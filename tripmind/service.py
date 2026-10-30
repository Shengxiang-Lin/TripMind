from __future__ import annotations

from datetime import date
import math
from typing import Any, List, Mapping

from .config import get_config, resolve_project_path
from .data import city_pois, load_pois
from .map_provider import FallbackMapProvider, LocalMapProvider, MapProvider
from .models import Itinerary, RealtimeSignal, UserProfile
from .planner import GreedyConstraintPlanner
from .rag import LocalKnowledgeRetriever
from .realtime import (
    CachingRealtimeProvider,
    DashScopeRealtimeProvider,
    NullRealtimeProvider,
    RealtimeProvider,
)
from .recommender import HeuristicPersonalizedRanker, RealtimeAdjustmentEngine


def build_map_provider(config: Mapping[str, Any] | None = None) -> MapProvider:
    config = dict(config or get_config())
    mode = str(config.get("providers", {}).get("route", "auto")).strip().lower()
    ors = config.get("ors", {})
    api_key = str(ors.get("api_key", "")).strip()

    if mode in {"auto", "ors"} and api_key:
        from .integrations.openrouteservice import OpenRouteServiceProvider

        provider = OpenRouteServiceProvider(
            api_key=api_key,
            base_url=ors.get("base_url"),
            timeout=float(ors.get("timeout_seconds", 15)),
            config=config,
        )
        return FallbackMapProvider(provider, LocalMapProvider())
    if mode == "ors":
        raise RuntimeError(
            "providers.route=ors but no ORS key is configured. "
            "Set ors.api_key in configs/config.local.yaml or ORS_API_KEY."
        )
    return LocalMapProvider()


def build_realtime_provider(
    config: Mapping[str, Any] | None = None,
) -> RealtimeProvider:
    config = dict(config or get_config())
    mode = str(config.get("providers", {}).get("realtime", "auto")).strip().lower()
    dashscope = config.get("dashscope", {})
    api_key = str(dashscope.get("api_key", "")).strip()

    if mode in {"auto", "dashscope"} and api_key:
        provider = DashScopeRealtimeProvider(config=config)
        ttl = int(config.get("realtime", {}).get("ttl_seconds", 3600))
        return CachingRealtimeProvider(provider, ttl_seconds=ttl)
    if mode == "dashscope":
        raise RuntimeError(
            "providers.realtime=dashscope but no DashScope key is configured. "
            "Set dashscope.api_key in configs/config.local.yaml or DASHSCOPE_API_KEY."
        )
    return NullRealtimeProvider()


class TripMindService:
    def __init__(
        self,
        map_provider: MapProvider | None = None,
        realtime_provider: RealtimeProvider | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        self.config = dict(config or get_config())
        data_cfg = self.config.get("data", {})

        poi_dir = resolve_project_path(data_cfg.get("poi_dir", "data/poi"))
        knowledge_file = resolve_project_path(
            data_cfg.get("knowledge_file", "data/knowledge/travel_knowledge.json")
        )

        self.pois = load_pois(poi_dir)
        self.rag = LocalKnowledgeRetriever(knowledge_file)
        self.ranker = HeuristicPersonalizedRanker()
        self.realtime_adjuster = RealtimeAdjustmentEngine()
        self.map = map_provider or build_map_provider(self.config)
        self.realtime = realtime_provider or build_realtime_provider(self.config)
        self.planner = GreedyConstraintPlanner(self.map)

    def plan(self, profile: UserProfile) -> Itinerary:
        if isinstance(profile.days, bool) or not isinstance(profile.days, int) or profile.days < 1:
            raise ValueError("Trip days must be a positive integer.")
        if not math.isfinite(profile.budget) or profile.budget < 0:
            raise ValueError("Budget must be a finite nonnegative number.")
        if profile.start_date:
            date.fromisoformat(profile.start_date)
        candidates = city_pois(self.pois, profile.city, profile.country)
        if not candidates:
            raise ValueError(
                f"No POIs available for city={profile.city!r}, "
                f"country={profile.country!r} in the current local dataset."
            )

        if any(poi.currency != profile.currency for poi in candidates):
            raise ValueError("Budget currency must match the POI currency; exchange-rate conversion is not configured.")
        knowledge = self.rag.retrieve(profile, top_k=5)
        ranked = self.ranker.rank(profile, candidates, knowledge)

        realtime_candidates = [x.poi for x in ranked[: min(12, len(ranked))]]
        warnings: list[str] = []
        try:
            signals = self.realtime.fetch(profile, realtime_candidates)
            warnings.extend(getattr(self.realtime, "last_errors", []))
        except Exception as exc:
            signals = []
            warnings.append(f"Realtime lookup failed ({type(exc).__name__}).")
        if isinstance(self.realtime, NullRealtimeProvider):
            realtime_status = "disabled"
        elif warnings:
            realtime_status = "partial" if signals else "failed"
        else:
            realtime_status = "ok"

        ranked = self.realtime_adjuster.apply(profile, ranked, signals)
        blocked = {
            signal.subject
            for signal in signals
            if signal.hard_constraint and signal.effect == "avoid"
        }

        days = self.planner.plan(profile, ranked, blocked_pois=blocked)
        total_cost = round(sum(day.estimated_cost for day in days), 2)
        if isinstance(self.map, FallbackMapProvider) and any(
            item.travel_from_previous and not item.travel_from_previous.grounded
            for day in days for item in day.items
        ):
            warnings.append("Road routing was unavailable or unsupported; some travel legs use local estimates.")
        return Itinerary(
            profile=profile,
            days=days,
            total_estimated_cost=total_cost,
            retrieved_knowledge=knowledge,
            realtime_signals=signals,
            warnings=warnings,
            realtime_status=realtime_status,
        )


def minute_str(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def _signal_markdown(signal: RealtimeSignal) -> str:
    hard = " · **hard constraint**" if signal.hard_constraint else ""
    return (
        f"- **{signal.kind}** · {signal.subject} · "
        f"{signal.confidence}{hard}: {signal.summary}"
    )


def itinerary_to_markdown(itinerary: Itinerary) -> str:
    lines: List[str] = []
    p = itinerary.profile
    location = f"{p.city}, {p.country}" if p.country else p.city
    lines.append(f"# {location} · {p.days}-Day Personalized Itinerary")
    lines.append(
        f"Profile: **{p.travel_group}** · **{p.pace}** · interests: "
        + ", ".join(p.interests)
    )
    if p.start_date:
        lines.append(f"Start date: **{p.start_date}**")
    lines.append(
        f"Budget: {p.currency} {p.budget:.0f} · currently planned ticket cost: "
        f"{p.currency} {itinerary.total_estimated_cost:.0f}"
    )
    lines.append("")

    if itinerary.warnings:
        lines.append("## Planning notices")
        lines.extend(f"- {warning}" for warning in itinerary.warnings)
        lines.append("")

    if itinerary.realtime_signals:
        lines.append("## Realtime context")
        lines.extend(_signal_markdown(s) for s in itinerary.realtime_signals)
        lines.append(
            "_Realtime web-search signals are planning context, not a substitute "
            "for official ticket/closure confirmation._"
        )
        lines.append("")

    used_grounded_route = False
    used_estimated_route = False
    for day in itinerary.days:
        lines.append(f"## Day {day.day}")
        if not day.items:
            lines.append("_No feasible POIs under current constraints._")
            continue
        for item in day.items:
            if item.travel_from_previous:
                leg = item.travel_from_previous
                used_grounded_route = used_grounded_route or leg.grounded
                used_estimated_route = used_estimated_route or not leg.grounded
                grounding = "grounded" if leg.grounded else "estimated"
                lines.append(
                    f"↳ {leg.mode}: **{leg.distance_km:.1f} km**, "
                    f"about **{leg.duration_min} min** "
                    f"({grounding}, {leg.source})"
                )
            reasons = ", ".join(item.poi.tags[:4])
            lines.append(
                f"- **{minute_str(item.start_min)}–{minute_str(item.end_min)}** "
                f"{item.poi.name} · {item.poi.currency} {item.poi.ticket_price:.0f} "
                f"· score {item.score:.2f}  \n"
                f"  {item.poi.description}  \n"
                f"  _tags: {reasons}_"
            )
        lines.append(
            f"\nDay travel: **{day.travel_distance_km:.1f} km / "
            f"{day.travel_duration_min} min**"
        )

    lines.append("\n---")
    if used_estimated_route:
        lines.append(
            "> Some route legs are local estimates. Configure an ORS key and use "
            "walking/driving/cycling for road-network-grounded routes."
        )
    elif used_grounded_route:
        lines.append("> Route distances/times are grounded by openrouteservice.")
    else:
        lines.append("> No inter-POI route leg was needed for this plan.")

    if itinerary.realtime_status in {"disabled", "failed"}:
        lines.append(
            "> Realtime enrichment is currently disabled/unavailable. Configure a "
            "DashScope key to enable Alibaba Cloud web-search context."
        )
    elif itinerary.realtime_status == "ok" and not itinerary.realtime_signals:
        lines.append("> Realtime lookup completed; no relevant current signals were returned.")
    return "\n".join(lines)
