from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

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

load_dotenv()


def build_map_provider() -> MapProvider:
    mode = os.getenv("TRIPMIND_ROUTE_PROVIDER", "auto").strip().lower()
    if mode in {"auto", "ors"} and os.getenv("ORS_API_KEY"):
        from .integrations.openrouteservice import OpenRouteServiceProvider

        return FallbackMapProvider(OpenRouteServiceProvider(), LocalMapProvider())
    if mode == "ors":
        raise RuntimeError("TRIPMIND_ROUTE_PROVIDER=ors but ORS_API_KEY is not set.")
    return LocalMapProvider()


def build_realtime_provider() -> RealtimeProvider:
    mode = os.getenv("TRIPMIND_REALTIME_PROVIDER", "auto").strip().lower()
    if mode in {"auto", "dashscope"} and os.getenv("DASHSCOPE_API_KEY"):
        provider = DashScopeRealtimeProvider()
        ttl = int(os.getenv("TRIPMIND_REALTIME_TTL_SECONDS", "3600"))
        return CachingRealtimeProvider(provider, ttl_seconds=ttl)
    if mode == "dashscope":
        raise RuntimeError(
            "TRIPMIND_REALTIME_PROVIDER=dashscope but DASHSCOPE_API_KEY is not set."
        )
    return NullRealtimeProvider()


class TripMindService:
    def __init__(
        self,
        map_provider: MapProvider | None = None,
        realtime_provider: RealtimeProvider | None = None,
    ):
        self.pois = load_pois()
        self.rag = LocalKnowledgeRetriever()
        self.ranker = HeuristicPersonalizedRanker()
        self.realtime_adjuster = RealtimeAdjustmentEngine()
        self.map = map_provider or build_map_provider()
        self.realtime = realtime_provider or build_realtime_provider()
        self.planner = GreedyConstraintPlanner(self.map)

    def plan(self, profile: UserProfile) -> Itinerary:
        candidates = city_pois(self.pois, profile.city, profile.country)
        if not candidates:
            raise ValueError(
                f"No POIs available for city={profile.city!r}, "
                f"country={profile.country!r} in the current local dataset."
            )

        knowledge = self.rag.retrieve(profile, top_k=5)
        ranked = self.ranker.rank(profile, candidates, knowledge)

        realtime_candidates = [x.poi for x in ranked[: min(12, len(ranked))]]
        try:
            signals = self.realtime.fetch(profile, realtime_candidates)
        except Exception:
            signals = []

        ranked = self.realtime_adjuster.apply(profile, ranked, signals)
        blocked = {
            signal.subject
            for signal in signals
            if signal.hard_constraint and signal.effect == "avoid"
        }

        days = self.planner.plan(profile, ranked, blocked_pois=blocked)
        total_cost = round(sum(day.estimated_cost for day in days), 2)
        return Itinerary(
            profile=profile,
            days=days,
            total_estimated_cost=total_cost,
            retrieved_knowledge=knowledge,
            realtime_signals=signals,
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
            "> Some route legs are local estimates. Configure `ORS_API_KEY` and use "
            "walking/driving/cycling for road-network-grounded routes."
        )
    elif used_grounded_route:
        lines.append("> Route distances/times are grounded by openrouteservice.")
    else:
        lines.append("> No inter-POI route leg was needed for this plan.")

    if not itinerary.realtime_signals:
        lines.append(
            "> Realtime enrichment is currently disabled/unavailable. Set "
            "`DASHSCOPE_API_KEY` to enable Alibaba Cloud web-search context."
        )
    return "\n".join(lines)
