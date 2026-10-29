from __future__ import annotations

from typing import List

from .data import city_pois, load_pois
from .map_provider import LocalMapProvider
from .models import Itinerary, UserProfile
from .planner import GreedyConstraintPlanner
from .rag import LocalKnowledgeRetriever
from .recommender import HeuristicPersonalizedRanker


class TripMindService:
    def __init__(self):
        self.pois = load_pois()
        self.rag = LocalKnowledgeRetriever()
        self.ranker = HeuristicPersonalizedRanker()
        self.map = LocalMapProvider()
        self.planner = GreedyConstraintPlanner(self.map)

    def plan(self, profile: UserProfile) -> Itinerary:
        candidates = city_pois(self.pois, profile.city)
        if not candidates:
            raise ValueError(f"No POIs available for city={profile.city!r} in local demo data.")

        knowledge = self.rag.retrieve(profile, top_k=5)
        ranked = self.ranker.rank(profile, candidates, knowledge)
        days = self.planner.plan(profile, ranked)
        total_cost = round(sum(day.estimated_cost for day in days), 2)
        return Itinerary(
            profile=profile,
            days=days,
            total_estimated_cost=total_cost,
            retrieved_knowledge=knowledge,
        )


def minute_str(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def itinerary_to_markdown(itinerary: Itinerary) -> str:
    lines: List[str] = []
    p = itinerary.profile
    lines.append(f"# {p.city} · {p.days}-Day Personalized Itinerary")
    lines.append(
        f"Profile: **{p.travel_group}** · **{p.pace}** · interests: "
        + ", ".join(p.interests)
    )
    lines.append(
        f"Budget: ¥{p.budget:.0f} · currently planned ticket cost: "
        f"¥{itinerary.total_estimated_cost:.0f}"
    )
    lines.append("")
    for day in itinerary.days:
        lines.append(f"## Day {day.day}")
        if not day.items:
            lines.append("_No feasible POIs under current constraints._")
            continue
        for item in day.items:
            if item.travel_from_previous:
                leg = item.travel_from_previous
                lines.append(
                    f"↳ {leg.mode}: **{leg.distance_km:.1f} km**, about **{leg.duration_min} min**"
                )
            reasons = ", ".join(item.poi.tags[:4])
            lines.append(
                f"- **{minute_str(item.start_min)}–{minute_str(item.end_min)}** "
                f"{item.poi.name} · ¥{item.poi.ticket_price:.0f} · score {item.score:.2f}  \n"
                f"  {item.poi.description}  \n"
                f"  _tags: {reasons}_"
            )
        lines.append(
            f"\nDay travel: **{day.travel_distance_km:.1f} km / {day.travel_duration_min} min**"
        )
    lines.append("\n---")
    lines.append(
        "> Local demo uses estimated straight-line-grounded travel time. "
        "Replace `LocalMapProvider` with the Bailian Amap MCP adapter for real routing."
    )
    return "\n".join(lines)
