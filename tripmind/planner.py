from __future__ import annotations

from collections import Counter
import logging
from typing import List

from .geo import city_center, haversine_km, reach_km
from .map_provider import MapProvider
from .models import DayPlan, POI, RankedPOI, RouteLeg, ScheduleItem, UserProfile

logger = logging.getLogger(__name__)


def _to_min(hhmm: str) -> int:
    parts = hhmm.split(":")
    if len(parts) != 2 or any(len(p) != 2 or not p.isdigit() for p in parts):
        raise ValueError(f"Invalid time: {hhmm!r}")
    hour, minute = map(int, parts)
    if not (0 <= hour <= 24 and 0 <= minute <= 59) or (hour == 24 and minute):
        raise ValueError(f"Invalid time: {hhmm!r}")
    return hour * 60 + minute


def _blocked(name: str, blocked_names: set[str]) -> bool:
    value = "".join(name.casefold().split())
    for blocked in blocked_names:
        other = "".join(blocked.casefold().split())
        if other and (value == other or value in other or other in value):
            return True
    return False


class GreedyConstraintPlanner:
    """Compact baseline planner with explicit time/budget constraints.

    It favors high personalized score while selecting geographically nearby POIs.
    OR-Tools can replace this class without changing upstream/downstream APIs.
    """

    def __init__(self, map_provider: MapProvider):
        self.map = map_provider

    def plan(
        self,
        profile: UserProfile,
        ranked: List[RankedPOI],
        blocked_pois: set[str] | None = None,
        *,
        all_pois: List[POI] | None = None,
    ) -> List[DayPlan]:
        blocked_pois = blocked_pois or set()
        candidates = [
            item
            for item in ranked
            if not _blocked(item.poi.name, blocked_pois)
        ]
        if candidates:
            center = city_center(profile.city, all_pois or [item.poi for item in candidates])
            radius = reach_km(profile.transport_preference)
            distances = [
                haversine_km(item.poi.latitude, item.poi.longitude, *center)
                for item in candidates
            ]
            for limit in (radius, 1.5 * radius):
                reachable = [item for item, distance in zip(candidates, distances) if distance <= limit]
                if len(reachable) >= 4:
                    candidates = reachable
                    break
        shortlist_size = max(8, profile.days * 5)
        remaining = candidates[:shortlist_size]
        category_supply = Counter(item.poi.category for item in remaining)
        # Keep enough alternatives for multiple days when the top scores favor one category.
        for item in candidates[shortlist_size:]:
            category = item.poi.category
            if category_supply[category] < profile.days:
                remaining.append(item)
                category_supply[category] += 1
        matrix = (
            self.map.matrix([item.poi for item in remaining], profile.transport_preference)
            if len(remaining) > 1 else None
        )
        indices = {item.poi.poi_id: idx for idx, item in enumerate(remaining)}
        daily_budget = profile.budget / max(1, profile.days)
        days: List[DayPlan] = []

        for day_idx in range(1, profile.days + 1):
            current_min = 9 * 60
            hard_end = 20 * 60
            day_cost = 0.0
            distance = 0.0
            travel_min = 0
            items: List[ScheduleItem] = []
            category_counts = Counter()
            previous = None
            initial_feasible_count = 0

            while remaining:
                feasible = []
                for cand in remaining:
                    poi = cand.poi
                    if category_counts[poi.category] >= 2:
                        continue
                    leg = None
                    if previous and matrix is not None:
                        a, b = indices[previous.poi.poi_id], indices[poi.poi_id]
                        duration = matrix.durations_min[a][b]
                        distance_km = matrix.distances_km[a][b]
                        if duration is None or distance_km is None:
                            continue
                        leg = RouteLeg(
                            previous.poi.name, poi.name, distance_km, duration,
                            profile.transport_preference, matrix.source, matrix.grounded,
                        )
                    travel = leg.duration_min if leg else 0
                    arrival = current_min + travel
                    opening = _to_min(poi.opening_time)
                    start = max(arrival, opening)
                    end = start + poi.recommended_duration
                    close = _to_min(poi.closing_time)

                    if end > min(close, hard_end):
                        continue
                    if day_cost + poi.ticket_price > daily_budget * 0.65:
                        continue

                    geo_penalty = 0 if leg is None else min(0.25, leg.distance_km / 30)
                    utility = cand.score - geo_penalty
                    feasible.append((utility, cand, leg, start, end))

                if not feasible:
                    break

                if not items:
                    initial_feasible_count = len(feasible)
                feasible.sort(key=lambda x: x[0], reverse=True)
                # Once the first stop is set, prefer the best feasible second category.
                # With no alternative, keep the original utility ordering within the cap.
                choice = feasible[0]
                if len(category_counts) == 1:
                    choice = next(
                        (entry for entry in feasible if entry[1].poi.category not in category_counts),
                        choice,
                    )
                _, chosen, leg, start, end = choice
                items.append(
                    ScheduleItem(
                        poi=chosen.poi,
                        start_min=start,
                        end_min=end,
                        score=chosen.score,
                        travel_from_previous=leg,
                        reasons=chosen.reasons,
                    )
                )
                category_counts[chosen.poi.category] += 1
                day_cost += chosen.poi.ticket_price
                if leg:
                    distance += leg.distance_km
                    travel_min += leg.duration_min
                current_min = end + 25
                previous = chosen
                remaining.remove(chosen)

                max_pois = {"relaxed": 3, "normal": 4, "intensive": 5}.get(
                    profile.pace, 4
                )
                if len(items) >= max_pois:
                    break

            if len(items) == 1:
                logger.warning(
                    "%s day %d: selected one POI, %d feasible candidates at start",
                    profile.city, day_idx, initial_feasible_count,
                )
            days.append(
                DayPlan(
                    day=day_idx,
                    items=items,
                    estimated_cost=round(day_cost, 2),
                    travel_distance_km=round(distance, 2),
                    travel_duration_min=travel_min,
                )
            )

        return days
