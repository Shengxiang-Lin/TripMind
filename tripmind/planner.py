from __future__ import annotations

from datetime import datetime
from typing import List

from .map_provider import MapProvider
from .models import DayPlan, RankedPOI, ScheduleItem, UserProfile


def _to_min(hhmm: str) -> int:
    dt = datetime.strptime(hhmm, "%H:%M")
    return dt.hour * 60 + dt.minute


class GreedyConstraintPlanner:
    """A compact baseline planner with explicit time/budget constraints.

    It favors high personalized score while selecting geographically nearby POIs.
    OR-Tools can replace this class without changing upstream/downstream APIs.
    """

    def __init__(self, map_provider: MapProvider):
        self.map = map_provider

    def plan(self, profile: UserProfile, ranked: List[RankedPOI]) -> List[DayPlan]:
        remaining = ranked[: max(8, profile.days * 5)]
        daily_budget = profile.budget / max(1, profile.days)
        days: List[DayPlan] = []

        for day_idx in range(1, profile.days + 1):
            current_min = 9 * 60
            hard_end = 20 * 60
            day_cost = 0.0
            distance = 0.0
            travel_min = 0
            items: List[ScheduleItem] = []
            previous = None

            while remaining:
                feasible = []
                for cand in remaining:
                    poi = cand.poi
                    leg = self.map.route(previous.poi, poi, profile.transport_preference) if previous else None
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

                feasible.sort(key=lambda x: x[0], reverse=True)
                _, chosen, leg, start, end = feasible[0]
                items.append(
                    ScheduleItem(
                        poi=chosen.poi,
                        start_min=start,
                        end_min=end,
                        score=chosen.score,
                        travel_from_previous=leg,
                    )
                )
                day_cost += chosen.poi.ticket_price
                if leg:
                    distance += leg.distance_km
                    travel_min += leg.duration_min
                current_min = end + 25
                previous = chosen
                remaining.remove(chosen)

                max_pois = {"relaxed": 3, "normal": 4, "intensive": 5}.get(profile.pace, 4)
                if len(items) >= max_pois:
                    break

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
