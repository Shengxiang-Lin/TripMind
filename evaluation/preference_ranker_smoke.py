from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.models import POI, UserProfile
from tripmind.preference_prompt import (
    candidate_text_from_poi,
    candidate_text_from_record,
    profile_text_from_record,
    profile_text_from_user,
)
from tripmind.recommender import LearnedPersonalizedRanker


class StubScorer:
    def score_pois(self, profile, candidates):
        mapping = {"culture": 3.0, "nature": 2.0, "shopping": -1.0}
        return [mapping.get(poi.category, 0.0) for poi in candidates]


def make(poi_id: str, category: str) -> POI:
    return POI(
        poi_id=poi_id,
        city="杭州",
        name=poi_id,
        category=category,
        longitude=120.0,
        latitude=30.0,
        ticket_price=0.0,
        recommended_duration=90,
        opening_time="09:00",
        closing_time="18:00",
        popularity=0.5,
        tags=[category],
        suitable_groups=["couple"],
    )


def main() -> None:
    profile = UserProfile(
        city="杭州",
        interests=["culture", "nature"],
        travel_history=["故宫", "苏州博物馆"],
    )
    candidates = [
        make("museum", "culture"),
        make("park", "nature"),
        make("mall", "shopping"),
    ]
    ranked = LearnedPersonalizedRanker(
        StubScorer(), heuristic_blend=0.0
    ).rank(profile, candidates)
    assert [item.poi.poi_id for item in ranked] == ["museum", "park", "mall"]
    assert all(0 <= item.score <= 1 for item in ranked)

    # Learned candidate semantics must be identical between GenUP training and
    # TripMind runtime. Rich POI attributes are intentionally excluded from the
    # learned prompt and handled by the heuristic/planner layers instead.
    record_candidate = {"poi_id": "123", "category": "Museum", "category_id": "10"}
    runtime_candidate = make("runtime-museum", "Museum")
    runtime_candidate.name = "A Famous Museum"
    runtime_candidate.ticket_price = 999.0
    runtime_candidate.description = "Rich runtime-only description"
    runtime_candidate.tags = ["museum", "history", "architecture"]
    assert candidate_text_from_record(record_candidate) == candidate_text_from_poi(
        runtime_candidate
    )

    # Dataset-local user IDs are removed from training prompts; both training and
    # runtime profiles use the same top-level "Traveler profile:" wrapper.
    record_profile = profile_text_from_record(
        {"profile_text": "You are user 7 and Age: adult; preferences: museums."}
    )
    runtime_profile = profile_text_from_user(profile)
    assert record_profile.startswith("Traveler profile:")
    assert runtime_profile.startswith("Traveler profile:")
    assert "user 7" not in record_profile.casefold()

    print("PREFERENCE_RANKER_SMOKE_OK")


if __name__ == "__main__":
    main()
