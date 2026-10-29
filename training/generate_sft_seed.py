"""Generate a small deterministic SFT seed set from the local POI catalogue.

This is NOT the final teacher-labeled dataset. It establishes the exact ChatML
schema we will later scale with stronger labels and human validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import random
from tripmind.data import load_pois
from tripmind.models import UserProfile
from tripmind.recommender import HeuristicPersonalizedRanker

OUT = ROOT / "data" / "sft_seed.jsonl"

PROFILES = [
    UserProfile(city="杭州", travel_group="couple", interests=["photography", "food", "culture"], pace="relaxed"),
    UserProfile(city="杭州", travel_group="family", interests=["nature", "museum", "food"], pace="normal"),
    UserProfile(city="杭州", travel_group="parents", interests=["history", "culture", "nature"], pace="relaxed"),
    UserProfile(city="杭州", travel_group="friends", interests=["food", "night_view", "shopping"], pace="intensive"),
]


def main(seed: int = 42):
    rng = random.Random(seed)
    pois = load_pois()
    ranker = HeuristicPersonalizedRanker()
    rows = []

    for idx in range(80):
        profile = PROFILES[idx % len(PROFILES)]
        candidates = rng.sample(pois, k=min(8, len(pois)))
        ranked = ranker.rank(profile, candidates)

        user_payload = {
            "profile": profile.__dict__,
            "candidate_pois": [
                {
                    "poi_id": p.poi_id,
                    "name": p.name,
                    "category": p.category,
                    "ticket_price": p.ticket_price,
                    "tags": p.tags,
                    "suitable_groups": p.suitable_groups,
                    "intensity": p.intensity,
                }
                for p in candidates
            ],
        }
        answer = {
            "ranking": [
                {
                    "poi_id": x.poi.poi_id,
                    "score": x.score,
                    "reason_tags": x.reasons,
                }
                for x in ranked
            ]
        }
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a personalized POI ranking model. Return valid JSON only.",
                    },
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
                ]
            }
        )

    OUT.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} seed examples to {OUT}")


if __name__ == "__main__":
    main()
