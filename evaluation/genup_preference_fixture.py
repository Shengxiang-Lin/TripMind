"""Small offline fixture test for the GenUP category-preference adapter."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from tripmind.processing.common import stable_id
from tripmind.processing.genup import (
    iter_preference_contexts,
    parse_history_visits,
    parse_target_poi_id,
    parse_user_id,
)


def profile(user: int) -> str:
    return (
        f"You are user {user} and your basic information is as follows: Age: adult; "
        "Gender: unspecified. You have the following preferences: museums, parks."
    )


def trajectory(user: int, visits: list[tuple[int, str, int]]) -> str:
    parts = [f"The following data is a trajectory of user {user}:"]
    for index, (poi, category, category_id) in enumerate(visits):
        parts.append(
            f"At 2026-01-0{index + 1} 10:00:00, user {user} visited POI id {poi} "
            f"which is a {category} and has Category id {category_id}."
        )
    return " ".join(parts)


def target(user: int, poi: int) -> str:
    return f"At 2026-01-09 10:00:00, user {user} will visit POI id {poi}."


def main() -> None:
    assert parse_user_id(profile(7)) == "7"
    assert parse_target_poi_id(target(7, 99)) == "99"
    parsed = parse_history_visits(trajectory(7, [(1, "Museum", 10), (2, "Park", 20)]))
    assert [x["poi_id"] for x in parsed] == ["1", "2"]

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        train = root / "train"
        test = root / "test"
        train.mkdir()
        test.mkdir()

        rows = [
            {
                "system_prompt": profile(0),
                "inputs": trajectory(0, [(1, "Museum", 10), (2, "Park", 20)]),
                "targets": target(0, 3),
            },
            {
                "system_prompt": profile(0),
                "inputs": trajectory(0, [(3, "Museum", 10)]),
                "targets": target(0, 4),
            },
            {
                "system_prompt": profile(1),
                "inputs": trajectory(
                    1,
                    [
                        (5, "Museum", 10),
                        (6, "Park", 20),
                        (7, "Museum", 10),
                        (8, "Cafe", 30),
                    ],
                ),
                "targets": target(1, 5),
            },
        ]
        # Supply unseen categories so the held-out user has two valid negatives.
        donor = {
            "system_prompt": profile(2),
            "inputs": trajectory(2, [(9, "Cafe", 30), (10, "Park", 20)]),
            "targets": target(2, 9),
        }
        pd.DataFrame([*rows, donor]).to_parquet(train / "part.parquet", index=False)
        pd.DataFrame(rows[-1:]).to_parquet(test / "part.parquet", index=False)

        records = list(
            iter_preference_contexts(
                root,
                dataset_name="fixture",
                source_url="https://example.invalid",
                negatives_per_positive=2,
                hard_negative_fraction=0.0,
                validation_fraction=0.0,
            )
        )
        assert records

        # Do not infer user identity from iteration order: test/ may be visited before train/.
        user0_key = stable_id("fixture", "user", "0")[:20]
        user0 = [record for record in records if record["user_key"] == user0_key]
        assert user0
        for record in user0:
            negative_ids = {x["poi_id"] for x in record["negative_candidates"]}
            assert not negative_ids & {"1", "2", "3", "4"}
            assert record["positive_candidate"]["poi_id"] not in negative_ids
            positive_category = record["positive_candidate"].get("category_id")
            assert all(
                candidate.get("category_id") != positive_category
                for candidate in record["negative_candidates"]
            )

        assert any(
            record["upstream_split"] == "test" and record["split"] == "test"
            for record in records
        )

    print("GENUP CATEGORY-PREFERENCE FIXTURE PASSED")


if __name__ == "__main__":
    main()
