"""Small offline fixture test for the GenUP preference adapter."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

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
        train.mkdir(); test.mkdir()

        # POIs 1..8 create a candidate catalogue. User 0 has observed 1,2 and
        # targets 3 and 4 across contexts, so none of 1..4 may become negatives.
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
                "inputs": trajectory(1, [(5, "Museum", 10), (6, "Park", 20), (7, "Museum", 10), (8, "Cafe", 30)]),
                "targets": target(1, 5),
            },
        ]
        pd.DataFrame(rows).to_parquet(train / "part.parquet", index=False)
        pd.DataFrame(rows[-1:]).to_parquet(test / "part.parquet", index=False)

        records = list(
            iter_preference_contexts(
                root,
                dataset_name="fixture",
                source_url="https://example.invalid",
                negatives_per_positive=2,
                hard_negative_fraction=0.5,
                validation_fraction=0.0,
            )
        )
        assert records
        user0 = [r for r in records if r["user_key"] == records[0]["user_key"]]
        assert user0
        for record in user0:
            negative_ids = {x["poi_id"] for x in record["negative_candidates"]}
            assert not negative_ids & {"1", "2", "3", "4"}
            assert record["positive_candidate"]["poi_id"] not in negative_ids
        assert any(r["upstream_split"] == "test" and r["split"] == "test" for r in records)

    print("GENUP PREFERENCE FIXTURE PASSED")


if __name__ == "__main__":
    main()
