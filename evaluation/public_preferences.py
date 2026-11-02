"""Validate the normalized public travel-preference corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="data/processed/public_preferences",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ids = set()
    users_by_split = {}
    counts = {}
    for path in sorted(root.rglob("*.jsonl")):
        count = 0
        for record in jsonl(path):
            record_id = record["record_id"]
            if record_id in ids:
                raise AssertionError(f"duplicate record_id: {record_id}")
            ids.add(record_id)
            task = record["task"]
            if task == "explicit_destination_preference":
                split = record["split"]
                users_by_split.setdefault(split, set()).add(record["user_key"])
                label = record["label"]
                if label["class"] not in {"yes", "maybe", "no"}:
                    raise AssertionError("invalid Stravl label")
                if label["ordinal"] not in {0, 1, 2}:
                    raise AssertionError("invalid Stravl ordinal")
                if not record["candidate"]["destination"]:
                    raise AssertionError("empty Stravl destination")
            elif task == "next_poi_prediction":
                roles = [m["role"] for m in record["messages"]]
                if roles != ["system", "user", "assistant"]:
                    raise AssertionError("invalid GenUP ChatML roles")
                if not record["input_text"] or not record["target_text"]:
                    raise AssertionError("empty GenUP input/target")
            else:
                raise AssertionError(f"unknown task: {task}")
            count += 1
        counts[str(path.relative_to(root))] = count

    splits = list(users_by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = users_by_split[left] & users_by_split[right]
            if overlap:
                raise AssertionError(
                    f"Stravl user leakage: {left}/{right}: {len(overlap)}"
                )

    print("VALIDATED")
    print("manifest sources:", sorted(manifest.get("sources", {})))
    print("files:", counts)
    print("records:", len(ids))


if __name__ == "__main__":
    main()
