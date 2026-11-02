"""Legacy/auxiliary public preference normalizer.

The main TripMind small-model path is now `build_genup_preference_dataset.py`,
which converts GenUP next-POI supervision into candidate-level preference data.
This script is retained for optional experiments and backward compatibility.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.config import get_config, resolve_project_path
from tripmind.processing.preferences import iter_genup_records, iter_stravl_records

GENUP_META = {
    "genup-nyc": (
        "FourSquare-NYC-POI",
        "https://huggingface.co/datasets/w11wo/FourSquare-NYC-POI",
        "genup-nyc",
    ),
    "genup-tky": (
        "FourSquare-TKY-POI",
        "https://huggingface.co/datasets/w11wo/FourSquare-TKY-POI",
        "genup-tky",
    ),
    "genup-ca": (
        "Gowalla-CA-POI",
        "https://huggingface.co/datasets/w11wo/Gowalla-CA-POI",
        "genup-ca",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["stravl", *GENUP_META],
        default=list(GENUP_META),
        help="Defaults to GenUP only. Add `stravl` explicitly for auxiliary experiments.",
    )
    parser.add_argument("--output", default="data/processed/public_preferences")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_stravl(raw_dir: Path, output: Path, *, seed: int) -> dict:
    folder = raw_dir / "stravl"
    csv_path = folder / "Stravl_Travel_Preference_Data.csv"
    destinations = folder / "destination_ids.txt"
    if not csv_path.is_file() or not destinations.is_file():
        raise FileNotFoundError(
            "Stravl raw files are missing. Run "
            "`python scripts/download_data.py --source stravl` first."
        )

    streams = {}
    counts = Counter()
    users = {split: set() for split in ("train", "validation", "test")}
    try:
        for split in users:
            path = output / "explicit_destination" / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            streams[split] = path.open("w", encoding="utf-8")
        for record in iter_stravl_records(csv_path, destinations, seed=seed):
            split = record["split"]
            streams[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            counts[split] += 1
            users[split].add(record["user_key"])
    finally:
        for stream in streams.values():
            stream.close()

    if (users["train"] & users["validation"]) or (users["train"] & users["test"]) or (
        users["validation"] & users["test"]
    ):
        raise AssertionError("Stravl user leakage across splits")
    return {
        "task": "explicit_destination_preference",
        "records": sum(counts.values()),
        "splits": dict(counts),
        "users": {split: len(values) for split, values in users.items()},
        "label_source": "original Stravl yes/maybe/no swipes",
        "status": "optional_auxiliary",
    }


def _write_genup(raw_dir: Path, output: Path, source: str) -> dict:
    dataset_name, source_url, raw_name = GENUP_META[source]
    root = raw_dir / raw_name
    if not root.is_dir():
        raise FileNotFoundError(
            f"{source} raw files are missing. Run "
            f"`python scripts/download_data.py --source {source}` first."
        )
    counts = Counter()
    output_dir = output / "next_poi"
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = {}
    try:
        for record in iter_genup_records(root, dataset_name=dataset_name, source_url=source_url):
            split = record["split"]
            if split not in streams:
                streams[split] = (output_dir / f"{source}-{split}.jsonl").open(
                    "w", encoding="utf-8"
                )
            streams[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    return {
        "task": "next_poi_prediction",
        "records": sum(counts.values()),
        "splits": dict(counts),
        "label_source": "original GenUP/LLM4POI next-POI target",
        "status": "legacy_generation_format",
    }


def main() -> None:
    args = parse_args()
    cfg = get_config()
    raw_dir = resolve_project_path(cfg["data"].get("raw_dir", "data/raw"))
    output = resolve_project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "status": "auxiliary_legacy_normalizer",
        "recommended_main_builder": "training/build_genup_preference_dataset.py",
        "sources": {},
    }
    for source in args.sources:
        summary = (
            _write_stravl(raw_dir, output, seed=args.seed)
            if source == "stravl"
            else _write_genup(raw_dir, output, source)
        )
        manifest["sources"][source] = summary
        print(source, summary)

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Wrote auxiliary public preference corpus to", output)
    print("Main scorer builder: python training/build_genup_preference_dataset.py")


if __name__ == "__main__":
    main()
