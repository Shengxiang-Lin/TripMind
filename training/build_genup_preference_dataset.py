"""Build GenUP candidate-preference data for TripMind's transferable scorer."""
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
from tripmind.processing.genup import iter_preference_contexts

SOURCES = {
    "nyc": ("FourSquare-NYC-POI", "https://huggingface.co/datasets/w11wo/FourSquare-NYC-POI"),
    "tky": ("FourSquare-TKY-POI", "https://huggingface.co/datasets/w11wo/FourSquare-TKY-POI"),
    "ca": ("Gowalla-CA-POI", "https://huggingface.co/datasets/w11wo/Gowalla-CA-POI"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--negatives-per-positive", type=int, default=None)
    parser.add_argument("--hard-negative-fraction", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = get_config()
    train_cfg = cfg.get("training", {}).get("preference", {})
    sources = args.sources or train_cfg.get("sources", ["nyc", "tky", "ca"])
    negatives = int(args.negatives_per_positive if args.negatives_per_positive is not None else train_cfg.get("negatives_per_positive", 4))
    same_category_fraction = float(args.hard_negative_fraction if args.hard_negative_fraction is not None else train_cfg.get("hard_negative_fraction", 0.0))
    validation_fraction = float(args.validation_fraction if args.validation_fraction is not None else train_cfg.get("validation_fraction", 0.1))
    output = resolve_project_path(args.output or train_cfg.get("output_dir", "data/processed/genup_preference"))
    raw_dir = resolve_project_path(cfg["data"].get("raw_dir", "data/raw"))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "task": "candidate_poi_preference",
        "objective": "score the real next POI above sampled POIs never observed for that user",
        "transfer_policy": "dataset-local POI IDs are provenance only; model prompts use semantic categories",
        "negatives_per_positive": negatives,
        "same_category_negative_fraction": same_category_fraction,
        "validation_fraction_from_upstream_train": validation_fraction,
        "sources": {},
    }
    for source in sources:
        dataset_name, source_url = SOURCES[source]
        root = raw_dir / f"genup-{source}"
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {root}. Run `python scripts/download_data.py --source genup-{source}` first.")
        streams, counts = {}, Counter()
        same_category = total_negatives = 0
        try:
            for split in ("train", "validation", "test"):
                streams[split] = (output / f"{source}-{split}.jsonl").open("w", encoding="utf-8")
            for record in iter_preference_contexts(root, dataset_name=dataset_name, source_url=source_url, negatives_per_positive=negatives, hard_negative_fraction=same_category_fraction, validation_fraction=validation_fraction):
                split = record["split"]
                streams[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts[split] += 1
                for candidate in record["negative_candidates"]:
                    total_negatives += 1
                    same_category += candidate.get("negative_type") == "same_category"
        finally:
            for stream in streams.values():
                stream.close()
        manifest["sources"][source] = {
            "dataset": dataset_name,
            "records": sum(counts.values()),
            "splits": dict(counts),
            "same_category_negative_rate": round(same_category / total_negatives, 4) if total_negatives else 0.0,
        }
        print(source, manifest["sources"][source])
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("Wrote GenUP preference data to", output)

if __name__ == "__main__":
    main()
