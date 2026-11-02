from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.config import get_config, resolve_project_path
from tripmind.preference_model import PreferenceScoringModel, resolve_device
from tripmind.preference_prompt import build_record_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TripMind preference checkpoint.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


@torch.inference_mode()
def evaluate_file(
    model,
    tokenizer,
    path: Path,
    *,
    device: torch.device,
    max_length: int,
    history_char_limit: int,
    max_contexts: int | None,
    no_history: bool,
    no_profile: bool,
) -> dict[str, float]:
    contexts = pairs = wins = top1 = 0
    reciprocal_rank = ndcg = average_rank = top_margin = 0.0

    for record in iter_jsonl(path):
        if max_contexts is not None and contexts >= max_contexts:
            break
        candidates = [record["positive_candidate"], *record["negative_candidates"]]
        prompts = [
            build_record_prompt(
                record,
                candidate,
                no_history=no_history,
                no_profile=no_profile,
                history_char_limit=history_char_limit,
            )
            for candidate in candidates
        ]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        scores = model(
            encoded["input_ids"],
            encoded["attention_mask"],
        ).float().cpu().tolist()

        positive = scores[0]
        negatives = scores[1:]
        pairs += len(negatives)
        wins += sum(positive > score for score in negatives)
        rank = 1 + sum(score >= positive for score in negatives)
        top1 += int(rank == 1)
        reciprocal_rank += 1.0 / rank
        ndcg += 1.0 / math.log2(rank + 1)
        average_rank += rank
        top_margin += positive - max(negatives)
        contexts += 1

    return {
        "contexts": contexts,
        "pairs": pairs,
        "pairwise_accuracy": wins / pairs if pairs else 0.0,
        "sampled_hit_at_1": top1 / contexts if contexts else 0.0,
        "sampled_mrr": reciprocal_rank / contexts if contexts else 0.0,
        "sampled_ndcg": ndcg / contexts if contexts else 0.0,
        "average_positive_rank": average_rank / contexts if contexts else 0.0,
        "average_top_margin": top_margin / contexts if contexts else 0.0,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    contexts = sum(int(row["contexts"]) for row in rows)
    pairs = sum(int(row["pairs"]) for row in rows)
    if not contexts:
        return {"contexts": 0, "pairs": 0}
    return {
        "contexts": contexts,
        "pairs": pairs,
        "pairwise_accuracy": (
            sum(row["pairwise_accuracy"] * row["pairs"] for row in rows) / pairs
            if pairs
            else 0.0
        ),
        "sampled_hit_at_1": sum(
            row["sampled_hit_at_1"] * row["contexts"] for row in rows
        ) / contexts,
        "sampled_mrr": sum(
            row["sampled_mrr"] * row["contexts"] for row in rows
        ) / contexts,
        "sampled_ndcg": sum(
            row["sampled_ndcg"] * row["contexts"] for row in rows
        ) / contexts,
        "average_positive_rank": sum(
            row["average_positive_rank"] * row["contexts"] for row in rows
        ) / contexts,
        "average_top_margin": sum(
            row["average_top_margin"] * row["contexts"] for row in rows
        ) / contexts,
    }


def main() -> None:
    args = parse_args()
    cfg = get_config()
    training_cfg = cfg.get("training", {}).get("preference", {})
    runtime_cfg = cfg.get("preference_model", {})

    checkpoint = resolve_project_path(
        args.checkpoint
        or runtime_cfg.get("checkpoint")
        or Path(training_cfg.get("checkpoint_dir", "checkpoints/tripmind-preference")) / "best"
    )
    data_root = resolve_project_path(
        args.data_root
        or training_cfg.get("output_dir", "data/processed/genup_preference")
    )
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_scope") != "sampled_cross_category_ranking_only":
        raise ValueError("Unexpected dataset evaluation scope")

    paths = sorted(data_root.glob(f"*-{args.split}.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No *-{args.split}.jsonl files under {data_root}")

    device = resolve_device(args.device)
    model, tokenizer, metadata = PreferenceScoringModel.load_checkpoint(
        checkpoint,
        device=device,
    )
    model.eval()
    max_length = int(metadata.get("max_length", 1024))
    history_char_limit = int(metadata.get("history_char_limit", 6000))

    per_source = {}
    metric_rows = []
    for path in paths:
        metrics = evaluate_file(
            model,
            tokenizer,
            path,
            device=device,
            max_length=max_length,
            history_char_limit=history_char_limit,
            max_contexts=args.max_contexts,
            no_history=args.no_history,
            no_profile=args.no_profile,
        )
        source = path.name.split("-", 1)[0]
        per_source[source] = metrics
        metric_rows.append(metrics)

    result = {
        "checkpoint": str(checkpoint),
        "split": args.split,
        "candidate_protocol": (
            "1 real next-POI category + sampled unobserved POIs from different categories"
        ),
        "evaluation_scope": "sampled_cross_category_ranking_only",
        "same_category_ranking_supported": False,
        "interpretation": (
            "Metrics measure category-level preference under sampled cross-category negatives; "
            "they do not demonstrate ranking quality between two POIs of the same category."
        ),
        "no_history": args.no_history,
        "no_profile": args.no_profile,
        "overall": aggregate(metric_rows),
        "per_source": per_source,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
