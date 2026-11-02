from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.config import get_config, resolve_project_path
from tripmind.preference_model import (
    PreferenceScoringModel,
    load_tokenizer,
    resolve_device,
)
from tripmind.preference_prompt import build_record_prompt
from tripmind.processing.common import stable_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TripMind's GenUP preference scorer.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--history-char-limit", type=int, default=None)
    parser.add_argument("--history-dropout", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-contexts", type=int, default=None)
    parser.add_argument("--max-validation-contexts", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def data_files(root: Path, split: str) -> list[Path]:
    paths = sorted(root.glob(f"*-{split}.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No *-{split}.jsonl files found under {root}")
    return paths


def count_contexts(paths: list[Path], limit: int | None = None) -> int:
    total = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    total += 1
                    if limit and total >= limit:
                        return total
    return total


def _drop_history(context_id: str, negative_id: str, fraction: float, seed: int) -> bool:
    if fraction <= 0:
        return False
    value = int(stable_id("history_dropout", seed, context_id, negative_id)[:12], 16) / 16**12
    return value < fraction


class PairwiseJsonlDataset(IterableDataset):
    def __init__(
        self,
        paths: list[Path],
        *,
        history_dropout: float,
        history_char_limit: int,
        seed: int,
        shuffle_buffer: int,
        max_contexts: int | None,
    ):
        super().__init__()
        self.paths = paths
        self.history_dropout = history_dropout
        self.history_char_limit = history_char_limit
        self.seed = seed
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.max_contexts = max_contexts
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _pairs(self) -> Iterable[dict[str, str]]:
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        seen_contexts = 0
        pair_index = 0
        paths = list(self.paths)
        random.Random(self.seed + self.epoch).shuffle(paths)

        for path in paths:
            for record in iter_jsonl(path):
                if self.max_contexts is not None and seen_contexts >= self.max_contexts:
                    return
                seen_contexts += 1
                positive = record["positive_candidate"]
                context_id = str(record["context_id"])
                for negative in record["negative_candidates"]:
                    if pair_index % num_workers != worker_id:
                        pair_index += 1
                        continue
                    negative_id = str(negative.get("poi_id", ""))
                    no_history = _drop_history(
                        context_id,
                        negative_id,
                        self.history_dropout,
                        self.seed,
                    )
                    common = {
                        "no_history": no_history,
                        "history_char_limit": self.history_char_limit,
                    }
                    yield {
                        "positive": build_record_prompt(record, positive, **common),
                        "negative": build_record_prompt(record, negative, **common),
                    }
                    pair_index += 1

    def __iter__(self):
        worker = get_worker_info()
        rng = random.Random(
            self.seed + 1009 * self.epoch + (worker.id if worker else 0)
        )
        buffer: list[dict[str, str]] = []
        for pair in self._pairs():
            buffer.append(pair)
            if len(buffer) >= self.shuffle_buffer:
                yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))


class PairCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        positives = [row["positive"] for row in rows]
        negatives = [row["negative"] for row in rows]
        return self.tokenizer(
            positives + negatives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


@torch.inference_mode()
def evaluate(
    model: PreferenceScoringModel,
    tokenizer,
    paths: list[Path],
    *,
    device: torch.device,
    max_length: int,
    history_char_limit: int,
    max_contexts: int | None,
) -> dict[str, float]:
    model.eval()
    contexts = pairs = wins = hits = 0
    rr = ndcg = avg_rank = 0.0

    for path in paths:
        for record in iter_jsonl(path):
            if max_contexts is not None and contexts >= max_contexts:
                break
            candidates = [record["positive_candidate"], *record["negative_candidates"]]
            prompts = [
                build_record_prompt(
                    record,
                    candidate,
                    history_char_limit=history_char_limit,
                )
                for candidate in candidates
            ]
            batch = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = model(batch["input_ids"], batch["attention_mask"]).float().cpu().tolist()
            positive = scores[0]
            negative_scores = scores[1:]
            pairs += len(negative_scores)
            wins += sum(positive > score for score in negative_scores)
            rank = 1 + sum(score >= positive for score in negative_scores)
            hits += int(rank == 1)
            rr += 1.0 / rank
            ndcg += 1.0 / math.log2(rank + 1)
            avg_rank += rank
            contexts += 1
        if max_contexts is not None and contexts >= max_contexts:
            break

    model.train()
    return {
        "contexts": float(contexts),
        "pairs": float(pairs),
        "pairwise_accuracy": wins / pairs if pairs else 0.0,
        "hit_at_1": hits / contexts if contexts else 0.0,
        "mrr": rr / contexts if contexts else 0.0,
        "ndcg": ndcg / contexts if contexts else 0.0,
        "average_positive_rank": avg_rank / contexts if contexts else 0.0,
    }


def main() -> None:
    args = parse_args()
    cfg = get_config()
    pref = cfg.get("training", {}).get("preference", {})

    seed = int(args.seed if args.seed is not None else pref.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_root = resolve_project_path(
        args.data_root or pref.get("output_dir", "data/processed/genup_preference")
    )
    output = resolve_project_path(
        args.output or pref.get("checkpoint_dir", "checkpoints/tripmind-preference")
    )
    base_model = args.model or pref.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
    epochs = int(args.epochs if args.epochs is not None else pref.get("epochs", 2))
    batch_size = int(args.batch_size if args.batch_size is not None else pref.get("batch_size", 2))
    grad_accum = int(
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else pref.get("gradient_accumulation_steps", 8)
    )
    lr = float(args.learning_rate if args.learning_rate is not None else pref.get("learning_rate", 2e-5))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else pref.get("weight_decay", 0.01))
    warmup_ratio = float(args.warmup_ratio if args.warmup_ratio is not None else pref.get("warmup_ratio", 0.05))
    max_grad_norm = float(args.max_grad_norm if args.max_grad_norm is not None else pref.get("max_grad_norm", 1.0))
    max_length = int(args.max_length if args.max_length is not None else pref.get("max_length", 1024))
    history_char_limit = int(
        args.history_char_limit
        if args.history_char_limit is not None
        else pref.get("history_char_limit", 6000)
    )
    history_dropout = float(
        args.history_dropout
        if args.history_dropout is not None
        else pref.get("history_dropout", 0.25)
    )
    gc_default = bool(pref.get("gradient_checkpointing", True))
    gradient_checkpointing = (
        False
        if args.no_gradient_checkpointing
        else True
        if args.gradient_checkpointing
        else gc_default
    )
    if epochs < 1 or batch_size < 1 or grad_accum < 1:
        raise ValueError("epochs, batch_size and gradient accumulation must be positive")
    if not 0 <= history_dropout <= 1:
        raise ValueError("history_dropout must be in [0,1]")

    train_paths = data_files(data_root, "train")
    validation_paths = data_files(data_root, "validation")
    train_contexts = count_contexts(train_paths, args.max_train_contexts)

    manifest_path = data_root / "manifest.json"
    negatives_per_context = 4
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        negatives_per_context = int(manifest.get("negatives_per_positive", 4))
    estimated_pairs = train_contexts * negatives_per_context
    batches_per_epoch = max(1, math.ceil(estimated_pairs / batch_size))
    optimizer_steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(base_model)
    model = PreferenceScoringModel.from_base_model(
        base_model,
        device=device,
        gradient_checkpointing=gradient_checkpointing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    dataset = PairwiseJsonlDataset(
        train_paths,
        history_dropout=history_dropout,
        history_char_limit=history_char_limit,
        seed=seed,
        shuffle_buffer=args.shuffle_buffer,
        max_contexts=args.max_train_contexts,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PairCollator(tokenizer, max_length),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "device": str(device),
                "base_model": base_model,
                "train_contexts": train_contexts,
                "estimated_pairs": estimated_pairs,
                "epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "total_optimizer_steps": total_steps,
                "history_dropout": history_dropout,
            },
            indent=2,
        )
    )

    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        micro_steps = 0

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            local_batch = batch["input_ids"].size(0) // 2
            scores = model(batch["input_ids"], batch["attention_mask"])
            positive = scores[:local_batch]
            negative = scores[local_batch:]
            loss = F.softplus(negative - positive).mean()
            (loss / grad_accum).backward()

            running_loss += float(loss.detach().cpu())
            running_acc += float((positive > negative).float().mean().detach().cpu())
            micro_steps += 1

            if micro_steps % grad_accum == 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if micro_steps % args.log_every == 0:
                progress.set_postfix(
                    loss=f"{running_loss / micro_steps:.4f}",
                    pair_acc=f"{running_acc / micro_steps:.3f}",
                    step=global_step,
                )

        if micro_steps % grad_accum:
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        metrics = evaluate(
            model,
            tokenizer,
            validation_paths,
            device=device,
            max_length=max_length,
            history_char_limit=history_char_limit,
            max_contexts=args.max_validation_contexts,
        )
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, micro_steps),
            "train_pairwise_accuracy": running_acc / max(1, micro_steps),
            "validation": metrics,
            "global_step": global_step,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, indent=2))

        save_extra = {
            "training_epoch": epoch + 1,
            "history_dropout": history_dropout,
            "validation_metrics": metrics,
        }
        model.save_checkpoint(
            output / "last",
            tokenizer,
            base_model=base_model,
            max_length=max_length,
            history_char_limit=history_char_limit,
            extra=save_extra,
        )
        if metrics["pairwise_accuracy"] > best_metric:
            best_metric = metrics["pairwise_accuracy"]
            model.save_checkpoint(
                output / "best",
                tokenizer,
                base_model=base_model,
                max_length=max_length,
                history_char_limit=history_char_limit,
                extra=save_extra,
            )

    summary = {
        "base_model": base_model,
        "best_validation_pairwise_accuracy": best_metric,
        "history": history,
        "best_checkpoint": str(output / "best"),
        "last_checkpoint": str(output / "last"),
    }
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Training complete:", output / "best")


if __name__ == "__main__":
    main()
