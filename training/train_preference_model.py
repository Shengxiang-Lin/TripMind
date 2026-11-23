from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from itertools import islice, zip_longest
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

TRAINING_STATE_FILE = "training_state.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TripMind's GenUP category-preference scorer.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume from a saved training checkpoint, e.g. checkpoints/tripmind-preference/last.",
    )
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
    parser.add_argument("--max-train-contexts-per-source", type=int, default=None)
    parser.add_argument("--max-validation-contexts-per-source", type=int, default=None)
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


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def iter_sampled(path: Path, quota: int, seed: int) -> Iterable[dict[str, Any]]:
    """Systematic sample with one reproducible random offset in the first stride."""
    if quota <= 0:
        raise ValueError("Per-source quota must be positive")
    total = count_lines(path)
    if quota >= total:
        yield from iter_jsonl(path)
        return
    # Integer form of floor(offset + sample * total / quota).
    offset = random.Random(seed).randrange(total)
    targets = ((offset + sample * total) // quota for sample in range(quota))
    target = next(targets)
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(line for line in stream if line.strip()):
            if index == target:
                yield json.loads(line)
                target = next(targets, None)
                if target is None:
                    return


def source_name(path: Path) -> str:
    return path.stem.rsplit("-", 1)[0]


def source_counts(
    paths: list[Path], limit: int | None, quota: int | None,
) -> dict[str, int]:
    counts = {}
    remaining = limit
    for path in paths:
        if quota is not None:
            count = min(quota, count_lines(path))
        elif remaining is None:
            count = count_lines(path)
        else:
            with path.open(encoding="utf-8") as stream:
                count = sum(1 for _ in islice((line for line in stream if line.strip()), remaining))
            remaining -= count
        counts[source_name(path)] = count
    return counts


def source_seed(seed: int, path: Path) -> int:
    return int(stable_id("genup_source_sample", seed, path.name)[:16], 16)


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
        max_contexts_per_source: int | None = None,
    ):
        super().__init__()
        self.paths = paths
        self.history_dropout = history_dropout
        self.history_char_limit = history_char_limit
        self.seed = seed
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.max_contexts = max_contexts
        if max_contexts is not None and max_contexts_per_source is not None:
            raise ValueError("Cannot combine global and per-source training limits")
        self.max_contexts_per_source = max_contexts_per_source
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _records(self) -> Iterable[dict[str, Any]]:
        paths = list(self.paths)
        random.Random(self.seed + self.epoch).shuffle(paths)
        if self.max_contexts_per_source is None:
            records = (record for path in paths for record in iter_jsonl(path))
            yield from islice(records, self.max_contexts)
            return
        samples = [
            iter_sampled(path, self.max_contexts_per_source, source_seed(self.seed + self.epoch, path))
            for path in paths
        ]
        # Stream one sample from each source before feeding the existing pair shuffle.
        for group in zip_longest(*samples):
            yield from (record for record in group if record is not None)

    def _pairs(self) -> Iterable[dict[str, str]]:
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        pairs = (
            (record, negative)
            for record in self._records()
            for negative in record["negative_candidates"]
        )
        for pair_index, (record, negative) in enumerate(pairs):
            if pair_index % num_workers != worker_id:
                continue
            common = {
                "no_history": _drop_history(
                    str(record["context_id"]),
                    str(negative.get("poi_id", "")),
                    self.history_dropout,
                    self.seed,
                ),
                "history_char_limit": self.history_char_limit,
            }
            yield {
                "positive": build_record_prompt(record, record["positive_candidate"], **common),
                "negative": build_record_prompt(record, negative, **common),
            }

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
    max_contexts_per_source: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    model.eval()
    remaining = max_contexts
    overall = Counter()
    per_source = {}

    for path in paths:
        counts = Counter()
        records = (
            islice(iter_jsonl(path), remaining)
            if max_contexts_per_source is None
            else iter_sampled(path, max_contexts_per_source, source_seed(seed, path))
        )
        for record in records:
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
            counts["pairs"] += len(negative_scores)
            counts["wins"] += sum(positive > score for score in negative_scores)
            rank = 1 + sum(score >= positive for score in negative_scores)
            counts["hits"] += int(rank == 1)
            counts["rr"] += 1.0 / rank
            counts["ndcg"] += 1.0 / math.log2(rank + 1)
            counts["rank"] += rank
            counts["contexts"] += 1
        if max_contexts_per_source is None and remaining is not None:
            remaining -= counts["contexts"]
        overall.update(counts)
        per_source[source_name(path)] = validation_metrics(counts)

    model.train()
    return {"overall": validation_metrics(overall), "per_source": per_source}


def validation_metrics(counts: Counter) -> dict[str, float]:
    contexts, pairs = counts["contexts"], counts["pairs"]
    return {
        "contexts": contexts,
        "pairs": pairs,
        "pairwise_accuracy": counts["wins"] / pairs if pairs else 0.0,
        "sampled_hit_at_1": counts["hits"] / contexts if contexts else 0.0,
        "sampled_mrr": counts["rr"] / contexts if contexts else 0.0,
        "sampled_ndcg": counts["ndcg"] / contexts if contexts else 0.0,
        "average_positive_rank": counts["rank"] / contexts if contexts else 0.0,
    }


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def _save_training_state(
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch_completed: int,
    global_step: int,
    best_metric: float,
    history: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "epoch_completed": int(epoch_completed),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "history": history,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "run_config": run_config,
        "rng": _capture_rng_state(),
    }
    torch.save(payload, checkpoint_dir / TRAINING_STATE_FILE)


def main() -> None:
    args = parse_args()
    train_quota = args.max_train_contexts_per_source
    validation_quota = args.max_validation_contexts_per_source
    if args.max_train_contexts is not None and train_quota is not None:
        raise ValueError("--max-train-contexts and --max-train-contexts-per-source are mutually exclusive")
    for quota in (train_quota, validation_quota):
        if quota is not None and quota <= 0:
            raise ValueError("Per-source quotas must be positive")
    for split, quota in (("train", train_quota), ("validation", validation_quota)):
        if quota is None:
            print(f"Warning: {split} uses legacy file-order sampling; a global cap can exclude sources.", file=sys.stderr)
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
    requested_base_model = args.model or pref.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
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
    requested_max_length = int(args.max_length if args.max_length is not None else pref.get("max_length", 1024))
    requested_history_char_limit = int(
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
    first_epoch_paths = list(train_paths)
    random.Random(seed).shuffle(first_epoch_paths)
    sources = {
        "train_sources": source_counts(first_epoch_paths, args.max_train_contexts, train_quota),
        "validation_sources": source_counts(validation_paths, args.max_validation_contexts, validation_quota),
    }
    train_contexts = sum(sources["train_sources"].values())

    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if manifest and manifest.get("evaluation_scope") != "sampled_cross_category_ranking_only":
        raise ValueError("Training expects the GenUP sampled cross-category dataset schema.")
    negatives_per_context = int(manifest.get("negatives_per_positive", 4))
    estimated_pairs = train_contexts * negatives_per_context
    batches_per_epoch = max(1, math.ceil(estimated_pairs / batch_size))
    optimizer_steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    device = resolve_device(args.device)
    resume_dir = resolve_project_path(args.resume_from) if args.resume_from else None
    start_epoch = 0
    global_step = 0
    best_metric = -1.0
    history: list[dict[str, Any]] = []

    if resume_dir is not None:
        state_path = resume_dir / TRAINING_STATE_FILE
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Resume state not found: {state_path}. Use a training 'last' checkpoint, not an inference-only checkpoint."
            )
        model, tokenizer, metadata = PreferenceScoringModel.load_checkpoint(
            resume_dir, device=device
        )
        base_model = str(metadata["base_model"])
        max_length = int(metadata.get("max_length", requested_max_length))
        history_char_limit = int(
            metadata.get("history_char_limit", requested_history_char_limit)
        )
        if args.model is not None and args.model != base_model:
            raise ValueError("--model cannot change when resuming training")
        if args.max_length is not None and args.max_length != max_length:
            raise ValueError("--max-length cannot change when resuming training")
        if (
            args.history_char_limit is not None
            and args.history_char_limit != history_char_limit
        ):
            raise ValueError("--history-char-limit cannot change when resuming training")
        if gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone.config, "use_cache"):
            model.backbone.config.use_cache = False
    else:
        base_model = requested_base_model
        max_length = requested_max_length
        history_char_limit = requested_history_char_limit
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

    if resume_dir is not None:
        state = _torch_load(resume_dir / TRAINING_STATE_FILE)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state.get("epoch_completed", 0))
        global_step = int(state.get("global_step", 0))
        best_metric = float(state.get("best_metric", -1.0))
        history = list(state.get("history", []))
        _restore_rng_state(state.get("rng", {}))
        print(f"Resumed from {resume_dir}: completed_epochs={start_epoch}, global_step={global_step}")

    dataset = PairwiseJsonlDataset(
        train_paths,
        history_dropout=history_dropout,
        history_char_limit=history_char_limit,
        seed=seed,
        shuffle_buffer=args.shuffle_buffer,
        max_contexts=args.max_train_contexts,
        max_contexts_per_source=train_quota,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PairCollator(tokenizer, max_length),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    run_config = {
        "data_root": str(data_root),
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "max_grad_norm": max_grad_norm,
        "max_length": max_length,
        "history_char_limit": history_char_limit,
        "history_dropout": history_dropout,
        "seed": seed,
        "max_train_contexts_per_source": train_quota,
        "max_validation_contexts_per_source": validation_quota,
        "evaluation_scope": "sampled_cross_category_ranking_only",
    }

    print(
        json.dumps(
            {
                "device": str(device),
                "base_model": base_model,
                "train_contexts": train_contexts,
                **sources,
                "estimated_pairs": estimated_pairs,
                "epochs": epochs,
                "start_epoch": start_epoch,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "total_optimizer_steps": total_steps,
                "history_dropout": history_dropout,
                "evaluation_scope": "sampled_cross_category_ranking_only",
            },
            indent=2,
        )
    )

    if start_epoch >= epochs:
        print(f"Nothing to do: checkpoint already completed {start_epoch} epochs (target={epochs}).")
        return

    for epoch in range(start_epoch, epochs):
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
            max_contexts_per_source=validation_quota,
            seed=seed,
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

        validation_accuracy = metrics["overall"]["pairwise_accuracy"]
        improved = validation_accuracy > best_metric
        if improved:
            best_metric = validation_accuracy

        save_extra = {
            "training_epoch": epoch + 1,
            "history_dropout": history_dropout,
            "validation_metrics": metrics,
            "evaluation_scope": "sampled_cross_category_ranking_only",
            "same_category_ranking_supported": False,
        }
        last_dir = output / "last"
        model.save_checkpoint(
            last_dir,
            tokenizer,
            base_model=base_model,
            max_length=max_length,
            history_char_limit=history_char_limit,
            extra=save_extra,
        )
        _save_training_state(
            last_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_completed=epoch + 1,
            global_step=global_step,
            best_metric=best_metric,
            history=history,
            run_config=run_config,
        )

        if improved:
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
        "sources": sources,
        "run_config": run_config,
        "best_validation_pairwise_accuracy": best_metric,
        "history": history,
        "best_checkpoint": str(output / "best"),
        "last_checkpoint": str(output / "last"),
        "resume_checkpoint": str(output / "last"),
        "evaluation_scope": "sampled_cross_category_ranking_only",
        "same_category_ranking_supported": False,
    }
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Training complete:", output / "best")
    print("Resume checkpoint:", output / "last")


if __name__ == "__main__":
    main()
