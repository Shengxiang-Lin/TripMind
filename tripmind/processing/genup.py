"""GenUP adapters for candidate-level preference learning."""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .common import stable_id

USER_RE = re.compile(r"\bYou are user\s+(\d+)\b", re.IGNORECASE)
TARGET_RE = re.compile(r"\bwill visit POI id\s+(\d+)\b", re.IGNORECASE)
VISIT_RE = re.compile(
    r"\bvisited POI id\s+(\d+)\s+which is\s+(?:an?\s+)?(.+?)\s+and has Category id\s+(\d+)\b",
    re.IGNORECASE,
)


def infer_parquet_split(path: Path) -> str:
    value = path.as_posix().casefold()
    for split in ("validation", "test", "train"):
        if split in value:
            return split
    return "unspecified"


def iter_raw_rows(root: Path) -> Iterable[dict[str, str]]:
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root}")
    for path in paths:
        split = infer_parquet_split(path)
        frame = pd.read_parquet(path, columns=["system_prompt", "inputs", "targets"])
        for row in frame.to_dict(orient="records"):
            system_prompt = str(row.get("system_prompt") or "").strip()
            inputs = str(row.get("inputs") or "").strip()
            targets = str(row.get("targets") or "").strip()
            if system_prompt and inputs and targets:
                yield {
                    "upstream_split": split,
                    "system_prompt": system_prompt,
                    "inputs": inputs,
                    "targets": targets,
                }


def parse_user_id(system_prompt: str) -> str | None:
    match = USER_RE.search(system_prompt)
    return match.group(1) if match else None


def parse_target_poi_id(target: str) -> str | None:
    match = TARGET_RE.search(target)
    return match.group(1) if match else None


def parse_history_visits(text: str) -> list[dict[str, str]]:
    return [
        {
            "poi_id": poi_id,
            "category": " ".join(category.split()),
            "category_id": category_id,
        }
        for poi_id, category, category_id in VISIT_RE.findall(text)
    ]


def context_identity(dataset_name: str, row: dict[str, str]) -> str:
    return stable_id(
        dataset_name,
        row["upstream_split"],
        row["system_prompt"],
        row["inputs"],
        row["targets"],
    )[:24]


def effective_split(upstream_split: str, context_id: str, validation_fraction: float = 0.1) -> str:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if upstream_split in {"validation", "test"}:
        return upstream_split
    fraction = int(stable_id("genup_validation", context_id)[:12], 16) / 16**12
    return "validation" if fraction < validation_fraction else "train"


def collect_catalog_and_user_targets(
    root: Path, dataset_name: str
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    observations: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    user_targets: dict[str, set[str]] = defaultdict(set)
    for row in iter_raw_rows(root):
        raw_user = parse_user_id(row["system_prompt"]) or stable_id(dataset_name, row["system_prompt"])[:16]
        user_key = stable_id(dataset_name, "user", raw_user)[:20]
        target = parse_target_poi_id(row["targets"])
        if target:
            user_targets[user_key].add(target)
        for visit in parse_history_visits(row["inputs"]):
            observations[visit["poi_id"]][(visit["category"], visit["category_id"])] += 1

    catalog: dict[str, dict[str, Any]] = {}
    for poi_id, counter in observations.items():
        (category, category_id), majority = counter.most_common(1)[0]
        catalog[poi_id] = {
            "poi_id": poi_id,
            "category": category,
            "category_id": category_id,
            "metadata_observations": sum(counter.values()),
            "majority_observations": majority,
        }
    return catalog, user_targets


def candidate_payload(poi_id: str, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return dict(catalog.get(poi_id, {"poi_id": poi_id, "category": None, "category_id": None}))


def sample_negatives(
    *,
    context_id: str,
    positive_id: str,
    history_ids: set[str],
    known_user_targets: set[str],
    catalog: dict[str, dict[str, Any]],
    count: int,
    hard_fraction: float,
) -> list[tuple[str, str]]:
    if count < 1:
        raise ValueError("negatives_per_positive must be at least 1")
    if not 0 <= hard_fraction <= 1:
        raise ValueError("hard_negative_fraction must be in [0, 1]")

    excluded = set(history_ids) | set(known_user_targets) | {positive_id}
    all_pool = [poi_id for poi_id in sorted(catalog) if poi_id not in excluded]
    if len(all_pool) < count:
        return []

    positive_category = catalog.get(positive_id, {}).get("category_id")
    hard_pool = [
        poi_id for poi_id in all_pool
        if positive_category and catalog.get(poi_id, {}).get("category_id") == positive_category
    ]
    rng = random.Random(int(stable_id("genup_negative", context_id)[:16], 16))
    rng.shuffle(hard_pool)
    desired_hard = min(count, round(count * hard_fraction))
    chosen = [(poi_id, "same_category_hard") for poi_id in hard_pool[:desired_hard]]
    chosen_ids = {poi_id for poi_id, _ in chosen}

    random_pool = [poi_id for poi_id in all_pool if poi_id not in chosen_ids]
    rng.shuffle(random_pool)
    chosen.extend((poi_id, "random") for poi_id in random_pool[: count - len(chosen)])
    return chosen if len(chosen) == count else []


def iter_preference_contexts(
    root: Path,
    *,
    dataset_name: str,
    source_url: str,
    negatives_per_positive: int = 4,
    hard_negative_fraction: float = 0.5,
    validation_fraction: float = 0.1,
) -> Iterable[dict[str, Any]]:
    catalog, user_targets = collect_catalog_and_user_targets(root, dataset_name)
    for row in iter_raw_rows(root):
        positive_id = parse_target_poi_id(row["targets"])
        if not positive_id:
            continue
        context_id = context_identity(dataset_name, row)
        raw_user = parse_user_id(row["system_prompt"]) or stable_id(dataset_name, row["system_prompt"])[:16]
        user_key = stable_id(dataset_name, "user", raw_user)[:20]
        visits = parse_history_visits(row["inputs"])
        history_ids = {visit["poi_id"] for visit in visits}
        negatives = sample_negatives(
            context_id=context_id,
            positive_id=positive_id,
            history_ids=history_ids,
            known_user_targets=user_targets.get(user_key, set()),
            catalog=catalog,
            count=negatives_per_positive,
            hard_fraction=hard_negative_fraction,
        )
        if len(negatives) != negatives_per_positive:
            continue
        negative_payloads = []
        for poi_id, kind in negatives:
            payload = candidate_payload(poi_id, catalog)
            payload["negative_type"] = kind
            negative_payloads.append(payload)
        yield {
            "schema_version": 1,
            "task": "candidate_poi_preference",
            "source_dataset": dataset_name,
            "source_url": source_url,
            "context_id": context_id,
            "upstream_split": row["upstream_split"],
            "split": effective_split(row["upstream_split"], context_id, validation_fraction),
            "user_key": user_key,
            "profile_text": row["system_prompt"],
            "history_text": row["inputs"],
            "history_poi_ids": [visit["poi_id"] for visit in visits],
            "positive_candidate": candidate_payload(positive_id, catalog),
            "negative_candidates": negative_payloads,
            "target_text": row["targets"],
            "supervision": "original_next_poi_positive_with_sampled_unobserved_negatives",
        }
