"""Adapters for public travel-preference datasets.

The adapters preserve each dataset's original supervision instead of forcing
explicit destination swipes and next-POI trajectories into one artificial label.
"""
from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .common import stable_id

STRAVL_MAPS = {
    "form_a": {"0": "0-19", "1": "20-39", "2": "40-59", "3": "60+"},
    "form_b": {"0": "$0-$49", "1": "$50-$99", "2": "$100-$249", "3": "$300+"},
    "form_c": {"0": "winter", "1": "spring", "2": "summer", "3": "fall"},
    "form_f": {
        "0": "beach", "1": "adventure", "2": "nature", "3": "culture",
        "4": "nightlife", "5": "history", "6": "shopping", "7": "cuisine",
    },
    "form_g": {
        "0": "urban", "1": "rural", "2": "sea", "3": "mountain",
        "4": "lake", "5": "desert", "6": "plains", "7": "jungle",
    },
    "form_h": {"0": "relaxed", "1": "balanced", "2": "active"},
    "form_i": {
        "0": "very_safety_conscious", "1": "balanced", "2": "ready_for_anything",
    },
    "form_j": {
        "0": "off_the_beaten_path", "1": "classic_spot", "2": "mainstream_trendy",
    },
    "form_r": {"0": "anywhere", "1": "specific_regions"},
    "form_rr": {
        "e": "europe", "n": "north_america", "c": "caribbean", "a": "asia",
        "s": "south_america", "m": "middle_east", "f": "africa", "o": "oceania",
    },
}

STRAVL_PROFILE_FIELDS = {
    "age_ranges": "form_a",
    "budget_per_person_per_night": "form_b",
    "season": "form_c",
    "experiences": "form_f",
    "scenery": "form_g",
    "activity_level": "form_h",
    "safety_preference": "form_i",
    "popularity_preference": "form_j",
    "destination_scope": "form_r",
    "regions": "form_rr",
}


def _literal_list(value: Any) -> list[Any]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def _mapped_list(row: dict[str, Any], source_field: str) -> list[str]:
    mapping = STRAVL_MAPS[source_field]
    values = _literal_list(row.get(source_field))
    return [mapping[str(value)] for value in values if str(value) in mapping]


def stravl_profile(row: dict[str, Any]) -> dict[str, Any]:
    return {
        target: _mapped_list(row, source)
        for target, source in STRAVL_PROFILE_FIELDS.items()
    }


def user_hash_split(source: str, user_key: str, seed: int = 42) -> str:
    fraction = int(stable_id(source, seed, user_key)[:12], 16) / 16**12
    return "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"


def _dedupe_ints(values: Iterable[Any]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def iter_stravl_records(
    csv_path: Path,
    destinations_path: Path,
    *,
    seed: int = 42,
) -> Iterable[dict[str, Any]]:
    destinations = [
        line.strip()
        for line in destinations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    label_fields = (
        ("yes", 2, "yes_swipes"),
        ("maybe", 1, "maybe_swipes"),
        ("no", 0, "no_swipes"),
    )
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_user_id = str(row.get("id", "")).strip()
            if not raw_user_id:
                continue
            user_key = stable_id("stravl_user", raw_user_id)[:20]
            split = user_hash_split("stravl", user_key, seed)
            profile = stravl_profile(row)

            labels_by_destination: dict[int, list[tuple[str, int]]] = defaultdict(list)
            for label, ordinal, field in label_fields:
                for destination_id in _dedupe_ints(_literal_list(row.get(field))):
                    labels_by_destination[destination_id].append((label, ordinal))

            for destination_id, labels in labels_by_destination.items():
                # A contradictory swipe is not reliable supervision.
                unique_labels = set(labels)
                if len(unique_labels) != 1 or not 0 <= destination_id < len(destinations):
                    continue
                label, ordinal = labels[0]
                destination = destinations[destination_id]
                yield {
                    "schema_version": 1,
                    "record_id": stable_id(
                        "stravl", user_key, destination_id, label
                    )[:24],
                    "source_dataset": "Stravl-Data",
                    "source_url": "https://github.com/Stravl/Stravl-Data",
                    "task": "explicit_destination_preference",
                    "split": split,
                    "user_key": user_key,
                    "profile": profile,
                    "candidate": {
                        "destination_id": destination_id,
                        "destination": destination,
                    },
                    "label": {"class": label, "ordinal": ordinal},
                }


def infer_parquet_split(path: Path) -> str:
    value = path.as_posix().casefold()
    for split in ("validation", "test", "train"):
        if split in value:
            return split
    return "unspecified"


def iter_genup_records(
    root: Path,
    *,
    dataset_name: str,
    source_url: str,
) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for path in sorted(root.rglob("*.parquet")):
        split = infer_parquet_split(path)
        frame = pd.read_parquet(
            path,
            columns=["system_prompt", "inputs", "targets"],
        )
        for row in frame.to_dict(orient="records"):
            system_prompt = str(row.get("system_prompt") or "").strip()
            inputs = str(row.get("inputs") or "").strip()
            target = str(row.get("targets") or "").strip()
            if not inputs or not target:
                continue
            record_id = stable_id(dataset_name, system_prompt, inputs, target)[:24]
            if record_id in seen:
                continue
            seen.add(record_id)
            yield {
                "schema_version": 1,
                "record_id": record_id,
                "source_dataset": dataset_name,
                "source_url": source_url,
                "task": "next_poi_prediction",
                "split": split,
                "system_prompt": system_prompt,
                "input_text": inputs,
                "target_text": target,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": inputs},
                    {"role": "assistant", "content": target},
                ],
            }


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    total = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            total += 1
            counts[str(record.get("split", "unspecified"))] += 1
    return {"records": total, "splits": dict(sorted(counts.items()))}
