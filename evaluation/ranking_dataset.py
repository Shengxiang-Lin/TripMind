from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def validate_label(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) not in {2, 3}:
        raise ValueError(f"{row.get('sample_id')}: messages must contain 2 or 3 turns")
    roles = [message.get("role") for message in messages]
    if roles[:2] != ["system", "user"] or (len(roles) == 3 and roles[2] != "assistant"):
        raise ValueError(f"{row.get('sample_id')}: invalid message roles")
    if len(messages) == 2:
        return False

    payload = json.loads(messages[2]["content"])
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError(f"{row.get('sample_id')}: missing ranking list")
    candidates = row.get("candidate_poi_ids")
    if not isinstance(candidates, list) or len(candidates) < 2 or len(candidates) != len(set(candidates)):
        raise ValueError(f"{row.get('sample_id')}: invalid candidate list")
    ranked_ids = [item.get("poi_id") for item in ranking if isinstance(item, dict)]
    if len(ranked_ids) != len(candidates) or set(ranked_ids) != set(candidates) or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError(f"{row.get('sample_id')}: ranking must contain every candidate exactly once")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TripMind ranking dataset splits.")
    parser.add_argument("--root", default="data/processed/ranking")
    args = parser.parse_args()
    root = Path(args.root)

    seen_sample_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = {}
    summary = {}
    for split in ["train", "validation", "test"]:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        groups: set[str] = set()
        labeled = 0
        for row in rows:
            if row.get("split") != split:
                raise ValueError(f"{row.get('sample_id')}: split field mismatch")
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate or empty sample_id: {sample_id!r}")
            seen_sample_ids.add(sample_id)
            group = str(row.get("split_group", ""))
            if not group:
                raise ValueError(f"{sample_id}: missing split_group")
            groups.add(group)
            labeled += int(validate_label(row))
        groups_by_split[split] = groups
        summary[split] = {"rows": len(rows), "labeled": labeled, "groups": len(groups)}

    split_names = list(groups_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ValueError(f"Split-group leakage between {left} and {right}: {len(overlap)} groups")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Ranking dataset validation: OK")


if __name__ == "__main__":
    main()
