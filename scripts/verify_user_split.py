"""Check that GenUP train, validation and test users are disjoint."""

import argparse
from itertools import combinations
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")


def read_split(path: Path, split: str) -> tuple[int, set[str]]:
    count = 0
    users = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["user_key"]
            if row["split"] != split or not isinstance(key, str) or not key:
                raise ValueError(f"Invalid split or user_key in {path} at context {count + 1}")
            users.add(key)
            count += 1
    return count, users


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    sources = sorted({
        path.stem.removesuffix(f"-{split}")
        for split in SPLITS
        for path in root.glob(f"*-{split}.jsonl")
    })
    if not sources:
        raise FileNotFoundError(f"No GenUP split files under {root}")

    all_users = {split: set() for split in SPLITS}
    overlap_found = False
    for source in sources:
        print(source)
        users = {}
        for split in SPLITS:
            count, users[split] = read_split(root / f"{source}-{split}.jsonl", split)
            all_users[split].update(users[split])
            print(f"  {split}: {count} contexts, {len(users[split])} users")
        for left, right in combinations(SPLITS, 2):
            overlap = len(users[left] & users[right])
            overlap_found |= overlap > 0
            print(f"  {left}/{right} user overlap = {overlap}")
    print("all sources")
    for left, right in combinations(SPLITS, 2):
        overlap = len(all_users[left] & all_users[right])
        overlap_found |= overlap > 0
        print(f"  {left}/{right} user overlap = {overlap}")
    if overlap_found:
        raise SystemExit("GenUP user splits overlap")


if __name__ == "__main__":
    main()
