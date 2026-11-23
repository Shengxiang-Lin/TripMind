"""Create the reproducible ChinaTravel SFT splits."""

import hashlib
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA / "sft_seed_chinatravel.jsonl"
SEED = 42
SIZES = {"train": 240, "val": 30, "test": 30}


def main() -> None:
    source = SOURCE.read_bytes()
    lines = source.decode("utf-8").splitlines()
    if len(lines) != sum(SIZES.values()):
        raise ValueError(f"Expected {sum(SIZES.values())} rows, got {len(lines)}")
    for line in lines:
        record = json.loads(line)
        if not all(isinstance(record.get(key), str) for key in ("instruction", "input", "output")):
            raise ValueError("Expected Alpaca string fields on every line")

    line_numbers = list(range(1, len(lines) + 1))
    random.Random(SEED).shuffle(line_numbers)
    manifest = {
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "seed": SEED,
        "sizes": SIZES,
        "rows": [],
    }
    offset = 0
    for split, size in SIZES.items():
        selected = line_numbers[offset:offset + size]
        payload = "".join(lines[number - 1] + "\n" for number in selected)
        (DATA / f"sft_{split}.jsonl").write_text(payload, encoding="utf-8", newline="\n")
        manifest["rows"].extend({"source_line": number, "split": split} for number in selected)
        offset += size
    (DATA / "sft_split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(SIZES)


if __name__ == "__main__":
    main()
