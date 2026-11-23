"""Evaluate base, v1 and v2 once on the held-out ChinaTravel split."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import V1_ADAPTER, V2_ADAPTER, generate_recommendation

TEST = ROOT / "data" / "sft_test.jsonl"
RESULT = ROOT / "data" / "sft_eval_result.json"
NOTE = "v1 trained on these test rows. Its metrics are for reference only."
METADATA = re.compile(r"出发地：(.+?)，目的地：(.+?)，天数：([0-9]+)，人数：([0-9]+)")
FIELDS = ("structure", "start", "target", "days", "people")


def score(text: str, start: str, target: str, days: str, people: str) -> dict:
    return {
        "structure": text.startswith(f"建议从{start}前往{target}") and f"安排{days}天行程" in text,
        "start": bool(re.search(rf"(?:从|出发地[：:]\s*){re.escape(start)}(?:出发|前往|到|[，。\s]|$)", text)),
        "target": bool(re.search(rf"(?:前往|抵达|到达?|目的地[：:]\s*){re.escape(target)}(?:[，。\s]|$)", text)),
        "days": bool(re.search(rf"(?<![0-9]){days}\s*天", text)),
        "people": bool(re.search(rf"(?<![0-9]){people}\s*(?:个?人|位)", text)),
        "characters": len(text),
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(f"Test was already evaluated: {RESULT}")
    rows = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 30:
        raise ValueError(f"Expected 30 held-out rows, got {len(rows)}")
    cases = []
    for row in rows:
        match = METADATA.fullmatch(row["input"].rsplit("\n", 1)[-1])
        if match is None:
            raise ValueError("Missing structured metadata in held-out input")
        cases.append((row, match.groups()))

    metrics = {}
    for name, adapter in (("base", None), ("v1", V1_ADAPTER), ("v2", V2_ADAPTER)):
        print(f"Evaluating {name} ({len(cases)} rows)", flush=True)
        scores = [
            score(generate_recommendation(row["input"], instruction=row["instruction"], adapter_dir=adapter), *expected)
            for row, expected in cases
        ]
        metrics[name] = {
            **{field: sum(row[field] for row in scores) / len(scores) for field in FIELDS},
            "average_characters": sum(row["characters"] for row in scores) / len(scores),
        }

    with RESULT.open("x", encoding="utf-8") as file:
        json.dump({"samples": len(rows), "note": NOTE, "metrics": metrics}, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("| Model | Structure | Start | Target | Days | People | Avg. chars |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, values in metrics.items():
        rates = " | ".join(f"{values[field]:.1%}" for field in FIELDS)
        print(f"| {name} | {rates} | {values['average_characters']:.1f} |")
    print(NOTE)


if __name__ == "__main__":
    main()
