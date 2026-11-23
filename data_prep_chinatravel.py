"""Convert the ChinaTravel easy split to Alpaca JSONL."""

import argparse
import json
from pathlib import Path

INSTRUCTION = "请根据用户的需求，制定一份合理的旅行规划。"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "sft_seed_chinatravel.jsonl"


def build_input(query: str, start: str, target: str, days: int, people: int) -> str:
    return (
        f"{query.strip()}\n"
        f"出发地：{start}，目的地：{target}，天数：{days}，人数：{people}"
    )


def build_output(start_city: str, target_city: str, days: int, people_number: int) -> str:
    days = int(days)
    schedule = ["第1天：抵达后在同一区域安排一个主要景点，附近用餐，减少来回奔波。"]
    for day in range(2, days):
        schedule.append(f"第{day}天：按兴趣选一个片区游览，午后留出休息时间。")
    if days > 1:
        schedule.append(f"第{days}天：在住宿地附近安排轻松活动，取行李后返程。")
    else:
        schedule.append("傍晚预留前往车站或机场的时间，当天返程。")

    return "\n".join([
        f"建议从{start_city}前往{target_city}，安排{days}天行程，适合{people_number}人出行。",
        *schedule,
        "先按预算预留往返交通和餐饮费用，需要过夜时再计入住宿。出发前确认景点开放时间和预约要求。",
    ])


def convert_row(row: dict) -> dict[str, str]:
    start_city = row["start_city"]
    target_city = row["target_city"]
    days = row["days"]
    people_number = row["people_number"]
    return {
        "instruction": INSTRUCTION,
        "input": build_input(row["nature_language"], start_city, target_city, days, people_number),
        "output": build_output(start_city, target_city, days, people_number),
    }


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-csv", type=Path, help="Path to a downloaded ChinaTravel easy.csv")
    args = parser.parse_args()

    if args.local_csv is None:
        dataset = load_dataset("LAMDA-NeSy/ChinaTravel", "default", split="easy")
    else:
        dataset = load_dataset("csv", data_files={"easy": str(args.local_csv)}, split="easy")
    records = [convert_row(row) for row in dataset]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"已写入 {len(records)} 条样本：{OUTPUT_PATH}")
    for record in records[:2]:
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
