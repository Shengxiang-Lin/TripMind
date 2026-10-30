from __future__ import annotations
import ast
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from .common import canonical_city, stable_id, write_json


def parse_programs(value: str, allow_single: bool = False) -> list[str]:
    if not value:
        return []
    if len(value) > 200000:
        raise ValueError("oversized_constraint_payload")
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(value)  # Literal containers only, never eval/exec.
        except (ValueError,SyntaxError):
            if allow_single:
                return [value]  # Some preference0 rows contain a single DSL program.
            raise
    if isinstance(result,str) and allow_single:
        return [result]
    if not isinstance(result,list) or not all(isinstance(x,str) for x in result):
        raise ValueError("constraint_programs_not_string_list")
    return result


def query_family(uid: str) -> str:
    uid = re.sub(r"^p\d+(?=[eh]\d)", "", uid)
    match = re.fullmatch(r"phase2_extended_(\d+)_\d+", uid)
    return match.group(1) if match else uid


def normalize_query(row: dict, source: str) -> dict:
    uid = row.get("uid","").strip()
    text = row.get("nature_language","").strip()
    if not uid or not text:
        raise ValueError("missing_uid_or_request")
    day_value, count_value = row.get("days"),row.get("people_number")
    days = int(day_value) if day_value else None
    people = int(count_value) if count_value else None
    if days is not None and days < 1 or people is not None and people < 1:
        raise ValueError("nonpositive_trip_metadata")
    return {
        "uid":uid,"family_key":query_family(uid),"request":text,
        "request_en":row.get("nature_language_en","").strip(),
        "start_city":canonical_city(row["start_city"]) if row.get("start_city") else None,
        "target_city":canonical_city(row["target_city"]) if row.get("target_city") else None,
        "days":days,"people_number":people,
        "constraint_programs":parse_programs(row.get("hard_logic_py","")),
        "constraint_descriptions":parse_programs(row.get("hard_logic_nl","")),
        "preference":row.get("preference", ""),"preference_en":row.get("preference_en", ""),
        "preference_programs":parse_programs(row.get("preference_py", ""),allow_single=True),
        "source_files":[source],"source_tags":[row.get("tag",Path(source).stem)],
        "official_holdout":Path(source).name in {"human1000.csv","competition_test.csv"}
            or str(row.get("official_phase2_evaluation","")).casefold()=="true",
        "profile_metadata_complete":bool(row.get("target_city") and days and people),
        "label_status":"unlabeled_request",
    }


def assign_splits(records: list[dict], seed: int = 42) -> None:
    # Union families AND identical request text, so duplicate IDs/text cannot leak.
    parent = {r["family_key"]:r["family_key"] for r in records}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    seen_text = {}
    for r in records:
        text = "".join(unicodedata.normalize("NFKC",r["request"]).casefold().split())
        if text in seen_text: union(r["family_key"],seen_text[text])
        else: seen_text[text] = r["family_key"]
    heldout = {find(r["family_key"]) for r in records if r["official_holdout"]}
    for r in records:
        group = find(r["family_key"])
        fraction = int(stable_id(seed,group)[:12],16) / 16**12
        r["split_group"] = group
        r["split"] = "test" if group in heldout else ("train" if fraction<.8 else "validation" if fraction<.9 else "test")
        r["split_reason"] = "official_holdout_family" if group in heldout else "seeded_family_hash"


def build_queries(raw: Path, out: Path, seed: int = 42, cities: list[str] | None = None) -> dict:
    root = raw/"chinatravel"
    records, quarantine, unique = [], [], {}
    input_rows = duplicates = 0
    for path in sorted(root.rglob("*.csv")):
        with path.open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                input_rows += 1
                source = str(path.relative_to(raw))
                try: record = normalize_query(row,source)
                except (ValueError,TypeError,SyntaxError) as exc:
                    quarantine.append({"source":source,"uid":row.get("uid"),"reason":str(exc)})
                    continue
                if cities and record["target_city"] and record["target_city"] not in cities: continue
                identity = stable_id({k:v for k,v in record.items() if k not in {"source_files","source_tags","official_holdout"}})
                if identity in unique:
                    existing = unique[identity]
                    existing["source_files"] = sorted(set(existing["source_files"]+record["source_files"]))
                    existing["source_tags"] = sorted(set(existing["source_tags"]+record["source_tags"]))
                    existing["official_holdout"] |= record["official_holdout"]
                    duplicates += 1
                else:
                    record["record_id"] = identity
                    records.append(record); unique[identity]=record
    assign_splits(records,seed)
    folder = out/"queries"; folder.mkdir()
    records.sort(key=lambda r:r["record_id"])
    for split in ["all","train","validation","test"]:
        with (folder/f"{split}.jsonl").open("w",encoding="utf-8") as f:
            for record in records:
                if split=="all" or record["split"]==split:
                    f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")
    write_json(out/"quarantine_queries.json",quarantine)
    return {"input_rows":input_rows,"records":len(records),"exact_duplicates":duplicates,
            "rejected":len(quarantine),"splits":dict(Counter(r["split"] for r in records)),
            "groups":len({r["split_group"] for r in records}),
            "official_holdout_records":sum(r["official_holdout"] for r in records),
            "partial_profile_records":sum(not r["profile_metadata_complete"] for r in records),
            "seed":seed,"labels":"none; these are request splits, not SFT targets"}
