from __future__ import annotations
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import asdict
import pandas as pd
from tripmind.models import POI
from tripmind.planner import _to_min
from .common import CATEGORY_TAGS, CITY_NAMES, canonical_city, city_registry, normalized, coordinate_distance, stable_id, write_json

TABLES = ("attractions", "restaurants", "accommodations", "trains", "flights", "poi", "subway_stations")


def normalized_hours(opening: str, closing: str) -> tuple[str, str, list[str]]:
    start, end = _to_min(opening), _to_min(closing)
    notes = []
    if start == 1440:
        raise ValueError("opening_at_end_of_day")
    if end < start or start == end == 0:
        closing = "24:00"
        notes.append("overnight_evening_window" if end < start else "midnight_pair_assumed_all_day")
    elif end == start:
        raise ValueError("empty_opening_interval")
    return opening, closing, notes


def normalize_attraction(row: dict, english: dict | None = None) -> POI:
    city = canonical_city(row["city"])
    if city not in CITY_NAMES or not str(row.get("name", "")).strip():
        raise ValueError("missing_identity")
    lat, lon, price = (float(row[k]) for k in ["lat", "lon", "price"])
    low, high = (float(row[k]) for k in ["recommendmintime", "recommendmaxtime"])
    if not all(map(math.isfinite, [lat, lon, price, low, high])):
        raise ValueError("nonfinite_numeric")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("invalid_coordinates")
    if price < 0 or low < 0 or high <= 0 or high < low:
        raise ValueError("invalid_cost_or_duration")
    opening, closing, notes = normalized_hours(str(row["opentime"]), str(row["endtime"]))
    category, tags = CATEGORY_TAGS.get(str(row["type"]), ("other", []))
    duration = max(1, round((low+high)*30))
    source_id = int(row["id"])
    aliases = [str(english["name"])] if english and english.get("name") else []
    return POI(
        poi_id=f"chinatravel:{city}:{source_id}", city=city, name=str(row["name"]).strip(),
        category=category, longitude=lon, latitude=lat, ticket_price=price,
        recommended_duration=duration, opening_time=opening, closing_time=closing,
        popularity=.5, tags=list(tags), suitable_groups=[],
        intensity="medium" if duration > 120 else "low",
        description=f"{row['type']}；价格与营业时间来自 ChinaTravel 静态快照。",
        indoor=True if category in {"museum", "gallery"} else None,
        source="https://huggingface.co/datasets/LAMDA-NeSy/ChinaTravel-Sandbox",
        aliases=aliases,
        source_metadata={
            "license": "CC BY 4.0", "source_rows": [{"city":city,"id":source_id}],
            "original_opening_time": str(row["opentime"]), "original_closing_time": str(row["endtime"]),
            "recommended_hours": [low,high], "coordinate_system": "unspecified_by_source",
            "derived_fields": ["category", "tags", "intensity", "neutral_popularity", "midpoint_duration_minutes"],
            "normalization_notes": notes,
        },
    )


def build_catalog(raw: Path, out: Path, cities: list[str] | None = None) -> dict:
    root = raw/"chinatravel-sandbox/data"
    target_cities = set(cities or CITY_NAMES)
    write_json(out/"cities.json", {k:v for k,v in city_registry().items() if k in target_cities})
    english = pd.read_parquet(root/"en/attractions.parquet").to_dict(orient="records")
    en_lookup = {(canonical_city(r["city"]),int(r["id"])):r for r in english}
    rows = pd.read_parquet(root/"zh/attractions.parquet").sort_values(["city","id"]).to_dict(orient="records")
    groups = defaultdict(list)
    pois, rejected, duplicates = [], [], []
    reasons, notes = Counter(), Counter()
    for row in rows:
        city = canonical_city(row["city"])
        if city not in target_cities:
            continue
        try:
            poi = normalize_attraction(row, en_lookup.get((city,int(row["id"]))))
        except (ValueError, TypeError, KeyError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            reasons[reason] += 1
            rejected.append({"reason":reason,"row":row})
            continue
        key = (poi.city, normalized(poi.name))
        duplicate = next((p for p in groups[key] if coordinate_distance(
            (p.latitude,p.longitude),(poi.latitude,poi.longitude)) <= 100), None)
        if duplicate:
            duplicate.aliases = sorted(set(duplicate.aliases+poi.aliases))
            duplicate.source_metadata["source_rows"].extend(poi.source_metadata["source_rows"])
            duplicates.append({"kept":duplicate.poi_id,"merged":poi.poi_id})
            continue
        groups[key].append(poi)
        pois.append(poi)
        notes.update(poi.source_metadata["normalization_notes"])
    if not pois:
        raise ValueError("No usable attractions survived quality checks")
    (out/"poi").mkdir()
    counts = Counter(p.city for p in pois)
    for city in sorted(counts):
        write_json(out/"poi"/f"{CITY_NAMES[city].lower()}.json", [asdict(p) for p in pois if p.city==city])
    with (out/"quarantine_pois.jsonl").open("w",encoding="utf-8") as f:
        for record in rejected:
            original = {k:(repr(v) if isinstance(v,float) and not math.isfinite(v) else v)
                        for k,v in record["row"].items()}
            f.write(json.dumps({**record,"row":original},ensure_ascii=False,allow_nan=False)+"\n")
    write_json(out/"poi_duplicates.json", duplicates)
    db = sqlite3.connect(out/"catalog.sqlite")
    summaries = {}
    try:
        for table in TABLES:
            db.execute(f'CREATE TABLE "{table}" (record_key TEXT PRIMARY KEY, city TEXT, language TEXT, source_file TEXT, license TEXT, payload TEXT)')
            total = inserted = 0
            for language in ["zh","en"]:
                path = root/language/f"{table}.parquet"
                frame = pd.read_parquet(path).astype(object)
                frame = frame.where(pd.notna(frame),None)
                frame = frame.replace([float("inf"),float("-inf")],None)
                for record in frame.to_dict(orient="records"):
                    city = canonical_city(record["city"]) if record.get("city") else "*"
                    if city != "*" and city not in target_cities:
                        continue
                    total += 1
                    payload = json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False)
                    cursor = db.execute(f'INSERT OR IGNORE INTO "{table}" VALUES(?,?,?,?,?,?)',
                        (stable_id(table,language,record),city,language,str(path.relative_to(raw)),"CC BY 4.0",payload))
                    inserted += cursor.rowcount
            summaries[table] = {"input_rows":total,"stored_rows":inserted,"exact_duplicates":total-inserted}
        db.commit()
    finally:
        db.close()
    return {"input_attractions":sum(1 for r in rows if canonical_city(r["city"]) in target_cities),
            "accepted_pois":len(pois),"rejected_pois":len(rejected),"merged_pois":len(duplicates),
            "rejection_reasons":dict(reasons),"normalization_notes":dict(notes),
            "cities":dict(sorted(counts.items())),"reference_tables":summaries}
