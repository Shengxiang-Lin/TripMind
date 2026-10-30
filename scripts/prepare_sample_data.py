"""Prepare a bounded Hangzhou sample from downloaded data, without calling APIs.

Category tags and neutral popularity are baseline defaults, not learned labels.
Raw source files and the bundled demo catalogue are never overwritten.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import html
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from tripmind.config import get_config, resolve_project_path
from tripmind.models import POI
from tripmind.planner import _to_min

CATEGORY_TAGS = {
    "历史古迹": ("culture", ["culture", "history", "architecture"]),
    "自然风光": ("nature", ["nature", "photography", "scenic", "outdoor"]),
    "公园": ("park", ["nature", "park", "walk", "outdoor"]),
    "博物馆/纪念馆": ("museum", ["museum", "culture", "education"]),
    "美术馆/艺术馆": ("gallery", ["museum", "culture", "photography"]),
    "商业街区": ("shopping", ["shopping", "food", "walk"]),
    "文化旅游区": ("culture", ["culture", "history", "photography"]),
    "人文景观": ("culture", ["culture", "architecture", "photography"]),
    "游乐园/体育娱乐": ("entertainment", ["entertainment", "family"]),
}


def distance_from_center(lat: float, lon: float) -> float:
    a, b = math.radians(30.25), math.radians(lat)
    delta = math.radians(lon - 120.15)
    h = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(delta/2)**2
    return 6371 * 2 * math.asin(min(1, math.sqrt(h)))


def clean_wikitext(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    def template(match):
        values = []
        for param in match.group(0)[2:-2].split("|")[1:]:
            key, sep, value = param.partition("=")
            if sep and key.strip().lower() in {"name", "name-local", "content", "description"}:
                values.append(value.strip())
        return " ".join(values)
    while re.search(r"\{\{[^{}]*\}\}", text):
        text = re.sub(r"\{\{[^{}]*\}\}", template, text)
    text = re.sub(r"\[\[(?:File|Image|文件|檔案|Category|分类|分類):[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[([^\[\]]+)\]\]", lambda m: m.group(1).split("|")[-1], text)
    text = re.sub(r"\[https?://\S+(?:\s+([^\]]*))?\]", lambda m: m.group(1) or "", text)
    text = re.sub(r"<[^>]+>", "", text).replace("'''", "").replace("''", "")
    text = html.unescape(text)
    return "\n".join(re.sub(r"^[*#:;]+\s*", "", line).strip() for line in text.splitlines())


def wiki_chunks(path: Path) -> list[dict]:
    text = None
    with bz2.open(path, "rb") as f:
        for _, page in ET.iterparse(f, events=("end",)):
            if not page.tag.endswith("}page"):
                continue
            title = next((x.text for x in page if x.tag.endswith("}title")), None)
            if title == "杭州":
                text = next((x.text for x in page.iter() if x.tag.endswith("}text")), "") or ""
                break
            page.clear()
    if not text:
        raise ValueError("Hangzhou article missing from Chinese Wikivoyage dump")
    docs = []
    heading = "概览"
    sections = re.split(r"(?m)^={2,6}\s*(.*?)\s*={2,6}\s*$", text)
    for index, section in enumerate(sections):
        if index % 2:
            heading = section.strip()
            continue
        cleaned = clean_wikitext(section)
        for start in range(0, len(cleaned), 800):
            chunk = cleaned[start:start+800].strip()
            if len(chunk) < 40:
                continue
            docs.append({
                "title": f"杭州 · {heading}", "city": "杭州", "country": "China",
                "tags": ["culture", "history", "nature", "photography", "food"],
                "text": chunk,
                "source": "https://zh.wikivoyage.org/wiki/" + quote("杭州"),
                "license": "CC BY-SA 4.0", "source_file": str(path.relative_to(ROOT)),
            })
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", default="data/processed/sample")
    args = parser.parse_args()
    if not 2 <= args.limit <= 25:
        parser.error("--limit must be between 2 and 25 for a small API run")
    import pandas as pd
    config = get_config()
    raw = resolve_project_path(config["data"].get("raw_dir", "data/raw"))
    out = resolve_project_path(args.output)
    table = raw / "chinatravel-sandbox/data/zh/attractions.parquet"
    df = pd.read_parquet(table)
    rows = df[df.city == "杭州"].sort_values("id").to_dict(orient="records")
    pois = []
    for row in rows:
        lat, lon = float(row["lat"]), float(row["lon"])
        if not all(map(math.isfinite, [lat, lon])) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if distance_from_center(lat, lon) > 25:
            continue
        opening, closing = str(row["opentime"]), str(row["endtime"])
        if _to_min(closing) <= _to_min(opening):
            continue  # Overnight intervals need a richer opening-hours model.
        category, tags = CATEGORY_TAGS.get(row["type"], ("other", ["culture"]))
        minimum, maximum = float(row["recommendmintime"]), float(row["recommendmaxtime"])
        price = float(row["price"])
        if not all(map(math.isfinite, [minimum, maximum, price])) or minimum <= 0 or maximum < minimum or price < 0:
            continue
        duration = max(15, round((minimum + maximum) / 2 * 60))
        poi = POI(
            poi_id=f"chinatravel:zh:杭州:{int(row['id'])}", city="杭州", name=row["name"],
            category=category, longitude=lon, latitude=lat, ticket_price=price,
            recommended_duration=duration, opening_time=opening, closing_time=closing,
            popularity=0.5, tags=list(tags), suitable_groups=[],
            intensity="medium" if duration > 120 else "low",
            indoor=True if category in {"museum", "gallery"} else None,
            description=f"ChinaTravel 静态景点数据：{row['type']}。票价和开放时间为数据快照。",
            source="https://huggingface.co/datasets/LAMDA-NeSy/ChinaTravel-Sandbox",
        )
        pois.append(poi.__dict__)
        if len(pois) == args.limit:
            break
    if len(pois) < 2:
        raise ValueError("Too few usable attractions")
    docs = wiki_chunks(raw / "wikivoyage/zh/zhwikivoyage-latest-pages-articles.xml.bz2")
    with (raw / "chinatravel/easy.csv").open(encoding="utf-8-sig", newline="") as f:
        queries = [r for r in csv.DictReader(f) if r["target_city"] == "杭州"][:3]
    if not queries:
        raise ValueError("No Hangzhou benchmark queries found")
    (out / "poi").mkdir(parents=True, exist_ok=True)
    for path, value in [(out/"poi/hangzhou.json", pois), (out/"knowledge.json", docs), (out/"queries.json", queries)]:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    # Copy only tracked defaults, never effective config or API keys.
    sample_config = yaml.safe_load((ROOT/"configs/config.yaml").read_text())
    for section in ("ors", "dashscope"):
        sample_config[section]["api_key"] = ""
    sample_config["data"]["use_processed"] = False
    sample_config["data"]["poi_dir"] = str(out/"poi")
    sample_config["data"]["knowledge_file"] = str(out/"knowledge.json")
    (out/"config.yaml").write_text(yaml.safe_dump(sample_config, allow_unicode=True, sort_keys=False))
    manifest = {
        "city": "杭州", "poi_count": len(pois), "knowledge_chunks": len(docs), "query_count": len(queries),
        "selection": "First source-ordered usable POIs within 25 km of 30.25N,120.15E",
        "derived_fields": "Category tags, intensity and neutral popularity are heuristic defaults; duration is midpoint in minutes.",
        "poi_source": str(table), "wiki_source": docs[0]["source"],
        "licenses": {"ChinaTravel": "CC BY 4.0", "Wikivoyage": "CC BY-SA 4.0"},
    }
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Sample config:", out/"config.yaml")


if __name__ == "__main__":
    main()
