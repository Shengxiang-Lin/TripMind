from __future__ import annotations
import hashlib
import json
import math
import os
import unicodedata
from functools import lru_cache
from pathlib import Path

CITY_NAMES = {
    "北京": "Beijing", "成都": "Chengdu", "重庆": "Chongqing", "广州": "Guangzhou",
    "杭州": "Hangzhou", "南京": "Nanjing", "上海": "Shanghai", "深圳": "Shenzhen",
    "苏州": "Suzhou", "武汉": "Wuhan",
}
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
    "红色景点": ("culture", ["culture", "history", "education"]),
    "大学校园": ("culture", ["culture", "architecture", "walk"]),
    "其它": ("other", []),
}


@lru_cache(maxsize=100000)
def normalized(text: object) -> str:
    from opencc import OpenCC
    # Created once per process; its dictionaries are local package data.
    if not hasattr(normalized, "converter"):
        normalized.converter = OpenCC("t2s")
    value = normalized.converter.convert(unicodedata.normalize("NFKC", str(text)))
    return "".join(c for c in value.casefold() if c.isalnum())


def city_registry() -> dict:
    return {
        city: {"country": "China", "aliases": [city, city+"市", english, english.lower()]}
        for city, english in CITY_NAMES.items()
    }


def canonical_city(value: object, registry: dict | None = None) -> str:
    key = normalized(value)
    for city, data in (registry or city_registry()).items():
        if key in {normalized(x) for x in data["aliases"]}:
            return city
    return str(value).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*values: object) -> str:
    data = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def atomic_json(path: Path, data: object) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_json(temp, data)
    os.replace(temp, path)


def coordinate_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1]-a[1])
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dl/2)**2
    return 6371000 * 2 * math.asin(min(1, math.sqrt(h)))
