from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import POI

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pois(path: str | Path | None = None) -> List[POI]:
    """Load one POI JSON file or every JSON file under data/poi.

    This keeps the v0.1 Hangzhou file working while making the data layer ready
    for multiple cities and countries.
    """
    target = Path(path) if path else PROJECT_ROOT / "data" / "poi"
    files = [target] if target.is_file() else sorted(target.rglob("*.json"))
    records = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"POI file must contain a JSON list: {file}")
        records.extend(payload)
    return [POI(**record) for record in records]


def city_pois(
    pois: Iterable[POI],
    city: str,
    country: str | None = None,
) -> List[POI]:
    city_key = city.strip().casefold()
    country_key = (country or "").strip().casefold()
    result = []
    for poi in pois:
        if poi.city.strip().casefold() != city_key:
            continue
        if country_key and poi.country.strip().casefold() != country_key:
            continue
        result.append(poi)
    return result
