from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import POI
from .processing.common import normalized

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICE_OVERRIDES = PROJECT_ROOT / "data" / "price_overrides.json"
PRICE_NAME_ALIASES = {
    "苏州": {"虎丘山风景名胜区": "虎丘"},
    "杭州": {"岳王庙景区": "岳王庙", "西溪国家湿地公园": "西溪湿地"},
}


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
    overrides = json.loads(PRICE_OVERRIDES.read_text(encoding="utf-8"))
    prices = {
        city: {normalized(name): price for name, price in city_prices.items()}
        for city, city_prices in overrides["prices"].items()
    }
    pois = [POI(**record) for record in records]
    for poi in pois:
        if poi.ticket_price > 0 or poi.currency != "CNY":
            continue
        city_prices = prices.get(poi.city, {})
        aliases = PRICE_NAME_ALIASES.get(poi.city, {})
        for name in (poi.name, *poi.aliases):
            key = normalized(aliases.get(name, name))
            if key in city_prices:
                poi.ticket_price = city_prices[key]
                poi.source_metadata.update(price_source="manual_demo", price_note=overrides["note"])
                break
    return pois


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
