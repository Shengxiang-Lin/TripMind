from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import POI

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pois(path: str | Path | None = None) -> List[POI]:
    path = Path(path) if path else PROJECT_ROOT / "data" / "poi" / "hangzhou.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return [POI(**record) for record in records]


def city_pois(pois: Iterable[POI], city: str) -> List[POI]:
    city = city.strip().lower()
    return [poi for poi in pois if poi.city.lower() == city]
