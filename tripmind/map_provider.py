from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .models import POI, RouteLeg


class MapProvider(ABC):
    @abstractmethod
    def route(self, origin: POI, destination: POI, mode: str = "transit") -> RouteLeg:
        raise NotImplementedError


class LocalMapProvider(MapProvider):
    """Offline fallback.

    Distances use Haversine distance and a detour factor; durations use simple
    average speeds. Values are estimates and are clearly separated from future
    Amap-grounded results.
    """

    SPEED_KMH = {"walking": 4.5, "transit": 22.0, "driving": 28.0}
    DETOUR = {"walking": 1.18, "transit": 1.32, "driving": 1.25}

    @staticmethod
    def _haversine(a: POI, b: POI) -> float:
        r = 6371.0
        lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
        lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * r * math.asin(math.sqrt(h))

    def route(self, origin: POI, destination: POI, mode: str = "transit") -> RouteLeg:
        mode = mode if mode in self.SPEED_KMH else "transit"
        distance = self._haversine(origin, destination) * self.DETOUR[mode]
        duration = max(5, round(distance / self.SPEED_KMH[mode] * 60 + (6 if mode == "transit" else 0)))
        return RouteLeg(
            origin=origin.name,
            destination=destination.name,
            distance_km=round(distance, 2),
            duration_min=int(duration),
            mode=mode,
        )
