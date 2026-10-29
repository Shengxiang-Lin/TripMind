from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

from .models import POI, RouteLeg, RouteMatrix


class MapProvider(ABC):
    @abstractmethod
    def route(self, origin: POI, destination: POI, mode: str = "walking") -> RouteLeg:
        raise NotImplementedError

    def supports(self, mode: str) -> bool:
        return True

    def matrix(self, pois: Sequence[POI], mode: str = "walking") -> RouteMatrix:
        """Default matrix implementation using pairwise route calls.

        Network providers should override this with their native matrix endpoint.
        """
        n = len(pois)
        distances: list[list[float | None]] = [[0.0] * n for _ in range(n)]
        durations: list[list[int | None]] = [[0] * n for _ in range(n)]
        grounded = True
        source = self.__class__.__name__
        for i, origin in enumerate(pois):
            for j, destination in enumerate(pois):
                if i == j:
                    continue
                leg = self.route(origin, destination, mode)
                distances[i][j] = leg.distance_km
                durations[i][j] = leg.duration_min
                grounded = grounded and leg.grounded
                source = leg.source
        return RouteMatrix(
            names=[p.name for p in pois],
            distances_km=distances,
            durations_min=durations,
            source=source,
            grounded=grounded,
        )


class LocalMapProvider(MapProvider):
    """Offline fallback based on Haversine distance.

    These values are estimates, not road-network ground truth.
    """

    SPEED_KMH = {
        "walking": 4.5,
        "transit": 22.0,
        "driving": 28.0,
        "cycling": 14.0,
    }
    DETOUR = {
        "walking": 1.18,
        "transit": 1.32,
        "driving": 1.25,
        "cycling": 1.20,
    }

    @staticmethod
    def _haversine(a: POI, b: POI) -> float:
        r = 6371.0
        lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
        lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        return 2 * r * math.asin(math.sqrt(h))

    def route(self, origin: POI, destination: POI, mode: str = "walking") -> RouteLeg:
        mode = mode if mode in self.SPEED_KMH else "walking"
        distance = self._haversine(origin, destination) * self.DETOUR[mode]
        duration = max(
            5,
            round(
                distance / self.SPEED_KMH[mode] * 60
                + (6 if mode == "transit" else 0)
            ),
        )
        return RouteLeg(
            origin=origin.name,
            destination=destination.name,
            distance_km=round(distance, 2),
            duration_min=int(duration),
            mode=mode,
            source="local_haversine_estimate",
            grounded=False,
        )


class FallbackMapProvider(MapProvider):
    """Use a real routing provider when possible, otherwise fall back locally."""

    def __init__(self, primary: MapProvider, fallback: MapProvider | None = None):
        self.primary = primary
        self.fallback = fallback or LocalMapProvider()

    def supports(self, mode: str) -> bool:
        return self.primary.supports(mode) or self.fallback.supports(mode)

    def route(self, origin: POI, destination: POI, mode: str = "walking") -> RouteLeg:
        if self.primary.supports(mode):
            try:
                return self.primary.route(origin, destination, mode)
            except Exception:
                pass
        return self.fallback.route(origin, destination, mode)

    def matrix(self, pois: Sequence[POI], mode: str = "walking") -> RouteMatrix:
        if self.primary.supports(mode):
            try:
                return self.primary.matrix(pois, mode)
            except Exception:
                pass
        return self.fallback.matrix(pois, mode)
