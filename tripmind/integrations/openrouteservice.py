from __future__ import annotations

import os
from typing import Sequence

import requests

from ..map_provider import MapProvider
from ..models import POI, RouteLeg, RouteMatrix


class OpenRouteServiceProvider(MapProvider):
    """Road-network routing backed by openrouteservice.

    ORS does not provide public-transit routing. TripMind therefore exposes
    walking/driving/cycling here and lets FallbackMapProvider handle unsupported
    modes explicitly instead of pretending they are grounded transit routes.
    """

    PROFILE_BY_MODE = {
        "walking": "foot-walking",
        "driving": "driving-car",
        "cycling": "cycling-regular",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ):
        self.api_key = api_key or os.getenv("ORS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ORS_API_KEY is not set.")
        self.base_url = (
            base_url
            or os.getenv("ORS_BASE_URL")
            or "https://api.openrouteservice.org"
        ).rstrip("/")
        self.timeout = timeout

    def supports(self, mode: str) -> bool:
        return mode in self.PROFILE_BY_MODE

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def route(self, origin: POI, destination: POI, mode: str = "walking") -> RouteLeg:
        if not self.supports(mode):
            raise ValueError(f"openrouteservice does not support mode={mode!r}")

        profile = self.PROFILE_BY_MODE[mode]
        url = f"{self.base_url}/v2/directions/{profile}/geojson"
        response = requests.post(
            url,
            headers=self._headers(),
            json={
                "coordinates": [
                    [origin.longitude, origin.latitude],
                    [destination.longitude, destination.latitude],
                ],
                "instructions": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        feature = payload["features"][0]
        summary = feature["properties"]["summary"]
        geometry = feature.get("geometry", {}).get("coordinates")
        return RouteLeg(
            origin=origin.name,
            destination=destination.name,
            distance_km=round(float(summary["distance"]) / 1000.0, 2),
            duration_min=max(1, round(float(summary["duration"]) / 60.0)),
            mode=mode,
            source="openrouteservice",
            grounded=True,
            geometry=geometry,
        )

    def matrix(self, pois: Sequence[POI], mode: str = "walking") -> RouteMatrix:
        if not self.supports(mode):
            raise ValueError(f"openrouteservice does not support mode={mode!r}")
        if not pois:
            return RouteMatrix([], [], [], "openrouteservice", True)

        profile = self.PROFILE_BY_MODE[mode]
        url = f"{self.base_url}/v2/matrix/{profile}"
        response = requests.post(
            url,
            headers=self._headers(),
            json={
                "locations": [[p.longitude, p.latitude] for p in pois],
                "metrics": ["distance", "duration"],
                "units": "km",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        distances = [
            [None if value is None else round(float(value), 3) for value in row]
            for row in payload.get("distances", [])
        ]
        durations = [
            [None if value is None else max(0, round(float(value) / 60.0)) for value in row]
            for row in payload.get("durations", [])
        ]
        return RouteMatrix(
            names=[p.name for p in pois],
            distances_km=distances,
            durations_min=durations,
            source="openrouteservice",
            grounded=True,
        )
