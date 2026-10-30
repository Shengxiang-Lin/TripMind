from __future__ import annotations

import ssl
from typing import Any, Mapping, Sequence

import requests
from requests.adapters import HTTPAdapter

from ..config import get_config
from ..map_provider import MapProvider
from ..models import POI, RouteLeg, RouteMatrix


class TLSAdapter(HTTPAdapter):
    """Optional TLS cap scoped to this provider, with certificate verification."""

    def __init__(self, maximum_version: str, **kwargs):
        versions = {"TLSv1_2": ssl.TLSVersion.TLSv1_2, "TLSv1_3": ssl.TLSVersion.TLSv1_3}
        if maximum_version not in versions:
            raise ValueError("ors.tls_maximum_version must be TLSv1_2 or TLSv1_3")
        self.context = ssl.create_default_context()
        self.context.maximum_version = versions[maximum_version]
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **kwargs):
        kwargs["ssl_context"] = self.context
        return super().proxy_manager_for(proxy, **kwargs)


class OpenRouteServiceProvider(MapProvider):
    """Road-network routing backed by openrouteservice / HeiGIT.

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
        timeout: float | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        config = dict(config or get_config())
        ors = config.get("ors", {})

        self.api_key = api_key or str(ors.get("api_key", "")).strip()
        if not self.api_key:
            raise RuntimeError(
                "No ORS API key configured. Set ors.api_key in "
                "configs/config.local.yaml or ORS_API_KEY."
            )

        self.base_url = (
            base_url
            or ors.get("base_url")
            or "https://api.heigit.org/openrouteservice"
        ).rstrip("/")
        self.timeout = float(
            timeout if timeout is not None else ors.get("timeout_seconds", 15)
        )

        self.session = requests.Session()
        maximum_version = ors.get("tls_maximum_version")
        if maximum_version:
            self.session.mount("https://", TLSAdapter(str(maximum_version)))
        self.api_calls = 0
        self._matrix_cache: dict[tuple, RouteMatrix] = {}

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
        self.api_calls += 1
        response = self.session.post(
            url,
            headers={**self._headers(), "Accept": "application/geo+json"},
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

        cache_key = (mode, tuple((p.name, p.longitude, p.latitude) for p in pois))
        if cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]
        profile = self.PROFILE_BY_MODE[mode]
        url = f"{self.base_url}/v2/matrix/{profile}"
        self.api_calls += 1
        response = self.session.post(
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
            [
                None if value is None else max(0, round(float(value) / 60.0))
                for value in row
            ]
            for row in payload.get("durations", [])
        ]
        n = len(pois)
        if len(distances) != n or len(durations) != n or any(
            len(row) != n for row in distances + durations
        ):
            raise ValueError("ORS returned a malformed route matrix")
        matrix = RouteMatrix(
            names=[p.name for p in pois],
            distances_km=distances,
            durations_min=durations,
            source="openrouteservice",
            grounded=True,
        )
        if len(self._matrix_cache) >= 32:
            self._matrix_cache.pop(next(iter(self._matrix_cache)))
        self._matrix_cache[cache_key] = matrix
        return matrix
