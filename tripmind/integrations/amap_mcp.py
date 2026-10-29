from __future__ import annotations

from . import dashscope_llm  # keeps integrations package importable for future wiring
from ..map_provider import MapProvider
from ..models import POI, RouteLeg


class AmapMCPProvider(MapProvider):
    """Adapter boundary for Bailian One-Key Amap Maps MCP.

    The MCP connection configuration is copied from the Bailian MCP Marketplace
    and can vary by workspace/region. This class intentionally raises until that
    configuration is supplied, instead of silently fabricating route facts.

    Next implementation step:
    1. read AMAP_MCP_URL / generated MCP client config;
    2. call the route-planning tool exposed by Amap Maps MCP;
    3. normalize its distance/duration into RouteLeg.
    """

    def route(self, origin: POI, destination: POI, mode: str = "transit") -> RouteLeg:
        raise NotImplementedError(
            "Amap MCP is not configured yet. Use LocalMapProvider for the scaffold, "
            "then wire the One-Key Amap Maps MCP configuration from Bailian."
        )
