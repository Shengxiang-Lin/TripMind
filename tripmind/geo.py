"""City-center reach estimates for itinerary candidate selection."""

from math import asin, cos, radians, sin, sqrt
from statistics import median

from .models import POI

CITY_CENTERS = {
    "shanghai": (31.2304, 121.4737), "beijing": (39.9042, 116.4074),
    "nanjing": (32.0603, 118.7969), "guangzhou": (23.1291, 113.2644),
    "chengdu": (30.5728, 104.0668), "hangzhou": (30.2741, 120.1551),
    "wuhan": (30.5928, 114.3055), "shenzhen": (22.5431, 114.0579),
    "suzhou": (31.2989, 120.5853), "chongqing": (29.5630, 106.5516),
}
_CITY_ALIASES = {
    "上海": "shanghai", "北京": "beijing", "南京": "nanjing", "广州": "guangzhou",
    "成都": "chengdu", "杭州": "hangzhou", "武汉": "wuhan", "深圳": "shenzhen",
    "苏州": "suzhou", "重庆": "chongqing",
}
_REACH = {"walking": 8, "cycling": 15, "transit": 25, "driving": 80}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    arc = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(min(1.0, arc)))


def _city_key(city: str) -> str:
    name = city.strip().casefold().removesuffix("市")
    return _CITY_ALIASES.get(name, name)


def city_center(city: str, pois: list[POI]) -> tuple[float, float]:
    key = _city_key(city)
    if key in CITY_CENTERS:
        return CITY_CENTERS[key]
    local = [poi for poi in pois if _city_key(poi.city) == key]
    return median(poi.latitude for poi in local), median(poi.longitude for poi in local)


def reach_km(transport: str) -> float:
    return _REACH.get(transport, _REACH["walking"])


def within_reach(poi: POI, city: str, transport: str, pois: list[POI]) -> bool:
    return haversine_km(poi.latitude, poi.longitude, *city_center(city, pois)) <= reach_km(transport)
