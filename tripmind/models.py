from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UserProfile:
    city: str
    days: int = 2
    budget: float = 1500.0
    age_group: str = "18-25"
    travel_group: str = "couple"
    interests: List[str] = field(default_factory=lambda: ["photography", "food", "culture"])
    pace: str = "relaxed"
    transport_preference: str = "walking"
    special_preferences: List[str] = field(default_factory=list)
    travel_history: List[str] = field(default_factory=list)
    country: str = "China"
    start_date: Optional[str] = None
    currency: str = "CNY"


@dataclass
class POI:
    poi_id: str
    city: str
    name: str
    category: str
    longitude: float
    latitude: float
    ticket_price: float
    recommended_duration: int
    opening_time: str
    closing_time: str
    popularity: float
    tags: List[str]
    suitable_groups: List[str]
    intensity: str = "low"
    description: str = ""
    image_url: Optional[str] = None
    country: str = "China"
    currency: str = "CNY"
    indoor: Optional[bool] = None
    source: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    source_metadata: dict = field(default_factory=dict)


@dataclass
class RankedPOI:
    poi: POI
    score: float
    reasons: List[str]


@dataclass
class RouteLeg:
    origin: str
    destination: str
    distance_km: float
    duration_min: int
    mode: str
    source: str = "local_estimate"
    grounded: bool = False
    geometry: Optional[List[List[float]]] = None


@dataclass
class RouteMatrix:
    names: List[str]
    distances_km: List[List[Optional[float]]]
    durations_min: List[List[Optional[int]]]
    source: str
    grounded: bool


@dataclass
class RealtimeSignal:
    kind: str
    subject: str
    summary: str
    effect: str = "info"
    confidence: str = "medium"
    source: str = "unknown"
    hard_constraint: bool = False
    retrieved_at: Optional[str] = None


@dataclass
class ScheduleItem:
    poi: POI
    start_min: int
    end_min: int
    score: float
    travel_from_previous: Optional[RouteLeg] = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class DayPlan:
    day: int
    items: List[ScheduleItem]
    estimated_cost: float
    travel_distance_km: float
    travel_duration_min: int


@dataclass
class Itinerary:
    profile: UserProfile
    days: List[DayPlan]
    total_estimated_cost: float
    retrieved_knowledge: List[str]
    realtime_signals: List[RealtimeSignal] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    realtime_status: str = "disabled"
    knowledge_sources: List[dict] = field(default_factory=list)
    personalization_status: str = "heuristic"
