from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .config import get_config
from .models import POI, RealtimeSignal, UserProfile


class RealtimeProvider(ABC):
    @abstractmethod
    def fetch(self, profile: UserProfile, pois: Sequence[POI]) -> list[RealtimeSignal]:
        raise NotImplementedError


class NullRealtimeProvider(RealtimeProvider):
    """No-network provider used by local tests and offline demos."""

    def fetch(self, profile: UserProfile, pois: Sequence[POI]) -> list[RealtimeSignal]:
        return []


class CachingRealtimeProvider(RealtimeProvider):
    """Tiny in-memory TTL cache to avoid repeated web searches while editing a plan."""

    def __init__(self, inner: RealtimeProvider, ttl_seconds: int = 3600):
        self.inner = inner
        self.ttl_seconds = ttl_seconds
        self._cache: dict[tuple, tuple[float, list[RealtimeSignal]]] = {}
        self.last_errors: list[str] = []

    def _key(self, profile: UserProfile, pois: Sequence[POI]) -> tuple:
        return (
            profile.country.casefold(),
            profile.city.casefold(),
            profile.start_date or "",
            profile.days,
            tuple(sorted(p.poi_id for p in pois)),
        )

    def fetch(self, profile: UserProfile, pois: Sequence[POI]) -> list[RealtimeSignal]:
        key = self._key(profile, pois)
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self.ttl_seconds:
            self.last_errors = []
            return list(hit[1])
        signals = self.inner.fetch(profile, pois)
        self.last_errors = list(getattr(self.inner, "last_errors", []))
        if not self.last_errors:
            self._cache[key] = (now, list(signals))
        return signals


class DashScopeRealtimeProvider(RealtimeProvider):
    """Realtime travel context via Alibaba Cloud Model Studio web search."""

    ALLOWED_KINDS = {
        "weather",
        "closure",
        "opening_hours",
        "promotion",
        "event",
        "alert",
    }
    ALLOWED_EFFECTS = {"avoid", "prefer", "prefer_indoor", "info"}
    ALLOWED_CONFIDENCE = {"high", "medium", "low"}

    def __init__(
        self,
        llm=None,
        poi_limit: int | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        self.config = dict(config or get_config())
        dashscope = self.config.get("dashscope", {})
        realtime = self.config.get("realtime", {})

        if llm is None:
            from .integrations.dashscope_llm import DashScopeLLM

            llm = DashScopeLLM(
                model=dashscope.get("realtime_model")
                or dashscope.get("model", "qwen-plus"),
                config=self.config,
            )
        self.llm = llm
        self.poi_limit = int(
            poi_limit if poi_limit is not None else realtime.get("poi_limit", 6)
        )
        self.strategy = str(dashscope.get("search_strategy", "turbo"))
        self.last_errors: list[str] = []
        self.usage_history: list[dict] = []

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Realtime search did not return a JSON object.")
        return json.loads(text[start : end + 1])

    def _query(self, prompt: str) -> list[RealtimeSignal]:
        system = """You extract current travel information from web search.
Return JSON only with this schema:
{"signals":[
  {
    "kind":"weather|closure|opening_hours|promotion|event|alert",
    "subject":"city or exact POI name",
    "summary":"short factual travel-relevant statement",
    "effect":"avoid|prefer|prefer_indoor|info",
    "confidence":"high|medium|low",
    "hard_constraint":false
  }
]}
Rules:
- Include only time-sensitive facts relevant to the requested travel date/location.
- Omit uncertain, stale, or unrelated information instead of guessing.
- Weather that makes outdoor visits less suitable should use prefer_indoor.
- Promotions are soft context: never set hard_constraint=true.
- hard_constraint=true is only for an explicit current closure/unavailability,
  and only when confidence=high.
- Do not turn an old promotion or old closure into a current fact.
"""
        text = self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            enable_search=True,
            force_search=True,
            search_strategy=self.strategy,
        )
        self.usage_history.append(dict(getattr(self.llm, "last_usage", {})))
        payload = self._extract_json(text)
        if not isinstance(payload.get("signals", []), list):
            raise ValueError("Realtime signals must be a JSON list")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        signals: list[RealtimeSignal] = []
        for item in payload.get("signals", []):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "alert"))
            effect = str(item.get("effect", "info"))
            confidence = str(item.get("confidence", "medium"))
            if kind not in self.ALLOWED_KINDS:
                continue
            if effect not in self.ALLOWED_EFFECTS:
                effect = "info"
            if confidence not in self.ALLOWED_CONFIDENCE:
                confidence = "medium"
            hard = item.get("hard_constraint") is True
            if kind not in {"closure", "alert"} or confidence != "high" or effect != "avoid":
                hard = False
            signals.append(
                RealtimeSignal(
                    kind=kind,
                    subject=str(item.get("subject", "")).strip(),
                    summary=str(item.get("summary", "")).strip(),
                    effect=effect,
                    confidence=confidence,
                    source="Alibaba Cloud Model Studio web search",
                    hard_constraint=hard,
                    retrieved_at=now,
                )
            )
        return [s for s in signals if s.subject and s.summary]

    def fetch(self, profile: UserProfile, pois: Sequence[POI]) -> list[RealtimeSignal]:
        travel_date = profile.start_date or "the current/near-term travel date"
        city_prompt = f"""Travel location: {profile.city}, {profile.country}
Travel start date: {travel_date}
Trip duration: {profile.days} days
Find only current travel-relevant city context:
1. weather/forecast that may change indoor-vs-outdoor planning;
2. major local events that could be useful to a visitor;
3. major alerts likely to affect sightseeing.
Keep the result concise."""

        selected = list(pois[: self.poi_limit])
        poi_names = ", ".join(p.name for p in selected)
        poi_prompt = f"""Travel location: {profile.city}, {profile.country}
Travel start date: {travel_date}
Trip duration: {profile.days} days
Candidate POIs: {poi_names}
For these POIs only, check current information that could change a plan:
temporary closure, changed opening hours, official/credible ticket promotion,
or a time-sensitive event. Omit ordinary static descriptions and omit anything
whose date is unclear."""

        self.last_errors = []
        signals: list[RealtimeSignal] = []
        for label, prompt in [("city", city_prompt), ("POI", poi_prompt)]:
            try:
                signals.extend(self._query(prompt))
            except Exception as exc:
                self.last_errors.append(f"Realtime {label} lookup failed ({type(exc).__name__}).")
        return signals
