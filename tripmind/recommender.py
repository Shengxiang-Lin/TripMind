from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence

from .models import POI, RankedPOI, RealtimeSignal, UserProfile

GROUP_HINTS = {
    "couple": {"romantic", "photography", "night_view", "walk", "food"},
    "family": {"family", "education", "animal", "museum", "park"},
    "parents": {"culture", "history", "scenic", "low_intensity"},
    "friends": {"food", "night_view", "shopping", "entertainment"},
    "solo": {"culture", "museum", "walk", "photography"},
}
PACE_PENALTY = {
    "relaxed": {"high": 0.18, "medium": 0.05, "low": 0.0},
    "normal": {"high": 0.08, "medium": 0.02, "low": 0.0},
    "intensive": {"high": 0.0, "medium": 0.0, "low": 0.0},
}
INDOOR_CATEGORIES = {"museum", "gallery", "shopping", "mall", "aquarium", "theatre", "theater"}
OUTDOOR_HINTS = {"nature", "park", "scenic", "walk", "hiking", "outdoor"}
AGE_TAG_BONUSES = {
    "<18": ({"family", "education", "animal", "park", "museum"}, 0.08),
    "18-25": ({"night_view", "shopping", "entertainment", "food"}, 0.08),
    "41-60": ({"culture", "history", "nature"}, 0.04),
    "60+": ({"park", "nature", "museum", "gallery"}, 0.06),
}


def age_adjustment(profile: UserProfile, poi: POI) -> float:
    age = profile.age_group
    if age not in AGE_TAG_BONUSES:
        return 0.0
    tags = {value.casefold() for value in (poi.category, *poi.tags)}
    favored, bonus = AGE_TAG_BONUSES[age]
    adjustment = bonus if tags & favored else 0.0
    if age in {"<18", "60+"} and tags & {"nightlife", "bar"}:
        adjustment -= 0.10
    if poi.recommended_duration > 180:
        adjustment -= {"<18": 0.04, "60+": 0.05}.get(age, 0.0)
    if age == "41-60" and poi.intensity == "high":
        adjustment -= 0.08
    if age == "60+":
        adjustment -= {"high": 0.15, "medium": 0.06}.get(poi.intensity, 0.0)
    return adjustment


def _normalize_model_scores(raw_scores: Sequence[float]) -> list[float]:
    values = [float(x) for x in raw_scores]
    if not values:
        return []
    if len(values) == 1:
        return [0.5]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.5 for _ in values]
    return [1.0 / (1.0 + math.exp(-(x - mean) / std)) for x in values]


class HeuristicPersonalizedRanker:
    name = "heuristic"

    def rank(self, profile: UserProfile, candidates: Iterable[POI], retrieved_knowledge: List[str] | None = None) -> List[RankedPOI]:
        ranked: List[RankedPOI] = []
        interests = {x.lower() for x in profile.interests}
        group_hints = GROUP_HINTS.get(profile.travel_group, set())
        knowledge_text = " ".join(retrieved_knowledge or []).lower()
        for poi in candidates:
            tags = {x.lower() for x in poi.tags}
            reasons: List[str] = []
            interest_hits = interests & tags
            interest_score = min(1.0, len(interest_hits) / max(1, min(3, len(interests))))
            if interest_hits:
                reasons.append("interest:" + ",".join(sorted(interest_hits)))
            group_hits = group_hints & tags
            group_score = min(1.0, len(group_hits) / 2)
            if profile.travel_group in poi.suitable_groups:
                group_score = max(group_score, 0.75)
                reasons.append(f"suitable_for:{profile.travel_group}")
            budget_ratio = poi.ticket_price / max(profile.budget, 1)
            budget_score = max(0.0, 1.0 - budget_ratio * max(profile.days, 1) * 3)
            pace_penalty = PACE_PENALTY.get(profile.pace, PACE_PENALTY["normal"]).get(poi.intensity, 0.0)
            knowledge_bonus = 0.08 if poi.name.lower() in knowledge_text else 0.0
            if knowledge_bonus:
                reasons.append("supported_by_retrieved_guide")
            popularity_score = max(0.0, min(1.0, poi.popularity))
            score = 0.38 * interest_score + 0.24 * group_score + 0.14 * budget_score + 0.16 * popularity_score + knowledge_bonus - pace_penalty
            age_delta = age_adjustment(profile, poi)
            if age_delta:
                score += age_delta
                reasons.append(f"age:{profile.age_group} {age_delta:+.2f}")
            ranked.append(RankedPOI(poi=poi, score=round(score, 4), reasons=reasons))
        return sorted(ranked, key=lambda x: x.score, reverse=True)


class LearnedPersonalizedRanker:
    name = "learned-genup"

    def __init__(self, scorer, *, heuristic_blend: float = 0.15):
        if not 0.0 <= heuristic_blend <= 1.0:
            raise ValueError("heuristic_blend must be in [0,1]")
        self.scorer = scorer
        self.heuristic_blend = float(heuristic_blend)
        self.heuristic = HeuristicPersonalizedRanker()

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, device: str = "auto", batch_size: int = 8, max_length: int | None = None, history_char_limit: int | None = None, heuristic_blend: float = 0.15) -> "LearnedPersonalizedRanker":
        from .preference_model import PreferenceScorer
        scorer = PreferenceScorer(checkpoint, device=device, batch_size=batch_size, max_length=max_length, history_char_limit=history_char_limit)
        return cls(scorer, heuristic_blend=heuristic_blend)

    def rank(self, profile: UserProfile, candidates: Iterable[POI], retrieved_knowledge: List[str] | None = None) -> List[RankedPOI]:
        candidate_list = list(candidates)
        if not candidate_list:
            return []
        raw_scores = self.scorer.score_pois(profile, candidate_list)
        if len(raw_scores) != len(candidate_list):
            raise RuntimeError("Preference scorer returned the wrong number of scores")
        learned_scores = _normalize_model_scores(raw_scores)
        heuristic_by_id = {}
        if self.heuristic_blend > 0:
            heuristic_by_id = {item.poi.poi_id: item for item in self.heuristic.rank(profile, candidate_list, retrieved_knowledge)}
        ranked: list[RankedPOI] = []
        for poi, raw, learned in zip(candidate_list, raw_scores, learned_scores):
            heuristic = heuristic_by_id.get(poi.poi_id)
            heuristic_score = heuristic.score if heuristic else 0.0
            score = (1.0 - self.heuristic_blend) * learned + self.heuristic_blend * heuristic_score
            reasons = ["learned_preference", f"learned_preference_raw:{raw:.4f}", f"learned_preference_normalized:{learned:.4f}"]
            age_delta = 0.15 * age_adjustment(profile, poi)
            if age_delta:
                score += age_delta
                reasons.append(f"age_adjustment:{age_delta:+.4f}")
            if heuristic and self.heuristic_blend > 0:
                reasons.append(f"heuristic_blend:{self.heuristic_blend:.2f}")
                reasons.extend(heuristic.reasons[:2])
            ranked.append(RankedPOI(poi=poi, score=round(max(0.0, min(1.0, score)), 4), reasons=reasons))
        return sorted(ranked, key=lambda x: x.score, reverse=True)


class RealtimeAdjustmentEngine:
    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(text.casefold().split())

    @classmethod
    def _matches_poi(cls, subject: str, poi_name: str) -> bool:
        a, b = cls._normalize(subject), cls._normalize(poi_name)
        return bool(a and b and (a == b or a in b or b in a))

    @staticmethod
    def _is_indoor(poi: POI) -> bool:
        if poi.indoor is not None:
            return poi.indoor
        return poi.category.casefold() in INDOOR_CATEGORIES

    @staticmethod
    def _is_outdoor(poi: POI) -> bool:
        tags = {x.casefold() for x in poi.tags}
        return bool(tags & OUTDOOR_HINTS)

    def apply(self, profile: UserProfile, ranked: List[RankedPOI], signals: List[RealtimeSignal]) -> List[RankedPOI]:
        adjusted: List[RankedPOI] = []
        city_key = self._normalize(profile.city)
        for item in ranked:
            delta = 0.0
            reasons = list(item.reasons)
            for signal in signals:
                subject_key = self._normalize(signal.subject)
                poi_match = self._matches_poi(signal.subject, item.poi.name)
                city_scope = subject_key in {city_key, self._normalize(profile.country)}
                if poi_match:
                    if signal.effect == "avoid":
                        delta -= 1.0 if signal.hard_constraint else 0.30
                        reasons.append(f"realtime_avoid:{signal.kind}")
                    elif signal.effect == "prefer":
                        delta += 0.08
                        reasons.append(f"realtime_prefer:{signal.kind}")
                if city_scope and signal.effect == "prefer_indoor":
                    if self._is_indoor(item.poi):
                        delta += 0.08
                        reasons.append("weather_indoor_bonus")
                    elif self._is_outdoor(item.poi):
                        delta -= 0.05
                        reasons.append("weather_outdoor_penalty")
            adjusted.append(RankedPOI(poi=item.poi, score=round(max(0.0, min(1.0, item.score + delta)), 4), reasons=reasons))
        return sorted(adjusted, key=lambda x: x.score, reverse=True)
