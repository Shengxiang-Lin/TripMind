from __future__ import annotations

from typing import Iterable, List

from .models import POI, RankedPOI, UserProfile


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


class HeuristicPersonalizedRanker:
    """Interpretable baseline that will later be replaced by the SFT ranker."""

    def rank(
        self,
        profile: UserProfile,
        candidates: Iterable[POI],
        retrieved_knowledge: List[str] | None = None,
    ) -> List[RankedPOI]:
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

            pace_penalty = PACE_PENALTY.get(profile.pace, PACE_PENALTY["normal"]).get(
                poi.intensity, 0.0
            )

            knowledge_bonus = 0.08 if poi.name.lower() in knowledge_text else 0.0
            if knowledge_bonus:
                reasons.append("supported_by_retrieved_guide")

            popularity_score = max(0.0, min(1.0, poi.popularity))
            score = (
                0.38 * interest_score
                + 0.24 * group_score
                + 0.14 * budget_score
                + 0.16 * popularity_score
                + knowledge_bonus
                - pace_penalty
            )

            ranked.append(RankedPOI(poi=poi, score=round(score, 4), reasons=reasons))

        return sorted(ranked, key=lambda x: x.score, reverse=True)
