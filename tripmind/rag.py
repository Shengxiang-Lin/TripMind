from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from .models import UserProfile
from .text import tokens

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LocalKnowledgeRetriever:
    """Small deterministic retriever used before a vector RAG backend is connected."""

    def __init__(self, path: str | Path | None = None):
        path = (
            Path(path)
            if path
            else PROJECT_ROOT / "data" / "knowledge" / "travel_knowledge.json"
        )
        self.docs: List[Dict] = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(tokens(text))

    def retrieve_documents(self, profile: UserProfile, top_k: int = 5) -> List[Dict]:
        query = " ".join(
            [
                profile.country,
                profile.city,
                profile.travel_group,
                profile.pace,
                *profile.interests,
                *profile.special_preferences,
            ]
        )
        q = self._tokens(query)
        scored = []
        for doc in self.docs:
            if doc.get("city", "*") not in {profile.city, "*", ""}:
                continue
            if doc.get("country", "*") not in {profile.country, "*", ""}:
                continue
            haystack = " ".join(
                [
                    doc.get("title", ""),
                    doc.get("text", ""),
                    " ".join(doc.get("tags", [])),
                    doc.get("country", ""),
                    doc.get("city", ""),
                ]
            )
            d = self._tokens(haystack)
            overlap = len(q & d)
            city_bonus = 3 if doc.get("city") in ("*", profile.city) else 0
            country_bonus = (
                1
                if doc.get("country", "*") in ("*", "", profile.country)
                else 0
            )
            scored.append((overlap + city_bonus + country_bonus, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**doc,"score":score} for score,doc in scored[:max(0,top_k)] if score>0]

    def retrieve(self, profile: UserProfile, top_k: int = 5) -> List[str]:
        return [d["text"] for d in self.retrieve_documents(profile,top_k)]
