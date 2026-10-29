from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from .models import UserProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LocalKnowledgeRetriever:
    """Small deterministic retriever used before Bailian RAG is connected.

    It deliberately mirrors the future cloud interface: retrieve(profile, top_k).
    """

    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path else PROJECT_ROOT / "data" / "knowledge" / "travel_knowledge.json"
        self.docs: List[Dict] = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", text.lower()))

    def retrieve(self, profile: UserProfile, top_k: int = 5) -> List[str]:
        query = " ".join([
            profile.city,
            profile.travel_group,
            profile.pace,
            *profile.interests,
            *profile.special_preferences,
        ])
        q = self._tokens(query)
        scored = []
        for doc in self.docs:
            haystack = " ".join([
                doc.get("title", ""),
                doc.get("text", ""),
                " ".join(doc.get("tags", [])),
                doc.get("city", ""),
            ])
            d = self._tokens(haystack)
            overlap = len(q & d)
            city_bonus = 3 if doc.get("city") in ("*", profile.city) else 0
            scored.append((overlap + city_bonus, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1]["text"] for x in scored[:top_k] if x[0] > 0]
