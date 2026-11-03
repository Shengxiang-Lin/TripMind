"""Offline age-rule checks against bundled POIs; no model or API required."""
from dataclasses import replace
from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.data import city_pois, load_pois
from tripmind.models import UserProfile
from tripmind.rag import LocalKnowledgeRetriever
from tripmind.recommender import HeuristicPersonalizedRanker


def main():
    pois = load_pois(ROOT / "data/poi")
    profile = UserProfile(city=pois[0].city, country=pois[0].country)
    candidates = city_pois(pois, profile.city, profile.country)
    # Keep retrieval fixed so only age can explain ranking differences.
    knowledge = LocalKnowledgeRetriever().retrieve(profile)
    ranker = HeuristicPersonalizedRanker()
    rankings = {
        age: ranker.rank(replace(profile, age_group=age), candidates, knowledge)
        for age in ["18-25", "60+", "26-40"]
    }
    scores = {age: {x.poi.poi_id: x.score for x in rows} for age, rows in rankings.items()}
    assert any(scores["18-25"][pid] != scores["60+"][pid] for pid in scores["18-25"])
    for poi in candidates:
        pid = poi.poi_id
        bonus = 0.04 if {t.lower() for t in poi.tags} & {"night_view", "shopping"} else 0
        penalty = {"high": 0.08, "medium": 0.03}.get(poi.intensity, 0)
        assert isclose(scores["18-25"][pid] - scores["26-40"][pid], bonus, abs_tol=1e-8)
        assert isclose(scores["26-40"][pid] - scores["60+"][pid], penalty, abs_tol=1e-8)
    # The bundled sample lacks high-intensity POIs; verify that branch explicitly.
    high = replace(candidates[0], intensity="high")
    neutral = ranker.rank(replace(profile, age_group="26-40"), [high], knowledge)[0]
    senior = ranker.rank(replace(profile, age_group="60+"), [high], knowledge)[0]
    assert isclose(neutral.score - senior.score, 0.08, abs_tol=1e-8)
    assert any("age:60+ intensity=high" in reason for reason in senior.reasons)
    assert ranker.rank(replace(profile, age_group=None), candidates, knowledge) == rankings["26-40"]
    assert ranker.rank(profile, candidates, knowledge) == rankings["18-25"]
    for age in ["18-25", "60+"]:
        print(f"\nAge {age} Top-5:")
        for item in rankings[age][:5]:
            print(f"  {item.poi.poi_id} {item.poi.name}: {item.score:.4f} | {'; '.join(item.reasons)}")
    print("\nPROFILE SENSITIVITY OK")


if __name__ == "__main__":
    main()
