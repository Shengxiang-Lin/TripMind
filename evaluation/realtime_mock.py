from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.data import city_pois, load_pois
from tripmind.models import RealtimeSignal, UserProfile
from tripmind.rag import LocalKnowledgeRetriever
from tripmind.recommender import HeuristicPersonalizedRanker, RealtimeAdjustmentEngine


def main():
    profile = UserProfile(
        country="China",
        city="杭州",
        travel_group="couple",
        interests=["photography", "culture"],
    )
    pois = city_pois(load_pois(), profile.city, profile.country)
    ranked = HeuristicPersonalizedRanker().rank(
        profile,
        pois,
        LocalKnowledgeRetriever().retrieve(profile),
    )
    before = {x.poi.name: x.score for x in ranked}

    signals = [
        RealtimeSignal(
            kind="closure",
            subject="西湖",
            summary="Synthetic test closure; not real-world data.",
            effect="avoid",
            confidence="high",
            source="unit-test",
            hard_constraint=True,
        )
    ]
    adjusted = RealtimeAdjustmentEngine().apply(profile, ranked, signals)
    after = {x.poi.name: x.score for x in adjusted}

    assert after["西湖"] < before["西湖"]
    print("REALTIME ADJUSTMENT OK")


if __name__ == "__main__":
    main()
