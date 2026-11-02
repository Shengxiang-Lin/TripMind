from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tripmind.models import POI,UserProfile
from tripmind.recommender import LearnedPersonalizedRanker
class StubScorer:
    def score_pois(self,profile,candidates):
        m={"culture":3.,"nature":2.,"shopping":-1.};return [m.get(p.category,0.) for p in candidates]
def make(i,c):return POI(poi_id=i,city="杭州",name=i,category=c,longitude=120.,latitude=30.,ticket_price=0.,recommended_duration=90,opening_time="09:00",closing_time="18:00",popularity=.5,tags=[c],suitable_groups=["couple"])
def main():
    p=UserProfile(city="杭州",interests=["culture","nature"],travel_history=["故宫","苏州博物馆"]);cand=[make("museum","culture"),make("park","nature"),make("mall","shopping")];r=LearnedPersonalizedRanker(StubScorer(),heuristic_blend=0.).rank(p,cand);assert [x.poi.poi_id for x in r]==["museum","park","mall"];assert all(0<=x.score<=1 for x in r);print("PREFERENCE_RANKER_SMOKE_OK")
if __name__=="__main__":main()
