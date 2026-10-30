"""Validate the active full bundle and generate offline plans in every city."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import json
from dataclasses import replace
from tripmind.config import get_config
from tripmind.bundle import resolve_runtime_data
from tripmind.processing.validate import validate_bundle
from tripmind.map_provider import LocalMapProvider
from tripmind.realtime import NullRealtimeProvider
from tripmind.models import UserProfile
from tripmind.service import TripMindService
from tripmind.planner import _to_min


def main():
    config=get_config(); config["data"]["use_processed"]=True
    paths=resolve_runtime_data(config["data"])
    validation=validate_bundle(Path(paths["knowledge_index"]).parent)
    service=TripMindService(config=config,map_provider=LocalMapProvider(),realtime_provider=NullRealtimeProvider())
    rows=[]
    for city in service.available_cities:
        for group,interests in [("couple",["photography","culture"]),("family",["museum","nature"]),("parents",["history","culture"])]:
            profile=UserProfile(city=city,travel_group=group,interests=interests,days=2,budget=1500)
            result=service.plan(profile)
            ids=[i.poi.poi_id for d in result.days for i in d.items]
            assert ids and len(ids)==len(set(ids))
            assert result.retrieved_knowledge and result.knowledge_sources
            assert all(s["city"] in {city,"*"} for s in result.knowledge_sources)
            assert result.total_estimated_cost<=profile.budget
            for day in result.days:
                for index,item in enumerate(day.items):
                    assert item.start_min>=_to_min(item.poi.opening_time)
                    assert item.end_min<=min(1200,_to_min(item.poi.closing_time))
                    if index: assert item.start_min>=day.items[index-1].end_min+25+item.travel_from_previous.duration_min
            rows.append({"city":city,"group":group,"pois":len(ids),"knowledge":len(result.retrieved_knowledge)})
        english=service.city_registry[city]["aliases"][2]
        alias=service.plan(UserProfile(city=english,country="中国",days=1))
        assert alias.profile.city==city and alias.days[0].items
    report={"bundle_id":service.data_bundle_id,"validation":validation,"offline_scenarios":rows,"alias_cases":len(service.available_cities)}
    output=ROOT/"data/processed/production_validation.json"
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("PRODUCTION DATA CHECKS PASSED",service.data_bundle_id,"cities",len(service.available_cities),"profiles",len(rows),"aliases",len(service.available_cities))
    print("Report:",output)


if __name__=="__main__": main()
