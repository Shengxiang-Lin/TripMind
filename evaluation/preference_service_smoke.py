from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tripmind.config import get_config,resolve_project_path
from tripmind.models import UserProfile
from tripmind.service import TripMindService
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",default="checkpoints/tripmind-preference/best");p.add_argument("--city",default="杭州");p.add_argument("--country",default="China");a=p.parse_args();cfg=copy.deepcopy(get_config());cfg.setdefault("providers",{});cfg["providers"].update({"preference":"learned","route":"local","realtime":"local"});cfg.setdefault("preference_model",{})["checkpoint"]=str(resolve_project_path(a.checkpoint));service=TripMindService(config=cfg);profile=UserProfile(city=a.city,country=a.country,days=2,budget=1500,currency="CNY",travel_group="couple",interests=["culture","nature","photography"],pace="relaxed",travel_history=["故宫","苏州博物馆","西安城墙"],special_preferences=["avoid_crowds"]);it=service.plan(profile);assert it.personalization_status=="learned-genup";selected=[{"day":d.day,"pois":[{"name":x.poi.name,"score":x.score} for x in d.items]} for d in it.days];assert any(x["pois"] for x in selected);print(json.dumps({"personalization_status":it.personalization_status,"selected":selected,"warnings":it.warnings},ensure_ascii=False,indent=2));print("PREFERENCE_SERVICE_SMOKE_OK")
if __name__=="__main__":main()
