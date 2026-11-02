from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tripmind.config import get_config,resolve_project_path
from tripmind.bundle import resolve_runtime_data
def main():
    cfg=get_config();assert cfg["providers"].get("preference","auto") in {"auto","heuristic","learned"};assert "preference_model" in cfg;assert "preference" in cfg.get("training",{});data=resolve_runtime_data(cfg["data"]);poi=resolve_project_path(data["poi_dir"]);knowledge=resolve_project_path(data.get("knowledge_index",data["knowledge_file"]));cp=resolve_project_path(cfg["preference_model"].get("checkpoint","checkpoints/tripmind-preference/best"));print("CONFIG OK");print("Data bundle:",data.get("bundle_id","bundled demo"));print("preference provider:",cfg["providers"].get("preference","auto"));print("preference checkpoint exists:",(cp/"preference_config.json").is_file());print("POI dir:",poi);print("knowledge:",knowledge);print("API keys are intentionally not printed.")
if __name__=="__main__":main()
