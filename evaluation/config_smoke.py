from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.config import get_config, resolve_project_path
from tripmind.bundle import resolve_runtime_data


def main() -> None:
    cfg = get_config()

    assert "providers" in cfg
    assert "ors" in cfg
    assert "dashscope" in cfg
    assert cfg["ors"]["base_url"].startswith("https://api.heigit.org/")

    data = resolve_runtime_data(cfg["data"])
    poi_dir = resolve_project_path(data["poi_dir"])
    knowledge = resolve_project_path(data.get("knowledge_index", data["knowledge_file"]))
    print("Data bundle:",data.get("bundle_id","bundled demo"))

    print("CONFIG OK")
    print("route provider:", cfg["providers"]["route"])
    print("realtime provider:", cfg["providers"]["realtime"])
    print("ORS base:", cfg["ors"]["base_url"])
    print("POI dir:", poi_dir)
    print("knowledge file:", knowledge)
    print("API keys are intentionally not printed.")


if __name__ == "__main__":
    main()
