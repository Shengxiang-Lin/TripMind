from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable

import requests
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.config import get_config, resolve_project_path

HTTP_USER_AGENT = "TripMind/0.5 (+https://github.com/Shengxiang-Lin/TripMind)"


def _download_hf(repo_id: str, revision: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"[HF] {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target),
    )


def _download_file(url: str, target: Path, force: bool = False) -> None:
    if target.exists() and not force:
        print(f"[skip] {target} already exists")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    print(f"[HTTP] {url}")
    with requests.get(
        url, stream=True, timeout=60, headers={"User-Agent": HTTP_USER_AGENT}
    ) as response:
        response.raise_for_status()
        with temp.open("wb") as f:
            shutil.copyfileobj(response.raw, f)
    temp.replace(target)
    print(f"[done] {target}")


def _download_wikivoyage(
    cfg: dict, languages: Iterable[str], raw_dir: Path, force: bool
) -> None:
    wiki_cfg = cfg["data"]["wikivoyage"]
    for lang in languages:
        filename = wiki_cfg["dump_filename"].format(lang=lang)
        url = wiki_cfg["dump_url"].format(lang=lang, filename=filename)
        _download_file(url, raw_dir / "wikivoyage" / lang / filename, force=force)


def _download_stravl(cfg: dict, raw_dir: Path, force: bool) -> None:
    """Optional legacy auxiliary dataset; not part of the default training route."""
    item = cfg["data"]["public_preferences"]["stravl"]
    target = raw_dir / "stravl"
    _download_file(item["csv_url"], target / "Stravl_Travel_Preference_Data.csv", force)
    _download_file(item["destinations_url"], target / "destination_ids.txt", force)


def _download_genup(cfg: dict, raw_dir: Path, source: str) -> None:
    item = cfg["data"]["public_preferences"]["genup"][source]
    _download_hf(
        item["repo_id"],
        item.get("revision", "main"),
        raw_dir / f"genup-{source}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public datasets used by TripMind.")
    parser.add_argument(
        "--source",
        choices=[
            "all",
            "core",
            "genup",
            "preferences",
            "chinatravel",
            "chinatravel-sandbox",
            "wikivoyage",
            "genup-nyc",
            "genup-tky",
            "genup-ca",
            "stravl",
        ],
        default="all",
        help=(
            "'core' = ChinaTravel + Sandbox + Wikivoyage; "
            "'genup'/'preferences' = GenUP NYC/Tokyo/California. "
            "Stravl is optional and must be requested explicitly."
        ),
    )
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    raw_dir = resolve_project_path(cfg["data"].get("raw_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    core = {"chinatravel", "chinatravel-sandbox", "wikivoyage"}
    genup = {"genup-nyc", "genup-tky", "genup-ca"}
    if args.source == "all":
        sources = core | genup
    elif args.source == "core":
        sources = core
    elif args.source in {"genup", "preferences"}:
        sources = genup
    else:
        sources = {args.source}

    if "chinatravel" in sources:
        item = cfg["data"]["chinatravel"]
        _download_hf(item["repo_id"], item.get("revision", "main"), raw_dir / "chinatravel")
    if "chinatravel-sandbox" in sources:
        item = cfg["data"]["chinatravel_sandbox"]
        _download_hf(item["repo_id"], item.get("revision", "main"), raw_dir / "chinatravel-sandbox")
    if "wikivoyage" in sources:
        languages = args.languages or cfg["data"]["wikivoyage"].get("languages", ["zh", "en"])
        _download_wikivoyage(cfg, languages, raw_dir, args.force)
    if "stravl" in sources:
        _download_stravl(cfg, raw_dir, args.force)
    for source in ("nyc", "tky", "ca"):
        if f"genup-{source}" in sources:
            _download_genup(cfg, raw_dir, source)

    print(f"\nRaw datasets are under: {raw_dir}")
    print("Downloaded raw data is gitignored by default.")


if __name__ == "__main__":
    main()
