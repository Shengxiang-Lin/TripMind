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

HTTP_USER_AGENT = "TripMind/0.3 (+https://github.com/Shengxiang-Lin/TripMind)"


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
    cfg: dict,
    languages: Iterable[str],
    raw_dir: Path,
    force: bool,
) -> None:
    wiki_cfg = cfg["data"]["wikivoyage"]
    filename_template = wiki_cfg["dump_filename"]
    url_template = wiki_cfg["dump_url"]

    for lang in languages:
        filename = filename_template.format(lang=lang)
        url = url_template.format(lang=lang, filename=filename)
        target = raw_dir / "wikivoyage" / lang / filename
        _download_file(url, target, force=force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public datasets used by TripMind."
    )
    parser.add_argument(
        "--source",
        choices=["all", "chinatravel", "chinatravel-sandbox", "wikivoyage"],
        default="all",
        help="Dataset source to download.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Wikivoyage language codes, e.g. zh en ja fr.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload existing Wikivoyage dump files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    raw_dir = resolve_project_path(cfg["data"].get("raw_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = (
        {"chinatravel", "chinatravel-sandbox", "wikivoyage"}
        if args.source == "all"
        else {args.source}
    )

    if "chinatravel" in sources:
        item = cfg["data"]["chinatravel"]
        _download_hf(
            item["repo_id"],
            item.get("revision", "main"),
            raw_dir / "chinatravel",
        )

    if "chinatravel-sandbox" in sources:
        item = cfg["data"]["chinatravel_sandbox"]
        _download_hf(
            item["repo_id"],
            item.get("revision", "main"),
            raw_dir / "chinatravel-sandbox",
        )

    if "wikivoyage" in sources:
        languages = (
            args.languages
            if args.languages
            else cfg["data"]["wikivoyage"].get("languages", ["zh", "en"])
        )
        _download_wikivoyage(cfg, languages, raw_dir, force=args.force)

    print(f"\nRaw datasets are under: {raw_dir}")
    print("Downloaded raw data is gitignored by default.")


if __name__ == "__main__":
    main()
