"""Resolve an immutable, validated data build selected by an atomic pointer."""
from __future__ import annotations
import json
from pathlib import Path
from .config import resolve_project_path
from .processing.common import sha256


def resolve_runtime_data(data: dict) -> dict:
    resolved=dict(data)
    if not data.get("use_processed",False):
        return resolved
    root=resolve_project_path(data.get("bundle_root","data/processed/production")).resolve()
    pointer_path=root/"current.json"
    if not pointer_path.is_file():
        raise FileNotFoundError("Processed data is enabled but no bundle is active. Run python scripts/build_data.py, or set data.use_processed=false for the demo.")
    pointer=json.loads(pointer_path.read_text(encoding="utf-8"))
    run=(root/pointer["path"]).resolve()
    if not run.is_relative_to(root/"runs") or run.name!=pointer["build_id"]:
        raise ValueError("Invalid data bundle pointer")
    if sha256(run/"manifest.json")!=pointer["manifest_sha256"]:
        raise ValueError("Data bundle manifest checksum mismatch")
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version")!=1 or not manifest.get("validation_passed") or manifest.get("build_id")!=pointer["build_id"]:
        raise ValueError("Data build has not passed validation")
    for relative,digest in manifest["artifacts"].items():
        file=(run/relative).resolve()
        if not file.is_relative_to(run) or sha256(file)!=digest:
            raise ValueError(f"Data artifact checksum mismatch: {relative}")
    resolved.update(poi_dir=str(run/"poi"),knowledge_index=str(run/"knowledge"),
                    city_aliases_file=str(run/"cities.json"),bundle_id=pointer["build_id"])
    return resolved
