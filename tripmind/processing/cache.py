"""Reuse only checksum-verified Wiki/index artifacts with matching dependencies."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from .common import sha256

CRITICAL_CODE = ["tripmind/processing/common.py","tripmind/processing/knowledge.py",
                 "tripmind/processing/index.py","tripmind/text.py"]


def reuse_knowledge(output: Path, stage: Path, inputs: dict, code: dict, settings: dict, runtime: dict):
    wiki_inputs={k:v for k,v in inputs.items() if k.startswith("wikivoyage/")}
    for manifest_file in sorted((output/"runs").glob("*/manifest.json")):
        run=manifest_file.parent.resolve()
        try:
            manifest=json.loads(manifest_file.read_text())
            if manifest.get("schema_version")!=1 or not manifest.get("validation_passed"): continue
            previous={k:v for k,v in manifest["inputs"].items() if k.startswith("wikivoyage/")}
            if previous!=wiki_inputs or manifest.get("runtime")!=runtime: continue
            if any(manifest["code"].get(k)!=code.get(k) for k in CRITICAL_CODE): continue
            if any(manifest["settings"].get(k)!=settings.get(k) for k in ["languages","chunk_size","overlap"]): continue
            expected={k:v for k,v in manifest["artifacts"].items() if k.startswith("knowledge/")}
            actual={str(p.relative_to(run)) for p in (run/"knowledge").rglob("*") if p.is_file()}
            if not expected or set(expected)!=actual: continue
            expected["quality_report.json"]=manifest["artifacts"]["quality_report.json"]
            if any(not (run/k).resolve().is_relative_to(run) or (run/k).is_symlink() or sha256(run/k)!=digest for k,digest in expected.items()): continue
            quality=json.loads((run/"quality_report.json").read_text())
        except (OSError,ValueError,KeyError,TypeError):
            continue
        shutil.copytree(run/"knowledge",stage/"knowledge")
        knowledge={**quality["knowledge"],"reused_from":manifest["build_id"]}
        index={**quality["index"],"reused_from":manifest["build_id"]}
        return knowledge,index
    return None
