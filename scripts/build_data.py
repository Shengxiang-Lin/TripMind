"""Build and atomically activate a versioned travel-data bundle, entirely offline."""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from filelock import FileLock
from tripmind.config import get_config,resolve_project_path
from tripmind.processing.common import CITY_NAMES,canonical_city,sha256,stable_id,write_json,atomic_json
from tripmind.processing.catalog import build_catalog
from tripmind.processing.queries import build_queries
from tripmind.processing.knowledge import build_knowledge
from tripmind.processing.index import build_index
from tripmind.processing.cache import reuse_knowledge
from tripmind.processing.validate import validate_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="data/processed/production")
    parser.add_argument("--cities",nargs="+",help="Default: all 10 Sandbox cities; Chinese/English names accepted")
    parser.add_argument("--languages",nargs="+",choices=["zh","en"],default=["zh","en"])
    parser.add_argument("--chunk-size",type=int,default=1200)
    parser.add_argument("--overlap",type=int,default=150)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--no-publish",action="store_true")
    args=parser.parse_args()
    cities=sorted({canonical_city(c) for c in args.cities}) if args.cities else sorted(CITY_NAMES)
    if not set(cities)<=set(CITY_NAMES): parser.error("Requested city is absent from ChinaTravel-Sandbox")
    if args.chunk_size<200 or not 0<=args.overlap<args.chunk_size: parser.error("Invalid chunk size or overlap")
    raw=resolve_project_path(get_config()["data"].get("raw_dir","data/raw"))
    output=resolve_project_path(args.output); (output/"runs").mkdir(parents=True,exist_ok=True)
    files=sorted(list((raw/"chinatravel-sandbox/data").rglob("*.parquet"))+list((raw/"chinatravel").rglob("*.csv")))
    for language in sorted(set(args.languages)):
        files.append(raw/f"wikivoyage/{language}/{language}wikivoyage-latest-pages-articles.xml.bz2")
    if not files or any(not p.is_file() for p in files): raise FileNotFoundError("Required raw input is missing; download datasets first")
    print("[1/6] Hashing raw inputs",flush=True)
    inputs={str(p.relative_to(raw)):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in files}
    code_files=sorted((ROOT/"tripmind/processing").glob("*.py"))+[ROOT/"tripmind/text.py",ROOT/"tripmind/models.py",ROOT/"tripmind/planner.py",Path(__file__).resolve()]
    code={str(p.relative_to(ROOT)):sha256(p) for p in code_files}
    settings={"cities":cities,"languages":sorted(set(args.languages)),"chunk_size":args.chunk_size,"overlap":args.overlap,"seed":args.seed}
    from importlib.metadata import version
    runtime={"python":platform.python_version(), **{name:version(name) for name in ["scikit-learn","scipy","pandas","pyarrow","mwparserfromhell","opencc-python-reimplemented"]}}
    build_id=stable_id(inputs,code,settings,runtime)[:20]
    run=output/"runs"/build_id
    with FileLock(str(output/".build.lock"),timeout=1):
        if not run.exists():
            stage=output/f".{build_id}.building-{os.getpid()}"; stage.mkdir()
            print("[2/6] Normalizing bilingual catalogue",flush=True)
            catalog=build_catalog(raw,stage,cities)
            print("[3/6] Deduplicating and splitting requests",flush=True)
            queries=build_queries(raw,stage,args.seed,cities if args.cities else None)
            reused=reuse_knowledge(output,stage,inputs,code,settings,runtime)
            if reused:
                knowledge,index=reused
                print("[4/6] Reused verified Wiki corpus from",knowledge["reused_from"],flush=True)
                print("[5/6] Reused matching local vector index",flush=True)
            else:
                print("[4/6] Cleaning bilingual Wikivoyage",flush=True)
                knowledge=build_knowledge(raw,stage,settings["languages"],args.chunk_size,args.overlap)
                print("[5/6] Building local sparse vector index",flush=True)
                index=build_index(stage/"knowledge")
            validation=validate_bundle(stage)
            quality={"catalog":catalog,"queries":queries,"knowledge":knowledge,"index":index,"validation":validation}
            write_json(stage/"quality_report.json",quality)
            artifacts={str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
            manifest={"schema_version":1,"build_id":build_id,"settings":settings,"inputs":inputs,"code":code,
                "runtime":runtime,
                "artifacts":artifacts,"validation_passed":True}
            write_json(stage/"manifest.json",manifest)
            stage.rename(run)
        else:
            manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
            for relative,digest in manifest["artifacts"].items():
                assert sha256(run/relative)==digest, f"Existing artifact changed: {relative}"
            print("Reusing matching verified build",build_id,flush=True)
        if not args.no_publish:
            atomic_json(output/"current.json",{"build_id":build_id,"path":f"runs/{build_id}","manifest_sha256":sha256(run/"manifest.json")})
        print("[6/6] VALIDATED",run,flush=True)
        print("Published:",not args.no_publish,flush=True)
        print(json.dumps(json.loads((run/"quality_report.json").read_text(encoding="utf-8"))["validation"],ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
