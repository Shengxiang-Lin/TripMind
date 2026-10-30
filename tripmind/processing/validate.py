from __future__ import annotations
import json
import math
import sqlite3
from pathlib import Path
from scipy import sparse
from tripmind.data import load_pois
from tripmind.planner import _to_min


def validate_bundle(folder: Path) -> dict:
    pois=load_pois(folder/"poi")
    assert pois, "Empty POI catalogue"
    assert len({p.poi_id for p in pois})==len(pois), "Duplicate POI IDs"
    cities={p.city for p in pois}
    for p in pois:
        assert math.isfinite(p.ticket_price) and p.ticket_price>=0
        assert -90<=p.latitude<=90 and -180<=p.longitude<=180
        assert 0<=_to_min(p.opening_time)<_to_min(p.closing_time)<=1440
        assert p.recommended_duration>0 and p.source and p.source_metadata.get("license")
    groups={}; uids={}; count=0
    for name in ["train","validation","test"]:
        for line in (folder/f"queries/{name}.jsonl").open(encoding="utf-8"):
            r=json.loads(line); count+=1
            assert r["split"]==name
            assert groups.setdefault(r["split_group"],name)==name, "Family leakage"
            assert uids.setdefault(r["uid"],name)==name, "UID leakage"
            assert not r["official_holdout"] or name=="test", "Official test leakage"
            assert r["label_status"]=="unlabeled_request"
    all_count=sum(1 for _ in (folder/"queries/all.jsonl").open(encoding="utf-8"))
    assert count==all_count>0
    db=sqlite3.connect(folder/"knowledge/documents.sqlite")
    try:
        assert db.execute("PRAGMA quick_check").fetchone()[0]=="ok"
        rows=db.execute("SELECT COUNT(*),MIN(row_idx),MAX(row_idx) FROM documents").fetchone()
        assert rows[0]>0 and rows[1]==0 and rows[2]==rows[0]-1
        coverage={c:db.execute("SELECT COUNT(*) FROM documents WHERE city=?",(c,)).fetchone()[0] for c in cities}
        assert all(coverage.values()), f"Missing destination knowledge: {coverage}"
        langs=dict(db.execute("SELECT language,COUNT(*) FROM documents GROUP BY language"))
    finally:
        db.close()
    matrix=sparse.load_npz(folder/"knowledge/tfidf.npz")
    assert matrix.shape[0]==rows[0] and matrix.shape[1]>0
    assert matrix.data.size and bool((matrix.data>=0).all())
    return {"passed":True,"pois":len(pois),"cities":len(cities),"query_records":count,
        "split_groups":len(groups),"knowledge_chunks":rows[0],"knowledge_languages":langs,
        "destination_knowledge_coverage":dict(sorted(coverage.items())),
        "vector_shape":list(matrix.shape),"checks":["poi_schema","unique_ids","opening_hours", "source_lineage",
        "split_group_disjointness","official_holdout_isolation","sqlite_integrity","city_coverage","vector_alignment"]}
