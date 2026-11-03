"""Small offline fixtures for normalization, leakage, Wiki parsing and retrieval."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import json
import sqlite3
import tempfile
import unittest
import xml.etree.ElementTree as ET
from tripmind.models import UserProfile
from tripmind.processing.common import canonical_city,stable_id,write_json,sha256
from tripmind.processing.catalog import normalize_attraction,normalized_hours,build_catalog
from tripmind.processing.queries import normalize_query,assign_splits,parse_programs,query_family,build_queries
from tripmind.processing.knowledge import clean_wikitext,text_chunks,page_documents
from tripmind.processing.index import build_index
from tripmind.vector_rag import VectorKnowledgeRetriever
from tripmind.bundle import resolve_runtime_data
from tripmind.processing.cache import reuse_knowledge,CRITICAL_CODE


def row(**changes):
    value=dict(city="杭州",id=7,name="测试博物馆",type="博物馆/纪念馆",lat=30.2,lon=120.1,
        price=20,opentime="09:00",endtime="24:00",recommendmintime=1,recommendmaxtime=2)
    value.update(changes); return value


class PipelineTests(unittest.TestCase):
    def test_duration_units_alias_and_lineage(self):
        poi=normalize_attraction(row(),{"name":"Test Museum"})
        self.assertEqual(poi.recommended_duration,90)
        self.assertEqual(poi.aliases,["Test Museum"])
        self.assertEqual(poi.source_metadata["license"],"CC BY 4.0")
        self.assertEqual(poi.popularity,.5)
        self.assertEqual(canonical_city("HANGZHOU"),"杭州")
        self.assertEqual(canonical_city("重慶市"),"重庆")

    def test_catalog_merges_entities_and_quarantines_bad_rows(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/"raw"; out=Path(temp)/"out"
            source=[row(id=1),row(id=2,lat=30.20001),row(id=3,lat=31.2),row(id=4,lat=float("nan"))]
            for language in ["zh","en"]:
                folder=raw/"chinatravel-sandbox/data"/language; folder.mkdir(parents=True)
                values=[{**r,"city":"Hangzhou","name":f"Museum {r['id']}"} for r in source] if language=="en" else source
                pd.DataFrame(values).to_parquet(folder/"attractions.parquet")
                for name in ["restaurants","accommodations","trains","flights","poi","subway_stations"]:
                    pd.DataFrame({"city":[]}).to_parquet(folder/f"{name}.parquet")
            summary=build_catalog(raw,out,["杭州"])
            self.assertEqual(summary["accepted_pois"],2)
            self.assertEqual(summary["merged_pois"],1)
            self.assertEqual(summary["rejected_pois"],1)
            pois=json.loads((out/"poi/hangzhou.json").read_text(encoding="utf-8"))
            self.assertEqual(set(pois[0]["aliases"]),{"Museum 1","Museum 2"})
            self.assertEqual(len(pois[0]["source_metadata"]["source_rows"]),2)
            rejected=json.loads((out/"quarantine_pois.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(rejected["reason"],"nonfinite_numeric")

    def test_invalid_source_values_are_rejected(self):
        for changes in [dict(lat=float("nan")),dict(lon=181),dict(price=-1),dict(recommendmintime=3,recommendmaxtime=1)]:
            with self.assertRaises(ValueError): normalize_attraction(row(**changes))
        self.assertEqual(normalize_attraction(row(recommendmintime=0)).recommended_duration,60)

    def test_overnight_is_explicitly_clipped_for_day_planner(self):
        opening,closing,notes=normalized_hours("18:00","02:00")
        self.assertEqual((opening,closing),("18:00","24:00"))
        self.assertIn("overnight_evening_window",notes)
        with self.assertRaises(ValueError): normalized_hours("09:00","09:00")

    def test_programs_are_data_not_executable(self):
        self.assertEqual(parse_programs('["result=True"]'),["result=True"])
        with self.assertRaises((ValueError,SyntaxError)):
            parse_programs("__import__('os').getcwd()")
        q=normalize_query({"uid":"p0e123","nature_language":"test","preference_py":"maximize attraction_count"},"preference0_base50.csv")
        self.assertEqual(q["preference_programs"],["maximize attraction_count"])
        self.assertEqual(q["family_key"],"e123")

    def test_related_queries_and_holdouts_cannot_leak(self):
        rows=[
            normalize_query({"uid":"e123","nature_language":"base"},"easy.csv"),
            normalize_query({"uid":"p1e123","nature_language":"preference"},"preference1_base50.csv"),
            normalize_query({"uid":"newid","nature_language":"base"},"human1000.csv"),
            normalize_query({"uid":"123","nature_language":"source seed"},"easy.csv"),
            normalize_query({"uid":"phase2_extended_123_00001","nature_language":"A"},"phase2/part-00000.csv"),
            normalize_query({"uid":"phase2_extended_123_00002","nature_language":"B"},"phase2/competition_test.csv"),
        ]
        assign_splits(rows,42)
        self.assertTrue(all(r["split"]=="test" for r in rows))
        before=[r["split_group"] for r in rows]; assign_splits(list(reversed(rows)),42)
        self.assertEqual(before,[r["split_group"] for r in rows])

    def test_duplicate_request_files_keep_holdout_provenance(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/"raw"; folder=raw/"chinatravel"; folder.mkdir(parents=True)
            out=Path(temp)/"out"; out.mkdir()
            for name in ["easy.csv","preference_base50.csv","competition_test.csv"]:
                with (folder/name).open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=["uid","nature_language"])
                    writer.writeheader(); writer.writerow(dict(uid="e123",nature_language="same request"))
            summary=build_queries(raw,out)
            self.assertEqual(summary["records"],1)
            self.assertEqual(summary["exact_duplicates"],2)
            result=json.loads((out/"queries/test.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(len(result["source_files"]),3)
            self.assertEqual(result["split_reason"],"official_holdout_family")

    def test_incremental_index_reuse_requires_matching_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); run=root/"runs/old"; (run/"knowledge").mkdir(parents=True)
            (run/"knowledge/test").write_text("fixture")
            quality=dict(knowledge={"chunks":1},index={"kind":"fixture"})
            write_json(run/"quality_report.json",quality)
            code={k:"same" for k in CRITICAL_CODE}
            inputs={"wikivoyage/en/file":{"sha256":"same"}}
            settings=dict(languages=["en"],chunk_size=1200,overlap=150)
            runtime={"python":"fixture"}
            manifest=dict(schema_version=1,build_id="old",validation_passed=True,code=code,inputs=inputs,
                settings=settings,runtime=runtime,artifacts={"knowledge/test":sha256(run/"knowledge/test"),"quality_report.json":sha256(run/"quality_report.json")})
            write_json(run/"manifest.json",manifest)
            stage=root/"stage"; stage.mkdir()
            reused=reuse_knowledge(root,stage,inputs,code,settings,runtime)
            self.assertEqual(reused[0]["reused_from"],"old")
            self.assertEqual((stage/"knowledge/test").read_text(encoding="utf-8"),"fixture")
            stage2=root/"other"; stage2.mkdir()
            self.assertIsNone(reuse_knowledge(root,stage2,inputs,code,{**settings,"overlap":20},runtime))
            (run/"knowledge/test").write_text("tampered")
            self.assertIsNone(reuse_knowledge(root,stage2,inputs,code,settings,runtime))

    def test_wiki_cleaner_keeps_listing_text_not_noise(self):
        value="<!-- hidden -->{{see|name=[[西湖]]|content=摄影与自然。{{icon|x}}}}<ref>citation noise</ref>[[File:test.jpg]]"
        result=clean_wikitext(value)
        self.assertIn("西湖",result); self.assertIn("摄影",result)
        self.assertNotIn("hidden",result); self.assertNotIn("citation noise",result)
        self.assertNotIn("test.jpg",result); self.assertNotIn("{{",result)
        chunks=list(text_chunks("自然摄影。"*200,200,30))
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(80<=len(x)<=200 for x in chunks))

    def test_wiki_destination_aliases_and_namespace_filter(self):
        page=ET.fromstring('<page xmlns="urn:test"><title>Hangzhou/West Lake</title><ns>0</ns><id>1</id><revision><id>2</id><text>'+('Photography and culture near West Lake. '*20)+'</text></revision></page>')
        docs=list(page_documents(page,"en","test.xml.bz2",300,50))
        self.assertTrue(docs)
        self.assertTrue(all(d["city"]=="杭州" and d["country"]=="China" for d in docs))
        self.assertTrue(all(d["source"].startswith("https://en.wikivoyage.org/") for d in docs))
        page.find('{urn:test}ns').text='14'
        self.assertEqual(list(page_documents(page,"en","test",300,50)),[])

    def test_vector_search_filters_other_cities_before_ranking(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            db=sqlite3.connect(folder/"documents.sqlite")
            db.execute("CREATE TABLE documents(row_idx INTEGER PRIMARY KEY,city TEXT,country TEXT,language TEXT,doc_id TEXT,metadata TEXT)")
            docs=[dict(doc_id="a",title="杭州",section="文化",city="杭州",country="China",language="zh",text="西湖文化摄影和博物馆",source="https://zh.wikivoyage.org/wiki/杭州"),
                  dict(doc_id="b",title="上海",section="文化",city="上海",country="China",language="zh",text="西湖文化摄影和博物馆 摄影摄影文化",source="https://zh.wikivoyage.org/wiki/上海")]
            for i,d in enumerate(docs): db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?)",(i,d["city"],d["country"],d["language"],d["doc_id"],json.dumps(d)))
            db.commit(); db.close(); build_index(folder)
            retriever=VectorKnowledgeRetriever(folder)
            results=retriever.retrieve_documents(UserProfile(city="杭州",interests=["photography","culture"]),5)
            self.assertEqual([d["doc_id"] for d in results],["a"])
            self.assertEqual(retriever.retrieve(UserProfile(city="Tokyo")),[])

    def test_bundle_rejects_path_escape_and_changed_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"runs/test").mkdir(parents=True)
            write_json(root/"current.json",dict(build_id="escape",path="../../escape",manifest_sha256="x"))
            with self.assertRaises(ValueError): resolve_runtime_data(dict(use_processed=True,bundle_root=str(root)))
            run=root/"runs/test"; (run/"x").write_text("original")
            manifest=dict(schema_version=1,build_id="test",validation_passed=True,artifacts={"x":sha256(run/"x")})
            write_json(run/"manifest.json",manifest)
            write_json(root/"current.json",dict(build_id="test",path="runs/test",manifest_sha256=sha256(run/"manifest.json")))
            (run/"x").write_text("changed")
            with self.assertRaises(ValueError): resolve_runtime_data(dict(use_processed=True,bundle_root=str(root)))


if __name__=="__main__": unittest.main(verbosity=2)
