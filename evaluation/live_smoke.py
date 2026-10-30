"""Small paid-API smoke run. Fails if realtime or road routes silently fall back."""
from __future__ import annotations
import argparse
import json
import os
import re
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tripmind.config import clear_config_cache, get_config, resolve_project_path
from tripmind.models import UserProfile
from tripmind.planner import _to_min, _blocked
from tripmind.service import TripMindService, itinerary_to_markdown


def validate(result):
    p = result.profile
    assert len(result.days) == p.days
    assert result.retrieved_knowledge, "No static knowledge was retrieved"
    assert result.realtime_status == "ok", result.warnings
    assert not result.warnings, result.warnings
    assert all(day.items for day in result.days), "Empty itinerary day"
    ids = [i.poi.poi_id for d in result.days for i in d.items]
    assert len(ids) == len(set(ids)), "Duplicate POI"
    assert result.total_estimated_cost <= p.budget
    assert abs(result.total_estimated_cost - sum(d.estimated_cost for d in result.days)) < .01
    blocked = {s.subject for s in result.realtime_signals if s.hard_constraint and s.effect == "avoid"}
    grounded = 0
    for day in result.days:
        assert day.estimated_cost <= p.budget / p.days * .65 + .01
        assert abs(day.estimated_cost-sum(i.poi.ticket_price for i in day.items)) < .01
        for index,item in enumerate(day.items):
            assert not _blocked(item.poi.name, blocked), "Closed POI selected"
            assert item.start_min >= max(540, _to_min(item.poi.opening_time))
            assert item.end_min <= min(1200, _to_min(item.poi.closing_time))
            assert item.end_min-item.start_min == item.poi.recommended_duration
            if index:
                leg = item.travel_from_previous
                assert leg and leg.grounded and leg.source == "openrouteservice", "Estimated road leg"
                assert item.start_min >= day.items[index-1].end_min + 25 + leg.duration_min
                grounded += 1
    assert grounded > 0, "No road leg was exercised"
    return grounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-config", default="data/processed/sample/config.yaml")
    parser.add_argument("--output", default="data/processed/live_smoke")
    parser.add_argument("--start-date", default=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat())
    parser.add_argument("--production", action="store_true", help="Use the validated full data bundle")
    parser.add_argument("--ui", action="store_true", help="Also submit a cached request through the Gradio HTTP API")
    args = parser.parse_args()
    if not args.production:
        os.environ["TRIPMIND_CONFIG"] = str(resolve_project_path(args.sample_config))
    clear_config_cache()
    config = get_config()
    if args.production:
        config["data"]["use_processed"] = True
    service = TripMindService(config=config)
    if args.production:
        assert service.data_bundle_id, "Production bundle was not selected"
    assert hasattr(service.map, "primary"), "ORS provider was not enabled"
    assert hasattr(service.realtime, "inner"), "DashScope provider was not enabled"
    if args.production:
        query_file = service.rag.folder.parent/"queries/all.jsonl"
        row = next(r for r in map(json.loads,query_file.open(encoding="utf-8"))
                   if "chinatravel/easy.csv" in r["source_files"] and r["target_city"]=="杭州" and r["days"]==1)
        query = {"uid":row["uid"],"nature_language":row["request"],"target_city":row["target_city"],
                 "days":row["days"],"people_number":row["people_number"]}
    else:
        query = json.loads((resolve_project_path(args.sample_config).parent/"queries.json").read_text())[0]
    budget = re.search(r"预算\s*(\d+)", query["nature_language"])
    first = UserProfile(
        city=query["target_city"], days=int(query["days"]),
        budget=float(budget.group(1)) if budget else 1500,
        travel_group="solo" if int(query["people_number"]) == 1 else "friends",
        interests=["culture", "photography"], start_date=args.start_date,
    )
    profiles = [first, UserProfile(
        city="杭州", days=2, budget=800, travel_group="family",
        interests=["museum", "nature", "culture"], pace="normal", start_date=args.start_date,
    )]
    out = resolve_project_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for number, profile in enumerate(profiles, 1):
        started = time.monotonic(); calls = service.map.primary.api_calls
        usage_start = len(service.realtime.inner.usage_history)
        print(f"LIVE case {number}: {profile.travel_group}, {profile.days} days", flush=True)
        result = service.plan(profile)
        grounded = validate(result)
        matrix_calls = service.map.primary.api_calls-calls
        assert matrix_calls <= 1, "Too many routing calls for a small plan"
        stem = f"case_{number}"
        (out/f"{stem}.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2)+"\n")
        (out/f"{stem}.md").write_text(itinerary_to_markdown(result)+"\n")
        summaries.append({
            "case": number, "benchmark_uid": query["uid"] if number == 1 else None,
            "seconds": round(time.monotonic()-started, 2), "days": profile.days,
            "poi_count": sum(len(d.items) for d in result.days), "ticket_cost": result.total_estimated_cost,
            "realtime_status": result.realtime_status, "realtime_signal_count": len(result.realtime_signals),
            "grounded_legs": grounded, "routing_api_calls": matrix_calls,
            "llm_usage": service.realtime.inner.usage_history[usage_start:],
        })
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    report = {"sample_config": None if args.production else args.sample_config, "bundle_id":service.data_bundle_id, "cases": summaries}
    if args.ui:
        import app
        import requests
        from gradio_client import Client
        app.service = service
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]
        llm_before = service.realtime.inner.llm.call_count
        route_before = service.map.primary.api_calls
        try:
            app.demo.launch(
                server_name="127.0.0.1", server_port=port, share=False,
                prevent_thread_lock=True, quiet=True,
            )
            url = f"http://127.0.0.1:{port}"
            assert requests.get(url, timeout=10).status_code == 200
            client = Client(url, verbose=False)
            p = profiles[0]
            markdown = client.predict(
                p.country, p.city, p.start_date, p.days, p.budget, p.currency,
                p.age_group, p.travel_group, p.interests, p.pace,
                p.transport_preference, "", api_name="/plan",
            )
            assert isinstance(markdown, str) and "Planning failed" not in markdown
            assert "grounded" in markdown and "Realtime context" in markdown
            (out/"ui_result.md").write_text(markdown+"\n")
            report["ui"] = {
                "http_status": 200, "prediction_ok": True,
                "additional_llm_calls": service.realtime.inner.llm.call_count-llm_before,
                "additional_route_calls": service.map.primary.api_calls-route_before,
            }
            assert report["ui"]["additional_llm_calls"] == 0, "Realtime cache did not prevent repeat searches"
            assert report["ui"]["additional_route_calls"] == 0, "Route matrix was not reused"
            print("GRADIO HTTP PREDICTION OK", json.dumps(report["ui"]), flush=True)
        finally:
            app.demo.close()
    (out/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print("LIVE END-TO-END CHECKS PASSED:", out, flush=True)


if __name__ == "__main__":
    main()
