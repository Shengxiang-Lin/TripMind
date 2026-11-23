"""Offline regressions for failures found during the real API smoke run."""
from collections import Counter
from pathlib import Path
import sys
import json
import ssl
import unittest
from dataclasses import replace
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tripmind.data import load_pois
from tripmind.geo import within_reach
from tripmind.models import POI, UserProfile, RankedPOI, RouteMatrix
from tripmind.map_provider import MapProvider, LocalMapProvider
from tripmind.planner import GreedyConstraintPlanner, _to_min
from tripmind.realtime import DashScopeRealtimeProvider, CachingRealtimeProvider, RealtimeProvider
from tripmind.service import TripMindService, itinerary_to_markdown
from tripmind.integrations.openrouteservice import TLSAdapter


class MatrixOnly(MapProvider):
    def __init__(self):
        self.calls = 0
    def route(self, *args):
        raise AssertionError("Planner must not issue pairwise HTTP calls")
    def matrix(self, pois, mode="walking"):
        self.calls += 1
        n = len(pois)
        d = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
        t = [[0 if i == j else 10 for j in range(n)] for i in range(n)]
        if n >= 3:
            d[0][1] = t[0][1] = None
            d[2][1] = t[2][1] = None
        return RouteMatrix([p.name for p in pois], d, t, "test_matrix", True)


class UnavailableRealtime(RealtimeProvider):
    def __init__(self):
        self.calls = 0
        self.last_errors = []
    def fetch(self, profile, pois):
        self.calls += 1
        self.last_errors = ["Realtime city lookup failed (AuthenticationError)."]
        return []


class RegressionTests(unittest.TestCase):
    def test_end_of_day_and_bad_times(self):
        self.assertEqual(_to_min("24:00"), 1440)
        for value in ["24:01", "25:00", "09:60", "9:00"]:
            with self.assertRaises(ValueError):
                _to_min(value)

    def test_matrix_once_and_unreachable_pair(self):
        pois = [replace(p, ticket_price=0, recommended_duration=30, opening_time="00:00", closing_time="24:00") for p in load_pois()[:3]]
        ranked = [RankedPOI(p, 1-i*.1, []) for i,p in enumerate(pois)]
        provider = MatrixOnly()
        days = GreedyConstraintPlanner(provider).plan(UserProfile(city="杭州", days=1), ranked)
        self.assertEqual(provider.calls, 1)
        self.assertEqual([i.poi.poi_id for i in days[0].items], [pois[0].poi_id, pois[2].poi_id])
        self.assertTrue(days[0].items[1].travel_from_previous.grounded)

    def test_suzhou_reach_filter_keeps_two_categories(self):
        fixture = ROOT / "evaluation" / "fixtures" / "suzhou_reach_pois.json"
        pois = [POI(**row) for row in json.loads(fixture.read_text(encoding="utf-8"))]
        # Jinji Lake is outside 8 km. A second real downtown non-culture POI
        # is needed to test two days of diversity without revisiting the museum.
        pois.append(replace(
            pois[-1], poi_id="suzhou-fixture:suzhou-park", name="苏州公园",
            latitude=31.31074, longitude=120.634291, recommended_duration=75,
        ))
        ranked = [
            RankedPOI(poi, 0.9 if index < 2 else 0.5, [])
            for index, poi in enumerate(pois)
        ]
        profile = UserProfile(city="苏州", days=2, interests=["culture", "history"], pace="relaxed")
        planner = GreedyConstraintPlanner(LocalMapProvider())
        for day in planner.plan(profile, ranked, all_pois=pois):
            counts = Counter(item.poi.category for item in day.items)
            self.assertGreaterEqual(len(counts), 2)
            self.assertGreaterEqual(len(day.items), 2)
            self.assertLessEqual(max(counts.values()), 2)
            self.assertTrue(all(within_reach(item.poi, profile.city, "walking", pois) for item in day.items))
        driving = planner.plan(replace(profile, transport_preference="driving"), ranked, all_pois=pois)
        selected = {item.poi.poi_id for day in driving for item in day.items}
        self.assertTrue({poi.poi_id for poi in pois[:2]} <= selected)

    def test_failed_lookup_is_visible_and_not_cached(self):
        inner = UnavailableRealtime()
        cache = CachingRealtimeProvider(inner)
        service = TripMindService(map_provider=LocalMapProvider(), realtime_provider=cache)
        profile = UserProfile(city="杭州", days=1)
        first = service.plan(profile)
        service.plan(profile)
        self.assertEqual(inner.calls, 2)
        self.assertEqual(first.realtime_status, "failed")
        self.assertIn("AuthenticationError", itinerary_to_markdown(first))
        self.assertNotIn("api_key", itinerary_to_markdown(first))

    def test_realtime_cache_distinguishes_trip_length(self):
        class Good(RealtimeProvider):
            def __init__(self): self.calls = 0
            def fetch(self, profile, pois): self.calls += 1; return []
        inner = Good(); cache = CachingRealtimeProvider(inner)
        profile = UserProfile(city="杭州", days=1)
        cache.fetch(profile, []); cache.fetch(profile, [])
        cache.fetch(replace(profile, days=2), [])
        self.assertEqual(inner.calls, 2)

    def test_model_string_false_cannot_close_a_poi(self):
        class FakeLLM:
            def chat(self, *args, **kwargs):
                return json.dumps({"signals": [{"kind":"closure", "subject":"西湖", "summary":"test", "effect":"avoid", "confidence":"high", "hard_constraint":"false"}]})
        provider = DashScopeRealtimeProvider(llm=FakeLLM())
        self.assertFalse(provider._query("test")[0].hard_constraint)

    def test_invalid_inputs_fail_before_external_lookups(self):
        inner = UnavailableRealtime()
        service = TripMindService(map_provider=LocalMapProvider(), realtime_provider=inner)
        profile = UserProfile(city="杭州", days=1)
        for invalid in [replace(profile, days=0), replace(profile, budget=float("nan")),
                        replace(profile, currency="USD"), replace(profile, start_date="2026-13-42")]:
            with self.assertRaises(ValueError):
                service.plan(invalid)
        self.assertEqual(inner.calls, 0)

    def test_tls_workaround_preserves_certificate_verification(self):
        adapter = TLSAdapter("TLSv1_2")
        self.assertEqual(adapter.context.maximum_version, ssl.TLSVersion.TLSv1_2)
        self.assertEqual(adapter.context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(adapter.context.check_hostname)


if __name__ == "__main__":
    unittest.main(verbosity=2)
