from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.models import UserProfile
from tripmind.service import TripMindService


def main():
    service = TripMindService()
    profile = UserProfile(
        city="杭州",
        days=2,
        budget=1500,
        travel_group="couple",
        interests=["photography", "food", "culture"],
        pace="relaxed",
    )
    result = service.plan(profile)
    assert len(result.days) == 2
    assert any(day.items for day in result.days)
    assert result.total_estimated_cost <= profile.budget
    names = [item.poi.name for day in result.days for item in day.items]
    print("SMOKE OK")
    print("Selected:", " -> ".join(names))
    print("Ticket cost:", result.total_estimated_cost)


if __name__ == "__main__":
    main()
