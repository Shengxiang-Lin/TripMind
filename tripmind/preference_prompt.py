from __future__ import annotations

import re
from typing import Any, Iterable

from .models import POI, UserProfile

PROMPT_VERSION = "tripmind-preference-v2-category"
NO_HISTORY = "No prior travel history is available."
NO_PROFILE = "No explicit traveler profile is available."
_USER_PREFIX_RE = re.compile(r"^\s*You are user\s+\d+\s*(?:and\s+)?", re.IGNORECASE)


def _compact(text: str, limit: int | None = None) -> str:
    value = " ".join(str(text or "").split())
    if limit and len(value) > limit:
        value = value[-limit:]
    return value


def _wrap_profile(text: str) -> str:
    value = _compact(text, 2600)
    return f"Traveler profile: {value}" if value else NO_PROFILE


def profile_text_from_record(record: dict[str, Any], *, no_profile: bool = False) -> str:
    if no_profile:
        return NO_PROFILE
    raw = _compact(record.get("profile_text", ""), 2600)
    # Remove the dataset-local user identifier and keep the semantic profile only.
    raw = _USER_PREFIX_RE.sub("", raw).strip(" .")
    return _wrap_profile(raw)


def profile_text_from_user(profile: UserProfile) -> str:
    interests = ", ".join(profile.interests) if profile.interests else "unspecified"
    special = ", ".join(profile.special_preferences) if profile.special_preferences else "none"
    value = (
        f"Age group: {profile.age_group}; "
        f"travel group: {profile.travel_group}; "
        f"interests: {interests}; "
        f"travel pace: {profile.pace}; "
        f"transport preference: {profile.transport_preference}; "
        f"budget: {profile.currency} {profile.budget:.2f} for {profile.days} days; "
        f"special preferences: {special}."
    )
    return _wrap_profile(value)


def history_text_from_record(record: dict[str, Any], *, no_history: bool = False) -> str:
    if no_history:
        return NO_HISTORY
    categories = [_compact(x) for x in record.get("history_categories", []) if _compact(x)]
    if categories:
        return "Observed historical place types: " + " -> ".join(categories[-80:])
    return NO_HISTORY


def history_text_from_user(profile: UserProfile) -> str:
    if not profile.travel_history:
        return NO_HISTORY
    # Runtime history can be names or category-like descriptions. The shared wrapper
    # keeps the textual surface close to GenUP; history dropout during training makes
    # this field optional rather than a hard dependency.
    values = [_compact(x) for x in profile.travel_history if _compact(x)]
    return "Observed historical places or place types: " + " -> ".join(values[-80:])


def candidate_category_text(category: str | None) -> str:
    return f"Category: {_compact(category or 'unknown')}."


def candidate_text_from_record(candidate: dict[str, Any]) -> str:
    # GenUP only supplies transferable category semantics once dataset-local POI IDs
    # are removed. Do not pretend that name/price/description supervision exists.
    return candidate_category_text(candidate.get("category"))


def candidate_text_from_poi(poi: POI) -> str:
    # IMPORTANT: learned inference uses the same information available during GenUP
    # training. Rich POI attributes are handled by the heuristic/planner layers.
    return candidate_category_text(poi.category)


def build_preference_prompt(
    profile_text: str,
    history_text: str,
    candidate_text: str,
    *,
    history_char_limit: int = 6000,
) -> str:
    profile = _compact(profile_text, 2600) or NO_PROFILE
    history = _compact(history_text, history_char_limit) or NO_HISTORY
    candidate = _compact(candidate_text, 1800)
    return (
        "Estimate the traveler's long-term preference for the candidate place type. "
        "Use traveler preference and historical place types only. Do not use realtime "
        "weather, temporary closures, route time, current promotions, or dataset-local POI IDs.\n\n"
        f"[TRAVELER PROFILE]\n{profile}\n\n"
        f"[TRAVEL HISTORY]\n{history}\n\n"
        f"[CANDIDATE PLACE TYPE]\n{candidate}\n\n"
        "[PREFERENCE REPRESENTATION]"
    )


def build_record_prompt(
    record: dict[str, Any],
    candidate: dict[str, Any],
    *,
    no_history: bool = False,
    no_profile: bool = False,
    history_char_limit: int = 6000,
) -> str:
    return build_preference_prompt(
        profile_text_from_record(record, no_profile=no_profile),
        history_text_from_record(record, no_history=no_history),
        candidate_text_from_record(candidate),
        history_char_limit=history_char_limit,
    )


def build_runtime_prompts(
    profile: UserProfile,
    candidates: Iterable[POI],
    *,
    history_char_limit: int = 6000,
) -> list[str]:
    profile_text = profile_text_from_user(profile)
    history_text = history_text_from_user(profile)
    return [
        build_preference_prompt(
            profile_text,
            history_text,
            candidate_text_from_poi(poi),
            history_char_limit=history_char_limit,
        )
        for poi in candidates
    ]
