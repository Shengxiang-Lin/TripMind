from __future__ import annotations

from typing import Any, Iterable

from .models import POI, UserProfile

PROMPT_VERSION = "tripmind-preference-v1"
NO_HISTORY = "No prior travel history is available."


def _compact(text: str, limit: int | None = None) -> str:
    value = " ".join(str(text or "").split())
    if limit and len(value) > limit:
        value = value[-limit:]
    return value


def profile_text_from_user(profile: UserProfile) -> str:
    interests = ", ".join(profile.interests) if profile.interests else "unspecified"
    special = ", ".join(profile.special_preferences) if profile.special_preferences else "none"
    return (
        f"Age group: {profile.age_group}. "
        f"Travel group: {profile.travel_group}. "
        f"Interests: {interests}. "
        f"Travel pace: {profile.pace}. "
        f"Transport preference: {profile.transport_preference}. "
        f"Budget: {profile.currency} {profile.budget:.2f} for {profile.days} days. "
        f"Special preferences: {special}."
    )


def history_text_from_user(profile: UserProfile) -> str:
    if not profile.travel_history:
        return NO_HISTORY
    return "Previously visited or liked places: " + " -> ".join(profile.travel_history)


def history_text_from_record(record: dict[str, Any], *, no_history: bool = False) -> str:
    if no_history:
        return NO_HISTORY
    categories = [
        _compact(x)
        for x in record.get("history_categories", [])
        if _compact(x)
    ]
    if categories:
        return "Observed historical POI categories: " + " -> ".join(categories[-80:])
    raw = _compact(record.get("history_text", ""))
    return raw or NO_HISTORY


def candidate_text_from_record(candidate: dict[str, Any]) -> str:
    # Deliberately omit dataset-local POI IDs. The scorer must learn transferable
    # semantic preference rather than memorize a city-specific identifier.
    category = _compact(candidate.get("category") or "unknown")
    return f"Category: {category}."


def candidate_text_from_poi(poi: POI) -> str:
    tags = ", ".join(poi.tags[:12]) if poi.tags else "none"
    suitable = ", ".join(poi.suitable_groups[:8]) if poi.suitable_groups else "unspecified"
    description = _compact(poi.description, 1200)
    return (
        f"Name: {poi.name}. "
        f"Category: {poi.category}. "
        f"Tags: {tags}. "
        f"Suitable groups: {suitable}. "
        f"Intensity: {poi.intensity}. "
        f"Ticket price: {poi.currency} {poi.ticket_price:.2f}. "
        f"Typical visit duration: {poi.recommended_duration} minutes. "
        f"Description: {description or 'not provided'}."
    )


def build_preference_prompt(
    profile_text: str,
    history_text: str,
    candidate_text: str,
    *,
    history_char_limit: int = 6000,
) -> str:
    profile = _compact(profile_text, 2600)
    history = _compact(history_text, history_char_limit) or NO_HISTORY
    candidate = _compact(candidate_text, 1800)
    return (
        "Estimate how well the candidate place fits this traveler. "
        "Use long-term traveler preference and visit history, not realtime weather, "
        "temporary closures, route time, or current promotions.\n\n"
        f"[TRAVELER PROFILE]\n{profile}\n\n"
        f"[TRAVEL HISTORY]\n{history}\n\n"
        f"[CANDIDATE PLACE]\n{candidate}\n\n"
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
    profile_text = (
        "No explicit traveler profile is available."
        if no_profile
        else str(record.get("profile_text", ""))
    )
    return build_preference_prompt(
        profile_text,
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
