from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tripmind.bundle import resolve_runtime_data
from tripmind.config import get_config, resolve_project_path
from tripmind.data import city_pois, load_pois
from tripmind.models import POI, UserProfile
from tripmind.processing.common import stable_id
from tripmind.text import INTEREST_HINTS, tokens
from tripmind.vector_rag import load_vector_retriever


SYSTEM_PROMPT = """You are a travel recommendation teacher model.
Rank every candidate POI exactly once for the given traveler request.
Use only the provided request, structured trip metadata, candidate POI facts, and
retrieved static travel knowledge. Do not invent realtime facts, weather, ticket
promotions, opening-hour changes, or route times.

Return JSON only with this schema:
{"ranking":[
  {"poi_id":"...","score":0-100,"reason_tags":["...", "..."]}
]}

Rules:
- Include every candidate poi_id exactly once.
- Higher score means better personalized fit, not shorter route distance.
- Prefer user-specific relevance over generic popularity.
- Respect explicit hard constraints in the request when the provided POI facts allow it.
- Use 1-3 concise reason_tags per POI.
- Do not add any keys outside the schema.
"""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _query_text(record: dict[str, Any]) -> str:
    return " ".join(
        x
        for x in [
            record.get("request", ""),
            record.get("request_en", ""),
            record.get("preference", ""),
            record.get("preference_en", ""),
            " ".join(record.get("constraint_descriptions", []) or []),
        ]
        if x
    )


def _infer_interests(text: str) -> list[str]:
    q = set(tokens(text))
    hits = []
    for name, hints in INTEREST_HINTS.items():
        if q & set(tokens(hints)):
            hits.append(name)
    return hits


def _candidate_score(record_id: str, query_tokens: set[str], poi: POI) -> float:
    poi_text = " ".join(
        [
            poi.name,
            poi.category,
            poi.description,
            " ".join(poi.tags),
            " ".join(poi.aliases),
        ]
    )
    p = set(tokens(poi_text))
    overlap = len(query_tokens & p)
    lexical = overlap / max(1.0, math.sqrt(len(query_tokens) * max(1, len(p))))
    jitter = int(stable_id(record_id, poi.poi_id)[:8], 16) / 16**8
    return 0.82 * lexical + 0.15 * max(0.0, min(1.0, poi.popularity)) + 0.03 * jitter


def _select_candidates(
    record: dict[str, Any],
    pois: list[POI],
    count: int,
) -> tuple[list[POI], dict[str, float]]:
    query_tokens = set(tokens(_query_text(record)))
    record_id = str(record["record_id"])
    scored = [(_candidate_score(record_id, query_tokens, poi), poi) for poi in pois]
    scored.sort(key=lambda item: (item[0], item[1].poi_id), reverse=True)

    chosen: list[POI] = []
    chosen_ids: set[str] = set()
    seen_categories: set[str] = set()
    diversity_target = min(count, max(2, count // 2))

    for _, poi in scored:
        if len(chosen) >= diversity_target:
            break
        if poi.category in seen_categories:
            continue
        chosen.append(poi)
        chosen_ids.add(poi.poi_id)
        seen_categories.add(poi.category)

    for _, poi in scored:
        if len(chosen) >= count:
            break
        if poi.poi_id in chosen_ids:
            continue
        chosen.append(poi)
        chosen_ids.add(poi.poi_id)

    scores = {poi.poi_id: score for score, poi in scored if poi.poi_id in chosen_ids}
    return chosen, scores


def _profile_for_retrieval(record: dict[str, Any]) -> UserProfile:
    request_text = _query_text(record)
    return UserProfile(
        city=record["target_city"],
        country="China",
        days=int(record.get("days") or 2),
        budget=100000.0,
        currency="CNY",
        age_group="unspecified",
        travel_group="unspecified",
        interests=_infer_interests(request_text),
        pace="normal",
        special_preferences=[request_text],
    )


def _candidate_payload(poi: POI) -> dict[str, Any]:
    return {
        "poi_id": poi.poi_id,
        "name": poi.name,
        "aliases": poi.aliases,
        "category": poi.category,
        "ticket_price": poi.ticket_price,
        "currency": poi.currency,
        "recommended_duration_min": poi.recommended_duration,
        "opening_time": poi.opening_time,
        "closing_time": poi.closing_time,
        "tags": poi.tags,
        "indoor": poi.indoor,
        "description": poi.description,
    }


def _teacher_input(
    record: dict[str, Any],
    candidates: list[POI],
    docs: list[dict[str, Any]],
    max_knowledge_chars: int,
) -> dict[str, Any]:
    return {
        "request": record.get("request", ""),
        "request_en": record.get("request_en", ""),
        "trip_metadata": {
            "target_city": record.get("target_city"),
            "start_city": record.get("start_city"),
            "days": record.get("days"),
            "people_number": record.get("people_number"),
        },
        "preference": record.get("preference", ""),
        "preference_en": record.get("preference_en", ""),
        "hard_constraints": record.get("constraint_descriptions", []),
        "candidate_pois": [_candidate_payload(poi) for poi in candidates],
        "retrieved_static_knowledge": [
            {
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "section": doc.get("section"),
                "language": doc.get("language"),
                "text": str(doc.get("text", ""))[:max_knowledge_chars],
                "source": doc.get("source"),
            }
            for doc in docs
        ],
    }


def _extract_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Teacher response does not contain a JSON object")
    return json.loads(value[start : end + 1])


def _validate_label(payload: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("Teacher ranking must be a list")
    expected = set(candidate_ids)
    seen: set[str] = set()
    normalized = []
    for item in ranking:
        if not isinstance(item, dict):
            raise ValueError("Each ranking item must be an object")
        poi_id = str(item.get("poi_id", ""))
        if poi_id not in expected or poi_id in seen:
            raise ValueError("Teacher ranking contains a missing/duplicate/unknown poi_id")
        score = float(item.get("score"))
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("Teacher score must be finite and in [0, 100]")
        reasons = item.get("reason_tags", [])
        if not isinstance(reasons, list) or not 1 <= len(reasons) <= 3:
            raise ValueError("reason_tags must contain 1-3 strings")
        reasons = [str(reason).strip() for reason in reasons if str(reason).strip()]
        if not reasons:
            raise ValueError("reason_tags cannot be empty")
        seen.add(poi_id)
        normalized.append(
            {"poi_id": poi_id, "score": round(score, 2), "reason_tags": reasons[:3]}
        )
    if seen != expected:
        raise ValueError("Teacher ranking must contain every candidate exactly once")
    return {"ranking": normalized}


def _heuristic_label(candidate_ids: list[str], selector_scores: dict[str, float]) -> dict[str, Any]:
    values = [selector_scores[x] for x in candidate_ids]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    ranked = sorted(candidate_ids, key=lambda x: (selector_scores[x], x), reverse=True)
    return {
        "ranking": [
            {
                "poi_id": poi_id,
                "score": round(25 + 70 * (selector_scores[poi_id] - low) / span, 2),
                "reason_tags": ["heuristic_debug_only"],
            }
            for poi_id in ranked
        ]
    }


def _teacher_label(llm, user_payload: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    user_text = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    first = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
    )
    try:
        return _validate_label(_extract_json(first), candidate_ids)
    except Exception:
        repair = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": first},
                {
                    "role": "user",
                    "content": "Repair the previous answer to satisfy the JSON schema and include every candidate exactly once. Return JSON only.",
                },
            ],
            temperature=0.0,
        )
        return _validate_label(_extract_json(repair), candidate_ids)


def _split_limit(args: argparse.Namespace, config: dict[str, Any], split: str) -> int | None:
    cli = getattr(args, f"{split}_limit")
    if cli is not None:
        return None if cli <= 0 else cli
    value = config.get("training", {}).get("ranking", {}).get(f"{split}_limit")
    if value in (None, 0, ""):
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TripMind personalized POI ranking instances.")
    parser.add_argument("--output", default="data/processed/ranking")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation", "test"])
    parser.add_argument("--label-mode", choices=["none", "heuristic", "dashscope"], default="none")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--knowledge-top-k", type=int, default=None)
    parser.add_argument("--max-knowledge-chars", type=int, default=800)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    ranking_cfg = config.get("training", {}).get("ranking", {})
    candidate_count = int(args.candidate_count or ranking_cfg.get("candidate_count", 8))
    knowledge_top_k = int(args.knowledge_top_k or ranking_cfg.get("knowledge_top_k", 3))
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    if knowledge_top_k < 0:
        raise ValueError("knowledge_top_k must be nonnegative")

    data_cfg = resolve_runtime_data(config.get("data", {}))
    if not data_cfg.get("knowledge_index"):
        raise RuntimeError("Ranking dataset generation requires an active processed bundle. Run scripts/build_data.py first.")

    poi_dir = resolve_project_path(data_cfg["poi_dir"])
    bundle_run = poi_dir.parent
    query_dir = bundle_run / "queries"
    retriever = load_vector_retriever(str(resolve_project_path(data_cfg["knowledge_index"])))
    all_pois = load_pois(poi_dir)
    pois_by_city = {city: city_pois(all_pois, city, "China") for city in sorted({p.city for p in all_pois})}

    llm = None
    label_source = args.label_mode
    if args.label_mode == "dashscope":
        from tripmind.integrations.dashscope_llm import DashScopeLLM

        model = args.teacher_model or ranking_cfg.get("teacher_model") or config.get("dashscope", {}).get("model", "qwen-plus")
        llm = DashScopeLLM(model=model, config=config)
        label_source = f"dashscope:{model}"
    elif args.label_mode == "heuristic":
        label_source = "heuristic_debug_only"

    output = resolve_project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    failure_path = output / "failures.jsonl"
    stats: dict[str, Any] = {
        "schema_version": 1,
        "bundle_id": data_cfg.get("bundle_id"),
        "label_mode": args.label_mode,
        "label_source": label_source,
        "candidate_count": candidate_count,
        "knowledge_top_k": knowledge_top_k,
        "seed": args.seed,
        "splits": {},
    }

    with failure_path.open("a" if args.resume else "w", encoding="utf-8") as failure_stream:
        for split in args.splits:
            source_path = query_dir / f"{split}.jsonl"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            records = _jsonl(source_path)
            records.sort(key=lambda r: stable_id(args.seed, r["record_id"]))
            limit = _split_limit(args, config, split)
            if limit is not None:
                records = records[:limit]

            dest = output / f"{split}.jsonl"
            existing: dict[str, dict[str, Any]] = {}
            if args.resume and dest.exists():
                existing = {row["sample_id"]: row for row in _jsonl(dest)}

            written = reused = skipped = failed = 0
            with dest.open("w", encoding="utf-8") as stream:
                for record in records:
                    city = record.get("target_city")
                    city_poi_list = pois_by_city.get(city or "", [])
                    if not city or len(city_poi_list) < 2:
                        skipped += 1
                        continue
                    candidates, selector_scores = _select_candidates(record, city_poi_list, min(candidate_count, len(city_poi_list)))
                    candidate_ids = [p.poi_id for p in candidates]
                    sample_id = stable_id(record["record_id"], candidate_ids)[:24]
                    if sample_id in existing:
                        stream.write(json.dumps(existing[sample_id], ensure_ascii=False, sort_keys=True) + "\n")
                        stream.flush()
                        written += 1
                        reused += 1
                        continue

                    try:
                        profile = _profile_for_retrieval(record)
                        docs = retriever.retrieve_documents(profile, top_k=knowledge_top_k)
                        teacher_input = _teacher_input(record, candidates, docs, args.max_knowledge_chars)
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(teacher_input, ensure_ascii=False, sort_keys=True)},
                        ]
                        label = None
                        if args.label_mode == "heuristic":
                            label = _heuristic_label(candidate_ids, selector_scores)
                        elif args.label_mode == "dashscope":
                            label = _teacher_label(llm, teacher_input, candidate_ids)
                        if label is not None:
                            messages.append({"role": "assistant", "content": json.dumps(label, ensure_ascii=False, sort_keys=True)})

                        row = {
                            "schema_version": 1,
                            "sample_id": sample_id,
                            "split": split,
                            "split_group": record.get("split_group"),
                            "source_record_id": record["record_id"],
                            "source_uid": record.get("uid"),
                            "official_holdout": bool(record.get("official_holdout")),
                            "label_source": label_source,
                            "candidate_poi_ids": candidate_ids,
                            "knowledge_doc_ids": [doc.get("doc_id") for doc in docs],
                            "messages": messages,
                        }
                        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        stream.flush()
                        written += 1
                    except Exception as exc:
                        failed += 1
                        failure_stream.write(
                            json.dumps(
                                {
                                    "split": split,
                                    "record_id": record.get("record_id"),
                                    "uid": record.get("uid"),
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        failure_stream.flush()
                        if args.fail_fast:
                            raise

            stats["splits"][split] = {
                "source_records_considered": len(records),
                "written": written,
                "reused": reused,
                "skipped_without_usable_city_candidates": skipped,
                "failed": failed,
            }
            print(split, stats["splits"][split])

    if llm is not None:
        stats["teacher_api_calls"] = llm.call_count
    (output / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Wrote ranking dataset to", output)
    print("Label mode:", args.label_mode)
    if args.label_mode == "heuristic":
        print("WARNING: heuristic labels are debug-only and should not be used as final SFT supervision.")


if __name__ == "__main__":
    main()
