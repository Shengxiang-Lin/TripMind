# GenUP Preference Learning

TripMind now uses **GenUP as the primary supervision source for the small preference model**. The main training path no longer requires Stravl, bulk Qwen teacher labeling, or synthetic ChinaTravel ranking labels.

The small model learns one stable function:

```text
behavior-derived user profile
        +
historical POI trajectory
        +
candidate POI
        |
        v
personalized preference score
```

At runtime TripMind applies this scorer to candidate POIs, then leaves weather, temporary closures, route time and itinerary feasibility to realtime tools, ORS and the deterministic planner.

## 1. GenUP data

TripMind uses:

- `w11wo/FourSquare-NYC-POI`
- `w11wo/FourSquare-TKY-POI`
- `w11wo/Gowalla-CA-POI`

The processed GenUP records expose:

```text
system_prompt -> behavior-derived natural-language user profile
inputs        -> chronological historical POI trajectory
targets       -> real next POI
```

The profile is behavior-derived by GenUP. Demographic/personality text should not be described as independently collected ground-truth demographics.

## 2. Download

```bash
python scripts/download_data.py --source genup
```

`--source preferences` is retained as a backward-compatible alias for the same GenUP group.

Individual domains:

```bash
python scripts/download_data.py --source genup-nyc
python scripts/download_data.py --source genup-tky
python scripts/download_data.py --source genup-ca
```

`--source all` downloads the core TripMind travel data plus GenUP. Stravl is no longer part of the default path; it is optional only.

## 3. Build scorer supervision

```bash
python training/build_genup_preference_dataset.py
```

Output:

```text
data/processed/genup_preference/
├── nyc-train.jsonl
├── nyc-validation.jsonl
├── nyc-test.jsonl
├── tky-train.jsonl
├── tky-validation.jsonl
├── tky-test.jsonl
├── ca-train.jsonl
├── ca-validation.jsonl
├── ca-test.jsonl
└── manifest.json
```

GenUP exposes train/test splits. TripMind preserves upstream test and deterministically carves 10% of upstream train into validation.

## 4. How each context is built

For every GenUP row:

1. `system_prompt` is kept as `profile_text`.
2. `inputs` is kept as `history_text` and parsed for historical POI/category IDs.
3. The original `targets` next POI is the **positive candidate**.
4. Historical POIs are never sampled as negatives.
5. Any POI known to be a target for the same user elsewhere in the downloaded GenUP data is also excluded from negatives, reducing false-negative noise.
6. By default four negatives are sampled: approximately half same-category hard negatives when category metadata is available, and half random unobserved POIs.

Example:

```json
{
  "task": "candidate_poi_preference",
  "profile_text": "You are user ... preferences ...",
  "history_text": "... visited POI id ...",
  "positive_candidate": {
    "poi_id": "142",
    "category": "Japanese Restaurant"
  },
  "negative_candidates": [
    {"poi_id": "...", "negative_type": "same_category_hard"},
    {"poi_id": "...", "negative_type": "random"}
  ]
}
```

No LLM API is called during this build.

## 5. Intended model objective

The preferred model interface is:

```text
s_theta(profile, history, candidate) -> scalar preference score
```

For every real next POI `p+` and sampled negative `p-`, use pairwise ranking loss:

```text
L_pair = -log sigmoid(s(profile, history, p+) - s(profile, history, p-))
```

A listwise softmax over one positive plus N negatives is also compatible. The model should learn **relative user-specific POI preference**, not an arbitrary 0-100 teacher score.

Runtime mapping:

```text
candidate POIs
    |
    +--> scorer(user, history, POI_1)
    +--> scorer(user, history, POI_2)
    +--> ...
    |
    v
personalized Top-K
    |
    v
realtime -> routing -> constraint planner
```

## 6. Validate

```bash
python evaluation/genup_preference.py \
  --root data/processed/genup_preference
```

The validator checks unique contexts, one original positive per context, negative counts, duplicate/invalid negatives, history leakage into negatives, and agreement with the original GenUP target.

## 7. Controls

Defaults live in `configs/config.yaml`:

```yaml
training:
  preference:
    sources: [nyc, tky, ca]
    validation_fraction: 0.10
    negatives_per_positive: 4
    hard_negative_fraction: 0.50
```

Override example:

```bash
python training/build_genup_preference_dataset.py \
  --sources nyc tky \
  --negatives-per-positive 6 \
  --hard-negative-fraction 0.5
```

## 8. What is no longer on the main path

`build_public_preference_dataset.py` and Stravl remain only for optional auxiliary experiments. `build_ranking_dataset.py` remains a research/debug utility, but paid Qwen teacher ranking is **not required** for the main TripMind preference model.

ChinaTravel / ChinaTravel-Sandbox still serve the planning side: realistic travel requests, structured destination data and itinerary evaluation. Wikivoyage remains static RAG. They do not need to manufacture preference labels for the small model.
