# Preference Learning Data

TripMind's small model is intended to learn **traveler preference**, not to memorize weather, realtime prices, road times, or temporary closures.

The preferred supervision strategy is now:

```text
Public preference data (main supervision)
        +
small ChinaTravel ranking calibration set
        |
        v
preference-aware small model
        |
        v
runtime RAG + realtime + routing + planner
```

This replaces the earlier idea of labeling thousands of ChinaTravel ranking instances with a paid Qwen teacher.

## 1. Public preference sources

### Stravl

Source: https://github.com/Stravl/Stravl-Data

Stravl contains 80k+ traveler profiles and 1M+ destination swipes. Each user supplies travel preferences such as age ranges, budget, season, desired experiences, scenery, activity level, safety preference, popularity preference, and target region. Destinations receive original `Yes / Maybe / No` swipe labels.

TripMind preserves these as the task:

```text
(profile, destination) -> yes / maybe / no
```

We do **not** invent a POI ranking from this dataset. User IDs are hashed again and all records from one user are assigned to the same deterministic 80/10/10 train/validation/test split.

### GenUP / LLM4POI processed POI datasets

Sources:

- https://huggingface.co/datasets/w11wo/FourSquare-NYC-POI
- https://huggingface.co/datasets/w11wo/FourSquare-TKY-POI
- https://huggingface.co/datasets/w11wo/Gowalla-CA-POI

These datasets contain user-profile / trajectory prompts with original next-POI targets. TripMind preserves the upstream supervision as:

```text
(user profile + check-in trajectory) -> next POI
```

The processed records already contain `system_prompt`, `inputs`, and `targets`; TripMind converts them into standard ChatML without changing the target semantics.

## 2. Download

Download only the public preference datasets:

```bash
python scripts/download_data.py --source preferences
```

Or individually:

```bash
python scripts/download_data.py --source stravl
python scripts/download_data.py --source genup-nyc
python scripts/download_data.py --source genup-tky
python scripts/download_data.py --source genup-ca
```

The original travel-planning data can still be downloaded separately:

```bash
python scripts/download_data.py --source core --languages zh en
```

`--source all` downloads both core travel data and public preference data.

## 3. Normalize the public preference corpus

```bash
python training/build_public_preference_dataset.py
```

Output:

```text
data/processed/public_preferences/
├── explicit_destination/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── next_poi/
│   ├── genup-nyc-*.jsonl
│   ├── genup-tky-*.jsonl
│   └── genup-ca-*.jsonl
└── manifest.json
```

Validate it with:

```bash
python evaluation/public_preferences.py \
  --root data/processed/public_preferences
```

The validator checks duplicate record IDs, Stravl user leakage across splits, label validity, and GenUP ChatML structure.

## 4. Why the tasks remain separate

Stravl is **explicit destination preference** while GenUP is **implicit next-POI behavior**. They are related but not identical supervision signals.

TripMind therefore does not coerce them into fabricated listwise ranking scores. A later multi-task trainer can use task-specific losses, for example:

```text
L = lambda_explicit * L_yes_maybe_no
  + lambda_nextpoi * L_next_poi
  + lambda_rank * L_chinatravel_rank
```

The public tasks should teach general preference/behavior patterns. ChinaTravel then provides a small target-domain alignment set for the final TripMind POI-ranking interface.

## 5. ChinaTravel ranking becomes calibration, not bulk supervision

The existing builder remains available:

```bash
python training/build_ranking_dataset.py \
  --label-mode none \
  --output data/processed/ranking
```

The default limits are now intentionally small:

```text
train       <= 300
validation  <= 100
test        <= 100
```

These ranking instances combine:

- ChinaTravel request / preference / constraints;
- target-city attraction candidates from ChinaTravel-Sandbox;
- city-filtered Wikivoyage static RAG.

`--label-mode heuristic` is still debug-only. If Qwen teacher labels are used, they should be restricted to a small calibration subset after the public-data pretraining stage rather than thousands of API calls.

Example:

```bash
python training/build_ranking_dataset.py \
  --label-mode dashscope \
  --splits train validation \
  --train-limit 100 \
  --validation-limit 30 \
  --output data/processed/ranking_teacher
```

## 6. Recommended training stages

```text
Stage A: preference pretraining
    Stravl explicit preference
    + GenUP next-POI behavior

Stage B: TripMind task alignment
    small ChinaTravel listwise ranking set

Stage C: evaluation
    held-out public splits
    + ChinaTravel held-out families
    + profile-sensitivity / NDCG / pairwise metrics
```

Dynamic route times, weather, temporary promotions and realtime closures remain outside the training corpus. They are handled by ORS, Alibaba realtime search and the deterministic planner at runtime.
