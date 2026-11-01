# Personalized POI Ranking Data

TripMind's final personalized model is intended to learn **which POIs fit a traveler**, not to memorize dynamic route times, weather, or current promotions.

The production data bundle already contains three useful ingredients:

- ChinaTravel requests and preference/constraint text in leakage-safe train/validation/test splits;
- normalized attraction candidates from ChinaTravel-Sandbox;
- city-filtered Wikivoyage retrieval results.

ChinaTravel does **not** provide a ground-truth POI ranking for each request. Therefore `build_ranking_dataset.py` separates **instance construction** from **label creation** instead of silently treating heuristic scores as supervision.

## 1. Build the production data bundle first

```bash
python scripts/download_data.py --source all --languages zh en
python scripts/build_data.py
```

The ranking builder reads the active validated bundle under `data/processed/production/current.json`.

## 2. Build unlabeled teacher-ready instances

```bash
python training/build_ranking_dataset.py \
  --label-mode none \
  --output data/processed/ranking
```

Default target limits are controlled by `configs/config.yaml`:

```text
train       <= 4000
validation  <= 400
test        <= 400
```

For each source request, the builder:

1. keeps the existing leakage-safe ChinaTravel split group;
2. selects a deterministic, category-diverse set of POIs from the target city;
3. retrieves static Wikivoyage context from the local production RAG index;
4. creates a strict ranking prompt containing only request/profile text, POI facts and static knowledge.

Dynamic facts are intentionally absent from the teacher prompt.

## 3. Smoke/debug labels

```bash
python training/build_ranking_dataset.py \
  --label-mode heuristic \
  --train-limit 100 \
  --validation-limit 20 \
  --test-limit 20 \
  --output data/processed/ranking_debug

python evaluation/ranking_dataset.py --root data/processed/ranking_debug
```

`heuristic_debug_only` labels exist only to test the SFT schema and pipeline. They should **not** be used as the final training target.

## 4. Qwen teacher labeling

Configure a DashScope key in `configs/config.local.yaml`, then start with a small batch:

```bash
python training/build_ranking_dataset.py \
  --label-mode dashscope \
  --teacher-model qwen-plus \
  --splits train validation \
  --train-limit 50 \
  --validation-limit 20 \
  --output data/processed/ranking_teacher
```

The Qwen teacher is called **without web search**. It must rank every candidate exactly once and return strict JSON. Invalid outputs get one repair attempt and then go to `failures.jsonl` rather than being silently accepted.

For an interrupted paid labeling run, use `--resume` to reuse already written sample IDs:

```bash
python training/build_ranking_dataset.py \
  --label-mode dashscope \
  --splits train validation \
  --resume \
  --output data/processed/ranking_teacher
```

Before scaling to thousands of samples, inspect a small labeled subset manually. The recommended final supervision workflow is:

```text
ChinaTravel request split
        +
city POI candidates
        +
Wikivoyage static RAG
        |
        v
strong Qwen teacher
        |
        v
schema / consistency filtering
        |
        v
human spot checks
        |
        v
SFT train / validation set
```

The official/held-out test families remain separated by the upstream ChinaTravel split logic. Do not merge train, validation and test JSONL files before training.

## 5. Validate the generated dataset

```bash
python evaluation/ranking_dataset.py --root data/processed/ranking_teacher
```

The validator checks:

- unique sample IDs;
- valid ChatML role order;
- every candidate ranked exactly once when a label exists;
- no `split_group` overlap across train/validation/test.

## Output schema

Each row contains metadata plus ChatML messages:

```json
{
  "sample_id": "...",
  "split": "train",
  "split_group": "...",
  "source_record_id": "...",
  "source_uid": "...",
  "official_holdout": false,
  "label_source": "dashscope:qwen-plus",
  "candidate_poi_ids": ["..."],
  "knowledge_doc_ids": ["..."],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"ranking\":[...]}"}
  ]
}
```

This format is deliberately independent of realtime search and routing so that the personalized ranker learns a stable preference policy.
