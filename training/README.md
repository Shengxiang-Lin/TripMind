# TripMind GenUP Preference Model

TripMind's learned personalization model is deliberately scoped to a task that the public GenUP data can actually supervise:

```text
(profile + historical place types + candidate category) -> category preference score
```

The original GenUP next POI supplies the positive category. Negatives are sampled only from POIs that were never observed for that user and whose category differs from the positive category. Dataset-local POI IDs are retained only as provenance and are never included in model prompts.

This means the learned model supports **cross-category preference ranking**, not fine-grained ranking between two POIs of the same category. Runtime candidate prompts therefore also contain only the POI category. TripMind's heuristic layer and planner continue to use rich attributes such as tags, group suitability, ticket price, popularity, duration and route feasibility to differentiate POIs within a category.

## Server commands

```bash
conda activate tripmind
python -m pip install -r requirements-training.txt
python scripts/download_data.py --source genup
python training/build_genup_preference_dataset.py
python evaluation/genup_preference.py --root data/processed/genup_preference
python evaluation/genup_preference_fixture.py
python evaluation/preference_ranker_smoke.py
```

The validator should explicitly report:

```text
task: sampled cross-category category-preference ranking
same-category ranking supported: False
```

## Small training first

```bash
python training/train_preference_model.py \
  --max-train-contexts 1000 \
  --max-validation-contexts 300 \
  --epochs 1 \
  --output checkpoints/tripmind-preference-smoke

python evaluation/evaluate_preference_model.py \
  --checkpoint checkpoints/tripmind-preference-smoke/best \
  --split test \
  --max-contexts 300
```

Training uses a Qwen backbone plus a scalar head and pairwise loss:

```text
softplus(score_negative - score_positive)
```

Twenty-five percent of training pairs mask history for cold-start robustness. The checkpoint under `best/` maximizes validation pairwise accuracy.

## Full training

```bash
python training/train_preference_model.py

python evaluation/evaluate_preference_model.py \
  --checkpoint checkpoints/tripmind-preference/best \
  --split test \
  --output data/processed/genup_preference/test_metrics.json
```

The reported ranking metrics are explicitly **sampled cross-category metrics**:

- `pairwise_accuracy`
- `sampled_hit_at_1`
- `sampled_mrr`
- `sampled_ndcg`
- `average_positive_rank`
- `average_top_margin`

They must not be interpreted as evidence that the model can distinguish two attractions from the same category.

## Ablations

```bash
python evaluation/evaluate_preference_model.py \
  --checkpoint checkpoints/tripmind-preference/best \
  --split test \
  --no-history \
  --max-contexts 1000

python evaluation/evaluate_preference_model.py \
  --checkpoint checkpoints/tripmind-preference/best \
  --split test \
  --no-profile \
  --max-contexts 1000
```

## Resume interrupted training

`last/` is a resumable training checkpoint. In addition to backbone/head/tokenizer weights it contains:

```text
training_state.pt
```

with optimizer state, scheduler state, completed epoch, global step, best validation metric, training history and Python/Torch/CUDA RNG state.

Resume with:

```bash
python training/train_preference_model.py \
  --resume-from checkpoints/tripmind-preference/last
```

To continue for more epochs, increase the target epoch count while resuming, for example:

```bash
python training/train_preference_model.py \
  --resume-from checkpoints/tripmind-preference/last \
  --epochs 4
```

`best/` is intended for inference/evaluation. Use `last/` for resume because only `last/` carries optimizer/scheduler state.

## Runtime integration

Runtime uses `providers.preference: auto`: a valid checkpoint activates the learned GenUP scorer; otherwise TripMind stays on the heuristic baseline.

The learned model receives the same candidate semantics used at training time:

```text
Category: Museum.
```

It does **not** receive runtime-only name, price, tags or description. Those richer POI features remain available to the heuristic ranker and planner. The default blend is:

```text
85% normalized learned category preference
15% heuristic POI preference
```

Pairwise logits are not probabilities. For the existing planner/realtime scale, logits are z-scored within the candidate set and then passed through sigmoid before blending.

The profile/history surface is also normalized: GenUP user IDs are stripped, both training and runtime profiles are wrapped as `Traveler profile: ...`, and historical data is represented under a shared place/place-type wrapper. Runtime history can still be noisier than GenUP category histories; history dropout is therefore retained and no-history evaluation should always be reported.

## End-to-end test

```bash
python evaluation/preference_service_smoke.py \
  --checkpoint checkpoints/tripmind-preference/best

python app.py
```

The Gradio endpoint accepts the added `travel_history` field. `evaluation/live_smoke.py --ui` now submits all 13 arguments.

Responsibility boundary:

```text
GenUP scorer -> long-term cross-category traveler preference
Heuristic    -> within-category POI differentiation / TripMind-specific static features
Wikivoyage   -> stable travel knowledge
Realtime     -> changing external context
ORS          -> road distance/time
Planner      -> budget/time/hours/route feasibility
```
