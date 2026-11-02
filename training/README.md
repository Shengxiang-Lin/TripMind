# TripMind GenUP Preference Model

Main task: `score(profile, history, candidate)` with a Qwen backbone plus scalar head. The real GenUP next POI is positive; POIs never observed for that user are negatives. Dataset-local POI IDs are not included in model prompts, so the scorer learns transferable profile/history/category preference rather than memorizing NYC/Tokyo/California IDs.

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

Small training first:

```bash
python training/train_preference_model.py --max-train-contexts 1000 --max-validation-contexts 300 --epochs 1 --output checkpoints/tripmind-preference-smoke
python evaluation/evaluate_preference_model.py --checkpoint checkpoints/tripmind-preference-smoke/best --split test --max-contexts 300
```

Full training:

```bash
python training/train_preference_model.py
python evaluation/evaluate_preference_model.py --checkpoint checkpoints/tripmind-preference/best --split test --output data/processed/genup_preference/test_metrics.json
python evaluation/evaluate_preference_model.py --checkpoint checkpoints/tripmind-preference/best --split test --no-history --max-contexts 1000
python evaluation/evaluate_preference_model.py --checkpoint checkpoints/tripmind-preference/best --split test --no-profile --max-contexts 1000
python evaluation/preference_service_smoke.py --checkpoint checkpoints/tripmind-preference/best
python app.py
```

Default base model is `Qwen/Qwen2.5-0.5B-Instruct`. Training uses pairwise `softplus(score_negative-score_positive)`. Twenty-five percent of training pairs mask history for cold-start robustness. The checkpoint selected as `best/` maximizes validation pairwise accuracy.

Runtime uses `providers.preference: auto`: a valid checkpoint activates `learned-genup`; otherwise TripMind stays on the heuristic baseline. Pairwise logits are not probabilities. For the existing planner/realtime scale, logits are z-scored within the candidate set and passed through sigmoid. A 15% heuristic blend is retained by default because GenUP does not directly supervise TripMind ticket budget, pace, or travel-group constraints.

Responsibility boundary:

```text
GenUP scorer -> long-term personalized POI preference
Wikivoyage   -> stable travel knowledge
Realtime     -> changing external context
ORS          -> road distance/time
Planner      -> budget/time/hours/route feasibility
```
