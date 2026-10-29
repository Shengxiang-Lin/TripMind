# TripMind

**Personalized · Knowledge-Grounded · Constraint-Aware Travel Planning**

TripMind is an LLM-enhanced travel itinerary recommender designed for the course project *Large Language Model for Recommendation System*. The implementation deliberately separates:

- **Preference learning** — which POIs fit this user?
- **Travel knowledge retrieval** — what combinations and planning patterns are useful?
- **Grounded map facts** — how far / how long does travel actually take?
- **Constraint planning** — can the itinerary fit time and budget constraints?
- **Presentation** — turn the structured plan into an interactive Gradio experience.

## Current status: v0.1 scaffold

The repository currently contains a runnable **Hangzhou local baseline**:

1. structured `UserProfile`;
2. local travel-knowledge retrieval;
3. interpretable personalized POI ranking;
4. route estimation behind a `MapProvider` interface;
5. time/budget-aware greedy day planner;
6. Gradio demo;
7. ChatML SFT seed-data generator;
8. DashScope LLM adapter;
9. explicit adapter boundary for Bailian Amap Maps MCP.

The local mode intentionally works **without API keys**, so the pipeline can be tested before cloud integrations are enabled.

## Architecture

```text
User Profile
    |
    +--> Travel RAG --------+
    |                       |
    +--> POI Database ------+--> Candidate POIs
                                    |
                                    v
                         Personalized Ranker
                         (heuristic -> SFT)
                                    |
                                    v
                             Top Candidate POIs
                                    |
                                    v
                         MapProvider / Amap MCP
                                    |
                                    v
                          Constraint Planner
                                    |
                                    v
                             Plan Verification
                                    |
                                    v
                                Gradio
```

The target cloud stack is **Alibaba Cloud Model Studio (Bailian)**:

- Qwen / fine-tuned Qwen for personalized ranking and generation;
- Bailian Knowledge Base for RAG;
- One-Key **Amap Maps MCP** for real POI / route facts.

## Quick start

```bash
git clone https://github.com/Shengxiang-Lin/TripMind.git
cd TripMind

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python evaluation/smoke.py
python app.py
```

Open the Gradio URL shown in the terminal.

The initial demo data covers **Hangzhou**. Try:

- group: `couple`
- interests: `photography`, `food`, `culture`
- pace: `relaxed`
- budget: `1500`

Then compare against `family` or `parents` to see profile-sensitive POI ranking.

## Generate the initial SFT schema

```bash
python training/generate_sft_seed.py
```

This creates `data/sft_seed.jsonl` in ChatML format. It is only a **seed/baseline dataset** generated from interpretable rules. The planned SFT pipeline is:

```text
POI database
    + synthetic profiles
    + candidate sampling
    + rule prior
    + strong teacher LLM
    + consistency filters
    + human spot checks
        -> train / val / test JSONL
        -> Qwen3.5-4B / 9B LoRA SFT
```

The SFT model will learn **profile -> POI preference ranking**, not memorize dynamic route facts.

## Cloud configuration

Copy:

```bash
cp .env.example .env
```

For Qwen on Alibaba Cloud Model Studio:

```bash
export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_MODEL="qwen-plus"
```

`tripmind/integrations/dashscope_llm.py` uses the OpenAI-compatible interface.

### Amap Maps MCP

`tripmind/integrations/amap_mcp.py` is intentionally an adapter stub in v0.1. Bailian's One-Key Amap Maps MCP connection configuration is workspace/region specific; after copying the generated MCP connection configuration from the MCP Marketplace, implement the adapter so every returned route is normalized to:

```python
RouteLeg(
    origin=...,
    destination=...,
    distance_km=...,
    duration_min=...,
    mode=...,
)
```

Until then, `LocalMapProvider` uses Haversine distance + a detour/speed heuristic. These values are **estimates**, not real route facts.

## Core data responsibilities

| Data | Storage / module |
|---|---|
| current user profile | request state |
| POI static metadata | structured JSON now, SQLite later |
| travel guides / planning knowledge | local JSON now, Bailian RAG later |
| preference policy | heuristic baseline now, SFT model later |
| real distance / route duration | Amap MCP |
| dynamic flight / hotel | future external tools |

This separation is intentional: dynamic facts should not be memorized through SFT or stale RAG documents.

## Near-term roadmap

- [ ] Replace local retriever with Bailian Knowledge Base
- [ ] Build 3k–5k teacher-labeled personalized ranking samples
- [ ] Fine-tune Qwen3.5-4B/9B and evaluate NDCG@5 / pairwise accuracy
- [ ] Wire the One-Key Amap Maps MCP adapter
- [ ] Replace greedy planner with OR-Tools time-window optimization
- [ ] Add budget categories (transport / hotel / food / tickets)
- [ ] Add interactive POI replace / delete / lock + local replanning
- [ ] Add route map, POI gallery, and budget visualization
- [ ] Add multi-city POI datasets
- [ ] Add ablation evaluation: Prompt / RAG / SFT / Map / Planner

## Evaluation plan

The final report should separate personalization and feasibility metrics:

**Personalization**
- NDCG@5
- Pairwise preference accuracy
- Profile sensitivity

**Feasibility**
- time-violation rate
- budget-violation rate
- opening-hour violation rate
- total travel distance/time
- feasible-itinerary rate

The key ablation is:

| Method | SFT | RAG | Map | Planner |
|---|---:|---:|---:|---:|
| Prompt-only Qwen |  |  |  |  |
| Qwen + RAG |  | ✓ |  |  |
| Personalized SFT | ✓ |  |  |  |
| SFT + RAG | ✓ | ✓ |  |  |
| SFT + RAG + Map | ✓ | ✓ | ✓ |  |
| **Full TripMind** | ✓ | ✓ | ✓ | ✓ |

## Design principle

> Use SFT to learn **preferences**, RAG to retrieve **travel knowledge**, tools to ground **dynamic facts**, and deterministic planning code to enforce **constraints**.
