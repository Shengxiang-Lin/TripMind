# TripMind

**Personalized · Knowledge-Grounded · Realtime-Aware · Constraint-Aware Travel Planning**

TripMind is an LLM-enhanced travel itinerary recommender for the course project **Large Language Model for Recommendation System**.

The core design separates four different kinds of information instead of asking one LLM to invent the whole trip:

1. **Preference learning** — which POIs fit this user?
2. **Static travel knowledge** — what is useful to know about destinations and itinerary patterns?
3. **Realtime context** — what has changed today or around the travel date?
4. **Route grounding and planning** — can the selected POIs actually fit the available time and budget?

## Current status: v0.2

The repository now contains a runnable offline baseline plus optional cloud integrations.

### Offline path

Works without any API key:

- structured `UserProfile`;
- multi-city/country-ready POI schema;
- local travel-knowledge retrieval;
- interpretable personalized POI ranking;
- Haversine route fallback;
- explicit time/budget-aware greedy planner;
- Gradio demo;
- SFT seed-data generator.

### Optional online enrichment

When keys are configured:

- **Alibaba Cloud Model Studio / Qwen web search**
  - weather;
  - local events;
  - temporary notices;
  - POI closure / opening-hour changes;
  - ticket or promotion signals;
- **openrouteservice**
  - real road-network distance;
  - real road-network duration;
  - route geometry;
  - time-distance matrix.

The online modules are deliberately optional. A failed realtime search or route request must not make the whole demo unusable.

---

## Architecture

```text
User Profile
    |
    +-----------------------+
    |                       |
    v                       v
Static Travel RAG       Structured POI DB
    |                       |
    +-----------+-----------+
                |
                v
        Personalized Ranker
                |
                v
          Top Candidate POIs
                |
                +-----------------------+
                |                       |
                v                       v
       Realtime Enrichment       Route Provider
       Alibaba Web Search        ORS / local fallback
                |                       |
                +-----------+-----------+
                            |
                            v
                 Constraint Planner
                 time / budget / hours
                            |
                            v
                    Structured Itinerary
                            |
                            v
                         Gradio
```

The intended final model stack is:

```text
SFT -> learn preference policy
RAG -> retrieve relatively stable travel knowledge
Realtime tools -> obtain changing external context
Routing API -> obtain road-network facts
Planner -> enforce constraints deterministically
LLM -> explain / verbalize the final structured plan
```

---

## Why realtime context is separate from RAG

Do **not** put rapidly changing facts into the long-term vector knowledge base.

### Static RAG

Good examples:

- attraction characteristics;
- cultural background;
- suitable visitor groups;
- classic itinerary combinations;
- neighborhood travel tips;
- typical visit style;
- long-term planning knowledge.

### Realtime layer

Good examples:

- weather forecast;
- temporary closure;
- changed opening hours;
- current local event;
- time-limited ticket promotion;
- travel alert.

`tripmind/realtime.py` models these as `RealtimeSignal`.

Realtime signals have:

```text
kind
subject
summary
effect
confidence
source
hard_constraint
retrieved_at
```

Only a high-confidence explicit closure/unavailability is allowed to become a hard planning constraint. Promotions are always soft context.

---

## Routing design

TripMind exposes a provider abstraction:

```python
MapProvider.route(...)
MapProvider.matrix(...)
```

### Local mode

`LocalMapProvider` uses:

- Haversine distance;
- a detour factor;
- simple average-speed assumptions.

It is only a development fallback and is labeled as **estimated** in the UI.

### openrouteservice

`OpenRouteServiceProvider` uses the ORS v2 API.

Currently supported grounded modes:

- `walking`
- `driving`
- `cycling`

ORS does not provide public-transit routing. If `transit` is selected while ORS is active, `FallbackMapProvider` transparently uses the local estimated transit baseline rather than falsely labeling it as a real ORS transit route.

Later, a dedicated transit provider can be added without changing the planner interface.

---

## Quick start

```bash
git clone https://github.com/Shengxiang-Lin/TripMind.git
cd TripMind

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Run the offline tests:

```bash
python evaluation/smoke.py
python evaluation/realtime_mock.py
```

Start Gradio:

```bash
python app.py
```

The current bundled POI data still focuses on Hangzhou, but the schema and loader are now country/city ready.

Example:

```text
Country: China
City: 杭州
Days: 2
Budget: 1500 CNY
Group: couple
Interests: photography, food, culture
Pace: relaxed
Transport: walking
```

---

## Environment configuration

Copy:

```bash
cp .env.example .env
```

### 1. Fully local mode

```bash
TRIPMIND_ROUTE_PROVIDER=local
TRIPMIND_REALTIME_PROVIDER=local
```

No API key is required.

### 2. openrouteservice grounding

```bash
TRIPMIND_ROUTE_PROVIDER=auto
ORS_API_KEY=...
```

With `auto`, TripMind uses ORS when a key exists, otherwise it falls back to local estimates.

### 3. Alibaba Cloud realtime search

```bash
TRIPMIND_REALTIME_PROVIDER=auto

DASHSCOPE_API_KEY=...
DASHSCOPE_MODEL=qwen-plus
DASHSCOPE_REALTIME_MODEL=qwen-plus
DASHSCOPE_SEARCH_STRATEGY=turbo
```

When enabled, TripMind performs two small realtime searches:

1. **city-level**
   - weather;
   - events;
   - major travel alerts;

2. **top-POI-level**
   - temporary closure;
   - changed opening hours;
   - ticket promotions;
   - time-sensitive POI events.

To avoid repeated search while a user edits the same itinerary:

```bash
TRIPMIND_REALTIME_TTL_SECONDS=3600
TRIPMIND_REALTIME_POI_LIMIT=6
```

---

## Global-ready POI schema

`POI` now includes:

```text
poi_id
country
city
name
category
longitude
latitude
ticket_price
currency
recommended_duration
opening_time
closing_time
popularity
tags
suitable_groups
intensity
indoor
description
source
image_url
```

The loader now scans all JSON files under:

```text
data/poi/**/*.json
```

Therefore future datasets can be organized as:

```text
data/poi/
├── china/
│   ├── hangzhou.json
│   ├── beijing.json
│   └── shanghai.json
├── japan/
│   ├── tokyo.json
│   └── kyoto.json
└── france/
    └── paris.json
```

No planner code needs to change when new countries are added.

---

## Realtime ranking adjustment

Realtime information does not replace the personalized ranker.

Instead:

```text
base preference score
        +
small realtime adjustment
        =
final ranking score
```

Examples:

- explicit high-confidence closure -> exclude the POI from planning;
- current promotion -> small positive adjustment;
- rain / bad outdoor weather -> small indoor bonus and outdoor penalty;
- uncertain information -> informational only.

This prevents volatile web-search content from overpowering stable user preferences.

---

## Personalized SFT

The intended fine-tuning task remains:

```text
(User Profile, Candidate POI Attributes, Retrieved Knowledge)
                    ->
              POI Preference Ranking
```

Do **not** fine-tune dynamic route facts or current ticket promotions into the model.

Generate the current seed dataset:

```bash
python training/generate_sft_seed.py
```

The next data milestone is:

```text
3k-5k training examples
300-500 validation examples
300+ held-out synthetic test examples
human-ranked evaluation subset
```

Candidate training labels should combine:

- interpretable preference rules;
- a stronger Qwen teacher;
- consistency filters;
- human spot checks.

---

## Near-term roadmap

### Data

- [ ] Replace hand-written Hangzhou POIs with open/global POI sources
- [ ] Add ChinaTravel structured travel data
- [ ] Add Wikivoyage ingestion pipeline
- [ ] Add several Chinese and international cities

### Recommendation

- [ ] Build 3k-5k personalized ranking examples
- [ ] Fine-tune a small Qwen model
- [ ] Add NDCG@5 / pairwise preference evaluation

### Planning

- [x] Map provider abstraction
- [x] ORS route and matrix adapter
- [x] transparent local fallback
- [ ] Replace greedy planner with OR-Tools time-window optimization
- [ ] Add hotel / food / transport budget categories
- [ ] Add local replanning after POI replace/delete/lock

### Realtime

- [x] Realtime provider abstraction
- [x] Alibaba Cloud web-search adapter
- [x] TTL cache
- [x] confidence / hard-vs-soft signal distinction
- [ ] add official-source-aware verification when available
- [ ] add flight / hotel / ticket provider tools

### UI

- [ ] route map
- [ ] POI image gallery
- [ ] budget visualization
- [ ] editable POI timeline

---

## Evaluation plan

### Personalization

- NDCG@5
- pairwise preference accuracy
- profile sensitivity
- cross-city / cross-country generalization

### Feasibility

- time-violation rate
- budget-violation rate
- opening-hour violation rate
- total travel distance/time
- feasible-itinerary rate

### Grounding

- estimated vs grounded route coverage
- realtime constraint hit rate
- stale-information error analysis

### Ablation

| Method | SFT | RAG | Realtime | Routing | Planner |
|---|---:|---:|---:|---:|---:|
| Heuristic |  |  |  |  | ✓ |
| Qwen + RAG |  | ✓ |  |  | ✓ |
| Personalized SFT | ✓ |  |  |  | ✓ |
| SFT + RAG | ✓ | ✓ |  |  | ✓ |
| + Realtime | ✓ | ✓ | ✓ |  | ✓ |
| **Full TripMind** | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Design principle

> Use SFT to learn **preferences**, RAG to retrieve **stable travel knowledge**, realtime tools to observe **changing external context**, routing services to ground **distance/time**, and deterministic planning code to enforce **constraints**.
