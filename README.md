# TripMind

**Personalized · Knowledge-Grounded · Realtime-Aware · Constraint-Aware Travel Planning**

TripMind is an LLM-enhanced travel itinerary recommender for the course project **Large Language Model for Recommendation System**.

The project deliberately separates five responsibilities:

1. **Personalization** — learn which POIs fit a user profile.
2. **Static RAG** — retrieve relatively stable travel knowledge.
3. **Realtime enrichment** — weather, temporary closures, events, promotions, etc.
4. **Route grounding** — obtain road-network distance and duration.
5. **Deterministic planning** — enforce time, opening-hour and budget constraints.

The current repository is a runnable baseline. It still ships with a small Hangzhou POI sample so the full pipeline can run immediately, while the data and provider interfaces are designed for China and international destinations.

---

## 1. Project structure

```text
TripMind/
├── app.py
├── configs/
│   ├── config.yaml                 # tracked defaults, no real secrets
│   └── config.local.yaml.example   # copy to config.local.yaml for keys
├── data/
│   ├── knowledge/
│   ├── poi/
│   ├── raw/                        # downloaded datasets, gitignored
│   └── processed/                  # generated data, gitignored
├── evaluation/
│   ├── config_smoke.py
│   ├── smoke.py
│   └── realtime_mock.py
├── scripts/
│   └── download_data.py
├── training/
│   └── generate_sft_seed.py
└── tripmind/
    ├── config.py
    ├── data.py
    ├── map_provider.py
    ├── models.py
    ├── planner.py
    ├── rag.py
    ├── realtime.py
    ├── recommender.py
    ├── service.py
    └── integrations/
        ├── dashscope_llm.py
        └── openrouteservice.py
```

---

## 2. Installation

```bash
git clone https://github.com/Shengxiang-Lin/TripMind.git
cd TripMind

conda create -n tripmind python=3.11 -y

conda activate tripmind 

pip install -r requirements.txt
```

Python 3.10+ is recommended.

---

## 3. Configuration

TripMind uses:

```text
configs/config.yaml
```

as the default configuration file.

The tracked file contains provider choices, default models, API endpoints, data paths and **empty API-key fields**. Do not commit real API keys.

### Recommended setup

```bash
cp configs/config.local.yaml.example configs/config.local.yaml
```

Then edit:

```yaml
dashscope:
  api_key: "YOUR_DASHSCOPE_API_KEY"

ors:
  api_key: "YOUR_ORS_API_KEY"
```

`configs/config.local.yaml` is ignored by git and overrides `configs/config.yaml`.

Configuration priority is:

```text
configs/config.yaml
        ↓
configs/config.local.yaml
        ↓
.env / system environment variables
```

Environment variables therefore remain supported, for example:

```bash
export ORS_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

You can also choose another YAML file:

```bash
export TRIPMIND_CONFIG=configs/config.yaml
```

### Main defaults

```yaml
providers:
  route: auto
  realtime: auto

dashscope:
  model: qwen-plus
  realtime_model: qwen-plus

ors:
  base_url: https://api.heigit.org/openrouteservice
```

`auto` means:

- use openrouteservice when an ORS key is available, otherwise use local route estimates;
- use Alibaba Cloud realtime web search when a DashScope key is available, otherwise disable realtime enrichment.

Verify the effective non-secret configuration:

```bash
python evaluation/config_smoke.py
```

---

## 4. API keys

### openrouteservice / HeiGIT

Create a HeiGIT account and obtain the free Standard API key from:

```text
https://account.heigit.org/
```

Put it in `configs/config.local.yaml`:

```yaml
ors:
  api_key: "..."
```

TripMind uses the current HeiGIT API base:

```text
https://api.heigit.org/openrouteservice
```

The old `api.openrouteservice.org` host is deprecated.

Current grounded route modes:

```text
walking
driving
cycling
```

openrouteservice does **not** provide public-transit schedules/routing. If `transit` is selected, TripMind currently falls back to an explicitly labeled local estimate.

### Alibaba Cloud Model Studio / DashScope

Configure:

```yaml
dashscope:
  api_key: "..."
  model: "qwen-plus"
  realtime_model: "qwen-plus"
  search_strategy: "turbo"
```

The current realtime layer uses Qwen web search for:

- weather / forecast;
- important local events;
- temporary closures;
- changed opening hours;
- current ticket or promotion signals;
- major travel alerts.

Realtime signals are kept separate from static RAG. Only high-confidence explicit closure/unavailability can become a hard planning constraint.

---

## 5. Download public datasets

TripMind provides a unified download script:

```bash
python scripts/download_data.py --help
```

Downloaded files go to:

```text
data/raw/
```

and are ignored by git.

### ChinaTravel query dataset

```bash
python scripts/download_data.py --source chinatravel
```

Source:

```text
LAMDA-NeSy/ChinaTravel
```

This dataset is useful for:

- realistic travel requests;
- constraints;
- preference tasks;
- SFT/evaluation query construction.

### ChinaTravel-Sandbox

```bash
python scripts/download_data.py --source chinatravel-sandbox
```

Source:

```text
LAMDA-NeSy/ChinaTravel-Sandbox
```

This is the companion structured travel database and is useful for:

- attractions;
- restaurants;
- accommodation;
- inter-city transport;
- benchmark-compatible static travel information.

### Wikivoyage dumps

Chinese + English:

```bash
python scripts/download_data.py --source wikivoyage --languages zh en
```

Other languages can be added directly:

```bash
python scripts/download_data.py --source wikivoyage --languages zh en ja fr
```

The script downloads the Wikimedia `pages-articles` dumps. Wikivoyage is intended to become the main source of relatively stable textual travel knowledge for RAG.

### Download all configured sources

```bash
python scripts/download_data.py --source all --languages zh en
```

> **Important:** dataset download and dataset ingestion are separate stages.  
> The current planner still reads the small processed POI files under `data/poi/` and the local knowledge file under `data/knowledge/`. The raw ChinaTravel/Wikivoyage downloads are inputs for the next preprocessing/index-building stage; they are not silently loaded into the demo as-is.

---

## 6. Run the project

### Offline smoke tests

No API key is required:

```bash
python evaluation/config_smoke.py
python evaluation/smoke.py
python evaluation/realtime_mock.py
```

### Start Gradio

```bash
python app.py
```

The bundled baseline currently has a Hangzhou sample. A useful test profile is:

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

### Force fully local mode

Edit `configs/config.local.yaml`:

```yaml
providers:
  route: local
  realtime: local
```

### Enable ORS + Alibaba realtime automatically

```yaml
providers:
  route: auto
  realtime: auto

ors:
  api_key: "YOUR_ORS_API_KEY"

dashscope:
  api_key: "YOUR_DASHSCOPE_API_KEY"
```

Then simply run:

```bash
python app.py
```

---

## 7. Current pipeline

```text
User Profile
    |
    +--------------------------+
    |                          |
    v                          v
Static Travel RAG       Structured POI DB
    |                          |
    +-------------+------------+
                  |
                  v
        Personalized Ranker
                  |
                  v
           Top Candidate POIs
                  |
                  v
        Realtime Enrichment
   weather / closure / events
       promotion / alerts
                  |
                  v
       Realtime Score Adjustment
                  |
          +-------+-------+
          |               |
          v               v
     Route Provider   Constraints
      ORS / local    budget / hours
          |               |
          +-------+-------+
                  |
                  v
          Constraint Planner
                  |
                  v
          Structured Itinerary
                  |
                  v
               Gradio
```

---

## 8. Data responsibility

TripMind intentionally does not put everything into one vector database.

| Information | Planned source / module |
|---|---|
| User profile | request state |
| POI metadata | structured database |
| Stable destination knowledge | Wikivoyage / RAG |
| China travel benchmark data | ChinaTravel / ChinaTravel-Sandbox |
| Personalized preference policy | heuristic baseline -> Qwen SFT |
| Weather / events / promotions | Alibaba Cloud realtime search |
| Road distance / duration | openrouteservice |
| Scheduling / constraints | deterministic planner |

Dynamic facts such as weather and temporary ticket promotions should **not** be memorized through SFT or stored as long-lived RAG facts.

---

## 9. Personalized SFT

The target fine-tuning task is:

```text
(User Profile, Candidate POI Attributes, Retrieved Knowledge)
                    ->
              POI Preference Ranking
```

Generate the current seed-format dataset:

```bash
python training/generate_sft_seed.py
```

The seed file validates the schema only. The intended next-stage dataset is approximately:

```text
3k-5k train
300-500 validation
300+ held-out test
+ human-ranked evaluation subset
```

Training labels should combine:

- interpretable preference rules;
- a stronger Qwen teacher;
- consistency filtering;
- human spot checks.

Do not fine-tune dynamic route times, weather or current promotions into the model.

---

## 10. Near-term data pipeline

The next implementation stage is:

```text
ChinaTravel / ChinaTravel-Sandbox / Wikivoyage
                    |
                    v
               raw downloads
                    |
                    v
             normalization
                    |
          +---------+---------+
          |                   |
          v                   v
   structured POI DB      RAG chunks
          |                   |
          +---------+---------+
                    |
                    v
        Personalized SFT generation
```

Planned preprocessing scripts will convert the downloaded raw data into:

```text
data/processed/
data/poi/
data/knowledge/
```

---

## 11. Evaluation

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


## Small end-to-end validation on downloaded data

Use the existing `orchestra-o1` environment on the server. For a new environment,
`requirements-runtime.txt` contains the minimal app/data dependencies; the larger
`requirements.txt` is an environment snapshot.

```bash
conda activate orchestra-o1
python -m pip install -r requirements-runtime.txt

# Environment variables have priority over local YAML. Remove an inherited stale
# key to use the key already provided in configs/config.local.yaml.
unset DASHSCOPE_API_KEY

# Offline: prepare 12 Hangzhou POIs, Wikivoyage text chunks and 3 benchmark queries.
python scripts/prepare_sample_data.py --limit 12

# Offline regression checks.
python evaluation/config_smoke.py
python evaluation/smoke.py
python evaluation/realtime_mock.py
python evaluation/regression.py

# Small real-API run: two profiles plus a Gradio HTTP prediction.
# This makes paid Qwen requests and consumes a small ORS quota.
python evaluation/live_smoke.py --ui
```

The sample importer writes only `data/processed/sample/`. It uses the Chinese
Sandbox attraction table, the Chinese Wikivoyage Hangzhou article and the
ChinaTravel easy queries; it does not overwrite raw files or the bundled demo.
It selects POIs within 25 km of the Hangzhou center and retains source/license
metadata. Tags, intensity and neutral popularity are baseline derivations, not
human labels; visit duration is the midpoint of source min/max hours converted
to minutes. The textual cleaner is a small sample extractor, not a complete
MediaWiki parser. ChinaTravel constraint programs are kept as data, never executed.

To start the app with the generated data and your local API configuration:

```bash
TRIPMIND_CONFIG=data/processed/sample/config.yaml python app.py
```

Results are saved under `data/processed/live_smoke/`: `report.json`, per-case JSON
and Markdown itineraries, and `ui_result.md` when `--ui` is used. A live check fails
if realtime fails, a selected road leg uses local estimates, or scheduling/budget
checks fail. Repeated UI requests exercise the in-memory realtime/matrix caches.
These checks cover the baseline pipeline, not every constraint of the full
ChinaTravel benchmark or a trained personalized model.

Server compatibility notes:

- An inherited `DASHSCOPE_API_KEY` can shadow a valid YAML key. Conflicts now emit
  a warning containing the variable name only; no key is logged.
- In this server's OpenSSL/network combination, default ORS TLS negotiation timed
  out. Setting `ors.tls_maximum_version: TLSv1_2` in local YAML restored verified
  HTTPS. This setting is optional and scoped to ORS; certificate verification stays on.
- ORS GeoJSON directions require `Accept: application/geo+json`; the matrix uses JSON.
- The planner requests one candidate matrix, preserves unreachable pairs, and
  accepts a source closing time of `24:00`.
- Realtime errors are shown as planning notices. Failed/partial lookups are not
  cached as successful empty results. Gradio analytics is disabled for local runs.
