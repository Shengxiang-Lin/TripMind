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

## Full offline data pipeline

The full pipeline replaces the small Hangzhou-only extraction with a versioned
bundle. Use the server's `orchestra-o1` environment:

```bash
conda activate orchestra-o1
python -m pip install -r requirements-data.txt
python scripts/build_data.py
```

No model, embedding or map API is called during ingestion. The builder reads all
14 bilingual Sandbox Parquet tables, every ChinaTravel CSV (including Phase 2),
and both downloaded Wikivoyage article dumps. It never modifies the raw inputs.

The active pointer is `data/processed/production/current.json`. Each immutable
build under `data/processed/production/runs/<build_id>/` contains:

- `poi/*.json`: canonical Chinese attraction records for all 10 source cities,
  with English aliases and source lineage. Chinese/English copies are one entity.
- `catalog.sqlite`: all seven Sandbox reference tables in both languages. Restaurant,
  accommodation and transport source payloads are retained with canonical destination
  metadata; they are not silently scheduled as sightseeing attractions.
- `cities.json`: supported destination names and aliases.
- `knowledge/chunks.jsonl` and `knowledge/documents.sqlite`: cleaned namespace-zero,
  non-redirect articles with source URLs, page/revision IDs, language and license.
- `knowledge/tfidf.npz`, `vectorizer.joblib`, `index.json`: a local sparse TF-IDF
  cosine index over Latin words and simplified-Chinese bigrams. This is lexical
  vector retrieval, not a neural semantic embedding model.
- `queries/{all,train,validation,test}.jsonl`: normalized requests and constraint
  programs as data. They contain no fabricated ranking answers or SFT labels.
- `quarantine_pois.jsonl`, `quarantine_queries.json`, `poi_duplicates.json`:
  rejected rows, reasons and merges.
- `quality_report.json` and `manifest.json`: counts, validation results, versions,
  settings, source SHA256 digests and artifact checksums.

Exact duplicates are removed. Same-name attraction records within 100 metres
are merged conservatively, retaining source IDs and English aliases. Missing or
invalid cost, coordinate and duration values are rejected rather than filled with
invented facts. Source minimum/maximum visit hours are converted to midpoint
minutes. Popularity and preference tags remain documented heuristic defaults.
For overnight venues, only the evening-to-midnight interval is exposed to the
09:00-20:00 planner, with original times and this normalization recorded. Coordinate
reference systems are preserved as unspecified where the dataset does not declare one.

Wiki parsing uses `mwparserfromhell`, removes navigation/media/reference noise,
preserves listing names/descriptions, and creates bounded overlapping text chunks.
Supported destination aliases map English and traditional-Chinese titles to the
canonical city. Other article titles remain separate destinations; country values
are not guessed. Retrieval filters city/country before ranking and returns source
citations. Region-to-city inheritance is not inferred in this version.

Query splitting groups preference variants by their base UID, Phase 2 variants by
source seed, and identical requests even when their UIDs differ. Official
`human1000` and competition-test families stay entirely in test. Other groups use
a seed-stable 80/10/10 hash split. Incomplete profile metadata, such as in
`human1000`, is retained and flagged. DSL programs are never executed by this
pipeline. Full benchmark constraint evaluation and model training remain separate.

Only a validated build is published, by atomic replacement of `current.json`.
Build IDs depend on raw content, transformation code, settings and package versions.
Repeating the same build reuses checksum-verified artifacts. Failed builds leave
the previously active pointer unchanged. Older complete builds remain available.
Use `--no-publish` to build without changing the active pointer, or `--output` to
create a separate bundle root. `--cities` supports bounded city subsets; the default
processes all source cities. The chunk size, overlap and split seed are configurable.

Default application configuration uses the active processed bundle. To select a
bundle root, set `TRIPMIND_DATA_BUNDLE`; to run the original bundled demo instead,
set `TRIPMIND_USE_PROCESSED=false`. An explicitly generated sample configuration
keeps `data.use_processed=false`. Processed bundles fail fast on missing manifests,
unsupported schemas or changed artifact checksums instead of silently reverting to
the demo. Only locally generated, verified vectorizer artifacts should be loaded.

```bash
python evaluation/data_pipeline.py       # small offline fixtures
python evaluation/production_data.py     # full bundle + all-city offline plans
unset DASHSCOPE_API_KEY                  # only if an inherited key shadows local YAML
python evaluation/live_smoke.py --production --ui --output data/processed/production_live
python app.py                           # now reads the processed bundle by default
```

This stage prepares reliable data and the baseline retrieval/planning path. It does
not train a personalized ranker, invent preference labels, or claim the complete
ChinaTravel benchmark has been solved.
