# Biomining Federal Grant Analysis Pipeline

**Output (PDF report)**: `visualization/biomining_highlights.pdf`
**Output (interactive site)**: `biomining-funding-viz/` (separate folder — deployed at the GapMap URL)

**Goal**: Identify and characterize federal funding for **biomining research** across all agencies (NSF, DOE, DOI, EPA, USDA, DOD, etc.) from 2016–2025. Show how much funding goes to biological vs. non-biological approaches, across research stages, mining types, materials, and orientation (public-good vs. industry-facing).

**Time period**: FY2016–FY2025 federal grants, sourced from **USASpending.gov bulk CSVs** + **NSF Awards API JSONs** (one bulk CSV per fiscal year, not per-keyword API calls).

---

## What counts as "biomining"?

**Biomining = extracting, recovering, processing, characterizing, sensing, or remediating metals/minerals using biological approaches** (bioleaching, biosorption, bioremediation, bioprecipitation, biosensing, phytomining) **or** conventional approaches in a mining context (hydrometallurgy, mine waste treatment, mineral exploration, critical-mineral supply chains).

**KEEP if**: The grant's *primary focus* is on a mining-relevant activity, as judged by the LLM against a detailed set of rules (hard filters for edge-case query terms like "biogeochemical" / "uranium" / "hyper-accumulation", BIOSENSING inclusion rule, TITLE-ONLY rule, ALL-CAPS-abstract rule, LOW-CONFIDENCE tiebreaker).

**REMOVE if**: The grant mentions a mining-adjacent term incidentally but isn't primarily about mining (data mining / genome mining, biomedical, materials science, pure oceanography, extraterrestrial resource utilization, battery manufacturing downstream of mining, etc.).

---

## The Classification System

Every kept grant is classified across **6 dimensions**:

| Axis | What it means | Categories |
|---|---|---|
| **biological** + **bio_subcategory** | Is the grant's primary research mechanism biological? If so, which subtype? | bool + one of [bioleaching / bioseparation_biosorption / bioremediation / bioprecipitation_mineral_formation / biosensing] |
| **mining_type** | What mining activity is the grant doing? | exploration / extraction / downstream_processing / remediation / infrastructure / collaborative_tools / general |
| **materials** (array) | Which metals/materials are targeted? | base_metals / precious_metals / lanthanides_REE / battery_metals / critical_transition_metals / platinum_group_metals / actinides / metalloids_specialty / polymetallic_general / coal_ash_secondary / secondary_feedstocks |
| **stage** | How mature is the work? | fundamental / early_technology_development / applied_translational / deployment |
| **orientation** | Who is the grant for? | public_good / industry_facing |
| **research_approach** | How is it structured? | collaborative_interdisciplinary / single_focus |

Plus two **post-classification keyword flags** (from `add_keyword_flags_multiyear.py`):
- **industry_framing** — does the abstract use TEA / LCA / commercial-viability / market-analysis language? (17 keywords)
- **open_access_sharing** — does it reference open-access / shared-facility / data-sharing commitments? (29 keywords)

Formula grants (block allocations like BIL AML) are classified on only `biological` + `bio_subcategory` — they're flagged via `is_formula_grant` and skip full classification to save LLM cost.

---

## Why this pipeline (vs. the old one)

The **old** biomining pipeline used the USASpending `spending_by_award` API with keyword searches, which returns *award-level* rollups. That conflated two different things:

- The **award's lifetime total obligation** (what the API returns)
- **Obligations that actually happened in 2016–2025** (what we want)

A grant whose project started in 2013 and had a single $0 administrative modification in 2016 was returned by the API and counted as its full lifetime amount (e.g., $1.65M) against 2013, even though *zero dollars* were obligated during our window.

The **new** pipeline switches to **USASpending bulk CSVs per fiscal year** + **NSF JSONs per year**. Each CSV row is a *transaction* (an obligation action) with its own `action_date` and `federal_action_obligation`. We group transactions by award_id *within each year* and sum the per-year obligation. This means:

1. **Attribution year = the year money actually flowed**, not the project's start year.
2. A grant contributing $0 in our window is correctly excluded (it'll sum to $0).
3. Multi-year grants appear in multiple years with the specific year's obligation.

The filter and classifier logic is preserved from the old pipeline — only the *data source and aggregation* changed.

---

## Pipeline (play-by-play)

### Step 0 — Download raw data

One-time setup. Place raw per-year data in `data/<year>/`:

**NSF** (JSON files, one per award):
Use the [NSF Awards API](https://www.nsf.gov/developer/). Pull every award whose start date falls in fiscal year FY. Save each JSON as `data/<year>/NSF<year>/<award_id>.json`.

**USASpending.gov** (per-year bulk CSV):
1. Go to https://www.usaspending.gov/download_center/award_data_archive
2. Select **Fiscal Year XXXX**, **Award Type: Grants (Prime)**
3. Download the CSV (1–3 GB per year)
4. Rename to `USASpending<year>.csv`
5. Save as `data/<year>/USASpending<year>.csv`

Expected layout (currently in place for 2016–2025):

```
data/
├── 2016/NSF2016/*.json      + USASpending2016.csv
├── 2017/NSF2017/*.json      + USASpending2017.csv
...
├── 2025/NSF2025/*.json      + USASpending2025.csv
└── federal_gapmap_master_future_updated_v2.csv   (hand-curated forward-looking data, Section 3 of PDF)
```

### Step 1 — Merge raw data into one CSV

**Script**: `scripts/grant_classifier/step1_merge_master_multiyear.py`

```bash
python scripts/grant_classifier/step1_merge_master_multiyear.py
```

**What it does**:
- For each year 2016–2025:
  - Loads NSF JSONs and USASpending CSV
  - Splits USASpending into NSF-agency rows and non-NSF rows
  - Groups NSF USASpending transactions by `award_id` (sum `federal_action_obligation`) and merges with NSF JSON metadata
  - Groups non-NSF USASpending transactions by `award_id`
  - Tags every row with `year = <fiscal year>` (from the transaction date, not the project's `startDate`)
- Concatenates all years → `output/merged_all_years.csv`

**Runtime**: ~5–15 min (pandas + I/O heavy).
**Output**: `scripts/grant_classifier/output/merged_all_years.csv` (~5 GB, depends on years included)

### Step 2 — Mining-only filter + abstract-length + formula flag

**Script**: `scripts/grant_classifier/step2_mining_filter_multiyear.py`

```bash
python scripts/grant_classifier/step2_mining_filter_multiyear.py
```

**What it does** — **filter rules identical to the old `federal_filter_mining_only_with_group_exclusion.py`**:
- Mining-only gating (Cat A/B + process terms)
- Exclude non-mining "data mining / genome mining / etc."
- Exclude physiology / biomedical (threshold-based)
- Exclude plastics (threshold-based)
- Exclude archaeology / paleo / heritage (threshold-based)
- Standalone auto-keep for `biomining, bioleaching, biooxidation, biohydrometallurgy, biometallurgy, phytomining` (and variants)
- CAT_B_ALLOWED_PROCESS_TERMS now includes hyperaccumulation family

Plus two additions from the climate_biotech pattern:
- **Flag (not drop)** `is_formula_grant` when `award_type == "FORMULA GRANT (A)"` — formula grants pass through to the LLM with a reduced Stage-2-Formula pass (biological + bio_subcategory only) so the Sankey's Formula Grants panel stays populated
- **Drop** rows with `abstract` < 92 characters (below LLM-classifiable context)

**Outputs** (in `scripts/grant_classifier/output/`):
- `mining_filtered_all_years.csv` — LLM-ready pool (kept + formula, with sufficient abstracts)
- `mining_excluded_all_years.csv` — rejected by mining filter
- `mining_formula_grants_all_years.csv` — formula subset (QA reference; also present in kept)
- `mining_insufficient_abstract_all_years.csv` — < 92-char abstracts

**Runtime**: ~2–5 min.

### Step 3 — Two-stage LLM classification

**Script**: `scripts/grant_classifier/step3_mining_two_stage_classifier_multiyear.py`

```bash
python scripts/grant_classifier/step3_mining_two_stage_classifier_multiyear.py
```

**Three-path architecture**:

| Stage | Model | Batch | Runs on | Output |
|---|---|---|---|---|
| **Stage 1** | Claude Haiku 4.5 | 20 | All step2-kept grants | `keep`, `confidence`, `remove_reason` |
| **Stage 2-Formula** | Claude Haiku 4.5 | 20 | Kept formula grants only | `biological`, `bio_subcategory`, `confidence` |
| **Stage 2-Full** | Claude Sonnet 4 | 10 | Kept non-formula grants | all 6 axes |

**Prompts**: preserved verbatim from the old biomining classifier — ~100 calibration examples, all axis definitions, all hard filters. The only adaptation is that hard filters keyed on `query_used` were rephrased to trigger on abstract text (the bulk pipeline has no `query_used` column). Orientation uses an explicit decision tree (SBIR/STTR → industry_facing, University/College → public_good, etc.) ported from climate_biotech step3 so no post-classification manual correction is needed.

**Test mode**: defaults to `TEST_MODE = True` which runs on the 70-grant holdout (`VALIDATION_GRANT_IDS` in the script, identical to the old `score_holdout.py` holdout). Set `TEST_MODE = False` for a full-dataset run.

**Resume**: `RESUME = True` (default) — checkpoints every 10 batches. Re-running after a crash resumes where it left off.

> ⚠️ **Clear the output files when switching between TEST_MODE and production, OR between two test runs you want isolated.** Stage 1/2 checkpoints (`stage1_mining_relevance_all_years.csv`, `stage2_formula_all_years.csv`, `stage2_full_all_years.csv`, `stage2_short_abstract_all_years.csv`) persist across runs because of `RESUME`. If you run in `TEST_MODE = True` (only the 70 holdout grants get classified), those 70 rows land in the stage1/stage2 CSVs. If you then flip `TEST_MODE = False` with `RESUME = True`, the classifier sees those 70 keys as "already done" and **skips them from the full-dataset run** — quietly inheriting the test output and leaving the rest of the ~2,800-grant pool partially classified or misaligned. Same risk going the other direction, or between two independent test runs.
>
> Before switching modes, wipe the checkpoints:
> ```bash
> rm scripts/grant_classifier/output/stage1_mining_relevance_all_years.csv \
>    scripts/grant_classifier/output/stage1_excluded_all_years.csv \
>    scripts/grant_classifier/output/stage1_review_all_years.csv \
>    scripts/grant_classifier/output/stage2_formula_all_years.csv \
>    scripts/grant_classifier/output/stage2_full_all_years.csv \
>    scripts/grant_classifier/output/stage2_short_abstract_all_years.csv \
>    scripts/grant_classifier/output/mining_llm_classified_all_years.csv \
>    scripts/grant_classifier/output/two_stage_classification_log_all_years.json
> ```
> Leave step1/step2 outputs (`merged_all_years.csv`, `mining_filtered_all_years.csv`) alone — those are deterministic filters and don't need re-running.

**Outputs**:
- `stage1_mining_relevance_all_years.csv` — Stage 1 decisions
- `stage1_excluded_all_years.csv` — Stage 1 REMOVEs
- `stage1_review_all_years.csv` — Stage 1 low-confidence
- `stage2_formula_all_years.csv` — Stage 2-Formula results
- `stage2_full_all_years.csv` — Stage 2-Full results
- `mining_llm_classified_all_years.csv` — **merged final** (all three branches unified)
- `two_stage_classification_log_all_years.json` — raw LLM responses

**Runtime (full dataset)**: ~4–8 hours.
**Cost estimate** (for ~10,000 grants full run):
- Stage 1: ~$0.001/grant × 10K = ~$10
- Stage 2-Formula: ~$0.001/formula-grant × ~200 = ~$0.20
- Stage 2-Full: ~$0.01/grant × ~1K kept non-formula = ~$10
- Total: **~$20–30** (varies with actual kept count)

### Step 4 — Keyword flags (post-classification)

**Script**: `scripts/add_keyword_flags_multiyear.py`

```bash
python scripts/add_keyword_flags_multiyear.py
```

**Section 1A filter** (LLM-kept + ≥ 92 char abstract + non-formula) then adds:
- `industry_framing` — 17 keywords (TEA, LCA, commercial viability, etc.)
- `open_access_sharing` — 29 keywords (open access, shared facility, data repository, etc.)

Interdisciplinary is *not* included here — the LLM's `research_approach` axis replaces the keyword-based interdisciplinary flag.

**Output**: `mining_llm_classified_with_keyword_flags_all_years.csv` — **FINAL DATASET** used by the PDF generator.

**Runtime**: ~1–2 min.

### Step 5 — Accuracy validation (optional)

**Script**: `scripts/test_llm_classifier_accuracy.py`

```bash
python scripts/test_llm_classifier_accuracy.py
```

Scores the LLM classification against the **70-grant manually-labeled holdout** embedded in the script. Reports per-axis accuracy (keep/remove, mining_type, bio_subcategory, stage, orientation, research_approach, materials as set-overlap). Threshold for "production-ready": ≥ 92% keep/remove.

**Latest validation (2026-04-17):** 95.0% overall weighted accuracy — materials, orientation, and research_approach at 100%. See [`VALIDATION_RESULTS.md`](VALIDATION_RESULTS.md) for the full per-axis breakdown, remaining known errors, and changes applied to reach this score.

### Step 6 — PDF report generation

**Script**: `scripts/biomining_highlights.py`

```bash
python scripts/biomining_highlights.py
```

Produces `visualization/biomining_highlights.pdf` — a ~12-page meeting-ready document covering:

- Cover page + portfolio donut
- Bio vs Non-Bio timeseries (all grants, deployment-excluded)
- Stage relative abundance by year + stage definitions sidebar
- Mining type relative abundance by year + type definitions sidebar
- Bio subcategory + materials vertical stacked bars + definitions
- Bio % by stage + Average grant size by stage
- Section 2: Formula Grants breakdown
- Section 3: Forward-looking federal funding landscape (funnel, stage breakdown, early-stage table) — reads `data/federal_gapmap_master_future_updated_v2.csv`

Depends on `scripts/biomining_shared.py` (library module, never run directly — auto-imported).

**Runtime**: ~2–5 min.

---

## Project Structure

```
biomining_federal_grant_funding/
├── data/
│   ├── 2016/NSF2016/*.json + USASpending2016.csv
│   ├── 2017/NSF2017/*.json + USASpending2017.csv
│   ├── 2018/NSF2018/*.json + USASpending2018.csv
│   ├── 2019/NSF2019/*.json + USASpending2019.csv
│   ├── 2020/NSF2020/*.json + USASpending2020.csv
│   ├── 2021/NSF2021/*.json + USASpending2021.csv
│   ├── 2022/NSF2022/*.json + USASpending2022.csv
│   ├── 2023/NSF2023/*.json + USASpending2023.csv
│   ├── 2024/NSF2024/*.json + USASpending2024.csv
│   ├── 2025/NSF2025/*.json + USASpending2025.csv
│   └── federal_gapmap_master_future_updated_v2.csv   # forward-looking, Section 3 of PDF
│
├── scripts/
│   ├── add_keyword_flags_multiyear.py       # step 4 (post-classification flags)
│   ├── biomining_highlights.py              # PDF report generator
│   ├── biomining_shared.py                  # shared library (never run directly)
│   ├── test_llm_classifier_accuracy.py      # score vs 70-grant holdout
│   └── grant_classifier/
│       ├── output/                                                        # pipeline outputs
│       │   ├── merged_all_years.csv
│       │   ├── mining_filtered_all_years.csv
│       │   ├── mining_excluded_all_years.csv
│       │   ├── mining_formula_grants_all_years.csv
│       │   ├── mining_insufficient_abstract_all_years.csv
│       │   ├── stage1_mining_relevance_all_years.csv
│       │   ├── stage1_excluded_all_years.csv
│       │   ├── stage1_review_all_years.csv
│       │   ├── stage2_formula_all_years.csv
│       │   ├── stage2_full_all_years.csv
│       │   ├── mining_llm_classified_all_years.csv
│       │   ├── mining_llm_classified_with_keyword_flags_all_years.csv    # FINAL
│       │   └── two_stage_classification_log_all_years.json
│       ├── step1_merge_master_multiyear.py
│       ├── step2_mining_filter_multiyear.py
│       └── step3_mining_two_stage_classifier_multiyear.py
│
└── visualization/
    └── biomining_highlights.pdf                                           # PDF report
```

---

## Complete Workflow

```bash
cd ~/Desktop/biomining_federal_grant_funding

# 1. Merge raw bulk data (slow, only rerun when raw data changes)
python scripts/grant_classifier/step1_merge_master_multiyear.py

# 2. Apply mining-only filter + formula flag + abstract-length filter
python scripts/grant_classifier/step2_mining_filter_multiyear.py

# 3. LLM classify — TEST_MODE=True by default (runs on 70-grant holdout)
#    Set TEST_MODE=False in the script for a full-dataset run
python scripts/grant_classifier/step3_mining_two_stage_classifier_multiyear.py

# 4. Add keyword flags (industry_framing, open_access_sharing)
python scripts/add_keyword_flags_multiyear.py

# 5. (optional) Validate accuracy against 70-grant manual holdout
python scripts/test_llm_classifier_accuracy.py

# 6. Generate the PDF report
python scripts/biomining_highlights.py
open visualization/biomining_highlights.pdf
```

**Total end-to-end time** (full run, 2016–2025): ~6–12 hours, mostly the LLM classification. Use `RESUME=True` for crash safety.

---

## Requirements

```bash
# Python 3.9+
python3 --version

# Required packages
pip install pandas numpy tqdm anthropic matplotlib scipy
```

**API key** — needed for step3 only:

```bash
# Option A: copy the template and fill it in
cp .env.example .env
# edit .env, paste your key after ANTHROPIC_API_KEY=

# Option B: environment variable
export ANTHROPIC_API_KEY=sk-ant-...
```

Get an Anthropic key at https://console.anthropic.com/settings/keys.

---

## Setting up from GitHub (fresh clone)

```bash
# 1. Clone
git clone <repo-url> biomining_federal_grant_funding
cd biomining_federal_grant_funding

# 2. Install dependencies
pip install pandas numpy tqdm anthropic matplotlib scipy

# 3. Set up your API key (never commit this file — .env is gitignored)
cp .env.example .env
# edit .env: paste your Anthropic key

# 4. Download raw data into data/<year>/ — see "Step 0 — Download raw data" above.
#    The raw NSF JSONs and USASpending CSVs are NOT tracked by git
#    (they're large and freely downloadable).
#    You also need to obtain the forward-looking CSV for Section 3 of the PDF:
#      data/federal_gapmap_master_future_updated_v2.csv
#    This is hand-curated from announced federal critical-minerals programs.

# 5. Run the pipeline (see "Complete Workflow" below).
```

**What's in the repo** (tracked by git):
- All Python scripts in `scripts/`
- `README.md`, `.env.example`, `.gitignore`

**What's NOT in the repo** (gitignored):
- `.env` (real API key)
- `data/**/*.json`, `data/**/*.csv` (raw NSF + USASpending data — users download their own)
- `scripts/grant_classifier/output/` (pipeline outputs — regenerated locally)
- `__pycache__/`, `.DS_Store`, `*.pyc`

**Note for forks / transitions:** two files contain author-specific fallback paths that look in `~/Desktop/Homeworld/Projects/GapMap/` for legacy data. These are harmless for other users (they return "not found" and the script either errors clearly or skips gracefully) but are specific to the author's machine:
- `scripts/biomining_shared.py` — FEDERAL_CSV fallback
- `scripts/biomining_highlights.py` — biomining_shared import path fallback

If you fork the repo, you can remove those fallbacks; the primary `__file__`-relative paths will still resolve correctly.

---

## Troubleshooting

### "Input file not found: ..."
Each step depends on the prior step's output. Check that the expected CSV exists in `scripts/grant_classifier/output/`.

### Step1 takes forever / runs out of memory
Raw data is large (~16 GB across 10 years). On low-memory machines, process years one at a time by editing the `YEARS = [2016, ..., 2025]` list.

### Step3 TEST_MODE confusion
`TEST_MODE = True` (default) runs only on the 70-grant holdout. Set it to `False` for a real full-dataset run. Check cost estimates before running — a full run is ~$20–30.

### Step3 crashes mid-run
Re-run with `RESUME = True` (default). Checkpoints every 10 batches mean you'll pick up where you left off. Don't delete `stage1_mining_relevance_all_years.csv` / `stage2_*_all_years.csv` between attempts.

### "Column X not found" when running PDF generator
`biomining_shared.py` has a schema adapter that maps new-pipeline columns (`unique_key`, `s1_keep`, `award_amount`, `start_date`) to old-pipeline names (`unique_award_key`, `llm_keep`, `amt`, `startDate`). If a chart still breaks, confirm `mining_llm_classified_all_years.csv` has all the expected `llm_*` columns from step3.

### PDF Section 3 missing
Requires `data/federal_gapmap_master_future_updated_v2.csv`. If absent, `biomining_shared.FEDERAL_CSV` falls back to `~/Desktop/Homeworld/Projects/GapMap/federal_gapmap_master_future_updated_v2.csv` for backwards compatibility. Copy the file into the project's `data/` folder to decouple from the legacy location.

### Out-of-range grants in the final output
See `analyze years` below. Some grants have project `startDate` before 2016 or after 2025 because their *first action* in that window caused them to match. The new pipeline attributes by transaction year (not project start), so this should be much smaller than the old pipeline's ~11 records. If it still happens, inspect the raw per-year USASpending CSVs.

---

## Model Specifications

| Stage | Model | Version | Purpose | Cost/grant | Typical speed |
|---|---|---|---|---|---|
| Stage 1 | Claude Haiku | 4.5 (Oct 2025) | Binary KEEP/REMOVE | ~$0.001 | ~1–2 sec |
| Stage 2-Formula | Claude Haiku | 4.5 | biological + bio_subcategory only | ~$0.001 | ~1–2 sec |
| Stage 2-Full | Claude Sonnet | 4 (May 2025) | 6-axis classification | ~$0.01 | ~2–3 sec |

Per-grant cost depends on how much Stage 2 work is needed. On a ~10K-grant dataset, typical total is ~$20–30.

---

## What Worked and What Didn't

### Worked ✅

- **Bulk CSVs over keyword-search API**: fixed the fundamental attribution bug where the API returned lifetime award totals dated to project start.
- **Two-stage classification**: cheap Haiku for keep/remove (~80% of cost reduction), Sonnet only for nuanced multi-axis work.
- **Formula grant routing**: running formula grants through a reduced Stage-2-Formula pass (bio only) keeps the Sankey panel populated without the full 6-axis cost.
- **Verbatim prompt preservation**: porting the old biomining classifier's prompts unchanged preserves prior accuracy, avoiding a re-validation cycle.
- **Resume-safe checkpointing**: every 10 batches. Multiple hour-long runs survive laptop reboots.
- **Schema adapter**: `biomining_shared._adapt_schema` renames new-pipeline columns to the old pipeline's names, so downstream plot code keeps working unchanged.

### Didn't Work ❌

- **Attribution by project `startDate`**: the original mistake. Led to ~$1.65M worth of 2013-era chitin work being attributed to 2013 despite $0 actually obligated in our 2016–2025 window.
- **`query_used` column in hard filter rules**: the old pipeline's keyword-search results carried a `query_used` column that the filter depended on. The bulk pipeline has no such column, so those rules were rephrased to trigger on abstract text content.
- **Keyword-based interdisciplinary flag**: noisy (too many false positives on "collaborative" as marketing language). Replaced by LLM-computed `research_approach` axis in Stage 2-Full.

---

## Model of the Dataset

**589 categorizable project grants totaling ~$1.46B (2016–2025)** from the old pipeline. New-pipeline numbers will differ slightly (more accurate year attribution). Bio share: ~3.3% of funding, ~11.5% of grants.

Full findings are re-computed with each run — see the PDF for current stats.

---

## Data Sources

- **NSF Awards API** — https://www.nsf.gov/developer/
- **USASpending.gov bulk CSV archive** — https://www.usaspending.gov/download_center/award_data_archive
- **Federal forward-looking dataset** (`federal_gapmap_master_future_updated_v2.csv`) — hand-curated from announced federal critical-minerals programs; used in PDF Section 3 and in the website's commitments page (as frozen aggregates).

---

## Support

For issues:
1. Check the script's docstring at the top.
2. Check `scripts/grant_classifier/output/two_stage_classification_log_all_years.json` for raw LLM responses when step3 produces surprising results.
3. Compare against the 70-grant holdout in `scripts/test_llm_classifier_accuracy.py` — disagreements usually surface systematic prompt issues.
