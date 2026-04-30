"""
score_holdout.py
================
Scores the biomining two-stage LLM classifier against a manually labeled holdout set.

USAGE:
    python score_holdout.py

INPUT:
    scripts/grant_classifier/output/mining_llm_classified_all_years.csv
    (output of step3_mining_two_stage_classifier_multiyear.py)

LABEL SCHEMA per row:
    "keep"            : True / False
    "mining_type"     : one of the valid values, or None if keep=False
    "bio_subcategory" : one of the valid values, or None
    "materials"       : list of valid values, or [] if keep=False
    "stage"           : one of the valid values, or None if keep=False
    "orientation"     : "public_good" or "industry_facing", or None if keep=False

Leave any taxonomy field as None if you don't want it scored.

HOLDOUT_LABELS below are verbatim from the old biomining pipeline's
score_holdout.py — 70 grants with manual ground truth. This lets you
compare new-pipeline accuracy directly against the old pipeline's scores.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent              # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent                            # biomining_federal_grant_funding/
OUTPUT_DIR   = PROJECT_ROOT / "scripts" / "grant_classifier" / "output"

# Default: score the merged final output from step3
CLASSIFIED_CSV = OUTPUT_DIR / "mining_llm_classified_all_years.csv"

# Column name for unique grant identifier (step1 output schema)
ID_COL = "unique_key"

# Column names in the new pipeline's classified output
COL_KEEP          = "s1_keep"
COL_KEEP_CONF     = "s1_confidence"
COL_REMOVE_REASON = "s1_remove_reason"
COL_LLM_PREFIX    = "llm_"    # llm_mining_type, llm_bio_subcategory, llm_stage, llm_orientation, llm_materials, llm_infrastructure_subtype


# =============================================================================
# HOLDOUT LABELS — 70 grants, verbatim from old biomining pipeline
# =============================================================================
HOLDOUT_LABELS = {

    # --- CONFIRMED REMOVES from initial session ---
    "USASpending::NI23SLSNXXXXG004": {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::26K75IL000025":    {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2447809":                  {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # --- CONFIRMED KEEPS from initial session ---
    "USASpending::GE146225": {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::SAQMIP23GR0356": {
        "keep": True,
        "mining_type": "extraction", "bio_subcategory": None,
        "materials": ["precious_metals"], "stage": "applied_translational",
        "orientation": "public_good",
        "research_approach": "single_focus",
    },
    "USASpending::20253361044958": {
        "keep": True,
        "mining_type": "remediation", "bio_subcategory": None,
        "materials": ["coal_ash_secondary", "polymetallic_general"],
        "stage": "early_technology_development", "orientation": "public_good",
        "research_approach": "single_focus",
    },
    "USASpending::R21ES034591": {
        "keep": True,
        "mining_type": "remediation", "bio_subcategory": None,
        "materials": ["base_metals", "polymetallic_general"],
        "stage": "applied_translational", "orientation": "public_good",
        "research_approach": "single_focus",
    },
    "USASpending::R43ES034319": {
        "keep": True,
        "mining_type": "remediation", "bio_subcategory": "bioremediation",
        "materials": ["polymetallic_general"], "stage": "early_technology_development",
        "orientation": "industry_facing",
        "research_approach": None,
    },
    "USASpending::R44ES034319": {
        "keep": True,
        "mining_type": "remediation", "bio_subcategory": "bioremediation",
        "materials": ["polymetallic_general"], "stage": "applied_translational",
        "orientation": "industry_facing",
        "research_approach": None,
    },
    "USASpending::80NSSC24K0805": {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::80NSSC21M0158": {
        # Extraterrestrial rule — Mars/space application = REMOVE
        "keep": False, "mining_type": None, "bio_subcategory": None,
        "materials": [], "stage": None, "orientation": None,
        "research_approach": None,
    },
    "USASpending::DEEE0006746": {
        "keep": True,
        "mining_type": "extraction", "bio_subcategory": None,
        "materials": ["battery_metals"], "stage": "early_technology_development",
        "orientation": "public_good",  # DOE funder, no SBIR marker, title-only → public_good default,
        "research_approach": "single_focus",
    },
    "USASpending::N000142112362": {
        "keep": True,
        "mining_type": "extraction", "bio_subcategory": "bioseparation_biosorption",
        "materials": ["lanthanides_REE"], "stage": "fundamental",
        "orientation": "public_good",
        "research_approach": "single_focus",
    },

    # --- BATCH 2: 36 additional rows ---
    "NSF::2342967": {
        "keep": True,
        "mining_type": "extraction", "bio_subcategory": "bioleaching",
        "materials": ["secondary_feedstocks", "battery_metals"], "stage": "early_technology_development",
        "orientation": "public_good",
        "research_approach": "single_focus",
    },
    "NSF::2446094": {
        "keep": True,
        "mining_type": "exploration", "bio_subcategory": None,
        "materials": ["lanthanides_REE", "secondary_feedstocks"], "stage": "fundamental",
        "orientation": "public_good",
        "research_approach": "single_focus",
    },
    "USASpending::DESC0020204": {
        "keep": True,
        "mining_type": "extraction", "bio_subcategory": None,
        "materials": ["lanthanides_REE"], "stage": "early_technology_development",
        "orientation": "public_good",
        "research_approach": "single_focus",
    },
    "NSF::1945401":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2550847":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2052585":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2240755":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::1953245":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::1954336":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::1736995":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::1709322":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::1650152":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2238415":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::1903685":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2403854":              {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::F22AP03247":   {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::NNX17AE87G":   {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::R00GM149822":  {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::R56DK106135":  {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # --- BATCH 3: 18 new grants ---
    "NSF::1805550": {"keep": True, "mining_type": "downstream_processing", "bio_subcategory": None, "materials": ["polymetallic_general"], "stage": "early_technology_development", "orientation": "public_good", "research_approach": "single_focus"},
    "NSF::2437985": {"keep": True, "mining_type": "extraction", "bio_subcategory": None, "materials": ["lanthanides_REE", "actinides", "secondary_feedstocks"], "stage": "early_technology_development", "orientation": "industry_facing", "research_approach": None},
    "NSF::2527909": {"keep": True, "mining_type": "extraction", "bio_subcategory": "bioleaching", "materials": ["lanthanides_REE", "secondary_feedstocks"], "stage": "applied_translational", "orientation": "industry_facing", "research_approach": None},
    "USASpending::96347301": {"keep": True, "mining_type": "remediation", "bio_subcategory": None, "materials": ["polymetallic_general"], "stage": "deployment", "orientation": "public_good", "research_approach": "single_focus"},
    "NSF::2112301": {"keep": True, "mining_type": "extraction", "bio_subcategory": None, "materials": ["secondary_feedstocks", "battery_metals"], "stage": "applied_translational", "orientation": "industry_facing", "research_approach": None},
    "USASpending::S26AF00008": {"keep": True, "mining_type": "remediation", "bio_subcategory": None, "materials": ["polymetallic_general"], "stage": "deployment", "orientation": "public_good", "research_approach": "single_focus"},
    "USASpending::00350022": {"keep": True, "mining_type": "remediation", "bio_subcategory": None, "materials": ["polymetallic_general"], "stage": "deployment", "orientation": "public_good", "research_approach": None},
    "NSF::2150925": {"keep": True, "mining_type": "infrastructure", "bio_subcategory": None, "materials": ["base_metals"], "stage": "applied_translational", "orientation": "industry_facing", "research_approach": None},
    "NSF::2212919": {"keep": True, "mining_type": "infrastructure", "bio_subcategory": None, "materials": ["polymetallic_general"], "stage": "applied_translational", "orientation": "industry_facing", "research_approach": None},
    "USASpending::G25AC00336": {"keep": True, "mining_type": "collaborative_tools", "bio_subcategory": None, "materials": ["polymetallic_general"], "stage": "deployment", "orientation": "public_good", "research_approach": "single_focus"},
    "USASpending::NNX15AP69H": {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2127732": {"keep": True, "mining_type": "extraction", "bio_subcategory": "bioseparation_biosorption", "materials": ["lanthanides_REE"], "stage": "fundamental", "orientation": "public_good", "research_approach": "single_focus"},
    "USASpending::DEAR0001339": {
        "keep": True,
        "mining_type": "extraction", "bio_subcategory": "bioseparation_biosorption",
        "materials": ["polymetallic_general"], "stage": "early_technology_development",
        "orientation": "public_good",
        "research_approach": "single_focus",
    },
    "NSF::2404728": {"keep": True, "mining_type": "extraction", "bio_subcategory": None, "materials": ["polymetallic_general"], "stage": "fundamental", "orientation": "public_good", "research_approach": "collaborative_interdisciplinary"},

    "NSF::1845683":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::047907710":      {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "USASpending::20186701528300": {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2423645":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # =========================================================================
    # BATCH 4: 20 grants targeting failure modes identified in production run
    # =========================================================================

    # SBIR explicitly naming biomining as target application — KEEP but axis not scored
    "NSF::2409142": {
        "keep": True,
        "mining_type": None, "bio_subcategory": None,
        "materials": None, "stage": None, "orientation": None,
        "research_approach": None,
    },

    # Phosphorus sensing for global biogeochemistry — food/water/energy context, not mining
    "NSF::2317826":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2317822":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2317823":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # Confocal microscope — general departmental instrument, no explicit mining use
    "NSF::1828523":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # Wastewater phosphorus recovery — agriculture/eutrophication context, not mining commodity
    "NSF::1554511":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # N2 fixation — molybdenum as enzyme cofactor, not a mining target
    "NSF::1937843":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # Deep-earth carbon cycling — pure geoscience, no mining context
    "NSF::1917681":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::1917682":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # Freshwater watershed ecology — AMD incidental
    "USASpending::20193882129065": {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # EF-hand protein with lanthanide as structural biology research tool — biomedical
    "USASpending::R35GM160470":    {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # Sensor dyes for metal plating/circuit board wastewater — not mining application
    "NSF::1621759":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # Extraterrestrial — space resource extraction (SBIR Phase II)
    "NSF::2304615":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},
    "NSF::2136875":                {"keep": False, "mining_type": None, "bio_subcategory": None, "materials": [], "stage": None, "orientation": None, "research_approach": None},

    # --- SUBCATEGORY CORRECTIONS — 6 grants: keep=True, wrong bio_subcategory originally assigned ---

    # Geobacter protein nanowires precipitating Fe/Mn oxides — bioprecipitation, not bioseparation
    "NSF::1629439": {
        "keep": True,
        "mining_type": "extraction", "bio_subcategory": "bioprecipitation_mineral_formation",
        "materials": ["base_metals"], "stage": "fundamental", "orientation": "public_good",
        "research_approach": "single_focus",
    },

    # Phytoremediation of mine waste sites — bioremediation, not bioseparation
    "NSF::2107469": {
        "keep": True,
        "mining_type": "remediation", "bio_subcategory": "bioremediation",
        "materials": ["polymetallic_general"], "stage": "fundamental", "orientation": "public_good",
        "research_approach": "collaborative_interdisciplinary",
    },
    "NSF::2107177": {
        "keep": True,
        "mining_type": "remediation", "bio_subcategory": "bioremediation",
        "materials": ["polymetallic_general"], "stage": "fundamental", "orientation": "public_good",
        "research_approach": "collaborative_interdisciplinary",
    },

    # Arsenic speciation/bioavailability at mine sites — bioremediation
    "NSF::1952817": {
        "keep": True,
        "mining_type": "remediation", "bio_subcategory": "bioremediation",
        "materials": ["metalloids_specialty"], "stage": "fundamental", "orientation": "public_good",
        "research_approach": "collaborative_interdisciplinary",
    },

    # Glycolipids for uranium remediation at legacy uranium mine sites — bioremediation
    "NSF::1940373": {
        "keep": True,
        "mining_type": "remediation", "bio_subcategory": "bioremediation",
        "materials": ["actinides"], "stage": "early_technology_development", "orientation": "industry_facing",
        "research_approach": None,
    },

    # AA-anchored 2D silicoaluminophosphate zeolites — not biological mechanism
    "NSF::2502122": {
        "keep": False,
        "mining_type": None, "bio_subcategory": None,
        "materials": [], "stage": None, "orientation": None,
        "research_approach": None,
    },
}


# =============================================================================
# SCORING
# =============================================================================
TAXONOMY_FIELDS = ["mining_type", "bio_subcategory", "stage", "orientation", "research_approach", "infrastructure_subtype"]


def parse_materials(val):
    """Parse materials field from CSV (stored as JSON string or list)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        # Old-pipeline sometimes stored Python-literal list-like strings.
        s = str(val).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                return ast.literal_eval(s)
            except Exception:
                return []
        return []


def score(classified_csv, labels: dict):
    classified_csv = Path(classified_csv)
    if not classified_csv.exists():
        print(f"❌ Classified CSV not found: {classified_csv}")
        print(f"   Run step3_mining_two_stage_classifier_multiyear.py first.")
        return

    df = pd.read_csv(classified_csv, low_memory=False)

    if ID_COL not in df.columns:
        print(f"❌ Input CSV is missing '{ID_COL}' column.")
        print(f"   Available columns: {list(df.columns)[:15]}...")
        return

    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    # Filter to only labeled rows
    labeled_keys = set(labels.keys())
    df_scored = df[df[ID_COL].isin(labeled_keys)].copy()

    missing = labeled_keys - set(df_scored[ID_COL])
    if missing:
        print(f"\n⚠️  {len(missing)} labeled row(s) not found in classified CSV:")
        for k in sorted(missing):
            print(f"   {k}")

    n = len(df_scored)
    if n == 0:
        print(f"❌ No matching rows found. Check that {ID_COL} values match the holdout keys.")
        return

    print(f"\n{'=' * 60}")
    print(f"HOLDOUT SCORING RESULTS")
    print(f"{'=' * 60}")
    print(f"Classified CSV:            {classified_csv}")
    print(f"Labeled rows found in CSV: {n} / {len(labeled_keys)}")

    # ------------------------------------------------------------------
    # 1. KEEP / REMOVE accuracy  (new pipeline: s1_keep)
    # ------------------------------------------------------------------
    keep_results = []
    tp = fp = tn = fn = 0

    for _, row in df_scored.iterrows():
        uid = row[ID_COL]
        truth = labels[uid]

        # New pipeline exposes s1_keep (bool or NA). Missing → assume kept so
        # we don't silently mark everything as REMOVE when the column's absent.
        raw_keep = row.get(COL_KEEP)
        if pd.isna(raw_keep):
            predicted_keep = True
        elif isinstance(raw_keep, (bool, int)):
            predicted_keep = bool(raw_keep)
        else:
            predicted_keep = str(raw_keep).strip().lower() in ("true", "1", "yes", "keep")

        actual_keep = truth["keep"]

        correct = predicted_keep == actual_keep
        keep_results.append({
            "uid": uid,
            "actual": actual_keep,
            "predicted": predicted_keep,
            "correct": correct,
            "confidence": row.get(COL_KEEP_CONF, ""),
            "remove_reason": row.get(COL_REMOVE_REASON, ""),
        })

        if actual_keep and predicted_keep:      tp += 1
        elif not actual_keep and predicted_keep: fp += 1
        elif actual_keep and not predicted_keep: fn += 1
        else:                                    tn += 1

    accuracy  = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- KEEP / REMOVE (Stage 1, Haiku) ---")
    print(f"  Accuracy  : {accuracy:.1%}  ({tp + tn}/{n} correct)")
    print(f"  Precision : {precision:.1%}  (of rows LLM kept, how many should be kept)")
    print(f"  Recall    : {recall:.1%}  (of rows that should be kept, how many LLM kept)")
    print(f"  F1        : {f1:.1%}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    errors = [r for r in keep_results if not r["correct"]]
    if errors:
        print(f"\n  ❌ Errors ({len(errors)}):")
        for e in errors:
            direction = "FALSE KEEP" if e["predicted"] and not e["actual"] else "FALSE REMOVE"
            print(f"     [{direction}] {e['uid']}")
            if e["remove_reason"]:
                print(f"       LLM reason: {e['remove_reason']}")
    else:
        print(f"\n  ✅ No keep/remove errors")

    # ------------------------------------------------------------------
    # 2. TAXONOMY accuracy  (Stage 2 — on correctly kept rows only)
    # ------------------------------------------------------------------
    kept_keys = {r["uid"] for r in keep_results if r["actual"] and r["predicted"]}
    df_kept = df_scored[df_scored[ID_COL].isin(kept_keys)]

    print(f"\n--- TAXONOMY (Stage 2, on {len(df_kept)} correctly kept rows) ---")

    for field in TAXONOMY_FIELDS:
        llm_col = f"{COL_LLM_PREFIX}{field}"
        correct_count = 0
        scoreable = 0
        field_errors = []

        for _, row in df_kept.iterrows():
            uid = row[ID_COL]
            truth_val = labels[uid].get(field)
            if truth_val is None:
                continue  # not labeled for this field, skip
            scoreable += 1
            predicted_val = row.get(llm_col)
            if pd.isna(predicted_val):
                predicted_val = None
            if str(predicted_val) == str(truth_val):
                correct_count += 1
            else:
                field_errors.append((uid, truth_val, predicted_val))

        if scoreable == 0:
            print(f"  {field:<20}: no labels provided")
            continue

        acc = correct_count / scoreable
        print(f"  {field:<20}: {acc:.1%}  ({correct_count}/{scoreable})")
        if field_errors:
            for uid, truth_val, pred_val in field_errors:
                print(f"    ❌ {uid}")
                print(f"       expected={truth_val}  got={pred_val}")

    # ------------------------------------------------------------------
    # 3. MATERIALS accuracy (kept rows, set overlap)
    # ------------------------------------------------------------------
    mat_exact = mat_partial = mat_wrong = 0
    mat_scoreable = 0

    for _, row in df_kept.iterrows():
        uid = row[ID_COL]
        truth_mats_raw = labels[uid].get("materials")
        if truth_mats_raw is None:
            continue  # not labeled
        truth_mats = set(truth_mats_raw)
        mat_scoreable += 1

        pred_mats = set(parse_materials(row.get(f"{COL_LLM_PREFIX}materials")))

        if pred_mats == truth_mats:
            mat_exact += 1
        elif pred_mats & truth_mats:
            mat_partial += 1
        else:
            mat_wrong += 1

    if mat_scoreable > 0:
        print(f"\n  {'materials':<20}: {mat_exact / mat_scoreable:.1%} exact  "
              f"{mat_partial / mat_scoreable:.1%} partial  "
              f"{mat_wrong / mat_scoreable:.1%} wrong  "
              f"({mat_scoreable} rows)")
        for _, row in df_kept.iterrows():
            uid = row[ID_COL]
            truth_mats_raw = labels[uid].get("materials")
            if truth_mats_raw is None:
                continue
            truth_mats = set(truth_mats_raw)
            pred_mats = set(parse_materials(row.get(f"{COL_LLM_PREFIX}materials")))
            if pred_mats != truth_mats:
                tag = "⚠️ PARTIAL" if pred_mats & truth_mats else "❌ WRONG"
                print(f"    {tag} {uid}")
                print(f"       expected={sorted(truth_mats)}  got={sorted(pred_mats)}")

    # ------------------------------------------------------------------
    # 4. CONFIDENCE calibration  (Stage 1)
    # ------------------------------------------------------------------
    print(f"\n--- CONFIDENCE CALIBRATION (Stage 1) ---")
    conf_buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in keep_results:
        c = str(r["confidence"]).strip().lower() if r["confidence"] else ""
        conf_buckets[c]["total"] += 1
        if r["correct"]:
            conf_buckets[c]["correct"] += 1

    for conf in ["high", "medium", "low"]:
        b = conf_buckets[conf]
        if b["total"] > 0:
            print(f"  {conf:<8}: {b['correct'] / b['total']:.1%} accurate  ({b['correct']}/{b['total']} rows)")

    # ------------------------------------------------------------------
    # 5. Summary verdict + ranked category accuracy table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Keep/Remove accuracy : {accuracy:.1%}  (F1: {f1:.1%})")
    print()

    category_scores = [("keep/remove", accuracy, n, n - (tp + tn))]

    for field in TAXONOMY_FIELDS:
        llm_col = f"{COL_LLM_PREFIX}{field}"
        correct_count = 0
        scoreable = 0
        errors = 0
        for _, row in df_kept.iterrows():
            uid = row[ID_COL]
            truth_val = labels[uid].get(field)
            if truth_val is None:
                continue
            scoreable += 1
            predicted_val = row.get(llm_col)
            if pd.isna(predicted_val):
                predicted_val = None
            if str(predicted_val) == str(truth_val):
                correct_count += 1
            else:
                errors += 1
        if scoreable > 0:
            category_scores.append((field, correct_count / scoreable, scoreable, errors))

    if mat_scoreable > 0:
        mat_acc = mat_exact / mat_scoreable
        category_scores.append(("materials (exact)", mat_acc, mat_scoreable, mat_scoreable - mat_exact))

    category_scores.sort(key=lambda x: x[1])

    print(f"  {'CATEGORY':<25} {'ACCURACY':>10}  {'CORRECT':>8}  {'ERRORS':>8}  PRIORITY")
    print(f"  {'-' * 65}")
    for cat, acc, total, errs in category_scores:
        if acc < 0.80:
            priority = "🔴 HIGH"
        elif acc < 0.90:
            priority = "🟡 MEDIUM"
        else:
            priority = "🟢 LOW"
        print(f"  {cat:<25} {acc:>9.1%}  {total - errs:>6}/{total:<2}  {errs:>6} err  {priority}")

    print()
    overall = sum(s[1] * s[2] for s in category_scores) / sum(s[2] for s in category_scores)
    print(f"  Overall weighted accuracy : {overall:.1%}")

    if accuracy >= 0.92:
        verdict = "✅ STRONG — ready for production run"
    elif accuracy >= 0.85:
        verdict = "⚠️  ACCEPTABLE — consider reviewing remaining errors before full run"
    else:
        verdict = "❌ NEEDS WORK — prompt requires further calibration"
    print(f"  Keep/Remove verdict       : {verdict}")
    print()


if __name__ == "__main__":
    score(CLASSIFIED_CSV, HOLDOUT_LABELS)
