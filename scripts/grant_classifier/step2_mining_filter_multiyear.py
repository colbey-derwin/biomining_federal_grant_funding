#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_mining_filter_multiyear.py

Mining-only filter for the NEW bulk-download pipeline.

Runs AFTER step1_merge_master_multiyear.py. Reads merged_all_years.csv
(step1 output, schema: source, award_id, title, abstract, por_text,
start_date, end_date, award_amount, award_type, funder, year, ...).

FILTER RULES — identical in spirit and effect to the predecessor script
`federal_filter_mining_only_with_group_exclusion.py` from the original
(API-pull) biomining pipeline.

  Any single grant presented to either filter will receive the SAME
  keep/exclude decision based on:
    - mining-only gating (Cat A/B + process terms)
    - exclude non-mining "data mining / genome mining / etc."
    - exclude physiology / biomedical / clinical (threshold-based)
    - exclude plastics (threshold-based)
    - exclude archaeology / paleo / heritage (threshold-based)
    - standalone auto-keep: biomining / bioleaching / biooxidation /
      biohydrometallurgy / biometallurgy

  One adaptation:
    Standalone auto-keep in the OLD filter was keyed on the `query_used`
    column (present only when grants were pulled by keyword search). The
    new pipeline has no keyword-query context, so standalone keeping is
    detected by regex-matching the same keyword set against
    title + abstract + por_text. Any grant containing those exact
    keywords in its text will auto-keep — equivalent behavior for
    on-topic grants, since those keywords appear in the text almost
    by definition.

ADDITIONS:
  1. Formula-grant FLAG (not drop) — adds is_formula_grant column where
                                     award_type == "FORMULA GRANT (A)".
                                     Formula grants pass through step2
                                     unchanged so the LLM can classify them
                                     (mirrors the old biomining site which
                                     shows formula grants in their own
                                     Sankey panel with full LLM detail).
                                     A separate CSV is also saved for QA.
  2. Abstract-length filter        — drops abstract < 92 chars
                                     (matches OLD biomining site threshold)
                                     — these are the only hard drops after
                                     the mining filter.

OUTPUTS (in scripts/grant_classifier/output/):
  mining_filtered_all_years.csv              — final kept (LLM-ready)
  mining_excluded_all_years.csv              — excluded by mining filter
  mining_formula_grants_all_years.csv        — passed mining, removed as formula
  mining_insufficient_abstract_all_years.csv — passed mining+non-formula,
                                               abstract < 92 chars
"""

import re
import pandas as pd
from pathlib import Path


# =========================
# PATHS
# =========================
SCRIPT_DIR   = Path(__file__).resolve().parent           # scripts/grant_classifier/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                  # biomining_federal_grant_funding/
OUTPUT_DIR   = PROJECT_ROOT / "scripts" / "grant_classifier" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INFILE         = OUTPUT_DIR / "merged_all_years.csv"   # from step1
OUT_KEPT       = OUTPUT_DIR / "mining_filtered_all_years.csv"
OUT_EXCL       = OUTPUT_DIR / "mining_excluded_all_years.csv"
OUT_FORMULA    = OUTPUT_DIR / "mining_formula_grants_all_years.csv"
OUT_INSUFF     = OUTPUT_DIR / "mining_insufficient_abstract_all_years.csv"


# =========================
# SETTINGS — identical to old filter
# =========================
PHYS_EXCLUDE_THRESHOLD     = 3
PLASTICS_EXCLUDE_THRESHOLD = 3
ARCH_EXCLUDE_THRESHOLD     = 3

# Additions
ABSTRACT_MIN_CHARS         = 92
FORMULA_AWARD_TYPE_LABEL   = "FORMULA GRANT (A)"


# =========================
# STANDALONE AUTO-KEEP
# =========================
STANDALONE_KEEP_QUERIES = {
    "biomining",
    "bioleaching",  "bioleach",
    "biooxidation",
    "biohydrometallurgy",
    "biometallurgy",
    "phytomining",  "phytomine",  "phytomines",
}


# =========================
# USER RULES: TERMS — identical to old filter
# =========================
BIO_NEEDS_A_OR_B = [
    "bioseparations",
    "bioextraction",
    "chelation",
    "hyper-accumulation",
    "biosorption",
    "biomineralization",
    "biogeochemical",
    "microbial oxidation",
]

CAT_A = [
    "mineral processing",
    #"ore",
    "tailings",
    "heap leach",
    "beneficiation",
    "hydrometallurgy",
    "acid mine drainage",
    "mine waste",
    "waste rock",
    "tailings reprocessing",
    "ore processing",
    "extractive metallurgy",
    "sulfide ore",
    "sulfide mineral",
]

CAT_B = [
    "rare earth",
    "lanthanide",
    "scandium",
    "yttrium",
    "lithium",
    "cobalt",
    "nickel",
    "copper",
    "uranium",
    "manganese",
    "graphite",
    "phosphate",
    "critical mineral",
    "rare earth element",
    "chalcopyrite",
    "coal ash",
    "ore",
    "mining",
]

CAT_B_ALLOWED_PROCESS_TERMS = [
    "extract",
    "extraction",
    "leach",
    "leaching",
    "leachate",
    "recover",
    "recovery",
    "separate",
    "separation",
    "concentrate",
    "concentration",
    "metal recovery",
    "flotation",
    # Hyperaccumulation — plant-based metal uptake (phytomining process)
    "hyperaccumulation",
    "hyperaccumulator",
    "hyperaccumulators",
    "hyperaccumulate",
]


# =========================
# EXCLUSION: "mining" in non-mining sense
# =========================
NON_MINING_MINING_SENSE = [
    "data mining",
    "text mining",
    "web mining",
    "process mining",
    "opinion mining",
    "sentiment mining",
    "genome mining",
    "genomic mining",
    "metagenome mining",
    "metagenomic mining",
    "gene mining",
    "sequence mining",
    "literature mining",
    "knowledge mining",
    "pattern mining",
    "association mining",
    "mining of data",
    "mining the literature",
    "mined data",
]


# =========================
# EXCLUSION: physiology / biomedical / clinical (THRESHOLDED)
# =========================
PHYSIOLOGY_BIOMED_EXCLUDE = [
    "clinical",
    "patient",
    "patients",
    "hospital",
    "clinical trial",
    "randomized",
    "placebo",
    "human subjects",
    "disease",
    "cancer",
    "tumor",
    "biomedical",
    "medical device",
    "medical devices",
    "drug delivery",
    "physiology",
    "healthcare",
    "COVID-19",
    "pandemic",
    "viruses",
    "human behavior",
    "psychological",
    "trauma",
    "gut",
    "oral",
]


# =========================
# EXCLUSION: plastics / microplastics (THRESHOLDED)
# =========================
PLASTICS_EXCLUDE = [
    "plastic",
    "plastics",
    "microplastic",
    "microplastics",
    "nanoplastic",
    "nanoplastics",
    "polyethylene",
    "polypropylene",
    "polystyrene",
    "polyvinyl chloride",
    "pvc",
    "pet",
    "polyethylene terephthalate",
    "polymer waste",
    "plastic waste",
]


# =========================
# EXCLUSION: archaeology / anthropology / paleo / heritage (THRESHOLDED)
# =========================
ARCHAEOLOGY_EXCLUDE = [
    "archaeology",
    "archaeological",
    "anthropology",
    "indigenous",
    "colonialism",
    "colonial",
    "pottery",
    "ceramic",
    "ceramics",
    "glaze",
    "artifact",
    "artifacts",
    "excavation",
    "stratigraphy",
    "radiocarbon",
    "carbon dating",
    "provenance",
    "paleontology",
    "palaeontology",
    "fossil",
    "fossils",
    "taphonomy",
    "paleoecology",
    "palaeoecology",
    "pleistocene",
    "holocene",
    "isotope sourcing",
    "lead isotope",
    "petrography",
]


# =========================
# HELPERS — identical to old filter
# =========================
def _norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _build_phrase_regex(phrases):
    parts = []
    for p in phrases:
        p = p.strip().lower()
        esc = re.escape(p)
        esc = esc.replace(r"\ ", r"[\s\-]+")
        parts.append(esc)
    return re.compile(r"\b(" + "|".join(parts) + r")\b", flags=re.IGNORECASE)


def _count_phrase_matches(phrases, text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    t = text.lower()
    hits = 0
    for p in phrases:
        p2 = p.strip().lower()
        if not p2:
            continue
        esc = re.escape(p2).replace(r"\ ", r"[\s\-]+")
        rx = re.compile(r"\b" + esc + r"\b", flags=re.IGNORECASE)
        if rx.search(t):
            hits += 1
    return hits


RX_A          = _build_phrase_regex(CAT_A)
RX_B          = _build_phrase_regex(CAT_B)
RX_BPROC      = _build_phrase_regex(CAT_B_ALLOWED_PROCESS_TERMS)
RX_BIO_NEEDS  = _build_phrase_regex(BIO_NEEDS_A_OR_B)
RX_NONMINING  = _build_phrase_regex(NON_MINING_MINING_SENSE)

RX_GENERIC_MINING = re.compile(
    r"\b(mining|mine|mines|mined|miner|mineral|mineralogy)\b",
    flags=re.IGNORECASE,
)

RX_STANDALONE = re.compile(
    r"\b(" + "|".join(re.escape(q) for q in STANDALONE_KEEP_QUERIES) + r")\b",
    flags=re.IGNORECASE,
)


# =========================
# FILTER
# =========================
def apply_mining_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - text_blob
      - hit_* flags
      - standalone_keep
      - keep_mining_only
      - filter_reason
      - physiology_count / plastics_count / arch_count
    """
    df = df.copy()

    for c in ["title", "abstract", "por_text"]:
        if c not in df.columns:
            df[c] = ""

    # text_blob over the fields step1 gives us
    df["text_blob"] = (
        df["title"].fillna("").astype(str) + " | " +
        df["abstract"].fillna("").astype(str) + " | " +
        df["por_text"].fillna("").astype(str)
    ).map(_norm)

    # standalone_keep — keyword appears anywhere in the text blob
    df["standalone_keep"] = df["text_blob"].apply(
        lambda s: bool(RX_STANDALONE.search(s))
    )

    # hit flags
    df["hit_cat_a"]               = df["text_blob"].apply(lambda s: bool(RX_A.search(s)))
    df["hit_cat_b"]               = df["text_blob"].apply(lambda s: bool(RX_B.search(s)))
    df["hit_bproc"]               = df["text_blob"].apply(lambda s: bool(RX_BPROC.search(s)))
    df["hit_bio_needs"]           = df["text_blob"].apply(lambda s: bool(RX_BIO_NEEDS.search(s)))
    df["hit_nonmining"]           = df["text_blob"].apply(lambda s: bool(RX_NONMINING.search(s)))
    df["hit_generic_mining_word"] = df["text_blob"].apply(lambda s: bool(RX_GENERIC_MINING.search(s)))

    df["physiology_count"] = df["text_blob"].apply(
        lambda s: _count_phrase_matches(PHYSIOLOGY_BIOMED_EXCLUDE, s)
    )
    df["hit_physiology"] = df["physiology_count"] >= PHYS_EXCLUDE_THRESHOLD

    df["plastics_count"] = df["text_blob"].apply(
        lambda s: _count_phrase_matches(PLASTICS_EXCLUDE, s)
    )
    df["hit_plastics"] = df["plastics_count"] >= PLASTICS_EXCLUDE_THRESHOLD

    df["arch_count"] = df["text_blob"].apply(
        lambda s: _count_phrase_matches(ARCHAEOLOGY_EXCLUDE, s)
    )
    df["hit_archaeology"] = df["arch_count"] >= ARCH_EXCLUDE_THRESHOLD

    # decision rules — identical to old filter
    rule1_ok = (~df["hit_bio_needs"]) | (df["hit_cat_a"] | df["hit_cat_b"])
    rule2_ok = (~df["hit_cat_b"])     | (df["hit_cat_a"] | df["hit_bproc"])
    base_mining_ok = (df["hit_cat_a"] | df["hit_cat_b"]) & rule1_ok & rule2_ok

    strong_mining_evidence = (
        (df["hit_cat_a"] & (df["hit_cat_b"] | df["hit_bproc"])) |
        (df["hit_cat_b"] & df["hit_bproc"]) |
        (df["hit_cat_a"] & df["hit_generic_mining_word"])
    )
    rule3_ok = (~df["hit_nonmining"]) | strong_mining_evidence
    rule4_ok = ~df["hit_physiology"]
    rule5_ok = ~df["hit_plastics"]
    rule6_ok = ~df["hit_archaeology"]

    df["keep_mining_only"] = (
        base_mining_ok & rule3_ok & rule4_ok & rule5_ok & rule6_ok
    ) | df["standalone_keep"]

    def _reason(row):
        if row["keep_mining_only"] and row.get("standalone_keep", False):
            return "kept_standalone_query"
        if row["keep_mining_only"]:
            return "kept"
        if row["hit_physiology"]:
            return "excluded_physiology_biomed"
        if row["hit_plastics"]:
            return "excluded_plastics"
        if row["hit_archaeology"]:
            return "excluded_archaeology"
        if row["hit_nonmining"] and not strong_mining_evidence.loc[row.name]:
            return "excluded_non_mining_sense"
        if row["hit_bio_needs"] and not (row["hit_cat_a"] or row["hit_cat_b"]):
            return "excluded_bio_term_without_cat_a_or_b"
        if row["hit_cat_b"] and not (row["hit_cat_a"] or row["hit_bproc"]):
            return "excluded_cat_b_without_cat_a_or_process_terms"
        if not (row["hit_cat_a"] or row["hit_cat_b"]):
            return "excluded_no_cat_a_or_b_evidence"
        return "excluded_other"

    df["filter_reason"] = df.apply(_reason, axis=1)
    return df


# =========================
# MAIN
# =========================
def main():
    print(f"Loading: {INFILE}")
    df = pd.read_csv(INFILE, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    print("\nApplying mining-only filter...")
    out = apply_mining_filter(df)

    kept_raw = out.loc[out["keep_mining_only"]].copy()
    excl     = out.loc[~out["keep_mining_only"]].copy()

    # ---- formula-grant FLAG (not drop) ----
    # Formula grants pass through step2 unchanged so the LLM can classify them.
    # Downstream display separates them via is_formula_grant.
    if "award_type" in kept_raw.columns:
        kept_raw["is_formula_grant"] = (
            kept_raw["award_type"].fillna("").astype(str) == FORMULA_AWARD_TYPE_LABEL
        )
    else:
        kept_raw["is_formula_grant"] = False

    formula_grants = kept_raw[kept_raw["is_formula_grant"]].copy()
    n_formula      = len(formula_grants)
    has_amt        = "award_amount" in formula_grants.columns
    amt_formula    = formula_grants["award_amount"].sum() if (n_formula > 0 and has_amt) else 0

    print(f"\n--- Formula Grant Flagging (kept for LLM) ---")
    print(f"Formula grants flagged: {n_formula:,} (${amt_formula:,.0f})")

    # Rough bio vs non-bio breakdown (keyword proxy — not LLM-defined bio).
    # A formula grant is "bio-ish" if its text tripped any bio-flavored
    # filter signal: standalone_keep (biomining/bioleaching/phytomining/etc.)
    # or hit_bio_needs (biosorption/biomineralization/biogeochemical/etc.).
    if n_formula > 0:
        is_bioish = (
            formula_grants["standalone_keep"].fillna(False).astype(bool) |
            formula_grants["hit_bio_needs"].fillna(False).astype(bool)
        )
        n_bio    = int(is_bioish.sum())
        n_nonbio = n_formula - n_bio
        amt_bio    = formula_grants.loc[is_bioish,  "award_amount"].sum() if has_amt else 0
        amt_nonbio = formula_grants.loc[~is_bioish, "award_amount"].sum() if has_amt else 0
        print(f"  Bio-ish (keyword proxy): {n_bio:,}  (${amt_bio:,.0f})")
        print(f"  Non-bio:                 {n_nonbio:,}  (${amt_nonbio:,.0f})")

    # ---- abstract-length filter (only hard drop post-mining) ----
    kept_raw["abstract_length"]     = kept_raw["abstract"].fillna("").astype(str).str.len()
    kept_raw["sufficient_abstract"] = kept_raw["abstract_length"] >= ABSTRACT_MIN_CHARS

    insufficient = kept_raw[~kept_raw["sufficient_abstract"]].copy()
    kept_final   = kept_raw[kept_raw["sufficient_abstract"]].copy()

    print(f"\n--- Abstract Length Filter (≥ {ABSTRACT_MIN_CHARS} chars) ---")
    print(f"Sufficient abstract:    {len(kept_final):,}  (LLM-ready, includes formula)")
    print(f"Insufficient abstract:  {len(insufficient):,}")

    # ---- save ----
    kept_final.to_csv(OUT_KEPT, index=False)

    # Excluded CSV used to include every column (incl. the fat text_blob),
    # which produced multi-GB files on large input. Save a slim version with
    # only the columns useful for QA: id, title, funder, year, $, and why.
    EXCL_SLIM_COLS = [
        "unique_key", "source", "award_id", "title", "funder",
        "year", "award_amount", "award_type", "filter_reason",
        # hit flags — lightweight, useful for debugging exclusion decisions
        "hit_cat_a", "hit_cat_b", "hit_bproc", "hit_bio_needs",
        "hit_nonmining", "hit_physiology", "hit_plastics", "hit_archaeology",
        "standalone_keep",
    ]
    excl_slim = excl[[c for c in EXCL_SLIM_COLS if c in excl.columns]].copy()
    try:
        excl_slim.to_csv(OUT_EXCL, index=False)
    except OSError as e:
        print(f"  ⚠  Could not save excluded CSV ({e}); skipping — primary output was saved.")
    if n_formula > 0:
        # Reference-only QA file; these rows are also present in kept_final.
        try:
            formula_grants.to_csv(OUT_FORMULA, index=False)
        except OSError as e:
            print(f"  ⚠  Could not save formula grants CSV ({e}); skipping.")
    if len(insufficient) > 0:
        try:
            insufficient.to_csv(OUT_INSUFF, index=False)
        except OSError as e:
            print(f"  ⚠  Could not save insufficient-abstract CSV ({e}); skipping.")

    # ---- summary ----
    print("\n" + "=" * 70)
    print("FILTERING COMPLETE")
    print("=" * 70)
    print(f"Input rows:                  {len(df):,}")
    print(f"Excluded by mining filter:   {len(excl):,}")
    print(f"Passed mining filter:        {len(kept_raw):,}")
    print(f"  Formula grants flagged:    {n_formula:,}  (kept; flagged via is_formula_grant)")
    print(f"  Insufficient abstract:     {len(insufficient):,}  (dropped)")
    print(f"Final kept (LLM-ready):      {len(kept_final):,}")
    print()
    print(f"Saved kept:                  {OUT_KEPT}")
    print(f"Saved excluded:              {OUT_EXCL}")
    if n_formula > 0:     print(f"Saved formula grants (QA):   {OUT_FORMULA}")
    if len(insufficient): print(f"Saved insufficient:          {OUT_INSUFF}")

    print("\n--- Exclusion reasons (top 30) ---")
    if len(excl) > 0:
        print(excl["filter_reason"].value_counts(dropna=False).head(30))

    print("\n--- Targeted exclusion totals ---")
    print(f"  physiology/biomed: {int((excl['filter_reason'] == 'excluded_physiology_biomed').sum()):,}")
    print(f"  plastics:          {int((excl['filter_reason'] == 'excluded_plastics').sum()):,}")
    print(f"  archaeology:       {int((excl['filter_reason'] == 'excluded_archaeology').sum()):,}")


if __name__ == "__main__":
    main()
