#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_keyword_flags_multiyear.py

POST-PROCESSING SCRIPT: add keyword-based flags to the new-pipeline
classified dataset, equivalent to the old biomining
`add_keyword_flags_mining_SECTION1A.py`.

Runs AFTER step3_mining_two_stage_classifier_multiyear.py.

Input:  scripts/grant_classifier/output/mining_llm_classified_all_years.csv
Output: scripts/grant_classifier/output/mining_llm_classified_with_keyword_flags_all_years.csv

SECTION 1A filter — same logic as the old pipeline:
  1. Categorizable — abstract ≥ 92 chars (already enforced by step2; we
     double-check here for safety)
  2. Project grants — not formula grants (is_formula_grant == False)
  3. LLM-kept — s1_keep == True (Stage 1 KEPT grants only)

Then adds TWO flag columns based on abstract text analysis:
  1. industry_framing     - TEA, LCA, commercial viability analysis
  2. open_access_sharing  - Open access, data sharing, shared facilities

Note: the old script also computed an `interdisciplinary` flag via keyword
matching. That flag is DROPPED here because the step3 LLM now computes
`llm_research_approach` ∈ {collaborative_interdisciplinary, single_focus}
directly — more accurate than keyword matching. If you need the old-style
flag for compatibility, derive it as
    (df["llm_research_approach"] == "collaborative_interdisciplinary")

Runtime: ~1-2 minutes on the full dataset.
"""

import os
import re
import pandas as pd
from typing import List
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent      # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent                    # biomining_federal_grant_funding/
OUTPUT_DIR   = PROJECT_ROOT / "scripts" / "grant_classifier" / "output"

INFILE  = OUTPUT_DIR / "mining_llm_classified_all_years.csv"
OUTFILE = OUTPUT_DIR / "mining_llm_classified_with_keyword_flags_all_years.csv"


# =============================================================================
# FILTERING SETTINGS (Section 1A logic)
# =============================================================================
BENCHMARK_ABSTRACT_LENGTH = 92


# =============================================================================
# KEYWORD LISTS — verbatim from old biomining pipeline
# =============================================================================

# INDUSTRY FRAMING KEYWORDS — 17 total
INDUSTRY_KEYWORDS = [
    # Techno-economic analysis
    "techno-economic analysis",
    "technoeconomic",
    "TEA",

    # Life cycle assessment
    "life cycle assessment",
    "lifecycle assessment",
    "LCA",

    # Economic viability
    "economic feasibility",
    "economic viability",

    # Commercial viability
    "commercial feasibility",
    "commercial viability",
    "commercialization pathway",

    # Cost analysis
    "cost analysis",
    "cost-benefit analysis",

    # Scalability & market
    "scalability analysis",
    "scale-up economics",
    "market analysis",
    "market potential",
]

# OPEN ACCESS / SHARING KEYWORDS — 29 total
SHARING_KEYWORDS = [
    # Open access
    "open access",
    "open-access",
    "openly available",
    "publicly available",
    "public access",
    "free access",

    # Shared resources
    "shared facility",
    "shared platform",
    "shared resource",
    "shared database",
    "shared infrastructure",
    "community resource",
    "community facility",
    "community platform",

    # Data sharing
    "data sharing",
    "data repository",
    "open data",
    "open source",
    "open-source",

    # Availability language
    "available to researchers",
    "available to the community",
    "available to the public",
    "made available",
    "will be shared",
    "will be made available",

    # Collaborative access
    "collaborative facility",
    "multi-user facility",
    "user facility",
    "shared access",

    # ─── Tier 1 additions (parallel to existing patterns; identical to
    # climate biotech step5_post_classification_industry_relevance_flags) ───
    # Pedagogical content
    "open educational resource",
    "open educational resources",

    # Open platforms / repositories
    "open platform",
    "open repository",
    "open repositories",

    # Public dissemination phrases (single-match safe — multi-word phrases
    # that don't generate the false positives a bare "proceedings" or
    # "freely available" would).
    "conference proceedings",
    "present at conferences",
    "presented at conferences",
    "present at a conference",
    "presented at a conference",
]


# =============================================================================
# OPEN ACCESS / SHARING — Tier 2 paired logic
# =============================================================================
# Catches dissemination / proceedings / "freely available" language that the
# Tier 1 keyword list misses, while requiring nearby public-context signals
# AND excluding restrictive (member-only) language. Identical to climate
# biotech step5_post_classification_industry_relevance_flags_multiyear.py
# so the two analyses share methodology.
#
# Rationale:
#   - "disseminate" alone is ambiguous; needs a public signal nearby.
#   - "freely available" alone matches both output-sharing ("results made
#     freely available") AND input-resource availability ("freely available
#     soil", "freely available WGS data"). Requires output-noun pairing.
#   - "proceedings" alone is weak; "conference proceedings" is in Tier 1.

TIER2_DISSEMINATION_TRIGGER = r"\bdisseminat\w+"
TIER2_PROCEEDINGS_TRIGGER   = r"\bproceedings\b"
TIER2_FREELY_AVAILABLE      = r"\bfreely\s+available\b"

PUBLIC_SIGNALS = [
    r"\bcommunity\b", r"\bpublicly\b", r"\bbroadly\b", r"\bwidely\b",
    r"\bscientific\s+community\b", r"\bresearch\s+community\b",
    r"\bthe\s+public\b", r"\bpublished\b", r"\bdistributed\b",
    r"\bonline\b", r"\bopen\b",
]

RESTRICTIVE_SIGNALS = [
    r"\bmember[-\s]?only\b", r"\bmembers\s+only\b",
    r"\bconsortium\s+members\s+only\b",
    r"\brestricted\s+to\b", r"\blimited\s+to\b",
    r"\binternal\s+use\s+only\b", r"\bproprietary\b",
]

# Output-noun pairing for "freely available" (must appear within 80 chars).
# Bare "data" deliberately excluded — too ambiguous (matches input data too,
# e.g. "using freely available WGS data" → input, not output sharing).
OUTPUT_NOUNS_PATTERN = (
    r"\b(results?|findings?|datasets?|data\s+(archive|product|portal)|"
    r"tools?|code|software|packages?|models?|protocols?|publications?|"
    r"lectures?|kits?|videos?|training\s+materials?|platforms?|"
    r"frameworks?|repositor\w+|databases?|R\s+packages?|"
    r"educational\s+materials?|outputs?|deliverables?)\b"
)


def has_open_access_supplement(abstract) -> bool:
    """Tier 2 paired-logic check for open_access_sharing.

    Catches grants whose abstract contains explicit dissemination, proceedings,
    or 'freely available' language that the Tier 1 SHARING_KEYWORDS list misses.

    Rules:
      - Restrictive language anywhere in the abstract → False (override).
      - 'disseminat\\w+' or bare 'proceedings' + any public signal → True.
      - 'freely available' + an output-noun within 80 chars → True.
    """
    if pd.isna(abstract):
        return False
    text = str(abstract)

    if any(re.search(p, text, re.IGNORECASE) for p in RESTRICTIVE_SIGNALS):
        return False

    has_dissem_or_proc = bool(
        re.search(TIER2_DISSEMINATION_TRIGGER, text, re.IGNORECASE)
        or re.search(TIER2_PROCEEDINGS_TRIGGER, text, re.IGNORECASE)
    )
    if has_dissem_or_proc:
        if any(re.search(p, text, re.IGNORECASE) for p in PUBLIC_SIGNALS):
            return True

    for m in re.finditer(TIER2_FREELY_AVAILABLE, text, re.IGNORECASE):
        s, e = m.span()
        ctx = text[max(0, s - 80):min(len(text), e + 80)]
        if re.search(OUTPUT_NOUNS_PATTERN, ctx, re.IGNORECASE):
            return True

    return False


# =============================================================================
# SCHEMA CONSTANTS (new pipeline output)
# =============================================================================
COL_KEEP        = "s1_keep"               # Stage 1 KEEP/REMOVE
COL_FORMULA     = "is_formula_grant"      # flagged in step2
COL_ORIENTATION = "llm_orientation"       # Stage 2 orientation axis
COL_ABSTRACT    = "abstract"
COL_TITLE       = "title"


# =============================================================================
# FILTERING (Section 1A logic)
# =============================================================================
def filter_to_section1a(df: pd.DataFrame):
    """
    Filter to Section 1A: LLM-kept, categorizable, project (non-formula) grants.

    Returns (filtered_df, exclusion_stats).
    """
    original_count = len(df)

    # --- LLM-kept filter (Stage 1) ---
    if COL_KEEP not in df.columns:
        print(f"⚠️  '{COL_KEEP}' column missing — assuming all rows are kept.")
        llm_kept = pd.Series(True, index=df.index)
    else:
        # s1_keep may be NA for rows that skipped Stage 1 — treat NA as NOT kept.
        llm_kept = df[COL_KEEP].fillna(False).astype(bool)
    excluded_not_kept = int((~llm_kept).sum())

    df = df[llm_kept].copy()

    # --- Categorizable (abstract ≥ 92 chars) ---
    df["abstract_length"] = df[COL_ABSTRACT].fillna("").astype(str).str.len()
    abstract_missing = df[COL_ABSTRACT].isna() | (df[COL_ABSTRACT].fillna("").astype(str).str.strip() == "")
    abstract_short   = (df["abstract_length"] < BENCHMARK_ABSTRACT_LENGTH) & ~abstract_missing

    excluded_missing = int(abstract_missing.sum())
    excluded_short   = int(abstract_short.sum())

    categorizable_mask = ~(abstract_short | abstract_missing)
    df = df[categorizable_mask].copy()

    # --- Project (not formula grants) ---
    if COL_FORMULA in df.columns:
        is_formula = df[COL_FORMULA].fillna(False).astype(bool)
    elif "award_type" in df.columns:
        is_formula = df["award_type"].fillna("").astype(str).str.contains("FORMULA", case=False, na=False)
    else:
        is_formula = pd.Series(False, index=df.index)

    excluded_formula = int(is_formula.sum())
    df = df[~is_formula].copy()

    return df, {
        "original":                   original_count,
        "excluded_not_llm_kept":      excluded_not_kept,
        "excluded_missing_abstract":  excluded_missing,
        "excluded_short_abstract":    excluded_short,
        "excluded_formula":           excluded_formula,
        "final":                      len(df),
    }


# =============================================================================
# KEYWORD DETECTION
# =============================================================================
def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def has_keyword(abstract, keywords: List[str]) -> bool:
    """Case-insensitive, word-boundary match for any keyword in list."""
    if pd.isna(abstract):
        return False
    normalized = normalize_text(abstract)
    for keyword in keywords:
        norm_kw = normalize_text(keyword)
        pattern = r"\b" + re.escape(norm_kw) + r"\b"
        if re.search(pattern, normalized):
            return True
    return False


def find_matching_keywords(abstract, keywords: List[str]) -> List[str]:
    if pd.isna(abstract):
        return []
    normalized = normalize_text(abstract)
    matched = []
    for keyword in keywords:
        norm_kw = normalize_text(keyword)
        pattern = r"\b" + re.escape(norm_kw) + r"\b"
        if re.search(pattern, normalized):
            matched.append(keyword)
    return matched


def verify_flagged_grant(row: pd.Series, keywords: List[str]) -> dict:
    abstract = str(row.get(COL_ABSTRACT, ""))
    abstract_lower = abstract.lower()
    for keyword in keywords:
        if keyword.lower() in abstract_lower:
            idx = abstract_lower.find(keyword.lower())
            context_start = max(0, idx - 100)
            context_end   = min(len(abstract), idx + 150)
            context = abstract[context_start:context_end]
            return {"keyword": keyword, "context": f"...{context}..."}
    return {"keyword": None, "context": "NO KEYWORD FOUND"}


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("ADD KEYWORD FLAGS TO MINING DATA (Section 1A filter)")
    print("=" * 80)
    print()

    if not INFILE.exists():
        print(f"❌ Input file not found: {INFILE}")
        print(f"   Run step3_mining_two_stage_classifier_multiyear.py first.")
        return

    print(f"Loading: {INFILE}")
    df = pd.read_csv(INFILE, low_memory=False)
    print(f"✓ Loaded {len(df):,} rows")
    print()

    # ---- Section 1A filter ----
    print("=" * 80)
    print("APPLYING SECTION 1A FILTER")
    print("=" * 80)
    print()
    print("Filter criteria:")
    print(f"  1. {COL_KEEP} == True           (LLM-kept in Stage 1)")
    print(f"  2. abstract ≥ {BENCHMARK_ABSTRACT_LENGTH} chars    (categorizable)")
    print(f"  3. {COL_FORMULA} == False  (project grants, not formula)")
    print()

    df_filtered, stats = filter_to_section1a(df)

    print("Filtering results:")
    print(f"  Original grants:           {stats['original']:,}")
    print(f"  - Not LLM-kept:            {stats['excluded_not_llm_kept']:,}")
    print(f"  - Missing abstract:        {stats['excluded_missing_abstract']:,}")
    print(f"  - Short abstract (< 92):   {stats['excluded_short_abstract']:,}")
    print(f"  - Formula grants:          {stats['excluded_formula']:,}")
    print(f"  ─────────────────────────────────")
    print(f"  Section 1A grants:         {stats['final']:,}")
    print()

    if stats["final"] == 0:
        print("❌ No Section 1A grants remain after filtering. Nothing to flag.")
        return

    # ---- Required columns ----
    if COL_ABSTRACT not in df_filtered.columns:
        print(f"❌ Missing required column: {COL_ABSTRACT}")
        return

    # ---- Keyword lists ----
    print("=" * 80)
    print("KEYWORD LISTS")
    print("=" * 80)
    print(f"\nIndustry Framing ({len(INDUSTRY_KEYWORDS)} keywords):")
    for i, kw in enumerate(INDUSTRY_KEYWORDS, 1):
        print(f"  {i:2d}. {kw}")
    print(f"\nOpen Access / Sharing ({len(SHARING_KEYWORDS)} keywords):")
    for i, kw in enumerate(SHARING_KEYWORDS, 1):
        print(f"  {i:2d}. {kw}")
    print()

    # ---- Orientation breakdown (if column available) ----
    if COL_ORIENTATION in df_filtered.columns:
        print("Orientation breakdown (Section 1A grants):")
        print(df_filtered[COL_ORIENTATION].value_counts(dropna=False))
        print()

    # ---- Apply keyword flags ----
    print("=" * 80)
    print("PROCESSING KEYWORD FLAGS")
    print("=" * 80)
    print()

    print("1. Industry Framing...")
    df_filtered["industry_framing"] = df_filtered[COL_ABSTRACT].apply(
        lambda x: has_keyword(x, INDUSTRY_KEYWORDS)
    )
    industry_count = int(df_filtered["industry_framing"].sum())
    print(f"   ✓ {industry_count:,} grants flagged with industry-framing keywords")

    print("2. Open Access / Sharing...")
    df_filtered["open_access_sharing"] = df_filtered[COL_ABSTRACT].apply(
        lambda x: has_keyword(x, SHARING_KEYWORDS) or has_open_access_supplement(x)
    )
    sharing_count = int(df_filtered["open_access_sharing"].sum())
    print(f"   ✓ {sharing_count:,} grants flagged with open-access / sharing keywords")
    print()

    # (Interdisciplinary flag intentionally omitted — LLM's llm_research_approach
    # supersedes the keyword-based interdisciplinary flag.)

    # ---- Summary ----
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    total = len(df_filtered)
    print(f"Section 1A grants:          {total:,}")
    print(f"Industry Framing = True:    {industry_count:,}  ({industry_count / total * 100:.1f}%)")
    print(f"Open Access/Sharing = True: {sharing_count:,}  ({sharing_count / total * 100:.1f}%)")
    print()

    if COL_ORIENTATION in df_filtered.columns:
        print("Breakdown by Orientation:")
        print("-" * 80)
        for orientation in df_filtered[COL_ORIENTATION].dropna().unique():
            subset = df_filtered[df_filtered[COL_ORIENTATION] == orientation]
            n = len(subset)
            if n == 0:
                continue
            ind = int(subset["industry_framing"].sum())
            shr = int(subset["open_access_sharing"].sum())
            print(f"\n{orientation}:")
            print(f"  Total grants:        {n:,}")
            print(f"  Industry Framing:    {ind:,}  ({ind / n * 100:.1f}%)")
            print(f"  Open Access/Sharing: {shr:,}  ({shr / n * 100:.1f}%)")
        print()

    # ---- Verification — show a flagged example per category with context ----
    print("=" * 80)
    print("VERIFICATION: first flagged grant per category (with abstract context)")
    print("=" * 80)
    print()

    ind_flagged = df_filtered[df_filtered["industry_framing"]]
    if len(ind_flagged) > 0:
        print("INDUSTRY FRAMING — first flagged grant:")
        print("-" * 80)
        sample = ind_flagged.iloc[0]
        v = verify_flagged_grant(sample, INDUSTRY_KEYWORDS)
        print(f"Title:         {sample.get(COL_TITLE, '(no title)')}")
        print(f"Keyword Found: '{v['keyword']}'")
        print(f"Context:       {v['context']}")
        print()

    shr_flagged = df_filtered[df_filtered["open_access_sharing"]]
    if len(shr_flagged) > 0:
        print("OPEN ACCESS / SHARING — first flagged grant:")
        print("-" * 80)
        sample = shr_flagged.iloc[0]
        v = verify_flagged_grant(sample, SHARING_KEYWORDS)
        print(f"Title:         {sample.get(COL_TITLE, '(no title)')}")
        print(f"Keyword Found: '{v['keyword']}'")
        print(f"Context:       {v['context']}")
        print()

    # ---- Save ----
    print("=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)
    print()
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(OUTFILE, index=False)
    print(f"✓ Saved: {OUTFILE}")
    print(f"  Total rows: {len(df_filtered):,}")
    print(f"  New columns:")
    print(f"    - industry_framing (bool)")
    print(f"    - open_access_sharing (bool)")
    print()

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
