#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_insufficient_abstracts_multiyear.py
===========================================
Analyze grants with INSUFFICIENT ABSTRACTS (<92 characters) by funding agency.
Shows which agencies had biomining grants excluded due to short abstracts.
Compares to total kept grants to show % of biomining removed.

Adapted for the biomining pipeline from the climate-biotech version:
  /Users/colbeyderwin/Desktop/climate_biotech_federal_grant_funding/scripts/
  analyze_insufficient_abstracts_multiyear.py

Threshold changed from 150 → 92 chars to match step2's SHORT_THRESHOLD.

Processes ALL YEARS (2016-2025) from the combined datasets.
"""

import pandas as pd
from pathlib import Path

# =========================
# PATHS
# =========================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR   = PROJECT_ROOT / "scripts" / "grant_classifier" / "output"

KEPT_FILE         = OUTPUT_DIR / "mining_filtered_all_years.csv"
INSUFFICIENT_FILE = OUTPUT_DIR / "mining_insufficient_abstract_all_years.csv"

SHORT_THRESHOLD = 92  # must match build_viz_data.py / step2

print("=" * 70)
print("INSUFFICIENT ABSTRACT ANALYSIS BY AGENCY (ALL YEARS)")
print(f"Grants that passed the mining keyword filter but have <{SHORT_THRESHOLD} char abstracts")
print("=" * 70)

print("\nLoading data...")
try:
    kept = pd.read_csv(KEPT_FILE, low_memory=False)
    print(f"Loaded {len(kept):,} filter-kept grants (sufficient abstracts)")
except FileNotFoundError:
    print(f"\n⚠️  File not found: {KEPT_FILE}")
    print("Run step2_mining_filter_multiyear.py first to generate this file.")
    raise SystemExit(1)

try:
    insuf = pd.read_csv(INSUFFICIENT_FILE, low_memory=False)
    print(f"Loaded {len(insuf):,} grants with insufficient abstracts")
except FileNotFoundError:
    print(f"\n⚠️  File not found: {INSUFFICIENT_FILE}")
    print("Run step2_mining_filter_multiyear.py first to generate this file.")
    raise SystemExit(1)

# Determine agency column
agency_col = next((c for c in ("funder", "funding_agency", "awarding_agency") if c in insuf.columns), None)
if agency_col is None:
    print("\n⚠️  Cannot find agency column!")
    print("Available columns:", insuf.columns.tolist()[:20])
    raise SystemExit(1)
print(f"Using agency column: '{agency_col}'")

amount_col = next((c for c in ("award_amount", "amount", "amt") if c in insuf.columns), None)
if amount_col is None:
    print("\n⚠️  Cannot find amount column!")
    raise SystemExit(1)
print(f"Using amount column: '{amount_col}'")

# Normalize numeric columns so downstream arithmetic is safe.
for df_x in (kept, insuf):
    df_x[amount_col] = pd.to_numeric(df_x[amount_col], errors="coerce").fillna(0)
    if "abstract_length" not in df_x.columns and "abstract" in df_x.columns:
        df_x["abstract_length"] = df_x["abstract"].fillna("").astype(str).str.len()

# =========================
# OVERALL STATISTICS
# =========================
print("\n" + "=" * 70)
print("OVERALL STATISTICS (ALL YEARS)")
print("=" * 70)

total_biomining        = len(kept) + len(insuf)
total_biomining_funding = kept[amount_col].sum() + insuf[amount_col].sum()
insuf_funding          = insuf[amount_col].sum()
avg_abs_len            = insuf.get("abstract_length", pd.Series([0])).mean()

print(f"\nTOTAL MINING-RELEVANT GRANTS (post-keyword filter):")
print(f"  {total_biomining:,} grants")
print(f"  ${total_biomining_funding:,.0f}")

print(f"\nKept (sufficient abstracts ≥{SHORT_THRESHOLD} chars):")
print(f"  {len(kept):,} grants ({len(kept)/total_biomining*100:.1f}%)")
print(f"  ${kept[amount_col].sum():,.0f} ({kept[amount_col].sum()/total_biomining_funding*100:.1f}%)")

print(f"\nRemoved (insufficient abstracts <{SHORT_THRESHOLD} chars):")
print(f"  {len(insuf):,} grants ({len(insuf)/total_biomining*100:.1f}%)")
print(f"  ${insuf_funding:,.0f} ({insuf_funding/total_biomining_funding*100:.1f}%)")
print(f"  Average abstract length: {avg_abs_len:.1f} characters")

print(f"\n📊 IMPACT: {len(insuf)/total_biomining*100:.1f}% of mining-relevant grants")
print(f"           {insuf_funding/total_biomining_funding*100:.1f}% of mining-relevant funding")
print(f"           REMOVED due to insufficient abstract length")

if "abstract_length" in insuf.columns:
    print("\nAbstract length distribution (insufficient):")
    print(f"  0 chars (missing):   {(insuf['abstract_length'] == 0).sum():,}")
    print(f"  1-30 chars:          {((insuf['abstract_length'] > 0) & (insuf['abstract_length'] <= 30)).sum():,}")
    print(f"  31-60 chars:         {((insuf['abstract_length'] > 30) & (insuf['abstract_length'] <= 60)).sum():,}")
    print(f"  61-{SHORT_THRESHOLD - 1} chars:        {((insuf['abstract_length'] > 60) & (insuf['abstract_length'] < SHORT_THRESHOLD)).sum():,}")

# =========================
# BY YEAR
# =========================
if "year" in insuf.columns and "year" in kept.columns:
    print("\n" + "=" * 70)
    print("INSUFFICIENT ABSTRACTS BY YEAR")
    print("=" * 70)

    id_col = next((c for c in ("award_id", "unique_key") if c in insuf.columns), None)
    if id_col is not None:
        year_insuf = insuf.groupby("year").agg(
            insuf_count=(id_col, "count"),
            insuf_funding=(amount_col, "sum"),
        )
        year_kept = kept.groupby("year").agg(
            kept_count=(id_col, "count"),
            kept_funding=(amount_col, "sum"),
        )
        yr = year_insuf.join(year_kept, how="outer").fillna(0)
        yr["total_count"] = yr["insuf_count"] + yr["kept_count"]
        yr["pct_insuf"]   = yr["insuf_count"] / yr["total_count"].replace(0, 1) * 100

        print(f"\n{'Year':<6} {'Kept':>8} {'Insuf':>8} {'Total':>8} {'% Insuf':>9}")
        print("-" * 45)
        for year, row in yr.iterrows():
            print(f"{int(year):<6} {int(row['kept_count']):>8,} {int(row['insuf_count']):>8,} "
                  f"{int(row['total_count']):>8,} {row['pct_insuf']:>8.1f}%")

# =========================
# BY AGENCY
# =========================
print("\n" + "=" * 70)
print("INSUFFICIENT ABSTRACTS BY AGENCY")
print("Compared to each agency's total kept mining-relevant grants")
print("=" * 70)

id_col = next((c for c in ("award_id", "unique_key") if c in insuf.columns), None)
kept_by_agency = kept.groupby(agency_col).agg(
    kept_count=(id_col, "count"),
    kept_funding=(amount_col, "sum"),
)
insuf_by_agency = insuf.groupby(agency_col).agg(
    insuf_count=(id_col, "count"),
    insuf_funding=(amount_col, "sum"),
)
comparison = insuf_by_agency.join(kept_by_agency, how="left").fillna(0)
comparison["total_funding"]           = comparison["insuf_funding"] + comparison["kept_funding"]
comparison["pct_of_agency_funding"]   = (comparison["insuf_funding"] / comparison["total_funding"].replace(0, 1) * 100)
comparison["pct_of_all_insufficient"] = (comparison["insuf_funding"] / (insuf_funding or 1) * 100)
comparison = comparison.sort_values("insuf_funding", ascending=False)

print(f"\n{'Agency':<45} {'Insuf':>7} {'Insuf $':>15} {'% All Insuf':>12} {'% of Agency':>12}")
print("-" * 95)
for agency, row in comparison.head(20).iterrows():
    print(f"{str(agency)[:45]:<45} "
          f"{int(row['insuf_count']):>7,} "
          f"${int(row['insuf_funding']):>14,.0f} "
          f"{row['pct_of_all_insufficient']:>11.1f}% "
          f"{row['pct_of_agency_funding']:>11.1f}%")

# =========================
# DOE, DOD, DOI DEEP-DIVE
# =========================
print("\n" + "=" * 70)
print("DOE / DOD / DOI DETAILED BREAKDOWN")
print("=" * 70)

for pat in ["Department of Energy", "Department of Defense", "Department of the Interior"]:
    mask = insuf[agency_col].astype(str).str.contains(pat, case=False, na=False)
    ag_insuf = insuf.loc[mask]
    if len(ag_insuf) == 0:
        continue
    ag_fund = ag_insuf[amount_col].sum()
    avg_len = ag_insuf.get("abstract_length", pd.Series([0])).mean()

    kept_mask = kept[agency_col].astype(str).str.contains(pat, case=False, na=False)
    dept_kept = kept.loc[kept_mask]
    dept_total_count   = len(dept_kept) + len(ag_insuf)
    dept_total_funding = dept_kept[amount_col].sum() + ag_fund
    pct_grants  = (len(ag_insuf) / dept_total_count * 100) if dept_total_count else 0
    pct_funding = (ag_fund / dept_total_funding * 100) if dept_total_funding else 0

    print(f"\n{pat}:")
    print(f"  Insufficient abstract grants: {len(ag_insuf):,}")
    print(f"  Insufficient abstract funding: ${ag_fund:,.0f}")
    print(f"  % of {pat} grants w/ short abstracts: {pct_grants:.1f}%")
    print(f"  % of {pat} funding w/ short abstracts: {pct_funding:.1f}%")
    print(f"  ({len(ag_insuf):,} of {dept_total_count:,} total {pat} grants)")
    print(f"  Average abstract length: {avg_len:.1f} chars")

    print(f"\n  Sample grants:")
    show_id = id_col or "unique_key"
    for _, row in ag_insuf.head(3).iterrows():
        al = row.get("abstract_length", 0)
        ap = str(row.get("abstract", ""))[:60]
        yr = row.get("year", "N/A")
        print(f"    [{yr}] {row[show_id]}: {str(row.get('title',''))[:60]}...")
        print(f"      Abstract ({al} chars): {ap}...")

# =========================
# FORMULA SPLIT (biomining-specific)
# =========================
if "is_formula_grant" in insuf.columns:
    print("\n" + "=" * 70)
    print("FORMULA vs. NON-FORMULA SPLIT IN THE SHORT-ABSTRACT POOL")
    print("=" * 70)
    is_f = insuf["is_formula_grant"].astype(str).str.lower().isin(["true", "1", "yes", "t"])
    print(f"  Formula (short abstract):     {int(is_f.sum()):>4,}  ${insuf.loc[is_f, amount_col].sum():>14,.0f}")
    print(f"  Non-formula (short abstract): {int((~is_f).sum()):>4,}  ${insuf.loc[~is_f, amount_col].sum():>14,.0f}")
    print("  → Formula grants with short abstracts should be counted under the")
    print("    Formula bucket (see build_viz_data.py). Non-formula rows populate")
    print("    the Non-Categorizable Sankey node.")

# =========================
# KEEP REASON BREAKDOWN
# =========================
if "filter_reason" in insuf.columns:
    print("\n" + "=" * 70)
    print("WHY WERE THESE KEPT? (Despite short abstracts)")
    print("=" * 70)
    print("\nKeep-reason breakdown:")
    for reason, count in insuf["filter_reason"].value_counts().items():
        amt = insuf.loc[insuf["filter_reason"] == reason, amount_col].sum()
        pct = count / len(insuf) * 100
        print(f"  {reason}: {count:,} grants ({pct:.1f}%) - ${amt:,.0f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
