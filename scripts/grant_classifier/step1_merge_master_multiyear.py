#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Multi-Year Merge Script for Grant Data (2016-2025)

Strategy:
1. For each year (2016-2025):
   - Process NSF grants: metadata from JSON + funding from USASpending
   - Process non-NSF grants: from USASpending only
   - Combine NSF + non-NSF for that year
2. Combine all years into single master file with year column

Output: Single unified dataset with all agencies and all years
"""

import os
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
# NEW:
# Get the project root directory
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/grant_classifier/
PROJECT_ROOT = SCRIPT_DIR.parent.parent       # climate_biotech_federal_grant_funding/

# Data and output directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "grant_classifier" / "output"
OUTPUT_FILE = OUTPUT_DIR / "merged_all_years.csv"

# Years to process
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Create output directory if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MASTER MULTI-YEAR MERGE: NSF JSONs + USASpending (2016-2025)")
print("="*80)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def clean_whitespace(s: str) -> str:
    """Normalize whitespace in text fields."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def first_nonempty(series):
    """Get first non-empty value from series."""
    for val in series:
        if pd.notna(val) and (not isinstance(val, str) or val.strip()):
            return val
    return pd.NA

# =============================================================================
# YEAR PROCESSING FUNCTION
# =============================================================================
def process_year(year):
    """Process a single year of data. Returns combined DataFrame."""
    print(f"\n{'='*80}")
    print(f"PROCESSING YEAR: {year}")
    print("="*80)
    
    # NEW:
    nsf_json_dir = DATA_DIR / str(year) / f"NSF{year}"
    usaspending_csv = DATA_DIR / str(year) / f"USASpending{year}.csv"
    
    # Check files exist
    if not nsf_json_dir.exists():
        print(f"⚠️  NSF directory not found: {nsf_json_dir}")
        print(f"   Skipping {year}")
        return None
    
    if not usaspending_csv.exists():
        print(f"⚠️  USASpending file not found: {usaspending_csv}")
        print(f"   Skipping {year}")
        return None
    
    # =========================================================================
    # PART 1: LOAD NSF JSON METADATA
    # =========================================================================
    print(f"\nLoading NSF JSON files for {year}...")
    json_files = list(Path(nsf_json_dir).glob("*.json"))
    print(f"Found {len(json_files):,} JSON files")
    
    nsf_metadata = {}
    
    for json_file in tqdm(json_files, desc=f"NSF {year} JSONs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            award_id = str(data.get("awd_id", ""))
            if not award_id:
                continue
            
            # Extract POR (Project Outcomes Report) text if available
            por_text = ""
            if "por" in data and data["por"]:
                por_cntn = data["por"].get("por_cntn", "")
                por_txt = data["por"].get("por_txt_cntn", "")
                por_text = por_txt if por_txt else por_cntn
            
            # Get PI info
            pi_name = ""
            pi_email = ""
            if "pi" in data and data["pi"] and len(data["pi"]) > 0:
                pi_name = data["pi"][0].get("pi_full_name", "")
                pi_email = data["pi"][0].get("pi_email_addr", "")
            
            # Get institution info
            inst_name = ""
            inst_city = ""
            inst_state = ""
            if "inst" in data and data["inst"]:
                inst_name = data["inst"].get("inst_name", "")
                inst_city = data["inst"].get("inst_city_name", "")
                inst_state = data["inst"].get("inst_state_code", "")
            
            nsf_metadata[award_id] = {
                'award_id': award_id,
                'title': clean_whitespace(data.get('awd_titl_txt', '')),
                'abstract': clean_whitespace(data.get('awd_abstract_narration', '')),
                'por_text': clean_whitespace(por_text),
                'start_date': data.get('awd_eff_date', ''),
                'end_date': data.get('awd_exp_date', ''),
                'pi_name': pi_name,
                'pi_email': pi_email,
                'institution': inst_name,
                'inst_city': inst_city,
                'inst_state': inst_state,
                'directorate': data.get('dir_abbr', ''),
                'division': data.get('div_abbr', ''),
                'program': data.get('org_code', ''),
                'award_type': data.get('awd_istr_txt', ''),
            }
            
        except Exception as e:
            print(f"\nError loading {json_file.name}: {e}")
            continue
    
    print(f"Loaded metadata for {len(nsf_metadata):,} NSF grants")
    
    # =========================================================================
    # PART 2: LOAD USASPENDING DATA
    # =========================================================================
    print(f"\nLoading USASpending CSV for {year}...")
    df = pd.read_csv(usaspending_csv, low_memory=False)
    print(f"Loaded {len(df):,} total USASpending rows")
    
    # =========================================================================
    # PART 3: PROCESS NSF GRANTS FROM USASPENDING
    # =========================================================================
    print(f"\nProcessing NSF grants from USASpending for {year}...")
    
    # Identify NSF grants
    is_nsf = (
        df["awarding_agency_name"].fillna("").str.upper().str.contains("NATIONAL SCIENCE FOUNDATION|\\bNSF\\b", regex=True) |
        df["funding_agency_name"].fillna("").str.upper().str.contains("NATIONAL SCIENCE FOUNDATION|\\bNSF\\b", regex=True)
    )
    
    nsf_usa = df[is_nsf].copy()
    print(f"Found {len(nsf_usa):,} NSF transactions in USASpending")
    
    # Extract NSF award ID (7-digit number)
    nsf_usa['award_id'] = nsf_usa['award_id_fain'].astype(str).str.extract(r'(\d{7})', expand=False)
    
    # Remove rows without valid award ID
    nsf_usa = nsf_usa[nsf_usa['award_id'].notna()].copy()
    print(f"  {len(nsf_usa):,} transactions with valid NSF award IDs")
    
    # Group by award_id and sum ALL transaction types (A+B+C+D)
    print(f"  Grouping NSF transactions by award_id (summing ALL types A+B+C+D)...")
    
    nsf_grouped = nsf_usa.groupby('award_id').agg({
        'federal_action_obligation': 'sum',
        'period_of_performance_start_date': 'first',
        'period_of_performance_current_end_date': 'first',
        'transaction_description': 'first',
        'recipient_name': 'first',
        'recipient_city_name': 'first',
        'recipient_state_code': 'first',
    }).reset_index()
    
    print(f"  Grouped into {len(nsf_grouped):,} unique NSF grants")
    print(f"  Total NSF funding from USASpending: ${nsf_grouped['federal_action_obligation'].sum():,.0f}")
    
    # =========================================================================
    # PART 4: MERGE NSF METADATA + USASPENDING AMOUNTS
    # =========================================================================
    print(f"\nMerging NSF metadata with USASpending amounts for {year}...")
    
    # Convert NSF metadata to DataFrame
    nsf_meta_df = pd.DataFrame.from_dict(nsf_metadata, orient='index').reset_index(drop=True)
    
    # Merge NSF metadata with USASpending amounts
    nsf_merged = pd.merge(
        nsf_meta_df,
        nsf_grouped[['award_id', 'federal_action_obligation']],
        on='award_id',
        how='outer',
        indicator=True
    )
    
    # Handle 3 cases for NSF grants
    print(f"  NSF Merge Results:")
    print(f"    In both (JSON + USASpending): {(nsf_merged['_merge'] == 'both').sum():,}")
    print(f"    Only in JSON: {(nsf_merged['_merge'] == 'right_only').sum():,}")
    print(f"    Only in USASpending: {(nsf_merged['_merge'] == 'left_only').sum():,}")
    
    # Create final NSF dataset
    nsf_final = nsf_merged.copy()
    nsf_final['source'] = 'NSF'
    nsf_final['funder'] = 'National Science Foundation'
    nsf_final['year'] = year
    
    # Use USASpending amount if available, otherwise use 0 (for JSON-only grants)
    nsf_final['award_amount'] = nsf_final['federal_action_obligation'].fillna(0)
    
    # Clean up
    nsf_final = nsf_final.drop(['federal_action_obligation', '_merge'], axis=1)
    
    print(f"  Final NSF dataset: {len(nsf_final):,} grants")
    print(f"  Total NSF funding: ${nsf_final['award_amount'].sum():,.0f}")
    
    # =========================================================================
    # PART 5: PROCESS NON-NSF AGENCIES FROM USASPENDING
    # =========================================================================
    print(f"\nProcessing non-NSF agencies from USASpending for {year}...")
    
    # Get all NON-NSF grants
    non_nsf = df[~is_nsf].copy()
    print(f"Found {len(non_nsf):,} non-NSF transactions")
    
    # Standardize schema
    non_nsf_standardized = pd.DataFrame({
        'source': 'USASpending',
        'award_id': non_nsf['award_id_fain'].fillna(non_nsf['award_id_uri']),
        'title': non_nsf['transaction_description'].fillna(
            non_nsf['prime_award_base_transaction_description'].fillna(
                non_nsf['cfda_title'].fillna(non_nsf['award_id_fain'])
            )
        ).apply(clean_whitespace),
        'abstract': non_nsf['funding_opportunity_goals_text'].fillna(
            non_nsf['transaction_description'].fillna("")
        ).apply(clean_whitespace),
        'por_text': '',
        'start_date': non_nsf['period_of_performance_start_date'],
        'end_date': non_nsf['period_of_performance_current_end_date'],
        'award_amount': pd.to_numeric(non_nsf['federal_action_obligation'], errors='coerce'),
        'pi_name': non_nsf['recipient_name'],
        'pi_email': '',
        'institution': non_nsf['recipient_name'],
        'inst_city': non_nsf['recipient_city_name'],
        'inst_state': non_nsf['recipient_state_code'],
        'directorate': '',
        'division': '',
        'program': non_nsf['cfda_number'],
        'award_type': non_nsf['assistance_type_description'],
        'funder': non_nsf['awarding_agency_name'].fillna(non_nsf['funding_agency_name']),
        'year': year,
    })
    
    # Group by award_id and sum amounts
    print(f"  Grouping non-NSF transactions by award_id...")
    
    agg_dict = {
        'source': 'first',
        'title': first_nonempty,
        'abstract': first_nonempty,
        'por_text': 'first',
        'start_date': 'first',
        'end_date': 'first',
        'award_amount': 'sum',
        'pi_name': first_nonempty,
        'pi_email': 'first',
        'institution': first_nonempty,
        'inst_city': first_nonempty,
        'inst_state': first_nonempty,
        'directorate': 'first',
        'division': 'first',
        'program': first_nonempty,
        'award_type': first_nonempty,
        'funder': first_nonempty,
        'year': 'first',
    }
    
    non_nsf_grouped = non_nsf_standardized.groupby('award_id', as_index=False).agg(agg_dict)
    
    print(f"  Grouped into {len(non_nsf_grouped):,} unique non-NSF grants")
    print(f"  Total non-NSF funding: ${non_nsf_grouped['award_amount'].sum():,.0f}")
    
    # =========================================================================
    # PART 6: COMBINE NSF + NON-NSF FOR THIS YEAR
    # =========================================================================
    print(f"\nCombining NSF + non-NSF for {year}...")
    
    # Ensure both have same columns
    all_cols = ['source', 'award_id', 'title', 'abstract', 'por_text', 
                'start_date', 'end_date', 'award_amount', 'pi_name', 'pi_email',
                'institution', 'inst_city', 'inst_state', 'directorate', 'division',
                'program', 'award_type', 'funder', 'year']
    
    # Add missing columns to NSF if needed
    for col in all_cols:
        if col not in nsf_final.columns:
            nsf_final[col] = ''
    
    # Reorder columns
    nsf_final = nsf_final[all_cols]
    non_nsf_grouped = non_nsf_grouped[all_cols]
    
    # Combine
    year_data = pd.concat([nsf_final, non_nsf_grouped], ignore_index=True)
    
    print(f"\n{year} SUMMARY:")
    print(f"  Total grants: {len(year_data):,}")
    print(f"  Total funding: ${year_data['award_amount'].sum():,.0f}")
    print(f"  NSF grants: {len(nsf_final):,}")
    print(f"  Non-NSF grants: {len(non_nsf_grouped):,}")
    
    return year_data

# =============================================================================
# MAIN: PROCESS ALL YEARS AND COMBINE
# =============================================================================
def main():
    all_years_data = []
    
    for year in YEARS:
        year_df = process_year(year)
        if year_df is not None:
            all_years_data.append(year_df)
    
    if not all_years_data:
        print("\n❌ No data processed! Check file paths.")
        return
    
    # =========================================================================
    # COMBINE ALL YEARS
    # =========================================================================
    print(f"\n{'='*80}")
    print("COMBINING ALL YEARS")
    print("="*80)
    
    master = pd.concat(all_years_data, ignore_index=True)
    
    # Create unique key
    master['unique_key'] = master['source'] + '::' + master['award_id'].astype(str)
    
    # Convert dates
    master['start_date'] = pd.to_datetime(master['start_date'], errors='coerce')
    master['end_date'] = pd.to_datetime(master['end_date'], errors='coerce')
    
    # Save
    master.to_csv(OUTPUT_FILE, index=False)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n✅ Master file saved: {OUTPUT_FILE}")
    print(f"\n{'='*80}")
    print("MULTI-YEAR MERGE COMPLETE")
    print("="*80)
    
    print(f"\nTotal grants (all years): {len(master):,}")
    print(f"Total funding (all years): ${master['award_amount'].sum():,.0f}")
    
    print(f"\nBreakdown by year:")
    year_summary = master.groupby('year').agg({
        'award_id': 'count',
        'award_amount': 'sum'
    }).rename(columns={'award_id': 'grants', 'award_amount': 'funding'})
    for year, row in year_summary.iterrows():
        print(f"  {year}: {int(row['grants']):,} grants (${int(row['funding']):,.0f})")
    
    print(f"\nBreakdown by source:")
    print(master['source'].value_counts())
    
    print(f"\nTop 10 funders (all years):")
    print(master['funder'].value_counts().head(10))
    
    print(f"\nNSF grants breakdown:")
    nsf_only = master[master['source'] == 'NSF']
    print(f"  Total NSF grants: {len(nsf_only):,}")
    print(f"  Total NSF funding: ${nsf_only['award_amount'].sum():,.0f}")
    print(f"  With USASpending amounts: {(nsf_only['award_amount'] > 0).sum():,}")
    print(f"  Without USASpending amounts: {(nsf_only['award_amount'] == 0).sum():,}")
    
    print("\n" + "="*80)
    print("NEXT STEP: Run climate_biotech_filter_multiyear.py on this master file")
    print("="*80)

if __name__ == "__main__":
    main()
