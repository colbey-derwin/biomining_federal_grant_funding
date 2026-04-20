"""
biomining_shared.py
Shared constants, colour palette, category definitions, and helpers.
Imported by both plot scripts — do not run directly.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────────────────
# BASE resolves to the biomining_federal_grant_funding project root.
_SCRIPT_DIR = Path(__file__).resolve().parent        # scripts/
BASE = str(_SCRIPT_DIR.parent)                        # biomining_federal_grant_funding/

# The new pipeline produces a single merged classified CSV. KEPT_CSV and
# ALL_CSV both point at the same file; load_base_data() filters kept
# internally (llm_keep == True). If step5 (keyword flags) has been run,
# prefer the flagged CSV so downstream plots see industry_framing /
# open_access_sharing columns.
_OUTPUT_DIR      = _SCRIPT_DIR / "grant_classifier" / "output"
_CLASSIFIED      = _OUTPUT_DIR / "mining_llm_classified_all_years.csv"
_CLASSIFIED_FLAG = _OUTPUT_DIR / "mining_llm_classified_with_keyword_flags_all_years.csv"
KEPT_CSV = str(_CLASSIFIED_FLAG if _CLASSIFIED_FLAG.exists() else _CLASSIFIED)
ALL_CSV  = str(_CLASSIFIED)

# Section 3 forward-looking federal data — optional external dataset.
# Looks in this project's data/ first, then falls back to the legacy GapMap
# location so the highlights PDF still builds during the transition.
FEDERAL_CSV = str(Path(BASE) / "data" / "federal_gapmap_master_future_updated_v2.csv")
if not Path(FEDERAL_CSV).exists():
    _LEGACY_FED = Path.home() / "Desktop" / "Homeworld" / "Projects" / "GapMap" / "federal_gapmap_master_future_updated_v2.csv"
    if _LEGACY_FED.exists():
        FEDERAL_CSV = str(_LEGACY_FED)

# ── MASTER COLOUR PALETTE ──────────────────────────────────────────────────
# Bio = GREEN always, Non-bio = BLUE always
BIO_COLOR    = "#27AE60"   # green
NONBIO_COLOR = "#2E75B6"   # blue
PUB_COLOR    = "#5B8DD9"   # mid-blue (public good)
IND_COLOR    = "#E67E22"   # orange  (industry facing)

DARK       = "#2C3E50"
LIGHT_GREY = "#ECF0F1"
GREY       = "#95A5A6"
TEAL       = "#1ABC9C"
RED        = "#E74C3C"
GREEN      = "#27AE60"

MINING_TYPE_COLORS = {
    "extraction":            "#2E75B6",   # blue
    "remediation":           "#27AE60",   # green
    "collaborative_tools":   "#8E44AD",   # purple
    "exploration":           "#E67E22",   # orange
    "infrastructure":        "#E74C3C",   # red
    "downstream_processing": "#F39C12",   # gold
    "general":               "#95A5A6",   # grey
}

STAGE_COLORS = {
    "fundamental":                  "#8E44AD",   # purple
    "early_technology_development": "#2E75B6",   # blue
    "applied_translational":        "#27AE60",   # green
    "deployment":                   "#E67E22",   # orange
}

BIO_SUB_COLORS = {
    "bioleaching":                        "#2E75B6",
    "bioseparation_biosorption":          "#1ABC9C",
    "bioremediation":                     "#27AE60",
    "biosensing":                         "#E67E22",
    "biooxidation":                       "#E74C3C",
    "bioprecipitation_mineral_formation": "#8E44AD",
}

MATERIAL_COLORS = {
    "lanthanides_REE":            "#E74C3C",
    "base_metals":                "#2E75B6",
    "battery_metals":             "#27AE60",
    "polymetallic_general":       "#8E44AD",
    "secondary_feedstocks":       "#E67E22",
    "coal_ash_secondary":         "#F39C12",
    "precious_metals":            "#1ABC9C",
    "critical_transition_metals": "#95A5A6",
    "actinides":                  "#C0392B",
    "metalloids_specialty":       "#16A085",
    "platinum_group_metals":      "#D4AC0D",
}

ORIENT_COLORS = {"public_good": PUB_COLOR, "industry_facing": IND_COLOR}

# ── CATEGORY DEFINITIONS (from mining_llm_classifier.py prompt) ────────────
MINING_TYPE_DEFS = {
    "extraction":
        "Getting metal/mineral out of any source — primary ore, waste streams, e-waste, ash, brine. "
        "Includes bioleaching, heap leaching, hydrometallurgy, phytomining. Input = solid/mixed source; "
        "output = metal or metal-bearing solution.",
    "remediation":
        "Directly remediating a mining-impacted site, waste stream, acid mine drainage, or tailings. "
        "Also covers environmental monitoring where mining waste is the explicit cause.",
    "exploration":
        "Locating and characterising subsurface mineral resources — geophysical, geochemical, "
        "and remote sensing approaches.",
    "downstream_processing":
        "Refining or purifying already-extracted material: electrowinning, smelting, flotation of "
        "concentrates, solvent extraction of leachates. Input = extracted concentrate; output = pure metal.",
    "infrastructure":
        "Physical tools, instruments, assays, sensors, equipment, software, and databases built to "
        "enable mining operations or research.",
    "collaborative_tools":
        "Conferences, workshops, hubs, regional innovation centres, workforce development programmes, "
        "policy studies, economic or sociological studies about mining.",
    "general":
        "Grant spans the full mining value chain (exploration + extraction + processing + remediation) "
        "with no dominant focus. Used sparingly.",
}

STAGE_DEFS = {
    "fundamental":
        "Basic science, understanding mechanisms, characterising organisms/materials. "
        "No immediate application target.",
    "early_technology_development":
        "Proof-of-concept, lab-scale demonstrations, novel process proposals.",
    "applied_translational":
        "Optimisation of known approaches, pilot-scale testing, techno-economic analysis, scale-up.",
    "deployment":
        "Near-commercial or commercial-scale implementation, field demonstrations, technology transfer.",
}

BIO_SUB_DEFS = {
    "bioleaching":
        "Microbial oxidation of sulphide minerals to solubilise metals for recovery "
        "(e.g., Acidithiobacillus leaching chalcopyrite for copper).",
    "bioseparation_biosorption":
        "Biological or bio-inspired molecules/organisms that selectively bind, adsorb, or separate "
        "metals — including ligand design for REE selectivity and protein-based metal chelation.",
    "bioremediation":
        "Microorganisms or plants to detoxify/remove contaminants from mining-impacted environments. "
        "Focus is ecological restoration; also includes siderophore production and microbial metal cycling.",
    "biosensing":
        "Biological tools for sensing metals, redox state, or analytes in mining streams/environments. "
        "Includes molecular assays for process control and detection technology development.",
    "biooxidation":
        "Microbial pre-treatment of refractory gold ores to liberate gold prior to cyanidation.",
    "bioprecipitation_mineral_formation":
        "Controlled microbial precipitation of metals as minerals "
        "(e.g., sulphate-reducing bacteria precipitating metal sulphides). Emphasis on product formation.",
}

MATERIAL_DEFS = {
    "lanthanides_REE":            "Rare earth elements: Nd, Dy, Tb, La, Ce, Y, Sc and all REEs.",
    "base_metals":                "Cu, Zn, Ni, Pb, Fe, Al.",
    "battery_metals":             "Li, Co, Ni, Mn (Ni may also appear in base_metals).",
    "polymetallic_general":       "Multiple metals without dominant focus; 'critical minerals'; 'strategic metals'.",
    "secondary_feedstocks":       "Recovery from e-waste, spent batteries, industrial by-products, "
                                  "process streams, or recycling — any secondary/waste-derived source.",
    "coal_ash_secondary":         "Recovery from coal ash or tailings specifically.",
    "precious_metals":            "Au, Ag.",
    "critical_transition_metals": "W, Mo, V, Cr, Ti.",
    "actinides":                  "U, Th.",
    "metalloids_specialty":       "Si, Ge, Ga, Te, In, As.",
    "platinum_group_metals":      "Pt, Pd, Rh, Ir, Ru, Os.",
}

# ── MATPLOTLIB DEFAULTS ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

# ── HELPERS ────────────────────────────────────────────────────────────────
def fmt_millions(x, pos=None):
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def save_fig(fig, name, pdf, out_dir):
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def parse_materials(s):
    if pd.isna(s) or str(s).strip() in ("", "[]"):
        return []
    try:
        return json.loads(s)
    except Exception:
        return [x.strip().strip("\"'") for x in str(s).strip("[]").split(",") if x.strip()]


# ── SCHEMA ADAPTER ─────────────────────────────────────────────────────────
# The new bulk-download pipeline renames some columns vs. the old API-pull
# pipeline. Rename at load time so every downstream consumer (plots, tables,
# calibration, PDFs) keeps seeing the names it was written against.
#     new_pipeline_name  →  old_pipeline_name
_SCHEMA_MAP = {
    "unique_key":    "unique_award_key",
    "s1_keep":       "llm_keep",
    "award_amount":  "amt",
    "start_date":    "startDate",
    "end_date":      "expDate",
    "funder":        "funder",        # same in both
}

# Alias columns that should be DUPLICATED (kept under both names) rather than
# renamed — used when downstream consumers of the shared loader expect the
# legacy name but we want the new name preserved for other consumers.
_SCHEMA_ALIASES = {
    "llm_stage":           "llm_research_stage",
    "llm_bio_subcategory": "llm_biological_subcategory",
}


def _adapt_schema(df):
    """Rename new-pipeline columns to old-pipeline names if old names are missing.
    Also duplicate a small set of columns under an alias so code expecting either
    the new or the old name keeps working."""
    rename = {old: new for old, new in _SCHEMA_MAP.items()
              if old in df.columns and new not in df.columns and old != new}
    if rename:
        df = df.rename(columns=rename)
    for src, dst in _SCHEMA_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    return df


def _coerce_bool(series):
    """Coerce mixed string/bool column to a clean bool series."""
    def _one(v):
        if pd.isna(v):                   return False
        if isinstance(v, (bool, np.bool_)): return bool(v)
        return str(v).strip().lower() in ("true", "1", "yes")
    return series.map(_one)


def load_base_data():
    """Load and parse the classified CSV; return (kept, alldf).

    In the new pipeline KEPT_CSV and ALL_CSV point at the same file, so we
    load once and filter locally to keep the behaviour of the old two-file
    loader identical downstream.
    """
    raw = pd.read_csv(ALL_CSV, low_memory=False)
    raw = _adapt_schema(raw)

    # Normalise llm_keep to bool regardless of how CSV stored it.
    if "llm_keep" in raw.columns:
        raw["llm_keep"] = _coerce_bool(raw["llm_keep"])
    else:
        raw["llm_keep"] = True  # fall back to "everything kept" if column missing

    alldf = raw.copy()
    kept  = raw.loc[raw["llm_keep"]].copy()

    for df in [kept, alldf]:
        df["startDate"] = pd.to_datetime(df.get("startDate", pd.NaT), errors="coerce")
        df["year"]      = pd.to_numeric(df.get("year", np.nan), errors="coerce")
        df.loc[df["year"].isna(), "year"] = df.loc[df["year"].isna(), "startDate"].dt.year
        df["year"]      = df["year"].astype("Int64")
        df["amt"]       = pd.to_numeric(df.get("amt", np.nan), errors="coerce")

    kept["materials_list"] = kept.get("llm_materials", pd.Series([], dtype=object)).apply(parse_materials)
    kept["is_bio"]         = _coerce_bool(kept.get("llm_biological", pd.Series([], dtype=object)))
    return kept, alldf


def build_matdf(kept):
    """Explode materials with split funding (no double-counting)."""
    rows = []
    for _, row in kept.iterrows():
        mats = row["materials_list"]
        if not mats:
            continue
        split = row["amt"] / len(mats) if pd.notna(row["amt"]) else np.nan
        for mat in mats:
            rows.append(dict(
                material=mat, amt=split, full_amt=row["amt"],
                is_bio=row["is_bio"], year=row["year"],
                llm_mining_type=row["llm_mining_type"],
                llm_stage=row["llm_stage"],
                llm_orientation=row["llm_orientation"],
                is_outlier=row.get("is_outlier", False),
            ))
    df = pd.DataFrame(rows)
    # Legacy column alias — some PDF slides read `mat_category` (pre-rename).
    if not df.empty and "mat_category" not in df.columns:
        df["mat_category"] = df["material"]
    return df


def cover_page(pdf, title_line2, kept, outlier_note):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK); ax.axis("off")
    ax.text(0.5, 0.68, "BIOMINING GRANT\nGAP ANALYSIS",
            ha="center", va="center", color="white",
            fontsize=36, fontweight="bold", transform=ax.transAxes, linespacing=1.4)
    ax.text(0.5, 0.50, title_line2,
            ha="center", va="center", color=TEAL,
            fontsize=16, transform=ax.transAxes, linespacing=1.5)
    pub_n = (kept["llm_orientation"] == "public_good").sum()
    ind_n = (kept["llm_orientation"] == "industry_facing").sum()
    ax.text(0.5, 0.32,
            f"n={len(kept):,} grants  |  ${kept['amt'].sum()/1e6:.0f}M total  |  "
            f"{kept['is_bio'].sum()} bio ({kept['is_bio'].mean()*100:.0f}%)  |  "
            f"{pub_n} public / {ind_n} industry",
            ha="center", va="center", color="#BDC3C7",
            fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.18, outlier_note,
            ha="center", va="center", color="#7F8C8D",
            fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)
