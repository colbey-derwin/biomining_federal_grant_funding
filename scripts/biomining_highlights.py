"""
biomining_highlights_v8.py  —  Professional meeting-ready highlights PDF

SECTION 1 — FEDERAL RESEARCH & INNOVATION GRANTS (2016–2025)
  Cover       Grant types listed; all stats as % of funding except n=
  Page 1      Portfolio overview: bio/non-bio donut, mining type $, stage $  (no stats table)
  Page 2      Timeseries all grants — bio vs non-bio (linear)
  Page 3      Timeseries deployment-excluded — bio vs non-bio (linear)
  Page 4      Stage relative abundance by year + Stage definitions sidebar
  Page 5      Mining type relative abundance by year + Mining type definitions sidebar
  Page 6      Bio subcategory: vertical funding bars (% labels) + definitions sidebar
  Page 7      Material type: vertical stacked bars bio-bottom/non-bio-top + definitions sidebar

SECTION 2 — FORMULA GRANTS
  Divider     Concise one-paragraph context + key stats
  Page 8      Bio vs Non-Bio bar + Mining type relative abundance + Stage relative abundance

SECTION 3 — FEDERAL FUTURE FUNDING LANDSCAPE
  Divider     Context paragraph + key stats
  Page 9      HIERARCHICAL FUNNEL: All CM $ → Mining → Grants → Public → Bio
  Page 10     BIO GRANTS: Sliding stage bars (thickness = funding) + Mining type breakdown
  Page 11     Early stage grants table
"""

import os, sys, textwrap
import numpy as np
import pandas as pd
import matplotlib
# Many labels in this PDF include literal `$` for dollar amounts. matplotlib
# parses `$...$` as mathtext by default, which crashes on strings like
# "$30.8B  (... of prev $)". Disable math parsing globally.
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# ── ROBUST PATH SETUP FOR MULTIPLE EXECUTION CONTEXTS ────────────────────────
# This handles: regular Python, Jupyter %runfile, IPython, and __main__ contexts

def setup_import_path():
    """Add the script directory to Python path for biomining_shared import."""
    # Try multiple methods to find the script directory
    script_dir = None
    
    # Method 1: __file__ (works in normal Python execution)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    
    # Method 2: Check if we're in IPython/Jupyter
    if script_dir is None:
        try:
            # Get IPython instance if available
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                # Try to get the working directory
                script_dir = os.getcwd()
        except:
            pass
    
    # Method 3: Use current working directory as fallback
    if script_dir is None:
        script_dir = os.getcwd()
    
    # biomining_shared.py lives alongside this script (scripts/biomining_shared.py).
    # Fall back to the legacy GapMap copy only if the expected one is missing.
    # The legacy copy shadows the new one if it appears earlier on sys.path and
    # breaks downstream imports (e.g., FEDERAL_CSV).
    paths_to_add = []
    if os.path.exists(os.path.join(script_dir, "biomining_shared.py")):
        paths_to_add.append(script_dir)
    else:
        legacy = os.path.expanduser("~/Desktop/Homeworld/Projects/GapMap")
        if os.path.exists(os.path.join(legacy, "biomining_shared.py")):
            paths_to_add.append(legacy)
        if script_dir not in paths_to_add:
            paths_to_add.append(script_dir)

    # Insert in reverse so the first path ends up at sys.path[0] (highest priority).
    for path in reversed(paths_to_add):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    return script_dir

# Run path setup before imports
_script_directory = setup_import_path()

# Now try to import - if this fails, provide helpful error message
try:
    from biomining_shared import (
        BASE, load_base_data, build_matdf,
        BIO_COLOR, NONBIO_COLOR, PUB_COLOR, IND_COLOR,
        DARK, LIGHT_GREY, GREY, RED, GREEN, TEAL,
        MINING_TYPE_COLORS, STAGE_COLORS, BIO_SUB_COLORS, MATERIAL_COLORS,
        fmt_millions as _fmt_millions_imported,  # Import but will override
    )
except ModuleNotFoundError as e:
    print("\n" + "="*70)
    print("ERROR: Cannot find biomining_shared.py module")
    print("="*70)
    print(f"\nSearched in these directories:")
    for p in sys.path[:5]:
        print(f"  • {p}")
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"\nPlease ensure biomining_shared.py is in one of these locations:")
    print(f"  1. Same directory as this script")
    print(f"  2. ~/Desktop/Homeworld/Projects/GapMap/")
    print(f"  3. Current working directory")
    print("\nSee SETUP_GUIDE.md for detailed instructions.")
    print("="*70 + "\n")
    raise

# ── Override build_matdf to use orientation_corrected ───────────────────────
def build_matdf(kept):
    """
    Explode materials with split funding (no double-counting).
    Uses orientation_corrected if available (manual fixes), otherwise llm_orientation.
    """
    import json
    
    def parse_materials(mat_str):
        """Parse LLM materials field into list."""
        if pd.isna(mat_str) or str(mat_str).strip() == "":
            return []
        try:
            parsed = json.loads(mat_str)
            if isinstance(parsed, list):
                return [str(m).strip() for m in parsed if str(m).strip()]
            return [str(parsed).strip()]
        except:
            return [str(mat_str).strip()]
    
    rows = []
    # Use orientation_corrected if available (manual fixes), otherwise llm_orientation
    orientation_col = 'orientation_corrected' if 'orientation_corrected' in kept.columns else 'llm_orientation'
    
    for _, row in kept.iterrows():
        # Handle materials_list if it exists, otherwise parse from llm_materials
        if 'materials_list' in row and isinstance(row['materials_list'], list):
            mats = row['materials_list']
        else:
            mats = parse_materials(row.get('llm_materials', ''))
        
        if not mats:
            continue
            
        split = row["amt"] / len(mats) if pd.notna(row.get("amt")) else np.nan
        for mat in mats:
            rows.append(dict(
                material=mat, 
                amt=split, 
                full_amt=row.get("amt"),
                is_bio=row.get("is_bio", False), 
                year=row.get("year"),
                llm_mining_type=row.get("llm_mining_type"),
                llm_stage=row.get("llm_stage"),
                llm_orientation=row.get(orientation_col),  # Use corrected if available
                is_outlier=row.get("is_outlier", False),
            ))
    return pd.DataFrame(rows)

# ── Override fmt_millions to show billions ───────────────────────────────────
def fmt_millions(amt):
    """Format dollar amounts: billions for ≥$1000M, otherwise millions"""
    if amt >= 1e9:
        return f"${amt/1e9:.1f}B"
    else:
        return f"${amt/1e6:.1f}M"

# ── New TRL-ordered stage names ──────────────────────────────────────────────
STAGE_NAME_MAP = {
    'deployment': 'Deployment',
    'applied_translational': 'Piloting',
    'applied translational': 'Piloting',
    'early_technology_development': 'Bench Scale Tech Development',
    'fundamental': 'Use Inspired Research',
}

def get_stage_display_name(stage_value):
    """Convert internal stage name to display name"""
    if pd.isna(stage_value):
        return 'Unknown'
    return STAGE_NAME_MAP.get(str(stage_value).lower(), str(stage_value).replace('_', ' ').title())

# ── rcParams ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "figure.figsize":    (16, 9),   # lock default — every fig must also pass figsize=W explicitly
    "savefig.bbox":      None,      # NEVER use tight — every page must be exactly 16×9
})

W          = (16, 9)   # ALL pages use this — never override
YEAR_RANGE = range(2016, 2026)

# ── Updated definitions ───────────────────────────────────────────────────────
MINING_TYPE_DEFS = {
    "exploration":
        "Locating and characterizing mineral resources using geophysical, geochemical, or "
        "remote sensing methods.",

    "extraction":
        "Recovering metals or minerals from ores, wastes, or other sources. Includes bioleaching, "
        "heap leaching, hydrometallurgy, selective separation, and phytomining.",

    "downstream_processing":
        "Refining or purifying already-extracted materials. Includes electrowinning, smelting, "
        "flotation, and solvent extraction.",

    "remediation":
        "Cleaning up mining-impacted environments such as acid mine drainage, tailings, or "
        "contaminated soils and waters.",

    "infrastructure":
        "Tools, instruments, software, databases, sensors, or facilities that enable mining "
        "operations or research.",

    "collaborative_tools":
        "Conferences, workshops, shared databases, innovation centers, workforce programs, "
        "and policy or economic research related to mining.",

    "general":
        "Projects that span multiple parts of the mining value chain with no single dominant focus.",
}

STAGE_DEFS = {
   "fundamental":
    "Knowledge generation only. Basic science, mechanisms, numerical modeling, characterization, geologic surveys, and mine-waste inventories. Output is data or understanding, not a technology.",

   "early_technology_development":
    "First proof-of-concept of a new technology, process, or system at lab or bench scale. Core question: does it work?",

   "applied_translational":
    "Improving, optimizing, or scaling a previously demonstrated technology. Includes pilots, scale-up, techno-economic analyses, and later-stage SBIR work.",

   "deployment":
    "Full-scale implementation of an established technology or program with little or no research component.",
}

BIO_SUB_DEFS = {
    "bioleaching_biooxidation":
        "Microbial processes that oxidize sulfide minerals to release or expose valuable metals. " 
        "Bioleaching dissolves metals into solution, while biooxidation breaks down sulfide " 
        "minerals to make metals accessible for downstream processing.",
        
    "bioseparation_biosorption":
        "Biological molecules or organisms that selectively bind and concentrate metals already " 
        "dissolved in solution. Includes proteins, peptides, siderophores, engineered binding " 
        "molecules, polymers, whole cells, and bio-based resins.",
        
    "bioremediation":
        "Use of microorganisms or plants to remove or detoxify contaminants in mining-impacted " 
        "environments. The primary goal is environmental cleanup rather than metal recovery.",
    
    "bioprecipitation_mineral_formation":
        "Microbially driven precipitation of dissolved metals into solid minerals, concentrating " 
        "or immobilizing them. Examples include sulfate-reducing bacteria forming metal sulfides " 
        "and other biomineralization processes.",
    
    "biosensing":
        "Biological tools used to detect metals, redox conditions, or other chemical signals in " 
        "mining or hydrometallurgical systems for monitoring and process control.",
}

MATERIAL_DEFS = {
    "lanthanides_REE":            "Rare earth elements (REEs): the lanthanides plus yttrium and scandium (e.g., Nd, Dy, Tb, La, Ce, Y, Sc).",

    "base_metals":                "Common industrial metals such as Cu, Zn, Ni, Pb, Fe, and Al.",

    "battery_metals":             "Metals used in battery chemistries (Li, Co, Ni, Mn) when the grant focuses on extraction or recovery from ores, wastes, or secondary sources, not battery manufacturing or storage technology.",

    "polymetallic_general":       "Projects targeting multiple metals without a dominant focus, often described as 'critical minerals' or 'strategic metals'. Used only when no specific metal category clearly dominates.",

    "secondary_feedstocks":       "Metal recovery from secondary or waste-derived sources such as e-waste, spent batteries, acid mine drainage sludge, phosphogypsum, or industrial by-products (excluding primary ores and coal ash).",

    "coal_ash_secondary":         "Metal recovery specifically from coal combustion residues such as fly ash or bottom ash from coal-fired power plants.",

    "precious_metals":            "Precious metals including gold (Au) and silver (Ag).",

    "critical_transition_metals": "Transition metals commonly classified as critical materials such as W, Mo, V, Cr, and Ti.",

    "actinides":                  "Actinide elements such as uranium (U) and thorium (Th), often associated with phosphogypsum or rare earth processing wastes.",

    "metalloids_specialty":       "Specialty metalloids and semiconductor-related elements such as Si, Ge, Ga, Te, In, and As.",

    "platinum_group_metals":      "Platinum group metals (PGMs): Pt, Pd, Rh, Ir, Ru, and Os.",
    }

# ── Money formatters ──────────────────────────────────────────────────────────
def fmt_money(x):
    if pd.isna(x): return "N/A"
    a = abs(x)
    if a >= 1e9: return f"${x/1e9:.2f}B"
    if a >= 1e6: return f"${x/1e6:.1f}M"
    if a >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

def fmt_money_ax(x, _=None):
    a = abs(x)
    if a >= 1e9: return f"${x/1e9:.1f}B"
    if a >= 1e6: return f"${x/1e6:.0f}M"
    if a >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

# ── Page number ───────────────────────────────────────────────────────────────
def _page_num(fig, n):
    fig.text(0.97, 0.01, f"Page {n}", ha="right", va="bottom",
             fontsize=8, color=GREY, transform=fig.transFigure)


# ── Definitions panel ─────────────────────────────────────────────────────────
def _draw_defs_panel(ax, title, defs_dict, color_map):
    """Wider coloured pill (so labels never clip) + definition text."""
    ax.axis("off")
    ax.text(0.01, 0.995, title, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=DARK, va="top")

    items = list(defs_dict.items())
    n     = len(items)
    row_h = 0.93 / n
    y     = 0.93
    
    # Stage name mapping for display
    stage_display_map = {
        'fundamental': 'Use Inspired Research',
        'early_technology_development': 'Bench Scale Tech Development',
        'applied_translational': 'Piloting',
        'deployment': 'Deployment'
    }

    for cat, defn in items:
        color = color_map.get(cat, GREY)
        ax.add_patch(plt.Rectangle(
            (0.01, y - row_h * 0.82), 0.34, row_h * 0.78,
            color=color, alpha=0.88, transform=ax.transAxes, clip_on=False
        ))
        # Use stage display name if it's a stage, otherwise just format the category name
        label_text = stage_display_map.get(cat, cat.replace("_", " "))
        ax.text(0.175, y - row_h * 0.41,
                label_text,
                ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold", transform=ax.transAxes,
                wrap=False)
        wrapped = textwrap.fill(defn, width=55)
        ax.text(0.37, y - row_h * 0.41, wrapped,
                ha="left", va="center", fontsize=8.5,
                color=DARK, transform=ax.transAxes)
        y -= row_h


def _draw_defs_panel_half(ax, defs_dict, color_map, title, y_start=1.0, y_end=0.5):
    """Like _draw_defs_panel but renders within a y_start–y_end band of the axes."""
    ax.axis("off")
    band_h = y_start - y_end
    ax.text(0.01, y_start - 0.01, title, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=DARK, va="top")
    items = list(defs_dict.items())
    n     = len(items)
    row_h = (band_h - 0.06) / n   # reserve 0.06 for the title
    y     = y_start - 0.06
    
    # Stage name mapping for display
    stage_display_map = {
        'fundamental': 'Use Inspired Research',
        'early_technology_development': 'Bench Scale Tech Development',
        'applied_translational': 'Piloting',
        'deployment': 'Deployment'
    }
    
    for cat, defn in items:
        color = color_map.get(cat, GREY)
        ax.add_patch(plt.Rectangle(
            (0.01, y - row_h * 0.82), 0.34, row_h * 0.78,
            color=color, alpha=0.88, transform=ax.transAxes, clip_on=False
        ))
        # Use stage display name if it's a stage, otherwise just format the category name
        label_text = stage_display_map.get(cat, cat.replace("_", " "))
        ax.text(0.175, y - row_h * 0.41, label_text,
                ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", transform=ax.transAxes)
        wrapped = textwrap.fill(defn, width=52)
        ax.text(0.37, y - row_h * 0.41, wrapped,
                ha="left", va="center", fontsize=7.5,
                color=DARK, transform=ax.transAxes)
        y -= row_h


# ── Relative-abundance stacked bar helper ─────────────────────────────────────
def _rel_abundance_bars(ax, yr, cat_col, cat_order, cat_colors, years):
    """100% stacked bars per year.  Bio = solid, Non-Bio = hatch ////."""
    yr_known    = yr[yr[cat_col].isin(cat_order)]
    total_by_yr = yr_known.groupby("year")["amt"].sum().reindex(years, fill_value=0)

    def pct(cat, bio_flag):
        sub  = yr_known[(yr_known[cat_col] == cat) & (yr_known["is_bio"] == bio_flag)]
        vals = sub.groupby("year")["amt"].sum().reindex(years, fill_value=0)
        return (vals / total_by_yr.replace(0, np.nan) * 100).fillna(0).values

    bar_w = 0.72
    for yi, year in enumerate(years):
        bottom = 0.0
        for cat in cat_order:
            color  = cat_colors.get(cat, GREY)
            nb_pct = pct(cat, False)[yi]
            b_pct  = pct(cat, True)[yi]
            if nb_pct > 0:
                ax.bar(year, nb_pct, bottom=bottom, width=bar_w,
                       color=color, hatch="////", edgecolor="white",
                       linewidth=0.4, alpha=0.55)
                bottom += nb_pct
            if b_pct > 0:
                ax.bar(year, b_pct, bottom=bottom, width=bar_w,
                       color=color, edgecolor="white", linewidth=0.4, alpha=0.92)
                bottom += b_pct

    ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    ax.set_ylim(0, 100)
    ax.set_xticks(list(years))
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylabel("% of Annual Funding", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.2)


def _abundance_legend(ax, cat_order=None, cat_colors=None, title=""):
    patches = [
        mpatches.Patch(facecolor=GREY, alpha=0.92,               label="Bio (solid)"),
        mpatches.Patch(facecolor=GREY, alpha=0.55, hatch="////", label="Non-Bio (hatched)"),
    ]
    ax.legend(handles=patches,
          fontsize=9,
          loc="lower right",
          bbox_to_anchor=(1.0, 1.02),
          ncol=2,
          framealpha=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED bar-label helper — used everywhere so the format is identical
# ─────────────────────────────────────────────────────────────────────────────
def _bar_label(ax, bar, value, total, scale, offset_frac=0.012):
    """
    Place  "$X.XM\n(XX%)"  above a bar.
    - value  : the bar value
    - total  : denominator for the % calculation
    - scale  : max value in the same axis (used for vertical offset)
    - offset_frac : fraction of scale to add above the bar top
    """
    pct = value / total * 100 if total > 0 else 0
    if pct == 0:
        pct_str = "0%"
    elif pct < 1:
        pct_str = "< 1%"
    elif pct > 99 and pct < 100:
        pct_str = "> 99%"
    elif pct >= 100:
        pct_str = "> 99%"
    else:
        pct_str = f"{pct:.0f}%"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + scale * offset_frac,
        f"{fmt_money(value)}\n({pct_str})",
        ha="center", va="bottom", fontsize=7.5, linespacing=1.3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TITLE PAGE — comes before everything else
# ─────────────────────────────────────────────────────────────────────────────
def _title_page(pdf):
    fig = plt.figure(figsize=W)
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK); ax.axis("off")

    # ── Main title ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.82,
            "BIOMINING FUNDING LANDSCAPE",
            ha="center", va="center", color="white",
            fontsize=34, fontweight="bold", transform=ax.transAxes, linespacing=1.3)
    ax.text(0.5, 0.73,
            "AND GAP ANALYSIS",
            ha="center", va="center", color="white",
            fontsize=34, fontweight="bold", transform=ax.transAxes)

    # ── Divider line ──────────────────────────────────────────────────────────
    ax.axhline(0.645, xmin=0.12, xmax=0.88, color=TEAL, linewidth=1.5)

    # ── Sections ──────────────────────────────────────────────────────────────
    ax.text(0.5, 0.595, "CONTENTS",
            ha="center", va="center", color=TEAL,
            fontsize=13, fontweight="bold", transform=ax.transAxes)

    sections = [
        ("Section 1", "Federal Research & Innovation Public Grants  (2016–2025)",
         "Federal grants supporting research, technology development, and innovation that advance the science and engineering of mining, mineral processing, and the environmental stewardship of mine-impacted lands."),
        ("Section 2", "Formula Grants (2016-2025)",
         "Formula-based federal funding allocated to states, tribes, territories, or local governments to carry out authorized public programs and activities."),
        ("Section 3", "Future Federal Funding Landscape",
         "Federal funding that has been announced, authorized, or publicly committed but not yet fully obligated or disbursed, as of Feb 2026.")
    ]

    y = 0.515
    for sec_label, sec_title, sec_desc in sections:
        # Section label pill
        ax.add_patch(plt.Rectangle((0.10, y - 0.025), 0.10, 0.055,
                                   color=TEAL, alpha=0.85,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(0.150, y + 0.002, sec_label,
                ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", transform=ax.transAxes)
        # Title
        ax.text(0.22, y + 0.016, sec_title,
                ha="left", va="center", color="white",
                fontsize=12, fontweight="bold", transform=ax.transAxes)
        # Description
        ax.text(0.22, y - 0.012, sec_desc,
                ha="left", va="center", color="#BDC3C7",
                fontsize=9, transform=ax.transAxes, linespacing=1.4)
        y -= 0.155


    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# COVER PAGE
# ─────────────────────────────────────────────────────────────────────────────
def _cover(pdf, kept, label, outlier_note):
    fig = plt.figure(figsize=W)
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK); ax.axis("off")

    total_amt = kept["amt"].sum()
    bio_pct   = kept[kept["is_bio"]]["amt"].sum() / total_amt * 100
    pub_pct   = kept[kept["llm_orientation"] == "public_good"]["amt"].sum() / total_amt * 100
    ind_pct   = kept[kept["llm_orientation"] == "industry_facing"]["amt"].sum() / total_amt * 100

    ax.text(0.5, 0.88, "SECTION 1 - FEDERAL GRANTS",
            ha="center", va="center", color="white",
            fontsize=28, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.79,
            "Federal Research & Innovation Grants — 2016 to 2025",
            ha="center", va="center", color=TEAL,
            fontsize=15, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.71,
            "Federal grants supporting the research, development, and implementation advancing "
            "the science and technology of mining, mineral processing, and environmental "
            "stewardship of mine-impacted lands",
            ha="center", va="center", color="#BDC3C7",
            fontsize=11, transform=ax.transAxes, linespacing=1.5)

    ax.add_patch(plt.Rectangle((0.07, 0.515), 0.86, 0.135,
                               color="#1A252F", transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.630, "Grant Types Included:",
            ha="center", va="center", color=TEAL,
            fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.567,
            "NSF (SBE, GEO, ENG, BIO, DMR)  ·  DOE (ARPA-E, Basic Energy Sciences, Fossil Energy, EERE, Office of Science)"
            "  ·  USDA (NIFA, ARS)\n"
            "USGS  ·  DOD (ONR, DARPA, SERDP, AFRL)  ·  EPA  ·  DOI (OSMRE, BLM)  ·  SBIR / STTR programs",
            ha="center", va="center", color="#ECF0F1",
            fontsize=9.5, transform=ax.transAxes, linespacing=1.65)

    ax.add_patch(plt.Rectangle((0.07, 0.385), 0.86, 0.095,
                               color="#1A252F", transform=ax.transAxes, clip_on=False))
    stats_txt = (
        f"n = {len(kept):,} grants   •   Total: {fmt_money(total_amt)}   •   "
        f"Biological: {bio_pct:.0f}% of funding   •   "
        f"Public Good: {pub_pct:.0f}% of funding   •   "
        f"Industry Facing: {ind_pct:.0f}% of funding"
    )
    ax.text(0.5, 0.432, stats_txt,
            ha="center", va="center", color="#ECF0F1",
            fontsize=11, transform=ax.transAxes)


    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# NEW PAGE 1 - CATEGORIZATION FUNNEL WITHOUT BIO-RELEVANT BUBBLE
# ─────────────────────────────────────────────────────────────────────────────
def _slide_section1_categorization_funnel_no_bio(pdf, kept_raw, page_num):
    """
    NEW FIRST PAGE - Simplified categorization funnel (STOPS at Public vs Industry).
    
    Shows progression from all grants to project grants split by orientation WITHOUT bio bubble.
    """
    from matplotlib.patches import Circle
    
    fig = plt.figure(figsize=W)
    fig.suptitle("Section 1 Grant Categorization — From All Grants to Project Grants by Orientation",
                 fontsize=14, fontweight='bold', y=0.96)
    
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.85])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    BENCHMARK_ABSTRACT_LENGTH = 92
    
    def fmt_money_local(amt):
        if amt >= 1e9:
            return f"${amt/1e9:.1f}B"
        elif amt >= 1e6:
            return f"${amt/1e6:.1f}M"
        else:
            return f"${amt/1e3:.0f}K"
    
    all_grants = kept_raw.copy()
    all_grants['abstract_length'] = all_grants['abstract'].fillna('').str.len()
    abstract_missing = all_grants['abstract'].isna() | (all_grants['abstract'].str.strip() == '')
    abstract_short = (all_grants['abstract_length'] < BENCHMARK_ABSTRACT_LENGTH) & ~abstract_missing
    short_or_missing_mask = abstract_short | abstract_missing
    categorizable_mask = ~short_or_missing_mask
    
    total_all = all_grants['amt'].dropna().sum()
    amt_categorizable = all_grants.loc[categorizable_mask, 'amt'].dropna().sum()
    amt_noncategorizable = all_grants.loc[short_or_missing_mask, 'amt'].dropna().sum()
    
    def pct_of_all(amt):
        return (amt / total_all * 100) if total_all > 0 else 0
    
    bar_width = 0.8
    x1 = 1.2
    y1_base = 0.3
    h_categorizable = 5.5 * (amt_categorizable / total_all) if total_all > 0 else 0
    h_noncat = 5.5 * (amt_noncategorizable / total_all) if total_all > 0 else 0
    
    ax.bar(x1, h_categorizable, width=bar_width, bottom=y1_base,
           color='#2E86AB', edgecolor='white', linewidth=2)
    ax.bar(x1, h_noncat, width=bar_width, bottom=y1_base + h_categorizable,
           color='#B0B0B0', edgecolor='white', linewidth=2, alpha=0.7)
    
    ax.text(x1, y1_base + h_categorizable + h_noncat + 0.4, "All Federal\nGrants (2016-2025)",
            ha='center', va='bottom', fontsize=16, fontweight='bold', color=DARK)
    ax.text(x1, y1_base + h_categorizable/2, f"Categorizable\n{fmt_money_local(amt_categorizable)}",
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    callout_x = x1 + bar_width/2 + 0.8
    callout_y = y1_base + h_categorizable + h_noncat/2
    ax.plot([x1 + bar_width/2, callout_x - 0.3], 
            [y1_base + h_categorizable + h_noncat/2, callout_y],
            'k-', lw=1.5, alpha=0.6)
    callout_text = f"Short Abstract\n{fmt_money_local(amt_noncategorizable)}\n({pct_of_all(amt_noncategorizable):.0f}%)"
    ax.text(callout_x, callout_y, callout_text,
            ha='left', va='center', fontsize=9, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#B0B0B0', linewidth=2))
    
    cat_top_right = (x1 + bar_width/2, y1_base + h_categorizable)
    cat_bottom_right = (x1 + bar_width/2, y1_base)
    
    categorizable_df = all_grants[categorizable_mask].copy()
    is_program = categorizable_df.get("is_formula_grant", pd.Series(False, index=categorizable_df.index)).fillna(False)
    amt_program = categorizable_df.loc[is_program, 'amt'].dropna().sum()
    amt_nonprogr = categorizable_df.loc[~is_program, 'amt'].dropna().sum()
    
    circle_x = 5.5
    circle_y = 3.0
    circle_r = 2.0
    
    ax.text(circle_x, circle_y + circle_r + 0.3, "Categorizable Grants\nby Type",
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK, zorder=7)
    circle = Circle((circle_x, circle_y), circle_r, facecolor='white',
                   edgecolor='#666666', linewidth=2, zorder=5)
    ax.add_patch(circle)
    
    x_program = circle_x - 0.7
    x_nonprog = circle_x + 0.7
    y_base_inner = circle_y - 1.2
    h_scale = 2
    h_program = h_scale * (amt_program / amt_categorizable) if amt_categorizable > 0 else 0
    h_nonprog = h_scale * (amt_nonprogr / amt_categorizable) if amt_categorizable > 0 else 0
    
    ax.bar(x_program, h_program, width=0.7, bottom=y_base_inner,
           color='#B0B0B0', edgecolor='white', linewidth=1.5, alpha=0.7, zorder=6)
    ax.bar(x_nonprog, h_nonprog, width=0.7, bottom=y_base_inner,
           color='#457B9D', edgecolor='white', linewidth=1.5, zorder=6)
    
    ax.text(x_program, y_base_inner + h_program + 0.15, f"{fmt_money_local(amt_program)}\n{pct_of_all(amt_program):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=7)
    ax.text(x_program, y_base_inner - 0.15, "Formula",
            ha='center', va='top', fontsize=7, color=DARK, zorder=7)
    ax.text(x_nonprog, y_base_inner + h_nonprog + 0.15, f"{fmt_money_local(amt_nonprogr)}\n{pct_of_all(amt_nonprogr):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=7)
    ax.text(x_nonprog, y_base_inner - 0.15, "Project",
            ha='center', va='top', fontsize=7, color=DARK, zorder=7)
    
    circle_left = (circle_x - circle_r, circle_y)
    ax.plot([cat_top_right[0], circle_left[0]], 
            [cat_top_right[1], circle_left[1]],
            'k-', lw=1.5, alpha=0.7, zorder=4)
    circle_bottom = (circle_x, circle_y - circle_r)
    ax.plot([cat_bottom_right[0], circle_bottom[0]], 
            [cat_bottom_right[1], circle_bottom[1]],
            'k-', lw=1.5, alpha=0.7, zorder=4)
    
    nonprog_top_right = (x_nonprog + 0.35, y_base_inner + h_nonprog)
    nonprog_bottom_right = (x_nonprog + 0.35, y_base_inner)
    
    nonprog_df = categorizable_df[~is_program].copy()
    orientation_col = 'orientation_corrected' if 'orientation_corrected' in nonprog_df.columns else 'llm_orientation'
    public_mask = nonprog_df.get(orientation_col, pd.Series()).fillna('').str.lower().isin(['public_good', 'public', 'both'])
    amt_public = nonprog_df.loc[public_mask, 'amt'].dropna().sum()
    amt_industry = nonprog_df.loc[~public_mask, 'amt'].dropna().sum()
    
    circle2_x = 11.0
    circle2_y = 5.0
    circle2_r = 2.5
    
    ax.text(circle2_x, circle2_y + circle2_r + 0.3, "Project Grants\nby Orientation",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=DARK, zorder=12)
    circle2 = Circle((circle2_x, circle2_y), circle2_r, facecolor='white',
                    edgecolor='#666666', linewidth=2, zorder=10)
    ax.add_patch(circle2)
    
    x_industry = circle2_x - 0.9
    x_public = circle2_x + 0.9
    y_base_inner2 = circle2_y - 1.6
    h_scale2 = 2.8
    h_industry = h_scale2 * (amt_industry / amt_nonprogr) if amt_nonprogr > 0 else 0
    h_public = h_scale2 * (amt_public / amt_nonprogr) if amt_nonprogr > 0 else 0
    
    ax.bar(x_industry, h_industry, width=0.9, bottom=y_base_inner2,
           color='#B0B0B0', edgecolor='white', linewidth=1.5, alpha=0.7, zorder=11)
    ax.bar(x_public, h_public, width=0.9, bottom=y_base_inner2,
           color='#A8DADC', edgecolor='white', linewidth=1.5, zorder=11)
    
    ax.text(x_industry, y_base_inner2 + h_industry + 0.2, f"{fmt_money_local(amt_industry)}\n{pct_of_all(amt_industry):.0f}%",
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=DARK, zorder=12)
    ax.text(x_industry, y_base_inner2 - 0.2, "Industry",
            ha='center', va='top', fontsize=8, color=DARK, zorder=12)
    ax.text(x_public, y_base_inner2 + h_public + 0.2, f"{fmt_money_local(amt_public)}\n{pct_of_all(amt_public):.0f}%",
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=DARK, zorder=12)
    ax.text(x_public, y_base_inner2 - 0.2, "Public",
            ha='center', va='top', fontsize=8, color=DARK, zorder=12)
    
    circle2_left = (circle2_x - circle2_r, circle2_y)
    ax.plot([nonprog_top_right[0], circle2_left[0]], 
            [nonprog_top_right[1], circle2_left[1]],
            'k-', lw=1.5, alpha=0.7, zorder=8)
    circle2_bottom = (circle2_x, circle2_y - circle2_r)
    ax.plot([nonprog_bottom_right[0], circle2_bottom[0]], 
            [nonprog_bottom_right[1], circle2_bottom[1]],
            'k-', lw=1.5, alpha=0.7, zorder=8)
    
    public_df = nonprog_df[public_mask].copy()
    n_public = len(public_df)
    industry_df = nonprog_df[~public_mask].copy()
    n_industry = len(industry_df)
    
    # Escape dollar signs to prevent LaTeX interpretation
    public_money = fmt_money_local(amt_public).replace('$', r'\$')
    industry_money = fmt_money_local(amt_industry).replace('$', r'\$')
    
    fig.text(0.5, 0.92,
             f"Project Grants Split: {n_public} Public ({public_money}, {pct_of_all(amt_public):.1f}%) + {n_industry} Industry ({industry_money}, {pct_of_all(amt_industry):.1f}%)",
             ha='center', fontsize=11, color='#457B9D', fontweight='bold')
    
    _page_num(fig, page_num)
    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIRST PLOT — HIERARCHICAL FUNNEL: Section 1 Categorization Flow
# ─────────────────────────────────────────────────────────────────────────────
def _slide_section1_categorization_funnel(pdf, kept_raw, page_num):
    """
    NEW FIRST PLOT - Hierarchical funnel matching Section 3 style EXACTLY.
    
    Shows progression from all grants to final Gen 3 (bio/non-bio):
    - STACKED BAR: All Grants → Categorizable (bottom) + Non-Categorizable (top)
    - Circle 1: Categorizable → Formula vs Project
    - Circle 2: Project → Public vs Industry  
    - Circle 3: Public → Bio vs Non-Bio
    
    This funnel explains what data flows into subsequent Section 1 plots.
    """
    from matplotlib.patches import Circle
    
    fig = plt.figure(figsize=W)
    fig.suptitle("Section 1 Grant Categorization — From All Grants to Bio-Relevant Analysis Subset",
                 fontsize=14, fontweight='bold', y=0.96)
    
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.85])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Benchmark abstract length from grant_length_analysis_3.py
    BENCHMARK_ABSTRACT_LENGTH = 92
    
    def fmt_money_local(amt):
        if amt >= 1e9:
            return f"${amt/1e9:.1f}B"
        elif amt >= 1e6:
            return f"${amt/1e6:.1f}M"
        else:
            return f"${amt/1e3:.0f}K"
    
    # Calculate categorizable mask
    all_grants = kept_raw.copy()
    all_grants['abstract_length'] = all_grants['abstract'].fillna('').str.len()
    abstract_missing = all_grants['abstract'].isna() | (all_grants['abstract'].str.strip() == '')
    abstract_short = (all_grants['abstract_length'] < BENCHMARK_ABSTRACT_LENGTH) & ~abstract_missing
    short_or_missing_mask = abstract_short | abstract_missing
    categorizable_mask = ~short_or_missing_mask
    
    total_all = all_grants['amt'].dropna().sum()
    amt_categorizable = all_grants.loc[categorizable_mask, 'amt'].dropna().sum()
    amt_noncategorizable = all_grants.loc[short_or_missing_mask, 'amt'].dropna().sum()
    
    def pct_of_all(amt):
        return (amt / total_all * 100) if total_all > 0 else 0
    
    bar_width = 0.8
    
    # STACKED BAR (bottom-left corner) - Categorizable (bottom) + Non-Categorizable (top)
    x1 = 1.2
    y1_base = 0.3
    h_categorizable = 5.5 * (amt_categorizable / total_all) if total_all > 0 else 0
    h_noncat = 5.5 * (amt_noncategorizable / total_all) if total_all > 0 else 0
    
    ax.bar(x1, h_categorizable, width=bar_width, bottom=y1_base,
           color='#2E86AB', edgecolor='white', linewidth=2)
    ax.bar(x1, h_noncat, width=bar_width, bottom=y1_base + h_categorizable,
           color='#B0B0B0', edgecolor='white', linewidth=2, alpha=0.7)
    
    # Title for stacked bar
    ax.text(x1, y1_base + h_categorizable + h_noncat + 0.4, "All Federal\nGrants (2016-2025)",
            ha='center', va='bottom', fontsize=16, fontweight='bold', color=DARK)
    
    # Label for bottom (categorizable) - INSIDE the bar (this one was fine)
    ax.text(x1, y1_base + h_categorizable/2, f"Categorizable\n{fmt_money_local(amt_categorizable)}",
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Callout for top (SHORT ABSTRACT) - with leader line JUST OUTSIDE to the right
    # Calculate callout position (just to the right of the bar)
    callout_x = x1 + bar_width/2 + 0.8
    callout_y = y1_base + h_categorizable + h_noncat/2
    
    # Leader line from bar to callout
    ax.plot([x1 + bar_width/2, callout_x - 0.3], 
            [y1_base + h_categorizable + h_noncat/2, callout_y],
            'k-', lw=1.5, alpha=0.6)
    
    # Callout text box
    callout_text = f"Short Abstract\n{fmt_money_local(amt_noncategorizable)}\n({pct_of_all(amt_noncategorizable):.0f}%)"
    ax.text(callout_x, callout_y, callout_text,
            ha='left', va='center', fontsize=9, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#B0B0B0', linewidth=2))
    
    # Categorizable corners
    cat_top_right = (x1 + bar_width/2, y1_base + h_categorizable)
    cat_bottom_right = (x1 + bar_width/2, y1_base)
    
    # CIRCLE 1: Categorizable → Formula vs Project
    categorizable_df = all_grants[categorizable_mask].copy()
    is_program = categorizable_df.get("is_formula_grant", pd.Series(False, index=categorizable_df.index)).fillna(False)
    amt_program = categorizable_df.loc[is_program, 'amt'].dropna().sum()
    amt_nonprogr = categorizable_df.loc[~is_program, 'amt'].dropna().sum()
    
    circle_x = 4.2
    circle_y = 3.0
    circle_r = 2.0
    
    ax.text(circle_x, circle_y + circle_r + 0.3, "Categorizable Grants\nby Type",
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK, zorder=7)
    circle = Circle((circle_x, circle_y), circle_r, facecolor='white',
                   edgecolor='#666666', linewidth=2, zorder=5)
    ax.add_patch(circle)
    
    # Bars INSIDE circle 1
    x_program = circle_x - 0.7
    x_nonprog = circle_x + 0.7
    y_base_inner = circle_y - 1.2
    h_scale = 2
    h_program = h_scale * (amt_program / amt_categorizable) if amt_categorizable > 0 else 0
    h_nonprog = h_scale * (amt_nonprogr / amt_categorizable) if amt_categorizable > 0 else 0
    
    ax.bar(x_program, h_program, width=0.7, bottom=y_base_inner,
           color='#B0B0B0', edgecolor='white', linewidth=1.5, alpha=0.7, zorder=6)
    ax.bar(x_nonprog, h_nonprog, width=0.7, bottom=y_base_inner,
           color='#457B9D', edgecolor='white', linewidth=1.5, zorder=6)
    
    ax.text(x_program, y_base_inner + h_program + 0.15, f"{fmt_money_local(amt_program)}\n{pct_of_all(amt_program):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=7)
    ax.text(x_program, y_base_inner - 0.15, "Formula",
            ha='center', va='top', fontsize=7, color=DARK, zorder=7)
    ax.text(x_nonprog, y_base_inner + h_nonprog + 0.15, f"{fmt_money_local(amt_nonprogr)}\n{pct_of_all(amt_nonprogr):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=7)
    ax.text(x_nonprog, y_base_inner - 0.15, "Project",
            ha='center', va='top', fontsize=7, color=DARK, zorder=7)
    
    # LINES from Categorizable to Circle 1
    circle_left = (circle_x - circle_r, circle_y)
    ax.plot([cat_top_right[0], circle_left[0]], 
            [cat_top_right[1], circle_left[1]],
            'k-', lw=1.5, alpha=0.7, zorder=4)
    circle_bottom = (circle_x, circle_y - circle_r)
    ax.plot([cat_bottom_right[0], circle_bottom[0]], 
            [cat_bottom_right[1], circle_bottom[1]],
            'k-', lw=1.5, alpha=0.7, zorder=4)
    
    # Project bar corners for next generation
    nonprog_top_right = (x_nonprog + 0.35, y_base_inner + h_nonprog)
    nonprog_bottom_right = (x_nonprog + 0.35, y_base_inner)
    
    # GENERATION 2: Project → Public vs Industry
    nonprog_df = categorizable_df[~is_program].copy()
    # Use orientation_corrected if available (manual fixes), otherwise llm_orientation
    orientation_col = 'orientation_corrected' if 'orientation_corrected' in nonprog_df.columns else 'llm_orientation'
    public_mask = nonprog_df.get(orientation_col, pd.Series()).fillna('').str.lower().isin(['public_good', 'public', 'both'])
    amt_public = nonprog_df.loc[public_mask, 'amt'].dropna().sum()
    amt_industry = nonprog_df.loc[~public_mask, 'amt'].dropna().sum()
    
    circle2_x = 8.2
    circle2_y = 5.0
    circle2_r = 2.0
    
    ax.text(circle2_x, circle2_y + circle2_r + 0.3, "Project Grants\nby Orientation",
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK, zorder=12)
    circle2 = Circle((circle2_x, circle2_y), circle2_r, facecolor='white',
                    edgecolor='#666666', linewidth=2, zorder=10)
    ax.add_patch(circle2)
    
    # Bars INSIDE circle 2
    x_industry = circle2_x - 0.7
    x_public = circle2_x + 0.7
    y_base_inner2 = circle2_y - 1.3
    h_scale2 = 2.2
    h_industry = h_scale2 * (amt_industry / amt_nonprogr) if amt_nonprogr > 0 else 0
    h_public = h_scale2 * (amt_public / amt_nonprogr) if amt_nonprogr > 0 else 0
    
    ax.bar(x_industry, h_industry, width=0.7, bottom=y_base_inner2,
           color='#B0B0B0', edgecolor='white', linewidth=1.5, alpha=0.7, zorder=11)
    ax.bar(x_public, h_public, width=0.7, bottom=y_base_inner2,
           color='#A8DADC', edgecolor='white', linewidth=1.5, zorder=11)
    
    ax.text(x_industry, y_base_inner2 + h_industry + 0.15, f"{fmt_money_local(amt_industry)}\n{pct_of_all(amt_industry):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=12)
    ax.text(x_industry, y_base_inner2 - 0.15, "Industry",
            ha='center', va='top', fontsize=7, color=DARK, zorder=12)
    ax.text(x_public, y_base_inner2 + h_public + 0.15, f"{fmt_money_local(amt_public)}\n{pct_of_all(amt_public):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=12)
    ax.text(x_public, y_base_inner2 - 0.15, "Public",
            ha='center', va='top', fontsize=7, color=DARK, zorder=12)
    
    # LINES from Project to Circle 2
    circle2_left = (circle2_x - circle2_r, circle2_y)
    ax.plot([nonprog_top_right[0], circle2_left[0]], 
            [nonprog_top_right[1], circle2_left[1]],
            'k-', lw=1.5, alpha=0.7, zorder=8)
    circle2_bottom = (circle2_x, circle2_y - circle2_r)
    ax.plot([nonprog_bottom_right[0], circle2_bottom[0]], 
            [nonprog_bottom_right[1], circle2_bottom[1]],
            'k-', lw=1.5, alpha=0.7, zorder=8)
    
    # Public bar corners for next generation
    public_top_right = (x_public + 0.35, y_base_inner2 + h_public)
    public_bottom_right = (x_public + 0.35, y_base_inner2)
    
    # GENERATION 3: Public → Bio vs Non-Bio
    public_df = nonprog_df[public_mask].copy()
    bio_mask = public_df.get('is_bio', pd.Series(False, index=public_df.index)).fillna(False)
    amt_bio = public_df.loc[bio_mask, 'amt'].dropna().sum()
    amt_nonbio = public_df.loc[~bio_mask, 'amt'].dropna().sum()
    n_bio = bio_mask.sum()
    
    circle3_x = 12.2
    circle3_y = 7.0
    circle3_r = 2.0
    
    ax.text(circle3_x, circle3_y + circle3_r + 0.3, "Bio-Relevant\nPublic Grants",
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK, zorder=17)
    circle3 = Circle((circle3_x, circle3_y), circle3_r, facecolor='white',
                    edgecolor='#666666', linewidth=2, zorder=15)
    ax.add_patch(circle3)
    
    # Bars INSIDE circle 3
    x_nonbio = circle3_x - 0.7
    x_bio = circle3_x + 0.7
    y_base_inner3 = circle3_y - 1.4
    h_scale3 = 2.4
    h_nonbio = h_scale3 * (amt_nonbio / amt_public) if amt_public > 0 else 0
    h_bio = h_scale3 * (amt_bio / amt_public) if amt_public > 0 else 0
    
    ax.bar(x_nonbio, h_nonbio, width=0.7, bottom=y_base_inner3,
           color='#B0B0B0', edgecolor='white', linewidth=1.5, alpha=0.7, zorder=16)
    ax.bar(x_bio, h_bio, width=0.7, bottom=y_base_inner3,
           color='#4CAF50', edgecolor='white', linewidth=1.5, zorder=16)
    
    ax.text(x_nonbio, y_base_inner3 + h_nonbio + 0.15, f"{fmt_money_local(amt_nonbio)}\n{pct_of_all(amt_nonbio):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=17)
    ax.text(x_nonbio, y_base_inner3 - 0.15, "Non-Bio",
            ha='center', va='top', fontsize=7, color=DARK, zorder=17)
    ax.text(x_bio, y_base_inner3 + h_bio + 0.15, f"{fmt_money_local(amt_bio)}\n{pct_of_all(amt_bio):.1f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=17)
    ax.text(x_bio, y_base_inner3 - 0.15, "Bio",
            ha='center', va='top', fontsize=7, fontweight='bold', color=DARK, zorder=17)
    
    # LINES from Public to Circle 3
    circle3_left = (circle3_x - circle3_r, circle3_y)
    ax.plot([public_top_right[0], circle3_left[0]], 
            [public_top_right[1], circle3_left[1]],
            'k-', lw=1.5, alpha=0.7, zorder=13)
    circle3_bottom = (circle3_x, circle3_y - circle3_r)
    ax.plot([public_bottom_right[0], circle3_bottom[0]], 
            [public_bottom_right[1], circle3_bottom[1]],
            'k-', lw=1.5, alpha=0.7, zorder=13)
    
    # Summary at top
    fig.text(0.5, 0.92,
             f"Final Analysis Subset: {n_bio} Bio + {(~bio_mask).sum()} Non-Bio Public Grants = {fmt_money_local(amt_public)} ({pct_of_all(amt_public):.1f}% of all grants)",
             ha='center', fontsize=11, color='#4CAF50', fontweight='bold')
    
    _page_num(fig, page_num)
    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER FUNCTION FOR SECTION 1 - GEN 3 DATASET
# ─────────────────────────────────────────────────────────────────────────────
def _filter_to_gen2(kept_raw):
    """
    Filter to Gen 2: Categorizable, Project grants (Public + Industry).
    This includes both public and industry-facing grants.
    
    Filtering steps:
    1. Categorizable: abstract ≥ 92 chars (not short/missing)
    2. Project: not formula grants
    
    Returns the filtered dataframe.
    """
    BENCHMARK_ABSTRACT_LENGTH = 92
    
    df = kept_raw.copy()
    
    # Step 1: Categorizable (abstract ≥ 92 chars)
    df['abstract_length'] = df['abstract'].fillna('').str.len()
    abstract_missing = df['abstract'].isna() | (df['abstract'].str.strip() == '')
    abstract_short = (df['abstract_length'] < BENCHMARK_ABSTRACT_LENGTH) & ~abstract_missing
    categorizable_mask = ~(abstract_short | abstract_missing)
    df = df[categorizable_mask].copy()
    
    # Step 2: Project (not formula grants)
    is_program = df.get("is_formula_grant", pd.Series(False, index=df.index)).fillna(False)
    df = df[~is_program].copy()
    
    # Note: Does NOT filter by orientation - includes both public and industry
    
    return df


def _filter_to_gen3(kept_raw):
    """
    Filter to Gen 3: Categorizable, Project, Public grants only.
    This is the dataset ALL Section 1 PUBLIC plots should use.
    
    Filtering steps:
    1. Categorizable: abstract ≥ 92 chars (not short/missing)
    2. Project: not formula grants
    3. Public: public_good, public, or both orientation
    
    Returns the filtered dataframe.
    """
    BENCHMARK_ABSTRACT_LENGTH = 92
    
    df = kept_raw.copy()
    
    # Step 1: Categorizable (abstract ≥ 92 chars)
    df['abstract_length'] = df['abstract'].fillna('').str.len()
    abstract_missing = df['abstract'].isna() | (df['abstract'].str.strip() == '')
    abstract_short = (df['abstract_length'] < BENCHMARK_ABSTRACT_LENGTH) & ~abstract_missing
    categorizable_mask = ~(abstract_short | abstract_missing)
    df = df[categorizable_mask].copy()
    
    # Step 2: Project (not formula grants)
    is_program = df.get("is_formula_grant", pd.Series(False, index=df.index)).fillna(False)
    df = df[~is_program].copy()
    
    # Step 3: Public orientation (both counts as public!)
    # Use orientation_corrected if available (manual fixes), otherwise llm_orientation
    orientation_col = 'orientation_corrected' if 'orientation_corrected' in df.columns else 'llm_orientation'
    public_mask = df.get(orientation_col, pd.Series()).fillna('').str.lower().isin(['public_good', 'public', 'both'])
    df = df[public_mask].copy()
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def _slide_portfolio_overview(pdf, kept, label, page_num):
    fig = plt.figure(figsize=W)
    # Format: "Section 1A: All Project Grants - Portfolio Overview"
    fig.suptitle(f"{label} — Portfolio Overview", fontsize=14, fontweight="bold", y=0.98)

    # Layout: left col = Bio/NonBio funding (top) + count (bottom)
    #         right col = timeseries all grants (top) + timeseries deployment-excluded (bottom)
    ax_fund = fig.add_axes([0.05, 0.52, 0.16, 0.38])   # top-left: funding vertical bar
    ax_cnt  = fig.add_axes([0.05, 0.08, 0.16, 0.35])   # bottom-left: count vertical bar
    ax_ts1  = fig.add_axes([0.27, 0.54, 0.70, 0.36])   # top-right: timeseries all
    ax_ts2  = fig.add_axes([0.27, 0.10, 0.70, 0.36])   # bottom-right: timeseries no-dep

    total_amt   = kept["amt"].sum()
    total_n     = len(kept)
    bio_amt     = kept[kept["is_bio"]]["amt"].sum()
    nbio_amt    = total_amt - bio_amt
    bio_n       = kept["is_bio"].sum()
    nbio_n      = total_n - bio_n

    def _fmt_pct(p):
        if p == 0:  return "0%"
        if p < 1:   return "< 1%"
        if p > 99:  return "> 99%"
        return             f"{p:.1f}%"

    bio_pct_val  = bio_amt / total_amt * 100 if total_amt > 0 else 0
    nbio_pct_val = 100 - bio_pct_val
    bio_n_pct    = bio_n  / total_n  * 100 if total_n  > 0 else 0
    nbio_n_pct   = 100 - bio_n_pct

    # ── Funding vertical bar (top-left) ──────────────────────────────────────
    bar_cats   = ["Biological", "Non-Bio"]
    bar_vals   = [bio_amt, nbio_amt]
    bar_cols   = [BIO_COLOR, NONBIO_COLOR]
    bars_f = ax_fund.bar(bar_cats, bar_vals, color=bar_cols, edgecolor="white", width=0.55)
    ax_fund.set_ylabel("Total Funding", fontsize=9)
    ax_fund.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_fund.set_title("Bio vs Non-Bio\nFunding ($)", fontsize=10, pad=4)
    scale_f = max(bar_vals)
    for bar, v, pv in zip(bars_f, bar_vals, [bio_pct_val, nbio_pct_val]):
        ax_fund.text(bar.get_x() + bar.get_width()/2, v + scale_f * 0.02,
                     f"{fmt_money(v)}\n({_fmt_pct(pv)})",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_fund.set_ylim(0, scale_f * 1.30)
    ax_fund.tick_params(axis="x", labelsize=9)

    # ── Count vertical bar (bottom-left) ─────────────────────────────────────
    cnt_vals = [bio_n, nbio_n]
    bars_c = ax_cnt.bar(bar_cats, cnt_vals, color=bar_cols, edgecolor="white", width=0.55)
    ax_cnt.set_ylabel("Number of Grants", fontsize=9)
    ax_cnt.set_title("Bio vs Non-Bio\nGrant Count (n)", fontsize=10, pad=4)
    scale_c = max(cnt_vals)
    for bar, v, pv in zip(bars_c, cnt_vals, [bio_n_pct, nbio_n_pct]):
        ax_cnt.text(bar.get_x() + bar.get_width()/2, v + scale_c * 0.02,
                    f"{v:,}\n({_fmt_pct(pv)})",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_cnt.set_ylim(0, scale_c * 1.30)
    ax_cnt.tick_params(axis="x", labelsize=9)

    # ── Timeseries panels (right column) ─────────────────────────────────────
    years = list(YEAR_RANGE)
    def _build_ts_p1(data):
        yr  = data.dropna(subset=["year", "amt"])
        yr  = yr[yr["year"].between(2016, 2025)]
        bio = yr[yr["is_bio"]].groupby("year")["amt"].sum().reindex(years, fill_value=0)
        nb  = yr[~yr["is_bio"]].groupby("year")["amt"].sum().reindex(years, fill_value=0)
        return bio, nb

    bio_all, nb_all = _build_ts_p1(kept)
    no_dep    = kept[kept["llm_stage"] != "deployment"]
    n_dep     = (kept["llm_stage"] == "deployment").sum()
    amt_dep   = kept[kept["llm_stage"] == "deployment"]["amt"].sum()
    bio_nd, nb_nd = _build_ts_p1(no_dep)

    def _ts_panel(ax, bio_ts, nb_ts, title):
        scale = max(nb_ts.values.max(), 1)
        ax.fill_between(years, nb_ts.values, alpha=0.12, color=NONBIO_COLOR)
        ax.plot(years, nb_ts.values, "s-", color=NONBIO_COLOR, linewidth=2.0, markersize=5)
        ax.fill_between(years, bio_ts.values, alpha=0.12, color=BIO_COLOR)
        ax.plot(years, bio_ts.values, "o-", color=BIO_COLOR, linewidth=2.0, markersize=5)
        for y_val, amt in zip(years, nb_ts.values):
            if amt > 0:
                ax.text(y_val, amt + scale * 0.07, fmt_money(amt),
                        ha="center", va="bottom", fontsize=6.5, color=NONBIO_COLOR)
        for y_val, amt in zip(years, bio_ts.values):
            if amt > 0:
                ax.text(y_val, amt - scale * 0.055, fmt_money(amt),
                        ha="center", va="top", fontsize=6.5, color=BIO_COLOR)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_ylabel("Total Funding", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(years); ax.tick_params(axis="x", labelsize=8, rotation=0)
        ax.set_ylim(-scale * 0.12, scale * 1.28)
        ax.legend(handles=[
            mpatches.Patch(color=BIO_COLOR,    label="Biological"),
            mpatches.Patch(color=NONBIO_COLOR, label="Non-Biological"),
        ], fontsize=7.5, loc="upper left")

    _ts_panel(ax_ts1, bio_all, nb_all, "Bio vs Non-Bio Time Series")
    _ts_panel(ax_ts2, bio_nd,  nb_nd,
              f"Bio vs Non-Bio Time Series — Deployment Excluded ({n_dep} grants, {fmt_money(amt_dep)})")
    ax_ts2.set_xlabel("Year", fontsize=9)

    _page_num(fig, page_num)
    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Funding by Mining Type & Stage + combined definitions sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _slide_timeseries_combined(pdf, kept, label, page_num):
    """Now page 2: Mining Type bars (top) + Stage bars (bottom) + defs sidebar."""
    fig = plt.figure(figsize=W)
    fig.suptitle(f"{label} — Funding by Mining Type & Research Stage",
                 fontsize=12, fontweight="bold", y=0.99)

    # Charts shifted right to prevent y-axis labels being clipped
    ax_mt   = fig.add_axes([0.10, 0.54, 0.50, 0.37])   # top: mining type
    ax_st   = fig.add_axes([0.10, 0.08, 0.50, 0.37])   # bottom: stage
    ax_defs = fig.add_axes([0.64, 0.04, 0.35, 0.92])   # right: definitions

    total_amt = kept["amt"].sum()

    # ── Mining type bars — bio split ─────────────────────────────────────────
    mt_bio_s = kept[kept["is_bio"]].groupby("llm_mining_type")["amt"].sum()
    mt_nb_s  = kept[~kept["is_bio"]].groupby("llm_mining_type")["amt"].sum()
    mt_all_s = kept.groupby("llm_mining_type")["amt"].sum().sort_values(ascending=False)
    scale_mt = mt_all_s.values.max()
    MIN_VIS_MT = scale_mt * 0.008
    for xi, cat in enumerate(mt_all_s.index):
        color  = MINING_TYPE_COLORS.get(cat, GREY)
        b_val  = float(mt_bio_s.get(cat, 0))
        nb_val = float(mt_nb_s.get(cat, 0))
        render_b = max(b_val, MIN_VIS_MT) if b_val > 0 else 0
        if render_b > 0:
            ax_mt.bar(xi, render_b, color=color, alpha=0.95, edgecolor="white", width=0.6)
        if nb_val > 0:
            ax_mt.bar(xi, nb_val, bottom=render_b, color=color, alpha=0.55,
                      hatch="////", edgecolor="white", width=0.6)
        total = b_val + nb_val
        if total > 0:
            pct = total / total_amt * 100 if total_amt > 0 else 0
            pct_str = "< 1%" if pct < 1 else ("> 99%" if pct > 99 else f"{pct:.0f}%")
            ax_mt.text(xi, render_b + nb_val + scale_mt * 0.015,
                       f"{fmt_money(total)}\n({pct_str})",
                       ha="center", va="bottom", fontsize=7.5, linespacing=1.3)
    xlbls_mt = [c.replace("_", " ") for c in mt_all_s.index]
    ax_mt.set_xticks(range(len(xlbls_mt)))
    ax_mt.set_xticklabels(xlbls_mt, fontsize=8, rotation=0, ha="center")
    ax_mt.set_ylabel("Total Funding"); ax_mt.set_title("Funding by Mining Research Type")
    ax_mt.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_mt.set_ylim(0, scale_mt * 1.28)
    ax_mt.legend(handles=[
        mpatches.Patch(facecolor=GREY, alpha=0.92,               label="Bio (solid)"),
        mpatches.Patch(facecolor=GREY, alpha=0.55, hatch="////", label="Non-Bio (hatched)"),
    ], fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, framealpha=0.9)

    # ── Stage bars — bio split ────────────────────────────────────────────────
    stage_order  = ["fundamental", "early_technology_development",
                    "applied_translational", "deployment"]
    stage_labels = ["Use Inspired\nResearch", "Bench Scale\nTech Dev.", "Piloting", "Deployment"]
    st_bio_s = kept[kept["is_bio"]].groupby("llm_stage")["amt"].sum()
    st_nb_s  = kept[~kept["is_bio"]].groupby("llm_stage")["amt"].sum()
    sf       = kept.groupby("llm_stage")["amt"].sum().reindex(stage_order, fill_value=0)
    scale_st = max(sf.values.max(), 1)
    MIN_VIS = scale_st * 0.008   # minimum rendered height so tiny bio sliver is visible
    for xi, (stage, slbl) in enumerate(zip(stage_order, stage_labels)):
        color  = STAGE_COLORS.get(stage, GREY)
        b_val  = float(st_bio_s.get(stage, 0))
        nb_val = float(st_nb_s.get(stage, 0))
        render_b = max(b_val, MIN_VIS) if b_val > 0 else 0
        if render_b > 0:
            ax_st.bar(xi, render_b, color=color, alpha=0.95, edgecolor="white", width=0.6)
        if nb_val > 0:
            ax_st.bar(xi, nb_val, bottom=render_b, color=color, alpha=0.55,
                      hatch="////", edgecolor="white", width=0.6)
        
        # Total label at top
        total = b_val + nb_val
        if total > 0:
            pct = total / total_amt * 100 if total_amt > 0 else 0
            pct_str = "< 1%" if pct < 1 else ("> 99%" if pct > 99 else f"{pct:.0f}%")
            ax_st.text(xi, render_b + nb_val + scale_st * 0.015,
                       f"{fmt_money(total)}\n({pct_str})",
                       ha="center", va="bottom", fontsize=7.5, linespacing=1.3)
        
        # Bio label - always show, positioned to RIGHT of bar
        bio_pct = b_val / total_amt * 100 if total_amt > 0 else 0
        if b_val == 0:
            label_text = 'No Bio\nFunding'
            label_color = color  # Match the stage color
        else:
            label_text = f'{fmt_money(b_val)}\n({bio_pct:.1f}%)'
            label_color = color
        
        # Position to right of bar, vertically centered on bio portion (or bottom if bio=0)
        label_y = render_b/2 if b_val > 0 else scale_st * 0.05
        ax_st.text(xi + 0.35, label_y, label_text,
                  ha='left', va='center', fontsize=6.5,
                  color=label_color, fontweight='bold', linespacing=1.2,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor=label_color, linewidth=1.5, alpha=0.95))
    ax_st.set_xticks(range(len(stage_order)))
    ax_st.set_xticklabels(stage_labels, fontsize=9)
    ax_st.set_ylabel("Total Funding"); ax_st.set_title("Funding by Research Stage")
    ax_st.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_st.set_ylim(0, scale_st * 1.28)
    ax_st.legend(handles=[
        mpatches.Patch(facecolor=GREY, alpha=0.92,               label="Bio (solid)"),
        mpatches.Patch(facecolor=GREY, alpha=0.55, hatch="////", label="Non-Bio (hatched)"),
    ], fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, framealpha=0.9)

    # ── Combined definitions sidebar ──────────────────────────────────────────
    # Split ax_defs into two halves manually using text positioning
    _draw_defs_panel_half(ax_defs, MINING_TYPE_DEFS, MINING_TYPE_COLORS,
                          "Mining Research Type Definitions",
                          y_start=0.99, y_end=0.50)
    _draw_defs_panel_half(ax_defs, STAGE_DEFS, STAGE_COLORS,
                          "Research Stage Definitions",
                          y_start=0.48, y_end=0.00)

    _page_num(fig, page_num)
    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 (new) — BIO SUBCATEGORY RELATIVE ABUNDANCE BY RESEARCH STAGE
# ─────────────────────────────────────────────────────────────────────────────
def _slide_biosub_stage_abundance(pdf, kept, label, page_num):
    """
    Relative abundance chart: bio subcategories on x-axis,
    stacked bars showing % of annual funding by research stage.
    Only bio grants are included.
    """
    bio = kept[kept["is_bio"]].copy()
    bio = bio.dropna(subset=["llm_bio_subcategory", "amt"])

    fig = plt.figure(figsize=W)
    fig.suptitle(f"{label} — Bio Subcategory by Research Stage",
                 fontsize=13, fontweight="bold", y=0.99)

    ax_chart = fig.add_axes([0.05, 0.12, 0.56, 0.80])
    ax_defs  = fig.add_axes([0.64, 0.04, 0.35, 0.92])

    stage_order  = ["fundamental", "early_technology_development",
                    "applied_translational", "deployment"]

    # Get all bio subcategories sorted by total funding descending
    sub_totals = bio.groupby("llm_bio_subcategory")["amt"].sum().sort_values(ascending=False)
    sub_cats   = list(sub_totals.index)

    # For each subcategory, compute % of that subcategory's total going to each stage
    bar_w = 0.65
    bottoms = np.zeros(len(sub_cats))

    for stage in stage_order:
        stage_amts = []
        for sub in sub_cats:
            sub_total = sub_totals.get(sub, 0)
            stage_amt = bio.loc[
                (bio["llm_bio_subcategory"] == sub) & (bio["llm_stage"] == stage),
                "amt"
            ].sum()
            pct = stage_amt / sub_total * 100 if sub_total > 0 else 0
            stage_amts.append(pct)

        color = STAGE_COLORS.get(stage, GREY)
        bars  = ax_chart.bar(range(len(sub_cats)), stage_amts,
                             bottom=bottoms, width=bar_w,
                             color=color, alpha=0.88, edgecolor="white", linewidth=0.4,
                             label=stage.replace("_", " ").title())
        bottoms += np.array(stage_amts)

    # X-axis: bio subcategory labels
    xlbls = [c.replace("_", "\n") for c in sub_cats]
    ax_chart.set_xticks(range(len(sub_cats)))
    ax_chart.set_xticklabels(xlbls, fontsize=9)
    ax_chart.set_ylabel("% of Subcategory Funding", fontsize=11)
    ax_chart.set_ylim(0, 115)
    ax_chart.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_chart.set_xlabel("Biological Subcategory", fontsize=11)
    ax_chart.grid(axis="y", alpha=0.2)

    # Dollar total and bio percentage above each bar
    bio_total = bio["amt"].sum()
    for xi, sub in enumerate(sub_cats):
        total = sub_totals.get(sub, 0)
        bio_pct = (total / bio_total * 100) if bio_total > 0 else 0
        ax_chart.text(xi, bottoms[xi] + 1.5, f"{fmt_money(total)}\n({bio_pct:.1f}%)",
                      ha="center", va="bottom", fontsize=8, color=DARK, linespacing=1.2)

    # No legend — stage definitions panel on right serves as the key

    # Definitions sidebar — research stage defs
    _draw_defs_panel(ax_defs, "Research Stage Definitions",
                     STAGE_DEFS, STAGE_COLORS)

    _page_num(fig, page_num)
    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — STAGE RELATIVE ABUNDANCE
# ─────────────────────────────────────────────────────────────────────────────
def _slide_stage_abundance(pdf, kept, label, page_num):
    fig = plt.figure(figsize=W)
    fig.suptitle(f"{label} — Research Stage Over Time",
                 fontsize=13, fontweight="bold", y=0.99)
    ax_chart = fig.add_axes([0.04, 0.10, 0.61, 0.82])
    ax_defs  = fig.add_axes([0.67, 0.04, 0.32, 0.92])

    stage_order = ["fundamental", "early_technology_development",
                   "applied_translational", "deployment"]
    yr    = kept.dropna(subset=["year", "amt"])
    yr    = yr[yr["year"].between(2016, 2025)]
    years = list(YEAR_RANGE)

    _rel_abundance_bars(ax_chart, yr, "llm_stage", stage_order, STAGE_COLORS, years)
    ax_chart.set_xlabel("Year", fontsize=11)
    _abundance_legend(ax_chart, stage_order, STAGE_COLORS, title="Stage")
    _draw_defs_panel(ax_defs, "Research Stage Definitions", STAGE_DEFS, STAGE_COLORS)

    _page_num(fig, page_num)
    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — MINING TYPE RELATIVE ABUNDANCE
# ─────────────────────────────────────────────────────────────────────────────
def _slide_miningtype_abundance(pdf, kept, label, page_num):
    fig = plt.figure(figsize=W)
    fig.suptitle(f"{label} — Mining Type Over Time",
                 fontsize=13, fontweight="bold", y=0.99)
    ax_chart = fig.add_axes([0.04, 0.10, 0.55, 0.82])
    ax_defs  = fig.add_axes([0.61, 0.04, 0.38, 0.92])

    type_order = sorted(MINING_TYPE_COLORS.keys())
    yr    = kept.dropna(subset=["year", "amt"])
    yr    = yr[yr["year"].between(2016, 2025)]
    years = list(YEAR_RANGE)

    _rel_abundance_bars(ax_chart, yr, "llm_mining_type", type_order, MINING_TYPE_COLORS, years)
    ax_chart.set_xlabel("Year", fontsize=11)
    _abundance_legend(ax_chart, type_order, MINING_TYPE_COLORS, title="Mining Type")
    _draw_defs_panel(ax_defs, "Mining Research Type Definitions",
                     MINING_TYPE_DEFS, MINING_TYPE_COLORS)

    _page_num(fig, page_num)
    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — BIO SUBCATEGORY
# ─────────────────────────────────────────────────────────────────────────────
def _slide_bio_subcategory(pdf, kept, label, page_num):
    bio   = kept[kept["is_bio"]].copy()
    years = list(YEAR_RANGE)

    fig = plt.figure(figsize=W)
    fig.suptitle(f"{label} — Biological Subcategory Funding",
                 fontsize=13, fontweight="bold", y=0.99)
    ax_line = fig.add_axes([0.03, 0.55, 0.53, 0.36])
    ax_fund = fig.add_axes([0.03, 0.08, 0.53, 0.36])
    ax_defs = fig.add_axes([0.60, 0.04, 0.39, 0.92])

    sf     = bio.groupby("llm_bio_subcategory")["amt"].sum().sort_values(ascending=False)
    cats   = list(sf.index)
    xlbls  = [c.replace("_", "\n") for c in cats]
    colors = [BIO_SUB_COLORS.get(c, GREY) for c in cats]

    bars      = ax_fund.bar(xlbls, sf.values, color=colors, edgecolor="white", width=0.55)
    total_bio = sf.sum()
    scale_bio = sf.values.max()
    ax_fund.set_ylabel("Total Funding")
    ax_fund.set_title("Total Funding\nby Bio Subcategory", fontsize=10, fontweight="bold")
    ax_fund.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_fund.tick_params(axis="x", labelsize=8)
    for bar, v in zip(bars, sf.values):
        _bar_label(ax_fund, bar, v, total_bio, scale_bio)

    yr_bio = bio.dropna(subset=["year", "amt"])
    yr_bio = yr_bio[yr_bio["year"].between(2016, 2025)]
    bio_ts = yr_bio.groupby("year")["amt"].sum().reindex(years, fill_value=0)

    ax_line.fill_between(years, bio_ts.values, alpha=0.18, color=BIO_COLOR)
    ax_line.plot(years, bio_ts.values, "o-", color=BIO_COLOR, linewidth=2.2, markersize=6)
    _bio_nudge = {2020: 0.12}
    for y_val, amt in zip(years, bio_ts.values):
        if amt > 0:
            nudge = _bio_nudge.get(y_val, 0.04)
            ax_line.text(y_val, amt + bio_ts.values.max() * nudge,
                         fmt_money(amt), ha="center", va="bottom",
                         fontsize=6.5, color=BIO_COLOR)
    ax_line.set_title("Bio Funding\nOver Time", fontsize=10)
    ax_line.set_ylabel("Total Funding", fontsize=9)
    ax_line.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_line.set_xticks(years)
    ax_line.tick_params(axis="x", labelsize=8)
    ax_line.grid(axis="y", alpha=0.25)

    _draw_defs_panel(ax_defs, "Biological Subcategory Definitions",
                     BIO_SUB_DEFS, BIO_SUB_COLORS)

    _page_num(fig, page_num)
    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — MATERIAL TYPE
# ─────────────────────────────────────────────────────────────────────────────
def _slide_materials(pdf, kept, matdf, label, page_num):
    fig = plt.figure(figsize=W)
    fig.suptitle(f"{label} — Funding by Material Type", fontsize=12, fontweight="bold", y=0.99)
    ax_bars = fig.add_axes([0.05, 0.18, 0.46, 0.72])
    ax_defs = fig.add_axes([0.55, 0.04, 0.44, 0.92])

    sub    = matdf.dropna(subset=["material", "amt"])
    bio_m  = sub[sub["is_bio"]].groupby("material")["amt"].sum()
    nb_m   = sub[~sub["is_bio"]].groupby("material")["amt"].sum()
    all_m  = sub.groupby("material")["amt"].sum().sort_values(ascending=False)
    cats   = list(all_m.index)
    xlbls  = [c.replace("_", "\n") for c in cats]
    colors = [MATERIAL_COLORS.get(c, GREY) for c in cats]

    bio_vals = np.array([bio_m.get(c, 0) for c in cats])
    nb_vals  = np.array([nb_m.get(c, 0)  for c in cats])
    totals   = bio_vals + nb_vals
    total_mat = totals.sum()
    max_total = totals.max()

    ax_bars.bar(range(len(cats)), bio_vals, color=colors, edgecolor="white",
                width=0.6, label="Bio", alpha=0.92)
    ax_bars.bar(range(len(cats)), nb_vals, bottom=bio_vals,
                color=colors, edgecolor="white", hatch="////",
                width=0.6, label="Non-Bio", alpha=0.55)

    ax_bars.set_xticks(range(len(cats)))
    ax_bars.set_xticklabels(xlbls, fontsize=7.5)
    ax_bars.set_ylabel("Total Funding (split across co-tags)")
    ax_bars.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))

    for xi, (bv, nv) in enumerate(zip(bio_vals, nb_vals)):
        total = bv + nv
        if total > 0:
            pct = total / total_mat * 100 if total_mat > 0 else 0
            ax_bars.text(xi, total + max_total * 0.015,
                         f"{fmt_money(total)}\n({pct:.0f}%)",
                         ha="center", va="bottom", fontsize=7.5, linespacing=1.3)

    ax_bars.legend(handles=[
        mpatches.Patch(facecolor=GREY, alpha=0.92,               label="Bio (solid)"),
        mpatches.Patch(facecolor=GREY, alpha=0.55, hatch="////", label="Non-Bio (hatched)"),
    ], fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, framealpha=0.9)

    _draw_defs_panel(ax_defs, "Material Type Definitions", MATERIAL_DEFS, MATERIAL_COLORS)

    _page_num(fig, page_num)
    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# NEW PAGE — BIO % BY STAGE + AVERAGE GRANT SIZE BY STAGE
# ─────────────────────────────────────────────────────────────────────────────
def _slide_stage_bio_analysis(pdf, kept, label, page_num):
    """
    Two-plot page:
    LEFT: Bio % of funding in each research stage (bar chart)
    RIGHT: Average grant size by stage, bio vs non-bio (grouped bars with error bars)
    """
    fig = plt.figure(figsize=W)
    fig.suptitle(f"{label} — Research Stage Bio Share & Grant Size Comparison",
                 fontsize=13, fontweight='bold', y=0.97)
    
    stage_order = ['fundamental', 'early_technology_development', 'applied_translational', 'deployment']
    stage_labels = ['Use Inspired\nResearch', 'Bench Scale\nTech Dev.', 'Piloting', 'Deployment']
    
    ax_pct = fig.add_axes([0.08, 0.15, 0.38, 0.72])
    ax_avg = fig.add_axes([0.55, 0.15, 0.38, 0.72])
    
    # ─────────────────────────────────────────────────────────────────────────
    # LEFT PLOT: Bio % of funding by stage
    # ─────────────────────────────────────────────────────────────────────────
    bio_pct_list = []
    for stage in stage_order:
        stage_total = kept[kept['llm_stage'] == stage]['amt'].sum()
        stage_bio = kept[(kept['llm_stage'] == stage) & (kept['is_bio'])]['amt'].sum()
        bio_pct = (stage_bio / stage_total * 100) if stage_total > 0 else 0
        bio_pct_list.append(bio_pct)
    
    colors = [STAGE_COLORS.get(s, GREY) for s in stage_order]
    bars_pct = ax_pct.bar(range(len(stage_order)), bio_pct_list,
                          color=colors, edgecolor='white', linewidth=1.5, width=0.65)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars_pct, bio_pct_list):
        if pct > 0:
            ax_pct.text(bar.get_x() + bar.get_width()/2, pct + max(bio_pct_list)*0.02,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax_pct.set_xticks(range(len(stage_order)))
    ax_pct.set_xticklabels(stage_labels, fontsize=9)
    ax_pct.set_ylabel('Bio % of Stage Funding', fontsize=10, fontweight='bold')
    ax_pct.set_title('Biological Share by Stage', fontsize=11, pad=8)
    ax_pct.set_ylim(0, max(bio_pct_list) * 1.20)
    ax_pct.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ─────────────────────────────────────────────────────────────────────────
    # RIGHT PLOT: Grant size comparison by stage (bio vs non-bio)
    # ─────────────────────────────────────────────────────────────────────────
    
    # First, check if data is skewed to determine appropriate visualization/test
    all_amounts = kept['amt'].dropna()
    skewness = all_amounts.skew() if len(all_amounts) > 0 else 0
    
    # If skewness > 1 or < -1, data is highly skewed → use box plots on log scale + Mann-Whitney
    # Otherwise use bar charts + t-test
    use_nonparametric = abs(skewness) > 1
    
    if use_nonparametric:
        # BOX-AND-WHISKER PLOTS for skewed data (on log scale)
        box_data_bio = []
        box_data_nonbio = []
        p_values = []
        
        for stage in stage_order:
            # Bio
            bio_stage = kept[(kept['llm_stage'] == stage) & (kept['is_bio'])]['amt'].dropna()
            box_data_bio.append(bio_stage.values if len(bio_stage) > 0 else [])
            
            # Non-bio
            nonbio_stage = kept[(kept['llm_stage'] == stage) & (~kept['is_bio'])]['amt'].dropna()
            box_data_nonbio.append(nonbio_stage.values if len(nonbio_stage) > 0 else [])
            
            # Mann-Whitney U test (non-parametric)
            # Only test if both groups have sufficient samples (min 3 each)
            if len(bio_stage) >= 3 and len(nonbio_stage) >= 3:
                statistic, p_val = stats.mannwhitneyu(bio_stage, nonbio_stage, alternative='two-sided')
                p_values.append(p_val)
            else:
                p_values.append(1.0)  # No test if insufficient samples
        
        # Create box plots
        positions_bio = np.arange(len(stage_order)) - 0.2
        positions_nonbio = np.arange(len(stage_order)) + 0.2
        
        # Box plot properties for professional appearance
        box_props_bio = dict(facecolor=BIO_COLOR, edgecolor='white', linewidth=1.5, alpha=0.8)
        box_props_nonbio = dict(facecolor=NONBIO_COLOR, edgecolor='white', linewidth=1.5, alpha=0.8)
        whisker_props = dict(color='black', linewidth=1.2)
        cap_props = dict(color='black', linewidth=1.2)
        median_props = dict(color='white', linewidth=2)  # Thin white line for median
        flier_props = dict(marker='o', markersize=3, alpha=0.5)  # Small dots for outliers
        
        # Plot bio boxes
        bp_bio = ax_avg.boxplot(box_data_bio, positions=positions_bio, widths=0.35,
                                patch_artist=True, showfliers=True,
                                boxprops=box_props_bio, whiskerprops=whisker_props,
                                capprops=cap_props, medianprops=median_props,
                                flierprops=dict(**flier_props, markerfacecolor=BIO_COLOR, markeredgecolor=BIO_COLOR))
        
        # Plot non-bio boxes
        bp_nonbio = ax_avg.boxplot(box_data_nonbio, positions=positions_nonbio, widths=0.35,
                                   patch_artist=True, showfliers=True,
                                   boxprops=box_props_nonbio, whiskerprops=whisker_props,
                                   capprops=cap_props, medianprops=median_props,
                                   flierprops=dict(**flier_props, markerfacecolor=NONBIO_COLOR, markeredgecolor=NONBIO_COLOR))
        
        # Set log scale for y-axis
        ax_avg.set_yscale('log')
        
        ax_avg.set_xticks(np.arange(len(stage_order)))
        ax_avg.set_xticklabels(stage_labels, fontsize=9)
        ax_avg.set_ylabel('Grant Size ($, log scale)', fontsize=10, fontweight='bold')
        ax_avg.set_title('Grant Size Distribution by Stage (Bio vs Non-Bio)', fontsize=11, pad=8)
        ax_avg.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
        
        # Custom legend
        legend_elements = [
            mpatches.Patch(facecolor=BIO_COLOR, edgecolor='white', linewidth=1.5, alpha=0.8, label='Bio'),
            mpatches.Patch(facecolor=NONBIO_COLOR, edgecolor='white', linewidth=1.5, alpha=0.8, label='Non-Bio')
        ]
        ax_avg.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        ax_avg.grid(axis='y', alpha=0.3, linestyle='--', which='both')
        
        # Add significance brackets
        # Get y-axis limits for bracket placement
        y_lim = ax_avg.get_ylim()
        y_max_log = y_lim[1]
        
        for i, p_val in enumerate(p_values):
            if p_val < 0.05:
                # Find the actual highest point (whisker or outlier) for both boxes
                bio_data = box_data_bio[i]
                nonbio_data = box_data_nonbio[i]
                
                if len(bio_data) > 0 and len(nonbio_data) > 0:
                    # Find max value from either group (could be outlier beyond whisker)
                    bio_max = np.max(bio_data)
                    nonbio_max = np.max(nonbio_data)
                    max_val = max(bio_max, nonbio_max)
                    
                    # Position bracket higher above the highest point on log scale
                    bracket_y = max_val * 2.0  # Doubled from 1.5 to give more space
                    
                    # Get the top of each box (75th percentile) for bracket legs
                    bio_q3 = np.percentile(bio_data, 75)
                    nonbio_q3 = np.percentile(nonbio_data, 75)
                    
                    x1 = positions_bio[i]
                    x2 = positions_nonbio[i]
                    
                    # Draw bracket: vertical lines down to just above highest point, connected by horizontal line
                    # Small gap above data
                    gap = max_val * 0.05  # 5% gap above the highest point
                    # Left leg down to just above bio max
                    ax_avg.plot([x1, x1], [bracket_y, bio_max + gap], color='black', linewidth=1.2)
                    # Horizontal connector
                    ax_avg.plot([x1, x2], [bracket_y, bracket_y], color='black', linewidth=1.2)
                    # Right leg down to just above nonbio max
                    ax_avg.plot([x2, x2], [bracket_y, nonbio_max + gap], color='black', linewidth=1.2)
                    
                    # Add significance stars above bracket (even higher)
                    sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                    ax_avg.text((x1 + x2)/2, bracket_y * 1.3, sig_text,
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Statistical summary
        stats_text = f"Distribution skewness: {skewness:.2f}. Mann-Whitney U test: "
        sig_count = sum(1 for p in p_values if p < 0.05)
        stats_text += f"{sig_count}/{len(stage_order)} stages show significant difference (p<0.05). "
        stats_text += "* p<0.05, ** p<0.01, *** p<0.001. Box = IQR, line = median, dots = outliers."
        
    else:
        # BAR CHARTS for normally distributed data
        bio_avgs = []
        bio_sems = []
        nonbio_avgs = []
        nonbio_sems = []
        p_values = []
        
        for stage in stage_order:
            # Bio
            bio_stage = kept[(kept['llm_stage'] == stage) & (kept['is_bio'])]['amt'].dropna()
            if len(bio_stage) > 0:
                bio_avgs.append(bio_stage.mean())
                bio_sems.append(bio_stage.sem())
            else:
                bio_avgs.append(0)
                bio_sems.append(0)
            
            # Non-bio
            nonbio_stage = kept[(kept['llm_stage'] == stage) & (~kept['is_bio'])]['amt'].dropna()
            if len(nonbio_stage) > 0:
                nonbio_avgs.append(nonbio_stage.mean())
                nonbio_sems.append(nonbio_stage.sem())
            else:
                nonbio_avgs.append(0)
                nonbio_sems.append(0)
            
            # T-test (only test if both groups have sufficient samples, min 3 each)
            if len(bio_stage) >= 3 and len(nonbio_stage) >= 3:
                t_stat, p_val = stats.ttest_ind(bio_stage, nonbio_stage)
                p_values.append(p_val)
            else:
                p_values.append(1.0)  # No test if insufficient samples
        
        x_pos = np.arange(len(stage_order))
        bar_width = 0.35
        
        # Plot grouped bars with SEM error bars
        bars_bio = ax_avg.bar(x_pos - bar_width/2, bio_avgs, bar_width,
                              yerr=bio_sems, capsize=4,
                              color=BIO_COLOR, edgecolor='white', linewidth=1.5,
                              label='Bio', error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        bars_nonbio = ax_avg.bar(x_pos + bar_width/2, nonbio_avgs, bar_width,
                                 yerr=nonbio_sems, capsize=4,
                                 color=NONBIO_COLOR, edgecolor='white', linewidth=1.5,
                                 label='Non-Bio', error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        ax_avg.set_xticks(x_pos)
        ax_avg.set_xticklabels(stage_labels, fontsize=9)
        ax_avg.set_ylabel('Average Grant Size ($)', fontsize=10, fontweight='bold')
        ax_avg.set_title('Average Grant Size by Stage (Bio vs Non-Bio)', fontsize=11, pad=8)
        ax_avg.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
        ax_avg.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_avg.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add significance brackets
        y_max = max(max(bio_avgs), max(nonbio_avgs))
        for i, (p_val, bio_avg, nonbio_avg, bio_sem, nonbio_sem) in enumerate(zip(p_values, bio_avgs, nonbio_avgs, bio_sems, nonbio_sems)):
            if p_val < 0.05:
                bar_top = max(bio_avg + bio_sem, nonbio_avg + nonbio_sem)
                gap = y_max * 0.02  # Small gap above error bars
                bracket_y = bar_top + y_max * 0.08  # Position bracket above bars
                
                x1 = i - bar_width/2
                x2 = i + bar_width/2
                # Vertical lines down to just above error bars, connected by horizontal line
                ax_avg.plot([x1, x1], [bracket_y, bar_top + gap], color='black', linewidth=1.2)
                ax_avg.plot([x1, x2], [bracket_y, bracket_y], color='black', linewidth=1.2)
                ax_avg.plot([x2, x2], [bracket_y, bar_top + gap], color='black', linewidth=1.2)
                
                sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                ax_avg.text(i, bracket_y + y_max*0.02, sig_text,
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Statistical summary
        stats_text = f"Distribution skewness: {skewness:.2f}. t-test: "
        sig_count = sum(1 for p in p_values if p < 0.05)
        stats_text += f"{sig_count}/{len(stage_order)} stages show significant difference (p<0.05). "
        stats_text += "* p<0.05, ** p<0.01, *** p<0.001. Error bars = SEM."
    
    fig.text(0.5, 0.04, stats_text, ha='center', fontsize=7.5, style='italic',
             color=GREY)
    
    _page_num(fig, page_num)
    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 DIVIDER
# ─────────────────────────────────────────────────────────────────────────────
def _section1b_divider(pdf, kept_gen3=None):
    """Divider slide introducing Section 1B: Public Project Grants"""
    fig = plt.figure(figsize=W)
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK)
    ax.axis("off")

    ax.text(0.5, 0.88, "SECTION 1B — PUBLIC PROJECT GRANTS",
            ha="center", va="center", color="white",
            fontsize=28, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.79, "Public-Oriented Project Grants (2016–2025)",
            ha="center", va="center", color=TEAL,
            fontsize=15, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.71,
            "This section analyzes public-oriented project grants only, excluding industry-facing grants. "
            "These grants focus on advancing public benefit and scientific knowledge.",
            ha="center", va="center", color="#BDC3C7",
            fontsize=11, transform=ax.transAxes, linespacing=1.5)

    ax.add_patch(plt.Rectangle((0.07, 0.515), 0.86, 0.135,
                               color="#1A252F", transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.630, "Data Scope:",
            ha="center", va="center", color=TEAL,
            fontsize=10, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.567,
            "Section 1B is a subset of Section 1A. It includes only public-oriented grants, "
            "excluding industry-facing grants. All grants have sufficient abstracts (≥92 characters) and exclude formula grants.",
            ha="center", va="center", color="#ECF0F1",
            fontsize=9.5, transform=ax.transAxes, linespacing=1.65)

    ax.add_patch(plt.Rectangle((0.07, 0.385), 0.86, 0.095,
                               color="#1A252F", transform=ax.transAxes, clip_on=False))

    if kept_gen3 is not None and len(kept_gen3) > 0:
        total = kept_gen3["amt"].sum()
        bio_amt = kept_gen3[kept_gen3["is_bio"]]["amt"].sum()
        bio_pct = bio_amt / total * 100 if total > 0 else 0

        stats_txt = (
            f"n = {len(kept_gen3):,} public project grants   •   "
            f"Total funding (2016–2025): {fmt_money(total)}   •   "
            f"Biological: {fmt_money(bio_amt)} ({bio_pct:.1f}% of funding)"
        )
    else:
        stats_txt = "No public project grants available for summary."

    ax.text(0.5, 0.432, stats_txt,
            ha="center", va="center", color="#ECF0F1",
            fontsize=11, transform=ax.transAxes)

    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 DIVIDER
# ─────────────────────────────────────────────────────────────────────────────
def _section_divider(pdf, kept_prog=None):
    fig = plt.figure(figsize=W)
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK)
    ax.axis("off")

    ax.text(0.5, 0.88, "SECTION 2 - FORMULA GRANTS",
            ha="center", va="center", color="white",
            fontsize=28, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.79, "Formula Grants (Non-competitive)",
            ha="center", va="center", color=TEAL,
            fontsize=15, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.71,
            "Formula grants are federal funds distributed to states, tribes, territories, or local governments according to a formula set in law rather than through competitive applications.",
            ha="center", va="center", color="#BDC3C7",
            fontsize=11, transform=ax.transAxes, linespacing=1.5)

    ax.add_patch(plt.Rectangle((0.07, 0.515), 0.86, 0.135,
                               color="#1A252F", transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.630, "Grant Types Included:",
            ha="center", va="center", color=TEAL,
            fontsize=10, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.567,
            "Forumula Grant (A) - Primarily Abandoned Mine Land (AML) grants under SMCRA, expanded by the Bipartisan Infrastructure Law "
            "(2021), which appropriated $11.3B nationally as annual tranches over 15 years.\n"
            "Each row in this dataset represents a separately obligated annual award to a specific state, tribe, or territory.",
            ha="center", va="center", color="#ECF0F1",
            fontsize=9.5, transform=ax.transAxes, linespacing=1.65)

    ax.add_patch(plt.Rectangle((0.07, 0.385), 0.86, 0.095,
                               color="#1A252F", transform=ax.transAxes, clip_on=False))

    if kept_prog is not None and len(kept_prog) > 0:
        total = kept_prog["amt"].sum()
        bio_pct = kept_prog[kept_prog["is_bio"]]["amt"].sum() / total * 100 if total > 0 else 0
        pub_pct = kept_prog[kept_prog["llm_orientation"] == "public_good"]["amt"].sum() / total * 100 if total > 0 else 0

        stats_txt = (
            f"n = {len(kept_prog):,} formula grants   •   "
            f"Total obligated (2016–2025): {fmt_money(total)}   •   "
            f"Biological: < 1% of funding   •   "
            f"Public Good: {pub_pct:.0f}% of funding"
        )
    else:
        stats_txt = "No formula grants available for summary."

    ax.text(0.5, 0.432, stats_txt,
            ha="center", va="center", color="#ECF0F1",
            fontsize=11, transform=ax.transAxes)

    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — FORMULA GRANTS OVERVIEW
# All bar labels use $X.XM\n(XX%) format consistent with Section 1
# ─────────────────────────────────────────────────────────────────────────────
def _slide_prog_overview(pdf, kept_prog, page_num):
    if kept_prog is None or len(kept_prog) == 0:
        return

    fig = plt.figure(figsize=W)
    fig.suptitle("Formula Grants: Funding by Category",
                 fontsize=14, fontweight="bold", y=0.99)

    ax_donut = fig.add_axes([0.03, 0.08, 0.36, 0.82])
    ax_mt    = fig.add_axes([0.45, 0.54, 0.51, 0.38])
    ax_st    = fig.add_axes([0.45, 0.08, 0.51, 0.38])

    total_amt = kept_prog["amt"].sum()
    bio_amt   = kept_prog[kept_prog["is_bio"]]["amt"].sum()
    nbio_amt  = total_amt - bio_amt

    def _fmt_pct(p):
        if p == 0:    return "0%"
        if p < 1:     return "< 1%"
        if p > 99:    return "> 99%"
        return              f"{p:.1f}%"

    bio_pct_val = bio_amt / total_amt * 100 if total_amt > 0 else 0

    # Bio vs Non-Bio donut
    wedges, _ = ax_donut.pie(
        [bio_amt, nbio_amt], labels=None,
        colors=[BIO_COLOR, NONBIO_COLOR],
        startangle=90, wedgeprops={"width": 0.55},
    )
    pct_vals = [bio_pct_val, 100.0 - bio_pct_val]
    lbl_vals = [f"Bio\n{fmt_money(bio_amt)}", f"Non-Bio\n{fmt_money(nbio_amt)}"]
    lbl_cols = [BIO_COLOR, NONBIO_COLOR]
    for wedge, pv, lbl, col in zip(wedges, pct_vals, lbl_vals, lbl_cols):
        ang = (wedge.theta1 + wedge.theta2) / 2
        rad = np.deg2rad(ang)
        ax_donut.text(0.72 * np.cos(rad), 0.72 * np.sin(rad),
                     _fmt_pct(pv), ha="center", va="center",
                     fontsize=11, fontweight="bold", color="white")
        ro = 1.28
        ha = "left" if np.cos(rad) > 0.1 else ("right" if np.cos(rad) < -0.1 else "center")
        ax_donut.text(ro * np.cos(rad), ro * np.sin(rad),
                     lbl, ha=ha, va="center",
                     fontsize=10, fontweight="bold", color=col)
    ax_donut.set_xlim(-1.9, 1.9); ax_donut.set_ylim(-1.9, 1.9)
    ax_donut.set_title("Bio vs Non-Bio", fontsize=12, pad=6)

    # ── Mining type bars — $X.XM\n(XX%) labels ──────────────────────────────
    mt_bio = kept_prog[kept_prog["is_bio"]].groupby("llm_mining_type")["amt"].sum()
    mt_nb  = kept_prog[~kept_prog["is_bio"]].groupby("llm_mining_type")["amt"].sum()
    mt_all = kept_prog.groupby("llm_mining_type")["amt"].sum().sort_values(ascending=False)
    cats_mt      = list(mt_all.index)
    scale_mt_all = mt_all.values.max() if len(mt_all) > 0 else 1

    for xi, cat in enumerate(cats_mt):
        color  = MINING_TYPE_COLORS.get(cat, GREY)
        b_val  = mt_bio.get(cat, 0)
        nb_val = mt_nb.get(cat, 0)
        if b_val  > 0:
            ax_mt.bar(xi, b_val, color=color, alpha=0.92, edgecolor="white", width=0.55)
        if nb_val > 0:
            ax_mt.bar(xi, nb_val, bottom=b_val, color=color, alpha=0.55,
                      hatch="////", edgecolor="white", width=0.55)
        total = b_val + nb_val
        if total > 0:
            pct = total / total_amt * 100
            if pct < 1:
                pct_str = "< 1%"
            elif pct > 99:
                pct_str = "> 99%"
            else:
                pct_str = f"{pct:.0f}%"
            ax_mt.text(xi, total + scale_mt_all * 0.015,
                       f"{fmt_money(total)}\n({pct_str})",
                       ha="center", va="bottom", fontsize=7.5, linespacing=1.3)

    ax_mt.set_xticks(range(len(cats_mt)))
    ax_mt.set_xticklabels([c.replace("_", " ") for c in cats_mt], fontsize=8.5)
    ax_mt.set_ylabel("Total Funding")
    ax_mt.set_title("By Mining Research Type")
    ax_mt.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_mt.set_ylim(0, scale_mt_all * 1.22)
    ax_mt.legend(handles=[
        mpatches.Patch(facecolor=GREY, alpha=0.92, label="Bio"),
        mpatches.Patch(facecolor=GREY, alpha=0.55, hatch="////", label="Non-Bio"),
    ], fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, framealpha=0.9)

    # ── Stage bars — $X.XM\n(XX%) labels ────────────────────────────────────
    stage_order  = ["fundamental", "early_technology_development",
                    "applied_translational", "deployment"]
    stage_labels = ["Use Inspired\nResearch", "Bench Scale\nTech Dev.", "Piloting", "Deployment"]
    st_bio = kept_prog[kept_prog["is_bio"]].groupby("llm_stage")["amt"].sum()
    st_nb  = kept_prog[~kept_prog["is_bio"]].groupby("llm_stage")["amt"].sum()
    st_all = kept_prog.groupby("llm_stage")["amt"].sum().reindex(stage_order, fill_value=0)
    scale_st_all = max(st_all.values.max(), 1)

    for xi, (stage, slbl) in enumerate(zip(stage_order, stage_labels)):
        color  = STAGE_COLORS.get(stage, GREY)
        b_val  = st_bio.get(stage, 0)
        nb_val = st_nb.get(stage, 0)
        if b_val  > 0:
            ax_st.bar(xi, b_val, color=color, alpha=0.92, edgecolor="white", width=0.55)
        if nb_val > 0:
            ax_st.bar(xi, nb_val, bottom=b_val, color=color, alpha=0.55,
                      hatch="////", edgecolor="white", width=0.55)
        total = b_val + nb_val
        if total > 0:
            pct = total / total_amt * 100
            if pct < 1:
                pct_str = "< 1%"
            elif pct > 99:
                pct_str = "> 99%"
            else:
                pct_str = f"{pct:.0f}%"
            ax_st.text(xi, total + scale_st_all * 0.015,
                       f"{fmt_money(total)}\n({pct_str})",
                       ha="center", va="bottom", fontsize=7.5, linespacing=1.3)

    ax_st.set_xticks(range(len(stage_order)))
    ax_st.set_xticklabels(stage_labels, fontsize=9)
    ax_st.set_ylabel("Total Funding")
    ax_st.set_title("By Research Stage")
    ax_st.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_st.set_ylim(0, scale_st_all * 1.22)
    ax_st.legend(handles=[
        mpatches.Patch(facecolor=GREY, alpha=0.92, label="Bio"),
        mpatches.Patch(facecolor=GREY, alpha=0.55, hatch="////", label="Non-Bio"),
    ], fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, framealpha=0.9)

    _page_num(fig, page_num)
    fig.set_size_inches(W); pdf.savefig(fig); plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: NEW HIERARCHICAL BREAKDOWN
# ═════════════════════════════════════════════════════════════════════════════

def _page_num(fig, n):
    """Add page number to figure."""
    fig.text(0.98, 0.02, f"Page {n}", ha='right', va='bottom',
             fontsize=8, color=GREY)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 DIVIDER
# ─────────────────────────────────────────────────────────────────────────────
def _section3_divider(pdf, all_federal_df=None):
    """Section 3 divider with updated context."""
    fig = plt.figure(figsize=W)
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK)
    ax.axis("off")

    ax.text(0.5, 0.88, "SECTION 3 - FUTURE FEDERAL FUNDING",
        ha="center", va="center", color="white",
        fontsize=28, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.79,
        "Critical Minerals Federal Investment Landscape (as of Feb 2026)",
        ha="center", va="center", color=TEAL,
        fontsize=15, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.71,
        "Federal funding that has been announced, authorized, or publicly committed but not yet fully obligated or disbursed.",
        ha="center", va="center", color="#BDC3C7",
        fontsize=11, transform=ax.transAxes, linespacing=1.5)

    ax.add_patch(plt.Rectangle((0.07, 0.515), 0.86, 0.135,
                               color="#1A252F", transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.630, "Sources Included:",
            ha="center", va="center", color=TEAL,
            fontsize=10, fontweight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.567,
        "State Department Critical Minerals Ministerial (Feb. 2026)  ·  DOE NOFOs and FOAs "
        "(FECM, MESC, EERE, AMMTO)  ·  NSF and Grants.gov funding opportunities\n"
        "DOE Loan Programs Office commitments  ·  EXIM announcements  ·  DoD and DFC authorizations",
        ha="center", va="center", color="#ECF0F1",
        fontsize=9.5, transform=ax.transAxes, linespacing=1.65)

    ax.add_patch(plt.Rectangle((0.07, 0.360), 0.86, 0.120,
                               color="#1A252F", transform=ax.transAxes, clip_on=False))

    if all_federal_df is not None and len(all_federal_df) > 0:
        total_all  = all_federal_df["amt"].dropna().sum()
        
        # Calculate funnel stats
        mining_mask = all_federal_df['reason_to_remove'].isna()
        amt_mining = all_federal_df.loc[mining_mask, 'amt'].dropna().sum()
        
        mining_df = all_federal_df[mining_mask].copy()
        grant_mask = mining_df.get('excel_instrument_type', pd.Series()).fillna('').str.contains(
            'Grant|FOA|NOFO', case=False, na=False)
        amt_grants = mining_df.loc[grant_mask, 'amt'].dropna().sum()
        
        grants_df = mining_df[grant_mask].copy()
        public_mask = grants_df.get('orientation', pd.Series()).fillna('').str.lower() == 'public'
        amt_public = grants_df.loc[public_mask, 'amt'].dropna().sum()
        
        public_df = grants_df[public_mask].copy()
        bio_mask = public_df.get('biomining_fit', pd.Series()).fillna('').str.lower() == 'yes'
        amt_bio = public_df.loc[bio_mask, 'amt'].dropna().sum()
        n_bio = bio_mask.sum()

        line1 = (
            f"Total Critical Minerals Announced: {fmt_money(total_all)}   •   "
            f"Mining-Relevant: {fmt_money(amt_mining)} ({amt_mining/total_all*100:.1f}%)"
        )
        line2 = (
            f"Grant Funding: {fmt_money(amt_grants)} ({amt_grants/amt_mining*100:.1f}% of mining)   •   "
            f"Public Grants: {fmt_money(amt_public)} ({amt_public/amt_grants*100:.1f}% of grants)   •   "
            f"Bio-Applicable: n={n_bio} grants ({fmt_money(amt_bio)}, {amt_bio/amt_public*100:.1f}% of public)"
        )
        stats_txt = line1 + "\n" + line2
    else:
        stats_txt = "No future federal funding initiatives available for summary."

    ax.text(0.5, 0.432, stats_txt,
            ha="center", va="center", color="#ECF0F1",
            fontsize=10.5, transform=ax.transAxes, linespacing=1.7)

    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 REPLACEMENT - Clearer hierarchy + bar charts for bio grants
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 9 — HIERARCHICAL FUNNEL (Stacked Horizontal Bars)
# ─────────────────────────────────────────────────────────────────────────────
def _slide_future_funnel_v8(pdf, all_federal_df, page_num):
    """Testing first stage only."""
    fig = plt.figure(figsize=W)
    fig.patch.set_facecolor('white')
    
    # Large title at top of chart
    fig.text(0.5, 0.95, "Future Federal Funding Initiatives for Critical Minerals",
             ha='center', va='top', fontsize=14, fontweight='bold', color=DARK)
    
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.88])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    from matplotlib.patches import Circle
    
    # Calculate
    total_all = all_federal_df['amt'].dropna().sum()
    mining_mask = all_federal_df['reason_to_remove'].isna()
    amt_mining = all_federal_df.loc[mining_mask, 'amt'].dropna().sum()
    amt_mfg = all_federal_df.loc[~mining_mask, 'amt'].dropna().sum()
    
    mining_df = all_federal_df[mining_mask].copy()
    grant_mask = mining_df.get('excel_instrument_type', pd.Series()).fillna('').str.contains(
        'Grant|FOA|NOFO', case=False, na=False)
    amt_grants = mining_df.loc[grant_mask, 'amt'].dropna().sum()
    amt_loans = mining_df.loc[~grant_mask, 'amt'].dropna().sum()
    
    def pct_of_mining(amt):
        return (amt / amt_mining * 100) if amt_mining > 0 else 0
    
    bar_width = 0.8
    
    # LEVEL 1: Stacked bar (bottom-left corner)
    x1 = 1.2
    y1_base = 0.3
    h_mining = 5.5 * (amt_mining / total_all) if total_all > 0 else 0
    h_mfg = 5.5 * (amt_mfg / total_all) if total_all > 0 else 0
    
    ax.bar(x1, h_mining, width=bar_width, bottom=y1_base,
           color='#2E86AB', edgecolor='white', linewidth=2)
    ax.bar(x1, h_mfg, width=bar_width, bottom=y1_base + h_mining,
           color='#B0B0B0', edgecolor='white', linewidth=2, alpha=0.7)
    
    # Title for stacked bar
    ax.text(x1, y1_base + h_mining + h_mfg + 0.4, "All Future Critical Minerals\nFederal Funding",
            ha='center', va='bottom', fontsize=16, fontweight='bold', color=DARK)
    
    # Labels INSIDE bars
    ax.text(x1, y1_base + h_mining/2, f"Mining\n{fmt_money(amt_mining)}",
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    ax.text(x1, y1_base + h_mining + h_mfg/2, f"Mfg.\n{fmt_money(amt_mfg)}",
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Mining corners
    mining_top_right = (x1 + bar_width/2, y1_base + h_mining)
    mining_bottom_right = (x1 + bar_width/2, y1_base)
    
    # CIRCLE around Loans/Grants (much closer to stacked bar)
    circle_x = 4.2
    circle_y = 3.0
    circle_r = 2.0
    
        
    # Title for Circle 1
    ax.text(circle_x, circle_y + circle_r + 0.3, "Mining-Relevant\nFunding Mechanisms",
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK, zorder=7)
    circle = Circle((circle_x, circle_y), circle_r, facecolor='white',
                   edgecolor='#666666', linewidth=2, zorder=5)
    ax.add_patch(circle)
    
    # Bars INSIDE circle
    x_loans = circle_x - 0.7
    x_grants = circle_x + 0.7
    y_base_inner = circle_y - 1.2
    h_scale = 2
    h_loans = h_scale * (amt_loans / amt_mining) if amt_mining > 0 else 0
    h_grants = h_scale * (amt_grants / amt_mining) if amt_grants > 0 else 0
    
    ax.bar(x_loans, h_loans, width=0.7, bottom=y_base_inner,
           color='#B0B0B0', edgecolor='white', linewidth=1.5, alpha=0.7, zorder=6)
    ax.bar(x_grants, h_grants, width=0.7, bottom=y_base_inner,
           color='#457B9D', edgecolor='white', linewidth=1.5, zorder=6)
    
    ax.text(x_loans, y_base_inner + h_loans + 0.15, f"{fmt_money(amt_loans)}\n{pct_of_mining(amt_loans):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=7)
    ax.text(x_loans, y_base_inner - 0.15, "Loans",
            ha='center', va='top', fontsize=7, color=DARK, zorder=7)
    ax.text(x_grants, y_base_inner + h_grants + 0.15, f"{fmt_money(amt_grants)}\n{pct_of_mining(amt_grants):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=7)
    ax.text(x_grants, y_base_inner - 0.15, "Grants",
            ha='center', va='top', fontsize=7, color=DARK, zorder=7)
    
    # LINES from Mining to Circle 1
    # Top-right corner → middle-left of circle
    circle_left = (circle_x - circle_r, circle_y)
    ax.plot([mining_top_right[0], circle_left[0]], 
            [mining_top_right[1], circle_left[1]],
            'k-', lw=1.5, alpha=0.7, zorder=4)
    
    # Bottom-right corner → bottom of circle
    circle_bottom = (circle_x, circle_y - circle_r)
    ax.plot([mining_bottom_right[0], circle_bottom[0]], 
            [mining_bottom_right[1], circle_bottom[1]],
            'k-', lw=1.5, alpha=0.7, zorder=4)
    
    # Grants bar corners for next generation
    grants_top_right = (x_grants + 0.35, y_base_inner + h_grants)
    grants_bottom_right = (x_grants + 0.35, y_base_inner)
    
    # GENERATION 2: Calculate Industry/Public
    grants_df = mining_df[grant_mask].copy()
    public_mask = grants_df.get('orientation', pd.Series()).fillna('').str.lower().isin(['public', 'both'])
    amt_public = grants_df.loc[public_mask, 'amt'].dropna().sum()
    amt_industry = grants_df.loc[~public_mask, 'amt'].dropna().sum()
    
    # CIRCLE 2 around Industry/Public (adjusted spacing)
    circle2_x = 8.2
    circle2_y = 5.0
    circle2_r = 2.0
    
        
    # Title for Circle 2
    ax.text(circle2_x, circle2_y + circle2_r + 0.3, "Grant Funding\nBeneficiaries",
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK, zorder=12)
    circle2 = Circle((circle2_x, circle2_y), circle2_r, facecolor='white',
                    edgecolor='#666666', linewidth=2, zorder=10)
    ax.add_patch(circle2)
    
    # Bars INSIDE circle 2 (higher z-order)
    x_industry = circle2_x - 0.7
    x_public = circle2_x + 0.7
    y_base_inner2 = circle2_y - 1.3
    h_scale2 = 2.2
    h_industry = h_scale2 * (amt_industry / amt_grants) if amt_grants > 0 else 0
    h_public = h_scale2 * (amt_public / amt_grants) if amt_grants > 0 else 0
    
    ax.bar(x_industry, h_industry, width=0.7, bottom=y_base_inner2,
           color='#B0B0B0', edgecolor='white', linewidth=1.5, alpha=0.7, zorder=11)
    ax.bar(x_public, h_public, width=0.7, bottom=y_base_inner2,
           color='#A8DADC', edgecolor='white', linewidth=1.5, zorder=11)
    
    ax.text(x_industry, y_base_inner2 + h_industry + 0.15, f"{fmt_money(amt_industry)}\n{pct_of_mining(amt_industry):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=12)
    ax.text(x_industry, y_base_inner2 - 0.15, "Industry",
            ha='center', va='top', fontsize=7, color=DARK, zorder=12)
    ax.text(x_public, y_base_inner2 + h_public + 0.15, f"{fmt_money(amt_public)}\n{pct_of_mining(amt_public):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=12)
    ax.text(x_public, y_base_inner2 - 0.15, "Public",
            ha='center', va='top', fontsize=7, color=DARK, zorder=12)
    
    # LINES from Grants to Circle 2 (higher z-order so visible)
    # Top-right corner → middle-left of circle 2
    circle2_left = (circle2_x - circle2_r, circle2_y)
    ax.plot([grants_top_right[0], circle2_left[0]], 
            [grants_top_right[1], circle2_left[1]],
            'k-', lw=1.5, alpha=0.7, zorder=8)
    
    # Bottom-right corner → bottom of circle 2
    circle2_bottom = (circle2_x, circle2_y - circle2_r)
    ax.plot([grants_bottom_right[0], circle2_bottom[0]], 
            [grants_bottom_right[1], circle2_bottom[1]],
            'k-', lw=1.5, alpha=0.7, zorder=8)
    
    # Public bar corners for next generation
    public_top_right = (x_public + 0.35, y_base_inner2 + h_public)
    public_bottom_right = (x_public + 0.35, y_base_inner2)
    
    # GENERATION 3: Calculate Bio/Non-Bio
    public_df = grants_df[public_mask].copy()
    bio_mask = public_df.get('biomining_fit', pd.Series()).fillna('').str.lower() == 'yes'
    amt_bio = public_df.loc[bio_mask, 'amt'].dropna().sum()
    amt_nonbio = public_df.loc[~bio_mask, 'amt'].dropna().sum()
    n_bio = bio_mask.sum()
    
    # CIRCLE 3 around Bio/Non-Bio (adjusted spacing)
    circle3_x = 12.2
    circle3_y = 7.0
    circle3_r = 2.0
    
        
    # Title for Circle 3
    ax.text(circle3_x, circle3_y + circle3_r + 0.3, "Bio-Relevant\nPublic Grants",
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK, zorder=17)
    circle3 = Circle((circle3_x, circle3_y), circle3_r, facecolor='white',
                    edgecolor='#666666', linewidth=2, zorder=15)
    ax.add_patch(circle3)
    
    # Bars INSIDE circle 3
    x_nonbio = circle3_x - 0.7
    x_bio = circle3_x + 0.7
    y_base_inner3 = circle3_y - 1.4
    h_scale3 = 2.4
    h_nonbio = h_scale3 * (amt_nonbio / amt_public) if amt_public > 0 else 0
    h_bio = h_scale3 * (amt_bio / amt_public) if amt_public > 0 else 0
    
    ax.bar(x_nonbio, h_nonbio, width=0.7, bottom=y_base_inner3,
           color='#B0B0B0', edgecolor='white', linewidth=1.5, alpha=0.7, zorder=16)
    ax.bar(x_bio, h_bio, width=0.7, bottom=y_base_inner3,
           color='#4CAF50', edgecolor='white', linewidth=1.5, zorder=16)
    
    ax.text(x_nonbio, y_base_inner3 + h_nonbio + 0.15, f"{fmt_money(amt_nonbio)}\n{pct_of_mining(amt_nonbio):.0f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=17)
    ax.text(x_nonbio, y_base_inner3 - 0.15, "Non-Bio",
            ha='center', va='top', fontsize=7, color=DARK, zorder=17)
    ax.text(x_bio, y_base_inner3 + h_bio + 0.15, f"{fmt_money(amt_bio)}\n{pct_of_mining(amt_bio):.1f}%",
            ha='center', va='bottom', fontsize=7, fontweight='bold', color=DARK, zorder=17)
    ax.text(x_bio, y_base_inner3 - 0.15, "Bio",
            ha='center', va='top', fontsize=7, fontweight='bold', color=DARK, zorder=17)
    
    # LINES from Public to Circle 3
    # Top-right corner → middle-left of circle 3
    circle3_left = (circle3_x - circle3_r, circle3_y)
    ax.plot([public_top_right[0], circle3_left[0]], 
            [public_top_right[1], circle3_left[1]],
            'k-', lw=1.5, alpha=0.7, zorder=13)
    
    # Bottom-right corner → bottom of circle 3
    circle3_bottom = (circle3_x, circle3_y - circle3_r)
    ax.plot([public_bottom_right[0], circle3_bottom[0]], 
            [public_bottom_right[1], circle3_bottom[1]],
            'k-', lw=1.5, alpha=0.7, zorder=13)
    
    # Summary
    fig.text(0.5, 0.92,
             f"{n_bio} Bio-Applicable Grants = {fmt_money(amt_bio)} ({pct_of_mining(amt_bio):.1f}% of mining funding)",
             ha='center', fontsize=11, color='#4CAF50', fontweight='bold')
    
    _page_num(fig, page_num)
    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


def _slide_bio_grants_bars_v7(pdf, all_federal_df, page_num):
    """
    Page 10 - Bar charts for bio-applicable public grants ONLY:
    - Mining type breakdown (vertical bars)
    - Research stage breakdown (vertical bars)
    Similar to v2 style but filtered to bio public grants only
    """
    fig = plt.figure(figsize=W)
    fig.suptitle("Bio-Relevant Public Grants — Mining Type & Research Stage Breakdown",
                 fontsize=13, fontweight='bold', y=0.97)
    
    # Filter to bio-applicable public grants
    mining_mask = all_federal_df['reason_to_remove'].isna()
    mining_df = all_federal_df[mining_mask].copy()
    
    grant_mask = mining_df.get('excel_instrument_type', pd.Series()).fillna('').str.contains(
        'Grant|FOA|NOFO', case=False, na=False)
    grants_df = mining_df[grant_mask].copy()
    
    # "Both" counts as Public!
    public_mask = grants_df.get('orientation', pd.Series()).fillna('').str.lower().isin(['public', 'both'])
    public_df = grants_df[public_mask].copy()
    
    bio_mask = public_df.get('biomining_fit', pd.Series()).fillna('').str.lower() == 'yes'
    bio_df = public_df[bio_mask].copy()
    
    if len(bio_df) == 0:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        ax.text(0.5, 0.5, "No bio-applicable public grants found",
                ha='center', va='center', fontsize=14, color=GREY,
                transform=ax.transAxes)
        fig.set_size_inches(W)
        pdf.savefig(fig)
        plt.close(fig)
        return
    
    # Create two subplots
    ax_mt = fig.add_axes([0.08, 0.54, 0.84, 0.35])  # Top: mining type
    ax_st = fig.add_axes([0.08, 0.12, 0.84, 0.35])  # Bottom: stage
    
    total_bio = bio_df['amt'].dropna().sum()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Mining Type Breakdown
    # ─────────────────────────────────────────────────────────────────────────
    def _norm_mining_type(v):
        if not isinstance(v, str): 
            return 'unknown'
        import re
        return re.split(r'[;,]', v)[0].strip()
    
    bio_df['_mt_clean'] = bio_df.get('mining_research_type', pd.Series()).apply(_norm_mining_type)
    mt_amt = bio_df.groupby('_mt_clean')['amt'].sum().sort_values(ascending=False)
    
    colors_mt = [MINING_TYPE_COLORS.get(cat, GREY) for cat in mt_amt.index]
    bars_mt = ax_mt.bar(range(len(mt_amt)), mt_amt.values,
                        color=colors_mt, edgecolor='white', linewidth=1.5, width=0.6)
    
    # Add labels
    scale_mt = mt_amt.max() if len(mt_amt) > 0 else 1
    for i, (bar, amt) in enumerate(zip(bars_mt, mt_amt.values)):
        pct = (amt / total_bio * 100) if total_bio > 0 else 0
        pct_str = f"{pct:.0f}%" if pct >= 1 else "< 1%"
        ax_mt.text(bar.get_x() + bar.get_width()/2, amt + scale_mt*0.02,
                  f"{fmt_money(amt)}\n({pct_str})",
                  ha='center', va='bottom', fontsize=8, linespacing=1.3, fontweight='bold')
    
    ax_mt.set_xticks(range(len(mt_amt)))
    ax_mt.set_xticklabels([c.replace('_', ' ').title() for c in mt_amt.index], fontsize=9)
    ax_mt.set_ylabel("Total Funding", fontsize=10)
    ax_mt.set_title("By Mining Research Type", fontsize=11, pad=8)
    ax_mt.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_mt.set_ylim(0, scale_mt * 1.25)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Research Stage Breakdown
    # ─────────────────────────────────────────────────────────────────────────
    stage_order = ['fundamental', 'early_technology_development', 
                   'applied_translational', 'applied translational', 'deployment']
    stage_labels_map = {
        'fundamental': 'Use Inspired\nResearch',
        'early_technology_development': 'Bench Scale\nTech Development',
        'applied_translational': 'Piloting',
        'applied translational': 'Piloting',
        'deployment': 'Deployment'
    }
    
    # Use stage_primary
    stage_col = bio_df.get('stage_primary', bio_df.get('research_stage', pd.Series()))
    st_amt = bio_df.groupby(stage_col)['amt'].sum()
    
    # Combine applied translational variants
    if 'applied translational' in st_amt.index and 'applied_translational' in st_amt.index:
        st_amt['applied_translational'] += st_amt['applied translational']
        st_amt = st_amt.drop('applied translational')
    
    # Reindex to standard order
    display_order = ['fundamental', 'early_technology_development', 'applied_translational', 'deployment']
    st_amt = st_amt.reindex(display_order, fill_value=0)
    display_labels = [stage_labels_map.get(s, s.replace('_', ' ').title()) for s in display_order]
    
    colors_st = [STAGE_COLORS.get(s, GREY) for s in display_order]
    bars_st = ax_st.bar(range(len(st_amt)), st_amt.values,
                        color=colors_st, edgecolor='white', linewidth=1.5, width=0.6)
    
    # Add labels
    scale_st = st_amt.max() if len(st_amt) > 0 else 1
    for i, (bar, amt) in enumerate(zip(bars_st, st_amt.values)):
        if amt > 0:
            pct = (amt / total_bio * 100) if total_bio > 0 else 0
            pct_str = f"{pct:.0f}%" if pct >= 1 else "< 1%"
            ax_st.text(bar.get_x() + bar.get_width()/2, amt + scale_st*0.02,
                      f"{fmt_money(amt)}\n({pct_str})",
                      ha='center', va='bottom', fontsize=8, linespacing=1.3, fontweight='bold')
    
    ax_st.set_xticks(range(len(display_labels)))
    ax_st.set_xticklabels(display_labels, fontsize=9)
    ax_st.set_ylabel("Total Funding", fontsize=10)
    ax_st.set_title("By Research Stage", fontsize=11, pad=8)
    ax_st.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_money_ax))
    ax_st.set_ylim(0, scale_st * 1.25)
    
    # Summary
    fig.text(0.5, 0.04,
             f"Total Bio-Applicable Public Grant Funding: {fmt_money(total_bio)} across {len(bio_df)} grants  •  "
             f"'Both' orientation counted with Public",
             ha='center', fontsize=9, style='italic', color=GREY)
    
    _page_num(fig, page_num)
    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# LAST PAGE — Use Inspired Research & Bench Scale Tech Development Grants
# ─────────────────────────────────────────────────────────────────────────────
# One-sentence summaries derived from notes/abstract + source URL links
EARLY_GRANTS = [
    {
        "title":   "Mine of the Future – Proving Ground Initiative (DE-FOA-0003390)",
        "agency":  "DOE – FECM / NETL",
        "amt":     "$80M",
        "stage":   "Bench Scale Tech Dev.",
        "bio":     True,
        "summary": "Funds up to 4 awards at TRL 3–6 to demonstrate novel in-situ and emerging "
                   "extraction technologies at field or pilot scale, with in-situ leaching "
                   "explicitly listed as an eligible R&D area.",
        "url":     "https://www.grants.gov/search-results-detail/360862",
    },
    {
        "title":   "ROCKS – Reliable Ore Characterization with Keystone Sensing (DE-FOA-0003592)",
        "agency":  "DOE – EERE",
        "amt":     "$40M",
        "stage":   "Bench Scale Tech Dev.",
        "bio":     False,
        "summary": "Supports development of advanced sensor-based ore characterization systems "
                   "to improve grade prediction and reduce waste in hard-rock mining operations.",
        "url":     "https://simpler.grants.gov/opportunity/914dfd2b-4dd0-44d1-8a7e-cc8624e2b7b5",
    },
    {
        "title":   "MAGNITO – Magnetic Acceleration Generating New Innovations (NOFO Aug 2025)",
        "agency":  "DOE – EERE",
        "amt":     "$20M",
        "stage":   "Bench Scale Tech Dev.",
        "bio":     False,
        "summary": "Advances next-generation magnetic separation technologies to recover critical "
                   "minerals from low-grade ores and secondary feedstocks more efficiently.",
        "url":     "https://simpler.grants.gov/opportunity/aad9fcc3-0fb7-4fb3-b36a-29accc831695",
    },
    {
        "title":   "TRACE-Ga – Technology for Recovery of Gallium (PIA Sep 2025)",
        "agency":  "DOE – EERE",
        "amt":     "$6M",
        "stage":   "Bench Scale Tech Dev.",
        "bio":     True,
        "summary": "Funds early-stage technologies to recover gallium from industrial waste "
                   "streams including semiconductor manufacturing effluents and coal byproducts.",
        "url":     "https://www.energy.gov/hgeo/funding-notice-technology-recovery-and-advanced-critical-material-extraction-gallium-trace-ga",
    },
    {
        "title":   "Advancing CMM Technology Development (DE-FOA-0002956)",
        "agency":  "DOE – FECM / NETL",
        "amt":     "$19.5M",
        "stage":   "Bench Scale Tech Dev.",
        "bio":     True,
        "summary": "Broad critical minerals and materials R&D FOA explicitly listing "
                   "bio-based recovery concepts as eligible — the most biomining-friendly "
                   "NETL funding opportunity to date (now closed).",
        "url":     "https://www.energy.gov/hgeo/funding-notice-advancing-technology-development-securing-domestic-supply-critical-minerals-and",
    },
    {
        "title":   "CORE-CM – Carbon Ore, Rare Earth & Critical Minerals Initiative (DE-FOA-0003077)",
        "agency":  "DOE – FECM / NETL",
        "amt":     "$45M",
        "stage":   "Use Inspired Research",
        "bio":     False,
        "summary": "Supports regional-scale, multi-institution collaborations to characterize "
                   "domestic coal and carbon ore resources as sources of rare earth and critical "
                   "mineral supply (now closed, funds allocated to regional hubs).",
        "url":     "https://www.energy.gov/hgeo/project-selections-foa-3077-regional-scale-collaboration-facilitate-domestic-critical-minerals",
    },
    {
        "title":   "Critical Minerals and Materials Accelerator (DE-FOA-0003588)",
        "agency":  "DOE – EERE / AMMTO",
        "amt":     "$50M",
        "stage":   "Bench Scale Tech Dev.",
        "bio":     True,
        "summary": "Accelerates development and scale-up of novel extraction and processing "
                   "technologies for critical minerals and materials, with up to $50M available "
                   "across multiple awards.",
        "url":     "https://www.grants.gov/search-results-detail/360290",
    },
]


def _slide_early_stage_grants(pdf, all_federal_df, page_num):
    """Last page — BIO-ONLY grants from CSV for ALL stages."""
    
    # Filter to bio-only
    mining_mask = all_federal_df['reason_to_remove'].isna()
    mining_df = all_federal_df[mining_mask].copy()
    
    grant_mask = mining_df.get('excel_instrument_type', pd.Series()).fillna('').str.contains(
        'Grant|FOA|NOFO', case=False, na=False)
    grants_df = mining_df[grant_mask].copy()
    
    public_mask = grants_df.get('orientation', pd.Series()).fillna('').str.lower().isin(['public', 'both'])
    public_df = grants_df[public_mask].copy()
    
    bio_df = public_df[
        public_df.get('biomining_fit', pd.Series()).fillna('').str.lower() == 'yes'
    ].copy()
    
    if len(bio_df) == 0:
        fig = plt.figure(figsize=W)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("white")
        ax.axis("off")
        ax.text(0.5, 0.5, "No bio-applicable public grants found",
                ha='center', va='center', fontsize=14, color=GREY)
        _page_num(fig, page_num)
        fig.set_size_inches(W)
        pdf.savefig(fig)
        plt.close(fig)
        return
    
    # Sort by stage then amount
    stage_order_map = {
        'fundamental': 0, 'early_technology_development': 1,
        'applied_translational': 2, 'applied translational': 2, 'deployment': 3
    }
    bio_df['_stage_sort'] = bio_df.get('stage_primary', 
        bio_df.get('research_stage', pd.Series())).map(stage_order_map).fillna(999)
    bio_df['_amt_sort'] = bio_df['amt'].fillna(0)
    bio_df = bio_df.sort_values(['_stage_sort', '_amt_sort'], ascending=[True, False])
    
    fig = plt.figure(figsize=W)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.axis("off")
    
    # Title
    ax.add_patch(plt.Rectangle((0, 0.91), 1, 0.09, transform=ax.transAxes,
                                facecolor=DARK, clip_on=False))
    ax.text(0.5, 0.955, "Bio-Relevant Public Grants — All Research Stages",
            ha="center", va="center", color="white", fontsize=15, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.918, 
            f"Complete landscape: {len(bio_df)} bio-relevant public grants (funded + unfunded opportunities)",
            ha="center", va="center", color=TEAL, fontsize=9, transform=ax.transAxes)
    
    # Columns
    COL = {"stage": 0.01, "amt": 0.11, "agency": 0.19, "title": 0.30, "notes": 0.70}
    HDR = [("Stage", 0.01), ("Amount", 0.11), ("Agency", 0.19), 
           ("Grant Title", 0.30), ("Notes", 0.70)]
    
    # Header
    hdr_y = 0.895
    ax.add_patch(plt.Rectangle((0, hdr_y - 0.018), 1, 0.028,
                                transform=ax.transAxes, facecolor="#E8EDF2", clip_on=False))
    for lbl, x in HDR:
        ax.text(x + 0.005, hdr_y - 0.002, lbl, transform=ax.transAxes, 
                fontsize=8, fontweight="bold", color=DARK, va="center")
    ax.plot([0.01, 0.99], [hdr_y - 0.018, hdr_y - 0.018],
            color="#AAAAAA", lw=0.6, transform=ax.transAxes)
    
    # Rows
    n = min(len(bio_df), 18)
    row_h = 0.86 / n
    top_y = 0.875
    
    for i, (idx, row) in enumerate(bio_df.head(n).iterrows()):
        y_center = top_y - row_h * i - row_h / 2
        y_bot_r = top_y - row_h * i - row_h
        
        bg = "#F5F8FA" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0.005, y_bot_r + 0.003), 0.99, row_h - 0.006,
                                   transform=ax.transAxes, facecolor=bg, clip_on=False))
        
        # Stage
        stage_val = row.get('stage_primary', row.get('research_stage', 'unknown'))
        if pd.isna(stage_val): stage_val = 'unknown'
        
        # Map to new display names
        stage_map = {
            'fundamental': 'Use Inspired',
            'early_technology_development': 'Bench Scale',
            'applied_translational': 'Piloting',
            'applied translational': 'Piloting',
            'deployment': 'Deployment'
        }
        stage_display = stage_map.get(str(stage_val).lower(), stage_val.replace('_', ' ').title())
        
        # Normalize piloting variants to single key for consistent color
        stage_val_normalized = stage_val
        if str(stage_val).lower() in ['applied translational', 'applied_translational']:
            stage_val_normalized = 'applied_translational'
        
        stage_color = STAGE_COLORS.get(stage_val_normalized, GREY)
        pill_h = min(row_h * 0.5, 0.04)
        ax.add_patch(plt.Rectangle((COL["stage"], y_center - pill_h/2), 0.09, pill_h,
                                   transform=ax.transAxes, facecolor=stage_color, alpha=0.90))
        ax.text(COL["stage"] + 0.045, y_center, stage_display,
                transform=ax.transAxes, fontsize=6.5, color="white",
                fontweight="bold", ha="center", va="center")
        
        # Amount (N/A not TBD)
        amt_val = row.get('amt', 0)
        if pd.isna(amt_val) or amt_val == 0:
            amt_display = "N/A"
            amt_color = GREY
        else:
            amt_display = fmt_money(amt_val)
            amt_color = DARK
        ax.text(COL["amt"] + 0.005, y_center, amt_display,
                transform=ax.transAxes, fontsize=8, color=amt_color,
                fontweight="bold", va="center")
        
        # Agency
        agency = row.get('excel_agency', row.get('funder', ''))
        if pd.isna(agency): agency = ''
        ax.text(COL["agency"] + 0.005, y_center, str(agency)[:20],
                transform=ax.transAxes, fontsize=7.5, color=GREY, va="center")
        
        # Title
        title = row.get('title', '')
        if pd.isna(title): title = ''
        title_short = textwrap.fill(str(title), width=45)
        ax.text(COL["title"] + 0.005, y_center, title_short,
                transform=ax.transAxes, fontsize=7, color=DARK, va="center", linespacing=1.2)
        
        # Notes - ONLY for N/A amounts
        if amt_display == "N/A":
            title_lower = str(title).lower()
            if 'prommis' in title_lower:
                notes = "Total funding not publicly available"
            elif 'ember' in title_lower:
                notes = "Rolling funding opportunity"
            else:
                notes = ""
            
            if notes:
                ax.text(COL["notes"] + 0.005, y_center, notes,
                        transform=ax.transAxes, fontsize=6.5, color='#666666',
                        va="center", style="italic")
        
        # Separator
        ax.plot([0.005, 0.995], [y_bot_r + 0.002, y_bot_r + 0.002],
                color="#DDDDDD", lw=0.4, transform=ax.transAxes)
    
    # Summary
    total_funded = bio_df[bio_df['amt'] > 0]['amt'].sum()
    n_funded = (bio_df['amt'] > 0).sum()
    n_unfunded = (bio_df['amt'].fillna(0) == 0).sum()
    
    fig.text(0.5, 0.02,
             f"{n_funded} funded ({fmt_money(total_funded)})  •  "
             f"{n_unfunded} unfunded (N/A)  •  Showing {n} of {len(bio_df)} grants",
             ha='center', fontsize=9, color=GREY, style='italic')
    
    _page_num(fig, page_num)
    fig.set_size_inches(W)
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def make_highlights_v8(
    kept, alldf, matdf, out_path, out_dir, label, outlier_note,
    program_kept=None, program_matdf=None,
    federal_df=None,
    all_federal_df=None,
    kept_raw=None,  # NEW: full dataset for Section 1 funnel
):
    """
    Build the full v9 highlights PDF.

    Section 1  — Federal Research & Innovation Grants
      Cover
      NEW FIRST PLOT: Categorization funnel (before page 1)
      Pages 1–7: Project Grants (Gen 2: public + industry)
      Pages 8–14: Public Grants (Gen 3: public only)

    Section 2  — Formula Grants
      Divider, Page 15

    Section 3  — Federal Future Funding Landscape
      Divider, Pages 16-18
    """
    _PAGE_W_PTS = 16 * 72
    _PAGE_H_PTS =  9 * 72
    with PdfPages(out_path) as pdf:

        # ── Title page ───────────────────────────────────────────────────────
        _title_page(pdf)

        # ── Section 1 ────────────────────────────────────────────────────────
        
        # CRITICAL: Filter to Gen 2 (Project) and Gen 3 (Public) for Section 1 plots
        if kept_raw is not None:
            kept_gen2 = _filter_to_gen2(kept_raw)  # Project (public + industry)
            kept_gen3 = _filter_to_gen3(kept_raw)  # Public only
            # Rebuild matdf with filtered data
            matdf_gen2 = build_matdf(kept_gen2)
            matdf_gen3 = build_matdf(kept_gen3)
        else:
            # Fallback if kept_raw not provided
            kept_gen2 = kept
            kept_gen3 = kept
            matdf_gen2 = matdf
            matdf_gen3 = matdf
        
        # Use Gen 2 for cover (shows combined stats)
        _cover(pdf, kept_gen2, label, outlier_note)

        page = 1
        
        # NEW PAGE 1: Categorization funnel WITHOUT Bio-Relevant bubble
        if kept_raw is not None:
            _slide_section1_categorization_funnel_no_bio(pdf, kept_raw, page); page += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 1A: ALL PROJECT GRANTS (Gen 2: Public + Industry)
        # ═══════════════════════════════════════════════════════════════════════
        section_1a_prefix = "Section 1A: All Project Grants (2016–2025)"
        _slide_portfolio_overview(pdf, kept_gen2, section_1a_prefix, page);    page += 1
        _slide_timeseries_combined(pdf, kept_gen2, section_1a_prefix, page);   page += 1
        _slide_bio_subcategory(pdf, kept_gen2, section_1a_prefix, page);       page += 1
        _slide_biosub_stage_abundance(pdf, kept_gen2, section_1a_prefix, page); page += 1
        _slide_stage_abundance(pdf, kept_gen2, section_1a_prefix, page);        page += 1
        _slide_miningtype_abundance(pdf, kept_gen2, section_1a_prefix, page);  page += 1
        _slide_materials(pdf, kept_gen2, matdf_gen2, section_1a_prefix, page);      page += 1
        _slide_stage_bio_analysis(pdf, kept_gen2, section_1a_prefix, page);    page += 1  # NEW
        
        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 1B: PUBLIC PROJECT GRANTS ONLY (Gen 3)
        # ═══════════════════════════════════════════════════════════════════════
        section_1b_prefix = "Section 1B: Public Project Grants (2016–2025)"
        
        # Section 1B divider (unnumbered)
        _section1b_divider(pdf, kept_gen3)
        
        # Full categorization funnel (WITH Bio-Relevant bubble) BEFORE Section 1B analysis
        if kept_raw is not None:
            _slide_section1_categorization_funnel(pdf, kept_raw, page); page += 1
        
        _slide_portfolio_overview(pdf, kept_gen3, section_1b_prefix, page);    page += 1
        _slide_timeseries_combined(pdf, kept_gen3, section_1b_prefix, page);   page += 1
        _slide_bio_subcategory(pdf, kept_gen3, section_1b_prefix, page);       page += 1
        _slide_biosub_stage_abundance(pdf, kept_gen3, section_1b_prefix, page); page += 1
        _slide_stage_abundance(pdf, kept_gen3, section_1b_prefix, page);        page += 1
        _slide_miningtype_abundance(pdf, kept_gen3, section_1b_prefix, page);  page += 1
        _slide_materials(pdf, kept_gen3, matdf_gen3, section_1b_prefix, page);      page += 1
        _slide_stage_bio_analysis(pdf, kept_gen3, section_1b_prefix, page);    page += 1  # NEW

        # ── Section 2 ────────────────────────────────────────────────────────
        _section_divider(pdf, kept_prog=program_kept)

        if program_kept is not None and len(program_kept) > 0:
            _slide_prog_overview(pdf, program_kept, page);    page += 1

        # ── Section 3 ────────────────────────────────────────────────────────
        if all_federal_df is not None and len(all_federal_df) > 0:
            _section3_divider(pdf, all_federal_df=all_federal_df)
            _slide_future_funnel_v8(pdf, all_federal_df, page);  page += 1
            _slide_bio_grants_bars_v7(pdf, all_federal_df, page);      page += 1
            _slide_early_stage_grants(pdf, all_federal_df, page);              page += 1

    print(f"  Highlights v9 PDF → {out_path}  ({page - 1} content pages + cover + dividers)")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")

    print("Loading data …")
    kept_raw, alldf = load_base_data()

    # Prefer the pipeline's explicit is_formula_grant flag (set in step2). Fall
    # back to award_type string-match if the flag column is missing.
    if "is_formula_grant" in kept_raw.columns:
        kept_raw["is_formula_grant"] = kept_raw["is_formula_grant"].apply(
            lambda v: str(v).strip().lower() in ("true", "1", "yes", "t"))
    else:
        kept_raw["is_formula_grant"] = (
            kept_raw.get("award_type", pd.Series(dtype=str)).fillna("").astype(str)
            .str.contains("FORMULA", case=False, na=False)
        )
    n_formula   = int(kept_raw["is_formula_grant"].sum())
    amt_formula = kept_raw.loc[kept_raw["is_formula_grant"], "amt"].sum()
    print(f"Formula grants (program section): {n_formula} ({fmt_money(amt_formula)})")

    program_kept = kept_raw[kept_raw["is_formula_grant"]].copy()
    kept         = kept_raw[~kept_raw["is_formula_grant"]].copy()

    # Fold in formula grants that step2 filtered for short abstracts — they're
    # real formula awards without usable abstracts. Matches the website's
    # Sankey Formula bucket (see build_viz_data.compute_totals).
    from biomining_shared import KEPT_CSV as _PIPE_CLASSIFIED_CSV
    _insuf_csv = os.path.join(
        os.path.dirname(_PIPE_CLASSIFIED_CSV), "mining_insufficient_abstract_all_years.csv"
    )
    if os.path.exists(_insuf_csv):
        from biomining_shared import _adapt_schema as _short_adapt
        _insuf = pd.read_csv(_insuf_csv, low_memory=False)
        if "is_formula_grant" in _insuf.columns:
            _is_f = _insuf["is_formula_grant"].apply(
                lambda v: str(v).strip().lower() in ("true", "1", "yes", "t"))
            _sf = _short_adapt(_insuf.loc[_is_f].copy())
            _pk_id = "unique_award_key" if "unique_award_key" in program_kept.columns else (
                "unique_key" if "unique_key" in program_kept.columns else None)
            _sf_id = "unique_award_key" if "unique_award_key" in _sf.columns else (
                "unique_key" if "unique_key" in _sf.columns else None)
            if _sf_id:
                _sf = _sf.drop_duplicates(_sf_id, keep="first")
            if _pk_id and _sf_id:
                _sf = _sf.loc[~_sf[_sf_id].astype(str).isin(
                    program_kept[_pk_id].astype(str))].reset_index(drop=True)
            if "award_amount" in _sf.columns and "amt" not in _sf.columns:
                _sf["amt"] = pd.to_numeric(_sf["award_amount"], errors="coerce").fillna(0.0)
            _sf["llm_biological"]        = False
            _sf["llm_bio_subcategory"]   = None
            _sf["llm_mining_type"]       = "remediation"
            _sf["llm_stage"]             = "deployment"
            _sf["llm_research_stage"]    = "deployment"
            _sf["llm_orientation"]       = "public_good"
            _sf["llm_materials"]         = "[]"
            _sf["materials_list"]        = [[] for _ in range(len(_sf))]
            _sf["is_formula_grant"]      = True
            _sf = _sf.reindex(columns=program_kept.columns)
            program_kept = pd.concat([program_kept, _sf], ignore_index=True, sort=False)
            print(f"  + {len(_sf)} formula grants from short-abstract pool folded into program_kept "
                  f"(now {len(program_kept)} total)")
    n_formula   = len(program_kept)
    amt_formula = program_kept["amt"].sum()
    print(f"Formula grants (after short-abstract fold-in): {n_formula} ({fmt_money(amt_formula)})")

    def _parse_bio(series):
        s = series.fillna(False)
        if s.dtype == object:
            return s.astype(str).str.lower().isin(["true", "1", "yes"])
        return s.astype(bool)

    kept["is_bio"]         = _parse_bio(kept["llm_biological"])
    program_kept["is_bio"] = _parse_bio(program_kept["llm_biological"])

    from biomining_shared import parse_materials
    if "materials_list" not in program_kept.columns:
        program_kept["materials_list"] = program_kept["llm_materials"].apply(parse_materials)

    matdf      = build_matdf(kept)
    matdf_prog = build_matdf(program_kept) if len(program_kept) > 0 else None

    # ── Section 3: Federal future funding landscape ───────────────────────────
    # biomining_shared exports FEDERAL_CSV with legacy-path fallback so the PDF
    # still builds during the transition to the new bulk pipeline.
    from biomining_shared import FEDERAL_CSV
    federal_df     = None   # kept (mining-relevant) rows — used for all charts
    all_federal_df = None   # all rows including manufacturing — used for total $ stats only
    if os.path.exists(FEDERAL_CSV):
        _fed_raw = pd.read_csv(FEDERAL_CSV, low_memory=False)
        # Apply the same schema adaptation so downstream slides see legacy column names.
        from biomining_shared import _adapt_schema as _fed_adapt_schema
        _fed_raw = _fed_adapt_schema(_fed_raw)
        _fed_raw["llm_biological"] = (
            _fed_raw["llm_biological"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
        _fed_raw["llm_keep"] = (
            _fed_raw["llm_keep"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
        all_federal_df = _fed_raw.copy()
        federal_df = _fed_raw[_fed_raw["llm_keep"] == True].copy()
        n_removed  = (~_fed_raw["llm_keep"]).sum()
        amt_removed = _fed_raw.loc[~_fed_raw["llm_keep"], "amt"].dropna().sum()
        print(f"Federal GapMap CSV loaded: {len(_fed_raw)} total rows from {FEDERAL_CSV}")
        print(f"  → {len(federal_df)} mining-relevant rows for charts")
        print(f"  → {n_removed} manufacturing/downstream rows excluded from charts ({fmt_money(amt_removed)})")
    else:
        print(f"⚠  Federal GapMap CSV not found at {FEDERAL_CSV} — Section 3 will be skipped")

    # PDF and any supplementary PNG figures land in the pipeline's visualization/ folder.
    _out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualization")
    os.makedirs(_out_dir, exist_ok=True)
    out_path = os.path.join(_out_dir, "biomining_highlights.pdf")
    label    = "formula grants removed — research & innovation only"
    note     = f"{n_formula} FORMULA GRANT (A) awards shown separately in Section 2"

    make_highlights_v8(
        kept, alldf, matdf,
        out_path=out_path, out_dir=_out_dir,
        label=label, outlier_note=note,
        program_kept=program_kept if len(program_kept) > 0 else None,
        program_matdf=matdf_prog,
        federal_df=federal_df,
        all_federal_df=all_federal_df,
        kept_raw=kept_raw,
    )
    print(f"✓ {out_path}")
