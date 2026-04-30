"""
step3_mining_two_stage_classifier_multiyear.py

Two-stage LLM classification pipeline for the biomining funding dataset.

Runs AFTER step2_mining_filter_multiyear.py. Reads:
    scripts/grant_classifier/output/mining_filtered_all_years.csv

STAGE 1  (Haiku, batch=20, ALL sufficient-abstract grants)
    - Binary KEEP/REMOVE decision for "is this genuinely a mining-research grant?"
    - Prompt = biomining KEEP/REMOVE rules verbatim from the old pipeline
      (hard filters for biogeochemical / uranium / hyper-accumulation keywords,
      BIOSENSING inclusion rule, TITLE-ONLY rule, ALL-CAPS rule, LOW CONFIDENCE
      rule). Output fields: keep, confidence, remove_reason.

STAGE 2-FORMULA  (Haiku, batch=20, KEPT formula grants ONLY)
    - Formula grants (is_formula_grant=True from step2) get ONLY
      biological + bio_subcategory classification — saves Sonnet cost
      for the non-formula pool. No stage/materials/mining_type/orientation.

STAGE 2-FULL  (Sonnet, batch=10, KEPT non-formula grants ONLY)
    - Full characterization across 6 axes:
        biological, bio_subcategory, mining_type, materials, stage,
        orientation, research_approach.
    - research_approach (collaborative_interdisciplinary / single_focus) is
      adopted from climate_biotech's step3 — replaces the old keyword-based
      interdisciplinary flag with an LLM call.
    - All other axis definitions and calibration examples are preserved
      verbatim from the old biomining classifier so classifications are
      consistent with the existing site data.

VALIDATION_GRANT_IDS = the 70 unique_award_keys from the old biomining
holdout (score_holdout.py) so TEST_MODE runs are directly comparable to the
old pipeline's accuracy calibration.

OUTPUTS (scripts/grant_classifier/output/):
    stage1_mining_relevance_all_years.csv        — Stage 1 decisions
    stage1_excluded_all_years.csv                — Stage 1 REMOVEs
    stage1_review_all_years.csv                  — Stage 1 low-confidence
    stage2_formula_all_years.csv                 — Stage 2-Formula results
    stage2_full_all_years.csv                    — Stage 2-Full results
    mining_llm_classified_all_years.csv          — merged final classification
    two_stage_classification_log_all_years.json  — raw LLM responses

Requirements:
    pip install anthropic pandas tqdm
    export ANTHROPIC_API_KEY=sk-ant-...   (or .env file in project root)
"""

import os
import re
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import anthropic


# =============================================================================
# API KEY LOADING
# =============================================================================
def _load_api_key() -> str:
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key

    project_root = script_dir.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key

    raise EnvironmentError(
        "\n\nNo Anthropic API key found. Fix with ONE of:\n"
        "  Option A (recommended): create a .env file in project root or script directory:\n"
        f"    echo 'ANTHROPIC_API_KEY=sk-ant-...' > {project_root}/.env\n"
        "  Option B: set the environment variable:\n"
        "    export ANTHROPIC_API_KEY=sk-ant-...\n"
        "\nGet your key at: https://console.anthropic.com/settings/keys\n"
    )


ANTHROPIC_API_KEY = _load_api_key()


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent           # scripts/grant_classifier/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                  # biomining_federal_grant_funding/
OUTPUT_DIR   = PROJECT_ROOT / "scripts" / "grant_classifier" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INFILE          = OUTPUT_DIR / "mining_filtered_all_years.csv"
SHORT_INFILE    = OUTPUT_DIR / "mining_insufficient_abstract_all_years.csv"

OUT_STAGE1           = OUTPUT_DIR / "stage1_mining_relevance_all_years.csv"
OUT_STAGE1_EXCL      = OUTPUT_DIR / "stage1_excluded_all_years.csv"
OUT_STAGE1_REVIEW    = OUTPUT_DIR / "stage1_review_all_years.csv"
OUT_STAGE2_FORMULA   = OUTPUT_DIR / "stage2_formula_all_years.csv"
OUT_STAGE2_FULL      = OUTPUT_DIR / "stage2_full_all_years.csv"
OUT_STAGE2_SHORT     = OUTPUT_DIR / "stage2_short_abstract_all_years.csv"
OUT_CLASSIFIED       = OUTPUT_DIR / "mining_llm_classified_all_years.csv"
OUT_LOG              = OUTPUT_DIR / "two_stage_classification_log_all_years.json"


# =============================================================================
# SETTINGS
# =============================================================================
# Stage 1: binary keep/remove (cheap + fast)
STAGE1_MODEL      = "claude-haiku-4-5-20251001"
STAGE1_BATCH_SIZE = 20
STAGE1_MAX_TOKENS = 4096

# Stage 2-Formula: biological + bio_subcategory ONLY (formula grants)
STAGE2_FORMULA_MODEL      = "claude-haiku-4-5-20251001"
STAGE2_FORMULA_BATCH_SIZE = 20
STAGE2_FORMULA_MAX_TOKENS = 4096

# Stage 2-Short: biological + bio_subcategory ONLY (non-formula short-abstract grants)
# Same lightweight path as Stage 2-Formula — we can only answer bio vs not with so
# little abstract text (median ~14 chars), so we ask for bio + bio_subcategory only.
STAGE2_SHORT_MODEL      = "claude-haiku-4-5-20251001"
STAGE2_SHORT_BATCH_SIZE = 20
STAGE2_SHORT_MAX_TOKENS = 4096

# Stage 2-Full: full 6-axis classification (non-formula kept grants)
STAGE2_FULL_MODEL      = "claude-sonnet-4-20250514"
STAGE2_FULL_BATCH_SIZE = 10
STAGE2_FULL_MAX_TOKENS = 4096

SLEEP_S     = 1.5
MAX_RETRIES = 4
RESUME      = True   # skip already-classified rows

# Test mode — use the 70 biomining holdout grant IDs from score_holdout.py
# so results are directly comparable to the old pipeline's validation scores.
TEST_MODE = False
TEST_SAMPLE_SIZE = 70  # fallback if none of the holdout keys match

VALIDATION_GRANT_IDS = [
    # NSF holdout keys (42 grants)
    "NSF::1554511", "NSF::1621759", "NSF::1629439", "NSF::1650152",
    "NSF::1709322", "NSF::1736995", "NSF::1805550", "NSF::1828523",
    "NSF::1845683", "NSF::1903685", "NSF::1917681", "NSF::1917682",
    "NSF::1937843", "NSF::1940373", "NSF::1945401", "NSF::1952817",
    "NSF::1953245", "NSF::1954336", "NSF::2052585", "NSF::2107177",
    "NSF::2107469", "NSF::2112301", "NSF::2127732", "NSF::2136875",
    "NSF::2150925", "NSF::2212919", "NSF::2238415", "NSF::2240755",
    "NSF::2304615", "NSF::2317822", "NSF::2317823", "NSF::2317826",
    "NSF::2342967", "NSF::2403854", "NSF::2404728", "NSF::2409142",
    "NSF::2423645", "NSF::2437985", "NSF::2446094", "NSF::2447809",
    "NSF::2502122", "NSF::2527909", "NSF::2550847",
    # USAspending holdout keys (28 grants)
    "USASpending::00350022", "USASpending::047907710",
    "USASpending::20186701528300", "USASpending::20193882129065",
    "USASpending::20253361044958", "USASpending::26K75IL000025",
    "USASpending::80NSSC21M0158", "USASpending::80NSSC24K0805",
    "USASpending::96347301", "USASpending::DEAR0001339",
    "USASpending::DEEE0006746", "USASpending::DESC0020204",
    "USASpending::F22AP03247", "USASpending::G25AC00336",
    "USASpending::GE146225", "USASpending::N000142112362",
    "USASpending::NI23SLSNXXXXG004", "USASpending::NNX15AP69H",
    "USASpending::NNX17AE87G", "USASpending::R00GM149822",
    "USASpending::R21ES034591", "USASpending::R35GM160470",
    "USASpending::R43ES034319", "USASpending::R44ES034319",
    "USASpending::R56DK106135", "USASpending::S26AF00008",
    "USASpending::SAQMIP23GR0356",
]


# =============================================================================
# STAGE 1 SYSTEM PROMPT — biomining KEEP/REMOVE rules, verbatim from old pipeline
# =============================================================================
# Note: The old pipeline had hard filters keyed on `query_used` (a column
# populated by the API-keyword-search retrieval step). In the new bulk-download
# pipeline there is no query_used column. The hard filter rules below have
# been rephrased from "if query_used = X, apply strict test" to "if the grant
# text contains X-style content, apply strict test" — same substantive rule,
# just sources the trigger from the abstract instead of a separate column.
# Calibration examples retain their original "| query: X |" annotations for
# reference; you will NOT see those annotations in the actual grant inputs.
STAGE1_SYSTEM_PROMPT = """You are an expert classifier for federal research grants in mining, metallurgy, and mineral processing.

Your ONLY job in this stage is to decide whether each grant is genuinely mining-relevant (KEEP) or not (REMOVE). You will NOT classify any other axes. Axis classification happens in a later stage only on KEPT grants.

## KEEP vs REMOVE

KEEP if: The grant's PRIMARY focus is on extracting, recovering, processing, characterizing, sensing, or remediating metals/minerals in a mining or mineral processing context. This includes biological approaches (bioleaching, biosorption, bioremediation, etc.), hydrometallurgy, mine waste treatment, mineral exploration, and critical mineral supply chains.

REMOVE if: The grant mentions a mining-adjacent term incidentally but is NOT primarily about mining or mineral extraction. Common false positives:
- "data mining" / "text mining" / "genome mining" (informatics — not mineral extraction)
- Materials science grants studying a metal's properties with no extraction/recovery angle
- Biomedical grants mentioning a metal (e.g., copper in enzymes, cobalt in vitamin B12)
- Agriculture grants mentioning phosphate as a plant nutrient, not a mined commodity
- Pure geology/earth science with no resource extraction angle
- Environmental grants studying metal contamination with no mining connection
- General-purpose lab equipment (SEM, mass spectrometer, NMR, ICP-OES, confocal microscope, GPU clusters) purchased for a university department where mining is incidentally listed as one possible use among many. CRITICAL: the mining application must appear explicitly in the ABSTRACT. If the abstract describes broad departmental use with no explicit mining application named, REMOVE. EXCEPTION: KEEP if the abstract explicitly names mining-relevant research tasks as primary uses (e.g. named mine sites, AMD-impacted watersheds, mine waste characterization).
- Extraterrestrial and space resource utilization grants — Mars regolith, lunar mining, asteroid mining — even if the abstract uses terrestrial mining vocabulary. The application is space, not terrestrial mining. REMOVE regardless of funder (including NASA).
- Battery manufacturing, cathode/anode engineering, energy storage device performance, or battery materials science — even if Li, Co, Ni, or Mn are central. The test is whether the grant is about GETTING those metals out of the ground or waste stream. If the grant starts after the metal has already been refined and is about what to do with it in a battery, REMOVE.
- Wastewater treatment grants where the PRIMARY goal is nutrient removal (nitrogen, phosphorus, eutrophication control) and metal/REE recovery is incidental, aspirational, or a minor side-benefit. Key test: if you removed all mention of metals/REE from the abstract, would the grant still make full sense as a wastewater study? If yes → REMOVE. KEEP only if metal/mineral/REE recovery is the stated primary objective, or if the waste water being treated is coming from a mined source, even if N/P co-removal is also mentioned.

## HARD FILTER RULES FOR SPECIFIC TEXT CONTENT

### If the grant's text emphasizes "biogeochemical" cycling
Grants using "biogeochemical" vocabulary often cover pure earth/ocean science unrelated to mining. Apply a STRICT test:
KEEP only if the abstract explicitly mentions at least ONE of: ore, leaching, heap, mine site, tailings, acid mine drainage, mineral extraction, metal recovery, hydrometallurgy, bioleaching, or a specific industrial mining process.
REMOVE if the grant is about: marine biogeochemistry, oceanography, paleoclimate, paleoceanography, agricultural nutrient cycles, carbon cycles, volcanic geochemistry, isotope tracing in natural systems, ecosystem nutrient dynamics, subsurface hypersaline environments, deep fracture fluid geochemistry, or nitrogen cycling — even if metals or rare earths appear as tracers or analytes.
Key distinction: studying how metals move through natural Earth systems = REMOVE. Using biogeochemical processes to extract or recover metals for industrial use = KEEP.

### If the grant's text mentions "hyperaccumulation" or "hyper-accumulation"
These concepts capture phytomining research (genuinely mining-relevant) but also pull in plant biology, nutrient sensing, and data analytics.
KEEP only if the abstract explicitly discusses: metal accumulation for recovery/harvesting, phytomining, phytoremediation of mine sites, or hyperaccumulator plants in extraction or recovery contexts.
REMOVE if the grant is about: plant nutrient physiology, agricultural phosphate/nutrient management, data analytics, drinking water treatment, or ecological contamination studies where metal accumulation is a pollution concern rather than a resource opportunity.

### If the grant prominently features "uranium"
Uranium vocabulary captures genuine uranium mining/processing grants but also pulls in agricultural, environmental health, and ecological grants where uranium appears as a contaminant or trace element.
KEEP only if the abstract's PRIMARY focus is on: uranium extraction, uranium ore processing, uranium mine waste remediation, uranium recovery from secondary sources, or instrumentation/sensing explicitly for uranium mining operations.
REMOVE if the grant is about: agricultural practices, crop science, soil health, farming extension programs, ecological restoration, public health exposure studies, or any other domain where uranium appears incidentally as a contaminant or analyte — even if phytoremediation of uranium-contaminated soils is mentioned as one sub-component among many non-mining activities.
Key distinction: uranium as the TARGET of a mining/recovery operation = KEEP. Uranium as a contaminant mentioned in passing within a predominantly non-mining grant = REMOVE.

## BIOSENSING INCLUSION RULE
KEEP a grant that involves biosensing ONLY if the sensor is explicitly designed for or deployed in a mining, hydrometallurgical, or mine-impacted environment — or if the grant explicitly states the sensor targets a metal contaminant from a mining source (AMD, tailings, mine drainage).
Do NOT keep a biosensing grant if the sensor detects a mining-adjacent element (phosphate, lithium, copper, manganese) but the application is agriculture, food science, ocean biogeochemistry, or general environmental monitoring with no explicit mining connection. Phosphorus sensors for agriculture or ocean monitoring = REMOVE. Metal sensors for general industrial wastewater (plating, circuit boards) with no mining application = REMOVE.
Exception: REMOVE if the analyte is purely a biological molecule (protein, DNA, glucose) with no connection to mined materials.

## ALL-CAPS ABSTRACTS (USAspending / USDA format)
Some grants — particularly from USDA and tribal/community programs via USAspending — have abstracts written entirely in uppercase. These often contain multiple sections: an administrative/educational framing at the top followed by substantive research content further down. READ THE FULL ABSTRACT before classifying, not just the opening sentences.
Common pattern to watch for: grant opens with curriculum, workforce, or community development language, then later sections explicitly describe acid mine drainage treatment, REE recovery, bioremediation of mine sites, or other clearly mining-relevant research activities.
If the research component (not just the framing) describes mining-relevant work, KEEP based on the research content.
Example: A USDA grant opening with "ADDRESSES SHORTAGE OF GRADUATES IN AGRICULTURAL SCIENCES" that later states "RESEARCH WILL EXPLORE ACID MINE DRAINAGE AND ASSESS REE BIOGEOCHEMICAL PROCESSES AND IF PASSIVE TREATMENT FOR ACID MINE DRAINAGE COULD BE USED AS A POTENTIAL RECOVERY SYSTEM FOR REE" — the research content is AMD + REE recovery = KEEP.

## TITLE-ONLY RULE (no abstract available)
Some grants have no abstract — the abstract field is empty or just repeats the title. In these cases:
KEEP with confidence:medium if the TITLE alone contains any of these unambiguous mining terms: tailings, tailing, heap leach, acid mine drainage, bioleaching, hydrometallurgy, beneficiation, ore processing, extractive metallurgy, mine waste, waste rock, flotation, chalcopyrite, sulfide ore, sulfide mineral.
EXCEPTION: "beneficiation" must appear in a mining context. "Beneficiation of building materials", "beneficiation of construction aggregates", or similar non-mining uses of the term = REMOVE even under the title-only rule.
These terms are mining-exclusive vocabulary — their presence in a title alone is sufficient evidence to keep the grant.
Do NOT default to REMOVE simply because no abstract is present — that unfairly penalizes USAspending grants which routinely omit abstracts.

## LOW CONFIDENCE RULE
When you are genuinely uncertain whether a grant is mining-relevant and would assign confidence:low, apply this tiebreaker:
- If the title and abstract contain NO explicit mining vocabulary (ore, tailings, leaching, mine, mineral extraction, AMD, heap, beneficiation in a mining context, etc.) → REMOVE with confidence:low and a brief remove_reason.
- Only assign keep:true with confidence:low if the abstract contains some mining-relevant content but the classification is uncertain. Do NOT use keep:true + confidence:low as a default holding pattern for grants that are clearly non-mining.

## OUTPUT FORMAT

Return ONLY a valid JSON array — no preamble, no explanation, no markdown fences.
One object per grant, in the same order as input.

[
  {
    "grant_id": "<unique_key from input>",
    "keep": true or false,
    "confidence": "high" | "medium" | "low",
    "remove_reason": null or "brief reason if REMOVE"
  }
]

## CALIBRATION EXAMPLES (keep / remove only)

Grant: "Hydrothermal Estuaries: What Sets the Hydrothermal Flux of Fe and Mn to the Oceans?" | query: biogeochemical | abstract: seafloor hot springs dispersing iron and manganese into ocean water; marine geochemistry of hydrothermal vents
→ keep:false, confidence:high, remove_reason:"Marine hydrothermal geochemistry — Fe and Mn are ocean tracers not mining targets. No ore, leaching, mine site, or extraction context."

Grant: "ABI Innovation: A New Framework to Analyze Plant Energy-related Phenomics Data" | query: mining | abstract: plant photosynthesis phenotyping platforms; crop productivity; light energy capture
→ keep:false, confidence:high, remove_reason:"Plant phenomics/photosynthesis framework — 'mining' appears in an informatics context. Zero mining vocabulary in abstract."

Grant: "HIGH EFFICIENCY SINTERING VIA BENEFICIATION OF THE BUILDING MATERIAL" | abstract: [title only]
→ keep:false, confidence:high, remove_reason:"Beneficiation of building material — construction/sintering context, not mineral extraction. 'Beneficiation' in construction context = REMOVE under title-only rule."

Grant: "Acquisition of an ICP-OES and IC to Increase Hydrological and Geochemical Research and Education" | abstract: instruments for geochemical and hydrological research at university; broad departmental use; no explicit mining application named
→ keep:false, confidence:high, remove_reason:"General university instrument acquisition — abstract describes broad departmental research with no explicit mining application named."

Grant: "An Economic, Sustainable, Green, Gold Isolation Process" | title_prefix: SBIR Phase II | abstract: improving gold mining process economics; replacing cyanide in gold isolation from tailings
→ keep:true, confidence:high

Grant: "Acidithiobacillus thiooxidans for sulfide mineral leaching in heap configurations"
→ keep:true, confidence:high

Grant: "Genome mining for novel natural product biosynthetic gene clusters in soil bacteria"
→ keep:false, confidence:high, remove_reason:"Genome mining — informatics/genomics, not mineral extraction."

Grant: "Copper chaperone proteins and their role in neurological disease"
→ keep:false, confidence:high, remove_reason:"Biomedical study of copper metabolism — no mining/extraction context."

Grant: "OXBOW TAILING RESTORATION PROJECT PHASE 3A" | abstract: [no abstract — title only]
→ keep:true, confidence:medium
Note: No abstract, but title contains "tailing" — unambiguous mining vocabulary. Apply title-only rule.

Grant: "US GEOTRACES GP-17-OCE: Molecular speciation of trace element-ligand complexes in the South Pacific Ocean" | abstract: marine siderophore biogeochemistry; characterizing metal-ligand complexes across ocean environments
→ keep:false, confidence:high, remove_reason:"Marine biogeochemistry — ocean metal-ligand characterization, not mining. Despite siderophore content, this is oceanographic science, not a mining-relevant extraction or remediation context."

Grant: "COMMUNITY PROJECT FUNDING ... VICTIMS OF MILLS TAILINGS EXPOSURE CANCER SCREENING PROGRAM" | abstract: funding cancer screenings for residents exposed to uranium mill tailings
→ keep:false, confidence:high, remove_reason:"Public health/medical services grant — mining is historical context only. Grant funds cancer screenings, not extraction, remediation, or any mining activity."
Note: Contrast with a grant that funds actual tailings remediation or characterization — those would KEEP as remediation.

Grant: "McCoy Creek Wet Meadow Restoration — habitat for Yellowstone Cutthroat Trout"
→ keep:false, confidence:high, remove_reason:"Ecological habitat restoration for fish — no mining content."

Grant: "FARMING PRACTICES ..." | abstract: agricultural extension — cover cropping, drought management, saline irrigation, soil health, crop production; one sentence mentions uranium phytoremediation incidentally
→ keep:false, confidence:high, remove_reason:"Agricultural extension grant — uranium as contaminant in a single incidental sentence. Single incidental mining mention in a predominantly non-mining grant = REMOVE."

Grant: "TRACE: A U.S.-DRC PARTNERSHIP TO ADDRESS CHILD AND FORCED LABOR IN SUPPLY CHAIN" | abstract: eliminate child and forced labor from DRC cobalt supply chains
→ keep:false, confidence:high, remove_reason:"Supply chain labor ethics program — policy/social intervention without a mining science or technology component."

Grant: "COMMINUTION OF REGOLITH USING MILLING FOR BENEFICIATION OF LUNAR EXTRACT (CRUMBLE)" | funder: NASA
→ keep:false, confidence:high, remove_reason:"Extraterrestrial/space resource utilization — lunar regolith. REMOVE regardless of NASA funder."

Grant: "INVESTIGATE PHOSPHORUS AND MINERAL EXTRACTION BIOMECHANISMS UTILIZED BY PALOVERDE TO GROW ON BASALT AND MARTIAN REGOLITH SIMULANT" | funder: NASA
→ keep:false, confidence:high, remove_reason:"Extraterrestrial/space resource utilization — Mars colonization application. REMOVE regardless of biological extraction mechanism."

Grant: "LTREB Renewal - River ecosystem responses to floodplain restoration" | abstract: long-term ecological study of river ecosystem recovery from historical metal pollution
→ keep:false, confidence:high, remove_reason:"River ecology research — metal pollution is historical context only, research focus is ecosystem recovery not mining remediation."

Grant: "IDBR: Development of a yeast-based continuous culture system for detecting bioavailable phosphate in water" | abstract: yeast biosensor for detecting bioavailable phosphate; framed as water quality and agricultural nutrient management tool; no mining application named
→ keep:false, confidence:high, remove_reason:"Phosphate biosensor for agricultural water quality — no mining process application named. BIOSENSING rule requires explicit mining context."

Grant: "GCR: Convergence on Phosphorus Sensing for Understanding Global Biogeochemistry and Enabling Pollution Management" | abstract: phosphorus sensors for global biogeochemical cycles, food/energy/water resources, agricultural runoff, pollution control
→ keep:false, confidence:high, remove_reason:"Phosphorus sensing for global biogeochemistry and agricultural pollution — no mining connection. BIOSENSING rule requires explicit mining/hydrometallurgical context."

Grant: "Testing the Existence of Magma Mush Zones ... with In situ Mineral Geochemistry" | abstract: volcanic geology; ore deposits as background context for crustal evolution, not as research target
→ keep:false, confidence:high, remove_reason:"Volcanic/magma geology — 'ore deposit' appears as background context, not as the research target."

Grant: "MRI: Acquisition of a Liquid-Chromatograph Mass Spectrometer (LC-MS) for Research and Training" | abstract: mentions REE ligand separation as a stated application
→ keep:true, confidence:high
Note: Explicit mining-relevant application language in the abstract = KEEP under instrument rule.

Grant: "OUR INTEGRATED PROJECT COMBINES TEACHING AND RESEARCH ..." | abstract: all-caps USDA; opens with curriculum/workforce language; later: "RESEARCH WILL EXPLORE ACID MINE DRAINAGE ... POTENTIAL RECOVERY SYSTEM FOR REE"
→ keep:true, confidence:high
Note: All-caps USDA abstract — read the full abstract; classify on research content, not administrative framing.

Grant: "IUCRC Phase III: Center for Resource Recovery & Recycling (CR3)" | abstract: multi-institution center for stewardship of ALL material resources — metals, plastics, bulk industrial materials
→ keep:false, confidence:high, remove_reason:"Broad all-sectors materials stewardship — mining is one domain among many. Remove mining and the center still exists = REMOVE."

Grant: "ERI: Engineering Amino Acid-Anchored 2D Silicoaluminophosphates and Aluminosilicates for Advanced Adsorption and Biomedical Applications" | abstract: 2D zeolites for catalysis, drug delivery, sensors, adsorption; REE adsorption is one application among many
→ keep:false, confidence:high, remove_reason:"Multi-domain materials platform — REE adsorption is one application among catalysis, drug delivery, and sensors. No primary mining focus."

Grant: "NSF Engines Development Award: Advancing microelectronics technologies (MO, KS)" | abstract: regional innovation ecosystem for microelectronics; mining mentioned once as potential application area
→ keep:false, confidence:high, remove_reason:"Microelectronics innovation hub — mining is incidental. When mining is one of many applications in a primarily non-mining program, REMOVE."

Grant: "Conference: Resilient Supply of Critical Minerals, Rolla, MO" | abstract: conference covering exploration, material flow, sustainability, mining methods, policy, mineral economics
→ keep:true, confidence:high
Note: Conference on mining topics is mining-relevant.

Grant: "Microbial iron reduction in the formation of iron ore caves" | abstract: geomicrobiology of cave formation through microbial H2S oxidation
→ keep:false, confidence:high, remove_reason:"Geomicrobiology of cave formation — iron ore appears as geological substrate being studied, not as a mining target."

Grant: "Developing scale-up manufacturing of engineered waste coal ash based lightweight aggregate for concrete applications"
→ keep:false, confidence:high, remove_reason:"Coal ash used as construction material — materials/construction engineering, not critical mineral recovery."
Note: Coal ash for REE/critical mineral recovery = KEEP. Coal ash as construction material = REMOVE.

Grant: "Heavy Mineral Mining Impeller Accelerated Separator" | title_prefix: SBIR Phase I | abstract: centrifugal gravity separator for mining
→ keep:true, confidence:high

Grant: "Mine Waste Inventory and Phase 1 Mine Waste Characterization in Washington State" | abstract: field sampling + inventory of critical minerals in mine waste
→ keep:true, confidence:high

Grant: "Kentucky Geological Survey FY2023 Geologic Data Preservation Project" | abstract: digitize/catalog existing geologic records for future mineral resource research
→ keep:true, confidence:high

Grant: "REU Site: InSPiRES: Integrated Circuits and Semiconductor Processes" | abstract: undergraduate research in integrated circuits and semiconductors; copper in circuits
→ keep:false, confidence:high, remove_reason:"Semiconductor engineering REU — copper in circuit manufacturing, not mining. REU in non-mining discipline with incidental metal mention = REMOVE."

Grant: "REU Site: Back to the Future at the South Dakota School of Mines and Technology" | abstract: 10-week undergraduate research in mineral beneficiation, resource recovery, strategic metals, metallurgical engineering
→ keep:true, confidence:high
Note: REU at a mining school with substantive mining research content = KEEP.

Grant: "Role of protein nanowires in metal cycling and mineralization" | abstract: Geobacter using pili to reductively precipitate Co/Cd/Ag as minerals
→ keep:true, confidence:high

Grant: "Spectroelectrochemical Measurements on Intact Microorganisms Under Oxic and Anoxic Conditions" | abstract: microorganisms exchanging electrons with solid minerals within an ore body
→ keep:true, confidence:high

Grant: "Bioavailability of mineral-associated molybdenum as a cofactor of Nif nitrogenase for N2 fixation" | abstract: N2 fixation biochemistry; molybdenum as enzyme cofactor in soil microbes
→ keep:false, confidence:high, remove_reason:"Nitrogen fixation biochemistry — molybdenum appears as enzyme cofactor, not mining target."

Grant: "MODELING ION SELECTIVITY IN EF-HAND PROTEINS — CALMODULIN, LANMODULIN, TROPONIN C" | abstract: biomedical structural biology; lanthanide used as calcium surrogate
→ keep:false, confidence:high, remove_reason:"Biomedical structural biology — lanthanide used as calcium surrogate, not as mining target."

IMPORTANT:
- Your response must be ONLY the JSON array
- Return one object per grant in the same order as input
- Include EVERY grant in the response
"""


# =============================================================================
# STAGE 2-FORMULA SYSTEM PROMPT — formula grants get biological + bio_subcategory ONLY
# =============================================================================
STAGE2_FORMULA_SYSTEM_PROMPT = """You are an expert classifier for federal research grants in mining, metallurgy, and mineral processing.

These grants are FORMULA GRANTS — block-allocation awards distributed by statutory formula (not competitively reviewed research). They have already passed the mining-relevance check. In this stage your ONLY job is to determine whether the grant is biological in nature and, if so, which bio subcategory applies. Do NOT classify mining_type, stage, materials, or orientation — those are assigned statically for formula grants downstream.

## BIOLOGICAL CLASSIFICATION

Assign biological=true ONLY if the grant's PRIMARY research mechanism is explicitly biological AND that biological mechanism is directly applied to a mining-relevant goal. Do NOT assign biological=true simply because a biological term (bacteria, enzyme, microbe, plant) appears somewhere in a long abstract alongside a mining keyword. The biological mechanism must be the CORE of the grant's approach to mining, extraction, or remediation.

BIOLOGICAL=FALSE even if biological terms appear: general ecology studies, natural biogeochemical cycling, nitrogen fixation studies where a metal appears only as an enzyme cofactor, marine biology, agricultural biology, plant physiology where metal uptake is a contamination concern not a recovery goal, carbon cycling, medical/biomedical studies of metal-protein interactions.

## BIO SUBCATEGORIES (assign one when biological=true, else null)

- **bioremediation**: Microorganisms or plants to detoxify/remove contaminants from mining-impacted environments. Focus is ecological restoration or environmental management of mine waste, not metal recovery. Phytoremediation (plants used to clean contaminated soil) = bioremediation.
  CRITICAL DISTINCTION — bioremediation vs bioseparation_biosorption:
  Ask: what is the PRIMARY GOAL of the biological mechanism?
  → CLEANUP / DETOXIFICATION of a contaminated site → bioremediation
  → RECOVERY / HARVEST of metal as a product → bioseparation_biosorption

- **bioleaching**: Microbial oxidation or acid generation acting on a SOLID SOURCE (ore, tailings, coal ash, refractory ore) to solubilize metals for recovery. Includes direct sulfide mineral oxidation (e.g., Acidithiobacillus leaching chalcopyrite for copper) AND microbial pre-treatment of refractory gold ores. Key indicators: sulfide oxidation, biolixiviant, refractory ore pre-treatment, acid-generating bacteria acting on solid minerals, Acidithiobacillus, Leptospirillum.

- **bioseparation_biosorption**: Biological or bio-inspired molecules, organisms, or materials that selectively bind, adsorb, or separate metals — including fundamental chemistry research on metal-ligand selectivity mechanisms and protein-metal complexes. Phytomining (plants harvesting metal for recovery) = bioseparation_biosorption (with mining_type: extraction).

- **bioprecipitation_mineral_formation**: Controlled microbial precipitation of metals as minerals (e.g., sulfate-reducing bacteria precipitating metal sulfides). NOT remediation-focused — emphasis on product formation. Metals move FROM solution INTO a solid mineral product. OPPOSITE direction from bioleaching.

- **biosensing**: Biological tools for sensing metals, redox state, or analytes in mining streams/environments. Includes molecular assays as process control signals, assay development, benchmarking. Must connect explicitly to a mining, hydrometallurgical, or mine-impacted environment.

BIOLEACHING vs BIOSEPARATION_BIOSORPTION — KEY DISAMBIGUATION:
- bioleaching → microbe acts on a SOLID SOURCE and DISSOLVES metals into solution. The SOLID is the input.
- bioseparation_biosorption → biomolecules SELECTIVELY BIND metals ALREADY IN SOLUTION.
- MULTI-STEP PROCESSES: classify by the FIRST / PRIMARY extraction step, not downstream purification. Bacterial leaching of a solid followed by bacterial purification → bioleaching.

## OUTPUT FORMAT

Return ONLY a valid JSON array — no preamble, no explanation, no markdown fences.
One object per grant, in the same order as input.

[
  {
    "grant_id": "<unique_key from input>",
    "biological": true or false,
    "bio_subcategory": null or one of [bioremediation, bioleaching, bioseparation_biosorption, bioprecipitation_mineral_formation, biosensing],
    "confidence": "high" | "medium" | "low"
  }
]

IMPORTANT:
- Your response must be ONLY the JSON array
- Return one object per grant in the same order as input
- biological=true requires PRIMARY biological mechanism applied to a mining goal
- bio_subcategory must be null when biological=false
"""


# =============================================================================
# STAGE 2-SHORT SYSTEM PROMPT — non-formula short-abstract grants get
# biological + bio_subcategory ONLY (full 6-axis classification is impossible
# with <92 character abstracts — typical abstract length in this pool is ~14
# characters, so judgements must be driven primarily by the title).
# =============================================================================
STAGE2_SHORT_SYSTEM_PROMPT = """You are an expert classifier for federal research grants in mining, metallurgy, and mineral processing.

These grants passed the mining-relevance keyword filter but have VERY SHORT abstracts (fewer than 92 characters — often just placeholder text, a repeat of the title, or "NOT APPLICABLE"). They are NOT formula grants. You are only being asked to classify whether the grant is biological in nature and, if so, which bio subcategory applies. You CANNOT classify mining_type, stage, materials, orientation, or research_approach with so little abstract context — those remain unknown for this pool.

Because the abstract is usually unusable, the grant's TITLE is your primary signal. The title often names the mechanism, organism, or target clearly enough to reach a bio/non-bio judgement (e.g., "DISCOVERY OF MICROBES FOR THE EXTRACTION OF RARE EARTH ELEMENTS" is clearly biological; "ION EXCHANGE MATERIALS FOR LITHIUM EXTRACTION" is clearly not).

## BIOLOGICAL CLASSIFICATION

Assign biological=true ONLY if the grant's PRIMARY research mechanism is explicitly biological AND that biological mechanism is directly applied to a mining-relevant goal. Apply the same strict rule as the main pipeline — do not infer biology from adjacency. If the title is ambiguous (e.g., contains "biology" only as a department name, or a generic word like "organic") and the abstract doesn't clarify, default to biological=false with low confidence.

BIOLOGICAL=FALSE even if biological terms appear in adjacency: general ecology studies, natural biogeochemical cycling, nitrogen fixation where a metal is only a cofactor, marine biology, agricultural biology, plant physiology where metal uptake is a contamination concern rather than a recovery goal, carbon cycling, medical/biomedical studies of metal-protein interactions.

## BIO SUBCATEGORIES (assign one when biological=true, else null)

- **bioremediation**: Microorganisms or plants to detoxify / remove contaminants from mining-impacted environments. Focus is ecological restoration of mine waste, not metal recovery. Phytoremediation of contaminated soil = bioremediation.

- **bioleaching**: Microbial oxidation or acid generation acting on a SOLID SOURCE (ore, tailings, coal ash, refractory ore) to solubilize metals for recovery. Includes Acidithiobacillus / Leptospirillum leaching of sulfide minerals, microbial pre-treatment of refractory gold ores.

- **bioseparation_biosorption**: Biological or bio-inspired molecules, organisms, or materials that selectively bind, adsorb, or separate metals. Phytomining (plants harvesting metal for recovery) = bioseparation_biosorption.

- **bioprecipitation_mineral_formation**: Controlled microbial precipitation of metals as minerals (e.g., sulfate-reducing bacteria precipitating metal sulfides). NOT remediation-focused — emphasis on product formation.

- **biosensing**: Biological tools for sensing metals, redox state, or analytes in mining streams / environments. Molecular assays used as process control signals in mining, hydrometallurgical, or mine-impacted contexts.

BIOLEACHING vs BIOSEPARATION_BIOSORPTION — KEY DISAMBIGUATION:
- bioleaching → microbe acts on a SOLID SOURCE and DISSOLVES metals into solution. The SOLID is the input.
- bioseparation_biosorption → biomolecules SELECTIVELY BIND metals ALREADY IN SOLUTION.
- MULTI-STEP PROCESSES: classify by the FIRST / PRIMARY extraction step, not downstream purification.

## CONFIDENCE

With so little text, confidence should skew "low" or "medium". Use "high" ONLY when the title unambiguously names a biological mechanism applied to a mining goal (e.g., "Acidithiobacillus Leaching of Chalcopyrite for Copper Recovery").

## OUTPUT FORMAT

Return ONLY a valid JSON array — no preamble, no explanation, no markdown fences.
One object per grant, in the same order as input.

[
  {
    "grant_id": "<unique_key from input>",
    "biological": true or false,
    "bio_subcategory": null or one of [bioremediation, bioleaching, bioseparation_biosorption, bioprecipitation_mineral_formation, biosensing],
    "confidence": "high" | "medium" | "low"
  }
]

IMPORTANT:
- Your response must be ONLY the JSON array
- Return one object per grant in the same order as input
- Title is typically the strongest signal; abstracts may be unusable
- biological=true requires PRIMARY biological mechanism applied to a mining goal
- bio_subcategory must be null when biological=false
- Default to biological=false with low confidence when signal is ambiguous
"""


# =============================================================================
# STAGE 2-FULL SYSTEM PROMPT — full 6-axis classification for non-formula kept grants
# =============================================================================
# Preserves biomining axis definitions verbatim from the old classifier plus
# the full calibration example set. Adds one NEW axis: research_approach
# (adopted from climate_biotech's step3 Stage 2 prompt).
STAGE2_FULL_SYSTEM_PROMPT = """You are an expert classifier for federal research grants in mining, metallurgy, and mineral processing.

These grants have already passed the mining-relevance check. Your job in this stage is to characterize each grant across SIX axes:
  1. biological + bio_subcategory
  2. mining_type
  3. materials
  4. stage
  5. orientation
  6. research_approach

Calibration examples below use the form "query: X" to denote how the grant was retrieved in the original API-based pipeline. You will NOT see "query:" annotations in the actual grant inputs — that metadata is only for reference in the examples.

## AXIS 1 — BIOLOGICAL SUBCATEGORY
Assign biological=true ONLY if the grant's PRIMARY research mechanism is explicitly biological AND that biological mechanism is directly applied to a mining-relevant goal. Do NOT assign biological=true simply because a biological term (bacteria, enzyme, microbe, plant) appears somewhere in a long abstract alongside a mining keyword. The biological mechanism must be the CORE of the grant's approach to mining, extraction, or remediation.

BIOLOGICAL=FALSE even if biological terms appear: general ecology studies, natural biogeochemical cycling, nitrogen fixation studies where a metal appears only as an enzyme cofactor, marine biology, agricultural biology, plant physiology where metal uptake is a contamination concern not a recovery goal, carbon cycling, medical/biomedical studies of metal-protein interactions.

- bioremediation: Microorganisms or plants to detoxify/remove contaminants from mining-impacted environments. Focus is ecological restoration or environmental management of mine waste, not metal recovery. Fundamental mechanism research (e.g. siderophore production, microbial iron/manganese cycling) is only in scope if the abstract explicitly connects the mechanism to mine site remediation, AMD treatment, or tailings management. Ocean, marine, agricultural, and natural ecosystem studies are OUT OF SCOPE even if the mechanism is theoretically relevant.
  CRITICAL DISTINCTION — bioremediation vs bioseparation_biosorption:
  Ask: what is the PRIMARY GOAL of the biological mechanism?
  → CLEANUP / DETOXIFICATION of a contaminated site or water source → bioremediation
  → RECOVERY / HARVEST of metal as a product → bioseparation_biosorption
  This distinction holds even when the abstract mentions BOTH cleanup and metal production. If the framing is "restore this legacy mine site" or "remove heavy metal contamination", classify as bioremediation — the secondary mention of metal value does not override the cleanup goal.
  Phytoremediation (plants used to clean contaminated soil) = bioremediation, even if metal uptake is mentioned.
  Phytomining (plants used to harvest metal for recovery) = bioseparation_biosorption (mining_type: extraction).
  Bio-inspired ligands or sorbents used to REMOVE uranium/heavy metals from contaminated water at a legacy mine site = bioremediation, NOT bioseparation_biosorption — the goal is contaminant removal, not metal product recovery.

- bioleaching: Microbial oxidation or acid generation acting on a SOLID SOURCE (ore, tailings, coal ash, refractory ore) to solubilize metals for recovery. Includes both direct sulfide mineral oxidation (e.g., Acidithiobacillus leaching chalcopyrite for copper) AND microbial pre-treatment of refractory gold ores to liberate gold prior to cyanidation (previously called biooxidation). These are the same fundamental mechanism — microbial oxidation of a solid mineral substrate — and are grouped together. Key indicators: sulfide oxidation, biolixiviant, refractory ore pre-treatment, acid-generating bacteria acting on solid minerals, Acidithiobacillus, Leptospirillum.

- bioseparation_biosorption: Biological or bio-inspired molecules, organisms, or materials that selectively bind, adsorb, or separate metals — including fundamental chemistry research on metal-ligand selectivity mechanisms and protein-metal complexes. KEEP if selective binding or separation of a mined commodity (REEs, lanthanides, transition metals, actinides) is the core research question, regardless of whether the study context is a mine leachate or a lab flask. Examples: ligand design for REE selectivity, organometallic complexation of cerium/lanthanides, cyclic peptide metal-binding motifs, protein-based metal chelation chemistry.

- bioprecipitation_mineral_formation: Controlled microbial precipitation of metals as minerals (e.g., sulfate-reducing bacteria precipitating metal sulfides). NOT remediation-focused — emphasis on product formation.
  CRITICAL DIRECTIONALITY: metals move FROM solution INTO a solid mineral product. This is the OPPOSITE direction from bioleaching (solid → solution). Key indicators: "reductive precipitation", "mineralization", "mineral phase", "microbial reduction of dissolved metal cations", protein nanowires or pili precipitating metals, XANES/EXAFS characterizing mineral products formed by microbes. Do NOT assign bioleaching when the abstract describes microbes precipitating or mineralizing dissolved metals — that is bioprecipitation_mineral_formation.

- biosensing: Biological tools for sensing metals, redox state, or analytes in mining streams/environments. Includes molecular assays as process control signals, assay development, benchmarking.

BIOLEACHING vs BIOSEPARATION_BIOSORPTION — KEY DISAMBIGUATION:
Ask: what is the microbe/biomolecule acting ON, and what is it doing?
- bioleaching → microbe acts on a SOLID SOURCE (ore, tailings, coal ash, refractory ore) and DISSOLVES metals into solution via acid generation, redox cycling, or ligand production. The solid is the input. Key indicators: "acid-producing bacteria", "biolixiviant", "sulfide oxidation", "metal solubilization from ore/tailings", "refractory ore pre-treatment", Acidithiobacillus, natural microbial metabolism producing acidic conditions. This category also covers what was previously called biooxidation (microbial pre-treatment of refractory gold ores) — both involve microbes oxidizing a solid mineral substrate.
- bioseparation_biosorption → biomolecules (proteins, peptides, siderophores, engineered ligands, whole cells) SELECTIVELY BIND or CAPTURE metals that are already dissolved or mobilized. The metal is already in solution. Key indicators: "selective binding", "metal-protein interaction", "siderophore", "engineered protein", "biosorption from solution", "purification of metal-bearing solution".
- DEFAULT: microbes acting on a SOLID to dissolve metals → bioleaching. Biomolecules capturing metals from SOLUTION → bioseparation_biosorption.
- MULTI-STEP PROCESSES: When a grant describes a pipeline with multiple biological steps (e.g., step 1: bacteria leach metals from solid waste; step 2: separate REEs from solution using selective bacteria), classify by the FIRST / PRIMARY extraction step, not the downstream purification step. If step 1 is bacterial leaching from a solid source material → bioleaching, even if later steps involve selective binding or separation. Only assign bioseparation_biosorption when there is NO upstream leaching step and the grant's core innovation is the selective capture/separation mechanism itself.

## AXIS 2 — MINING RESEARCH TYPE
Assign the ONE type that best describes the PRIMARY purpose of the grant. When a grant includes multiple activities, classify by the dominant activity as determined by the core research question and the majority of described work. When the abstract is ambiguous, use the TITLE as the tiebreaker.

Key decision rule — ask: "Is this grant DOING something or BUILDING/CONVENING something?"
- DOING research (conducting experiments, running processes, remediating sites, characterizing deposits) → extraction / downstream_processing / remediation / exploration
- BUILDING something (tool, physical infrastructure instrument, database, platform, sensor, software) → infrastructure
  Exception: if the sole stated purpose of the tool is remediation → remediation. Multi-use tools → infrastructure.
- CONVENING or STUDYING (conferences, workshops, shared centers, policy/economic/sociological studies, workforce development) → collaborative_tools
- SPANNING the full mining value chain with no dominant focus → general (NOT for SBIR grants — those always have a specific technical goal; assign by that goal)

- exploration: Locating and characterizing subsurface mineral resources — geophysical, geochemical, remote sensing approaches. ALSO includes grants where the primary deliverable is NEW mining-relevant geoscience data generated through active field work: physical site visits, geochemical sampling of mine waste or mineral deposits, resource estimation for extraction, or geophysical surveys. KEY OUTPUT TEST: if the grant generates new field data (samples, analyses, resource estimates) about mining-relevant geology → exploration, even if results are subsequently archived into a shared database. If the work is purely digitizing, compiling, or archiving EXISTING records without generating new field data → collaborative_tools.

- extraction: Getting metal/mineral out of ANY source material — primary ore, waste streams, e-waste, ash, brine, soil, secondary feedstocks. Includes bioleaching, heap leaching, hydrometallurgy, selective binding/separation of metals from a source material, phytomining. The input is a solid or mixed-phase source, the output is a metal or metal-bearing solution. Bioseparation research on metal selective binding = extraction. Phytomining (plants used to harvest metals from soil) = extraction, bio_subcategory=bioseparation_biosorption.

- downstream_processing: Refining, upgrading, or purifying already-extracted material — electrowinning, smelting, flotation of concentrates, solvent extraction of leachates, purification of a metal-bearing solution into pure metal. The input is an already-extracted concentrate or solution; the output is a purified metal or compound.

- remediation: (1) Directly remediating a mining-impacted site, waste stream, acid mine drainage, or tailings. (2) Environmental monitoring/ecological assessment where mining waste streams are the explicit cause. (3) Tools whose SOLE stated purpose is mine site environmental management or remediation monitoring.
  CONTAMINANT REMOVAL OVERRIDES BENEFICIATION: If a title or abstract frames the primary goal as "contaminant removal", "pollution cleanup", or "remediation" of a mining-impacted material, classify as remediation even if the word "beneficiation" also appears. The contaminant/cleanup framing takes precedence over beneficiation vocabulary when both are present.

- infrastructure: Physical tools, instruments, assays, sensors, equipment, software, and platforms built to enable mining operations or research. Includes process simulation tools, ML platforms for mining, sensors for mining applications, and purpose-built analytical instruments. ALSO INCLUDES: grants funding construction of physical production facilities or plants — if the primary deliverable is a physical facility (plant, processing centre, hub), classify as infrastructure regardless of what process occurs inside of proposed building. BOUNDARY: if a grant is primarily conducting research AND produces a tool as a secondary output, classify by the primary research activity not the secondary deliverable.

- collaborative_tools: Conferences, workshops, regional innovation centers, shared databases, atlases, shared inventories, workforce development programs, policy studies, economic studies, and sociological studies about mining. Also includes grants whose PRIMARY deliverable is digitizing, compiling, or archiving EXISTING mining-relevant records into a shared resource. Key test: is the primary output a convening, a policy recommendation, a workforce program, or a shared community resource built from existing data? STRICT EXCLUSIONS — do NOT assign collaborative_tools to: (1) grants generating NEW field data (sampling, geochemical analysis, resource estimates) even if results are subsequently archived → exploration; (2) REU/undergraduate training programs where mining is incidental to a broader science/engineering mission; (3) grants where mining is incidental to a broader technology or materials science mission.

- general: Grant genuinely spans the full mining value chain (exploration + extraction + processing + remediation) with no dominant focus. Use sparingly — most grants have a primary focus. NEVER assign to SBIR grants (always classify by specific technical goal).

COLLABORATIVE_TOOLS vs INFRASTRUCTURE — KEY DISAMBIGUATION:
- Database, atlas, inventory, or shared knowledge resource whose deliverable is used by MANY organizations as a reference → collaborative_tools
- Sensor, software tool, instrument, or platform built to perform a specific mining OPERATION or research TASK → infrastructure
- Quick test: "Is this something many organizations look up, or something one organization uses to do work?" → look up = collaborative_tools, do work = infrastructure

## AXIS 3 — MATERIAL TYPES
Tag ALL that apply (array, can be multiple):

- base_metals: Cu, Zn, Ni, Pb, Fe, Al
- precious_metals: Au, Ag, Pt, Pd
- lanthanides_REE: Nd, Dy, Tb, La, Ce, Y, Sc and all rare earth elements
- battery_metals: Li, Co, Ni, Mn — ONLY when the grant is about extracting, recovering, or processing these metals from ore or waste streams. Do NOT assign battery_metals (or keep the grant) if the focus is battery manufacturing, cathode/anode engineering, energy storage performance, or battery materials science — those are downstream of mining.
- critical_transition_metals: W, Mo, V, Cr, Ti
- platinum_group_metals: Pt, Pd, Rh, Ir, Ru, Os
- actinides: U, Th
- metalloids_specialty: Si, Ge, Ga, Te, In, As
- polymetallic_general: Multiple metals without dominant focus; "critical minerals"; "strategic metals". USE THIS when the grant discusses ores, minerals, or metals generally without naming specific metal types. Do NOT combine with a specific metal tag if you have already tagged a specific metal — only add polymetallic_general when there is genuine multi-metal focus beyond the specific tag.
- coal_ash_secondary: Recovery from coal FLY ASH or BOTTOM ASH — the solid combustion byproduct of coal-fired power plants. Do NOT assign for acid mine drainage (AMD), coal mine drainage, or AMD sludge — those are secondary_feedstocks. AMD ≠ coal ash. Only tag coal_ash_secondary when the abstract explicitly mentions fly ash, bottom ash, coal combustion byproducts, or coal ash piles.
- secondary_feedstocks: Recovery from e-waste, electronic waste, spent batteries, industrial byproducts, process streams, fertilizer waste (phosphogypsum), AMD sludge, recycling streams — any secondary or waste-derived source that is not primary ore or coal fly/bottom ash.

MATERIALS TAGGING RULES:
1. Tag SPECIFIC metals if named (base_metals, lanthanides_REE, battery_metals, etc.). Only ALSO add polymetallic_general if there is additional multi-metal focus beyond the specific tag. Do NOT automatically add polymetallic_general alongside every specific tag.
2. Tag actinides (U, Th, Ra) whenever uranium, thorium, or radium are explicitly named — these commonly appear in phosphogypsum and REE processing waste.
3. Do NOT tag battery_metals (Li, Co, Ni, Mn) based on the motivating application alone. Only tag battery_metals if the grant is specifically extracting/recovering those metals. If the grant studies a general separation mechanism that happens to use lithium as one example, use polymetallic_general instead.
4. Do NOT tag coal_ash_secondary for AMD, mine drainage, or sludge. Coal ash = combustion byproduct. AMD = drainage from mine workings. These are distinct.
5. Tag secondary_feedstocks when the SOURCE MATERIAL is waste-derived (AMD sludge, e-waste, phosphogypsum, spent batteries, process streams).

## AXIS 4 — RESEARCH STAGE
Assign one. Ask: what is the MATURITY of what this grant is doing?

- fundamental: Output is KNOWLEDGE, not a technology or process. Builds from first-principles science — the grant is generating the foundational understanding that future technology development will depend on. The grant asks a WHY or HOW question — mechanism studies, characterization, basic science — with nothing being built or tested. Includes: organism/material characterization, geologic surveys, mine site ecological baseline surveys, mine waste inventories, atlases, geochemical databases, reconnaissance mapping, molecular mechanism studies. Key test: is the only output knowledge or data, with nothing being built or tested? If yes → fundamental. If a tool or process is being built even as a means to study a mechanism, classify as early_technology_development instead — fundamental is reserved for purely observational or computational work. Key signals: "mechanisms of", "characterization of", "role of X in Y", "understand how", computation or MD simulation studying mechanisms.

- early_technology_development: Builds directly off fundamental science — takes established mechanistic understanding and asks whether it can be turned into a working process or device for the first time. A NEW technology, process, or system is being built and tested for the FIRST TIME at lab or bench scale. The core question is "does this work at all?" Output is a proof-of-concept or initial prototype demonstrating feasibility. The approach has NOT been previously demonstrated at any scale. Key signals: "proof of concept", "EAGER", "SBIR Phase I", first-ever lab demonstration, future-tense descriptions of an unbuilt system with no mention of prior demonstration, Phase I results, or predecessor work. DEFAULT for title-only grants with novel process names. DISTINCTION from fundamental: early_technology_development builds something and tests it; fundamental only produces knowledge. DISTINCTION from applied_translational: the technology is new and unproven — no prototype or working process yet exists.

- applied_translational: A KNOWN, PREVIOUSLY DEMONSTRATED technology or process is being optimized, scaled, or piloted. The core question is "how do we make this work better or bigger?" Output is performance improvement data, scale-up results, or a pilot/field demonstration. Key signals: "pilot-scale", "scale-up", "techno-economic analysis", "SBIR Phase II" (always applied_translational), "demonstration at X scale", explicit reference to prior Phase I or earlier work being advanced. DISTINCTION from early_technology_development: a prototype or working process already exists and has been demonstrated — the grant is refining or scaling it, not inventing it. When in doubt between ETD and applied_translational, ask whether a prototype or working process already exists — if yes → applied_translational.

- deployment: An APPROVED PROGRAM is being EXECUTED at full operational or commercial scale with no meaningful R&D component — minor monitoring or reporting does not qualify as R&D. No new science, no optimization, no pilot — established methods are being applied. Key signals: large federal program grants distributing funds to states/tribes for pre-approved plans, facilities, or remediation efforts; infrastructure law implementation ($1M+ operational programs); "executing" or "installing" established technology on a named site or for an approved program; SBIR Phase III; DOE loan programs. Key test: is anyone learning anything new? If no → deployment. DISTINCTION from applied_translational: deployment has zero learning objective; applied_translational still has a research or optimization objective.

- EAGER GRANTS: Any grant with title_prefix "EAGER" (EArly-concept Grants for Exploratory Research) always has a specific technology target. Never assign fundamental — assign early_technology_development minimum.

STAGE DECISION TREE — go in order, stop at first match:
1. Is the output purely knowledge/data with no technology being built or tested? → fundamental
2. Is an approved operational program being executed with no R&D? → deployment
3. Is a brand new technology being proposed/tested for the first time? → early_technology_development
4. Is a known technology being optimized, scaled, or piloted? → applied_translational

Do NOT assign fundamental to title-only grants describing WORKSHOP ATTENDANCE or PROGRAM PARTICIPATION — check step 2 first. If the title describes attending a workshop, participating in a federal program (Earth MRI, IIJA, SMCRA), or executing a pre-defined activity, assign deployment regardless of whether an abstract is present.

## AXIS 5 — ORIENTATION

Apply the decision tree below IN ORDER. Stop at the first match.

1. **Award Type contains "SBIR" or "STTR" OR Title contains "I-Corps"?** → `industry_facing`
2. **Abstract contains "non-profit", "not-for-profit", or "nonprofit"?** → `public_good`
3. **Recipient institution contains University, College, Institute, or School?** → `public_good`
4. **Recipient institution matches a state-university / engineering-school pattern** (e.g., "Georgia Tech", "Ohio State", "Colorado School of Mines", "Montana Tech", "South Dakota School of Mines and Technology")? → `public_good`
5. **Recipient is a national lab, federal agency, or tribal/Indigenous organization** (e.g., "National Laboratory", "USGS", "DOE", "Tribe", "Nation", "Pueblo", "Confederated")? → `public_good`
6. **Recipient institution contains Foundation, Action, Restoration, Community, or Corps** (NOT "I-Corps")? → `public_good`
7. **Recipient institution looks like a private company** (contains "LLC", "Inc", "Inc.", "Corp", "Corporation", "Ltd", "Co.", "Company")? → `industry_facing`
8. **Abstract contains commercialization / IP language** (patent, royalties, licensing, "bring to market", "commercialization pathway", "commercial viability")? → `industry_facing`
9. **Default** → `public_good`

### BIOMINING-SPECIFIC OVERRIDES (apply AFTER the tree above if triggered)

**BIL / large federal infrastructure override.** Large grants awarded to a PRIVATE COMPANY for construction of an industrial-scale production facility (plant, processing hub, commercial production capacity) = `industry_facing`, even when funded through a public law (BIL, DOE infrastructure programs). Signals: named private-company awardee + explicit production-capacity target + commercial-scale or industrial-scale language. If the tree returned `public_good` only because rule 3/6 hit a non-specific word, but the awardee is plainly a private company with an industrial-scale facility objective, reclassify as `industry_facing`.

**University-with-DOE-Cooperative-Agreement exception.** If the recipient is a university (rule 3) AND the award mechanism is a DOE Cooperative Agreement (not SBIR/STTR), keep `public_good` even if the abstract contains commercialization language. Rule 3 takes precedence over rule 8 here.

**Title-only grants.** If no abstract is present and only the title is visible, default to `public_good` unless the title itself contains SBIR/STTR/I-Corps markers OR a clearly private company awardee is named.

## AXIS 6 — RESEARCH APPROACH

**Gate:** research_approach applies ONLY to public_good grants. If orientation = industry_facing, set research_approach = null and skip the rest of this axis.

For public_good grants, assign one of two values based ONLY on the research-description text in the abstract (NOT funder program boilerplate, NOT multi-institution collaboration, NOT SBIR/grant-type language):

- **collaborative_interdisciplinary** — ONLY if the abstract EXPLICITLY describes integrating or combining multiple disciplines as a CORE feature of the research approach. Positive signals include:
  - "interdisciplinary approach"
  - "integration of [discipline] and [discipline]"
  - "combining [field] and [field]"
  - "integrating X and Y across disciplines"
  - "Collaborative Research:" ONLY when paired with explicit cross-discipline integration language in the abstract body
  Example hits: "integration of advanced modeling methods from engineering, environmental science, natural science, and data science" ✓; "interdisciplinary approach combining ecology and engineering" ✓.

- **single_focus** — Everything else. This includes:
  - Using multiple techniques/methods from one primary discipline (e.g., "using genomics, proteomics, and metabolomics" — multiple techniques, one field)
  - Mentioning multiple fields without stating integration (e.g., "biochemistry and bioengineering approaches" — related subfields, not integration)
  - CAREER grants — default to single_focus unless the abstract body explicitly states interdisciplinary integration
  - "Collaborative Research:" appearing in the TITLE alone (multi-institution ≠ interdisciplinary)
  - Funder-program boilerplate describing the GRANT PROGRAM as "multi-project, interdisciplinary" — this describes the program, not this grant's approach
  - Multiple disciplines named ONLY in the Broader Impacts / education / outreach / training section (e.g., "training students at the interface of biology, geology, physics, chemistry, and engineering") — this is educational scope, not research approach. Evaluate ONLY the research description (aims, methods, hypotheses), not the education/outreach plan.

**Key test:** Does the abstract explicitly use words like "interdisciplinary", "integration of disciplines", or "combining [field A] and [field B]" in describing THIS grant's research approach? If YES → collaborative_interdisciplinary. Otherwise → single_focus.

## AXIS 7 — INFRASTRUCTURE SUBTYPE

**Gate:** infrastructure_subtype applies ONLY when mining_type = infrastructure. For all other mining_type values, set infrastructure_subtype = null and skip this axis.

For mining_type = infrastructure grants, assign one of seven subtypes using the following 7-way taxonomy.

**Apply ALL of these rules:**
1. Classify by the grant's PRIMARY DELIVERABLE. If the infrastructure-flavored output is just a "broader impacts" side activity of a research grant, the parent grant_type would not be infrastructure. Assume the input you receive is genuinely infrastructure; choose the subtype that fits its primary deliverable.
2. Funding to BUILD a facility/resource and funding to OPERATE the same facility/resource go in the SAME subtype. Lifecycle stage of the funding does not change the classification.
3. If a grant funds multiple elements (e.g., a building plus instruments), classify by the dominant element — usually the headline deliverable or the larger dollar share.
4. **Multi-component programs that all serve a single coherent purpose go in that purpose's subtype, NOT `other`.** Example: an extension program with workshops + educational materials + demo sites + monitoring networks all serving one IPM extension goal → `community_building`. The components individually look like multiple subtypes, but together they constitute one program with a clear primary character (extension/training). Use `other` only when the components serve genuinely independent purposes with no overarching program — see the `other` definition for details.
5. **Centers and hubs with both a physical home AND multi-institution coordination:** classify by the dominant tangible deliverable. If a specific physical building/lab/equipment at a named institution is central to the grant's delivery (e.g., "User Facility at CCRC at UGA", "regional analytical chemistry laboratory at Michigan State"), the physical home wins → `physical_facilities`. If the coordination/network is the primary deliverable and the physical home is incidental (e.g., I/UCRC across 4 universities, RCN, NSF Engines spanning industry+academia+nonprofits), → `community_building`. If a center has multiple genuinely co-equal deliverables (physical equipment + user facility + research outputs + education + community-wide standards) with no single dominant element, → `other`.
6. The "Includes" lists are illustrative, not exhaustive. Apply the key test to grants that don't exactly match an example.

**physical_facilities** — Buildings, spaces, or structures that researchers occupy or operate from. The deliverable is the space itself, not specific instruments inside it.
- Includes: lab buildings; research centers/hubs with a physical home; greenhouses; walk-in plant growth rooms; plant phenotyping facilities; field stations; pilot fermentation/biomanufacturing facilities (the building, not the fermenters); photobioreactor facilities (the room, not the individual reactors).
- Key test: Is the deliverable a SPACE researchers OCCUPY or OPERATE FROM?
- Not this: a discrete instrument or piece of equipment (use `instrumentation`); a curated material collection (use `physical_repositories`); a virtual coordinating entity with no physical building (use `community_building`).

**instrumentation** — Discrete equipment or hardware that researchers operate as a tool — inputs go in, data or outputs come out. Single-purpose, even if large.
- Includes: mass spectrometers; sequencers; gas chromatographs; individual bioreactors/fermenters; individual photobioreactors; bench-top growth chambers; automated culture/screening systems; instrument suites (NSF MRI awards).
- Key test: Is the deliverable a SPECIFIC TOOL or UNIT researchers OPERATE, performing a defined function?
- Not this: a building or occupied space (use `physical_facilities`); software platforms or computational models (use `data_repositories`).

**Disambiguating physical_facilities vs instrumentation** — apply the deliverable test: space researchers occupy → physical_facilities; tool/unit researchers operate → instrumentation. Size isn't the test — occupy vs. operate is.
- "Construction of a pilot fermentation facility" → physical_facilities (building)
- "Acquisition of a 100L pilot fermenter" → instrumentation (the fermenter)
- "Greenhouse complex for plant breeding" → physical_facilities (occupied space)
- "Bench-top plant growth chamber" → instrumentation (discrete unit)
- "Walk-in environmental growth room" → physical_facilities (researchers occupy it)

**data_repositories** — Digital collections built or maintained for community use.
- Includes: microbial/plant/soil microbiome genome databases; biocatalyst and enzyme databases; metabolic pathway databases; synthetic biology design tools; bioprocess simulation models; LCA tools for bioproducts; climate-biotech reference datasets; software platforms; code libraries; computational models.
- Key test: Is the deliverable a DIGITAL RESOURCE that other researchers will access or use?
- Not this: physical materials (use `physical_repositories`); structured pedagogy (use `learning_materials`); a discrete instrument (use `instrumentation`).

**physical_repositories** — Curated physical collections of biological or reference materials, built or maintained for community use.
- Includes: microbial culture collections (e.g., ATCC, NRRL); algae and cyanobacteria culture collections; seed banks (e.g., USDA NPGS); plant tissue archives; soil/sediment sample archives; type-strain collections; certified physical reference material libraries for environmental analytes.
- Key test: Is the deliverable a CURATED COLLECTION of physical samples that other researchers will request from?
- Not this: digital data (use `data_repositories`); a building where research happens (use `physical_facilities`).

**learning_materials** — Produced pedagogical content.
- Includes: synthetic biology curricula; bioenergy/bioproducts training modules; bioremediation lab manuals; climate-biotech educational websites; MOOCs on biomanufacturing or microbial engineering; instructional guides; open educational resources (OER) for climate biotech.
- Key test: Is the deliverable PEDAGOGICAL CONTENT that learners will read, watch, or work through?
- Not this: workshops/training events (use `community_building`); reference data (use `data_repositories`).

**community_building** — Multi-institution coordination, training programs, and convening activities. The deliverable is human/institutional capacity, not an artifact.
- Includes: bioenergy/synthetic biology workshops and conferences; REUs (Research Experiences for Undergraduates) in climate biotech labs; training programs in metabolic engineering or bioremediation; multi-institution networks (e.g., bioenergy research networks, climate biotech consortia); RCN-type awards; virtual research centers and hubs (no physical home, network of researchers and institutions).
- Key test: Is the deliverable HUMAN or INSTITUTIONAL CAPACITY built through events, training, or coordinating networks?
- Not this: produced content artifacts (use `learning_materials`); a physical building or facility (use `physical_facilities`).

**other** — Infrastructure grant whose deliverable genuinely cannot be classified under any of the six specific subtypes above.
- Use when: (a) the grant looks borderline-deployment (community-scale implementation of established practice rather than building a tool/facility for others); (b) the deliverable genuinely spans multiple subtypes with NO single dominant element AND the components do NOT all serve one coherent program (a center that is simultaneously a research producer, equipment provider, training program, user facility, and standards body with all elements equally weighted); or (c) it's an unusual infrastructure case that doesn't match any of the 6 patterns.
- Key test: After applying each of the 6 specific subtype tests honestly, AND after asking "do the multiple components all serve one coherent overarching program?", is the deliverable still a poor fit for all 6?
- IMPORTANT — do NOT use `other` when:
  - The grant has multiple components but they all serve one coherent program (e.g., extension program → `community_building`; research center with a clear physical anchor → `physical_facilities`).
  - One of the 6 subtypes fits even imperfectly. Pick the closest fit before resorting to `other`.
  - The grant has just two or three component types — `other` is for genuinely irreducible multi-deliverable cases.
- Note: choosing `other` is a useful signal — these grants warrant a closer look (often indicates an upstream `grant_type` edge case or a true integrated multi-deliverable program). It should be RARE — most infrastructure grants will fit one of the 6 specific subtypes after careful reading.

## OUTPUT FORMAT

Return ONLY a valid JSON array — no preamble, no explanation, no markdown fences.
One object per grant, in the same order as input.

CRITICAL COMPLETENESS RULE: For every grant, mining_type, stage, and orientation must be non-null. research_approach must be non-null when orientation=public_good and must be null when orientation=industry_facing. infrastructure_subtype must be non-null when mining_type=infrastructure and must be null otherwise. A classification with null mining_type or null stage is an error. If you are uncertain about mining_type but the grant is clearly mining-relevant, make your best call and use confidence:medium rather than leaving the field null.

[
  {
    "grant_id": "<unique_key from input>",
    "confidence": "high" | "medium" | "low",
    "biological": true or false,
    "bio_subcategory": null or one of [bioremediation, bioleaching, bioseparation_biosorption, bioprecipitation_mineral_formation, biosensing],
    "mining_type": one of [exploration, extraction, downstream_processing, remediation, infrastructure, collaborative_tools, general],
    "materials": [array of material tags],
    "stage": one of [fundamental, early_technology_development, applied_translational, deployment],
    "orientation": "public_good" or "industry_facing",
    "research_approach": "collaborative_interdisciplinary" | "single_focus" | null  (must be null when orientation=industry_facing),
    "infrastructure_subtype": "physical_facilities" | "instrumentation" | "data_repositories" | "physical_repositories" | "learning_materials" | "community_building" | "other" | null  (must be null when mining_type != infrastructure)
  }
]

## CALIBRATION EXAMPLES

Grant: "An Economic, Sustainable, Green, Gold Isolation Process" | query: mining | title_prefix: SBIR Phase II | abstract: improving gold mining process economics; replacing cyanide in gold isolation from tailings
→ biological:false, mining_type:extraction, materials:[precious_metals], stage:applied_translational, orientation:industry_facing
Note: mining_type must always be assigned. Gold isolation from tailings = extraction.

Grant: "DEVELOP AN UNDERSTANDING OF ACIDIC AND TOXIC MINE DRAINAGE ..." | query: acid mine drainage | abstract: SMCRA-authorized mine drainage technology initiative; coordinating AMD research and technology transfer across states
→ biological:false, mining_type:remediation, materials:[polymetallic_general], stage:applied_translational, orientation:public_good

Grant: "Acidithiobacillus thiooxidans for sulfide mineral leaching in heap configurations"
→ biological:true, bio_subcategory:bioleaching, mining_type:extraction, materials:[base_metals], stage:early_technology_development, orientation:public_good

Grant: "SBIR Phase II: Selective solvent extraction process for lithium recovery from geothermal brines" | title_prefix: SBIR Phase II
→ biological:false, mining_type:extraction, materials:[battery_metals], stage:applied_translational, orientation:industry_facing

Grant: "Froth flotation optimization for rare earth mineral separation from coal fly ash" | hit_cat_a: True
→ biological:false, mining_type:downstream_processing, materials:[lanthanides_REE, coal_ash_secondary], stage:applied_translational, orientation:public_good

Grant: "Sulfate-reducing bacterial consortia for selective precipitation of cobalt and nickel from acid mine drainage" | query: biohydrometallurgy
→ biological:true, bio_subcategory:bioprecipitation_mineral_formation, mining_type:extraction, materials:[battery_metals, base_metals], stage:fundamental, orientation:public_good
Note: bioprecipitation_mineral_formation = controlled microbial precipitation of specific metals as recoverable mineral products. mining_type=extraction not remediation — the goal is selectively recovering cobalt and nickel as products. The AMD source material does not determine mining_type — the stated goal does.

Grant: "OXBOW TAILING RESTORATION PROJECT PHASE 3A" | abstract: [title only]
→ biological:false, mining_type:remediation, materials:[polymetallic_general], stage:applied_translational, orientation:public_good

Grant: "SBIR Phase I: Machine learning platform for automated mineral identification from hyperspectral drill core imagery" | title_prefix: SBIR Phase I
→ biological:false, mining_type:infrastructure, materials:[polymetallic_general], stage:early_technology_development, orientation:industry_facing
Note: Software tool built specifically for mining exploration = infrastructure.

Grant: "GCR: Convergence on Phosphorus Sensing for Understanding Global Biogeochemistry and Enabling Pollution Management (mining/hydrometallurgy context)"
→ biological:true, bio_subcategory:biosensing, mining_type:infrastructure, materials:[polymetallic_general], stage:fundamental, orientation:public_good
Note: Phosphorus sensing tool = infrastructure (building a tool). Multi-use but mining explicitly named.

Grant: "Development of a yeast-based continuous culture system for detecting bioavailable phosphate in water" | abstract: yeast biosensor designed as transferable process monitoring tool applicable to mining streams and hydrometallurgical operations
→ biological:true, bio_subcategory:biosensing, mining_type:infrastructure, materials:[polymetallic_general], stage:early_technology_development, orientation:public_good
Note: KEEP because the abstract explicitly connects phosphate sensing to mining process streams.

Grant: "Synthesis, Characterization, and Reactivity Studies of Cerium-Ligand Multiple Bonds and Organometallics" | abstract mentions: cerium, thorium, uranium, metal-ligand multiple bonds, ion selectivity; synthetic organometallic and inorganic ligand chemistry
→ biological:false, bio_subcategory:null, mining_type:extraction, materials:[lanthanides_REE, actinides], stage:fundamental, orientation:public_good
Note: Synthetic organometallic and inorganic ligand chemistry = biological:false. Do NOT assign bioseparation_biosorption to synthetic ligand chemistry. The bio fields are reserved for actual biological organisms, proteins, peptides, or bio-derived molecules.

Grant: "Spectroelectrochemical Measurements on Intact Microorganisms Under Oxic and Anoxic Conditions" | abstract: measuring electron exchange between microorganisms and solid minerals within an ore body
→ biological:true, bio_subcategory:bioleaching, mining_type:extraction, materials:[], stage:fundamental, orientation:public_good
Note: Microorganisms exchanging electrons with solid minerals = bioleaching. Do NOT default to bioremediation for any microbe-mineral interaction.

Grant: "Uncovering rare earth elements biochemistry: From enzymes to ecosystems" | abstract: characterizing enzymes and biological pathways that bind and process rare earth elements in environmental bacteria
→ biological:true, bio_subcategory:bioseparation_biosorption, mining_type:extraction, materials:[lanthanides_REE], stage:fundamental, orientation:public_good

Grant: "Molecular Specificity of Transition Metal and Lanthanide-Cyclic Depsipeptide Complexes"
→ biological:true, bio_subcategory:bioseparation_biosorption, mining_type:extraction, materials:[lanthanides_REE, critical_transition_metals], stage:fundamental, orientation:public_good
Note: Cyclic depsipeptides are bio-derived molecules — biological:true.

Grant: "RUI: CAS: Development of Tripodal Ligands for Next-Generation Rare Earth Element Separations" | abstract: developing SYNTHETIC metal chelating agents (tripodal ligands with carbamoylmethylphosphine oxide groups)
→ biological:false, bio_subcategory:null, mining_type:extraction, materials:[lanthanides_REE, actinides], stage:fundamental, orientation:public_good

Grant: "MRI: Acquisition of a Liquid-Chromatograph Mass Spectrometer (LC-MS)" | abstract mentions: development of ligands for the separation of critical elements such as rare earth (f) elements
→ biological:false, mining_type:infrastructure, materials:[lanthanides_REE], stage:applied_translational, orientation:public_good

Grant: "OUR INTEGRATED PROJECT COMBINES TEACHING AND RESEARCH ... AMD ... PASSIVE TREATMENT FOR ACID MINE DRAINAGE COULD BE USED AS A POTENTIAL RECOVERY SYSTEM FOR REE"
→ biological:false, mining_type:remediation, materials:[lanthanides_REE, polymetallic_general], stage:fundamental, orientation:public_good

Grant: "GEO OSE Track 2: Sustainable Open Science Tools to Democratize Use of 3D Geomaterial Data" | abstract: rare-earth mineral recovery listed alongside groundwater, carbon sequestration, contaminant transport
→ biological:false, mining_type:infrastructure, materials:[lanthanides_REE, polymetallic_general], stage:early_technology_development, orientation:public_good

Grant: "Confederated Tribes of the Colville Reservation: Mining-Associated Waste Streams and Burbot Health" | abstract: "assess the influence of mining-associated waste streams on contaminant exposure patterns"
→ biological:false, mining_type:remediation, materials:[base_metals, metalloids_specialty], stage:applied_translational, orientation:public_good

Grant: "Next generation enzyme engineering: high-throughput directed evolution of spore-displayed enzymes" | title_prefix: SBIR Phase I | abstract: enzyme platform listing biomining alongside pharmaceuticals, biofuels, food production, carbon capture
→ biological:true, bio_subcategory:bioseparation_biosorption, mining_type:extraction, materials:[], stage:early_technology_development, orientation:industry_facing
Note: SBIR Phase I — biomining is explicitly named as a target application. Classify on the mining application.

Grant: "MRI: Acquisition of high-resolution mass spectrometry system for metal-organic environmental and biological chemistry" | abstract: instrument acquisition to build database linking biological systems to metals they bind
→ biological:true, bio_subcategory:bioseparation_biosorption, mining_type:infrastructure, materials:[], stage:fundamental, orientation:public_good

Grant: "RAISE: CET: REE-Selective Protein Hydrogels (REESPECT) for Lanthanide Recovery from Recycling Feedstocks"
→ biological:true, bio_subcategory:bioseparation_biosorption, mining_type:extraction, materials:[lanthanides_REE, secondary_feedstocks], stage:early_technology_development, orientation:public_good

Grant: "GEO-CM: Evaluating hydrogeochemical controls on REE and Yttrium mobility during rock weathering" | abstract: explores how waste products of coal production could be an important source of REEs
→ biological:false, mining_type:exploration, materials:[lanthanides_REE, secondary_feedstocks], stage:fundamental, orientation:public_good
Note: MUST tag secondary_feedstocks alongside lanthanides_REE — coal waste/byproducts are the source material.

Grant: "DESIGN AND APPLICATION OF A GENETIC TOOLBOX TO DOMESTICATE ODONTARRHENA CORSICA AND ODONTARRHENA CHALCIDICA FOR NICKEL PHYTOMINING" | abstract: developing genetic tools to domesticate nickel hyperaccumulating plants for phytomining
→ biological:true, bio_subcategory:bioseparation_biosorption, mining_type:extraction, materials:[battery_metals], stage:early_technology_development, orientation:public_good
Note: Phytomining = plants as extraction mechanism = extraction, bio_subcategory=bioseparation_biosorption. NOT bioremediation.

Grant: "NSF-BSF: Elucidating the role of ion dehydration in regulating transport and selectivity of monovalent ions in polyamide nanofiltration membranes" | abstract: novel ion dehydration mechanism; lithium extraction from brines as motivating application; technology applies broadly to monovalent ion separation
→ biological:false, mining_type:extraction, materials:[polymetallic_general], stage:fundamental, orientation:public_good
Note: polymetallic_general — lithium is the motivating example but the grant develops a general ion separation mechanism.

Grant: "BIOINSPIRED GREEN GLYCOLIPIDS AS FUGITIVE DUST MITIGATION AGENTS" | query: mining | abstract: biologically-derived glycolipid surfactants for dust suppression on mine tailings and unpaved haul roads
→ biological:true, bio_subcategory:bioremediation, mining_type:remediation, materials:[polymetallic_general], stage:early_technology_development, orientation:industry_facing
Note: Bio-derived agents for mine site dust suppression = bioremediation + remediation.

Grant: "THE TOWN OF SUPERIOR ... UPPER QUEEN CREEK WATERSHED GROUP" | abstract: stakeholder engagement to create watershed group addressing water quality issues from copper mining impacts
→ biological:false, mining_type:collaborative_tools, materials:[base_metals], stage:applied_translational, orientation:public_good
Note: Community stakeholder engagement = collaborative_tools. NOT remediation — no active cleanup being built.

Grant: "Heavy Mineral Mining Impeller Accelerated Separator" | title_prefix: SBIR Phase I
→ biological:false, mining_type:extraction, materials:[polymetallic_general], stage:early_technology_development, orientation:industry_facing

Grant: "Conference: Resilient Supply of Critical Minerals, Rolla, MO"
→ biological:false, mining_type:collaborative_tools, materials:[polymetallic_general], stage:fundamental, orientation:public_good

Grant: "Global Centers Track 2: Green Energy Transitions in the Far North (GET North)" | abstract: international center studying green energy transitions including raw mineral mining, labor, infrastructure, Indigenous populations across Arctic regions
→ biological:false, mining_type:collaborative_tools, materials:[polymetallic_general], stage:fundamental, orientation:public_good

Grant: "EAGER: CET: The Dissolution of Li-ion Batteries and Recycling of their Precious Components"
→ biological:false, mining_type:extraction, materials:[battery_metals, secondary_feedstocks], stage:early_technology_development, orientation:public_good
Note: Recovering metals FROM spent batteries = extraction from secondary feedstocks.

Grant: "Sustainable Rare Earth Element Production from Coal Combustion Byproducts" | title_prefix: SBIR Phase I
→ biological:false, mining_type:extraction, materials:[lanthanides_REE, coal_ash_secondary], stage:early_technology_development, orientation:industry_facing

Grant: "COMMUNITY PROJECT FUNDING ... REESTABLISH THE VICTIMS OF MILLS TAILINGS EXPOSURE CANCER SCREENING PROGRAM" | funder: HHS
→ biological:false, mining_type:remediation, materials:[actinides], stage:applied_translational, orientation:public_good

Grant: "TO SUPPORT THE IMPLEMENTATION OF THE MINAMATA CONVENTION ON MERCURY BY REDUCING THE USE OF MERCURY IN ARTISANAL AND SMALL SCALE GOLD MINING" | funder: Department of State
→ biological:false, mining_type:extraction, materials:[precious_metals], stage:applied_translational, orientation:public_good

Grant: "SYNTHETIC GYPSUM AND COAL ASH CONTAMINANT REMOVAL AND BENEFICIATION" | funder: USDA | abstract: title only
→ biological:false, mining_type:remediation, materials:[coal_ash_secondary, polymetallic_general], stage:early_technology_development, orientation:public_good
Note: Title describes a new technology being proposed, not established process being optimized.

Grant: "WILDFIRES AND FLASH FLOODS: USING EXPOSURE SCIENCE TO IDENTIFY RURAL ARIZONA MINING COMMUNITIES AT RISK FROM THE RELEASE AND REMOBILIZATION OF CONTAMINANTS"
→ biological:false, mining_type:remediation, materials:[base_metals, polymetallic_general], stage:applied_translational, orientation:public_good

Grant: "BIOINSPIRED GREEN GLYCOLIPIDS AS FUGITIVE DUST MITIGATION AGENTS" [Phase I = early_technology_development / Phase II = applied_translational]
→ biological:true, bio_subcategory:bioremediation, mining_type:remediation, materials:[polymetallic_general], orientation:industry_facing

Grant: "GEOTHERMAL THERMOELECTRIC GENERATION (G-TEG) WITH INTEGRATED TEMPERATURE DRIVEN MEMBRANE DISTILLATION AND NOVEL MANGANESE OXIDE LITHIUM EXTRACTION" | funder: DOE | abstract: title only
→ biological:false, mining_type:extraction, materials:[battery_metals], stage:early_technology_development, orientation:public_good

Grant: "BIOLOGICAL EXTRACTION OF RARE EARTH ELEMENTS INSPIRED BY PROCESSES IN THE EARTH (BE RIPE)" | funder: DOD | abstract: title only
→ biological:true, bio_subcategory:bioseparation_biosorption, mining_type:extraction, materials:[lanthanides_REE], stage:fundamental, orientation:public_good

Grant: "BIPARTISAN INFRASTRUCTURE LAW ABANDONED MINED LANDS (BIL AML) PROGRAM" | funder: DOI | abstract: $11.3B federal program distributing grants to states/tribes to execute pre-approved reclamation plans
→ biological:false, mining_type:remediation, materials:[polymetallic_general], stage:deployment, orientation:public_good
Note: DEPLOYMENT. Large federal infrastructure law grants executing established programs = deployment.

Grant: "EARTH MRI MINE WASTE PROPOSAL: CREATE VIRGINIAS MINE WASTE INVENTORY" | abstract: compile geospatial database of 25+ mine waste sites; deliverable to USMIN national database
→ biological:false, mining_type:collaborative_tools, materials:[polymetallic_general, secondary_feedstocks], stage:fundamental, orientation:public_good

Grant: "A Microbe-Mineral Atlas for a Sustainable Energy Future" | funder: NSF Global Centers | abstract: international multi-stakeholder partnership building a shared atlas of microbe-mineral interactions relevant to biological metal extraction
→ biological:true, bio_subcategory:bioleaching, mining_type:collaborative_tools, materials:[polymetallic_general], stage:fundamental, orientation:public_good

Grant: "LOW-COST RARE-EARTH-ELEMENT (REE) RECOVERY FROM ACID MINE DRAINAGE SLUDGE" | funder: DOE | abstract: title only
→ biological:false, mining_type:extraction, materials:[lanthanides_REE, secondary_feedstocks], stage:early_technology_development, orientation:public_good
Note: EXTRACTION not remediation. AMD sludge is the SOURCE MATERIAL being processed to recover REEs.

Grant: "WSGS Mine Waste FY2025 Earth MRI Workshop Attendance" | funder: DOI | abstract: title only — attending Earth MRI workshop
→ biological:false, mining_type:collaborative_tools, materials:[polymetallic_general], stage:deployment, orientation:public_good
Note: Workshop attendance as part of established federal program = deployment.

Grant: "BIL-Integrated Sustainable Battery Precursor Production Plant" | funder: DOE | awardee: private company | abstract: establishing industrial-scale battery precursor production capacity
→ biological:false, mining_type:infrastructure, materials:[battery_metals, secondary_feedstocks], stage:deployment, orientation:industry_facing
Note: Primary deliverable is a physical production plant = infrastructure. Private company awardee + industrial-scale = industry_facing.

Grant: "REV Nickel: Partnership Between Critical Mineral Processing Company and US Nickel Mine" | funder: DOE | awardee: private company
→ biological:false, mining_type:extraction, materials:[base_metals, polymetallic_general], stage:deployment, orientation:industry_facing

Grant: "IRES Track I: Rapid, Integrated Geotechnical and Geochemical Characterization of Mine Waste for Phytoremediation and Biofuel/Bioenergy Production"
→ biological:true, bio_subcategory:bioremediation, mining_type:remediation, materials:[polymetallic_general], stage:fundamental, orientation:public_good
Note: BIOREMEDIATION not bioseparation_biosorption — primary goal is ecological restoration of contaminated mine waste sites.

Grant: "Glycolipids as Inexpensive Solid Supported Ligands for Uranium Remediation" | title_prefix: STTR Phase I
→ biological:true, bio_subcategory:bioremediation, mining_type:remediation, materials:[actinides], stage:early_technology_development, orientation:industry_facing
Note: BIOREMEDIATION not bioseparation_biosorption — goal is CLEANUP of uranium from legacy mine water, not uranium recovery as a product.

Grant: "Role of protein nanowires in metal cycling and mineralization" | abstract: Geobacter pili reductively precipitating Co/Cd/Ag as minerals
→ biological:true, bio_subcategory:bioprecipitation_mineral_formation, mining_type:extraction, materials:[base_metals], stage:fundamental, orientation:public_good
Note: BIOPRECIPITATION not bioleaching — metals move FROM solution INTO solid mineral product. Do NOT assign bioleaching to Geobacter/microbe-mineral interactions when microbes are PRECIPITATING, not DISSOLVING.

Grant: "Heap leaching and slope saturation monitoring with muon detectors" | title_prefix: SBIR Phase II
→ biological:false, mining_type:infrastructure, materials:[base_metals], stage:applied_translational, orientation:industry_facing
Note: materials=[base_metals] ONLY. Do NOT add polymetallic_general alongside a specific metal tag.

Grant: "Separation of Clean Gypsum from Phosphate Ore Processing Waste" | title_prefix: SBIR Phase I | abstract: separating gypsum; isolating uranium, thorium, and radium; recovering lanthanide REEs
→ biological:false, mining_type:extraction, materials:[lanthanides_REE, actinides, secondary_feedstocks], stage:early_technology_development, orientation:industry_facing
Note: materials MUST include actinides when U/Th/Ra are named.

Grant: "Sustainable System for Mineral Beneficiation Using Engineered Microspheres" | abstract: novel engineered microspheres for froth flotation separation of minerals; no specific target metal
→ biological:false, mining_type:downstream_processing, materials:[polymetallic_general], stage:early_technology_development, orientation:public_good
Note: Froth flotation = downstream_processing. No specific metal → polymetallic_general ONLY.

Grant: "GEO-CM: Evaluating hydrogeochemical controls on Rare Earth Element and Yttrium mobility during rock weathering from the micro- to landscape-scale" | abstract: explores how waste products of coal production could be an important source of REEs; hydrogeochemical controls on REE mobility in coal waste
→ biological:false, mining_type:exploration, materials:[lanthanides_REE, secondary_feedstocks], stage:fundamental, orientation:public_good
Note: MUST tag secondary_feedstocks alongside lanthanides_REE — coal waste/byproducts are the source material. Whenever coal waste, coal fly ash, or industrial waste streams are the feedstock being explored for REE recovery, always co-tag secondary_feedstocks. Do NOT add coal_ash_secondary unless fly ash or bottom ash is explicitly named — AMD sludge and coal mine waste are secondary_feedstocks.

Grant: "SBIR Phase II: A biological solution to improving the United States rare earth supply chain" | abstract: three-step biological REE recovery pipeline — step 1: bacteria leach REE from solid waste into solution; step 2: total REE recovery; step 3: selective bacterial attachment for purification
→ biological:true, bio_subcategory:bioleaching, mining_type:extraction, materials:[lanthanides_REE, secondary_feedstocks], stage:applied_translational, orientation:industry_facing
Note: bio_subcategory=bioleaching — step 1 is bacteria acting on SOLID waste. MULTI-STEP RULE: classify by primary extraction step.

## RESEARCH_APPROACH CALIBRATION

Grant: orientation=industry_facing (any SBIR commercial dust-suppressant, any industry-facing recovery pilot)
→ research_approach: null
Note: research_approach is gated on orientation=public_good. For industry_facing grants, always return null regardless of abstract language.

Grant: orientation=public_good, abstract says "integrating observations and modeling across the earth system"
→ research_approach: collaborative_interdisciplinary

Grant: orientation=public_good, abstract says "interdisciplinary approach combining biogeochemistry and materials science"
→ research_approach: collaborative_interdisciplinary

Grant: orientation=public_good, abstract says "use XRD and SEM to characterize mineral phases"
→ research_approach: single_focus
Note: Using multiple techniques does NOT equal interdisciplinary. Only explicit cross-discipline integration language triggers collaborative_interdisciplinary.

Grant: orientation=public_good, title="Collaborative Research: Role of Fe-oxide surfaces in uranium retention" | abstract describes geochemistry experiments only, no cross-discipline integration language
→ research_approach: single_focus
Note: "Collaborative Research:" in the title = multi-institution, NOT multi-discipline. Requires explicit integration language in the abstract body.

Grant: orientation=public_good, title="CAREER: Microbial recovery of lithium from spent batteries" | abstract describes bioleaching experiments and kinetic modeling
→ research_approach: single_focus
Note: CAREER grants default to single_focus unless the abstract explicitly states cross-discipline integration.

Grant: orientation=public_good, por_text contains funder boilerplate "MULTI-PROJECT, INTERDISCIPLINARY GRANTS" describing the FUNDING PROGRAM, abstract body describes focused glycolipid chemistry research
→ research_approach: single_focus
Note: Ignore funder-program boilerplate. Evaluate only what THIS grant's research approach describes. Program-level "interdisciplinary" language is not a signal.

Grant: orientation=public_good, abstract says "this EAGER project uses microorganisms for battery metal recovery; hallmark Clean Energy Technology initiative; transformative biohydrometallurgy"
→ research_approach: single_focus
Note: Multi-domain vocabulary (CET, biohydrometallurgy, clean energy) is not interdisciplinary integration. No explicit "integration of X and Y" or "interdisciplinary approach" language in the abstract body → single_focus.

Grant: orientation=public_good, abstract body describes focused Geobacter microbiology experiments, then concludes: "The research is also intertwined with educational efforts directed at training young professionals at the interface of biology and geology, but with deep understanding of physics, chemistry, and engineering"
→ research_approach: single_focus
Note: The multi-discipline language ("biology and geology...physics, chemistry, engineering") appears ONLY in the Broader Impacts / education section, describing training scope — not the research approach. The research itself is focused microbiology. Education/outreach mentions never upgrade a grant to collaborative_interdisciplinary.

IMPORTANT:
- Your response must be ONLY the JSON array
- Return one object per grant in the same order as input
- CRITICAL: every kept grant must have non-null mining_type, stage, orientation, research_approach
- materials is an array — may be empty but must be present
- bio_subcategory must be null when biological=false
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _safe_str(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _extract_json_array(text: str):
    """Extract JSON array from LLM response, handling markdown fences."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        raise ValueError(f"Could not parse JSON: {e}\nText: {text[:500]}")


def _build_grant_text(row) -> str:
    """Render a grant for LLM input. Uses step1/step2 schema."""
    key       = _safe_str(row.get("unique_key", ""))
    title     = _safe_str(row.get("title", ""))
    abstract  = _safe_str(row.get("abstract", ""))
    funder    = _safe_str(row.get("funder", ""))
    award_type = _safe_str(row.get("award_type", ""))
    recipient = _safe_str(row.get("institution") or row.get("recipient_name") or row.get("pi_name") or "")

    return (
        f"Grant ID: {key}\n"
        f"Title: {title}\n"
        f"Funder: {funder}\n"
        f"Award Type: {award_type}\n"
        f"Recipient: {recipient}\n"
        f"Abstract: {abstract}"
    )


# =============================================================================
# STAGE 1 — KEEP/REMOVE (Haiku)
# =============================================================================
def classify_stage1_batch(client, batch):
    grant_texts = [_build_grant_text(row) for row in batch]
    user_msg = "Classify these grants for mining relevance (KEEP vs REMOVE):\n\n" + "\n\n---\n\n".join(grant_texts)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=STAGE1_MODEL,
                max_tokens=STAGE1_MAX_TOKENS,
                system=STAGE1_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text
            results = _extract_json_array(raw_text)

            expected_ids = {_safe_str(r.get("unique_key")) for r in batch}
            returned_ids = {_safe_str(r.get("grant_id")) for r in results}
            if expected_ids != returned_ids:
                print(f"  ⚠️  Stage 1 ID mismatch (expected {len(expected_ids)}, got {len(returned_ids)})")
            return results, raw_text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  ⚠️  Stage 1 attempt {attempt+1} failed: {e}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Stage 1 failed after {MAX_RETRIES} attempts: {e}")
                return [{"grant_id": _safe_str(r.get("unique_key")),
                         "keep": False, "confidence": "low",
                         "remove_reason": "Classification failed"} for r in batch], str(e)


def apply_stage1_results(df, results_dict):
    for idx, row in df.iterrows():
        gid = str(row.get("unique_key", "")).strip()
        if gid in results_dict:
            r = results_dict[gid]
            df.at[idx, "s1_keep"]          = bool(r.get("keep", False))
            df.at[idx, "s1_confidence"]    = r.get("confidence", "low")
            df.at[idx, "s1_remove_reason"] = r.get("remove_reason", "")


# =============================================================================
# STAGE 2-FORMULA — biological + bio_subcategory (Haiku)
# =============================================================================
def classify_stage2_formula_batch(client, batch):
    grant_texts = [_build_grant_text(row) for row in batch]
    user_msg = "Classify these FORMULA grants (biological? bio_subcategory?):\n\n" + "\n\n---\n\n".join(grant_texts)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=STAGE2_FORMULA_MODEL,
                max_tokens=STAGE2_FORMULA_MAX_TOKENS,
                system=STAGE2_FORMULA_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text
            results = _extract_json_array(raw_text)
            return results, raw_text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  ⚠️  Stage 2-Formula attempt {attempt+1} failed: {e}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Stage 2-Formula failed after {MAX_RETRIES} attempts: {e}")
                return [{"grant_id": _safe_str(r.get("unique_key")),
                         "biological": False, "bio_subcategory": None,
                         "confidence": "low"} for r in batch], str(e)


def apply_stage2_formula_results(df, results_dict):
    for idx, row in df.iterrows():
        gid = str(row.get("unique_key", "")).strip()
        if gid in results_dict:
            r = results_dict[gid]
            df.at[idx, "llm_biological"]      = bool(r.get("biological", False))
            df.at[idx, "llm_bio_subcategory"] = r.get("bio_subcategory")
            df.at[idx, "llm_confidence"]      = r.get("confidence", "low")
            # Static axes for formula grants (block allocations — deployment, not research)
            df.at[idx, "llm_mining_type"]            = "remediation"
            df.at[idx, "llm_stage"]                  = "deployment"
            df.at[idx, "llm_materials"]              = json.dumps(["polymetallic_general"])
            df.at[idx, "llm_orientation"]            = "public_good"
            df.at[idx, "llm_research_approach"]      = None
            df.at[idx, "llm_infrastructure_subtype"] = None


# =============================================================================
# STAGE 2-SHORT — biological + bio_subcategory for non-formula short-abstract grants
# =============================================================================
def classify_stage2_short_batch(client, batch):
    grant_texts = [_build_grant_text(row) for row in batch]
    user_msg = ("Classify these grants (biological? bio_subcategory?). "
                "Abstracts are very short — rely primarily on titles:\n\n"
                + "\n\n---\n\n".join(grant_texts))
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=STAGE2_SHORT_MODEL,
                max_tokens=STAGE2_SHORT_MAX_TOKENS,
                system=STAGE2_SHORT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text
            results = _extract_json_array(raw_text)
            return results, raw_text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  ⚠️  Stage 2-Short attempt {attempt+1} failed: {e}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Stage 2-Short failed after {MAX_RETRIES} attempts: {e}")
                return [{"grant_id": _safe_str(r.get("unique_key")),
                         "biological": False, "bio_subcategory": None,
                         "confidence": "low"} for r in batch], str(e)


def apply_stage2_short_results(df, results_dict):
    """Apply Stage 2-Short results. Sets llm_biological + llm_bio_subcategory only.
    Other axes intentionally left None — short abstracts can't support full classification."""
    for idx, row in df.iterrows():
        gid = str(row.get("unique_key", "")).strip()
        if gid in results_dict:
            r = results_dict[gid]
            df.at[idx, "llm_biological"]      = bool(r.get("biological", False))
            df.at[idx, "llm_bio_subcategory"] = r.get("bio_subcategory")
            df.at[idx, "llm_confidence"]      = r.get("confidence", "low")
            # Other axes remain None — not determinable from a short abstract.
            df.at[idx, "llm_mining_type"]            = None
            df.at[idx, "llm_stage"]                  = None
            df.at[idx, "llm_materials"]              = None
            df.at[idx, "llm_orientation"]            = None
            df.at[idx, "llm_research_approach"]      = None
            df.at[idx, "llm_infrastructure_subtype"] = None


# =============================================================================
# STAGE 2-FULL — 6-axis classification (Sonnet)
# =============================================================================
def classify_stage2_full_batch(client, batch):
    grant_texts = [_build_grant_text(row) for row in batch]
    user_msg = "Characterize these mining-relevant grants across all 6 axes:\n\n" + "\n\n---\n\n".join(grant_texts)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=STAGE2_FULL_MODEL,
                max_tokens=STAGE2_FULL_MAX_TOKENS,
                system=STAGE2_FULL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text
            results = _extract_json_array(raw_text)
            return results, raw_text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  ⚠️  Stage 2-Full attempt {attempt+1} failed: {e}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Stage 2-Full failed after {MAX_RETRIES} attempts: {e}")
                return [{"grant_id": _safe_str(r.get("unique_key")),
                         "biological": False, "bio_subcategory": None,
                         "mining_type": None, "materials": [],
                         "stage": None, "orientation": None,
                         "research_approach": None,
                         "infrastructure_subtype": None,
                         "confidence": "low"} for r in batch], str(e)


def apply_stage2_full_results(df, results_dict):
    for idx, row in df.iterrows():
        gid = str(row.get("unique_key", "")).strip()
        if gid in results_dict:
            r = results_dict[gid]
            df.at[idx, "llm_biological"]            = bool(r.get("biological", False))
            df.at[idx, "llm_bio_subcategory"]       = r.get("bio_subcategory")
            df.at[idx, "llm_mining_type"]           = r.get("mining_type")
            materials = r.get("materials", []) or []
            df.at[idx, "llm_materials"]             = json.dumps(materials) if isinstance(materials, list) else str(materials)
            df.at[idx, "llm_stage"]                 = r.get("stage")
            df.at[idx, "llm_orientation"]           = r.get("orientation")
            df.at[idx, "llm_research_approach"]     = r.get("research_approach")
            df.at[idx, "llm_infrastructure_subtype"] = r.get("infrastructure_subtype")
            df.at[idx, "llm_confidence"]            = r.get("confidence", "low")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("BIOMINING TWO-STAGE LLM CLASSIFIER")
    print("Stage 1 (Haiku): keep/remove · Stage 2-Formula (Haiku): bio only · Stage 2-Full (Sonnet): 6 axes")
    print("=" * 70)
    print()

    # ---- Load step2 output ----
    print(f"Loading: {INFILE}")
    df = pd.read_csv(INFILE, low_memory=False)
    print(f"  {len(df):,} grants loaded")

    if "unique_key" not in df.columns:
        raise RuntimeError("Input CSV is missing unique_key column (should come from step1).")
    if "is_formula_grant" not in df.columns:
        print("  (no is_formula_grant column — treating all grants as non-formula)")
        df["is_formula_grant"] = False
    df["is_formula_grant"] = df["is_formula_grant"].fillna(False).astype(bool)

    # ---- Initialize Stage 1 columns ----
    for col in ("s1_keep", "s1_confidence", "s1_remove_reason"):
        if col not in df.columns:
            df[col] = pd.NA

    # ---- Initialize Stage 2 columns ----
    s2_cols = [
        "llm_biological", "llm_bio_subcategory",
        "llm_mining_type", "llm_materials",
        "llm_stage", "llm_orientation",
        "llm_research_approach", "llm_infrastructure_subtype",
        "llm_confidence",
    ]
    for col in s2_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # ==========================================================================
    # STAGE 1
    # ==========================================================================
    print()
    print("=" * 70)
    print("STAGE 1: KEEP/REMOVE (Haiku)")
    print("=" * 70)

    # Resume — skip already-classified rows
    if RESUME and OUT_STAGE1.exists():
        print(f"  Resuming from: {OUT_STAGE1}")
        existing = pd.read_csv(OUT_STAGE1)
        done_keys = set(existing.loc[existing["s1_keep"].notna(), "unique_key"].astype(str))
        print(f"  Already classified: {len(done_keys):,}")
    else:
        done_keys = set()

    s1_todo_mask = ~df["unique_key"].astype(str).isin(done_keys)
    df_s1_todo   = df.loc[s1_todo_mask].copy().reset_index(drop=True)
    df_s1_done   = df.loc[~s1_todo_mask].copy()

    # Test mode — filter to 70 biomining holdout keys
    if TEST_MODE:
        hits = df_s1_todo["unique_key"].astype(str).isin(VALIDATION_GRANT_IDS)
        n_hits = int(hits.sum())
        if n_hits > 0:
            df_s1_todo = df_s1_todo.loc[hits].reset_index(drop=True)
            print(f"  TEST MODE: {n_hits}/{len(VALIDATION_GRANT_IDS)} holdout grants matched in input")
        else:
            n_fallback = min(TEST_SAMPLE_SIZE, len(df_s1_todo))
            print(f"  TEST MODE: 0 holdout grants matched; falling back to random sample of {n_fallback}")
            df_s1_todo = df_s1_todo.sample(n=n_fallback, random_state=42).reset_index(drop=True)

    print(f"  Grants needing Stage 1: {len(df_s1_todo):,}")

    s1_log = []
    if len(df_s1_todo) > 0:
        client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        rows    = [df_s1_todo.iloc[i] for i in range(len(df_s1_todo))]
        batches = [rows[i:i + STAGE1_BATCH_SIZE] for i in range(0, len(rows), STAGE1_BATCH_SIZE)]

        all_s1 = {}
        print(f"\nProcessing {len(batches)} Stage 1 batches ({STAGE1_BATCH_SIZE} grants each)...\n")
        for bi, batch in enumerate(tqdm(batches, desc="Stage 1")):
            results, raw = classify_stage1_batch(client, batch)
            s1_log.append({"stage": 1, "batch": bi, "raw_response": raw,
                           "ids": [_safe_str(r.get("unique_key")) for r in batch]})
            for r in results:
                gid = str(r.get("grant_id", "")).strip()
                if gid:
                    all_s1[gid] = r
            time.sleep(SLEEP_S)
            if (bi + 1) % 10 == 0:
                apply_stage1_results(df_s1_todo, all_s1)
                pd.concat([df_s1_done, df_s1_todo], ignore_index=True).to_csv(OUT_STAGE1, index=False)
                print(f"\n  Checkpoint after batch {bi+1} — saved {OUT_STAGE1.name}")

        apply_stage1_results(df_s1_todo, all_s1)
        df = pd.concat([df_s1_done, df_s1_todo], ignore_index=True)
        df.to_csv(OUT_STAGE1, index=False)
    else:
        print("  All grants already have Stage 1 decisions. Loading existing file.")
        df = pd.read_csv(OUT_STAGE1, low_memory=False)

    s1_kept   = df.loc[df["s1_keep"] == True].copy()
    s1_excl   = df.loc[df["s1_keep"] == False].copy()
    s1_review = df.loc[df["s1_confidence"] == "low"].copy()
    s1_excl.to_csv(OUT_STAGE1_EXCL, index=False)
    s1_review.to_csv(OUT_STAGE1_REVIEW, index=False)

    print()
    print(f"Stage 1 complete:  KEEP={len(s1_kept):,}  REMOVE={len(s1_excl):,}  low-conf={len(s1_review):,}")

    # ==========================================================================
    # STAGE 2-FORMULA (kept formula grants only)
    # ==========================================================================
    print()
    print("=" * 70)
    print("STAGE 2-FORMULA: biological + bio_subcategory (Haiku)")
    print("=" * 70)

    kept_formula     = s1_kept.loc[s1_kept["is_formula_grant"] == True].copy().reset_index(drop=True)
    kept_non_formula = s1_kept.loc[s1_kept["is_formula_grant"] == False].copy().reset_index(drop=True)

    print(f"  Kept formula grants (→ Stage 2-Formula): {len(kept_formula):,}")
    print(f"  Kept non-formula grants (→ Stage 2-Full): {len(kept_non_formula):,}")

    s2f_log = []
    if len(kept_formula) > 0:
        # Resume
        if RESUME and OUT_STAGE2_FORMULA.exists():
            existing = pd.read_csv(OUT_STAGE2_FORMULA)
            done_fkeys = set(existing.loc[existing["llm_biological"].notna(), "unique_key"].astype(str))
            kept_formula = kept_formula.loc[~kept_formula["unique_key"].astype(str).isin(done_fkeys)].reset_index(drop=True)
            print(f"  Stage 2-Formula resume: {len(done_fkeys):,} already done, {len(kept_formula):,} remaining")

        if len(kept_formula) > 0:
            client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            rows    = [kept_formula.iloc[i] for i in range(len(kept_formula))]
            batches = [rows[i:i + STAGE2_FORMULA_BATCH_SIZE] for i in range(0, len(rows), STAGE2_FORMULA_BATCH_SIZE)]

            all_s2f = {}
            print(f"\nProcessing {len(batches)} Stage 2-Formula batches ({STAGE2_FORMULA_BATCH_SIZE} grants each)...\n")
            for bi, batch in enumerate(tqdm(batches, desc="Stage 2-Formula")):
                results, raw = classify_stage2_formula_batch(client, batch)
                s2f_log.append({"stage": "2-formula", "batch": bi, "raw_response": raw,
                                "ids": [_safe_str(r.get("unique_key")) for r in batch]})
                for r in results:
                    gid = str(r.get("grant_id", "")).strip()
                    if gid:
                        all_s2f[gid] = r
                time.sleep(SLEEP_S)
                if (bi + 1) % 10 == 0:
                    apply_stage2_formula_results(kept_formula, all_s2f)
                    kept_formula.to_csv(OUT_STAGE2_FORMULA, index=False)

            apply_stage2_formula_results(kept_formula, all_s2f)
            kept_formula.to_csv(OUT_STAGE2_FORMULA, index=False)

    # ==========================================================================
    # STAGE 2-FULL (kept non-formula grants)
    # ==========================================================================
    print()
    print("=" * 70)
    print("STAGE 2-FULL: 6-axis classification (Sonnet)")
    print("=" * 70)

    s2u_log = []
    if len(kept_non_formula) > 0:
        if RESUME and OUT_STAGE2_FULL.exists():
            existing = pd.read_csv(OUT_STAGE2_FULL)
            done_ukeys = set(existing.loc[existing["llm_mining_type"].notna(), "unique_key"].astype(str))
            kept_non_formula = kept_non_formula.loc[~kept_non_formula["unique_key"].astype(str).isin(done_ukeys)].reset_index(drop=True)
            print(f"  Stage 2-Full resume: {len(done_ukeys):,} already done, {len(kept_non_formula):,} remaining")

        if len(kept_non_formula) > 0:
            client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            rows    = [kept_non_formula.iloc[i] for i in range(len(kept_non_formula))]
            batches = [rows[i:i + STAGE2_FULL_BATCH_SIZE] for i in range(0, len(rows), STAGE2_FULL_BATCH_SIZE)]

            all_s2u = {}
            print(f"\nProcessing {len(batches)} Stage 2-Full batches ({STAGE2_FULL_BATCH_SIZE} grants each)...\n")
            for bi, batch in enumerate(tqdm(batches, desc="Stage 2-Full")):
                results, raw = classify_stage2_full_batch(client, batch)
                s2u_log.append({"stage": "2-full", "batch": bi, "raw_response": raw,
                                "ids": [_safe_str(r.get("unique_key")) for r in batch]})
                for r in results:
                    gid = str(r.get("grant_id", "")).strip()
                    if gid:
                        all_s2u[gid] = r
                time.sleep(SLEEP_S)
                if (bi + 1) % 10 == 0:
                    apply_stage2_full_results(kept_non_formula, all_s2u)
                    kept_non_formula.to_csv(OUT_STAGE2_FULL, index=False)

            apply_stage2_full_results(kept_non_formula, all_s2u)
            kept_non_formula.to_csv(OUT_STAGE2_FULL, index=False)

    # ==========================================================================
    # STAGE 2-SHORT (non-formula grants with abstracts too short for step3's
    # main input pool). These never hit Stage 1 because step2 filtered them
    # into mining_insufficient_abstract_all_years.csv. We run a lightweight
    # Haiku pass on them — bio + bio_subcategory only — so the dashboard's
    # Non-Categorizable bucket can still report a Biological Share.
    # Formula short-abstract grants are intentionally SKIPPED here: they're
    # counted under the Formula bucket by build_viz_data.py with static
    # non-bio defaults (AML reclamation is overwhelmingly non-biological).
    # ==========================================================================
    print()
    print("=" * 70)
    print("STAGE 2-SHORT: biological + bio_subcategory on short-abstract grants (Haiku)")
    print("=" * 70)

    s2s_log = []
    if SHORT_INFILE.exists():
        df_short = pd.read_csv(SHORT_INFILE, low_memory=False)
        if "is_formula_grant" in df_short.columns:
            is_formula_short = df_short["is_formula_grant"].apply(
                lambda v: str(v).strip().lower() in ("true", "1", "yes", "t"))
            df_short = df_short.loc[~is_formula_short].copy()
        elif "award_type" in df_short.columns:
            is_formula_short = df_short["award_type"].fillna("").astype(str).str.contains(
                "FORMULA", case=False, na=False)
            df_short = df_short.loc[~is_formula_short].copy()
        if "unique_key" in df_short.columns:
            df_short = df_short.drop_duplicates("unique_key", keep="first").reset_index(drop=True)
        # Seed llm_* cols so apply_stage2_short_results can write to them.
        for col in ("llm_biological", "llm_bio_subcategory", "llm_confidence",
                    "llm_mining_type", "llm_stage", "llm_materials",
                    "llm_orientation", "llm_research_approach",
                    "llm_infrastructure_subtype"):
            if col not in df_short.columns:
                df_short[col] = None

        print(f"  Non-formula short-abstract grants: {len(df_short):,}")

        if RESUME and OUT_STAGE2_SHORT.exists():
            existing = pd.read_csv(OUT_STAGE2_SHORT)
            done_skeys = set(existing.loc[existing["llm_biological"].notna(), "unique_key"].astype(str))
            df_short = df_short.loc[~df_short["unique_key"].astype(str).isin(done_skeys)].reset_index(drop=True)
            print(f"  Stage 2-Short resume: {len(done_skeys):,} already done, {len(df_short):,} remaining")

        if len(df_short) > 0:
            client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            rows    = [df_short.iloc[i] for i in range(len(df_short))]
            batches = [rows[i:i + STAGE2_SHORT_BATCH_SIZE] for i in range(0, len(rows), STAGE2_SHORT_BATCH_SIZE)]

            all_s2s = {}
            print(f"\nProcessing {len(batches)} Stage 2-Short batches ({STAGE2_SHORT_BATCH_SIZE} grants each)...\n")
            for bi, batch in enumerate(tqdm(batches, desc="Stage 2-Short")):
                results, raw = classify_stage2_short_batch(client, batch)
                s2s_log.append({"stage": "2-short", "batch": bi, "raw_response": raw,
                                "ids": [_safe_str(r.get("unique_key")) for r in batch]})
                for r in results:
                    gid = str(r.get("grant_id", "")).strip()
                    if gid:
                        all_s2s[gid] = r
                time.sleep(SLEEP_S)
                if (bi + 1) % 10 == 0:
                    apply_stage2_short_results(df_short, all_s2s)
                    # Merge with any existing stage2_short file so resume state is preserved.
                    parts_s = [df_short]
                    if OUT_STAGE2_SHORT.exists():
                        parts_s.insert(0, pd.read_csv(OUT_STAGE2_SHORT, low_memory=False))
                    out_s = pd.concat(parts_s, ignore_index=True, sort=False)
                    out_s = out_s.drop_duplicates("unique_key", keep="last")
                    out_s.to_csv(OUT_STAGE2_SHORT, index=False)

            apply_stage2_short_results(df_short, all_s2s)
            parts_s = [df_short]
            if OUT_STAGE2_SHORT.exists():
                parts_s.insert(0, pd.read_csv(OUT_STAGE2_SHORT, low_memory=False))
            out_s = pd.concat(parts_s, ignore_index=True, sort=False)
            out_s = out_s.drop_duplicates("unique_key", keep="last")
            out_s.to_csv(OUT_STAGE2_SHORT, index=False)
    else:
        print(f"  Skipped — {SHORT_INFILE.name} not found.")

    # ==========================================================================
    # MERGE FINAL CLASSIFIED OUTPUT
    # ==========================================================================
    # Load whatever is on disk from both 2-Formula and 2-Full so restart paths
    # still produce a complete merged file.
    print()
    print("=" * 70)
    print("MERGING FINAL CLASSIFIED OUTPUT")
    print("=" * 70)

    parts = []
    if OUT_STAGE2_FORMULA.exists():
        parts.append(pd.read_csv(OUT_STAGE2_FORMULA, low_memory=False))
    if OUT_STAGE2_FULL.exists():
        parts.append(pd.read_csv(OUT_STAGE2_FULL, low_memory=False))

    # Also fold the Stage 1 REMOVE set so the final CSV covers every input grant
    s1_excl_for_merge = df.loc[df["s1_keep"] == False].copy()
    if not s1_excl_for_merge.empty:
        parts.append(s1_excl_for_merge)

    if parts:
        merged = pd.concat(parts, ignore_index=True, sort=False)
        merged = merged.drop_duplicates(subset=["unique_key"], keep="first")
        merged.to_csv(OUT_CLASSIFIED, index=False)
        print(f"  Saved: {OUT_CLASSIFIED}  ({len(merged):,} rows)")
    else:
        print("  No classification outputs found to merge.")

    # Save log
    with open(OUT_LOG, "w") as f:
        json.dump(s1_log + s2f_log + s2u_log, f, indent=2, default=str)
    print(f"  Log: {OUT_LOG}")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"Stage 1 KEEP:            {len(s1_kept):,}")
    print(f"Stage 1 REMOVE:          {len(s1_excl):,}")
    print(f"  formula (bio only):    {len(s1_kept.loc[s1_kept['is_formula_grant'] == True]):,}")
    print(f"  non-formula (6 axes):  {len(s1_kept.loc[s1_kept['is_formula_grant'] == False]):,}")


if __name__ == "__main__":
    main()
