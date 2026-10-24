# Infant EEG Functional Connectivity: Universality & Longitudinal Identifiability

This repository implements a rigorous, end-to-end computational pipeline to decouple **universal population organization** from **subject-specific neural fingerprinting (differential identifiability)** across 26 infant EEG network feature sets.

The cohort consists of **$N = 19$ longitudinal infants** scanned across two developmental sessions (`ses-1` at ~3 months, `ses-2` at ~6 months) with $K = 12$ artifact-free eyes-closed sleep epochs (2.56 s duration) per session.

---

## Folder and File Structure

Below is a complete map of the project files and directories with short, simple descriptions of their roles:

```text
HonoursM26/
├── run_full_analysis.py            # Master one-click runner executing all analysis phases (1 to 9)
├── ANALYSIS_PIPELINE.md  (Q)          # Complete theoretical framework, mathematical axioms, and blueprint
├── Methodology-Analysis.md         # Detailed methodological notes on Jaccard comparison framework
├── Dataset-Documentation.md        # Summary of the BIDS infant dataset, channels, and recording setup
│
├── analysis_pipeline/              # Core production analysis codebase (Phases 1 through 9)
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Central configuration: N=19 cohort, channels, 26 feature definitions
│   ├── utils.py                    # Jaccard engines, 10-20 layout, Midline 'M' classifiers, bootstrap helpers
│   ├── 01_extract_tensors.py       # Phase 1: Builds standardized (19, 2, 12, n) feature tensors
│   ├── 02_compute_universality.py  # Phase 2: Comparison A (Universality U_1, U_2, bootstrap, permutations)
│   ├── 03_compute_within_session.py# Phase 3: Comparison B (Within-session stability ceilings W_1, W_2)
│   ├── 04_compute_identifiability.py# Phase 4: Comparisons C & D (Cross-session M, I_diff, accuracy)
│   ├── 05_taxonomy_and_variance.py # Phase 5: 4-quadrant taxonomy & 3-way LMM state-trait variance decomposition
│   ├── 06_developmental_and_twins.py# Phase 6: Chronological age Mantel tests, interval decay, twin audit
│   ├── 07_robustness_checks.py     # Phase 7: Evaluates 88 check conditions across 9 sensitivity variations
│   ├── 08_export_tables.py         # Phase 8: Compiles publication Tables T1 through T6 (CSV & Markdown)
│   └── 09_generate_figures.py      # Phase 9: Generates publication Figures 1 through 7 (Vector PDF & PNG)
│
├── tables/                         # Final publication-ready tables (CSV and Markdown formats)
│   ├── Table_T1_Global_Summary.csv/.md   # 26 features x 28 global metrics (U, W, I_diff, Acc, Mantel)
│   ├── Table_T2_Element_Results.csv      # 3,230 element evaluations (region, 3D distance, FDR q, taxonomy)
│   ├── Table_T3_Variance_Partition.csv   # 3,230 element LMM state-trait variance partitions
│   ├── Table_T4_Robustness.csv/.md       # 88 sensitivity evaluations across 9 methodological checks
│   ├── Table_T5_Bookkeeping.csv/.md      # Complete metadata and scan intervals for all 19 infants
│   └── Table_T6_Sibling_Analysis.csv/.md # Male twin pair (sub-NORB00086 & 00087) similarity audit
│
├── figures/                        # Final publication-ready figures (Vector PDF & 300 DPI PNG)
│   ├── README.md                   # Complete figure inventory and scientific captions
│   ├── Figure_1_Identifiability_Matrices # 19x19 cross-session similarity matrices M with highlighted diagonals
│   ├── Figure_1_Supplementary_All_26_Matrices # Complete 26-feature identifiability gallery
│   ├── Figure_2_Universality_Identifiability_Forest # Forest plot comparing U_1, U_2, W_1, I_self, I_others
│   ├── Figure_3_Universality_Individuality_Plane # 2D four-quadrant functional taxonomy plane
│   ├── Figure_4_Topography_Hubs_Fingerprints # 10-20 scalp maps (node identifiability & FDR edges)
│   ├── Figure_5_Developmental_Interval_Scatter # Delta U developmental shifts & scan interval decay
│   ├── Figure_6_Variance_Partition_Stacked # Stacked bar charts of state-trait variance decomposition
│   ├── Figure_7_Ordered_Element_Heatmaps # Age-sorted 171x171 element heatmaps with twin highlighted
│   └── Figure_7_Supplementary_Twin_Distributions # Distribution of pair similarities with twin marked
│
├── processed_tensors/              # Precomputed intermediate arrays and result databases (374 files)
│   ├── README.md                   # Full inventory of all arrays, shapes, and axis ordering
│   ├── X_{fid}.npy (26 files)      # Standardized feature tensors of shape (19, 2, 12, n)
│   ├── J_pair_{fid}_{ses}.npy (52) # Pairwise element Jaccard arrays of shape (171, n)
│   ├── S_mat_{fid}_{ses}.npy (52)  # Symmetric 19x19 recording distance matrices
│   ├── M_mat_{fid}.npy (26 files)  # 19x19 cross-session identifiability matrices
│   ├── M_elem_{fid}.npy (26 files) # 19x19xn cross-session element similarity tensors
│   └── Results JSONs (7 files)     # universality, within_session, identifiability, taxonomy, etc.
│
├── Infants_data/                   # Raw BIDS infant EEG dataset (103 subjects, EDF signals, TSV annotations)
│
└── [Legacy / Exploratory Folders]  # Historical exploratory notebooks (superseded by analysis_pipeline/)
    ├── weighted/                   # Early notebooks for broadband weighted metrics
    ├── binary/                     # Early notebooks for GCC-binarized metrics
    ├── spectralBandSpecificConnectivity/ # Early notebooks for delta, theta, alpha, beta bands
    ├── correlation_matrices_segment_wise_absolute/ # Legacy per-epoch CSV correlation matrices
    └── changePlots/                # Legacy exploratory PNG plots
```

---

## The 4 Comparison Classes

The analysis evaluates network constituents across four fundamental comparison classes using the **weighted Jaccard index**:

* **Comparison A (Between-subject, within-session):** Quantifies population **Universality** ($U_1, U_2$).
* **Comparison B (Within-subject, within-session):** Quantifies **Within-session stability ceiling** ($W_1, W_2$).
* **Comparison C (Within-subject, between-session):** Quantifies longitudinal self-similarity ($I_{\text{self}}$).
* **Comparison D (Between-subject, between-session):** Quantifies the null developmental baseline ($I_{\text{others}}$).

> **Differential Identifiability:** True individual fingerprinting is defined as:
> $$I_{\text{diff}} = I_{\text{self}} - I_{\text{others}}$$
> Subtracting $I_{\text{others}}$ removes shared developmental maturation, isolating true subject specificity.

---

## Running the Pipeline

### Quick Setup (`uv` or `venv`)

Using Python 3.12:

```bash
# Install dependencies
uv sync
# OR with pip
pip install -r requirements.txt
```

### Execute Full Pipeline (Phases 1 to 9)

To run all phases end-to-end (feature extraction, universality, stability, identifiability, taxonomy, developmental analysis, 9 robustness checks, table compilation, and figure generation):

```bash
.venv/bin/python run_full_analysis.py --start-phase 1 --permutations 10000 --workers 8
```

Individual phases can also be executed independently:
```bash
.venv/bin/python -m analysis_pipeline.08_export_tables   # Phase 8: Exports Tables T1-T6
.venv/bin/python -m analysis_pipeline.09_generate_figures # Phase 9: Generates Figures 1-7
```
