# Infant EEG Brain Networks: What Is Shared and What Is Personal?

This project looks at brain-network patterns in infant EEG recordings and asks two simple questions:

1. **What do all infants have in common?** We call this *universality*.
2. **What makes each infant unique, and does it stay the same as they grow?** We call this *identifiability*: can we pick out the same baby again from a later recording, like a fingerprint?

We look at 26 different ways of describing a brain network (the "features"), and run the same analysis on each one.

## The data at a glance

- **19 infants**, each recorded twice: `ses-1` usually at about 3 months old and `ses-2` usually at about 6 months old. Exact ages and gaps vary; see Table T5.
- From each recording we use **12 clean, eyes-closed sleep segments** ("epochs"), each 2.56 seconds long.
- The raw recordings come from a public infant EEG dataset with 103 babies (OpenNeuro ds004577; see `Infants_data/`). We use the 19 who have enough clean data in both sessions.

---

## Quick start

**1. Install** (needs Python 3.12):

```bash
uv sync
```

No `uv`? Use a normal virtual environment instead:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e .
```

**2. Run the analysis:**

```bash
.venv/bin/python run_full_analysis.py
```

This runs every step from phase 2 onwards and writes the results to `tables/`, `figures/` and `processed_tensors/`.

It skips phase 1 by default, because phase 1 only rebuilds the input data, which is already saved in `processed_tensors/`. To rebuild everything from the raw recordings, start at phase 1:

```bash
.venv/bin/python run_full_analysis.py --start-phase 1 --permutations 10000 --workers 8
```

- `--permutations`: how many random shuffles to use for the significance tests (default 10,000; fewer runs faster).
- `--workers`: how many CPU cores to use (default 8).

You can also run a single step on its own, for example:

```bash
.venv/bin/python -m analysis_pipeline.08_export_tables     # rebuild the tables
.venv/bin/python -m analysis_pipeline.09_generate_figures  # redraw the figures
```

---

## The main idea: four kinds of comparison

Every result comes from comparing two recordings and asking how similar they are. We measure similarity with the **weighted Jaccard index**, a score from 0 (nothing in common) to 1 (identical).

There are four kinds of pairs we can compare:

| | **Same session** | **Different sessions (3 → 6 months)** |
| :--- | :--- | :--- |
| **Same baby** | **B: Short-term stability** ($W$): how consistent a baby is within one recording | **C: Self-similarity** ($I_{\text{self}}$): how much a baby resembles their own earlier recording |
| **Different babies** | **A: Universality** ($U$): how much babies resemble each other | **D: Baseline** ($I_{\text{others}}$): how much a baby resembles *other* babies' recordings from the other session |

**The key number is the difference between C and D:**

$$
I_{\text{diff}} = I_{\text{self}} - I_{\text{others}}
$$

All babies change in similar ways between 3 and 6 months. Subtracting D removes that shared change, so whatever is left is what is really personal to each baby. If $I_{\text{diff}}$ is above zero, babies look more like their own past selves than like anyone else.

---

## The analysis steps (phases)

| Phase | File | What it does |
| :---: | :--- | :--- |
| 1 | `01_extract_tensors.py` | Turns the raw recordings into the 26 feature sets and saves them. |
| 2 | `02_compute_universality.py` | Comparison A: how similar babies are to each other. |
| 3 | `03_compute_within_session.py` | Comparison B: how stable each baby is within one recording. |
| 4 | `04_compute_identifiability.py` | Comparisons C and D: can we recognise each baby across sessions? |
| 5 | `05_taxonomy_and_variance.py` | Sorts each brain connection into "shared", "personal", "both" or "neither", and measures how much of the variation comes from the baby, the session and moment-to-moment noise. |
| 6 | `06_developmental_and_twins.py` | Checks how results relate to age and to the time between sessions, and looks at the one twin pair in the group. |
| 7 | `07_robustness_checks.py` | Repeats the key results with 9 different method choices (88 runs in total) to make sure they hold up. |
| 8 | `08_export_tables.py` | Writes the final tables T1–T6. |
| 9 | `09_generate_figures.py` | Draws the final Figures 1–7. |

---

## What's in each folder

```text
HonoursM26/
├── run_full_analysis.py            # Runs all the phases in order (start here)
├── pyproject.toml / uv.lock        # List of required Python packages
├── .python-version                 # Python version used (3.12)
├── ANALYSIS_PIPELINE.md            # Full write-up of the method and the maths
├── Dataset-Documentation.md        # Notes on the dataset and the recording setup
├── eegMontage.png                  # Picture of where the EEG electrodes sit on the head
│
├── analysis_pipeline/              # The analysis code (one file per phase)
│   ├── config.py                   # Settings: file paths, the 19 infants, channels, the 26 features
│   ├── utils.py                    # Shared helper functions (similarity scores, electrode layout, statistics)
│   └── 01_…py to 09_…py            # The nine phases (see the table above)
│
├── tables/                         # Final results tables (CSV, plus Markdown for most)
│   ├── Table_T1_Global_Summary     # Headline numbers for each of the 26 features
│   ├── Table_T2_Element_Results    # Results for every single node and connection (3,230 rows)
│   ├── Table_T3_Variance_Partition # Baby vs. session vs. noise breakdown for each node/connection
│   ├── Table_T4_Robustness         # Results of the 88 robustness runs
│   ├── Table_T5_Bookkeeping        # Ages and recording dates for the 19 infants
│   └── Table_T6_Sibling_Analysis   # How similar the twin pair (sub-NORB00086 & 00087) is
│
├── figures/                        # Final figures (each as PDF and PNG)
│   ├── README.md                   # What each figure shows
│   ├── Figure_1 …                  # Baby-vs-baby similarity grids across sessions
│   ├── Figure_2 …                  # Side-by-side comparison of all the similarity scores
│   ├── Figure_3 …                  # "Shared vs. personal" map of every node and connection
│   ├── Figure_4 …                  # Head maps showing which brain areas are most personal
│   ├── Figure_5 …                  # Changes with age and with time between sessions
│   ├── Figure_6 …                  # Baby vs. session vs. noise breakdown
│   └── Figure_7 …                  # Pair-by-pair similarity heatmaps and the twin check
│
├── processed_tensors/              # Saved in-between results, so later phases can rerun quickly (374 files)
│   ├── README.md                   # What every file contains and its shape
│   ├── X_*.npy                     # The 26 feature sets (output of phase 1)
│   ├── J_pair_*, S_mat_*           # Baby-to-baby similarity within a session (phase 2)
│   ├── W_elem_*, O_elem_*, diff_within_elem_*  # Within-session stability (phase 3)
│   ├── M_*, M_mat_*, M_elem_*      # Cross-session similarity (phase 4)
│   ├── metadata.json               # The infant list, electrode order and feature names
│   └── *_results.json              # Summary results from each phase
│
├── Infants_data/                   # The raw EEG dataset (103 infants, standard BIDS format)
│
└── legacyCode/                     # Older exploratory notebooks and scripts, kept for reference
    ├── correlation_matrices_segment_wise_absolute/  # Still used: input data for phase 1 (see below)
    ├── weighted/, binary/, spectralBandSpecificConnectivity/  # Early notebooks
    ├── verify_tensors_against_notebooks.py           # Checks the new code matches the old notebooks
    └── …                                             # Other early experiments and plots
```

**Where phase 1 gets its data:**

- Features f01–f18 (whole-signal features) are built from the correlation files in `legacyCode/correlation_matrices_segment_wise_absolute/`. This location is set by `CORR_MATRICES_DIR` in `analysis_pipeline/config.py`.
- Features f19–f26 (frequency-band features: delta, theta, alpha, beta) are built directly from the raw recordings in `Infants_data/`.

Some local notes (`Methodology-Analysis.md`, `1-ANALYSIS_PIPELINE.md`, `furtherAnalysis.pdf`) are deliberately left out of git.

---

## Glossary

- **EEG**: a recording of the brain's electrical activity from electrodes on the scalp.
- **Session**: one recording visit (`ses-1` ≈ 3 months, `ses-2` ≈ 6 months).
- **Epoch**: a short 2.56-second slice of a recording.
- **Node**: one electrode (there are 19).
- **Edge / connection**: a link between two electrodes (there are 171 pairs).
- **Feature**: one way of describing the network, for example how strongly each pair of electrodes is linked, or how central each electrode is. There are 26.
- **Weighted Jaccard index**: a similarity score between 0 and 1.
- **Permutation test**: checks whether a result could have happened by chance, by shuffling the data many times.
- **FDR correction**: an adjustment that stops us finding false positives when we run many tests at once.
