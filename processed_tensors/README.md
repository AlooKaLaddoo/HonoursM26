# `processed_tensors/`: Saved In-Between Results

This folder holds the numbers the pipeline calculates along the way. Each phase saves its results here, and later phases (and the tables and figures) load them back. This means you can rerun a later step without redoing everything before it.

There are **374 files**: 364 NumPy arrays (`.npy`), 9 JSON files and this README.

---

## How to read the file names

- **`{fid}`** is the feature ID, `f01` to `f26`. Each one is a different way of describing the brain network (full list [below](#the-26-features)).
- **`{ses}`** is the session: `ses-1` (≈ 3 months) or `ses-2` (≈ 6 months).
- **`n`** is the number of *elements* in a feature:
  - **19** for node features (one value per electrode)
  - **171** for edge features (one value per pair of electrodes)

### Loading a file

```python
import numpy as np, json
X = np.load("processed_tensors/X_f01.npy")                  # a NumPy array
res = json.load(open("processed_tensors/identifiability_results.json"))
res["f01"]["I_diff"]                                        # results are grouped by feature ID
```

---

## Everything in this folder

| Files | How many | Shape | Made by | What it is |
| :--- | :---: | :---: | :---: | :--- |
| `X_{fid}.npy` | 26 | `(19, 2, 12, n)` | Phase 1 | The feature values for every infant, session and epoch. Everything else is built from these. |
| `metadata.json` | 1 | – | Phase 1 | The list of infants, electrode and connection names, feature names, and network-threshold summary numbers. |
| `J_pair_{fid}_{ses}.npy` | 52 | `(171, n)` | Phase 2 | How similar each **pair of infants** is, element by element, within one session. |
| `S_mat_{fid}_{ses}.npy` | 52 | `(19, 19)` | Phase 2 | The same similarity as one overall score per pair of infants, laid out as a grid. |
| `universality_results.json` | 1 | – | Phase 2 | Universality results (how alike infants are). |
| `W_elem_{fid}_{ses}.npy` | 52 | `(19, n)` | Phase 3 | How similar each infant is **to themselves** within one recording. |
| `O_elem_{fid}_{ses}.npy` | 52 | `(19, n)` | Phase 3 | The comparison baseline: each infant compared with **everyone else**. |
| `diff_within_elem_{fid}_{ses}.npy` | 52 | `(19, n)` | Phase 3 | `W_elem` minus `O_elem`: how much more each infant looks like themselves than like others. |
| `within_session_results.json` | 1 | – | Phase 3 | Summary of the within-recording results. |
| `M_{fid}.npy` | 26 | `(19, 19)` | Phase 4 | Similarity of every infant at 3 months to every infant at 6 months. |
| `M_mat_{fid}.npy` | 26 | `(19, 19)` | Phase 4 | An exact copy of `M_{fid}.npy`, kept under both names. |
| `M_elem_{fid}.npy` | 26 | `(19, 19, n)` | Phase 4 | The same 3-month vs. 6-month comparison, broken down element by element. |
| `identifiability_results.json` | 1 | – | Phase 4 | Identifiability results (can we recognise each infant across sessions?). |
| `taxonomy_results.json` | 1 | – | Phase 5 | Which elements are shared, personal, both or neither. |
| `variance_results.json` | 1 | – | Phase 5 | How much of the variation comes from the infant, the session, or moment-to-moment noise. |
| `developmental_results.json` | 1 | – | Phase 6 | How results relate to age and to the time between sessions. |
| `twin_results.json` | 1 | – | Phase 6 | How similar the twin pair is compared with everyone else. |
| `robustness_results.json` | 1 | – | Phase 7 | Results of the 88 robustness re-runs. |

---

## The arrays in more detail

### `X_{fid}.npy`: the feature values (phase 1)

Shape `(19, 2, 12, n)`. Each position along the shape means:

| Position | Size | Meaning |
| :---: | :---: | :--- |
| 1st | 19 | Infant, in the order listed in `metadata.json` (`sub-NORB00064` … `sub-NORB00103`) |
| 2nd | 2 | Session: `0` = `ses-1`, `1` = `ses-2` |
| 3rd | 12 | Epoch: the 12 clean 2.56-second segments, in time order |
| 4th | n | Element: an electrode (n = 19) or an electrode pair (n = 171) |

So `X[0, 1, 5, :]` is the first infant, second session, sixth epoch, all elements.

**Element order:**
- **Electrodes (19):** `Fp1, Fp2, F3, F4, F7, F8, FZ, C3, C4, CZ, T3, T4, T5, T6, P3, P4, PZ, O1, O2` (front of the head to the back).
- **Electrode pairs (171):** every pair of the electrodes above, in that order: `Fp1-Fp2, Fp1-F3, Fp1-F4, …, O1-O2`.

Both lists are saved in `metadata.json` under `elements`.

### `J_pair` and `S_mat`: infant vs. infant within a session (phase 2)

- **`J_pair_{fid}_{ses}.npy`**, shape `(171, n)`: one row for each of the 171 possible pairs of infants, and one column per element. Each value is a similarity score from 0 to 1, averaged over the 12 epochs. The pairs are in the order `(0,1), (0,2), …, (17,18)`, the same as Python's `itertools.combinations(range(19), 2)`.
- **`S_mat_{fid}_{ses}.npy`**, shape `(19, 19)`: one overall similarity score for each pair of infants, using all elements together. The grid is symmetric (`S[a, b] == S[b, a]`). The diagonal is blank (`NaN`) because an infant isn't compared with themselves here.

### `W_elem`, `O_elem`, `diff_within_elem`: stability within one recording (phase 3)

Each recording's 12 epochs are split in half: epochs 1–6 and epochs 7–12.

- **`W_elem`**: each infant's first half compared with **their own** second half.
- **`O_elem`**: each infant's first half compared with **every other infant's** second half, then averaged.
- **`diff_within_elem`**: `W_elem − O_elem`. Above 0 means the infant is more like themselves than like others, even within a single recording.

All three have shape `(19, n)`: one row per infant, one column per element.

### `M`, `M_mat`, `M_elem`: 3 months vs. 6 months (phase 4)

- **`M_{fid}.npy`** (and its copy **`M_mat_{fid}.npy`**), shape `(19, 19)`. Rows are infants at `ses-1`, columns are infants at `ses-2`. `M[a, b]` is how similar infant `a` at 3 months is to infant `b` at 6 months.
  - **Diagonal** (`M[a, a]`): an infant compared with their own later self.
  - **Everything else**: an infant compared with somebody else.
- **`M_elem_{fid}.npy`**, shape `(19, 19, n)`: the same grid, but with a separate score for each element.

The identifiability score `I_diff` is (average of the diagonal) − (average of everything else), multiplied by 100.

### Blank values (`NaN`)

Some node-feature arrays (for example `J_pair_f02_*` and `M_elem_f02`) contain a few `NaN` values. These appear when **both** recordings being compared have a value of 0 for an element, so the similarity score is 0 ÷ 0 and can't be calculated. The pipeline skips these with `np.nanmean`. Do the same if you average these arrays yourself.

---

## What's inside each JSON file

Except for `metadata.json` and `robustness_results.json`, every JSON file is a dictionary keyed by feature ID (`"f01"` … `"f26"`). Each entry also repeats the feature's `fid` and `name`.

| File | Main contents |
| :--- | :--- |
| `metadata.json` | `cohort` (the 19 infant IDs, sessions, number of epochs), `elements` (electrode and pair names), `features` (name, type, size, band for each feature), `tau_density_summary` (threshold and edge-count summary for the broadband binary networks). The current code also writes `frequency_band_summaries` (the same summary for each frequency band), but the saved file predates that and doesn't have it yet. |
| `universality_results.json` | For each session: the overall score `U_global`, its p-value and bootstrap confidence interval, per-element scores `U_elem` with p-values and FDR-corrected q-values, per-infant scores, and averages by brain region. `developmental_shift` holds the change from session 1 to 2 (`delta_U`). |
| `within_session_results.json` | For each session: `W` (self), `O` (others) and their difference, each as an overall score with a bootstrap interval, per element, and per infant, plus an `odd_even_sensitivity` check (splitting odd vs. even epochs instead of first vs. second half). |
| `identifiability_results.json` | `I_self`, `I_others`, `I_diff` and its p-value; identification accuracy in % (`acc_forward`: 3→6 months, `acc_backward`: 6→3 months, `acc_mean`) with p-values; bootstrap intervals; per-element values with p- and q-values; per-infant values. |
| `taxonomy_results.json` | `counts`: how many elements fall into each group. `classes`: the group for every element, in element order. The groups are **`Universal`** (shared by everyone), **`Individual`** (personal), **`Universal and Individual`**, and **`Neither`**. An element counts as "universal" if it is significant in both sessions, and as "individual" if its `I_diff` is significant (q < 0.05). |
| `variance_results.json` | `elements`: one entry per element giving the share of variation due to the **infant** (`trait_prop`), the **session** (`session_prop`) and **epoch-to-epoch noise** (`epoch_prop`). Also included: the same three shares after adjusting for age (`*_adj`), and whether the model fit worked (`converged`, `skipped`). |
| `developmental_results.json` | The change in universality between sessions (`delta_U_global`, confidence interval, p-value); Mantel tests checking whether infants closer in age are more similar (`mantel_ses1`, `mantel_ses2`); correlations with the time gap between sessions (`interval_corr`); and a close look at two infants with unusually short gaps (`individual_infants`: `sub-NORB00064`, 4 days; `sub-NORB00069`, 42 days). |
| `twin_results.json` | For the twins `sub-NORB00086` and `sub-NORB00087`: their similarity in each session, and how it ranks among all 171 infant pairs, among pairs that include either twin, and among pairs of infants of a similar age. Also included: whether the pipeline confuses one twin for the other across sessions (`cross_identification`), and the elements where the twins are unusually similar in both sessions (`shared_elements`). |
| `robustness_results.json` | A **list** of 88 rows (this is Table T4). Each row has `Check ID`, `Sensitivity Analysis`, `Feature ID`, `Metric` (`U_1` or `I_diff (%)`), `Baseline`, `Perturbed`, `Delta`, `Pct Change (%)` and a one-line `Conclusion`. |

### The 9 robustness checks

Each check changes one method choice and re-runs the key numbers to see whether the conclusions still hold.

| Check | What changes | Rows |
| :--- | :--- | :---: |
| 1 | How 0 ÷ 0 similarity is handled (count it as 0 instead of skipping it) | 8 |
| 2 | Compare all 12 × 12 epoch combinations instead of matching epoch 1 to 1, 2 to 2, … | 16 |
| 3 | Use only 6 epochs instead of 12 | 16 |
| 4 | Leave out the twins (17 infants) | 20 |
| 5 | Compare only the *pattern* of values, not their overall size | 14 |
| 6 | Use a different EEG reference (average reference) | 2 |
| 7 | Use a connectivity measure (dwPLI) that ignores signal leaking between nearby electrodes | 2 |
| 8 | Build binary networks from the 58 strongest connections instead of the adaptive threshold | 2 |
| 9 | Remove the effect of the physical distance between electrodes | 8 |

---

## The 26 features

"Weighted" features keep the connection strengths. "Binary" features only record whether a connection is present or not. "Broadband" uses the whole EEG signal; the other bands each use one frequency range.

| ID | Name | Band | Weighted / Binary | n |
| :--- | :--- | :--- | :--- | :---: |
| f01 | `broadband_weighted_correlation` | Broadband | Weighted | 171 |
| f02 | `broadband_weighted_node_betweenness` | Broadband | Weighted | 19 |
| f03 | `broadband_weighted_edge_betweenness` | Broadband | Weighted | 171 |
| f04 | `broadband_weighted_node_strength` | Broadband | Weighted | 19 |
| f05 | `broadband_weighted_edge_strength` | Broadband | Weighted | 171 |
| f06 | `broadband_weighted_node_eigenvector` | Broadband | Weighted | 19 |
| f07 | `broadband_weighted_edge_eigenvector` | Broadband | Weighted | 171 |
| f08 | `broadband_weighted_node_closeness` | Broadband | Weighted | 19 |
| f09 | `broadband_weighted_edge_closeness` | Broadband | Weighted | 171 |
| f10 | `broadband_binary_adjacency` | Broadband | Binary | 171 |
| f11 | `broadband_binary_node_betweenness` | Broadband | Binary | 19 |
| f12 | `broadband_binary_edge_betweenness` | Broadband | Binary | 171 |
| f13 | `broadband_binary_node_degree` | Broadband | Binary | 19 |
| f14 | `broadband_binary_edge_strength` | Broadband | Binary | 171 |
| f15 | `broadband_binary_node_eigenvector` | Broadband | Binary | 19 |
| f16 | `broadband_binary_edge_eigenvector` | Broadband | Binary | 171 |
| f17 | `broadband_binary_node_closeness` | Broadband | Binary | 19 |
| f18 | `broadband_binary_edge_closeness` | Broadband | Binary | 171 |
| f19 | `delta_weighted_correlation` | Delta (0.5–4 Hz) | Weighted | 171 |
| f20 | `delta_binary_adjacency` | Delta (0.5–4 Hz) | Binary | 171 |
| f21 | `theta_weighted_correlation` | Theta (4–8 Hz) | Weighted | 171 |
| f22 | `theta_binary_adjacency` | Theta (4–8 Hz) | Binary | 171 |
| f23 | `alpha_weighted_correlation` | Alpha (8–13 Hz) | Weighted | 171 |
| f24 | `alpha_binary_adjacency` | Alpha (8–13 Hz) | Binary | 171 |
| f25 | `beta_weighted_correlation` | Beta (13–30 Hz) | Weighted | 171 |
| f26 | `beta_binary_adjacency` | Beta (13–30 Hz) | Binary | 171 |
