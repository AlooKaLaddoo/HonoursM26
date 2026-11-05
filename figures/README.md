# Figures

This folder holds the figures made by **Phase 9** of the analysis pipeline (`analysis_pipeline/09_generate_figures.py`), following the "Figures" list of the analysis plan. Each figure is saved as a vector PDF (sharp at any zoom) and, except for the multi-page PDFs, as a 300 DPI PNG.

**Every number on a figure is read from the results files in `processed_tensors/`;** nothing is typed in by hand, so the figures stay correct when the analysis is rerun. Titles describe what is plotted, not what it means. Where a figure shows only a few key features, a supplementary file covers all 26 feature sets.

$I_{\text{diff}} = I_{\text{self}} - I_{\text{others}}$ is shown in Jaccard units, unscaled (Step 7.5 of the plan), like every other similarity. "q" is the Benjamini–Hochberg FDR-corrected p-value.

---

## Main figures

| Figure | Files | What it shows |
| :--- | :--- | :--- |
| **Figure 1** | `Figure_1_Identifiability_Matrices` (.pdf, .png) | The $19 \times 19$ identifiability matrix $M$ for six key features (f01, f04, f06, f10, f13, f24): rows are infants at `ses-1`, columns the same infants at `ses-2`, in the same order. Cyan boxes mark the diagonal (each infant with itself). Each panel has its own colour scale and colour bar, and its title gives $I_{\text{diff}}$ with its q across the 26 features and the mean identification accuracy. |
| **Figure 2** | `Figure_2_Universality_Identifiability_Forest` (.pdf, .png) | For every feature: $W_1$, $W_2$ (within-session), $U_1$, $U_2$ (between infants), $I_{\text{self}}$ and $I_{\text{others}}$ (across sessions), each as its estimate with a 95% infant bootstrap CI. |
| **Figure 3** | `Figure_3_Universality_Individuality_Plane` (.pdf, .png) | The universality–individuality plane (Step 5a) for f01, f06, f10 and f24: one point per element at $x = U(e) = (U_1(e) + U_2(e))/2$ and $y = I_{\text{diff}}(e)$. Colour: region (node features) or region pair (edge features). Marker: circle for nodes and within-region edges, triangle for between-region edges. Filled marker: $U(e)$ significant at both sessions; open: not significant. Black ring: $I_{\text{diff}}(e)$ significant. The class counts are in each title. The classes come from the FDR-corrected tests, not from the position in the plane. |
| **Figure 4** | `Figure_4_Topography_Hubs_Fingerprints` (.pdf, .png) | Scalp maps of $I_{\text{diff}}(e)$ for two node features (f06, f04) and two edge features (f20, f12). Nodes are coloured by $I_{\text{diff}}(e)$ on a scale centred on 0 (red above, blue below), with a thick ring where $q < 0.05$ and the three largest values labelled. For edge features, only edges with $q < 0.05$ are drawn, and each panel says how many there are. |
| **Figure 5** | `Figure_5_Developmental_Interval_Scatter` (.pdf, .png) | A: $\Delta U = U_2 - U_1$ for every feature with its 95% bootstrap CI; green or red when $q < 0.05$ (positive or negative), grey otherwise. B: each infant's self-similarity $M[a, a]$ (f01) against the time between sessions, with `sub-NORB00064` and `sub-NORB00069` labelled with their interval and value, a least-squares line, and the Spearman $\rho$ with its 95% infant bootstrap CI, permutation p and q. |
| **Figure 6** | `Figure_6_Variance_Partition_Stacked` (.pdf, .png) | The variance partition of Phase 5b as stacked bars (infant / trait, session, epoch / state). A: the mean over the elements of each feature, with the number of converged fits. B: the mean per region pair for f01. Only converged fits are averaged; binary adjacency features are not modelled. |
| **Figure 7** | `Figure_7_Ordered_Element_Heatmaps` (.pdf, .png) | For f01, the similarity $J(e)$ of every infant pair (rows) and element (columns) at `ses-1` and `ses-2`, with rows sorted by the pair's age difference $\lvert\Delta\text{age}\rvert$ (smallest at the top) and a fixed colour scale from 0 to 1. Columns are grouped by region pair. The strip above each heatmap gives each column's hemispheric category: L (left), R (right) and **M (midline)**. A green line marks the twin pair, with its age difference. |
| **Sibling figure** | `Figure_7_Supplementary_Twin_Distributions` (.pdf, .png) | For f01, the distribution of the recording similarity $S$ over all 171 infant pairs at each session. Dashed red line: the twin pair, with its percentile rank. Tick marks: the two matched comparison sets of the sibling analysis (the pairs involving infant 86 or 87, and the non-sibling pairs whose ages differ by at most 7 days), each with its size and median. |

## Supplementary figures (all 26 feature sets)

| File | What it shows |
| :--- | :--- |
| `Figure_1_Supplementary_All_26_Matrices` (.pdf, .png) | Figure 1 for every feature, each with its own colour bar. |
| `Figure_3_Supplementary_All_26_Planes` (.pdf, .png) | Figure 3 for every feature. |
| `Figure_4_Supplementary_All_26_Scalp_Maps` (.pdf, .png) | Figure 4 for every feature. |
| `Figure_6_Supplementary_Regions_All_Features` (.pdf, .png) | Figure 6B for every modelled feature: by region (node features) or region pair (edge features). |
| `Figure_7_Supplementary_All_26_Heatmaps.pdf` | Figure 7 for every feature, one feature per page (26 pages). |
| `Figure_7_Supplementary_Twin_Distributions_All_26.pdf` | The sibling figure for every feature, one feature per page (26 pages). |
| `Figure_7_Supplementary_Threshold_Bars` (.pdf, .png) | Threshold bars for f01, the region summary of the legacy notebooks (not in the plan's list). For each session, the percentage of the element similarities $J(e)$ (all 171 infant pairs × the elements of a region pair) above 0.75 and 0.90 (top row, left and right bar) and above 0.50 and 0.30 (bottom row). Each bar is stacked by hemispheric category (L-L, R-R, L-R, L-M, R-M, M-M; L, R, M for node features), with the bar total written on top and the number of elements under each label. Undefined (0/0) values are left out, and the panel title says how many there are. |
| `Figure_7_Supplementary_Threshold_Bars_All_26.pdf` | The threshold bars for every feature, one feature per page (26 pages). Node features are grouped by region. |
| `Figure_7_Supplementary_Threshold_Bars_Weighted_Grid.pdf` | The threshold bars of the weighted node features (f02, f04, f06, f08; pages 1–2) and of the same four measures computed on edges (f03, f05, f07, f09; pages 3–4). Each page shows one threshold pair: J > 0.75 and 0.90 (pages 1 and 3) or J > 0.50 and 0.30 (pages 2 and 4), with one column per feature, ses-1 on top and ses-2 below. Same numbers as the per-feature pages, with larger text. On the edge pages the numbers and region-pair labels are written vertically to fit the 15 narrow bar pairs. Segments of 6% or less (nodes) or 12% or less (edges) are not labelled; every bar total is. |
| `Figure_7_Supplementary_Threshold_Bars_Weighted_{Nodes,Edges}_{J075_J090,J050_J030}.png` | The four pages of the grid above as 150 DPI PNGs, named by element type and threshold pair. |

All figures use Matplotlib with embedded TrueType fonts (Type 42) and the `DejaVu Sans` typeface.
