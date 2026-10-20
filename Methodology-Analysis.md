## 1. Subject Cohort & Demographic Information

### A. Demographic Table of the 20 Subjects (Age, Sex, Session Counts)
Demographic data were retrieved from [`Infants_data/participants.tsv`](file:///home/aloo/HonoursM26/Infants_data/participants.tsv) (biological sex) and [`age_acq_time_sorted.csv`](file:///home/aloo/HonoursM26/age_acq_time_sorted.csv) / BIDS `*_scans.tsv` files (decimal age in years at acquisition).

| Participant ID | Sex | `ses-1` Age (Years) | `ses-1` Age (Months) | `ses-1` Age (Days) | `ses-1` Eyes-Closed Windows | Total Longitudinal Sessions | All Sessions Available |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **sub-NORB00096** | **M** | **0.2274** | 2.73 mo | 83.1 d | 20 | 2 | ses-1 (0.227y), ses-2 (0.500y) |
| **sub-NORB00086** | **M** | **0.2377** | 2.85 mo | 86.8 d | 20 | 2 | ses-1 (0.238y), ses-2 (0.525y) |
| **sub-NORB00087** | **M** | **0.2377** | 2.85 mo | 86.8 d | 20 | 2 | ses-1 (0.238y), ses-2 (0.525y) |
| **sub-NORB00103** | **F** | **0.2410** | 2.89 mo | 88.0 d | 35 | 2 | ses-1 (0.241y), ses-2 (0.523y) |
| **sub-NORB00079** | **M** | **0.2459** | 2.95 mo | 89.8 d | 15 | 2 | ses-1 (0.246y), ses-2 (0.472y) |
| **sub-NORB00080** | **M** | **0.2486** | 2.98 mo | 90.8 d | 20 | 2 | ses-1 (0.249y), ses-2 (0.558y) |
| **sub-NORB00089** | **M** | **0.2500** | 3.00 mo | 91.3 d | 24 | 2 | ses-1 (0.250y), ses-2 (0.547y) |
| **sub-NORB00077** | **M** | **0.2514** | 3.02 mo | 91.8 d | 20 | 2 | ses-1 (0.251y), ses-2 (0.492y) |
| **sub-NORB00078** | **M** | **0.2514** | 3.02 mo | 91.8 d | 20 | 2 | ses-1 (0.251y), ses-2 (0.477y) |
| **sub-NORB00090** | **F** | **0.2514** | 3.02 mo | 91.8 d | 15 | 2 | ses-1 (0.251y), ses-2 (0.530y) |
| **sub-NORB00083** | **F** | **0.2525** | 3.03 mo | 92.2 d | 20 | 2 | ses-1 (0.252y), ses-2 (0.544y) |
| **sub-NORB00070** | **M** | **0.2541** | 3.05 mo | 92.8 d | 15 | 3 | ses-1 (0.254y), ses-2 (0.505y), ses-3 (0.797y) |
| **sub-NORB00094** | **F** | **0.2642** | 3.17 mo | 96.5 d | 12 | 2 | ses-1 (0.264y), ses-2 (0.542y) |
| **sub-NORB00075** | **M** | **0.2666** | 3.20 mo | 97.4 d | 20 | 3 | ses-1 (0.267y), ses-2 (0.505y), ses-3 (0.847y) |
| **sub-NORB00084** | **F** | **0.2775** | 3.33 mo | 101.4 d | 15 | 2 | ses-1 (0.278y), ses-2 (0.556y) |
| **sub-NORB00081** | **F** | **0.2923** | 3.51 mo | 106.8 d | 15 | 2 | ses-1 (0.292y), ses-2 (0.572y) |
| **sub-NORB00069** | **F** | **0.3852** | 4.62 mo | 140.7 d | 15 | 4 | ses-1 (0.385y), ses-2 (0.500y), ses-3 (0.754y), ses-4 (1.000y) |
| **sub-NORB00064** | **M** | **0.4262** | 5.11 mo | 155.7 d | 13 | 3 | ses-1 (0.426y), ses-2 (0.437y), ses-3 (0.751y) |
| **sub-NORB00065** | **M** | **0.4344** | 5.21 mo | 158.7 d | 20 | 2 | ses-1 (0.434y), ses-2 (0.798y) |
| **sub-NORB00082** | **M** | **0.5388** | 6.47 mo | 196.8 d | 20 | 2 | ses-1 (0.539y), ses-2 (0.746y) |

#### Cohort Demographic Summary:
- **Total Infants**: 20
- **Sex Breakdown**: **13 Males (65%)**, **7 Females (35%)**
- **Age Distribution at `ses-1`**:
  - **Mean Age**: 3.50 months (~106.6 days, 0.2917 years)
  - **Median Age**: 3.03 months (~92.0 days, 0.2519 years)
  - **Range**: 2.73 months to 6.47 months (83.1 days to 196.8 days, 0.2274 to 0.5388 years)
  - **Standard Deviation**: 1.05 months (~32.0 days)
- **Session Breakdown**:
  - 16 infants with 2 sessions (`ses-1`, `ses-2`)
  - 3 infants with 3 sessions (`ses-1`, `ses-2`, `ses-3`)
  - 1 infant with 4 sessions (`ses-1`, `ses-2`, `ses-3`, `ses-4`)

---

### B. Why Were These Exact 20 Subjects Chosen?
The subjects were chosen through a two-stage filtering process:

1. **Criterion 1: Longitudinal Follow-up (Multi-Session Availability)**:
   - The entire open-access dataset (`ds004577`) consists of 103 infants (`sub-NORB00001` to `sub-NORB00103`).
   - Out of the 103 infants, **81 infants** had only a single recording (`ses-1`), while only **22 infants** had longitudinal follow-up recordings (`ses-1` and at least `ses-2`).
   - Because this project was designed to study development and support intrasession/intersession stability and longitudinal comparisons, only multi-session candidates were shortlisted.

2. **Criterion 2: Sufficient Artifact-Free Quiet Sleep / Eyes-Closed Windows ($\ge 12$ Windows)**:
   - To compute robust, segment-wise topological distributions and inter-subject Jaccard similarities, each subject required a minimum common baseline of artifact-free segments.
   - In [`minCommonSegments.py`](file:///home/aloo/HonoursM26/minCommonSegments.py#L22-L55), the visual inspection annotation tables (`*_task-EEG_annotations.tsv` in `Infants_data/derivatives/NeuronicEEG/`) were audited across all 22 longitudinal infants:
     - `sub-NORB00074`: only **1** valid window in `ses-1` (insufficient data).
     - `sub-NORB00085`: only **8** valid windows in `ses-1` (insufficient data).
     - **The remaining 20 infants all had $\ge 12$ valid windows** in `ses-1` (minimum count was 12 for `sub-NORB00094`).
   - Therefore, the cohort was finalized as the **20 longitudinal infants who met the $\ge 12$ window threshold in session 1**:
     ```python
     subject_ids = [64, 65, 69, 70, 75, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 89, 90, 94, 96, 103]
     ```

---

### C. Window Selection Method
The 12 windows were chosen as the **first 12 chronological windows**:
- In the pipeline, segment files or extracted window lists are sorted chronologically and sliced using `[:12]` / `[:n_segments]`:
  ```python
  csv_files = sorted(subj_dir.glob('*.csv'))
  for csv_file in csv_files[:n_segments]:  # n_segments = 12
  ```
- *(Sources: [`weighted/correlationJaccard.py#L55-L62`](file:///home/aloo/HonoursM26/weighted/correlationJaccard.py#L55-L62), [`binary/binaryCorrelationJaccard.py#L92-L98`](file:///home/aloo/HonoursM26/binary/binaryCorrelationJaccard.py#L92-L98), [`segmentWIseChange_refactored.py#L268-L273`](file:///home/aloo/HonoursM26/segmentWIseChange_refactored.py#L268-L273))*.

---

## 2. Filtering & Signal Preprocessing

### A. Extra 0.5 Hz High-Pass Filter
- **Broad-Band / Pearson Analysis**: No extra software high-pass filter was applied in Python. In [`pearson_correlation_calculation.py#L55-L61`](file:///home/aloo/HonoursM26/pearson_correlation_calculation.py#L55-L61), `mne.io.read_raw_edf` loads the continuous data without calling `.filter()`. The recording hardware and acquisition system had already applied a physical **0.5 Hz high-pass and 30 Hz low-pass filter** during data collection (documented in [`Dataset-Documentation.md#L102-L105`](file:///home/aloo/HonoursM26/Dataset-Documentation.md#L102-L105) and `*_task-EEG_channels.tsv`: `low_cutoff: 0.5`, `high_cutoff: 30`).
- **Spectral Band Analysis**: In [`spectralBandSpecificConnectivity/delta.py#L140`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/delta.py#L140), a bandpass filter of 0.5–4.0 Hz was explicitly applied for the delta band:
  ```python
  raw_clean.filter(l_freq=0.5, h_freq=4.0, fir_design='firwin', verbose=False)
  ```

### B. Continuous Signal vs. Window-Level Filtering
- **Filtering was performed on the continuous signal BEFORE window extraction**:
  Across all spectral scripts ([`alpha.py#L134-L148`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/alpha.py#L134-L148), [`beta.py#L138`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/beta.py#L138), [`theta.py#L138`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/theta.py#L138), [`delta.py#L140`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/delta.py#L140)), the full continuous `raw_clean` recording is filtered first, and segments are sliced out afterwards:
  ```python
  # 1. Bandpass filter the continuous recording
  raw_clean.filter(l_freq=8.0, h_freq=13.0, fir_design='firwin', verbose=False)

  # 2. Extract 2.56s windows from the filtered continuous signal
  segments = segment_data_from_annotations(raw_clean, annotations_df, label='eyes_closed')
  ```
  Filtering the continuous recording avoids edge filter transients and ringing artifacts that occur when filtering short 2.56-second segments independently.

---

## 3. Division by Zero Handling in the Min/Max Ratio

In all workflows, **`0 / 0` evaluates to `0.0`**:

### A. Binary Adjacency Matrices
In [`binary/binaryCorrelationJaccard.py#L260-L265`](file:///home/aloo/HonoursM26/binary/binaryCorrelationJaccard.py#L260-L265):
```python
mins = np.minimum(u, v)
maxs = np.maximum(u, v)

with np.errstate(divide='ignore', invalid='ignore'):
    j_score = np.where(maxs != 0, mins / maxs, 0)
```
- If an edge is absent in both subjects ($u_i = 0, v_i = 0$), `maxs == 0`, and `np.where` evaluates to **`0`**.
- It yields `1.0` if and only if both edges are present ($u_i = 1, v_i = 1$), and `0.0` otherwise.

### B. Zero Betweenness (and Centrality Metrics)
In [`weighted/edgeBetweennessJaccard.py#L225-L232`](file:///home/aloo/HonoursM26/weighted/edgeBetweennessJaccard.py#L225-L232) and [`weighted/nodeBetweennessJaccard.py#L206-L211`](file:///home/aloo/HonoursM26/weighted/nodeBetweennessJaccard.py#L206-L211):
```python
val_A = subject_segment_eb[subj_A][s].get(e, 0)
val_B = subject_segment_eb[subj_B][s].get(e, 0)

if max(val_A, val_B) > 0:
    jaccard_s = min(val_A, val_B) / max(val_A, val_B)
else:
    jaccard_s = 0
```
- When both subjects have zero centrality on an edge/node ($val_A = 0$ and $val_B = 0$), `max(val_A, val_B) == 0` evaluates to `False`, and `jaccard_s` is assigned **`0`**.

---

## 4. Binary Network GCC Thresholding

### A. Threshold Level
- **The GCC threshold is determined dynamically per subject and per window**:
  In [`binary/binaryCorrelationJaccard.py#L98-L102`](file:///home/aloo/HonoursM26/binary/binaryCorrelationJaccard.py#L98-L102):
  ```python
  for csv_file in csv_files[:n_segments]:
      df = pd.read_csv(csv_file, index_col=0)
      bin_df, thr = build_binary_network(df)
  ```
  Every individual segment matrix evaluates its own highest threshold starting at $1.0$ (stepping down by $0.05$) such that the giant connected component ratio is $1.0$ (all 19 nodes connected).

### B. Cut-off Comparison Operator
- **The cut-off is strictly “$\ge$” (greater than or equal to)**:
  In [`binary/binaryCorrelationJaccard.py#L58`](file:///home/aloo/HonoursM26/binary/binaryCorrelationJaccard.py#L58) and [`AditiMaamCode/edge-wise-jaccard-for-intrasession-between-subjects.py#L224`](file:///home/aloo/HonoursM26/AditiMaamCode/edge-wise-jaccard-for-intrasession-between-subjects.py#L224):
  ```python
  adj = (abs_corr >= thr).astype(float)
  ```

---

## 5. Distance Metric for Weighted Betweenness

- **The distance definition used is $1 / |r|$**:
  - In NetworkX, shortest path algorithms interpret edge weights as distances/costs. To map high correlation to short topological distance:
  - In [`weighted/edgeBetweennessJaccard.py#L85-L91`](file:///home/aloo/HonoursM26/weighted/edgeBetweennessJaccard.py#L85-L91):
    ```python
    for u, v, d in G.edges(data=True):
        weight = d.get('weight', 0)
        if weight > 0:
            d['distance'] = 1.0 / weight
        else:
            d['distance'] = np.inf

    eb = nx.edge_betweenness_centrality(G, weight='distance')
    ```
  - In [`weighted/nodeBetweennessJaccard.py#L84-L90`](file:///home/aloo/HonoursM26/weighted/nodeBetweennessJaccard.py#L84-L90):
    ```python
    nb = nx.betweenness_centrality(G, weight='distance')
    ```
  - In [`segmentWIseChange_refactored.py#L96-L102`](file:///home/aloo/HonoursM26/segmentWIseChange_refactored.py#L96-L102):
    ```python
    def build_graph_with_distance(df: pd.DataFrame) -> nx.Graph:
        G = nx.from_pandas_adjacency(df)
        for _, _, d in G.edges(data=True):
            weight = d.get("weight", 0)
            d["distance"] = 1.0 / weight if weight > 0 else np.inf
        return G
    ```

---

## 6. Edge-Strength Formula & Line Graph Weights

### A. Exact Edge-Strength Formula
In [`weighted/edgeStrengthJaccard.py#L85-L92`](file:///home/aloo/HonoursM26/weighted/edgeStrengthJaccard.py#L85-L92) and [`segmentWIseChange_refactored.py#L125-L135`](file:///home/aloo/HonoursM26/segmentWIseChange_refactored.py#L125-L135):
$$\text{Edge Strength}(u, v) = \left( \frac{s_u + s_v - 2 w_{uv}}{|s_u - s_v| + 1} \right) \cdot w_{uv}$$

Where:
- $w_{uv}$ is the edge weight between $u$ and $v$ (absolute Pearson correlation).
- $s_u = \sum_{k \in N(u)} w_{uk}$ is the node strength (weighted degree) of node $u$.
- $s_v = \sum_{k \in N(v)} w_{vk}$ is the node strength (weighted degree) of node $v$.

Code implementation:
```python
wk = d.get('weight', 0)
cvs_u = G.degree(u, weight='weight')
cvs_v = G.degree(v, weight='weight')
es[(u, v)] = ((cvs_u + cvs_v - 2 * wk) / (abs(cvs_u - cvs_v) + 1)) * wk
```

### B. Line Graph Weights for Edge Eigenvector Centrality
In [`weighted/edgeEigenvectorCentralityJaccard.py#L83-L93`](file:///home/aloo/HonoursM26/weighted/edgeEigenvectorCentralityJaccard.py#L83-L93) and [`segmentWIseChange_refactored.py#L145-L157`](file:///home/aloo/HonoursM26/segmentWIseChange_refactored.py#L145-L157) (adapted from Bröhl & Lehnertz, *Scientific Reports* 2022, [`Brohl-Lehnertz_SciRep2022.pdf`](file:///home/aloo/HonoursM26/Brohl-Lehnertz_SciRep2022.pdf)):
- Adjacent edges $e_1 = (u, v)$ and $e_2 = (v, w)$ sharing node $v$ form an edge in the line graph $L(G)$.
- The weight assigned to this adjacency in $L(G)$ is the **arithmetic mean (average)** of the two constituent correlation weights:
  $$W_{L}(e_1, e_2) = \frac{w(e_1) + w(e_2)}{2} = \frac{w_{uv} + w_{vw}}{2}$$

Code implementation:
```python
L = nx.line_graph(G)
for e in L.edges():
    edge1, edge2 = e
    w1 = G[edge1[0]][edge1[1]].get(weight, 0)
    w2 = G[edge2[0]][edge2[1]].get(weight, 0)
    L[edge1][edge2]['weight'] = (w1 + w2) / 2.0

ec = nx.eigenvector_centrality(L, weight='weight', max_iter=1000)
```

---

## 7. Spectral Band Analysis Scope

- **Yes, the spectral band analysis in `spectralBandSpecificConnectivity/` is weighted-correlation only**:
  In [`alpha.py`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/alpha.py), [`beta.py`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/beta.py), [`delta.py`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/delta.py), and [`theta.py`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/theta.py):
  1. The code computes the Pearson correlation matrix for each segment of the bandpass-filtered signal: `corr_matrix = np.corrcoef(segment)`.
  2. The absolute value is taken: `abs_matrix = np.abs(corr_matrix)` ([`alpha.py#L152-L154`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/alpha.py#L152-L154)).
  3. The edge-wise Jaccard similarity across subject pairs is calculated directly on these **continuous absolute correlation weights**:
     $$\text{Jaccard}(u, v) = \frac{\min(|r_A|, |r_B|)}{\max(|r_A|, |r_B|)}$$
     ([`alpha.py#L220-L226`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/alpha.py#L220-L226)).
  4. Regional connection percentages ($> 0.30, > 0.50, > 0.75, > 0.90$) are computed directly from the correlation matrices ([`alpha.py#L339-L352`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/alpha.py#L339-L352), [`results_summary.txt`](file:///home/aloo/HonoursM26/spectralBandSpecificConnectivity/results_summary.txt)).
  - **No graph centrality metrics** (node/edge betweenness, eigenvector, or strength) and **no GCC binarization** are performed in the `spectralBandSpecificConnectivity/` workflows. Those metrics exist exclusively for the broad-band data in `weighted/` and `binary/`.
