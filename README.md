# Brain Functional Connectivity Analysis

This repository contains Jupyter notebooks and workflows for analyzing functional brain networks derived from EEG recordings across subjects and age groups.

---

## Workflow Overview

1. **Matrix Generation**: Extract absolute segment-wise correlation matrices from EEG timeseries ([`pearson_correlation_calculation.ipynb`](pearson_correlation_calculation.ipynb)).
2. **Graph Construction**: Build undirected graphs for each segment (Nodes = EEG Sensors, Edge Weights = Correlation / Binarized Connectivity).
3. **Centrality Computation**: Compute topological metrics for all segments across subjects:
   - **Node Betweenness**: [`weighted/nodeBetweennessJaccard.ipynb`](weighted/nodeBetweennessJaccard.ipynb)
   - **Edge Betweenness**: [`weighted/edgeBetweennessJaccard.ipynb`](weighted/edgeBetweennessJaccard.ipynb)
   - **Node Closeness**: [`weighted/nodeClosenessJaccard.ipynb`](weighted/nodeClosenessJaccard.ipynb)
   - **Edge Closeness**: [`weighted/edgeClosenessJaccard.ipynb`](weighted/edgeClosenessJaccard.ipynb)
   - **Node Eigenvector**: [`weighted/nodeEigenvectorCentralityJaccard.ipynb`](weighted/nodeEigenvectorCentralityJaccard.ipynb)
   - **Edge Eigenvector**: [`weighted/edgeEigenvectorCentralityJaccard.ipynb`](weighted/edgeEigenvectorCentralityJaccard.ipynb)
   - **Node Strength**: [`weighted/nodeStrengthJaccard.ipynb`](weighted/nodeStrengthJaccard.ipynb)
   - **Edge Strength / Nearest-Neighbor**: [`weighted/edgeStrengthJaccard.ipynb`](weighted/edgeStrengthJaccard.ipynb)
   *(Also available in [`binary/`](binary/) and [`spectralBandSpecificConnectivity/`](spectralBandSpecificConnectivity/))*.
4. **Similarity Analysis**: Calculate the Weighted / Binary Jaccard Index across subject pairs to compare their segment-wise centrality distributions ([`weighted/correlationJaccard.ipynb`](weighted/correlationJaccard.ipynb)).
5. **Visualization**: Plot Seaborn heatmaps of Jaccard indices, sorting subject pairs by their age differences ([`age_acq_time_sorted.csv`](age_acq_time_sorted.csv)) to analyze age-related topological changes.

---

## Quick Setup (`uv`)

### 1. Install `uv` (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Environment & Install Dependencies
```bash
cd HonoursM26
uv sync
```

### 3. Register Jupyter Kernel
```bash
uv run python -m ipykernel install --user --name honoursm26 --display-name "Python (HonoursM26 uv)"
```

---

## Usage

- **Launch JupyterLab**:
  ```bash
  uv run jupyter lab
  ```
- **Select Kernel**: Open any `.ipynb` notebook and choose the **Python (HonoursM26 uv)** kernel.
