# Brain Functional Connectivity Analysis

This repository contains notebooks for analyzing functional brain networks derived from EEG recordings. 

## Workflow

1. **Matrix Generation**: Extract absolute segment-wise correlation matrices from EEG timeseries (`pearson_correlation_calculation.ipynb`).
2. **Graph Construction**: Build undirected weighted graphs for each segment (Nodes = Sensors, Edge Weights = Absolute Correlation).
3. **Centrality Computation**: Compute topological metrics for all segments across subjects:
   - Node Betweenness (`correlationNodeBetweennessJaccard.ipynb`)
   - Edge Betweenness (`correlationEdgeBetweennessJaccard.ipynb`)
   - Node Eigenvector (`correlationNodeEigenvectorCentralityJaccard.ipynb`)
   - Node Strength (`nodeStrengthJaccard.ipynb`)
   - Edge Strength / Nearest-Neighbor (`edgeStrengthJaccard.ipynb`)
4. **Similarity Analysis**: Calculate the Weighted Jaccard Index between all subject pairs to compare their segment-wise centrality distributions (`correlationJaccard.ipynb`).
5. **Visualization**: Plot Seaborne heatmaps of Jaccard indices, sorting subject pairs by their age differences (`age_acq_time_sorted.csv`) to observe age-related topological changes.
