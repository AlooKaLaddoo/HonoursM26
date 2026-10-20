"""
Configuration and metadata constants for the Infant EEG Longitudinal Analysis Pipeline.
Adapted for N = 19 longitudinal subjects (ses-1 and ses-2, K = 12 epochs each).
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Infants_data"
DERIVATIVES_PATH = DATASET_PATH / "derivatives" / "NeuronicEEG"
CORR_MATRICES_DIR = BASE_DIR / "correlation_matrices_segment_wise_absolute"
OUTPUT_TENSORS_DIR = BASE_DIR / "processed_tensors"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"

# Cohort specification: 19 infants with >= 12 epochs in both ses-1 and ses-2
SUBJECT_IDS = [
    "sub-NORB00064", "sub-NORB00065", "sub-NORB00069", "sub-NORB00070",
    "sub-NORB00075", "sub-NORB00077", "sub-NORB00078", "sub-NORB00079",
    "sub-NORB00080", "sub-NORB00081", "sub-NORB00083", "sub-NORB00084",
    "sub-NORB00086", "sub-NORB00087", "sub-NORB00089", "sub-NORB00090",
    "sub-NORB00094", "sub-NORB00096", "sub-NORB00103"
]

SESSIONS = ["ses-1", "ses-2"]
N_EPOCHS = 12

# 19 Electrodes in standard recording order
CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"
]

# Standard element order: 19 Nodes grouped by region (F, C, T, P, O)
NODE_LABELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "FZ",
    "C3", "C4", "CZ",
    "T3", "T4", "T5", "T6",
    "P3", "P4", "PZ",
    "O1", "O2"
]

NODE_REGIONS = {
    "Fp1": "F", "Fp2": "F", "F3": "F", "F4": "F", "F7": "F", "F8": "F", "FZ": "F",
    "C3": "C", "C4": "C", "CZ": "C",
    "T3": "T", "T4": "T", "T5": "T", "T6": "T",
    "P3": "P", "P4": "P", "PZ": "P",
    "O1": "O", "O2": "O"
}

# 171 Edges ordered by region pair with increasing distance between regions
EDGE_LABELS = [
    'Fp1-Fp2', 'Fp1-F3', 'Fp1-F4', 'Fp1-F7', 'Fp1-F8', 'Fp1-FZ', 'Fp2-F3',
    'Fp2-F4', 'Fp2-F7', 'Fp2-F8', 'Fp2-FZ', 'F3-F4', 'F3-F7', 'F3-F8', 'F3-FZ',
    'F4-F7', 'F4-F8', 'F4-FZ', 'F7-F8', 'F7-FZ', 'F8-FZ', 'Fp1-C3', 'Fp1-C4',
    'Fp1-CZ', 'Fp2-C3', 'Fp2-C4', 'Fp2-CZ', 'F3-C3', 'F3-C4', 'F3-CZ', 'F4-C3',
    'F4-C4', 'F4-CZ', 'F7-C3', 'F7-C4', 'F7-CZ', 'F8-C3', 'F8-C4', 'F8-CZ',
    'FZ-C3', 'FZ-C4', 'FZ-CZ', 'Fp1-T3', 'Fp1-T4', 'Fp1-T5', 'Fp1-T6',
    'Fp2-T3', 'Fp2-T4', 'Fp2-T5', 'Fp2-T6', 'F3-T3', 'F3-T4', 'F3-T5', 'F3-T6',
    'F4-T3', 'F4-T4', 'F4-T5', 'F4-T6', 'F7-T3', 'F7-T4', 'F7-T5', 'F7-T6',
    'F8-T3', 'F8-T4', 'F8-T5', 'F8-T6', 'FZ-T3', 'FZ-T4', 'FZ-T5', 'FZ-T6',
    'Fp1-P3', 'Fp1-P4', 'Fp1-PZ', 'Fp2-P3', 'Fp2-P4', 'Fp2-PZ', 'F3-P3',
    'F3-P4', 'F3-PZ', 'F4-P3', 'F4-P4', 'F4-PZ', 'F7-P3', 'F7-P4', 'F7-PZ',
    'F8-P3', 'F8-P4', 'F8-PZ', 'FZ-P3', 'FZ-P4', 'FZ-PZ', 'Fp1-O1', 'Fp1-O2',
    'Fp2-O1', 'Fp2-O2', 'F3-O1', 'F3-O2', 'F4-O1', 'F4-O2', 'F7-O1', 'F7-O2',
    'F8-O1', 'F8-O2', 'FZ-O1', 'FZ-O2', 'C3-C4', 'C3-CZ', 'C4-CZ', 'C3-T3',
    'C3-T4', 'C3-T5', 'C3-T6', 'C4-T3', 'C4-T4', 'C4-T5', 'C4-T6', 'CZ-T3',
    'CZ-T4', 'CZ-T5', 'CZ-T6', 'C3-P3', 'C3-P4', 'C3-PZ', 'C4-P3', 'C4-P4',
    'C4-PZ', 'CZ-P3', 'CZ-P4', 'CZ-PZ', 'C3-O1', 'C3-O2', 'C4-O1', 'C4-O2',
    'CZ-O1', 'CZ-O2', 'T3-T4', 'T3-T5', 'T3-T6', 'T4-T5', 'T4-T6', 'T5-T6',
    'T3-P3', 'T3-P4', 'T3-PZ', 'T4-P3', 'T4-P4', 'T4-PZ', 'T5-P3', 'T5-P4',
    'T5-PZ', 'T6-P3', 'T6-P4', 'T6-PZ', 'T3-O1', 'T3-O2', 'T4-O1', 'T4-O2',
    'T5-O1', 'T5-O2', 'T6-O1', 'T6-O2', 'P3-P4', 'P3-PZ', 'P4-PZ', 'P3-O1',
    'P3-O2', 'P4-O1', 'P4-O2', 'PZ-O1', 'PZ-O2', 'O1-O2',
]

REGION_PAIRS = [
    "F-F", "F-C", "F-T", "F-P", "F-O",
    "C-C", "C-T", "C-P", "C-O",
    "T-T", "T-P", "T-O",
    "P-P", "P-O",
    "O-O"
]

# Frequency band frequency limits
FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0)
}

# The 26 Feature Definitions
FEATURE_DEFS = {
    "f01": {"name": "broadband_weighted_correlation", "type": "edge", "dim": 171, "network": "weighted", "band": "broadband"},
    "f02": {"name": "broadband_weighted_node_betweenness", "type": "node", "dim": 19, "network": "weighted", "band": "broadband"},
    "f03": {"name": "broadband_weighted_edge_betweenness", "type": "edge", "dim": 171, "network": "weighted", "band": "broadband"},
    "f04": {"name": "broadband_weighted_node_strength", "type": "node", "dim": 19, "network": "weighted", "band": "broadband"},
    "f05": {"name": "broadband_weighted_edge_strength", "type": "edge", "dim": 171, "network": "weighted", "band": "broadband"},
    "f06": {"name": "broadband_weighted_node_eigenvector", "type": "node", "dim": 19, "network": "weighted", "band": "broadband"},
    "f07": {"name": "broadband_weighted_edge_eigenvector", "type": "edge", "dim": 171, "network": "weighted", "band": "broadband"},
    "f08": {"name": "broadband_weighted_node_closeness", "type": "node", "dim": 19, "network": "weighted", "band": "broadband"},
    "f09": {"name": "broadband_weighted_edge_closeness", "type": "edge", "dim": 171, "network": "weighted", "band": "broadband"},
    "f10": {"name": "broadband_binary_adjacency", "type": "edge", "dim": 171, "network": "binary", "band": "broadband"},
    "f11": {"name": "broadband_binary_node_betweenness", "type": "node", "dim": 19, "network": "binary", "band": "broadband"},
    "f12": {"name": "broadband_binary_edge_betweenness", "type": "edge", "dim": 171, "network": "binary", "band": "broadband"},
    "f13": {"name": "broadband_binary_node_degree", "type": "node", "dim": 19, "network": "binary", "band": "broadband"},
    "f14": {"name": "broadband_binary_edge_strength", "type": "edge", "dim": 171, "network": "binary", "band": "broadband"},
    "f15": {"name": "broadband_binary_node_eigenvector", "type": "node", "dim": 19, "network": "binary", "band": "broadband"},
    "f16": {"name": "broadband_binary_edge_eigenvector", "type": "edge", "dim": 171, "network": "binary", "band": "broadband"},
    "f17": {"name": "broadband_binary_node_closeness", "type": "node", "dim": 19, "network": "binary", "band": "broadband"},
    "f18": {"name": "broadband_binary_edge_closeness", "type": "edge", "dim": 171, "network": "binary", "band": "broadband"},
    "f19": {"name": "delta_weighted_correlation", "type": "edge", "dim": 171, "network": "weighted", "band": "delta"},
    "f20": {"name": "delta_binary_adjacency", "type": "edge", "dim": 171, "network": "binary", "band": "delta"},
    "f21": {"name": "theta_weighted_correlation", "type": "edge", "dim": 171, "network": "weighted", "band": "theta"},
    "f22": {"name": "theta_binary_adjacency", "type": "edge", "dim": 171, "network": "binary", "band": "theta"},
    "f23": {"name": "alpha_weighted_correlation", "type": "edge", "dim": 171, "network": "weighted", "band": "alpha"},
    "f24": {"name": "alpha_binary_adjacency", "type": "edge", "dim": 171, "network": "binary", "band": "alpha"},
    "f25": {"name": "beta_weighted_correlation", "type": "edge", "dim": 171, "network": "weighted", "band": "beta"},
    "f26": {"name": "beta_binary_adjacency", "type": "edge", "dim": 171, "network": "binary", "band": "beta"}
}

