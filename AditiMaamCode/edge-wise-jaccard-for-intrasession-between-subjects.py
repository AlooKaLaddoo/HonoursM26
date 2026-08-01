# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:02:31 2026
@author: Aditi Bose
"""

# -*- coding: utf-8 -*-
"""
EEG Infant Sleep - Jaccard Edge Stability Pipeline (Between-Subjects)
======================================================================
Steps:
  1. Load EDF, retain 19 canonical channels.
  2. Use expert-annotated .tsv windows (onset, 2.56 s, 200 Hz → 512 samples).
  3. BINARIZE=True : Build binary adjacency matrix per window via adaptive
                     GCC thresholding on |Pearson| correlation.
     BINARIZE=False: Build weighted adjacency matrix per window using
                     |Pearson| correlation (no thresholding).
  4. For each subject, compute mean edge matrix across all windows.
  5. Compute pairwise Jaccard similarity matrices for all subject pairs.
  6. Plot heatmap: rows = subject pairs, columns = 171 edges grouped by
     anatomical region pairs (FF, FC, … OO).
"""

import re
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────
# 0.  USER INPUTS  –  fill these in
# ─────────────────────────────────────────────

# List every EDF file path (strings or Path objects).
EDF_FILES = [
#session-1
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00064_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00065_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00069_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00070_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00075_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00077_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00078_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00079_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00080_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00081_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00083_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00084_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00085_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00086_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00087_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00089_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00090_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00094_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00096_ses-1_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/1session/sub-NORB00103_ses-1_task-EEG_eeg.edf",
#session-2
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00064_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00065_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00069_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00070_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00075_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00077_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00078_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00079_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00080_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00081_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00083_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00084_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00085_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00086_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00087_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00089_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00090_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00094_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00096_ses-2_task-EEG_eeg.edf",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/dataset-for-filepaths/2sessions/sub-NORB00103_ses-2_task-EEG_eeg.edf",
]

# Corresponding TSV files IN THE SAME ORDER as EDF_FILES.
TSV_FILES = [
#session-1
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00064_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00065_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00069_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00070_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00075_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00077_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00078_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00079_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00080_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00081_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00083_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00084_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00085_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00086_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00087_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00089_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00090_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00094_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00096_ses-1_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/1session/sub-NORB00103_ses-1_task-EEG_annotations.tsv",
#session-2
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00064_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00065_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00069_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00070_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00075_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00077_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00078_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00079_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00080_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00081_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00083_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00084_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00085_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00086_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00087_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00089_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00090_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00094_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00096_ses-2_task-EEG_annotations.tsv",
"C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/annotations-for-filepaths/2sessions/sub-NORB00103_ses-2_task-EEG_annotations.tsv",
]

BINARIZE = True   # True → binary (GCC-adaptive), False → weighted (|Pearson|)

THRESHOLD_INIT = 1.0
THRESHOLD_STEP = 0.05
THRESHOLD_MIN  = 0.0

OUTPUT_DIR = Path("C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/plots/pearson-based_intrasession-subject-pair-jaccard_{}_thresh={}".format(
    "binary" if BINARIZE else "weighted",
    THRESHOLD_STEP if BINARIZE else "NA"
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_SUBJECTS = [("NORB00064", 1), ("NORB00065", 1)]  # (subject_id, session_id) pairs to save
DEBUG_SUBJECT = ("NORB00084", 1)  # set to None to disable debug prints

PER_SEGMENT_DIR = Path("C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/output/per_segment_matrices")
MEAN_MATRIX_DIR = Path("C:/Users/Aditi Bose/OneDrive - Research.iiit.ac.in - IIIT Hyderabad/IIITH-PhD/Research-works/EEG-infant/output/mean_matrices")

# ─────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────

TARGET_FS    = 200
WIN_DURATION = 2.56
WIN_SAMPLES  = int(WIN_DURATION * TARGET_FS)   # 512

KEEP_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1',  'O2',  'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ',  'CZ',  'PZ'
]
N_CH = len(KEEP_CHANNELS)   # 19

REGION_MAP = {
    'Fp1':'F','Fp2':'F','F3':'F','F4':'F','F7':'F','F8':'F','FZ':'F',
    'C3':'C','C4':'C','CZ':'C',
    'T3':'T','T4':'T','T5':'T','T6':'T',
    'P3':'P','P4':'P','PZ':'P',
    'O1':'O','O2':'O',
}
REGION_ORDER = ['F','C','T','P','O']

def _build_edge_info(channels, region_map, region_order):
    ch_to_reg  = {c: region_map[c] for c in channels}
    region_rank = {r: idx for idx, r in enumerate(region_order)}
    group_order = []
    for r1, r2 in itertools.combinations_with_replacement(region_order, 2):
        group_order.append(r1 + r2)

    edges, edge_groups = [], []
    for grp in group_order:
        r1, r2 = grp[0], grp[1]
        if r1 == r2:
            pairs = list(itertools.combinations(range(len(channels)), 2))
            for i, j in pairs:
                if ch_to_reg[channels[i]] == r1 and ch_to_reg[channels[j]] == r1:
                    edges.append((i, j))
                    edge_groups.append(grp)
        else:
            for i, chi in enumerate(channels):
                for j, chj in enumerate(channels):
                    if i >= j:
                        continue
                    ri, rj = ch_to_reg[chi], ch_to_reg[chj]
                    if region_rank[ri] > region_rank[rj]:
                        ri, rj = rj, ri
                    if ri + rj == grp:
                        edges.append((i, j))
                        edge_groups.append(grp)
    return edges, edge_groups, group_order

EDGES, EDGE_GROUPS, GROUP_ORDER = _build_edge_info(
    KEEP_CHANNELS, REGION_MAP, REGION_ORDER
)
N_EDGES = len(EDGES)

# ─────────────────────────────────────────────
# 2.  HELPER: GCC-ADAPTIVE BINARY NETWORK
# ─────────────────────────────────────────────

def gcc_ratio(adj):
    G = nx.from_numpy_array(adj)
    if G.number_of_nodes() == 0:
        return 0.0
    comps = list(nx.connected_components(G))
    return max(len(c) for c in comps) / G.number_of_nodes()

def build_binary_network(data_window):
    corr = np.corrcoef(data_window)
    np.fill_diagonal(corr, 0)
    abs_corr = np.abs(corr)

    thr = THRESHOLD_INIT
    while thr >= THRESHOLD_MIN:
        adj = (abs_corr >= thr).astype(float)
        np.fill_diagonal(adj, 0)
        if gcc_ratio(adj) == 1.0:
            return adj, round(thr, 10)
        thr -= THRESHOLD_STEP
    return None, None

# ─────────────────────────────────────────────
# 3.  LOAD DATA  →  networks per subject
# ─────────────────────────────────────────────

def parse_subject_session(filepath):
    name = Path(filepath).name
    sub  = re.search(r'sub-(\w+?)_', name)
    ses  = re.search(r'ses-(\d+)',   name)
    if not sub or not ses:
        raise ValueError(f"Cannot parse subject/session from: {name}")
    return sub.group(1), int(ses.group(1))

def load_subject_networks(edf_path, tsv_path, verbose=False):
    """
    Load one EDF + TSV pair.
    Returns list of (19×19) numpy arrays, one per valid window.
    Binary or weighted depending on BINARIZE flag.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=verbose)
    raw.rename_channels({ch: ch.strip() for ch in raw.ch_names})

    ch_map    = {ch.upper(): ch for ch in raw.ch_names}
    available = [ch_map[k.upper()] for k in KEEP_CHANNELS if k.upper() in ch_map]
    missing   = [k for k in KEEP_CHANNELS if k.upper() not in ch_map]

    if missing:
        print(f"  [WARN] Channels not found and dropped: {missing}")

    raw.pick(available)
    canonical_available = [c for c in KEEP_CHANNELS if c.upper() in ch_map]
    raw.reorder_channels(canonical_available)
    
    
    # ── DEBUG ──
    if DEBUG_SUBJECT is not None:
        sub_id_check, ses_id_check = DEBUG_SUBJECT
        current_path = Path(edf_path)
        if f"sub-{sub_id_check}" in current_path.name and f"ses-{ses_id_check}" in current_path.name:
            print(f"\n  [DEBUG] Channel order after reordering: {raw.ch_names}")
    # ── END DEBUG ──

    data = raw.get_data()   # (n_ch, n_times)

    tsv_df = pd.read_csv(tsv_path, sep='\t', index_col=False)
    tsv_df.columns = [c.strip().lower() for c in tsv_df.columns]
    print(f"  [DEBUG] TSV first 3 rows:\n{tsv_df.head(3)}")
    print(f"  [DEBUG] TSV shape: {tsv_df.shape}")


    networks       = []
    thresholds_used = []
    skipped        = 0

    win_idx = 0
    for _, row in tsv_df.iterrows():
        win_idx += 1
        onset_sec    = float(row['onset'])
        start_sample = int(round(onset_sec * TARGET_FS))
        end_sample   = start_sample + WIN_SAMPLES

        if end_sample > data.shape[1]:
            print(f"  [WARN] Window onset={onset_sec:.3f}s: end_sample={end_sample} "
                  f"exceeds recording length={data.shape[1]}. Skipping.")
            skipped += 1
            continue

        window = data[:, start_sample:end_sample]   # (n_ch, 512)
        
        # ── DEBUG ──
        if DEBUG_SUBJECT is not None:
            sub_id_check, ses_id_check = DEBUG_SUBJECT
            current_path = Path(edf_path)
            if f"sub-{sub_id_check}" in current_path.name and f"ses-{ses_id_check}" in current_path.name:
                if win_idx == 2:  # only first window
                    print(f"  [DEBUG] Window 1 → onset={onset_sec:.3f}s  "
                          f"start_sample={start_sample}  "
                          f"end_sample={end_sample}  "
                          f"window.shape={window.shape}")
                    print(f"  [DEBUG] Window 1 time series (first 5 samples, all 19 channels):")
                    print(window[:, :5])
                    print(f"  [DEBUG] Window 1 time series (last 5 samples, all 19 channels):")
                    print(window[:, -5:])
        # ── END DEBUG ──

        if BINARIZE:
            adj, thr = build_binary_network(window)
            if adj is None:
                print(f"  [WARN] Window onset={onset_sec:.3f}s: GCC never reached 1. Skipping.")
                skipped += 1
                continue
            networks.append(adj)
            thresholds_used.append(thr)
        else:
            corr = np.corrcoef(window)
            np.fill_diagonal(corr, 0)
            networks.append(np.abs(corr))

    print(f"  Windows used: {len(networks)}  |  Skipped: {skipped}")
    if BINARIZE:
        print(f"  Threshold stats → min:{min(thresholds_used):.1f}  "
              f"max:{max(thresholds_used):.1f}  "
              f"mean:{np.mean(thresholds_used):.2f}")

    return networks

# ─────────────────────────────────────────────
# 4.  JACCARD SIMILARITY (edge-centric)
# ─────────────────────────────────────────────

def edge_stability_jaccard(subject_a, subject_b):
    """
    subject_a, subject_b : lists of (19×19) arrays for subject A and subject B.
    Computes mean matrix across windows for each subject, then
    applies Jaccard: min(a,b) / max(a,b) per edge.
    
    subject_a and subject_b are lists of 19×19 numpy arrays — one array per window 
    for that subject. sum(...) adds them elementwise (numpy supports this), and 
    dividing by len(...) gives the elementwise mean. So p_a and p_b are each a 
    single 19×19 matrix where entry [i,j] is the average value of that edge 
    across all windows for that subject.

    """
    p_a = sum(subject_a) / len(subject_a)   # mean edge matrix, subject A
    p_b = sum(subject_b) / len(subject_b)   # mean edge matrix, subject B

    jaccard_matrix = np.zeros((N_CH, N_CH))
    for i in range(N_CH):
        for j in range(i + 1, N_CH):
            a, b = p_a[i, j], p_b[i, j]
            jac  = min(a, b) / max(a, b) if max(a, b) > 0 else 0.0
            jaccard_matrix[i, j] = jac
            jaccard_matrix[j, i] = jac
    return jaccard_matrix

def jaccard_to_edge_vector(jac_matrix):
    return np.array([jac_matrix[i, j] for i, j in EDGES])

def save_matrices_for_subject(edf_path, tsv_path, sub_id, ses_id):
    """
    Saves raw Pearson matrices per segment and the mean matrix
    for a given subject+session into their respective directories.
    """
    folder_name = f"{sub_id}_ses{ses_id}"
    seg_out_dir  = PER_SEGMENT_DIR / folder_name
    mean_out_dir = MEAN_MATRIX_DIR / folder_name
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    mean_out_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels({ch: ch.strip() for ch in raw.ch_names})
    ch_map    = {ch.upper(): ch for ch in raw.ch_names}
    available = [ch_map[k.upper()] for k in KEEP_CHANNELS if k.upper() in ch_map]
    raw.pick(available)
    canonical_available = [c for c in KEEP_CHANNELS if c.upper() in ch_map]
    raw.reorder_channels(canonical_available)
    data = raw.get_data()

    tsv_df = pd.read_csv(tsv_path, sep='\t', index_col=False)
    tsv_df.columns = [c.strip().lower() for c in tsv_df.columns]

    pearson_matrices = []
    seg_idx = 1
    
    for _, row in tsv_df.iterrows():
        onset_sec    = float(row['onset'])
        start_sample = int(round(onset_sec * TARGET_FS))
        end_sample   = start_sample + WIN_SAMPLES

        if end_sample > data.shape[1]:
            continue

        window = data[:, start_sample:end_sample]
        corr   = np.corrcoef(window)                  # raw Pearson (with negatives)
        np.fill_diagonal(corr, 0)

        df_corr = pd.DataFrame(corr, index=canonical_available, columns=canonical_available)
        fname   = f"seg_{seg_idx:03d}.csv"
        df_corr.to_csv(seg_out_dir / fname)
        pearson_matrices.append(corr)
        seg_idx += 1

    # mean of raw Pearson matrices across segments
    mean_mat = np.mean(np.stack([np.abs(m) for m in pearson_matrices], axis=0), axis=0)
    df_mean  = pd.DataFrame(mean_mat, index=canonical_available, columns=canonical_available)
    df_mean.to_csv(mean_out_dir / "mean_matrix.csv")

    print(f"  Saved {seg_idx - 1} segment CSVs → {seg_out_dir}")
    print(f"  Saved mean matrix CSV  → {mean_out_dir}")

# ─────────────────────────────────────────────
# 5.  PLOTTING
# ─────────────────────────────────────────────

def _group_boundaries(edge_groups):
    boundaries = []
    prev_grp, start = edge_groups[0], 0
    for idx, grp in enumerate(edge_groups):
        if grp != prev_grp:
            boundaries.append((start, idx - 1, prev_grp))
            start = idx
            prev_grp = grp
    boundaries.append((start, len(edge_groups) - 1, prev_grp))
    return boundaries

def plot_subject_pairs(subject_networks, ses_id):
    """
    subject_networks : dict {subject_id: list of (19×19) arrays}
    Rows = subject pairs, Columns = 171 edges.
    """
    subject_ids = sorted(subject_networks.keys())
    pairs       = list(itertools.combinations(subject_ids, 2))
    n_pairs     = len(pairs)

    data_matrix = np.zeros((n_pairs, N_EDGES))
    y_labels    = []

    print(f"\n  Computing Jaccard for {n_pairs} subject pairs...")
    for p_idx, (sid_a, sid_b) in enumerate(tqdm(pairs, desc="Subject pairs", unit="pair")):
        jac_mat          = edge_stability_jaccard(
                               subject_networks[sid_a],
                               subject_networks[sid_b]
                           )
        data_matrix[p_idx] = jaccard_to_edge_vector(jac_mat)

        # Strip leading zeros: "NORB00001" → "1", "NORB00020" → "20"
        label_a = str(int(re.search(r'\d+', sid_a).group()))
        label_b = str(int(re.search(r'\d+', sid_b).group()))
        y_labels.append(f"{label_a}_vs_{label_b}")

    fig_h = max(6, n_pairs * 0.35 + 2)
    fig, ax = plt.subplots(figsize=(max(18, N_EDGES // 8), fig_h))
    im = ax.imshow(data_matrix, aspect='auto', cmap='hot', vmin=0, vmax=1,
                   origin='lower')

    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_ylabel('Subject pair', fontsize=20)

    boundaries = _group_boundaries(EDGE_GROUPS)
    xtick_pos  = [(s + e) / 2 for s, e, _ in boundaries]
    xtick_lbl  = [grp for _, _, grp in boundaries]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=15, rotation=90, ha='center', va='top')
    ax.set_xlabel('Edge group (anatomical)', fontsize=20)

    for s, e, grp in boundaries:
        ax.axvline(s - 0.5, color='red', linewidth=1, alpha=1)

    plt.colorbar(im, ax=ax, label='Jaccard similarity', shrink=0.8)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"ses-{ses_id}_subject_pairs_jaccard.png"
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out_path}")

# ─────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    assert len(EDF_FILES) == len(TSV_FILES), \
        "EDF_FILES and TSV_FILES must have the same length."

    print("\n══════════════════════════════════════════")
    print("  STEP 1/2 : Loading EDF files & building networks")
    print("══════════════════════════════════════════\n")

    # To session-keyed dicts:
    session_networks = defaultdict(dict)  # {ses_id: {sub_id: [arrays]}}

    for edf_path, tsv_path in tqdm(zip(EDF_FILES, TSV_FILES),
                                   total=len(EDF_FILES),
                                   desc="Subjects", unit="subject"):
        sub_id, ses_id = parse_subject_session(edf_path)
        print(f"\n▶ Subject {sub_id}  Session {ses_id}")
        print(f"  EDF : {edf_path}")
        print(f"  TSV : {tsv_path}")

        nets = load_subject_networks(edf_path, tsv_path)

        if sub_id in session_networks[ses_id]:
            # safety check: should not happen if input is truly one session only
            print(f"  [WARN] Subject {sub_id} Ses {ses_id} already loaded. Appending.")
            session_networks[ses_id][sub_id].extend(nets)
        else:
            session_networks[ses_id][sub_id] = nets

    for ses_id in sorted(session_networks.keys()):
        print(f"✔ Session {ses_id} subjects loaded: {sorted(session_networks[ses_id].keys())}")

    print("\n══════════════════════════════════════════")
    print("  OPTIONAL : Saving raw Pearson matrices for selected subjects")
    print("══════════════════════════════════════════\n")

    save_lookup = {(sub, ses) for sub, ses in SAVE_SUBJECTS}
    for edf_path, tsv_path in zip(EDF_FILES, TSV_FILES):
        sub_id, ses_id = parse_subject_session(edf_path)
        if (sub_id, ses_id) in save_lookup:
            print(f"▶ Saving matrices for Subject {sub_id}  Session {ses_id}")
            save_matrices_for_subject(edf_path, tsv_path, sub_id, ses_id)        

    print("\n══════════════════════════════════════════")
    print("  STEP 2/2 : Computing Jaccard & plotting")
    print("══════════════════════════════════════════\n")

    for ses_id in sorted(session_networks.keys()):
        print(f"\n── Session {ses_id} ──")
        plot_subject_pairs(session_networks[ses_id],  ses_id)


    print("\n✔ All plots saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()