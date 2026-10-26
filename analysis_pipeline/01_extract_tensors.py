"""
Phase 1: Feature Tensor Construction (Step 1)
Extracts all 26 feature sets for N = 19 infants across 2 sessions and K = 12 epochs.
Outputs:
  - processed_tensors/X_f01.npy ... X_f26.npy  (shape: 19 x 2 x 12 x n)
  - processed_tensors/metadata.json
  - processed_tensors/README.md
"""
import json
import re
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# Support both direct script execution and package imports
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from analysis_pipeline.config import (
        BASE_DIR,
        CHANNELS,
        CORR_MATRICES_DIR,
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        FREQUENCY_BANDS,
        N_EPOCHS,
        NODE_LABELS,
        OUTPUT_TENSORS_DIR,
        SESSIONS,
        SUBJECT_IDS,
    )
    from analysis_pipeline.utils import (
        compute_all_broadband_features,
        extract_band_matrices,
        extract_matrix_edge_vector,
        load_annotations,
    )
else:
    from .config import (
        BASE_DIR,
        CHANNELS,
        CORR_MATRICES_DIR,
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        FREQUENCY_BANDS,
        N_EPOCHS,
        NODE_LABELS,
        OUTPUT_TENSORS_DIR,
        SESSIONS,
        SUBJECT_IDS,
    )
    from .utils import (
        compute_all_broadband_features,
        extract_band_matrices,
        extract_matrix_edge_vector,
        load_annotations,
    )

def _sort_segment_key(path: Path) -> int:
    """Sort key extracting integer segment index for strict chronological ordering."""
    match = re.search(r"segment-(\d+)", path.name)
    return int(match.group(1)) if match else 0

def run_extraction():
    start_time = time.time()
    OUTPUT_TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    
    n_subs = len(SUBJECT_IDS)
    n_ses = len(SESSIONS)
    n_epochs = N_EPOCHS
    
    # Resolve correlation matrices directory (root or legacyCode fallback)
    corr_dir = CORR_MATRICES_DIR
    if not corr_dir.exists():
        legacy_dir = BASE_DIR / "legacyCode" / "correlation_matrices_segment_wise_absolute"
        if legacy_dir.exists():
            corr_dir = legacy_dir
        else:
            raise FileNotFoundError(
                f"Correlation matrices directory not found at '{CORR_MATRICES_DIR}' or '{legacy_dir}'"
            )

    print("=" * 70)
    print("PHASE 1: FEATURE TENSOR EXTRACTION (STEP 1)")
    print(f"Cohort: {n_subs} infants | Sessions: {n_ses} | Epochs: {n_epochs}")
    print(f"Correlation matrices source: {corr_dir}")
    print("=" * 70)
    
    # Initialize dictionary of tensors: shape (19, 2, 12, n)
    tensors = {}
    for fid, meta in FEATURE_DEFS.items():
        dim = meta["dim"]
        tensors[fid] = np.zeros((n_subs, n_ses, n_epochs, dim), dtype=np.float64)
        
    bookkeeping_tau = []
    
    # -------------------------------------------------------------
    # 1. EXTRACT BROADBAND FEATURES (f01 - f18)
    # -------------------------------------------------------------
    print("\n[1/2] Extracting Broadband Network Features (f01 - f18)...")
    broadband_fids = [fid for fid, meta in FEATURE_DEFS.items() if meta["band"] == "broadband"]
    
    for sub_idx, subj in enumerate(tqdm(SUBJECT_IDS, desc="Broadband subjects")):
        for ses_idx, ses in enumerate(SESSIONS):
            subj_ses_dir = corr_dir / subj / ses
            if not subj_ses_dir.exists():
                raise FileNotFoundError(f"Missing correlation matrices directory: {subj_ses_dir}")
                
            csv_files = sorted(subj_ses_dir.glob("*.csv"), key=_sort_segment_key)
            
            if len(csv_files) < n_epochs:
                raise ValueError(
                    f"Subject {subj} {ses} has only {len(csv_files)} correlation matrices (< {n_epochs}) in {subj_ses_dir}"
                )
                
            for ep_idx in range(n_epochs):
                csv_path = csv_files[ep_idx]
                df = pd.read_csv(csv_path, index_col=0)
                
                features = compute_all_broadband_features(df)
                
                # Assign to broadband tensors
                for fid in broadband_fids:
                    tensors[fid][sub_idx, ses_idx, ep_idx, :] = features[fid]
                    
                bookkeeping_tau.append({
                    "subject": subj, "session": ses, "epoch": ep_idx + 1,
                    "tau_c": features["tau_c"], "density": features["density"]
                })
                
    # -------------------------------------------------------------
    # 2. EXTRACT FREQUENCY BAND FEATURES (f19 - f26)
    # -------------------------------------------------------------
    print("\n[2/2] Extracting Frequency Band Specific Features (f19 - f26)...")
    
    band_feature_map = {
        "delta": ("f19", "f20"),
        "theta": ("f21", "f22"),
        "alpha": ("f23", "f24"),
        "beta":  ("f25", "f26")
    }
    
    bookkeeping_bands = {b: [] for b in band_feature_map}
    
    # Iterate subjects and sessions on the outer loop to load raw EDF only once per recording (38 loads instead of 152)
    for sub_idx, subj in enumerate(tqdm(SUBJECT_IDS, desc="Frequency band subjects")):
        for ses_idx, ses in enumerate(SESSIONS):
            edf_path = DATASET_PATH / subj / ses / "eeg" / f"{subj}_{ses}_task-EEG_eeg.edf"
            if not edf_path.exists():
                raise FileNotFoundError(f"Missing EDF: {edf_path}")
                
            ann_df = load_annotations(subj, ses)
            if ann_df is None:
                raise FileNotFoundError(f"Missing annotations for {subj} {ses}")
                
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.pick(CHANNELS)
            
            for band_name, (w_fid, b_fid) in band_feature_map.items():
                l_freq, h_freq = FREQUENCY_BANDS[band_name]
                
                band_results = extract_band_matrices(
                    subject_id=subj, session_id=ses,
                    l_freq=l_freq, h_freq=h_freq, n_epochs=n_epochs,
                    raw=raw, ann_df=ann_df
                )
                
                for ep_idx, (corr_df, bin_df, thr, density) in enumerate(band_results):
                    w_vec = extract_matrix_edge_vector(corr_df)
                    b_vec = extract_matrix_edge_vector(bin_df)
                    
                    tensors[w_fid][sub_idx, ses_idx, ep_idx, :] = w_vec
                    tensors[b_fid][sub_idx, ses_idx, ep_idx, :] = b_vec
                    
                    bookkeeping_bands[band_name].append({
                        "subject": subj, "session": ses, "epoch": ep_idx + 1,
                        "tau_c": thr, "density": density
                    })
                    
    # -------------------------------------------------------------
    # 3. VALIDATE AND SAVE TENSORS
    # -------------------------------------------------------------
    print("\n[3/3] Validating and saving all 26 feature tensors to disk...")
    for fid, tensor in tensors.items():
        expected_dim = FEATURE_DEFS[fid]["dim"]
        expected_shape = (n_subs, n_ses, n_epochs, expected_dim)
        if tensor.shape != expected_shape:
            raise ValueError(f"Tensor {fid} shape mismatch: expected {expected_shape}, got {tensor.shape}")
        if np.isnan(tensor).any():
            raise ValueError(f"Tensor {fid} contains NaN values!")
        if np.isinf(tensor).any():
            raise ValueError(f"Tensor {fid} contains infinite values!")
            
        out_file = OUTPUT_TENSORS_DIR / f"X_{fid}.npy"
        np.save(out_file, tensor)
        
    # Save metadata JSON
    meta_path = OUTPUT_TENSORS_DIR / "metadata.json"
    metadata = {
        "cohort": {
            "n_subjects": n_subs,
            "subjects": SUBJECT_IDS,
            "sessions": SESSIONS,
            "n_epochs": n_epochs
        },
        "elements": {
            "n_nodes": len(NODE_LABELS),
            "node_labels": NODE_LABELS,
            "n_edges": len(EDGE_LABELS),
            "edge_labels": EDGE_LABELS
        },
        "features": FEATURE_DEFS,
        "tau_density_summary": {
            "mean_tau": float(np.mean([x["tau_c"] for x in bookkeeping_tau])),
            "min_tau": float(np.min([x["tau_c"] for x in bookkeeping_tau])),
            "max_tau": float(np.max([x["tau_c"] for x in bookkeeping_tau])),
            "median_density": float(np.median([x["density"] for x in bookkeeping_tau])),
            "min_density": float(np.min([x["density"] for x in bookkeeping_tau])),
            "max_density": float(np.max([x["density"] for x in bookkeeping_tau]))
        },
        "frequency_band_summaries": {
            band: {
                "mean_tau": float(np.mean([x["tau_c"] for x in records])),
                "min_tau": float(np.min([x["tau_c"] for x in records])),
                "max_tau": float(np.max([x["tau_c"] for x in records])),
                "median_density": float(np.median([x["density"] for x in records])),
                "min_density": float(np.min([x["density"] for x in records])),
                "max_density": float(np.max([x["density"] for x in records]))
            }
            for band, records in bookkeeping_bands.items() if records
        }
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    # Save README
    readme_path = OUTPUT_TENSORS_DIR / "README.md"
    readme_content = """# Processed Feature Tensors ($X_f$)

This directory contains the 26 standardized feature tensors constructed for the infant EEG universality and individuality study.

## Cohort & Tensor Shape
Each tensor has exact shape:
$$(19, 2, 12, n)$$
* **Axis 0 (Infants, $N=19$):** Fixed sorted alphanumeric order from `sub-NORB00064` to `sub-NORB00103`.
* **Axis 1 (Sessions, $S=2$):** `0: ses-1`, `1: ses-2`.
* **Axis 2 (Epochs, $K=12$):** Chronological artifact-free segments (1 to 12).
* **Axis 3 (Elements, $n$):** $n = 19$ for node features; $n = 171$ for edge features.

## Files
| Tensor File | Feature Name | Representation | Type | Dimension |
| :--- | :--- | :--- | :--- | :---: |
"""
    for fid, meta in FEATURE_DEFS.items():
        readme_content += f"| `X_{fid}.npy` | `{meta['name']}` | {meta['network'].capitalize()} | {meta['type'].capitalize()} | {meta['dim']} |\n"
        
    with open(readme_path, "w") as f:
        f.write(readme_content)
        
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"SUCCESS: All 26 tensors successfully extracted and verified in {elapsed:.1f} seconds!")
    print(f"Tensors saved in: {OUTPUT_TENSORS_DIR}")
    print(f"Broadband GCC Threshold tau_c: mean = {metadata['tau_density_summary']['mean_tau']:.3f}, range = [{metadata['tau_density_summary']['min_tau']}, {metadata['tau_density_summary']['max_tau']}]")
    print(f"Broadband GCC Edge Density m: median = {metadata['tau_density_summary']['median_density']:.0f}, range = [{metadata['tau_density_summary']['min_density']}, {metadata['tau_density_summary']['max_density']}]")
    print("=" * 70)
    
    return tensors, metadata

if __name__ == "__main__":
    run_extraction()
