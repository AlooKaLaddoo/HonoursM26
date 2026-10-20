"""
Phase 3: Within-Session Self-Similarity Analysis (Step 3 - Comparison B)
Evaluates short-term state variability and within-session stability ceiling across all 26 features.
Computes:
  - Split-half self-similarity W_s(a) and W_s(a, e) (epochs 1-6 vs 7-12, paired k with k+6)
  - Matched others baseline O_s(a) and O_s(a, e)
  - Within-session identifiability diff_within_s = W_s - O_s
  - Odd/Even split sensitivity check (odd k vs even k, paired k with k+1)
  - Infant-level bootstrap 95% Confidence Intervals (10,000 resamples)
"""
import json
import numpy as np
from pathlib import Path

from .config import (
    SUBJECT_IDS, SESSIONS, N_EPOCHS, NODE_LABELS, EDGE_LABELS,
    FEATURE_DEFS, OUTPUT_TENSORS_DIR
)
from .utils import element_J, recording_S

def infant_value_bootstrap(values: np.ndarray, B: int = 10000, seed: int = 42):
    """Infant-level ordinary bootstrap for infant-level metrics (Step 7.2 item 2)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(B, dtype=np.float64)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boot_means[i] = np.mean(values[idx])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "median": float(np.median(boot_means)),
        "ci_95": [float(ci_low), float(ci_high)]
    }

def run_within_session():
    n_subs = len(SUBJECT_IDS)
    half_K = N_EPOCHS // 2 # 6 epochs
    
    print("=" * 80)
    print("PHASE 3: WITHIN-SESSION STABILITY & COMPARISON B (STEP 3)")
    print(f"Cohort: {n_subs} infants | Sessions: 2 | Split: 6 vs 6 epochs")
    print("=" * 80)
    
    results = {}
    
    for fid_idx, (fid, meta) in enumerate(FEATURE_DEFS.items(), 1):
        fname = meta["name"]
        dim = meta["dim"]
        ftype = meta["type"]
        
        tensor_path = OUTPUT_TENSORS_DIR / f"X_{fid}.npy"
        X = np.load(tensor_path) # (19, 2, 12, dim)
        
        feat_res = {
            "fid": fid,
            "name": fname,
            "dim": dim,
            "type": ftype
        }
        
        for ses_idx, ses in enumerate(SESSIONS):
            # 1. Primary Split-Half (epochs 1-6 vs 7-12)
            W_self_rec = np.zeros(n_subs, dtype=np.float64)
            W_self_elem = np.zeros((n_subs, dim), dtype=np.float64)
            
            O_others_rec = np.zeros(n_subs, dtype=np.float64)
            O_others_elem = np.zeros((n_subs, dim), dtype=np.float64)
            
            for a in range(n_subs):
                x_a_h1 = X[a, ses_idx, :half_K, :]       # (6, dim)
                x_a_h2 = X[a, ses_idx, half_K:N_EPOCHS, :] # (6, dim)
                
                # Self
                W_self_rec[a] = recording_S(x_a_h1, x_a_h2)
                W_self_elem[a, :] = element_J(x_a_h1, x_a_h2)
                
                # Matched others: compare a's first half with each other infant b's second half
                others_s_list = []
                others_j_list = []
                for b in range(n_subs):
                    if b == a:
                        continue
                    x_b_h2 = X[b, ses_idx, half_K:N_EPOCHS, :]
                    others_s_list.append(recording_S(x_a_h1, x_b_h2))
                    others_j_list.append(element_J(x_a_h1, x_b_h2))
                    
                O_others_rec[a] = np.mean(others_s_list)
                O_others_elem[a, :] = np.nanmean(others_j_list, axis=0)
                
            # Within-session identifiability diff: W - O
            diff_within_rec = W_self_rec - O_others_rec
            diff_within_elem = W_self_elem - O_others_elem
            
            # Save intermediate arrays: (19, dim)
            np.save(OUTPUT_TENSORS_DIR / f"W_elem_{fid}_{ses}.npy", W_self_elem)
            np.save(OUTPUT_TENSORS_DIR / f"O_elem_{fid}_{ses}.npy", O_others_elem)
            np.save(OUTPUT_TENSORS_DIR / f"diff_within_elem_{fid}_{ses}.npy", diff_within_elem)
            
            # Bootstrap CIs
            boot_W = infant_value_bootstrap(W_self_rec, seed=100 + ses_idx)
            boot_O = infant_value_bootstrap(O_others_rec, seed=200 + ses_idx)
            boot_diff = infant_value_bootstrap(diff_within_rec, seed=300 + ses_idx)
            
            # Test diff > 0 bootstrap p-value
            rng_p = np.random.default_rng(400 + ses_idx)
            diff_boots = [np.mean(diff_within_rec[rng_p.integers(0, n_subs, n_subs)]) for _ in range(10000)]
            p_val_diff = float(max(1.0 / 10000, np.mean(np.array(diff_boots) <= 0)))
            
            # 2. Odd/Even Split Sensitivity Check
            W_oddeven_rec = np.zeros(n_subs, dtype=np.float64)
            for a in range(n_subs):
                odd_epochs = X[a, ses_idx, 0::2, :] # k = 0, 2, 4, 6, 8, 10
                even_epochs = X[a, ses_idx, 1::2, :] # k = 1, 3, 5, 7, 9, 11
                W_oddeven_rec[a] = recording_S(odd_epochs, even_epochs)
            boot_oddeven = infant_value_bootstrap(W_oddeven_rec, seed=500 + ses_idx)
            
            feat_res[ses] = {
                "W_global": float(np.mean(W_self_rec)),
                "W_bootstrap": boot_W,
                "W_elem": np.nanmean(W_self_elem, axis=0).tolist(),
                "W_per_infant": W_self_rec.tolist(),
                "O_global": float(np.mean(O_others_rec)),
                "O_bootstrap": boot_O,
                "O_elem": np.nanmean(O_others_elem, axis=0).tolist(),
                "diff_within_global": float(np.mean(diff_within_rec)),
                "diff_within_bootstrap": boot_diff,
                "diff_within_p": p_val_diff,
                "diff_within_elem": np.nanmean(diff_within_elem, axis=0).tolist(),
                "odd_even_sensitivity": boot_oddeven
            }
            
        results[fid] = feat_res
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:26]:<26}) | W_1: {feat_res['ses-1']['W_global']:.4f} [O_1: {feat_res['ses-1']['O_global']:.4f}, diff: {feat_res['ses-1']['diff_within_global']:+.4f}, p={feat_res['ses-1']['diff_within_p']:.4f}] | W_2: {feat_res['ses-2']['W_global']:.4f} [diff: {feat_res['ses-2']['diff_within_global']:+.4f}]")
        
    out_json = OUTPUT_TENSORS_DIR / "within_session_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 80)
    print(f"Within-session stability results saved to: {out_json}")
    print("=" * 80)
    return results

if __name__ == "__main__":
    run_within_session()

