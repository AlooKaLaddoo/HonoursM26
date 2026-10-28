"""
Phase 4: Longitudinal Identifiability & Fingerprinting (Step 4 - Comparisons C & D)
Evaluates neural fingerprinting and subject-specific identifiability across all 26 feature sets.
Computes:
  - 19 x 19 cross-session matrix M[a, b] (rows: ses-1, columns: ses-2)
  - I_self, I_others, and Differential Identifiability I_diff
  - 1-Nearest Neighbor Identification Accuracy (ses-1 -> ses-2, ses-2 -> ses-1, and Mean)
  - Element-level M_e, I_self(e), I_others(e), and I_diff(e)
  - Permutation tests (200,000 column permutations by default) for global I_diff, accuracy, and element I_diff(e)
  - Benjamini-Hochberg FDR correction across elements (q < 0.05)
  - Infant bootstrap 95% Confidence Intervals (as many resamples as permutations)
Saves M matrices as .npy and summary JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

# Support both direct script execution and package imports
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from analysis_pipeline.config import (
        FEATURE_DEFS,
        OUTPUT_TENSORS_DIR,
        SUBJECT_IDS,
    )
    from analysis_pipeline.utils import element_J, recording_S
else:
    from .config import (
        FEATURE_DEFS,
        OUTPUT_TENSORS_DIR,
        SUBJECT_IDS,
    )
    from .utils import element_J, recording_S


def infant_identifiability_bootstrap(M: np.ndarray, B: int = 200000, seed: int = 42):
    """
    Infant bootstrap (Step 7.2 item 3) for matrix M.
    I_self = mean(M[dp, dp]), I_others = mean(M[dp, dq] for dp != dq).
    """
    rng = np.random.default_rng(seed)
    n_subs = M.shape[0]
    p_grid, q_grid = np.meshgrid(range(n_subs), range(n_subs), indexing="ij")
    off_diag_mask = p_grid != q_grid
    
    boot_diff = np.empty(B, dtype=np.float64)
    boot_self = np.empty(B, dtype=np.float64)
    boot_others = np.empty(B, dtype=np.float64)
    
    for i in range(B):
        d = rng.integers(0, n_subs, n_subs)
        # diagonal self
        b_self = np.mean(M[d, d])
        # off-diagonal others with distinct infants
        dp = d[p_grid]
        dq = d[q_grid]
        valid_off = off_diag_mask & (dp != dq)
        b_others = np.mean(M[dp[valid_off], dq[valid_off]])
        
        boot_self[i] = b_self
        boot_others[i] = b_others
        boot_diff[i] = (b_self - b_others) * 100.0
        
    return {
        "self_ci_95": [float(np.percentile(boot_self, 2.5)), float(np.percentile(boot_self, 97.5))],
        "others_ci_95": [float(np.percentile(boot_others, 2.5)), float(np.percentile(boot_others, 97.5))],
        "idiff_median": float(np.median(boot_diff)),
        "idiff_ci_95": [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))],
        "idiff_boot_sd": float(np.std(boot_diff))
    }


def run_identifiability(n_permutations: int = 200000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_subs = len(SUBJECT_IDS)
    
    print("=" * 80)
    print("PHASE 4: LONGITUDINAL IDENTIFIABILITY & FINGERPRINTING (STEP 4)")
    print(f"Cohort: {n_subs} infants | Cross-Session Matrix: {n_subs}x{n_subs} | Permutations: {n_permutations}")
    print("=" * 80)
    
    results = {}
    
    for fid_idx, (fid, meta) in enumerate(FEATURE_DEFS.items(), 1):
        fname = meta["name"]
        dim = meta["dim"]
        ftype = meta["type"]
        
        tensor_path = OUTPUT_TENSORS_DIR / f"X_{fid}.npy"
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Feature tensor not found: {tensor_path}. Run 01_extract_tensors.py first."
            )
        X = np.load(tensor_path) # (19, 2, 12, dim)
        
        # 1. Construct 19x19 Identifiability Matrix M and element tensor M_elem (19, 19, dim)
        M = np.zeros((n_subs, n_subs), dtype=np.float64)
        M_elem = np.zeros((n_subs, n_subs, dim), dtype=np.float64)
        
        for a in range(n_subs):
            x_a_s1 = X[a, 0, :, :] # ses-1 (12, dim)
            for b in range(n_subs):
                x_b_s2 = X[b, 1, :, :] # ses-2 (12, dim)
                
                # Recording level: S( (a, 1), (b, 2) ) with epoch pairing (k, k)
                M[a, b] = recording_S(x_a_s1, x_b_s2)
                # Element level: J( (a, 1), (b, 2) )
                M_elem[a, b, :] = element_J(x_a_s1, x_b_s2)
                
        # Save M matrix to disk
        np.save(OUTPUT_TENSORS_DIR / f"M_mat_{fid}.npy", M)
        np.save(OUTPUT_TENSORS_DIR / f"M_{fid}.npy", M)
        np.save(OUTPUT_TENSORS_DIR / f"M_elem_{fid}.npy", M_elem)
        
        # 2. Global Identifiability
        diag_mask = np.eye(n_subs, dtype=bool)
        I_self = float(np.mean(M[diag_mask]))
        I_others = float(np.mean(M[~diag_mask]))
        I_diff = float((I_self - I_others) * 100.0)
        
        # 3. Identification Accuracy (1-NN)
        row_idx = np.arange(n_subs)
        # Forward: ses-1 -> ses-2 (row max)
        f_pred = np.argmax(M, axis=1)
        f_max = np.take_along_axis(M, f_pred[:, None], axis=1)
        f_unique = (np.sum(M == f_max, axis=1) == 1)
        forward_correct = int(np.sum((f_pred == row_idx) & f_unique))
        acc_forward = float((forward_correct / n_subs) * 100.0)
        
        # Backward: ses-2 -> ses-1 (column max)
        b_pred = np.argmax(M, axis=0)
        b_max = np.take_along_axis(M, b_pred[None, :], axis=0).squeeze()
        b_unique = (np.sum(M == b_max[None, :], axis=0) == 1)
        backward_correct = int(np.sum((b_pred == row_idx) & b_unique))
        acc_backward = float((backward_correct / n_subs) * 100.0)
        acc_mean = float((acc_forward + acc_backward) / 2.0)
        
        # 4. Element-level Identifiability
        with np.errstate(divide="ignore", invalid="ignore"):
            I_self_elem = np.nanmean(M_elem[diag_mask, :], axis=0) # shape (dim,)
            I_others_elem = np.nanmean(M_elem[~diag_mask, :], axis=0)
        I_diff_elem = (I_self_elem - I_others_elem) * 100.0
        
        # 5. Per-infant values: self(a), others(a), idiff(a)
        per_infant_self = np.diag(M)
        per_infant_others = np.zeros(n_subs, dtype=np.float64)
        for a in range(n_subs):
            row_off = [M[a, b] for b in range(n_subs) if b != a]
            col_off = [M[b, a] for b in range(n_subs) if b != a]
            per_infant_others[a] = np.mean(row_off + col_off)
        per_infant_idiff = (per_infant_self - per_infant_others) * 100.0
        
        # 6. Permutation Tests (n_permutations permutations of columns of M)
        null_idiff = np.empty(n_permutations, dtype=np.float64)
        null_acc_f = np.empty(n_permutations, dtype=np.float64)
        null_acc_b = np.empty(n_permutations, dtype=np.float64)
        null_elem_ge = np.zeros(dim, dtype=np.int32)
        
        total_M_sum = np.sum(M)
        with np.errstate(divide="ignore", invalid="ignore"):
            total_elem_sum = np.nansum(M_elem, axis=(0, 1))
            total_elem_count = np.sum(~np.isnan(M_elem), axis=(0, 1))
        
        for perm_i in range(n_permutations):
            col_perm = rng.permutation(n_subs)
            M_perm = M[:, col_perm]
            
            # Global null
            s_sum = np.sum(M[row_idx, col_perm])
            s_null = s_sum / n_subs
            o_null = (total_M_sum - s_sum) / (n_subs * (n_subs - 1))
            null_idiff[perm_i] = (s_null - o_null) * 100.0
            
            # Accuracy null
            p_f = np.argmax(M_perm, axis=1)
            m_f = np.take_along_axis(M_perm, p_f[:, None], axis=1)
            u_f = (np.sum(M_perm == m_f, axis=1) == 1)
            c_f = np.sum((p_f == row_idx) & u_f)
            null_acc_f[perm_i] = (c_f / n_subs) * 100.0
            
            p_b = np.argmax(M_perm, axis=0)
            m_b = np.take_along_axis(M_perm, p_b[None, :], axis=0).squeeze()
            u_b = (np.sum(M_perm == m_b[None, :], axis=0) == 1)
            c_b = np.sum((p_b == row_idx) & u_b)
            null_acc_b[perm_i] = (c_b / n_subs) * 100.0
            
            # Element null: identical column permutation across all element matrices
            diag_vals = M_elem[row_idx, col_perm, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                s_elem_null = np.nanmean(diag_vals, axis=0)
                diag_sum = np.nansum(diag_vals, axis=0)
                diag_count = np.sum(~np.isnan(diag_vals), axis=0)
                off_sum = total_elem_sum - diag_sum
                off_count = total_elem_count - diag_count
                o_elem_null = np.where(off_count > 0, off_sum / off_count, np.nan)
            diff_elem_null = (s_elem_null - o_elem_null) * 100.0
            null_elem_ge += (diff_elem_null >= I_diff_elem).astype(np.int32)
            
        p_idiff = float((1.0 + np.sum(null_idiff >= I_diff)) / (1.0 + n_permutations))
        p_acc_f = float((1.0 + np.sum(null_acc_f >= acc_forward)) / (1.0 + n_permutations))
        p_acc_b = float((1.0 + np.sum(null_acc_b >= acc_backward)) / (1.0 + n_permutations))
        p_elem = (1.0 + null_elem_ge) / (1.0 + n_permutations)
        # In a one-tailed test for individual identifiability (H1: I_diff > 0),
        # elements with non-positive differential identifiability or NaNs cannot reject the null
        p_elem[np.isnan(I_diff_elem) | (I_diff_elem <= 1e-6)] = 1.0
        
        # FDR correction across elements
        _, q_elem, _, _ = multipletests(p_elem, method="fdr_bh")
        
        # Bootstrap stats (draws follow n_permutations, so a quick run is quick throughout)
        boot_stats = infant_identifiability_bootstrap(M, B=n_permutations, seed=1000 + fid_idx)
        
        feat_res = {
            "fid": fid,
            "name": fname,
            "dim": dim,
            "type": ftype,
            "network": meta["network"],
            "band": meta["band"],
            "I_self": I_self,
            "I_others": I_others,
            "I_diff": I_diff,
            "p_idiff": p_idiff,
            "acc_forward": acc_forward,
            "p_acc_forward": p_acc_f,
            "acc_backward": acc_backward,
            "p_acc_backward": p_acc_b,
            "acc_mean": acc_mean,
            "bootstrap": boot_stats,
            "I_self_elem": I_self_elem.tolist(),
            "I_others_elem": I_others_elem.tolist(),
            "I_diff_elem": I_diff_elem.tolist(),
            "p_elem": p_elem.tolist(),
            "q_elem": q_elem.tolist(),
            "n_sig_elements": int(np.sum(q_elem < 0.05)),
            "per_infant_self": per_infant_self.tolist(),
            "per_infant_others": per_infant_others.tolist(),
            "per_infant_idiff": per_infant_idiff.tolist()
        }
        
        results[fid] = feat_res
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:26]:<26}) | I_self: {I_self:.4f} | I_others: {I_others:.4f} | I_diff: {I_diff:+.2f}% (p={p_idiff:.4f}) | Acc: {acc_mean:.1f}% (Fwd:{acc_forward:.1f}%, Bwd:{acc_backward:.1f}%) | Sig Elements: {feat_res['n_sig_elements']}/{dim}")
        
    out_json = OUTPUT_TENSORS_DIR / "identifiability_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 80)
    print(f"Identifiability results saved to: {out_json}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_identifiability()
