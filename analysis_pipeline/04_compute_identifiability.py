"""
Phase 4: Longitudinal Identifiability & Fingerprinting (Step 4 - Comparisons C & D)
Evaluates neural fingerprinting and subject-specific identifiability across all 26 feature sets.
Computes:
  - 19 x 19 cross-session matrix M[a, b] (rows: ses-1, columns: ses-2)
  - I_self, I_others, and Differential Identifiability I_diff = I_self - I_others (Jaccard units)
  - 1-Nearest Neighbor Identification Accuracy (ses-1 -> ses-2, ses-2 -> ses-1, and Mean)
  - Element-level M_e, I_self(e), I_others(e), and I_diff(e)
  - Permutation tests (200,000 column permutations by default) for global I_diff, each accuracy, the mean accuracy,
    and element I_diff(e)
  - Benjamini-Hochberg FDR correction across elements, and across the 26 features for each global test (q < 0.05)
  - Infant bootstrap 95% Confidence Intervals (as many resamples as permutations, one recorded seed)
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
        N_PERMUTATIONS,
        PERM_TIE_TOL,
        OUTPUT_TENSORS_DIR,
        SEED,
        SUBJECT_IDS,
    )
    from analysis_pipeline.utils import (
        bootstrap_ci,
        bootstrap_counts,
        bootstrap_draws,
        element_J,
        identification_accuracy,
        recording_S,
        weighted_nanmean,
    )
else:
    from .config import (
        FEATURE_DEFS,
        N_PERMUTATIONS,
        PERM_TIE_TOL,
        OUTPUT_TENSORS_DIR,
        SEED,
        SUBJECT_IDS,
    )
    from .utils import (
        bootstrap_ci,
        bootstrap_counts,
        bootstrap_draws,
        element_J,
        identification_accuracy,
        recording_S,
        weighted_nanmean,
    )


def infant_identifiability_bootstrap(M: np.ndarray, W_self: np.ndarray, W_others: np.ndarray):
    """
    Infant bootstrap (Step 7.2 item 3) for matrix M, from the shared infant resamples:
    I_self = mean of M[d_p, d_p] over positions p (diagonal cells weighted by c_a, W_self: (B, 19)), and
    I_others = mean of M[d_p, d_q] over positions p != q with d_p != d_q (off-diagonal cells, row-major, weighted
    by c_a * c_b, W_others: (B, 342)).
    """
    off_diag_mask = ~np.eye(M.shape[0], dtype=bool)
    boot_self = weighted_nanmean(W_self, np.diag(M))
    boot_others = weighted_nanmean(W_others, M[off_diag_mask])
    boot_diff = boot_self - boot_others

    return {
        "self_median": float(np.nanmedian(boot_self)),
        "self_ci_95": bootstrap_ci(boot_self).tolist(),
        "others_median": float(np.nanmedian(boot_others)),
        "others_ci_95": bootstrap_ci(boot_others).tolist(),
        "idiff_median": float(np.nanmedian(boot_diff)),
        "idiff_ci_95": bootstrap_ci(boot_diff).tolist(),
        "idiff_boot_sd": float(np.nanstd(boot_diff))
    }


def run_identifiability(n_permutations: int = N_PERMUTATIONS, seed: int = SEED):
    n_subs = len(SUBJECT_IDS)

    # One set of infant resamples for every feature (Step 0 item 8, Step 7.2)
    counts = bootstrap_counts(bootstrap_draws(n_subs, n_permutations, seed), n_subs)
    W_others = (counts[:, :, None] * counts[:, None, :])[:, ~np.eye(n_subs, dtype=bool)]

    print("=" * 80)
    print("PHASE 4: LONGITUDINAL IDENTIFIABILITY & FINGERPRINTING (STEP 4)")
    print(f"Cohort: {n_subs} infants | Cross-Session Matrix: {n_subs}x{n_subs} | Permutations: {n_permutations} | Seed: {seed}")
    print("=" * 80)

    results = {}

    for fid_idx, (fid, meta) in enumerate(FEATURE_DEFS.items(), 1):
        fname = meta["name"]
        dim = meta["dim"]
        ftype = meta["type"]
        # The same column permutations for every feature, from the one recorded seed
        rng = np.random.default_rng(seed)

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

        # 2. Global Identifiability (Jaccard units, Step 7.5)
        diag_mask = np.eye(n_subs, dtype=bool)
        I_self = float(np.nanmean(M[diag_mask]))
        I_others = float(np.nanmean(M[~diag_mask]))
        I_diff = I_self - I_others

        # 3. Identification Accuracy (1-NN): forward ses-1 -> ses-2 (row max), backward ses-2 -> ses-1 (column max)
        row_idx = np.arange(n_subs)
        acc_forward, acc_backward = identification_accuracy(M)
        acc_mean = float((acc_forward + acc_backward) / 2.0)

        # 4. Element-level Identifiability
        with np.errstate(divide="ignore", invalid="ignore"):
            I_self_elem = np.nanmean(M_elem[diag_mask, :], axis=0) # shape (dim,)
            I_others_elem = np.nanmean(M_elem[~diag_mask, :], axis=0)
        I_diff_elem = I_self_elem - I_others_elem

        # 5. Per-infant values: self(a), others(a), idiff(a)
        per_infant_self = np.diag(M)
        per_infant_others = np.zeros(n_subs, dtype=np.float64)
        for a in range(n_subs):
            row_off = [M[a, b] for b in range(n_subs) if b != a]
            col_off = [M[b, a] for b in range(n_subs) if b != a]
            per_infant_others[a] = np.nanmean(row_off + col_off)
        per_infant_idiff = per_infant_self - per_infant_others

        # 6. Permutation Tests (n_permutations permutations of columns of M)
        null_idiff = np.empty(n_permutations, dtype=np.float64)
        null_acc_f = np.empty(n_permutations, dtype=np.float64)
        null_acc_b = np.empty(n_permutations, dtype=np.float64)
        null_elem_ge = np.zeros(dim, dtype=np.int32)

        # Sums and counts of the defined cells; a column permutation keeps them and only moves cells on/off the diagonal
        M_valid = ~np.isnan(M)
        M_filled = np.where(M_valid, M, 0.0)
        total_M_sum, total_M_count = M_filled.sum(), M_valid.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            total_elem_sum = np.nansum(M_elem, axis=(0, 1))
            total_elem_count = np.sum(~np.isnan(M_elem), axis=(0, 1))

        for perm_i in range(n_permutations):
            col_perm = rng.permutation(n_subs)

            # Global null
            s_sum, s_count = M_filled[row_idx, col_perm].sum(), M_valid[row_idx, col_perm].sum()
            null_idiff[perm_i] = s_sum / s_count - (total_M_sum - s_sum) / (total_M_count - s_count)

            # Accuracy null
            null_acc_f[perm_i], null_acc_b[perm_i] = identification_accuracy(M[:, col_perm])

            # Element null: identical column permutation across all element matrices
            diag_vals = M_elem[row_idx, col_perm, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                s_elem_null = np.nanmean(diag_vals, axis=0)
                diag_sum = np.nansum(diag_vals, axis=0)
                diag_count = np.sum(~np.isnan(diag_vals), axis=0)
                off_sum = total_elem_sum - diag_sum
                off_count = total_elem_count - diag_count
                o_elem_null = np.where(off_count > 0, off_sum / off_count, np.nan)
            diff_elem_null = s_elem_null - o_elem_null
            # The observed I_diff(e) (np.nanmean, above) and the null (sums and counts) are computed in a different
            # floating-point order; the tolerance stops an exact tie from missing by rounding error
            null_elem_ge += (diff_elem_null >= I_diff_elem - PERM_TIE_TOL).astype(np.int32)

        null_acc_mean = (null_acc_f + null_acc_b) / 2.0
        p_idiff = float((1.0 + np.sum(null_idiff >= I_diff - PERM_TIE_TOL)) / (1.0 + n_permutations))
        p_acc_f = float((1.0 + np.sum(null_acc_f >= acc_forward - PERM_TIE_TOL)) / (1.0 + n_permutations))
        p_acc_b = float((1.0 + np.sum(null_acc_b >= acc_backward - PERM_TIE_TOL)) / (1.0 + n_permutations))
        p_acc_m = float((1.0 + np.sum(null_acc_mean >= acc_mean - PERM_TIE_TOL)) / (1.0 + n_permutations))
        p_elem = (1.0 + null_elem_ge) / (1.0 + n_permutations)
        # Elements that are 0/0 in every comparison have no observed I_diff(e); no null value is >= NaN,
        # so without this they would get the smallest possible p-value
        p_elem[np.isnan(I_diff_elem)] = 1.0

        # FDR correction across elements
        _, q_elem, _, _ = multipletests(p_elem, method="fdr_bh")

        # Bootstrap stats (draws follow n_permutations, so a quick run is quick throughout).
        # One recorded seed (Step 0 item 8): every feature uses the same infant resamples (Step 7.2 item 4)
        boot_stats = infant_identifiability_bootstrap(M, counts, W_others)

        feat_res = {
            "fid": fid,
            "name": fname,
            "dim": dim,
            "type": ftype,
            "network": meta["network"],
            "band": meta["band"],
            "seed": seed,
            "n_permutations": n_permutations,
            "I_self": I_self,
            "I_others": I_others,
            "I_diff": I_diff,
            "p_idiff": p_idiff,
            "acc_forward": acc_forward,
            "p_acc_forward": p_acc_f,
            "acc_backward": acc_backward,
            "p_acc_backward": p_acc_b,
            "acc_mean": acc_mean,
            "p_acc_mean": p_acc_m,
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
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:26]:<26}) | I_self: {I_self:.4f} | I_others: {I_others:.4f} | I_diff: {I_diff:+.4f} (p={p_idiff:.4f}) | Acc: {acc_mean:.1f}% (p={p_acc_m:.4f}; Fwd:{acc_forward:.1f}%, Bwd:{acc_backward:.1f}%) | Sig Elements: {feat_res['n_sig_elements']}/{dim}")

    # FDR correction of each global test across the 26 feature sets (Step 7.4 ii)
    for p_key, q_key in [("p_idiff", "q_idiff"), ("p_acc_forward", "q_acc_forward"), ("p_acc_backward", "q_acc_backward"), ("p_acc_mean", "q_acc_mean")]:
        q_global = multipletests([results[fid][p_key] for fid in results], method="fdr_bh")[1]
        for fid, q_val in zip(results, q_global):
            results[fid][q_key] = float(q_val)
    print(f"Global I_diff significant after FDR across 26 features (q < 0.05): {sum(results[fid]['q_idiff'] < 0.05 for fid in results)}/26")

    out_json = OUTPUT_TENSORS_DIR / "identifiability_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"Identifiability results saved to: {out_json}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_identifiability()
