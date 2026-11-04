"""
Phase 3: Within-Session Self-Similarity Analysis (Step 3 - Comparison B)
Evaluates short-term state variability and within-session stability ceiling across all 26 features.
Computes, for the primary split (epochs 1-6 vs 7-12, paired k with k+6) and, as the sensitivity check of
Step 3 item 5, for the odd/even split (odd k vs even k, paired k with k+1):
  - Self-similarity W_s(a) and W_s(a, e)
  - Matched others baseline O_s(a) and O_s(a, e)
  - Within-session identifiability W_s - O_s at recording and element level, tested against 0 with the infant
    bootstrap (two-sided, Step 7.2 item 4); element p-values are FDR-corrected across elements
  - Infant-level bootstrap 95% Confidence Intervals (200,000 resamples by default, one recorded seed)
"""
import json
import sys
import warnings
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
        N_EPOCHS,
        N_PERMUTATIONS,
        OUTPUT_TENSORS_DIR,
        SEED,
        SESSIONS,
        SUBJECT_IDS,
    )
    from analysis_pipeline.utils import (
        bootstrap_ci,
        bootstrap_counts,
        bootstrap_draws,
        bootstrap_p,
        element_J,
        recording_S,
        weighted_nanmean,
    )
else:
    from .config import (
        FEATURE_DEFS,
        N_EPOCHS,
        N_PERMUTATIONS,
        OUTPUT_TENSORS_DIR,
        SEED,
        SESSIONS,
        SUBJECT_IDS,
    )
    from .utils import (
        bootstrap_ci,
        bootstrap_counts,
        bootstrap_draws,
        bootstrap_p,
        element_J,
        recording_S,
        weighted_nanmean,
    )

def infant_summary(values: np.ndarray, boot: np.ndarray) -> dict:
    """Mean and SD over the infants, and median and 95% CI of the infant bootstrap means (Step 7.1, 7.2 item 2)."""
    ci_low, ci_high = bootstrap_ci(boot)
    return {
        "mean": float(np.nanmean(values)),
        "sd": float(np.nanstd(values)),
        "median": float(np.nanmedian(boot)),
        "ci_95": [float(ci_low), float(ci_high)]
    }

def split_half(X_s: np.ndarray, first: np.ndarray, second: np.ndarray, counts: np.ndarray):
    """
    Step 3 items 2-4 for one session X_s of shape (19, 12, dim), pairing epoch first[i] with epoch second[i]:
    self W(a) and W(a, e) between an infant's own two halves, matched others O(a) and O(a, e) between a's first half
    and every other infant's second half (averaged over the 18 others), and the differences W - O.
    Infant bootstrap means are c-weighted means over the infants (counts c: (B, 19)), i.e. an ordinary bootstrap of
    the 19 per-infant values; W - O is tested against 0 with the two-sided bootstrap p (Step 7.2 item 4).
    Returns the summary dict and the per-infant element arrays W(a, e), O(a, e) and W(a, e) - O(a, e).
    """
    n_subs = X_s.shape[0]
    W_self_rec = np.array([recording_S(X_s[a, first], X_s[a, second]) for a in range(n_subs)])
    W_self_elem = np.array([element_J(X_s[a, first], X_s[a, second]) for a in range(n_subs)])
    O_others_rec = np.zeros(n_subs, dtype=np.float64)
    O_others_elem = np.zeros_like(W_self_elem)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # elements that are 0/0 against every other infant
        for a in range(n_subs):
            others = [b for b in range(n_subs) if b != a]
            O_others_rec[a] = np.nanmean([recording_S(X_s[a, first], X_s[b, second]) for b in others])
            O_others_elem[a, :] = np.nanmean([element_J(X_s[a, first], X_s[b, second]) for b in others], axis=0)

        # Within-session identifiability diff: W - O
        diff_within_rec = W_self_rec - O_others_rec
        diff_within_elem = W_self_elem - O_others_elem

        boot_diff = weighted_nanmean(counts, diff_within_rec)
        boot_diff_elem = weighted_nanmean(counts, diff_within_elem) # (B, dim)
        p_elem = bootstrap_p(boot_diff_elem)
        q_elem = np.asarray(multipletests(p_elem, method="fdr_bh")[1])

        summary = {
            "W_global": float(np.nanmean(W_self_rec)),
            "W_bootstrap": infant_summary(W_self_rec, weighted_nanmean(counts, W_self_rec)),
            "W_elem": np.nanmean(W_self_elem, axis=0).tolist(),
            "W_per_infant": W_self_rec.tolist(),
            "O_global": float(np.nanmean(O_others_rec)),
            "O_bootstrap": infant_summary(O_others_rec, weighted_nanmean(counts, O_others_rec)),
            "O_elem": np.nanmean(O_others_elem, axis=0).tolist(),
            "diff_within_global": float(np.nanmean(diff_within_rec)),
            "diff_within_bootstrap": infant_summary(diff_within_rec, boot_diff),
            "diff_within_p": float(bootstrap_p(boot_diff)),
            "diff_within_elem": np.nanmean(diff_within_elem, axis=0).tolist(),
            "diff_within_elem_ci_95": bootstrap_ci(boot_diff_elem).T.tolist(),
            "diff_within_elem_p": p_elem.tolist(),
            "diff_within_elem_q": q_elem.tolist(),
            "n_sig_diff_within_elem": int(np.sum(q_elem < 0.05))
        }
    return summary, W_self_elem, O_others_elem, diff_within_elem

def run_within_session(n_bootstrap: int = N_PERMUTATIONS, seed: int = SEED):
    n_subs = len(SUBJECT_IDS)
    half_K = N_EPOCHS // 2 # 6 epochs
    first_half, second_half = np.arange(half_K), np.arange(half_K, N_EPOCHS) # k with k + 6
    odd_epochs, even_epochs = np.arange(0, N_EPOCHS, 2), np.arange(1, N_EPOCHS, 2) # k = 1, 3, ... with k + 1

    # One set of infant resamples for every bootstrap (Step 0 item 8, Step 7.2)
    counts = bootstrap_counts(bootstrap_draws(n_subs, n_bootstrap, seed), n_subs)

    print("=" * 80)
    print("PHASE 3: WITHIN-SESSION STABILITY & COMPARISON B (STEP 3)")
    print(f"Cohort: {n_subs} infants | Sessions: 2 | Split: 6 vs 6 epochs (sensitivity: odd vs even) | Bootstrap: {n_bootstrap} | Seed: {seed}")
    print("=" * 80)

    results = {}

    for fid_idx, (fid, meta) in enumerate(FEATURE_DEFS.items(), 1):
        fname = meta["name"]
        dim = meta["dim"]
        ftype = meta["type"]

        tensor_path = OUTPUT_TENSORS_DIR / f"X_{fid}.npy"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Feature tensor not found: {tensor_path}. Run Phase 1 first.")
        X = np.load(tensor_path) # (19, 2, 12, dim)

        feat_res = {
            "fid": fid,
            "name": fname,
            "dim": dim,
            "type": ftype,
            "seed": seed,
            "n_bootstrap": n_bootstrap
        }

        for ses_idx, ses in enumerate(SESSIONS):
            # 1. Primary Split-Half (epochs 1-6 vs 7-12)
            primary, W_self_elem, O_others_elem, diff_within_elem = split_half(X[:, ses_idx], first_half, second_half, counts)

            # Save intermediate arrays: (19, dim)
            np.save(OUTPUT_TENSORS_DIR / f"W_elem_{fid}_{ses}.npy", W_self_elem)
            np.save(OUTPUT_TENSORS_DIR / f"O_elem_{fid}_{ses}.npy", O_others_elem)
            np.save(OUTPUT_TENSORS_DIR / f"diff_within_elem_{fid}_{ses}.npy", diff_within_elem)

            # 2. Odd/Even Split Sensitivity Check: items 2-4 repeated with odd k paired with k + 1
            odd_even, *_ = split_half(X[:, ses_idx], odd_epochs, even_epochs, counts)

            feat_res[ses] = {**primary, "odd_even_sensitivity": odd_even}

        results[fid] = feat_res
        s1, s2 = feat_res["ses-1"], feat_res["ses-2"]
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:26]:<26}) | W_1: {s1['W_global']:.4f} [O_1: {s1['O_global']:.4f}, diff: {s1['diff_within_global']:+.4f}, p={s1['diff_within_p']:.4f}, sig elements={s1['n_sig_diff_within_elem']}/{dim}] | W_2: {s2['W_global']:.4f} [diff: {s2['diff_within_global']:+.4f}, p={s2['diff_within_p']:.4f}] | odd/even diff: {s1['odd_even_sensitivity']['diff_within_global']:+.4f}, {s2['odd_even_sensitivity']['diff_within_global']:+.4f}")

    out_json = OUTPUT_TENSORS_DIR / "within_session_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"Within-session stability results saved to: {out_json}")
    print("=" * 80)
    return results

if __name__ == "__main__":
    run_within_session()
