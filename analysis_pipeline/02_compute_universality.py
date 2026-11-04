"""
Phase 2: Complete Universality Analysis (Step 2 - Comparison A)
Evaluates population-wide shared organization across all 26 feature sets for ses-1 and ses-2.
Computes:
  - Element-level universality U_1(e), U_2(e), and Delta U(e)
  - Recording-level universality U_1, U_2, and Delta U
  - Per-infant universality u_s(a) and Delta u(a)
  - Regional universality U_s(g) (5 node regions, 15 edge region pairs) as the column means of the
    pairs x regions matrix of J (Step 2 item 4)
  - Element-shuffle null hypothesis permutation tests (200,000 permutations per feature/session by default)
  - Infant-level bootstrap 95% Confidence Intervals for U_s, U_s(g), Delta U and Delta U(e) (as many resamples as
    permutations); one recorded seed, so every statistic uses the same infant resamples
  - Benjamini-Hochberg FDR correction across elements (q < 0.05) for U_s(e) and for Delta U(e)
  - Undefined (0/0) occurrence tracking per element
Saves intermediate pair arrays and summary JSON.
"""
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

# Support both direct script execution and package imports
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from analysis_pipeline.config import (
        EDGE_LABELS,
        FEATURE_DEFS,
        N_PERMUTATIONS,
        PERM_TIE_TOL,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        REGION_PAIRS,
        SEED,
        SESSIONS,
        SUBJECT_IDS,
    )
    from analysis_pipeline.utils import (
        bootstrap_ci,
        bootstrap_counts,
        bootstrap_draws,
        bootstrap_p,
        weighted_nanmean,
    )
else:
    from .config import (
        EDGE_LABELS,
        FEATURE_DEFS,
        N_PERMUTATIONS,
        PERM_TIE_TOL,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        REGION_PAIRS,
        SEED,
        SESSIONS,
        SUBJECT_IDS,
    )
    from .utils import (
        bootstrap_ci,
        bootstrap_counts,
        bootstrap_draws,
        bootstrap_p,
        weighted_nanmean,
    )

# Permutations per task of the element-shuffle null. Each task gets a child seed of the one recorded seed,
# so the null values do not depend on the number of worker processes.
PERM_BLOCK = 500


def get_node_region(node: str) -> str:
    return NODE_REGIONS.get(node, "Unknown")

def get_edge_region_pair(edge: str) -> str:
    u, v = edge.split("-")
    r1, r2 = NODE_REGIONS.get(u, "F"), NODE_REGIONS.get(v, "F")
    region_order = {"F": 0, "C": 1, "T": 2, "P": 3, "O": 4}
    if region_order.get(r1, 0) <= region_order.get(r2, 0):
        return f"{r1}-{r2}"
    else:
        return f"{r2}-{r1}"

def region_groups(ftype: str) -> dict:
    """Element indices of every region (nodes: F, C, T, P, O) or region pair (edges: F-F ... O-O), in that order."""
    if ftype == "node":
        return {reg: [i for i, n in enumerate(NODE_LABELS) if get_node_region(n) == reg] for reg in ["F", "C", "T", "P", "O"]}
    groups = {rp: [i for i, e in enumerate(EDGE_LABELS) if get_edge_region_pair(e) == rp] for rp in REGION_PAIRS}
    return {rp: idxs for rp, idxs in groups.items() if idxs}

def pair_recording_S(min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
    """
    Recording-level S of every pair from the stacked minima and maxima (pairs, epochs, elements): the mean over the
    epoch pairs of sum(min) / sum(max). An epoch pair whose two vectors are all zero is 0/0 and left out, as in
    utils.recording_S.
    """
    num = np.sum(min_v, axis=2)
    den = np.sum(max_v, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_epochs = np.where(den > 0, num / den, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # pairs without any defined epoch pair stay NaN
        return np.nanmean(s_epochs, axis=1)

def boot_summary(boot: np.ndarray) -> dict:
    """Median, 95% percentile CI and SD of one statistic's infant bootstrap values (Step 7.2)."""
    ci_low, ci_high = bootstrap_ci(boot)
    return {
        "median": float(np.nanmedian(boot)),
        "ci_95": [float(ci_low), float(ci_high)],
        "boot_sd": float(np.nanstd(boot))
    }

def _run_perm_worker(seed_seq: np.random.SeedSequence, n_perms: int, A_stacked: np.ndarray, B_stacked: np.ndarray, obs_U_elem: np.ndarray, obs_U_global: float):
    """Worker function for one block of element-shuffle null permutations."""
    rng = np.random.default_rng(seed_seq)
    n_pairs, _, dim = A_stacked.shape

    null_elem_ge = np.zeros(dim, dtype=np.int64)
    null_global_ge = 0

    for _ in range(n_perms):
        # Generate random permutation per pair
        perms = np.array([rng.permutation(dim) for _ in range(n_pairs)]) # (n_pairs, dim)
        B_perm = np.take_along_axis(B_stacked, perms[:, np.newaxis, :], axis=2) # (n_pairs, 12, dim)

        min_v = np.minimum(A_stacked, B_perm)
        max_v = np.maximum(A_stacked, B_perm)

        # Element Jaccard: (n_pairs, 12, dim)
        with np.errstate(divide="ignore", invalid="ignore"):
            j_elem = np.where(max_v > 0, min_v / max_v, np.nan)
        # Average over epochs within each pair, then over pairs, exactly like the observed U_elem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # pairs that are 0/0 in all 12 epochs
            null_U_e = np.nanmean(np.nanmean(j_elem, axis=1), axis=0)
            null_elem_ge += (null_U_e >= obs_U_elem - PERM_TIE_TOL).astype(np.int64)

            # Recording Jaccard: sum over elements
            null_global_ge += int(np.nanmean(pair_recording_S(min_v, max_v)) >= obs_U_global - PERM_TIE_TOL)

    return null_elem_ge, null_global_ge

def run_element_shuffle_null(
    A_stacked: np.ndarray,
    B_stacked: np.ndarray,
    obs_U_elem: np.ndarray,
    obs_U_global: float,
    n_permutations: int = N_PERMUTATIONS,
    n_workers: int = 8,
    seed: int = SEED,
) -> tuple[np.ndarray, float]:
    """
    Runs n_permutations element-shuffle permutations in blocks of PERM_BLOCK across worker processes. Every block
    has its own child seed spawned from the recorded seed, so the p-values do not depend on n_workers.
    """
    n_blocks = -(-n_permutations // PERM_BLOCK)
    block_sizes = [min(PERM_BLOCK, n_permutations - b * PERM_BLOCK) for b in range(n_blocks)]
    block_seeds = np.random.SeedSequence(seed).spawn(n_blocks)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_run_perm_worker, block_seed, n_block, A_stacked, B_stacked, obs_U_elem, obs_U_global)
            for block_seed, n_block in zip(block_seeds, block_sizes)
        ]
        results = [f.result() for f in futures]

    total_elem_ge: np.ndarray = np.sum([r[0] for r in results], axis=0)
    total_global_ge = sum(r[1] for r in results)

    p_elem: np.ndarray = (1.0 + total_elem_ge) / (1.0 + n_permutations)
    # Elements that are 0/0 in every comparison have no observed U; no null value is >= NaN,
    # so without this they would get the smallest possible p-value
    p_elem[np.isnan(obs_U_elem)] = 1.0
    p_global = (1.0 + total_global_ge) / (1.0 + n_permutations)

    return p_elem, float(p_global)

def run_phase_2(n_permutations: int = N_PERMUTATIONS, n_workers: int = 8, seed: int = SEED):
    start_total = time.time()
    n_subs = len(SUBJECT_IDS)
    pairs = list(combinations(range(n_subs), 2))
    n_pairs = len(pairs)
    p_idx, q_idx = np.array(pairs).T

    # Infant bootstrap (Step 7.2 item 1): pair (a, b) weighted by c_a * c_b in every resample, which equals averaging
    # over all position pairs with d_p != d_q. The same resamples serve every feature, session and statistic.
    counts = bootstrap_counts(bootstrap_draws(n_subs, n_permutations, seed), n_subs)
    W_pairs = counts[:, p_idx] * counts[:, q_idx]

    print("=" * 80)
    print("PHASE 2: UNIVERSALITY & DEVELOPMENTAL STABILITY ANALYSIS (STEP 2)")
    print(f"Cohort: {n_subs} infants | Pairs: {n_pairs} | Permutations: {n_permutations} | Seed: {seed} | Workers: {n_workers}")
    print("=" * 80)

    results = {}

    for fid_idx, (fid, meta) in enumerate(FEATURE_DEFS.items(), 1):
        t0_feat = time.time()
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
            "network": meta["network"],
            "band": meta["band"],
            "seed": seed,
            "n_permutations": n_permutations
        }
        groups = region_groups(ftype)

        session_boot = {}

        for ses_idx, ses in enumerate(SESSIONS):
            # Pre-stack all 171 pairs: (n_pairs, 12, dim)
            A_stacked = X[p_idx, ses_idx]
            B_stacked = X[q_idx, ses_idx]

            # Element Jaccard: min / max
            min_v = np.minimum(A_stacked, B_stacked)
            max_v = np.maximum(A_stacked, B_stacked)

            # Undefined 0/0 tracking
            undef_mask = (max_v == 0)
            undef_frac: np.ndarray = np.asarray(np.mean(undef_mask, axis=(0, 1))) # per element

            with np.errstate(divide="ignore", invalid="ignore"):
                j_elem = np.where(max_v > 0, min_v / max_v, np.nan)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # elements that are 0/0 in every comparison
                # Pair Jaccard per element (nanmean across 12 epochs): shape (171, dim)
                pair_J = np.nanmean(j_elem, axis=1)
                # Element-level Universality U_s(e): mean across 171 pairs
                U_elem = np.nanmean(pair_J, axis=0) # shape (dim,)

            # Save pairwise element Jaccard array: processed_tensors/J_pair_{fid}_{ses}.npy
            np.save(OUTPUT_TENSORS_DIR / f"J_pair_{fid}_{ses}.npy", pair_J)

            # Recording-level Jaccard S, one value per pair: shape (171,)
            pair_S = pair_recording_S(min_v, max_v)
            U_global = float(np.nanmean(pair_S))

            # 19 x 19 symmetric matrix S
            S_mat = np.full((n_subs, n_subs), np.nan, dtype=np.float64)
            S_mat[p_idx, q_idx] = pair_S
            S_mat[q_idx, p_idx] = pair_S
            np.save(OUTPUT_TENSORS_DIR / f"S_mat_{fid}_{ses}.npy", S_mat)

            # Per-infant universality u_s(a)
            u_infant = np.nanmean(S_mat, axis=1)

            # Regional Universality (Step 2 item 4): mean J over the elements of each region for each pair gives a
            # pairs x regions matrix; U_s(g) is its column mean, bootstrapped like U_s (Step 7.2 item 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                J_region = np.column_stack([np.nanmean(pair_J[:, idxs], axis=1) for idxs in groups.values()])
                U_region = np.nanmean(J_region, axis=0)
                SD_region = np.nanstd(J_region, axis=0)
            np.save(OUTPUT_TENSORS_DIR / f"J_region_{fid}_{ses}.npy", J_region)
            boot_region = weighted_nanmean(W_pairs, J_region) # (B, G)
            ci_region = bootstrap_ci(boot_region)

            # Element-shuffle null permutations
            p_elem, p_global = run_element_shuffle_null(
                A_stacked, B_stacked, U_elem, U_global,
                n_permutations=n_permutations, n_workers=n_workers, seed=seed
            )

            # FDR correction across elements
            fdr_results = multipletests(p_elem, method="fdr_bh")
            q_elem: np.ndarray = np.asarray(fdr_results[1])

            # Infant bootstrap of U_s and of every U_s(e), kept for the paired Delta U below
            boot_U = weighted_nanmean(W_pairs, pair_S)
            session_boot[ses] = (boot_U, weighted_nanmean(W_pairs, pair_J))

            feat_res[ses] = {
                "U_global": U_global,
                "SD_pairs": float(np.nanstd(pair_S)),
                "p_global": p_global,
                "bootstrap": boot_summary(boot_U),
                "U_elem": U_elem.tolist(),
                "p_elem": p_elem.tolist(),
                "q_elem": q_elem.tolist(),
                "undef_frac": undef_frac.tolist(),
                "n_sig_elements": int(np.sum(q_elem < 0.05)),
                "u_infant": u_infant.tolist(),
                "regions": list(groups),
                "regional_U": {g: float(U_region[i]) for i, g in enumerate(groups)},
                "regional_U_SD_pairs": {g: float(SD_region[i]) for i, g in enumerate(groups)},
                "regional_U_median": {g: float(np.nanmedian(boot_region[:, i])) for i, g in enumerate(groups)},
                "regional_U_ci_95": {g: [float(ci_region[0, i]), float(ci_region[1, i])] for i, g in enumerate(groups)}
            }

        # -------------------------------------------------------------
        # Developmental shift: ses-2 - ses-1 (Step 6 item 1)
        # -------------------------------------------------------------
        U1 = feat_res["ses-1"]["U_global"]
        U2 = feat_res["ses-2"]["U_global"]
        delta_U_global = U2 - U1
        delta_U_elem = np.array(feat_res["ses-2"]["U_elem"]) - np.array(feat_res["ses-1"]["U_elem"])
        delta_u_infant = np.array(feat_res["ses-2"]["u_infant"]) - np.array(feat_res["ses-1"]["u_infant"])

        # Paired bootstrap (Step 7.2 item 4): both sessions are averaged over the same resampled pairs and the
        # difference is kept, at recording level and for every element
        boot_delta = session_boot["ses-2"][0] - session_boot["ses-1"][0]
        boot_delta_elem = session_boot["ses-2"][1] - session_boot["ses-1"][1]
        ci_d_low, ci_d_high = bootstrap_ci(boot_delta)
        p_delta = float(bootstrap_p(boot_delta))

        # Two-sided bootstrap p per element, FDR-corrected across elements
        p_delta_elem = bootstrap_p(boot_delta_elem)
        q_delta_elem = np.asarray(multipletests(p_delta_elem, method="fdr_bh")[1])

        feat_res["developmental_shift"] = {
            "delta_U_global": float(delta_U_global),
            "median_delta": float(np.nanmedian(boot_delta)),
            "ci_95": [float(ci_d_low), float(ci_d_high)],
            "p_value": p_delta,
            "delta_U_elem": delta_U_elem.tolist(),
            "delta_U_elem_ci_95": bootstrap_ci(boot_delta_elem).T.tolist(),
            "delta_U_elem_p": p_delta_elem.tolist(),
            "delta_U_elem_q": q_delta_elem.tolist(),
            "n_sig_delta_U_elem": int(np.sum(q_delta_elem < 0.05)),
            "delta_u_infant": delta_u_infant.tolist()
        }

        results[fid] = feat_res
        t_feat = time.time() - t0_feat
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:26]:<26}) | U_1: {U1:.4f} (p={feat_res['ses-1']['p_global']:.4f}, sig={feat_res['ses-1']['n_sig_elements']}/{dim}) | U_2: {U2:.4f} (p={feat_res['ses-2']['p_global']:.4f}, sig={feat_res['ses-2']['n_sig_elements']}/{dim}) | Delta U: {delta_U_global:+.4f} (p={p_delta:.4f}, sig elements={feat_res['developmental_shift']['n_sig_delta_U_elem']}) | {t_feat:.1f}s")

    # Save complete results
    out_json = OUTPUT_TENSORS_DIR / "universality_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_total
    print("=" * 80)
    print(f"PHASE 2 COMPLETED SUCCESSFULLY in {total_time:.1f} seconds ({total_time/60:.2f} minutes)!")
    print(f"Full results saved to: {out_json}")
    print("Intermediate arrays saved: J_pair_{fid}_{ses}.npy, J_region_{fid}_{ses}.npy and S_mat_{fid}_{ses}.npy")
    print("=" * 80)
    return results

if __name__ == "__main__":
    run_phase_2()
