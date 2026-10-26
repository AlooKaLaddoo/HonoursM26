"""
Phase 2: Complete Universality Analysis (Step 2 - Comparison A)
Evaluates population-wide shared organization across all 26 feature sets for ses-1 and ses-2.
Computes:
  - Element-level universality U_1(e), U_2(e), and Delta U(e)
  - Recording-level universality U_1, U_2, and Delta U
  - Per-infant universality u_s(a) and Delta u(a)
  - Regional universality (5 node regions, 15 edge region pairs)
  - Element-shuffle null hypothesis permutation tests (10,000 permutations per feature/session)
  - Infant-level bootstrap 95% Confidence Intervals (10,000 resamples)
  - Benjamini-Hochberg FDR correction across elements (q < 0.05)
  - Undefined (0/0) occurrence tracking per element
Saves intermediate pair arrays and summary JSON.
"""
import json
import sys
import time
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
        N_EPOCHS,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        REGION_PAIRS,
        SESSIONS,
        SUBJECT_IDS,
    )
else:
    from .config import (
        EDGE_LABELS,
        FEATURE_DEFS,
        N_EPOCHS,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        REGION_PAIRS,
        SESSIONS,
        SUBJECT_IDS,
    )

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

def infant_pair_bootstrap(S_matrix: np.ndarray, B: int = 10000, seed: int = 42):
    """
    Infant-level bootstrap (Step 7.2) for pair-based statistics.
    Resamples 19 infants with replacement, ignoring self-comparisons.
    """
    rng = np.random.default_rng(seed)
    n_subs = S_matrix.shape[0]
    p, q = np.triu_indices(n_subs, k=1)
    
    boot_vals = np.empty(B, dtype=np.float64)
    for i in range(B):
        d = rng.integers(0, n_subs, n_subs)
        keep = d[p] != d[q]
        boot_vals[i] = np.nanmean(S_matrix[d[p][keep], d[q][keep]])
        
    ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
    return {
        "median": float(np.median(boot_vals)),
        "ci_95": [float(ci_low), float(ci_high)],
        "boot_sd": float(np.std(boot_vals))
    }

def _run_perm_worker(seed: int, n_perms: int, A_stacked: np.ndarray, B_stacked: np.ndarray, dim: int, obs_U_elem: np.ndarray, obs_U_global: float):
    """Worker function for parallel element-shuffle null permutations."""
    rng = np.random.default_rng(seed)
    n_pairs = A_stacked.shape[0]
    
    null_elem_ge = np.zeros(dim, dtype=np.int32)
    null_global_vals = np.zeros(n_perms, dtype=np.float64)
    
    for i in range(n_perms):
        # Generate random permutation per pair
        perms = np.array([rng.permutation(dim) for _ in range(n_pairs)]) # (n_pairs, dim)
        B_perm = np.take_along_axis(B_stacked, perms[:, np.newaxis, :], axis=2) # (n_pairs, 12, dim)
        
        min_v = np.minimum(A_stacked, B_perm)
        max_v = np.maximum(A_stacked, B_perm)
        
        # Element Jaccard: (n_pairs, 12, dim)
        with np.errstate(divide="ignore", invalid="ignore"):
            j_elem = np.where(max_v > 0, min_v / max_v, np.nan)
        null_U_e = np.nanmean(j_elem, axis=(0, 1))
        null_elem_ge += (null_U_e >= obs_U_elem).astype(np.int32)
        
        # Recording Jaccard: sum over elements
        num = np.sum(min_v, axis=2)
        den = np.sum(max_v, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_pair = np.where(den > 0, num / den, 0.0)
        null_global_vals[i] = float(np.mean(s_pair))
        
    return null_elem_ge, null_global_vals

def run_element_shuffle_null(
    A_stacked: np.ndarray,
    B_stacked: np.ndarray,
    dim: int,
    obs_U_elem: np.ndarray,
    obs_U_global: float,
    n_permutations: int = 10000,
    n_workers: int = 8,
) -> tuple[np.ndarray, float]:
    """Runs 10,000 element-shuffle permutations across multiple worker processes."""
    perms_per_worker = n_permutations // n_workers
    remainder = n_permutations % n_workers
    
    worker_perms = [perms_per_worker + (1 if w < remainder else 0) for w in range(n_workers)]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_run_perm_worker, 1000 + w * 37, worker_perms[w], A_stacked, B_stacked, dim, obs_U_elem, obs_U_global)
            for w in range(n_workers)
        ]
        results = [f.result() for f in futures]
        
    total_elem_ge: np.ndarray = np.sum([r[0] for r in results], axis=0)
    all_null_globals = np.concatenate([r[1] for r in results])
    
    p_elem: np.ndarray = (1.0 + total_elem_ge) / (1.0 + n_permutations)
    p_global = (1.0 + np.sum(all_null_globals >= obs_U_global)) / (1.0 + n_permutations)
    
    return p_elem, float(p_global)

def run_phase_2(n_permutations: int = 10000, n_workers: int = 8):
    start_total = time.time()
    n_subs = len(SUBJECT_IDS)
    pairs = list(combinations(range(n_subs), 2))
    n_pairs = len(pairs)
    
    print("=" * 80)
    print("PHASE 2: UNIVERSALITY & DEVELOPMENTAL STABILITY ANALYSIS (STEP 2)")
    print(f"Cohort: {n_subs} infants | Pairs: {n_pairs} | Permutations: {n_permutations} | Workers: {n_workers}")
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
            "band": meta["band"]
        }
        
        session_S_mats = []
        
        for ses_idx, ses in enumerate(SESSIONS):
            # Pre-stack all 171 pairs
            A_stacked = np.zeros((n_pairs, N_EPOCHS, dim), dtype=np.float64)
            B_stacked = np.zeros((n_pairs, N_EPOCHS, dim), dtype=np.float64)
            
            for p_idx, (a, b) in enumerate(pairs):
                A_stacked[p_idx, :, :] = X[a, ses_idx, :, :]
                B_stacked[p_idx, :, :] = X[b, ses_idx, :, :]
                
            # Element Jaccard: min / max
            min_v = np.minimum(A_stacked, B_stacked)
            max_v = np.maximum(A_stacked, B_stacked)
            
            # Undefined 0/0 tracking
            undef_mask = (max_v == 0)
            undef_frac: np.ndarray = np.asarray(np.mean(undef_mask, axis=(0, 1))) # per element
            
            with np.errstate(divide="ignore", invalid="ignore"):
                j_elem = np.where(max_v > 0, min_v / max_v, np.nan)
                
            # Pair Jaccard per element (nanmean across 12 epochs): shape (171, dim)
            pair_J = np.nanmean(j_elem, axis=1)
            
            # Save pairwise element Jaccard array: processed_tensors/J_pair_{fid}_{ses}.npy
            np.save(OUTPUT_TENSORS_DIR / f"J_pair_{fid}_{ses}.npy", pair_J)
            
            # Element-level Universality U_s(e): mean across 171 pairs
            U_elem = np.nanmean(pair_J, axis=0) # shape (dim,)
            
            # Recording-level Jaccard S
            num = np.sum(min_v, axis=2)
            den = np.sum(max_v, axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_epochs = np.where(den > 0, num / den, 0.0)
            pair_S = np.mean(s_epochs, axis=1) # shape (171,)
            U_global = float(np.mean(pair_S))
            
            # 19 x 19 symmetric matrix S
            S_mat = np.full((n_subs, n_subs), np.nan, dtype=np.float64)
            for p_idx, (a, b) in enumerate(pairs):
                S_mat[a, b] = pair_S[p_idx]
                S_mat[b, a] = pair_S[p_idx]
            session_S_mats.append(S_mat)
            np.save(OUTPUT_TENSORS_DIR / f"S_mat_{fid}_{ses}.npy", S_mat)
            
            # Per-infant universality u_s(a)
            u_infant = np.nanmean(S_mat, axis=1)
            
            # Regional Universality
            reg_U = {}
            if ftype == "node":
                for reg in ["F", "C", "T", "P", "O"]:
                    idxs = [i for i, n in enumerate(NODE_LABELS) if get_node_region(n) == reg]
                    reg_U[reg] = float(np.nanmean(U_elem[idxs]))
            else:
                for rp in REGION_PAIRS:
                    idxs = [i for i, e in enumerate(EDGE_LABELS) if get_edge_region_pair(e) == rp]
                    if idxs:
                        reg_U[rp] = float(np.nanmean(U_elem[idxs]))
                        
            # Element-shuffle null permutations (10,000)
            p_elem, p_global = run_element_shuffle_null(
                A_stacked, B_stacked, dim, U_elem, U_global,
                n_permutations=n_permutations, n_workers=n_workers
            )
            
            # FDR correction across elements
            fdr_results = multipletests(p_elem, method="fdr_bh")
            q_elem: np.ndarray = np.asarray(fdr_results[1])
            
            # Infant bootstrap CI
            boot_stats = infant_pair_bootstrap(S_mat, B=10000, seed=42 + ses_idx)
            
            feat_res[ses] = {
                "U_global": U_global,
                "SD_pairs": float(np.std(pair_S)),
                "p_global": p_global,
                "bootstrap": boot_stats,
                "U_elem": U_elem.tolist(),
                "p_elem": p_elem.tolist(),
                "q_elem": q_elem.tolist(),
                "undef_frac": undef_frac.tolist(),
                "n_sig_elements": int(np.sum(q_elem < 0.05)),
                "u_infant": u_infant.tolist(),
                "regional_U": reg_U
            }
            
        # -------------------------------------------------------------
        # Developmental shift: ses-2 - ses-1
        # -------------------------------------------------------------
        U1 = feat_res["ses-1"]["U_global"]
        U2 = feat_res["ses-2"]["U_global"]
        delta_U_global = U2 - U1
        delta_U_elem = np.array(feat_res["ses-2"]["U_elem"]) - np.array(feat_res["ses-1"]["U_elem"])
        delta_u_infant = np.array(feat_res["ses-2"]["u_infant"]) - np.array(feat_res["ses-1"]["u_infant"])
        
        # Paired bootstrap for Delta U
        # For each bootstrap resample, compute difference between ses-2 mean and ses-1 mean
        rng_d = np.random.default_rng(999)
        p_idx, q_idx = np.triu_indices(n_subs, k=1)
        boot_delta = np.empty(10000, dtype=np.float64)
        S1, S2 = session_S_mats[0], session_S_mats[1]
        for b_i in range(10000):
            d = rng_d.integers(0, n_subs, n_subs)
            keep = d[p_idx] != d[q_idx]
            b_u1 = np.nanmean(S1[d[p_idx][keep], d[q_idx][keep]])
            b_u2 = np.nanmean(S2[d[p_idx][keep], d[q_idx][keep]])
            boot_delta[b_i] = b_u2 - b_u1
        ci_d_low, ci_d_high = np.percentile(boot_delta, [2.5, 97.5])
        
        # Two-sided bootstrap p-value for Delta U != 0
        p_delta = 2.0 * min(np.mean(boot_delta <= 0), np.mean(boot_delta >= 0))
        p_delta = max(p_delta, 1.0 / 10000)
        
        feat_res["developmental_shift"] = {
            "delta_U_global": float(delta_U_global),
            "median_delta": float(np.median(boot_delta)),
            "ci_95": [float(ci_d_low), float(ci_d_high)],
            "p_value": float(p_delta),
            "delta_U_elem": delta_U_elem.tolist(),
            "delta_u_infant": delta_u_infant.tolist()
        }
        
        results[fid] = feat_res
        t_feat = time.time() - t0_feat
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:26]:<26}) | U_1: {U1:.4f} (p={feat_res['ses-1']['p_global']:.4f}, sig={feat_res['ses-1']['n_sig_elements']}/{dim}) | U_2: {U2:.4f} (p={feat_res['ses-2']['p_global']:.4f}, sig={feat_res['ses-2']['n_sig_elements']}/{dim}) | Delta U: {delta_U_global:+.4f} (p={p_delta:.4f}) | {t_feat:.1f}s")
        
    # Save complete results
    out_json = OUTPUT_TENSORS_DIR / "universality_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    total_time = time.time() - start_total
    print("=" * 80)
    print(f"PHASE 2 COMPLETED SUCCESSFULLY in {total_time:.1f} seconds ({total_time/60:.2f} minutes)!")
    print(f"Full results saved to: {out_json}")
    print("Intermediate arrays saved: J_pair_{fid}_{ses}.npy and S_mat_{fid}_{ses}.npy")
    print("=" * 80)
    return results

if __name__ == "__main__":
    run_phase_2(n_permutations=10000, n_workers=8)
