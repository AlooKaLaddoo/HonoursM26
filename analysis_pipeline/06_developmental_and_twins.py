"""
Phase 6: Developmental Trajectories, Mantel Tests, & Sibling/Twin Analysis (Step 6)
Computes:
  - Mantel matrix correlation tests between pairwise similarity S and age gap |Delta age| (10,000 permutations)
  - Correlation between self-similarity M[a, a] / idiff(a) and inter-session time gap (in days)
  - Individual reports for short-gap infant sub-NORB00064 (4 days) and sub-NORB00069 (42 days)
  - Complete Sibling/Twin Analysis (Table T6) for male twin pair sub-NORB00086 and sub-NORB00087:
    * Sibling rank among all 171 pairs
    * Sibling rank among matched control sets (same-infant pairs & same-age pairs)
    * Cross-session cross-identification check (M[86, 87] vs M[86, 86])
    * Shared elements exceeding 95th percentile of non-sibling pairs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Support both direct script execution and package imports
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from analysis_pipeline.config import (
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        NODE_LABELS,
        OUTPUT_TENSORS_DIR,
        SESSIONS,
        SUBJECT_IDS,
    )
else:
    from .config import (
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        NODE_LABELS,
        OUTPUT_TENSORS_DIR,
        SESSIONS,
        SUBJECT_IDS,
    )


def load_exact_subject_ages():
    """Load exact infant ages (years and days) at ses-1 and ses-2."""
    ages_yr = {s: {} for s in SUBJECT_IDS}
    ages_days = {s: {} for s in SUBJECT_IDS}
    for subj in SUBJECT_IDS:
        for ses in SESSIONS:
            scan_file = DATASET_PATH / subj / ses / f"{subj}_{ses}_scans.tsv"
            if scan_file.exists():
                df = pd.read_csv(scan_file, sep="\t")
                val = float(df["age_acq_time"].iloc[0])
                ages_yr[subj][ses] = val
                ages_days[subj][ses] = val * 365.25
            else:
                ages_yr[subj][ses] = np.nan
                ages_days[subj][ses] = np.nan
    return ages_yr, ages_days


def mantel_test(S_mat: np.ndarray, D_mat: np.ndarray, n_permutations: int = 10000, seed: int = 42):
    """
    Mantel test (Step 6.3):
    Spearman correlation between 171 upper-triangle entries of S and D.
    Permutes infant rows and columns of D simultaneously 10,000 times.
    Uses precomputed standardized ranks for bit-exact Spearman correlation at ~10x speed.
    """
    rng = np.random.default_rng(seed)
    n = S_mat.shape[0]
    p, q = np.triu_indices(n, k=1)
    n_pairs = len(p)
    
    s_vec = S_mat[p, q]
    d_vec = D_mat[p, q]
    
    rx = stats.rankdata(s_vec)
    ry = stats.rankdata(d_vec)
    rx_std = (rx - np.mean(rx)) / (np.std(rx) * n_pairs)
    ry_std = (ry - np.mean(ry)) / np.std(ry)
    obs_rho = float(np.dot(rx_std, ry_std))
    
    null_rhos = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        d_perm_vec = D_mat[np.ix_(perm, perm)][p, q]
        ry_perm = stats.rankdata(d_perm_vec)
        ry_perm_std = (ry_perm - np.mean(ry_perm)) / np.std(ry_perm)
        null_rhos[i] = np.dot(rx_std, ry_perm_std)
        
    p_two_sided = float((1.0 + np.sum(np.abs(null_rhos) >= np.abs(obs_rho))) / (1.0 + n_permutations))
    return obs_rho, p_two_sided


def perm_spearman(x: np.ndarray, y: np.ndarray, n_permutations: int = 10000, seed: int = 42):
    """
    Spearman rank correlation with 10,000 permutations of vector y (Step 6.2).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx_std = (rx - np.mean(rx)) / (np.std(rx) * n)
    ry_std = (ry - np.mean(ry)) / np.std(ry)
    obs_rho = float(np.dot(rx_std, ry_std))
    
    null_rhos = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm_ry = rng.permutation(ry_std)
        null_rhos[i] = np.dot(rx_std, perm_ry)
        
    p_two_sided = float((1.0 + np.sum(np.abs(null_rhos) >= np.abs(obs_rho))) / (1.0 + n_permutations))
    return obs_rho, p_two_sided


def run_developmental_and_twins(n_permutations: int = 10000):
    print("=" * 80)
    print("PHASE 6: DEVELOPMENTAL ANALYSES & TWIN/SIBLING AUDIT (STEP 6)")
    print("=" * 80)
    
    _ages_yr, ages_days = load_exact_subject_ages()
    n_subs = len(SUBJECT_IDS)
    p_triu, q_triu = np.triu_indices(n_subs, k=1)
    
    # Inter-session interval in days per infant
    intervals_days = np.array([ages_days[s]["ses-2"] - ages_days[s]["ses-1"] for s in SUBJECT_IDS])
    ages_ses1_days = np.array([ages_days[s]["ses-1"] for s in SUBJECT_IDS])
    
    # Age difference matrices D[a, b] for ses-1 and ses-2 (in days)
    D_mats = {}
    for ses in SESSIONS:
        d_mat = np.zeros((n_subs, n_subs), dtype=np.float64)
        for i in range(n_subs):
            for j in range(n_subs):
                d_mat[i, j] = abs(ages_days[SUBJECT_IDS[i]][ses] - ages_days[SUBJECT_IDS[j]][ses])
        D_mats[ses] = d_mat
        
    idx_86 = SUBJECT_IDS.index("sub-NORB00086")
    idx_87 = SUBJECT_IDS.index("sub-NORB00087")
    idx_64 = SUBJECT_IDS.index("sub-NORB00064")
    idx_69 = SUBJECT_IDS.index("sub-NORB00069")
    
    # Load Universality results for developmental shift delta_U
    u_file = OUTPUT_TENSORS_DIR / "universality_results.json"
    if not u_file.exists():
        raise FileNotFoundError(f"Prerequisite file not found: {u_file}. Run Phase 2 first.")
    with open(u_file) as f:
        u_results = json.load(f)
        
    dev_results = {}
    twin_results = {}
    
    for fid_idx, (fid, meta) in enumerate(FEATURE_DEFS.items(), 1):
        fname = meta["name"]
        dim = meta["dim"]
        ftype = meta["type"]
        labels = NODE_LABELS if ftype == "node" else EDGE_LABELS
        
        # Load matrices
        s1_path = OUTPUT_TENSORS_DIR / f"S_mat_{fid}_ses-1.npy"
        s2_path = OUTPUT_TENSORS_DIR / f"S_mat_{fid}_ses-2.npy"
        m_path = OUTPUT_TENSORS_DIR / f"M_mat_{fid}.npy"
        j1_path = OUTPUT_TENSORS_DIR / f"J_pair_{fid}_ses-1.npy"
        j2_path = OUTPUT_TENSORS_DIR / f"J_pair_{fid}_ses-2.npy"
        
        if not s1_path.exists():
            raise FileNotFoundError(
                f"Prerequisite matrix not found: {s1_path}. Run Phase 2 and Phase 4 first."
            )
        
        S1 = np.load(s1_path)
        S2 = np.load(s2_path)
        M = np.load(m_path)
        J1_pairs = np.load(j1_path)
        J2_pairs = np.load(j2_path)
        
        # 1. Mantel test per session
        rho_m1, p_m1 = mantel_test(S1, D_mats["ses-1"], n_permutations=n_permutations, seed=101 + fid_idx)
        rho_m2, p_m2 = mantel_test(S2, D_mats["ses-2"], n_permutations=n_permutations, seed=102 + fid_idx)
        
        # 2. Correlation between self-similarity / idiff and inter-session interval
        self_vals = np.diag(M)
        others_vals = np.zeros(n_subs, dtype=np.float64)
        for a in range(n_subs):
            row_off = [M[a, b] for b in range(n_subs) if b != a]
            col_off = [M[b, a] for b in range(n_subs) if b != a]
            others_vals[a] = np.mean(row_off + col_off)
        idiff_vals = (self_vals - others_vals) * 100.0
        
        rho_self_int, p_self_int = perm_spearman(self_vals, intervals_days, n_permutations=n_permutations, seed=201 + fid_idx)
        rho_idiff_int, p_idiff_int = perm_spearman(idiff_vals, intervals_days, n_permutations=n_permutations, seed=202 + fid_idx)
        rho_self_age1, p_self_age1 = perm_spearman(self_vals, ages_ses1_days, n_permutations=n_permutations, seed=203 + fid_idx)
        
        u_shift = u_results[fid]["developmental_shift"]
        dev_results[fid] = {
            "fid": fid, "name": fname,
            "delta_U_global": u_shift["delta_U_global"],
            "median_delta": u_shift["median_delta"],
            "ci_95": u_shift["ci_95"],
            "p_value": u_shift["p_value"],
            "delta_U_elem": u_shift["delta_U_elem"],
            "delta_u_infant": u_shift["delta_u_infant"],
            "mantel_ses1": {"rho": rho_m1, "p": p_m1},
            "mantel_ses2": {"rho": rho_m2, "p": p_m2},
            "interval_corr": {
                "self_vs_interval": {"rho": float(rho_self_int), "p": float(p_self_int)},
                "idiff_vs_interval": {"rho": float(rho_idiff_int), "p": float(p_idiff_int)},
                "self_vs_age1": {"rho": float(rho_self_age1), "p": float(p_self_age1)}
            },
            "individual_infants": {
                "sub-NORB00064": {
                    "gap_days": float(intervals_days[idx_64]),
                    "self": float(self_vals[idx_64]),
                    "idiff": float(idiff_vals[idx_64])
                },
                "sub-NORB00069": {
                    "gap_days": float(intervals_days[idx_69]),
                    "self": float(self_vals[idx_69]),
                    "idiff": float(idiff_vals[idx_69])
                }
            }
        }
        
        # -------------------------------------------------------------
        # 3. SIBLING / TWIN ANALYSIS (Table T6)
        # -------------------------------------------------------------
        # Pair index for (86, 87)
        # All 171 upper-triangle values
        pairs_list = [(p_triu[k], q_triu[k]) for k in range(len(p_triu))]
        twin_pair_idx = pairs_list.index((idx_86, idx_87))
        
        # Sibling similarity in ses-1 and ses-2
        s_twin_1 = float(S1[idx_86, idx_87])
        s_twin_2 = float(S2[idx_86, idx_87])
        
        all_s1 = S1[p_triu, q_triu]
        all_s2 = S2[p_triu, q_triu]
        
        # Percentile rank among all 171 pairs (0 to 100%)
        rank_all_1 = float(stats.percentileofscore(all_s1, s_twin_1))
        rank_all_2 = float(stats.percentileofscore(all_s2, s_twin_2))
        
        # Matched control set 1: 34 other pairs involving 86 or 87
        matched_infant_pairs_idx = [
            k for k, (a, b) in enumerate(pairs_list)
            if (a in (idx_86, idx_87) or b in (idx_86, idx_87)) and k != twin_pair_idx
        ]
        s1_matched_infant = all_s1[matched_infant_pairs_idx]
        s2_matched_infant = all_s2[matched_infant_pairs_idx]
        rank_matched_infant_1 = float(stats.percentileofscore(s1_matched_infant, s_twin_1))
        rank_matched_infant_2 = float(stats.percentileofscore(s2_matched_infant, s_twin_2))
        
        # Matched control set 2: non-sibling pairs with |Delta age| <= 7 days
        matched_age_pairs_idx_1 = [
            k for k, (a, b) in enumerate(pairs_list)
            if D_mats["ses-1"][a, b] <= 7.0 and k != twin_pair_idx
        ]
        s1_matched_age = all_s1[matched_age_pairs_idx_1] if matched_age_pairs_idx_1 else np.array([])
        rank_matched_age_1 = float(stats.percentileofscore(s1_matched_age, s_twin_1)) if len(s1_matched_age) > 0 else np.nan

        matched_age_pairs_idx_2 = [
            k for k, (a, b) in enumerate(pairs_list)
            if D_mats["ses-2"][a, b] <= 7.0 and k != twin_pair_idx
        ]
        s2_matched_age = all_s2[matched_age_pairs_idx_2] if matched_age_pairs_idx_2 else np.array([])
        rank_matched_age_2 = float(stats.percentileofscore(s2_matched_age, s_twin_2)) if len(s2_matched_age) > 0 else np.nan
        
        # Cross-session cross-identification:
        # Check M[86, 87] and M[87, 86]
        m_86_86 = float(M[idx_86, idx_86])
        m_87_87 = float(M[idx_87, idx_87])
        m_86_87 = float(M[idx_86, idx_87])
        m_87_86 = float(M[idx_87, idx_86])
        
        # Rank of sibling within row and column
        row_86 = M[idx_86, :]
        rank_86_in_row = int(np.sum(row_86 > m_86_87) + 1) # 1 = highest
        col_87 = M[:, idx_87]
        rank_87_in_col = int(np.sum(col_87 > m_86_87) + 1)
        
        row_87 = M[idx_87, :]
        rank_87_in_row = int(np.sum(row_87 > m_87_86) + 1)
        col_86 = M[:, idx_86]
        rank_86_in_col = int(np.sum(col_86 > m_87_86) + 1)
        
        # Check if either sibling is misidentified as the other
        misidentified_86_as_87 = bool(np.argmax(row_86) == idx_87)
        misidentified_87_as_86 = bool(np.argmax(row_87) == idx_86)
        
        # Shared elements exceeding 95th percentile of non-sibling pairs in BOTH sessions
        non_twin_idxs = [k for k in range(171) if k != twin_pair_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            j1_nontwin_p95 = np.nanpercentile(J1_pairs[non_twin_idxs, :], 95, axis=0)
            j2_nontwin_p95 = np.nanpercentile(J2_pairs[non_twin_idxs, :], 95, axis=0)
        
        j1_twin = J1_pairs[twin_pair_idx, :]
        j2_twin = J2_pairs[twin_pair_idx, :]
        
        shared_elements = []
        for e in range(dim):
            v1, v2 = j1_twin[e], j2_twin[e]
            t1, t2 = j1_nontwin_p95[e], j2_nontwin_p95[e]
            if not (np.isnan(v1) or np.isnan(v2) or np.isnan(t1) or np.isnan(t2)) and v1 > t1 and v2 > t2:
                shared_elements.append(labels[e])
        
        twin_results[fid] = {
            "fid": fid, "name": fname,
            "ses1_similarity": s_twin_1,
            "ses2_similarity": s_twin_2,
            "rank_among_171_ses1": rank_all_1,
            "rank_among_171_ses2": rank_all_2,
            "rank_matched_infant_ses1": rank_matched_infant_1,
            "rank_matched_infant_ses2": rank_matched_infant_2,
            "rank_matched_age_ses1": rank_matched_age_1,
            "rank_matched_age_ses2": rank_matched_age_2,
            "cross_identification": {
                "M_86_86": m_86_86, "M_87_87": m_87_87,
                "M_86_87": m_86_87, "M_87_86": m_87_86,
                "rank_87_for_86": rank_86_in_row,
                "rank_86_for_87": rank_87_in_row,
                "rank_87_in_col": rank_87_in_col,
                "rank_86_in_col": rank_86_in_col,
                "misidentified_86_as_87": misidentified_86_as_87,
                "misidentified_87_as_86": misidentified_87_as_86
            },
            "n_shared_p95_elements": len(shared_elements),
            "shared_elements": shared_elements
        }
        
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:25]:<25}) | Mantel s1: rho={rho_m1:+.3f} (p={p_m1:.4f}), s2: rho={rho_m2:+.3f} (p={p_m2:.4f}) | Twin Rank: s1={rank_all_1:4.1f}%, s2={rank_all_2:4.1f}% | Shared P95: {len(shared_elements):<2}/{dim}")
        
    out_dev = OUTPUT_TENSORS_DIR / "developmental_results.json"
    out_twin = OUTPUT_TENSORS_DIR / "twin_results.json"
    with open(out_dev, "w") as f:
        json.dump(dev_results, f, indent=2)
    with open(out_twin, "w") as f:
        json.dump(twin_results, f, indent=2)
        
    print("=" * 80)
    print("Developmental & Twin/Sibling results successfully computed!")
    print(f"Developmental: {out_dev}")
    print(f"Twin (Table T6): {out_twin}")
    print("=" * 80)


if __name__ == "__main__":
    run_developmental_and_twins()
