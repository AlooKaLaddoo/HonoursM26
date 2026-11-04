"""
Phase 6: Developmental Trajectories, Mantel Tests, & Sibling/Twin Analysis (Step 6)
Computes:
  - Mantel matrix correlation tests between pairwise similarity S and age gap |Delta age| (200,000 permutations by default)
  - Correlation between self-similarity M[a, a] / idiff(a) and inter-session time gap and age at ses-1 (in days)
  - A 95% infant bootstrap CI for every correlation (Step 7.5): pair bootstrap for the Mantel rho, ordinary
    bootstrap of the 19 infants for the interval and age correlations
  - Mean |Delta age| over pairs at each session, reported next to Delta U (Step 6 item 1 caveat)
  - BH-FDR of every global test across the 26 feature sets (Step 7.4 ii); one seed for all permutations and resamples
  - Individual reports for short-gap infant sub-NORB00064 (4 days) and sub-NORB00069 (42 days)
  - Complete Sibling/Twin Analysis (Table T6) for male twin pair sub-NORB00086 and sub-NORB00087,
    at recording and element level:
    * Sibling rank among all 171 pairs
    * Sibling rank among matched control sets (same-infant pairs & same-age pairs), with their median and range
    * Cross-session cross-identification check (M[86, 87] vs M[86, 86]) in both directions
    * Shared elements exceeding 95th percentile of non-sibling pairs, and how many are "individual" (Step 5a)
    * Same-day control: not determinable (recording dates are anonymised in the dataset)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Support both direct script execution and package imports
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from analysis_pipeline.config import (
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        N_PERMUTATIONS,
        PERM_TIE_TOL,
        NODE_LABELS,
        OUTPUT_TENSORS_DIR,
        SEED,
        SESSIONS,
        SUBJECT_IDS,
    )
    from analysis_pipeline.utils import bootstrap_ci, bootstrap_draws
else:
    from .config import (
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        N_PERMUTATIONS,
        PERM_TIE_TOL,
        NODE_LABELS,
        OUTPUT_TENSORS_DIR,
        SEED,
        SESSIONS,
        SUBJECT_IDS,
    )
    from .utils import bootstrap_ci, bootstrap_draws

# Sibling item 5: scans.tsv holds no acquisition date and every EDF header shows the anonymised date 2023-01-01
SAME_DAY_CONTROL_NOTE = (
    "Not determinable: scans.tsv has no acquisition date and the EDF recording dates are anonymised "
    "(all 2023-01-01), so shared recording days cannot be identified."
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


def mantel_test(S_mat: np.ndarray, D_mat: np.ndarray, n_permutations: int = N_PERMUTATIONS, seed: int = SEED):
    """
    Mantel test (Step 6.3):
    Spearman correlation between 171 upper-triangle entries of S and D.
    Permutes infant rows and columns of D simultaneously n_permutations times.
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

    p_two_sided = float((1.0 + np.sum(np.abs(null_rhos) >= np.abs(obs_rho) - PERM_TIE_TOL)) / (1.0 + n_permutations))
    return obs_rho, p_two_sided


def perm_spearman(x: np.ndarray, y: np.ndarray, n_permutations: int = N_PERMUTATIONS, seed: int = SEED):
    """
    Spearman rank correlation with n_permutations permutations of vector y (Step 6.2).
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

    p_two_sided = float((1.0 + np.sum(np.abs(null_rhos) >= np.abs(obs_rho) - PERM_TIE_TOL)) / (1.0 + n_permutations))
    return obs_rho, p_two_sided


def spearman_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman rho between each row of X and the same row of Y, over the entries defined (not NaN) in both."""
    both = ~(np.isnan(X) | np.isnan(Y))
    rx = stats.rankdata(np.where(both, X, np.nan), axis=1, nan_policy="omit")
    ry = stats.rankdata(np.where(both, Y, np.nan), axis=1, nan_policy="omit")
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a resample without variation gives NaN
        rx = rx - np.nanmean(rx, axis=1, keepdims=True)
        ry = ry - np.nanmean(ry, axis=1, keepdims=True)
        return np.nansum(rx * ry, axis=1) / np.sqrt(np.nansum(rx ** 2, axis=1) * np.nansum(ry ** 2, axis=1))


def mantel_bootstrap(S_mat: np.ndarray, D_mat: np.ndarray, draws: np.ndarray, chunk: int = 10000) -> np.ndarray:
    """
    Mantel rho in every infant resample (pair-based statistic, Step 7.2 item 1): the pairs are all position pairs
    p < q of the draw with d_p != d_q, so a duplicated infant is never compared with itself.
    """
    p, q = np.triu_indices(S_mat.shape[0], k=1)
    rhos = []
    for start in range(0, len(draws), chunk):
        dp, dq = draws[start:start + chunk, p], draws[start:start + chunk, q]
        keep = dp != dq
        rhos.append(spearman_rows(np.where(keep, S_mat[dp, dq], np.nan), np.where(keep, D_mat[dp, dq], np.nan)))
    return np.concatenate(rhos)


def corr_result(rho: float, p: float, boot: np.ndarray) -> dict:
    """A correlation as rho with its 95% infant bootstrap CI (Step 7.5) and its permutation p."""
    ci_low, ci_high = bootstrap_ci(boot)
    return {"rho": float(rho), "ci_95": [float(ci_low), float(ci_high)], "p": float(p)}


def pct_rank(values: np.ndarray, x: float) -> float:
    """Percentile rank (0-100) of x among values, ignoring NaN; NaN if undefined."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if np.isnan(x) or len(values) == 0:
        return np.nan
    return float(stats.percentileofscore(values, x))


def set_summary(values: np.ndarray) -> dict:
    """Size, median and range of a comparison set (sibling item 2), ignoring NaN."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"n": 0, "median": np.nan, "min": np.nan, "max": np.nan}
    return {"n": int(len(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def identified_as(scores: np.ndarray, target: int) -> bool:
    """True if target holds the unique maximum of scores; a tie counts as no identification (Step 4 item 4c)."""
    if np.isnan(scores[target]):
        return False
    best = np.nanmax(scores)
    return bool(scores[target] == best and np.sum(scores == best) == 1)


def cross_identification(M: np.ndarray, i: int, j: int) -> dict:
    """
    Sibling item 3 for one matrix M (rows ses-1, columns ses-2) and siblings i (86) and j (87):
    M[i, j] and M[j, i] next to M[i, i] and M[j, j], the rank of each sibling value within its row and
    column (1 = highest), and whether either sibling is identified as the other in either direction
    (row: ses-1 -> ses-2, column: ses-2 -> ses-1).
    """
    def rank(vec, x):
        return int(np.sum(vec > x) + 1) if not np.isnan(x) else None

    fwd_86 = identified_as(M[i, :], j)  # 86 at ses-1 matched to 87 at ses-2
    bwd_86 = identified_as(M[:, i], j)  # 86 at ses-2 matched to 87 at ses-1
    fwd_87 = identified_as(M[j, :], i)
    bwd_87 = identified_as(M[:, j], i)
    return {
        "M_86_86": float(M[i, i]), "M_87_87": float(M[j, j]),
        "M_86_87": float(M[i, j]), "M_87_86": float(M[j, i]),
        "rank_87_for_86": rank(M[i, :], M[i, j]),
        "rank_86_for_87": rank(M[j, :], M[j, i]),
        "rank_87_in_col": rank(M[:, j], M[i, j]),
        "rank_86_in_col": rank(M[:, i], M[j, i]),
        "misidentified_86_as_87": fwd_86 or bwd_86,
        "misidentified_87_as_86": fwd_87 or bwd_87,
        "misidentified_86_as_87_fwd": fwd_86, "misidentified_86_as_87_bwd": bwd_86,
        "misidentified_87_as_86_fwd": fwd_87, "misidentified_87_as_86_bwd": bwd_87,
    }


def run_developmental_and_twins(n_permutations: int = N_PERMUTATIONS, seed: int = SEED):
    print("=" * 80)
    print("PHASE 6: DEVELOPMENTAL ANALYSES & TWIN/SIBLING AUDIT (STEP 6)")
    print(f"Permutations and bootstrap resamples: {n_permutations} | Seed: {seed}")
    print("=" * 80)

    _ages_yr, ages_days = load_exact_subject_ages()
    n_subs = len(SUBJECT_IDS)
    p_triu, q_triu = np.triu_indices(n_subs, k=1)
    # The infant resamples of every bootstrap in the pipeline (Step 0 item 8, Step 7.2)
    draws = bootstrap_draws(n_subs, n_permutations, seed)

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
    # Mean |Delta age| over all pairs at each session, reported next to Delta U (Step 6 item 1 caveat)
    mean_abs_dage = {ses: float(np.mean(D_mats[ses][p_triu, q_triu])) for ses in SESSIONS}

    idx_86 = SUBJECT_IDS.index("sub-NORB00086")
    idx_87 = SUBJECT_IDS.index("sub-NORB00087")
    idx_64 = SUBJECT_IDS.index("sub-NORB00064")
    idx_69 = SUBJECT_IDS.index("sub-NORB00069")

    # Load Universality results for developmental shift delta_U, and the Step 5a taxonomy
    u_file = OUTPUT_TENSORS_DIR / "universality_results.json"
    tax_file = OUTPUT_TENSORS_DIR / "taxonomy_results.json"
    if not u_file.exists():
        raise FileNotFoundError(f"Prerequisite file not found: {u_file}. Run Phase 2 first.")
    if not tax_file.exists():
        raise FileNotFoundError(f"Prerequisite file not found: {tax_file}. Run Phase 5 first.")
    with open(u_file) as f:
        u_results = json.load(f)
    with open(tax_file) as f:
        tax_results = json.load(f)

    # Sibling pair index and matched control sets (shared by all features)
    pairs_list = [(p_triu[k], q_triu[k]) for k in range(len(p_triu))]
    twin_pair_idx = pairs_list.index((idx_86, idx_87))
    non_twin_idxs = [k for k in range(len(pairs_list)) if k != twin_pair_idx]
    # Matched control set 1: the other pairs involving 86 or 87 (34 for N = 19)
    matched_infant_idx = [
        k for k, (a, b) in enumerate(pairs_list)
        if (a in (idx_86, idx_87) or b in (idx_86, idx_87)) and k != twin_pair_idx
    ]
    # Matched control set 2: non-sibling pairs with |Delta age| <= 7 days at that session
    matched_age_idx = {
        ses: [k for k, (a, b) in enumerate(pairs_list) if D_mats[ses][a, b] <= 7.0 and k != twin_pair_idx]
        for ses in SESSIONS
    }
    print(f"Sibling item 5 (same-day control): {SAME_DAY_CONTROL_NOTE}")

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
        m_elem_path = OUTPUT_TENSORS_DIR / f"M_elem_{fid}.npy"
        j1_path = OUTPUT_TENSORS_DIR / f"J_pair_{fid}_ses-1.npy"
        j2_path = OUTPUT_TENSORS_DIR / f"J_pair_{fid}_ses-2.npy"

        if not s1_path.exists():
            raise FileNotFoundError(
                f"Prerequisite matrix not found: {s1_path}. Run Phase 2 and Phase 4 first."
            )

        S1 = np.load(s1_path)
        S2 = np.load(s2_path)
        M = np.load(m_path)
        M_elem = np.load(m_elem_path) # (19, 19, dim)
        J1_pairs = np.load(j1_path)
        J2_pairs = np.load(j2_path)

        # 1. Mantel test per session (one seed for all permutation tests, Step 0 item 8), with an infant pair
        #    bootstrap CI for its rho
        mantel = {}
        for ses, S_ses in (("ses-1", S1), ("ses-2", S2)):
            rho_m, p_m = mantel_test(S_ses, D_mats[ses], n_permutations=n_permutations, seed=seed)
            mantel[ses] = corr_result(rho_m, p_m, mantel_bootstrap(S_ses, D_mats[ses], draws))

        # 2. Correlation of self-similarity / idiff with inter-session interval and with age at ses-1, each with an
        #    ordinary infant bootstrap CI (the same infants drawn for both variables)
        self_vals = np.diag(M)
        others_vals = np.zeros(n_subs, dtype=np.float64)
        for a in range(n_subs):
            row_off = [M[a, b] for b in range(n_subs) if b != a]
            col_off = [M[b, a] for b in range(n_subs) if b != a]
            others_vals[a] = np.nanmean(row_off + col_off)
        idiff_vals = self_vals - others_vals

        interval_corr = {}
        for corr_key, x_vals, y_vals in [("self_vs_interval", self_vals, intervals_days),
                                         ("idiff_vs_interval", idiff_vals, intervals_days),
                                         ("self_vs_age1", self_vals, ages_ses1_days),
                                         ("idiff_vs_age1", idiff_vals, ages_ses1_days)]:
            rho_c, p_c = perm_spearman(x_vals, y_vals, n_permutations=n_permutations, seed=seed)
            interval_corr[corr_key] = corr_result(rho_c, p_c, spearman_rows(x_vals[draws], y_vals[draws]))

        u_shift = u_results[fid]["developmental_shift"]
        dev_results[fid] = {
            "fid": fid, "name": fname,
            "seed": seed,
            "n_permutations": n_permutations,
            "delta_U_global": u_shift["delta_U_global"],
            "median_delta": u_shift["median_delta"],
            "ci_95": u_shift["ci_95"],
            "p_value": u_shift["p_value"],
            "mean_abs_dage_days": mean_abs_dage,
            "delta_U_elem": u_shift["delta_U_elem"],
            "delta_U_elem_q": u_shift["delta_U_elem_q"],
            "n_sig_delta_U_elem": u_shift["n_sig_delta_U_elem"],
            "delta_u_infant": u_shift["delta_u_infant"],
            "mantel_ses1": mantel["ses-1"],
            "mantel_ses2": mantel["ses-2"],
            "interval_corr": interval_corr,
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
        # 3. SIBLING / TWIN ANALYSIS (Table T6), recording level
        # -------------------------------------------------------------
        # Sibling similarity in ses-1 and ses-2
        s_twin_1 = float(S1[idx_86, idx_87])
        s_twin_2 = float(S2[idx_86, idx_87])

        all_s1 = S1[p_triu, q_triu]
        all_s2 = S2[p_triu, q_triu]

        # Item 1: percentile rank among all 171 pairs (0 to 100%)
        rank_all_1 = pct_rank(all_s1, s_twin_1)
        rank_all_2 = pct_rank(all_s2, s_twin_2)

        # Item 2: rank, median and range in each matched control set
        rank_matched_infant_1 = pct_rank(all_s1[matched_infant_idx], s_twin_1)
        rank_matched_infant_2 = pct_rank(all_s2[matched_infant_idx], s_twin_2)
        rank_matched_age_1 = pct_rank(all_s1[matched_age_idx["ses-1"]], s_twin_1)
        rank_matched_age_2 = pct_rank(all_s2[matched_age_idx["ses-2"]], s_twin_2)

        # Item 3: cross-session cross-identification in M, both directions
        cross_id = cross_identification(M, idx_86, idx_87)

        # Item 4: shared elements exceeding the 95th percentile of non-sibling pairs in BOTH sessions
        with np.errstate(divide="ignore", invalid="ignore"):
            j1_nontwin_p95 = np.nanpercentile(J1_pairs[non_twin_idxs, :], 95, axis=0)
            j2_nontwin_p95 = np.nanpercentile(J2_pairs[non_twin_idxs, :], 95, axis=0)

        j1_twin = J1_pairs[twin_pair_idx, :]
        j2_twin = J2_pairs[twin_pair_idx, :]

        tax_classes = tax_results[fid]["classes"]
        shared_elements = []
        shared_classes = {}
        for e in range(dim):
            v1, v2 = j1_twin[e], j2_twin[e]
            t1, t2 = j1_nontwin_p95[e], j2_nontwin_p95[e]
            if not (np.isnan(v1) or np.isnan(v2) or np.isnan(t1) or np.isnan(t2)) and v1 > t1 and v2 > t2:
                shared_elements.append(labels[e])
                shared_classes[labels[e]] = tax_classes[e]
        # "Individual" elements in Step 5a: I_diff(e) significant (classes Individual, Universal and Individual)
        n_shared_individual = sum(c in ("Individual", "Universal and Individual") for c in shared_classes.values())

        # Items 1-3 at element level
        element_level = []
        for e in range(dim):
            element_level.append({
                "element": labels[e],
                "J_ses1": float(j1_twin[e]),
                "J_ses2": float(j2_twin[e]),
                "rank_among_171_ses1": pct_rank(J1_pairs[:, e], j1_twin[e]),
                "rank_among_171_ses2": pct_rank(J2_pairs[:, e], j2_twin[e]),
                "rank_matched_infant_ses1": pct_rank(J1_pairs[matched_infant_idx, e], j1_twin[e]),
                "rank_matched_infant_ses2": pct_rank(J2_pairs[matched_infant_idx, e], j2_twin[e]),
                "rank_matched_age_ses1": pct_rank(J1_pairs[matched_age_idx["ses-1"], e], j1_twin[e]),
                "rank_matched_age_ses2": pct_rank(J2_pairs[matched_age_idx["ses-2"], e], j2_twin[e]),
                "cross_identification": cross_identification(M_elem[:, :, e], idx_86, idx_87)
            })
        n_elem_misidentified = sum(
            el["cross_identification"]["misidentified_86_as_87"] or el["cross_identification"]["misidentified_87_as_86"]
            for el in element_level
        )

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
            "matched_infant_set_ses1": set_summary(all_s1[matched_infant_idx]),
            "matched_infant_set_ses2": set_summary(all_s2[matched_infant_idx]),
            "matched_age_set_ses1": set_summary(all_s1[matched_age_idx["ses-1"]]),
            "matched_age_set_ses2": set_summary(all_s2[matched_age_idx["ses-2"]]),
            "cross_identification": cross_id,
            "n_shared_p95_elements": len(shared_elements),
            "shared_elements": shared_elements,
            "shared_element_classes": shared_classes,
            "n_shared_individual_elements": n_shared_individual,
            "n_elements_misidentified": n_elem_misidentified,
            "element_level": element_level,
            "same_day_control": SAME_DAY_CONTROL_NOTE
        }

        m1, m2 = mantel["ses-1"], mantel["ses-2"]
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:25]:<25}) | Mantel s1: rho={m1['rho']:+.3f} [{m1['ci_95'][0]:+.3f}, {m1['ci_95'][1]:+.3f}] (p={m1['p']:.4f}), s2: rho={m2['rho']:+.3f} (p={m2['p']:.4f}) | Twin Rank: s1={rank_all_1:4.1f}%, s2={rank_all_2:4.1f}% | Shared P95: {len(shared_elements):<2}/{dim} (individual: {n_shared_individual})")

    # BH-FDR of each global test across the 26 feature sets (Step 7.4 ii)
    for test_key in ["mantel_ses1", "mantel_ses2"]:
        tests = [dev_results[fid][test_key] for fid in dev_results]
        q_vals = multipletests([t["p"] for t in tests], method="fdr_bh")[1]
        for t, q_val in zip(tests, q_vals):
            t["q"] = float(q_val)
    for corr_key in ["self_vs_interval", "idiff_vs_interval", "self_vs_age1", "idiff_vs_age1"]:
        tests = [dev_results[fid]["interval_corr"][corr_key] for fid in dev_results]
        q_vals = multipletests([t["p"] for t in tests], method="fdr_bh")[1]
        for t, q_val in zip(tests, q_vals):
            t["q"] = float(q_val)
    q_delta = multipletests([dev_results[fid]["p_value"] for fid in dev_results], method="fdr_bh")[1]
    for fid, q_val in zip(dev_results, q_delta):
        dev_results[fid]["q_value"] = float(q_val)

    out_dev = OUTPUT_TENSORS_DIR / "developmental_results.json"
    out_twin = OUTPUT_TENSORS_DIR / "twin_results.json"
    with open(out_dev, "w") as f:
        json.dump(dev_results, f, indent=2)
    with open(out_twin, "w") as f:
        json.dump(twin_results, f, indent=2)

    print("=" * 80)
    print(f"Mean |Delta age| over pairs: ses-1 {mean_abs_dage['ses-1']:.1f} days, ses-2 {mean_abs_dage['ses-2']:.1f} days")
    print("Developmental & Twin/Sibling results successfully computed!")
    print(f"Developmental: {out_dev}")
    print(f"Twin (Table T6): {out_twin}")
    print("=" * 80)
    return dev_results, twin_results


if __name__ == "__main__":
    run_developmental_and_twins()
