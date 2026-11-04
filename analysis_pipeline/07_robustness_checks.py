"""
Phase 7: Robustness & Sensitivity Checks (Step 8 - Table T4)
Reruns the main results - U_1, U_2, I_diff, identification accuracy and the Step 5a class counts - with one
change at a time (furtherAnalysis Step 8); everything else is kept as in the main analysis:
  1. 0/0 rule: 0/0 set to 0 instead of excluded, for J and S (all features)
  2. Epoch pairing: J and S averaged over all K x K epoch pairs instead of (k, k) (all features)
  3. Number of epochs: K = 6 (first 6 epochs) instead of K = 12 (all features)
  4. Twins: sub-NORB00086 and sub-NORB00087 excluded, N = 17 (all features)
  5. Pattern only: each epoch's vector divided by its sum before computing S (all features, recording level)
  6. Reference: average reference before computing correlations; all 26 features rebuilt from the EDFs
  7. Volume conduction: debiased wPLI (mne_connectivity "wpli2_debiased", per band, negative values set to 0)
     instead of |r| (weighted edge weight features)
  8. Binarisation: the m strongest edges of every network instead of the GCC rule, m = median GCC edge count
     (all binary features)
  9. Electrode distance: Spearman rho of U(e) and of I_diff(e) with inter-electrode distance, and the within- vs
     between-region comparison of U(e) repeated on its residuals after linear regression on distance (edge features)
The element-level tests behind the class counts use n_permutations (default 200,000, as in the main analysis). The baseline is
recomputed with the same code and permutations, so class counts are compared like with like; its recording-level
values are checked against Phases 2 and 4.
Outputs Table T4 (checks 1-8), Table T4b (check 9) and robustness_results.json.
"""
import concurrent.futures
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import mne
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
        CHANNELS,
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        FREQUENCY_BANDS,
        N_EPOCHS,
        N_PERMUTATIONS,
        PERM_TIE_TOL,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        SEED,
        SESSIONS,
        SUBJECT_IDS,
        TABLES_DIR,
    )
    from analysis_pipeline.utils import (
        bootstrap_ci,
        bootstrap_counts,
        bootstrap_draws,
        bootstrap_p,
        compute_all_broadband_features,
        compute_binary_features,
        extract_band_matrices,
        extract_matrix_edge_vector,
        identification_accuracy,
        load_annotations,
        recording_S,
        segment_eeg_from_annotations,
        weighted_nanmean,
    )
else:
    from .config import (
        CHANNELS,
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        FREQUENCY_BANDS,
        N_EPOCHS,
        N_PERMUTATIONS,
        PERM_TIE_TOL,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        SEED,
        SESSIONS,
        SUBJECT_IDS,
        TABLES_DIR,
    )
    from .utils import (
        bootstrap_ci,
        bootstrap_counts,
        bootstrap_draws,
        bootstrap_p,
        compute_all_broadband_features,
        compute_binary_features,
        extract_band_matrices,
        extract_matrix_edge_vector,
        identification_accuracy,
        load_annotations,
        recording_S,
        segment_eeg_from_annotations,
        weighted_nanmean,
    )

# Band features: weighted edge weight and binary adjacency per band (as in Phase 1)
BAND_FEATURES = {"delta": ("f19", "f20"), "theta": ("f21", "f22"), "alpha": ("f23", "f24"), "beta": ("f25", "f26")}
# Check 7: weighted edge weight features and their frequency range; broadband spans the four bands
DWPLI_FEATURES = {
    "f01": (FREQUENCY_BANDS["delta"][0], FREQUENCY_BANDS["beta"][1]),
    "f19": FREQUENCY_BANDS["delta"],
    "f21": FREQUENCY_BANDS["theta"],
    "f23": FREQUENCY_BANDS["alpha"],
    "f25": FREQUENCY_BANDS["beta"],
}
# Check 8: binary features rebuilt from the weighted network they are thresholded from
FIXED_DENSITY_SOURCES = {"broadband": "f01", "delta": "f19", "theta": "f21", "alpha": "f23", "beta": "f25"}
EPOCH_SECONDS = 2.56
TWIN_IDS = ("sub-NORB00086", "sub-NORB00087")
METRICS = ["U_1", "U_2", "I_diff", "acc_forward", "acc_backward", "acc_mean"]
CLASSES = ["Universal", "Individual", "Universal and Individual", "Neither"]


# ----------------------------------------------------------------------------------------------
# Main results for one feature tensor (Steps 2, 4 and 5a)
# ----------------------------------------------------------------------------------------------

def epoch_pairing(K: int, pairing: str):
    """Epoch index pairs: (k, k) for the main analysis, or all K x K pairs (check 2)."""
    if pairing == "diag":
        return np.arange(K), np.arange(K)
    k1, k2 = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    return k1.ravel(), k2.ravel()


def nan_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """Mean ignoring NaN, NaN where no value is defined (without RuntimeWarnings)."""
    valid = ~np.isnan(values)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    count = valid.sum(axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def jaccard_ratio(x1: np.ndarray, x2: np.ndarray, zero_rule: bool) -> np.ndarray:
    """min / max per entry; 0/0 is NaN (main rule) or 0 (check 1)."""
    mn = np.minimum(x1, x2)
    mx = np.maximum(x1, x2)
    ok = mx > 0
    ratio = np.divide(mn, mx, out=np.zeros(np.broadcast_shapes(mn.shape, mx.shape)), where=ok)
    return ratio if zero_rule else np.where(ok, ratio, np.nan)


def cross_J(a: np.ndarray, b: np.ndarray, k1: np.ndarray, k2: np.ndarray, zero_rule: bool) -> np.ndarray:
    """
    G[e, e2] = J between a's element e and b's element e2, averaged over the epoch pairs.
    The diagonal is the observed J(e); the element-shuffle null reads G[e, pi(e)], so it averages
    epochs first and pairs second exactly like the observed value (Step 2 item 5).
    """
    ratio = jaccard_ratio(a[k1][:, :, None], b[k2][:, None, :], zero_rule)
    return nan_mean(ratio, axis=0)


def element_shuffle_p(G: np.ndarray, obs: np.ndarray, n_permutations: int, rng, chunk: int = 100) -> np.ndarray:
    """One-sided p of U_s(e) against the element-shuffle null (Step 2 item 5); undefined U_s(e) gets p = 1."""
    n_pairs, dim, _ = G.shape
    pair_idx = np.arange(n_pairs)[None, :, None]
    elem_idx = np.arange(dim)[None, None, :]
    count_ge = np.zeros(dim, dtype=np.int64)
    done = 0
    while done < n_permutations:
        c = min(chunk, n_permutations - done)
        perms = rng.permuted(np.tile(np.arange(dim), (c, n_pairs, 1)), axis=2)  # one permutation per pair
        null_U = nan_mean(G[pair_idx, elem_idx, perms], axis=1)  # (c, dim)
        count_ge += np.sum(null_U >= obs - PERM_TIE_TOL, axis=0)
        done += c
    p = (1.0 + count_ge) / (1.0 + n_permutations)
    p[np.isnan(obs)] = 1.0
    return p


def identifiability_elem_p(M_elem: np.ndarray, obs_diff: np.ndarray, n_permutations: int, rng) -> np.ndarray:
    """One-sided p of I_diff(e) from column permutations of every M_e together (Step 7.3 item 2), as in Phase 4."""
    n = M_elem.shape[0]
    rows = np.arange(n)
    valid = ~np.isnan(M_elem)
    filled = np.where(valid, M_elem, 0.0)
    total_sum, total_count = filled.sum(axis=(0, 1)), valid.sum(axis=(0, 1))
    count_ge = np.zeros(M_elem.shape[2], dtype=np.int64)
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        diag_sum, diag_count = filled[rows, perm].sum(axis=0), valid[rows, perm].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            null_diff = diag_sum / diag_count - (total_sum - diag_sum) / (total_count - diag_count)
        count_ge += null_diff >= obs_diff - PERM_TIE_TOL  # tolerance: observed and null are summed in a different order
    p = (1.0 + count_ge) / (1.0 + n_permutations)
    p[np.isnan(obs_diff)] = 1.0  # no observed I_diff(e): 0/0 in every comparison
    return p


def main_results(X: np.ndarray, n_permutations: int, seed: int, pairing: str = "diag",
                 zero_rule: bool = False, element_level: bool = True, keep_arrays: bool = False) -> dict:
    """
    U_1, U_2 (Step 2), I_diff and identification accuracy (Step 4), and the Step 5a class counts for one
    feature tensor X of shape (n_infants, 2, K, dim). Class counts need the element-level tests, so they are
    skipped (None) when element_level is False.
    """
    rng = np.random.default_rng(seed)
    n, _, K = X.shape[:3]
    k1, k2 = epoch_pairing(K, pairing)
    pairs = list(combinations(range(n), 2))
    out = {}
    q_u = []
    pair_J = []
    for s in range(2):
        out[f"U_{s + 1}"] = float(np.nanmean([recording_S(X[a, s][k1], X[b, s][k2], zero_rule) for a, b in pairs]))
        if element_level:
            G = np.stack([cross_J(X[a, s], X[b, s], k1, k2, zero_rule) for a, b in pairs])
            J_s = np.diagonal(G, axis1=1, axis2=2)  # (n_pairs, dim)
            U_elem = nan_mean(J_s, axis=0)
            q_u.append(np.asarray(multipletests(element_shuffle_p(G, U_elem, n_permutations, rng), method="fdr_bh")[1]))
            pair_J.append(J_s)
            out[f"U_elem_{s + 1}"] = U_elem

    M = np.array([[recording_S(X[a, 0][k1], X[b, 1][k2], zero_rule) for b in range(n)] for a in range(n)])
    off = ~np.eye(n, dtype=bool)
    out["I_self"] = float(np.nanmean(np.diag(M)))
    out["I_others"] = float(np.nanmean(M[off]))
    out["I_diff"] = out["I_self"] - out["I_others"]
    out["acc_forward"], out["acc_backward"] = identification_accuracy(M)
    out["acc_mean"] = (out["acc_forward"] + out["acc_backward"]) / 2.0

    out["class_counts"] = None
    if element_level:
        M_elem = np.array([[nan_mean(jaccard_ratio(X[a, 0][k1], X[b, 1][k2], zero_rule), axis=0)
                            for b in range(n)] for a in range(n)])  # (n, n, dim)
        I_diff_elem = nan_mean(M_elem[~off], axis=0) - nan_mean(M_elem[off], axis=0)
        q_idiff = np.asarray(multipletests(identifiability_elem_p(M_elem, I_diff_elem, n_permutations, rng), method="fdr_bh")[1])
        sig_u = (q_u[0] < 0.05) & (q_u[1] < 0.05)
        sig_i = q_idiff < 0.05
        out["class_counts"] = {
            "Universal": int(np.sum(sig_u & ~sig_i)),
            "Individual": int(np.sum(~sig_u & sig_i)),
            "Universal and Individual": int(np.sum(sig_u & sig_i)),
            "Neither": int(np.sum(~sig_u & ~sig_i)),
        }
        out["I_diff_elem"] = I_diff_elem
        if keep_arrays:  # for check 9
            out["pair_J"] = pair_J
            out["M_elem"] = M_elem
    if not keep_arrays:
        for key in ("U_elem_1", "U_elem_2", "I_diff_elem"):
            out.pop(key, None)
    return out


def _main_results_task(task):
    key, X, kwargs = task
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return key, main_results(X, **kwargs)


# ----------------------------------------------------------------------------------------------
# Rebuilt feature tensors (checks 6, 7 and 8)
# ----------------------------------------------------------------------------------------------

def _load_recording(subj: str, ses: str, reference: str | None):
    edf_path = DATASET_PATH / subj / ses / "eeg" / f"{subj}_{ses}_task-EEG_eeg.edf"
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(CHANNELS)
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    ann_df = load_annotations(subj, ses)
    if ann_df is None:
        raise FileNotFoundError(f"Missing annotations for {subj} {ses}")
    return raw, ann_df


def rebuild_recording_features(subj: str, ses: str, reference: str | None) -> dict:
    """
    All 26 features of one recording, built from the EDF as in Phase 1, optionally after re-referencing
    (check 6). Broadband f01 - f18 use |r| of the unfiltered epochs, which reproduces the Phase 1 correlation files;
    band features filter the whole recording before cutting epochs.
    """
    raw, ann_df = _load_recording(subj, ses, reference)
    out = {fid: np.zeros((N_EPOCHS, meta["dim"])) for fid, meta in FEATURE_DEFS.items()}
    segments = segment_eeg_from_annotations(raw, ann_df, label="eyes_closed")[:N_EPOCHS]
    if len(segments) < N_EPOCHS:
        raise ValueError(f"Insufficient segments for {subj} {ses}")
    for ep, seg in enumerate(segments):
        df = pd.DataFrame(np.abs(np.corrcoef(seg)), index=CHANNELS, columns=CHANNELS)
        feats = compute_all_broadband_features(df)
        for fid, meta in FEATURE_DEFS.items():
            if meta["band"] == "broadband":
                out[fid][ep] = feats[fid]
    for band, (w_fid, b_fid) in BAND_FEATURES.items():
        l_freq, h_freq = FREQUENCY_BANDS[band]
        band_results = extract_band_matrices(subj, ses, l_freq, h_freq, n_epochs=N_EPOCHS, raw=raw, ann_df=ann_df)
        for ep, (corr_df, bin_df, _thr, _density) in enumerate(band_results):
            out[w_fid][ep] = extract_matrix_edge_vector(corr_df)
            out[b_fid][ep] = extract_matrix_edge_vector(bin_df)
    return out


def dwpli_recording(subj: str, ses: str) -> dict:
    """
    Debiased wPLI (check 7) of one recording with mne_connectivity, one value per edge and band, estimated across
    the 12 epochs (each cut to 2.56 s); negative values set to 0. Returns fid -> (171,) in EDGE_LABELS order.
    """
    from mne_connectivity import spectral_connectivity_epochs

    raw, ann_df = _load_recording(subj, ses, None)
    sfreq = float(raw.info["sfreq"])
    segments = segment_eeg_from_annotations(raw, ann_df, label="eyes_closed")[:N_EPOCHS]
    n_samples = min(int(round(EPOCH_SECONDS * sfreq)), min(seg.shape[1] for seg in segments))
    data = np.stack([seg[:, :n_samples] for seg in segments])  # (12, 19, n_samples)
    ch_idx = {ch: i for i, ch in enumerate(CHANNELS)}
    out = {}
    for fid, (fmin, fmax) in DWPLI_FEATURES.items():
        con = spectral_connectivity_epochs(
            data, method="wpli2_debiased", sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=False
        )
        dense = np.asarray(con.get_data(output="dense"))[:, :, 0]
        dense = np.maximum(dense + dense.T, 0.0)  # lower triangle filled; negative values set to 0
        out[fid] = np.array([dense[ch_idx[e.split("-")[0]], ch_idx[e.split("-")[1]]] for e in EDGE_LABELS])
    return out


def fixed_density_recording(weighted: dict, m_edges: dict) -> dict:
    """
    Binary features of one recording from the m strongest edges of every weighted network (check 8).
    weighted: family -> (12, 171) weighted edge weights. Returns fid -> (12, dim).
    """
    out = {}
    node_idx = {node: i for i, node in enumerate(NODE_LABELS)}
    for family, W in weighted.items():
        m = m_edges[family]
        rows = []
        for w in W:
            keep = np.argsort(-w, kind="stable")[:m]
            adj = np.zeros((len(NODE_LABELS), len(NODE_LABELS)))
            for e in keep:
                u, v = EDGE_LABELS[e].split("-")
                adj[node_idx[u], node_idx[v]] = adj[node_idx[v], node_idx[u]] = 1.0
            bin_df = pd.DataFrame(adj, index=NODE_LABELS, columns=NODE_LABELS)
            rows.append(compute_binary_features(bin_df) if family == "broadband" else {"adj": extract_matrix_edge_vector(bin_df)})
        if family == "broadband":
            for fid in rows[0]:
                out[fid] = np.stack([r[fid] for r in rows])
        else:
            out[BAND_FEATURES[family][1]] = np.stack([r["adj"] for r in rows])
    return out


def _recording_task(task):
    kind, sub_idx, ses_idx, payload = task
    warnings.filterwarnings("ignore")
    subj, ses = SUBJECT_IDS[sub_idx], SESSIONS[ses_idx]
    if kind == "avg_ref":
        return kind, sub_idx, ses_idx, rebuild_recording_features(subj, ses, "average")
    if kind == "dwpli":
        return kind, sub_idx, ses_idx, dwpli_recording(subj, ses)
    return kind, sub_idx, ses_idx, fixed_density_recording(*payload)


# ----------------------------------------------------------------------------------------------
# Check 9: electrode distance
# ----------------------------------------------------------------------------------------------

def get_electrode_3d_distances():
    """Load 3D Cartesian coordinates for standard electrodes and compute 171 pairwise Euclidean distances."""
    elec_path = DATASET_PATH / "sub-NORB00001" / "ses-1" / "eeg" / "sub-NORB00001_ses-1_electrodes.tsv"
    if not elec_path.exists():
        raise FileNotFoundError(f"Electrode coordinates file not found: {elec_path}")
    df = pd.read_csv(elec_path, sep="\t")
    coords: dict[str, np.ndarray] = {}
    for name, x, y, z in zip(df["name"], df["x"], df["y"], df["z"]):
        if str(name) in NODE_LABELS:
            coords[str(name)] = np.array([float(x), float(y), float(z)])

    distances = []
    for edge in EDGE_LABELS:
        u, v = edge.split("-")
        d = float(np.linalg.norm(coords[u] - coords[v]))
        distances.append(d)
    return np.array(distances, dtype=np.float64)


def spearman_rows(Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Spearman rho between each row of Y and x, using the finite entries of that row."""
    out = np.full(Y.shape[0], np.nan)
    finite = np.isfinite(Y)
    full = finite.all(axis=1)
    if full.any():
        ry = stats.rankdata(Y[full], axis=1)
        ry -= ry.mean(axis=1, keepdims=True)
        rx = stats.rankdata(x)
        rx -= rx.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            out[full] = (ry @ rx) / (np.sqrt((ry ** 2).sum(axis=1)) * np.sqrt((rx ** 2).sum()))
    for i in np.where(~full)[0]:
        ok = finite[i]
        if ok.sum() > 2:
            out[i] = stats.spearmanr(Y[i, ok], x[ok])[0]
    return out


def within_minus_between(Y: np.ndarray, within: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    """
    Mean over within-region edges minus mean over between-region edges, per row of Y. With x, the comparison is
    made on the residuals of each row after linear regression on x (check 9).
    """
    out = np.full(Y.shape[0], np.nan)
    for i, y in enumerate(Y):
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        r = y[ok]
        if x is not None:
            slope, intercept = np.polyfit(x[ok], r, 1)
            r = r - (intercept + slope * x[ok])
        out[i] = np.mean(r[within[ok]]) - np.mean(r[~within[ok]])
    return out


def bootstrap_summary(obs: float, boot: np.ndarray) -> dict:
    """Observed value, 95% percentile CI and two-sided bootstrap p (Step 7.2 item 4, floor 1/B)."""
    ci_low, ci_high = bootstrap_ci(boot)
    return {"obs": float(obs), "ci_95": [float(ci_low), float(ci_high)], "p": float(bootstrap_p(boot))}


def distance_check(base: dict, distances: np.ndarray, B: int, seed: int) -> dict:
    """
    Check 9 for one edge feature. U(e) = mean of U_1(e) and U_2(e); every quantity is recomputed in the same
    infant resamples (Step 7.2): pair values weighted by c_i * c_j and M[i, i] by c_i, where c counts how often
    each infant was drawn, which equals using all position pairs with d_p != d_q.
    """
    n = base["M_elem"].shape[0]
    within = np.array([NODE_REGIONS[e.split("-")[0]] == NODE_REGIONS[e.split("-")[1]] for e in EDGE_LABELS])
    c = bootstrap_counts(bootstrap_draws(n, B, seed), n)
    p_i, q_i = np.triu_indices(n, k=1)
    W_pairs = c[:, p_i] * c[:, q_i]

    U_obs = (base["U_elem_1"] + base["U_elem_2"]) / 2.0
    U_boot = (weighted_nanmean(W_pairs, base["pair_J"][0]) + weighted_nanmean(W_pairs, base["pair_J"][1])) / 2.0
    M_elem = base["M_elem"]
    off = ~np.eye(n, dtype=bool)
    W_off = (c[:, :, None] * c[:, None, :])[:, off]
    I_diff_boot = weighted_nanmean(c, M_elem[np.arange(n), np.arange(n)]) - weighted_nanmean(W_off, M_elem[off])
    I_diff_obs = base["I_diff_elem"]

    return {
        "rho_U_distance": bootstrap_summary(spearman_rows(U_obs[None], distances)[0], spearman_rows(U_boot, distances)),
        "rho_Idiff_distance": bootstrap_summary(spearman_rows(I_diff_obs[None], distances)[0], spearman_rows(I_diff_boot, distances)),
        "within_minus_between_U": bootstrap_summary(within_minus_between(U_obs[None], within)[0], within_minus_between(U_boot, within)),
        "within_minus_between_U_resid": bootstrap_summary(
            within_minus_between(U_obs[None], within, distances)[0], within_minus_between(U_boot, within, distances)
        ),
    }


# ----------------------------------------------------------------------------------------------
# Table T4
# ----------------------------------------------------------------------------------------------

def t4_row(check_id: str, check_name: str, fid: str, base: dict, pert: dict, note: str = "") -> dict:
    row = {"Check ID": check_id, "Sensitivity Analysis": check_name, "Feature ID": fid,
           "Feature Name": FEATURE_DEFS[fid]["name"], "Note": note}
    for k in METRICS:
        row[f"{k} baseline"] = round(base[k], 4)
        row[f"{k} check"] = round(pert[k], 4)
        row[f"{k} change"] = round(pert[k] - base[k], 4)
    row["I_diff same sign"] = bool(np.sign(pert["I_diff"]) == np.sign(base["I_diff"]))
    for cls in CLASSES:
        row[f"{cls} baseline"] = base["class_counts"][cls]
        row[f"{cls} check"] = pert["class_counts"][cls] if pert["class_counts"] is not None else "n/a"
    return row


def run_robustness_checks(n_permutations: int = N_PERMUTATIONS, n_workers: int = 8, seed: int = SEED):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    n_subs = len(SUBJECT_IDS)
    print("=" * 80)
    print("PHASE 7: ROBUSTNESS & SENSITIVITY CHECKS (STEP 8 - TABLE T4)")
    print(f"Cohort: {n_subs} infants | 9 checks | Element-level permutations: {n_permutations} | Seed: {seed} | Workers: {n_workers}")
    print("=" * 80)

    u_file = OUTPUT_TENSORS_DIR / "universality_results.json"
    i_file = OUTPUT_TENSORS_DIR / "identifiability_results.json"
    if not u_file.exists() or not i_file.exists():
        raise FileNotFoundError("Prerequisite baseline results missing. Please run Phases 2 and 4 first.")
    with open(u_file) as f:
        u_main = json.load(f)
    with open(i_file) as f:
        i_main = json.load(f)

    X_all = {fid: np.load(OUTPUT_TENSORS_DIR / f"X_{fid}.npy") for fid in FEATURE_DEFS}
    all_fids = list(FEATURE_DEFS)
    binary_fids = [fid for fid, meta in FEATURE_DEFS.items() if meta["network"] == "binary"]
    edge_fids = [fid for fid, meta in FEATURE_DEFS.items() if meta["type"] == "edge"]
    recordings = [(a, s) for a in range(n_subs) for s in range(len(SESSIONS))]

    # m = median GCC edge count of each network family (Step 0 item 5), for check 8
    m_edges = {"broadband": int(round(float(np.median(X_all["f10"].sum(axis=-1)))))}
    for band, (_w, b_fid) in BAND_FEATURES.items():
        m_edges[band] = int(round(float(np.median(X_all[b_fid].sum(axis=-1)))))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Rebuilt tensors for checks 6, 7 and 8, one task per recording
        print("Rebuilding features: average reference (check 6), debiased wPLI (check 7), fixed density (check 8)...")
        rec_tasks = []
        for a, s in recordings:
            rec_tasks.append(("avg_ref", a, s, None))
            rec_tasks.append(("dwpli", a, s, None))
            weighted = {fam: X_all[src][a, s] for fam, src in FIXED_DENSITY_SOURCES.items()}
            rec_tasks.append(("fixed_density", a, s, (weighted, m_edges)))
        X_avg = {fid: np.zeros_like(X_all[fid]) for fid in all_fids}
        X_dwpli = {fid: np.zeros((n_subs, len(SESSIONS), 1, len(EDGE_LABELS))) for fid in DWPLI_FEATURES}
        X_fixed = {fid: np.zeros_like(X_all[fid]) for fid in binary_fids}
        for kind, a, s, feats in executor.map(_recording_task, rec_tasks):
            for fid, values in feats.items():
                if kind == "avg_ref":
                    X_avg[fid][a, s] = values
                elif kind == "dwpli":
                    X_dwpli[fid][a, s, 0] = values
                else:
                    X_fixed[fid][a, s] = values

        # Main results under every condition; one task per condition and feature
        non_twin = [i for i, s in enumerate(SUBJECT_IDS) if s not in TWIN_IDS]
        base_kw = {"n_permutations": n_permutations, "seed": seed}
        tasks = []
        for fid in all_fids:
            X = X_all[fid]
            sums = X.sum(axis=3, keepdims=True)
            X_pattern = np.divide(X, sums, out=np.zeros_like(X), where=sums > 0)
            tasks += [
                (("check2", fid), X, {**base_kw, "pairing": "all"}),
                (("baseline", fid), X, {**base_kw, "keep_arrays": fid in edge_fids}),
                (("check1", fid), X, {**base_kw, "zero_rule": True}),
                (("check3", fid), X[:, :, :6], base_kw),
                (("check4", fid), X[non_twin], base_kw),
                (("check5", fid), X_pattern, {**base_kw, "element_level": False}),
                (("check6", fid), X_avg[fid], base_kw),
            ]
        tasks += [(("check7", fid), X_dwpli[fid], base_kw) for fid in DWPLI_FEATURES]
        tasks += [(("check8", fid), X_fixed[fid], base_kw) for fid in binary_fids]
        print(f"Computing main results for {len(tasks)} condition x feature combinations...")
        res = dict(executor.map(_main_results_task, tasks))

    base = {fid: res[("baseline", fid)] for fid in all_fids}

    # The baseline must reproduce the Phase 2 and Phase 4 recording-level values
    max_diff = max(
        max(abs(base[fid]["U_1"] - u_main[fid]["ses-1"]["U_global"]),
            abs(base[fid]["U_2"] - u_main[fid]["ses-2"]["U_global"]),
            abs(base[fid]["I_diff"] - i_main[fid]["I_diff"]),
            abs(base[fid]["acc_mean"] - i_main[fid]["acc_mean"]))
        for fid in all_fids
    )
    print(f"Baseline vs Phases 2 and 4 (U_1, U_2, I_diff, accuracy): max |difference| = {max_diff:.2e}")

    checks = [
        ("Check 1", "0/0 Rule (0/0 = 0)", "check1", all_fids, "0/0 = 0 for J and for S (S is 0/0 only where both epoch vectors are all zero)"),
        ("Check 2", "Epoch Pairing (all K x K pairs)", "check2", all_fids, "J and S averaged over all 144 epoch pairs"),
        ("Check 3", "Number of Epochs (K = 6)", "check3", all_fids, "First 6 epochs"),
        ("Check 4", "Twin Exclusion (N = 17)", "check4", all_fids, "sub-NORB00086 and sub-NORB00087 excluded"),
        ("Check 5", "Pattern Only (vector / sum)", "check5", all_fids, "Recording level only, so no class counts"),
        ("Check 6", "Average Reference", "check6", all_fids, "All 26 features rebuilt after raw.set_eeg_reference('average')"),
        ("Check 7", "Volume Conduction (debiased wPLI)", "check7", list(DWPLI_FEATURES),
         "wpli2_debiased across the 12 epochs (one network per recording, K = 1), negative values set to 0; "
         "frequencies below 5 cycles per 2.56 s epoch (~2 Hz) are not estimated"),
        ("Check 8", "Fixed-Density Binarisation", "check8", binary_fids, ""),
    ]
    t4_rows = []
    for check_id, check_name, key, fids, note in checks:
        for fid in fids:
            row_note = note
            if key == "check7":
                lo, hi = DWPLI_FEATURES[fid]
                row_note = f"{note}; {lo}-{hi} Hz"
            elif key == "check8":
                family = FEATURE_DEFS[fid]["band"]
                row_note = f"m = {m_edges[family]} strongest edges ({family} median GCC edge count)"
            t4_rows.append(t4_row(check_id, check_name, fid, base[fid], res[(key, fid)], row_note))
        same_sign = sum(r["I_diff same sign"] for r in t4_rows if r["Check ID"] == check_id)
        print(f"[{check_id}] {check_name}: {len(fids)} features | I_diff sign kept in {same_sign}/{len(fids)}")

    # Check 9: electrode distance, edge features
    print("[Check 9] Electrode distance: Spearman rho with distance and within- vs between-region on residuals...")
    distances = get_electrode_3d_distances()
    dist_res = {fid: distance_check(base[fid], distances, B=n_permutations, seed=seed) for fid in edge_fids}
    # BH-FDR of each global test across the edge feature sets (Step 7.4 ii)
    for test in ("rho_U_distance", "rho_Idiff_distance", "within_minus_between_U", "within_minus_between_U_resid"):
        q_vals = multipletests([dist_res[fid][test]["p"] for fid in edge_fids], method="fdr_bh")[1]
        for fid, q_val in zip(edge_fids, q_vals):
            dist_res[fid][test]["q"] = float(q_val)
    t4b_rows = []
    for fid in edge_fids:
        row = {"Feature ID": fid, "Feature Name": FEATURE_DEFS[fid]["name"]}
        for test, label in [("rho_U_distance", "rho(U(e), distance)"), ("rho_Idiff_distance", "rho(I_diff(e), distance)"),
                            ("within_minus_between_U", "Within - between U(e)"),
                            ("within_minus_between_U_resid", "Within - between U(e) residuals")]:
            r = dist_res[fid][test]
            row[label] = round(r["obs"], 4)
            row[f"{label} 95% CI"] = f"[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]"
            row[f"{label} p"] = round(r["p"], 4)
            row[f"{label} q"] = round(r["q"], 4)
        t4b_rows.append(row)

    df_t4 = pd.DataFrame(t4_rows)
    out_csv = TABLES_DIR / "Table_T4_Robustness.csv"
    df_t4.to_csv(out_csv, index=False)
    out_csv_b = TABLES_DIR / "Table_T4b_Electrode_Distance.csv"
    pd.DataFrame(t4b_rows).to_csv(out_csv_b, index=False)

    def recording_level(r):
        return {k: r[k] for k in ["U_1", "U_2", "I_self", "I_others", "I_diff", "acc_forward", "acc_backward", "acc_mean", "class_counts"]}

    out_json = OUTPUT_TENSORS_DIR / "robustness_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_permutations": n_permutations,
            "seed": seed,
            "m_edges": m_edges,
            "baseline_max_abs_diff_vs_phases_2_4": max_diff,
            "baseline": {fid: recording_level(base[fid]) for fid in all_fids},
            "checks": {key: {fid: recording_level(res[(key, fid)]) for fid in fids} for _cid, _name, key, fids, _note in checks},
            "check9_distance": dist_res,
        }, f, indent=2)

    print("=" * 80)
    print(f"Table T4 successfully generated ({len(df_t4)} check x feature rows) and Table T4b ({len(t4b_rows)} edge features)!")
    print(f"CSV saved to: {out_csv} and {out_csv_b}")
    print(f"JSON saved to: {out_json}")
    print("=" * 80)
    return df_t4


if __name__ == "__main__":
    run_robustness_checks()
