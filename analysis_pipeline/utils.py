"""
Graph centrality computations, signal processing, Jaccard similarity and infant bootstrap functions.
"""
import warnings

import mne
import networkx as nx
import numpy as np
import pandas as pd

from .config import (
    CHANNELS,
    DATASET_PATH,
    DERIVATIVES_PATH,
    EDGE_LABELS,
    NODE_LABELS,
)


def gcc_ratio(adj_matrix: np.ndarray) -> float:
    """Return the fraction of nodes in the Giant Connected Component."""
    G = nx.from_numpy_array(adj_matrix)
    if G.number_of_nodes() == 0:
        return 0.0
    comps = list(nx.connected_components(G))
    return max(len(c) for c in comps) / G.number_of_nodes()


def build_binary_network(df: pd.DataFrame, step: float = 0.05):
    """
    GCC Adaptive Thresholding:
    Finds the maximum threshold tau where the network has GCC ratio == 1.0.
    Returns (bin_df, threshold, edge_count).
    """
    abs_corr = np.abs(df.values).copy()
    np.fill_diagonal(abs_corr, 0)

    thr = 1.0
    while thr >= 0.0:
        adj = (abs_corr >= thr).astype(float)
        np.fill_diagonal(adj, 0)
        if gcc_ratio(adj) == 1.0:
            edge_count = int(np.sum(adj) / 2)
            return pd.DataFrame(adj, index=df.index, columns=df.columns), round(thr, 4), edge_count
        thr -= step

    fallback = np.ones_like(abs_corr) - np.eye(df.shape[0])
    edge_count = int(np.sum(fallback) / 2)
    return pd.DataFrame(fallback, index=df.index, columns=df.columns), 0.0, edge_count


def compute_edge_closeness(G: nx.Graph, distance: str | None = "distance") -> dict:
    """
    Compute edge closeness according to Bröhl & Lehnertz (2022) / Communications Physics Eq. (9):
    ec(u, v) = (E - 1) * (c_u * c_v) / (c_u + c_v)
    """
    node_cc = nx.closeness_centrality(G, distance=distance)
    E = G.number_of_edges()
    ec = {}
    for u, v in G.edges():
        cu = node_cc.get(u, 0.0)
        cv = node_cc.get(v, 0.0)
        if cu + cv > 0:
            val = (E - 1) * (cu * cv) / (cu + cv)
        else:
            val = 0.0
        ec[tuple(sorted([u, v]))] = float(val)
    return ec


def compute_edge_strength(G: nx.Graph, weight: str | None = "weight") -> dict:
    """
    Compute nearest-neighbor edge strength:
    es(u, v) = ((k_u + k_v - 2 * w) / (|k_u - k_v| + 1)) * w
    """
    deg = dict(G.degree(weight=weight))
    es = {}
    for u, v, d in G.edges(data=True):
        w = d.get(weight, 1.0) if weight is not None else 1.0
        ku = deg[u]
        kv = deg[v]
        denom = abs(ku - kv) + 1.0
        val = ((ku + kv - 2.0 * w) / denom) * w
        es[tuple(sorted([u, v]))] = float(val)
    return es


def compute_edge_eigenvector(G: nx.Graph, weight: str | None = "weight") -> dict:
    """
    Compute edge eigenvector centrality using line graph with edge weights (w1 + w2)/2.
    """
    L = nx.line_graph(G)
    if L.number_of_edges() == 0:
        return {tuple(sorted(e)): 0.0 for e in G.edges()}

    if weight is not None:
        for e1, e2 in L.edges():
            w1 = G[e1[0]][e1[1]].get(weight, 1.0)
            w2 = G[e2[0]][e2[1]].get(weight, 1.0)
            L[e1][e2]["weight"] = (w1 + w2) / 2.0

    try:
        ec = nx.eigenvector_centrality(L, weight=weight, max_iter=2000)
    except nx.NetworkXException:
        try:
            ec = nx.eigenvector_centrality_numpy(L, weight=weight)
        except (nx.NetworkXException, np.linalg.LinAlgError):
            ec = {e: 1.0 / len(L) for e in L.nodes()}

    normalized_ec = {}
    for edge_tuple, val in ec.items():
        normalized_ec[tuple(sorted(edge_tuple))] = float(val)
    return normalized_ec


def extract_node_vector(node_dict: dict) -> np.ndarray:
    """Extract node centralities in standardized NODE_LABELS order (shape: 19)."""
    return np.array([float(node_dict.get(node, 0.0)) for node in NODE_LABELS], dtype=np.float64)


def extract_edge_vector(edge_dict: dict) -> np.ndarray:
    """Extract edge centralities in standardized EDGE_LABELS order (shape: 171)."""
    vec = []
    for label in EDGE_LABELS:
        u, v = label.split("-")
        key = tuple(sorted([u, v]))
        vec.append(float(edge_dict.get(key, 0.0)))
    return np.array(vec, dtype=np.float64)


def extract_matrix_edge_vector(df: pd.DataFrame) -> np.ndarray:
    """Extract edge weights directly from symmetric DataFrame in EDGE_LABELS order."""
    vec = []
    for label in EDGE_LABELS:
        u, v = label.split("-")
        vec.append(df.loc[u, v])
    return np.array(vec, dtype=np.float64)


def compute_all_broadband_features(df: pd.DataFrame) -> dict:
    """
    Computes all 18 broadband centralities from one segment's correlation matrix.
    Follows the exact notebook implementations in weighted/ and binary/.
    Returns dict mapping 'f01'..'f18' -> np.ndarray.
    """
    # 1. Weighted Graph: directly from df (matching weighted/*.ipynb)
    G_w = nx.from_pandas_adjacency(df)
    # Drop self-loops from the 1.0 diagonal; networkx counts each twice in degree, inflating strength by 2
    G_w.remove_edges_from(list(nx.selfloop_edges(G_w)))
    for _u, _v, d in G_w.edges(data=True):
        w = d.get("weight", 0.0)
        d["distance"] = 1.0 / w if w > 0 else np.inf

    # Weighted centralities
    f01 = extract_matrix_edge_vector(df)
    f02 = extract_node_vector(nx.betweenness_centrality(G_w, weight="distance", normalized=True))

    eb_w_raw = nx.edge_betweenness_centrality(G_w, weight="distance", normalized=True)
    eb_w = {tuple(sorted(k)): v for k, v in eb_w_raw.items()}
    f03 = extract_edge_vector(eb_w)

    f04 = extract_node_vector(dict(G_w.degree(weight="weight")))
    f05 = extract_edge_vector(compute_edge_strength(G_w, weight="weight"))

    try:
        ne_w = nx.eigenvector_centrality(G_w, weight="weight", max_iter=2000)
    except nx.NetworkXException:
        try:
            ne_w = nx.eigenvector_centrality_numpy(G_w, weight="weight")
        except (nx.NetworkXException, np.linalg.LinAlgError):
            ne_w = {n: 1.0 / len(G_w) for n in G_w.nodes()}
    f06 = extract_node_vector(ne_w)
    f07 = extract_edge_vector(compute_edge_eigenvector(G_w, weight="weight"))
    f08 = extract_node_vector(nx.closeness_centrality(G_w, distance="distance"))
    f09 = extract_edge_vector(compute_edge_closeness(G_w, distance="distance"))

    # 2. Binary Network via GCC adaptive thresholding (matching binary/*.ipynb)
    bin_df, thr, density = build_binary_network(df, step=0.05)
    binary = compute_binary_features(bin_df)

    return {
        "f01": f01, "f02": f02, "f03": f03, "f04": f04, "f05": f05,
        "f06": f06, "f07": f07, "f08": f08, "f09": f09, **binary,
        "tau_c": thr, "density": density
    }


def compute_binary_features(bin_df: pd.DataFrame) -> dict:
    """
    Computes the 9 binary-network features (f10 - f18) from one binary adjacency DataFrame.
    Used for the GCC-thresholded networks and for the fixed-density networks of the robustness checks.
    """
    G_b = nx.from_pandas_adjacency(bin_df)

    f10 = extract_matrix_edge_vector(bin_df)
    f11 = extract_node_vector(nx.betweenness_centrality(G_b, normalized=True))

    eb_b_raw = nx.edge_betweenness_centrality(G_b, normalized=True)
    eb_b = {tuple(sorted(k)): v for k, v in eb_b_raw.items()}
    f12 = extract_edge_vector(eb_b)

    f13 = extract_node_vector(dict(G_b.degree()))
    f14 = extract_edge_vector(compute_edge_strength(G_b, weight=None))

    try:
        ne_b = nx.eigenvector_centrality(G_b, max_iter=2000)
    except nx.NetworkXException:
        try:
            ne_b = nx.eigenvector_centrality_numpy(G_b)
        except (nx.NetworkXException, np.linalg.LinAlgError):
            ne_b = {n: 1.0 / len(G_b) for n in G_b.nodes()}
    f15 = extract_node_vector(ne_b)
    f16 = extract_edge_vector(compute_edge_eigenvector(G_b, weight=None))
    f17 = extract_node_vector(nx.closeness_centrality(G_b))
    f18 = extract_edge_vector(compute_edge_closeness(G_b, distance=None))

    return {
        "f10": f10, "f11": f11, "f12": f12, "f13": f13, "f14": f14,
        "f15": f15, "f16": f16, "f17": f17, "f18": f18
    }

# ----------------- EEG Filtering & Band Centralities -----------------

def load_annotations(subject_id: str, session_id: str) -> pd.DataFrame | None:
    """Load annotations TSV file for subject and session."""
    p = DERIVATIVES_PATH / subject_id / session_id / "eeg" / f"{subject_id}_{session_id}_task-EEG_annotations.tsv"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines()[1:]:
        fields = [f.strip() for f in line.split("\t")]
        if len(fields) >= 2 and fields[0] and fields[1]:
            try:
                label = fields[2] if len(fields) > 2 and fields[2] else "eyes_closed"
                rows.append([float(fields[0]), float(fields[1]), label])
            except ValueError:
                pass
    return pd.DataFrame(rows, columns=["onset", "duration", "label"]) if rows else None

def segment_eeg_from_annotations(raw: mne.io.BaseRaw, annotations_df: pd.DataFrame, label: str = "eyes_closed") -> list:
    """Segment continuous raw EEG data using annotations."""
    sfreq = float(raw.info["sfreq"])
    data: np.ndarray = np.asarray(raw.get_data())
    n_samples = data.shape[1]

    ann_labels = annotations_df["label"].fillna("").astype(str).str.lower().str.strip()
    mask = ann_labels.isin(["eyes_closed", "ojos_cerrados"]) if label == "eyes_closed" else (ann_labels == label.lower())

    segments = []
    for _, row in annotations_df[mask].iterrows():
        start = int(row["onset"] * sfreq)
        end = int((row["onset"] + row["duration"]) * sfreq)
        if 0 <= start <= end <= n_samples:
            segments.append(data[:, start:end])
    return segments

def extract_band_matrices(
    subject_id: str,
    session_id: str,
    l_freq: float,
    h_freq: float,
    n_epochs: int = 12,
    raw: mne.io.BaseRaw | None = None,
    ann_df: pd.DataFrame | None = None
):
    """
    Filters raw EDF in frequency band (l_freq - h_freq), extracts up to n_epochs,
    and returns list of (weighted_corr_df, binary_adj_df, tau_c, density).
    """
    if raw is None:
        edf_path = DATASET_PATH / subject_id / session_id / "eeg" / f"{subject_id}_{session_id}_task-EEG_eeg.edf"
        if not edf_path.exists():
            raise FileNotFoundError(f"Missing EDF: {edf_path}")
        raw_filtered = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw_filtered.pick(CHANNELS)
    else:
        raw_filtered = raw.copy()

    if ann_df is None:
        loaded_ann = load_annotations(subject_id, session_id)
        if loaded_ann is None:
            raise FileNotFoundError(f"Missing annotations for {subject_id} {session_id}")
        ann_df = loaded_ann

    # Filter in specified frequency band
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)

    segments = segment_eeg_from_annotations(raw_filtered, ann_df, label="eyes_closed")
    if len(segments) < n_epochs:
        raise ValueError(f"Insufficient segments: {len(segments)} < {n_epochs} for {subject_id} {session_id}")

    results = []
    for seg in segments[:n_epochs]:
        corr = np.corrcoef(seg)
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0)
        corr_df = pd.DataFrame(abs_corr, index=CHANNELS, columns=CHANNELS)
        bin_df, thr, density = build_binary_network(corr_df, step=0.05)
        results.append((corr_df, bin_df, thr, density))

    return results

# ----------------- Hemisphere Helpers -----------------

def get_node_hemisphere(node: str) -> str:
    """Classify node as Left (L), Right (R), or Midline (M)."""
    if node.endswith(("Z", "z")):
        return "M"
    digits = [c for c in node if c.isdigit()]
    if digits:
        val = int(digits[-1])
        return "R" if val % 2 == 0 else "L"
    return "M"

def get_edge_hemisphere_pair(edge: str) -> str:
    """Classify edge as L-L, R-R, L-R, L-M, R-M, M-M."""
    u, v = edge.split("-")
    h1 = get_node_hemisphere(u)
    h2 = get_node_hemisphere(v)
    order = {"L": 0, "R": 1, "M": 2}
    if order.get(h1, 0) <= order.get(h2, 0):
        pair = f"{h1}-{h2}"
    else:
        pair = f"{h2}-{h1}"
    return pair

# ----------------- Jaccard Similarity Functions -----------------

def element_J(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Computes element-wise weighted Jaccard index with 0/0 -> NaN rule.
    x1, x2: arrays of shape (n_pairs, n_elements) or (n_elements,)
    Returns nanmean across pairs: shape (n_elements,)
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    mn = np.minimum(x1, x2)
    mx = np.maximum(x1, x2)
    ok = mx > 0
    valid_count = np.sum(ok, axis=0)
    out = np.full(mn.shape[1], np.nan, dtype=np.float64)
    has_valid = valid_count > 0
    if np.any(has_valid):
        ratio_zero = np.zeros(mn.shape, dtype=np.float64)
        ratio_zero[ok] = mn[ok] / mx[ok]
        out[has_valid] = np.sum(ratio_zero, axis=0)[has_valid] / valid_count[has_valid]
    return out

def recording_S(x1: np.ndarray, x2: np.ndarray, zero_rule: bool = False) -> float:
    """
    Computes recording-level weighted Jaccard index across all elements:
    S = mean over pairs of [ sum(min(x1, x2)) / sum(max(x1, x2)) ]
    An epoch pair whose two vectors are all zero is 0/0: it is left out of the mean, as at element level
    (NaN if no pair is defined), or counted as 0 with zero_rule (the legacy rule, Step 8 check 1).
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    num = np.minimum(x1, x2).sum(axis=1)
    den = np.maximum(x1, x2).sum(axis=1)
    ok = den > 0
    if zero_rule:
        return float(np.sum(num[ok] / den[ok]) / len(den))
    return float(np.mean(num[ok] / den[ok])) if ok.any() else np.nan


def identification_accuracy(M: np.ndarray) -> tuple[float, float]:
    """
    1-NN identification accuracy in % (Step 4 item 4): ses-1 -> ses-2 (row maximum) and ses-2 -> ses-1 (column
    maximum). A tie for the maximum counts as incorrect; an undefined (NaN) cell is never the predicted identity.
    """
    M = np.where(np.isnan(M), -np.inf, M)
    idx = np.arange(M.shape[0])
    row_max, col_max = M.max(axis=1), M.max(axis=0)
    fwd = (np.argmax(M, axis=1) == idx) & (np.sum(M == row_max[:, None], axis=1) == 1)
    bwd = (np.argmax(M, axis=0) == idx) & (np.sum(M == col_max[None, :], axis=0) == 1)
    return float(np.mean(fwd) * 100.0), float(np.mean(bwd) * 100.0)

# ----------------- Infant Bootstrap (Step 7.2) -----------------

def bootstrap_draws(n: int, B: int, seed: int) -> np.ndarray:
    """
    B infant resamples: row i holds the n infant indices d_1 .. d_n drawn with replacement in resample i.
    Every bootstrap draws them from the one recorded seed, so all statistics share the same resamples and paired
    differences (Step 7.2 item 4) are always computed within the same resample.
    """
    return np.random.default_rng(seed).integers(0, n, (B, n))


def bootstrap_counts(draws: np.ndarray, n: int) -> np.ndarray:
    """
    How often each infant is drawn in each resample, shape (B, n). A mean over the drawn infants is the mean of the
    infants' values weighted by c_a (Step 7.2 item 2), and a mean over all position pairs p < q with d_p != d_q is
    the mean over infant pairs weighted by c_a * c_b (Step 7.2 items 1 and 3).
    """
    B = draws.shape[0]
    return np.bincount((draws + n * np.arange(B)[:, None]).ravel(), minlength=B * n).reshape(B, n).astype(np.float64)


def weighted_nanmean(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Weighted mean over the first axis of V for every row of weights W, leaving out NaN entries:
    W (B, m) with V (m,) or (m, d) gives (B,) or (B, d); NaN where no weighted entry is defined.
    """
    valid = ~np.isnan(V)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (W @ np.where(valid, V, 0.0)) / (W @ valid)


def bootstrap_ci(boot: np.ndarray) -> np.ndarray:
    """95% percentile CI: the 2.5th and 97.5th percentiles over the resamples (axis 0), NaN resamples left out."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a statistic undefined in every resample stays NaN
        return np.nanpercentile(boot, [2.5, 97.5], axis=0)


def bootstrap_p(boot: np.ndarray) -> np.ndarray:
    """
    Two-sided bootstrap p (Step 7.2 item 4) per statistic (axis 0 = resamples): 2 x min(fraction <= 0, fraction >= 0)
    with a floor of 1/B, at most 1. Undefined (NaN) resamples are left out; a statistic never defined gets p = 1.
    """
    B = boot.shape[0]
    valid = ~np.isnan(boot)
    n_valid = valid.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.minimum((valid & (boot <= 0)).sum(axis=0), (valid & (boot >= 0)).sum(axis=0)) / n_valid
    return np.where(n_valid > 0, np.clip(2.0 * frac, 1.0 / B, 1.0), 1.0)
