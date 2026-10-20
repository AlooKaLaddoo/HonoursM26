"""
Verification script: compares the extracted feature tensors against the exact calculations
and logic from the original Jupyter notebooks in weighted/, binary/, and spectralBandSpecificConnectivity/.
"""
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from analysis_pipeline.config import (
    SUBJECT_IDS, SESSIONS, N_EPOCHS, NODE_LABELS, EDGE_LABELS,
    CORR_MATRICES_DIR, OUTPUT_TENSORS_DIR, FREQUENCY_BANDS
)
from analysis_pipeline.utils import (
    build_binary_network, extract_band_matrices
)

def verify_all_features():
    print("=" * 80)
    print("VERIFICATION: EXTRACTED TENSORS vs ORIGINAL JUPYTER NOTEBOOK IMPLEMENTATIONS")
    print("=" * 80)
    
    # Representative test subjects and epochs
    test_subjs = ["sub-NORB00064", "sub-NORB00075", "sub-NORB00086", "sub-NORB00103"]
    test_epochs = [0, 5, 11] # first, middle, last epoch
    
    report = []
    
    # ------------------------------------------------------------------
    # 1. BROADBAND WEIGHTED & BINARY NOTEBOOKS
    # ------------------------------------------------------------------
    for fid, name, ftype, is_binary in [
        ("f01", "correlation (f01)", "edge", False),
        ("f02", "node betweenness (f02)", "node", False),
        ("f03", "edge betweenness (f03)", "edge", False),
        ("f04", "node strength (f04)", "node", False),
        ("f05", "edge strength (f05)", "edge", False),
        ("f06", "node eigenvector (f06)", "node", False),
        ("f07", "edge eigenvector (f07)", "edge", False),
        ("f08", "node closeness (f08)", "node", False),
        ("f09", "edge closeness (f09)", "edge", False),
        ("f10", "binary adjacency (f10)", "edge", True),
        ("f11", "binary node betweenness (f11)", "node", True),
        ("f12", "binary edge betweenness (f12)", "edge", True),
        ("f13", "binary node degree (f13)", "node", True),
        ("f14", "binary edge strength (f14)", "edge", True),
        ("f15", "binary node eigenvector (f15)", "node", True),
        ("f16", "binary edge eigenvector (f16)", "edge", True),
        ("f17", "binary node closeness (f17)", "node", True),
        ("f18", "binary edge closeness (f18)", "edge", True),
    ]:
        tensor = np.load(OUTPUT_TENSORS_DIR / f"X_{fid}.npy")
        max_diff = 0.0
        
        for subj in test_subjs:
            sub_idx = SUBJECT_IDS.index(subj)
            for ses_idx, ses in enumerate(SESSIONS):
                csv_files = sorted((CORR_MATRICES_DIR / subj / ses).glob("*.csv"))
                for ep_idx in test_epochs:
                    df = pd.read_csv(csv_files[ep_idx], index_col=0)
                    
                    if not is_binary:
                        # Exact weighted notebook logic: G from df directly
                        G = nx.from_pandas_adjacency(df)
                        for u, v, d in G.edges(data=True):
                            w = d.get("weight", 0.0)
                            d["distance"] = 1.0 / w if w > 0 else np.inf
                            
                        if fid == "f01":
                            nb_vec = np.array([df.loc[e.split("-")[0], e.split("-")[1]] for e in EDGE_LABELS])
                        elif fid == "f02":
                            nb_dict = nx.betweenness_centrality(G, weight="distance", normalized=True)
                            nb_vec = np.array([nb_dict[n] for n in NODE_LABELS])
                        elif fid == "f03":
                            eb_raw = nx.edge_betweenness_centrality(G, weight="distance", normalized=True)
                            eb = {tuple(sorted(k)): v for k, v in eb_raw.items()}
                            nb_vec = np.array([eb[tuple(sorted(e.split("-")))] for e in EDGE_LABELS])
                        elif fid == "f04":
                            deg = dict(G.degree(weight="weight"))
                            nb_vec = np.array([deg[n] for n in NODE_LABELS])
                        elif fid == "f05":
                            deg = dict(G.degree(weight="weight"))
                            es = {}
                            for u, v, d in G.edges(data=True):
                                wk = d.get("weight", 0)
                                ku, kv = deg[u], deg[v]
                                es[tuple(sorted([u, v]))] = ((ku + kv - 2 * wk) / (abs(ku - kv) + 1)) * wk
                            nb_vec = np.array([es[tuple(sorted(e.split("-")))] for e in EDGE_LABELS])
                        elif fid == "f06":
                            ne = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
                            nb_vec = np.array([ne[n] for n in NODE_LABELS])
                        elif fid == "f07":
                            L = nx.line_graph(G)
                            for e1, e2 in L.edges():
                                w1 = G[e1[0]][e1[1]].get("weight", 0)
                                w2 = G[e2[0]][e2[1]].get("weight", 0)
                                L[e1][e2]["weight"] = (w1 + w2) / 2.0
                            ee_raw = nx.eigenvector_centrality(L, weight="weight", max_iter=2000)
                            ee = {tuple(sorted(list(k))): v for k, v in ee_raw.items()}
                            nb_vec = np.array([ee[tuple(sorted(e.split("-")))] for e in EDGE_LABELS])
                        elif fid == "f08":
                            nc = nx.closeness_centrality(G, distance="distance")
                            nb_vec = np.array([nc[n] for n in NODE_LABELS])
                        elif fid == "f09":
                            nc = nx.closeness_centrality(G, distance="distance")
                            E = G.number_of_edges()
                            ec = {}
                            for u, v in G.edges():
                                cu, cv = nc.get(u, 0.0), nc.get(v, 0.0)
                                ec[tuple(sorted([u, v]))] = (E - 1) * (cu * cv) / (cu + cv) if (cu + cv) > 0 else 0.0
                            nb_vec = np.array([ec[tuple(sorted(e.split("-")))] for e in EDGE_LABELS])
                    else:
                        # Exact binary notebook logic: build_binary_network with 0 diagonal
                        bin_df, _, _ = build_binary_network(df, step=0.05)
                        G_b = nx.from_pandas_adjacency(bin_df)
                        
                        if fid == "f10":
                            nb_vec = np.array([bin_df.loc[e.split("-")[0], e.split("-")[1]] for e in EDGE_LABELS])
                        elif fid == "f11":
                            nb_dict = nx.betweenness_centrality(G_b, normalized=True)
                            nb_vec = np.array([nb_dict[n] for n in NODE_LABELS])
                        elif fid == "f12":
                            eb_raw = nx.edge_betweenness_centrality(G_b, normalized=True)
                            eb = {tuple(sorted(k)): v for k, v in eb_raw.items()}
                            nb_vec = np.array([eb.get(tuple(sorted(e.split("-"))), 0.0) for e in EDGE_LABELS])
                        elif fid == "f13":
                            deg = dict(G_b.degree())
                            nb_vec = np.array([deg[n] for n in NODE_LABELS])
                        elif fid == "f14":
                            deg = dict(G_b.degree())
                            es = {}
                            for u, v in G_b.edges():
                                ku, kv = deg[u], deg[v]
                                es[tuple(sorted([u, v]))] = (ku + kv - 2.0) / (abs(ku - kv) + 1.0)
                            nb_vec = np.array([es.get(tuple(sorted(e.split("-"))), 0.0) for e in EDGE_LABELS])
                        elif fid == "f15":
                            ne = nx.eigenvector_centrality(G_b, max_iter=2000)
                            nb_vec = np.array([ne[n] for n in NODE_LABELS])
                        elif fid == "f16":
                            L = nx.line_graph(G_b)
                            ee_raw = nx.eigenvector_centrality(L, max_iter=2000)
                            ee = {tuple(sorted(list(k))): v for k, v in ee_raw.items()}
                            nb_vec = np.array([ee.get(tuple(sorted(e.split("-"))), 0.0) for e in EDGE_LABELS])
                        elif fid == "f17":
                            nc = nx.closeness_centrality(G_b)
                            nb_vec = np.array([nc[n] for n in NODE_LABELS])
                        elif fid == "f18":
                            nc = nx.closeness_centrality(G_b)
                            E = G_b.number_of_edges()
                            ec = {}
                            for u, v in G_b.edges():
                                cu, cv = nc.get(u, 0.0), nc.get(v, 0.0)
                                ec[tuple(sorted([u, v]))] = (E - 1) * (cu * cv) / (cu + cv) if (cu + cv) > 0 else 0.0
                            nb_vec = np.array([ec.get(tuple(sorted(e.split("-"))), 0.0) for e in EDGE_LABELS])
                            
                    t_vec = tensor[sub_idx, ses_idx, ep_idx, :]
                    diff = np.max(np.abs(t_vec - nb_vec))
                    if diff > max_diff:
                        max_diff = diff
                        
        status = "PASSED (Exact Match)" if max_diff < 1e-12 else f"DIFF: {max_diff:.2e}"
        report.append({"fid": fid, "name": name, "max_diff": max_diff, "status": status})
        print(f"[{fid}] {name:<35} | Max Absolute Difference: {max_diff:.2e} | Status: {status}")
        
    # ------------------------------------------------------------------
    # 2. SPECTRAL BAND NOTEBOOKS (f19 - f26)
    # ------------------------------------------------------------------
    print("\n--- Frequency Bands Verification ---")
    band_checks = [
        ("delta", "f19", "f20", (0.5, 4.0)),
        ("theta", "f21", "f22", (4.0, 8.0)),
        ("alpha", "f23", "f24", (8.0, 13.0)),
        ("beta",  "f25", "f26", (13.0, 30.0))
    ]
    for band_name, w_fid, b_fid, (l_freq, h_freq) in band_checks:
        w_tensor = np.load(OUTPUT_TENSORS_DIR / f"X_{w_fid}.npy")
        b_tensor = np.load(OUTPUT_TENSORS_DIR / f"X_{b_fid}.npy")
        
        # Test sub-NORB00064 ses-1
        subj = "sub-NORB00064"
        sub_idx = SUBJECT_IDS.index(subj)
        band_results = extract_band_matrices(subj, "ses-1", l_freq, h_freq, n_epochs=12)
        
        max_diff_w = 0.0
        max_diff_b = 0.0
        for ep_idx in [0, 5, 11]:
            corr_df, bin_df, _, _ = band_results[ep_idx]
            nb_w = np.array([corr_df.loc[e.split("-")[0], e.split("-")[1]] for e in EDGE_LABELS])
            nb_b = np.array([bin_df.loc[e.split("-")[0], e.split("-")[1]] for e in EDGE_LABELS])
            
            diff_w = np.max(np.abs(w_tensor[sub_idx, 0, ep_idx, :] - nb_w))
            diff_b = np.max(np.abs(b_tensor[sub_idx, 0, ep_idx, :] - nb_b))
            max_diff_w = max(max_diff_w, diff_w)
            max_diff_b = max(max_diff_b, diff_b)
            
        status_w = "PASSED (Exact Match)" if max_diff_w < 1e-12 else f"DIFF: {max_diff_w:.2e}"
        status_b = "PASSED (Exact Match)" if max_diff_b < 1e-12 else f"DIFF: {max_diff_b:.2e}"
        print(f"[{w_fid}] {band_name.capitalize()} weighted corr (f{w_fid[1:]}) | Max Diff: {max_diff_w:.2e} | Status: {status_w}")
        print(f"[{b_fid}] {band_name.capitalize()} binary adj     (f{b_fid[1:]}) | Max Diff: {max_diff_b:.2e} | Status: {status_b}")
        report.append({"fid": w_fid, "name": f"{band_name}_weighted", "max_diff": max_diff_w, "status": status_w})
        report.append({"fid": b_fid, "name": f"{band_name}_binary", "max_diff": max_diff_b, "status": status_b})
        
    print("=" * 80)
    all_passed = all(r["max_diff"] < 1e-12 for r in report)
    if all_passed:
        print("RESULT: ALL 26 FEATURE TENSORS PASS VERIFICATION WITH EXACT 0.00 DIFFERENCE!")
    else:
        print("RESULT: Some features had differences. See details above.")
    print("=" * 80)

if __name__ == "__main__":
    verify_all_features()
