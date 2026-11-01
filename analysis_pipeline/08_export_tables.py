"""
Phase 8: Export Tables T1 through T6 (CSV & Markdown formats).
Compiles results from all preceding phases into publication-ready tables:
  - Table T1: Global Summary (26 features; each metric as mean +/- SD, median [95% CI], p and FDR q across features)
  - Table T1b: Interval and age correlations of Step 6 item 2 (rho with 95% CI, p and q), and the individually
    reported infants sub-NORB00064 and sub-NORB00069
  - Table T2: Element Results (3,230 element-level evaluations across features)
  - Table T2b: Regional universality U_s(g) per feature, session and region / region pair (Step 2 item 4)
  - Table T3: Variance Partition (Three-way LMM variance decomposition & age adjustments)
  - Table T4: Robustness & Sensitivity Checks (one row per check and feature, checks 1-8), and T4b (check 9)
  - Table T5: Cohort Bookkeeping (one row per infant: sex, ages and interval in days, epochs, twin status),
    and T5b: GCC threshold bookkeeping (tau_c and edge density per network and session)
  - Table T6: Sibling / Twin Analysis (Twin comparisons for sub-NORB00086 & sub-NORB00087)
"""
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
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
        N_EPOCHS,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        SESSIONS,
        SUBJECT_IDS,
        TABLES_DIR,
    )
    from analysis_pipeline.utils import (
        build_binary_network,
        get_edge_hemisphere_pair,
        get_node_hemisphere,
        load_annotations,
    )
else:
    from .config import (
        DATASET_PATH,
        EDGE_LABELS,
        FEATURE_DEFS,
        N_EPOCHS,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        SESSIONS,
        SUBJECT_IDS,
        TABLES_DIR,
    )
    from .utils import (
        build_binary_network,
        get_edge_hemisphere_pair,
        get_node_hemisphere,
        load_annotations,
    )


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as standard GitHub Flavored Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = [str(v) if pd.notna(v) else "" for v in row.values]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# Networks thresholded with the GCC rule: family -> (weighted feature, binary adjacency feature)
GCC_NETWORKS = {"broadband": ("f01", "f10"), "delta": ("f19", "f20"), "theta": ("f21", "f22"),
                "alpha": ("f23", "f24"), "beta": ("f25", "f26")}
TWIN_IDS = ("sub-NORB00086", "sub-NORB00087")
SHORT_GAP_IDS = ("sub-NORB00064", "sub-NORB00069")  # reported individually in Step 6


def fdr_across_features(p_values) -> np.ndarray:
    """Benjamini-Hochberg q-values of one global test across the feature sets (Step 7.4 ii)."""
    return np.asarray(multipletests(np.asarray(p_values, dtype=np.float64), method="fdr_bh")[1])


def fmt_ci(ci, spec: str = ".4f") -> str:
    return f"[{ci[0]:{spec}}, {ci[1]:{spec}}]"


def gcc_threshold_bookkeeping() -> pd.DataFrame:
    """
    Step 0 item 5: tau_c and edge count of every GCC-thresholded network, summarised (min, median, max) per
    network family and session. tau_c is recomputed from the weighted edge weights with the Phase 1 rule; the
    recomputed edge counts must equal the saved binary adjacency tensors.
    """
    node_idx = {node: i for i, node in enumerate(NODE_LABELS)}
    edge_uv = [(node_idx[e.split("-")[0]], node_idx[e.split("-")[1]]) for e in EDGE_LABELS]
    rows = []
    for family, (w_fid, b_fid) in GCC_NETWORKS.items():
        W = np.load(OUTPUT_TENSORS_DIR / f"X_{w_fid}.npy")
        A = np.load(OUTPUT_TENSORS_DIR / f"X_{b_fid}.npy")
        for s_idx, ses in enumerate(SESSIONS):
            taus, counts = [], []
            for a in range(W.shape[0]):
                for k in range(W.shape[2]):
                    mat = np.zeros((len(NODE_LABELS), len(NODE_LABELS)))
                    for e, (u, v) in enumerate(edge_uv):
                        mat[u, v] = mat[v, u] = W[a, s_idx, k, e]
                    _bin, tau, count = build_binary_network(pd.DataFrame(mat, index=NODE_LABELS, columns=NODE_LABELS))
                    if count != int(A[a, s_idx, k].sum()):
                        raise ValueError(f"Recomputed GCC network differs from X_{b_fid}: infant {a}, {ses}, epoch {k + 1}")
                    taus.append(tau)
                    counts.append(count)
            rows.append({
                "Network": family, "Session": ses, "Networks": len(taus),
                "tau_c min": min(taus), "tau_c median": float(np.median(taus)), "tau_c max": max(taus),
                "Edges min": min(counts), "Edges median": float(np.median(counts)), "Edges max": max(counts),
                "Density min": round(min(counts) / len(EDGE_LABELS), 4),
                "Density median": round(float(np.median(counts)) / len(EDGE_LABELS), 4),
                "Density max": round(max(counts) / len(EDGE_LABELS), 4),
            })
    return pd.DataFrame(rows)


def get_electrode_3d_distances_dict():
    """Returns mapping from edge label 'u-v' to Euclidean distance in mm."""
    elec_path = DATASET_PATH / "sub-NORB00001" / "ses-1" / "eeg" / "sub-NORB00001_ses-1_electrodes.tsv"
    if not elec_path.exists():
        raise FileNotFoundError(f"Electrode coordinates file not found: {elec_path}")
    df = pd.read_csv(elec_path, sep="\t")
    coords: dict[str, np.ndarray] = {}
    for name, x, y, z in zip(df["name"], df["x"], df["y"], df["z"]):
        if str(name) in NODE_LABELS:
            coords[str(name)] = np.array([float(x), float(y), float(z)])
            
    dist_map = {}
    for edge in EDGE_LABELS:
        u, v = edge.split("-")
        d = float(np.linalg.norm(coords[u] - coords[v]))
        dist_map[edge] = round(d, 2)
    return dist_map


def export_all_tables():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("PHASE 8: COMPILING & EXPORTING TABLES T1 THROUGH T6")
    print("=" * 80)
    
    # Load results databases
    u_file = OUTPUT_TENSORS_DIR / "universality_results.json"
    w_file = OUTPUT_TENSORS_DIR / "within_session_results.json"
    i_file = OUTPUT_TENSORS_DIR / "identifiability_results.json"
    tax_file = OUTPUT_TENSORS_DIR / "taxonomy_results.json"
    var_file = OUTPUT_TENSORS_DIR / "variance_results.json"
    dev_file = OUTPUT_TENSORS_DIR / "developmental_results.json"
    twin_file = OUTPUT_TENSORS_DIR / "twin_results.json"
    
    for fpath in (u_file, w_file, i_file, tax_file, var_file, dev_file, twin_file):
        if not fpath.exists():
            raise FileNotFoundError(f"Prerequisite results database missing: {fpath}. Run earlier pipeline phases first.")
            
    with open(u_file) as f:
        u_res = json.load(f)
    with open(w_file) as f:
        w_res = json.load(f)
    with open(i_file) as f:
        i_res = json.load(f)
    with open(tax_file) as f:
        tax_res = json.load(f)
    with open(var_file) as f:
        var_res = json.load(f)
    with open(dev_file) as f:
        dev_res = json.load(f)
    with open(twin_file) as f:
        twin_res = json.load(f)
        
    # Distance dictionary for Table T2
    edge_dists = get_electrode_3d_distances_dict()

    # ------------------------------------------------------------------
    # TABLE T1: GLOBAL SUMMARY (26 rows)
    # ------------------------------------------------------------------
    # Each metric as mean +/- SD (stating what the SD is over), median [95% bootstrap CI], and p with its
    # FDR q across the 26 feature sets (Step 7.1, 7.4 ii); I_diff and Delta U in Jaccard units (Step 7.5)
    p_triu, q_triu = np.triu_indices(len(SUBJECT_IDS), k=1)
    t1_rows = []
    for fid, meta in FEATURE_DEFS.items():
        u = u_res[fid]
        w = w_res[fid]
        ident = i_res[fid]
        dev = dev_res[fid]
        M = np.load(OUTPUT_TENSORS_DIR / f"M_mat_{fid}.npy")
        off = ~np.eye(M.shape[0], dtype=bool)
        S1 = np.load(OUTPUT_TENSORS_DIR / f"S_mat_{fid}_ses-1.npy")
        S2 = np.load(OUTPUT_TENSORS_DIR / f"S_mat_{fid}_ses-2.npy")
        delta_pairs = S2[p_triu, q_triu] - S1[p_triu, q_triu]

        row = {
            "Feature ID": fid,
            "Feature Name": meta["name"],
            "Network": meta["network"],
            "Type": meta["type"],
            "Dim": meta["dim"],
        }
        # Universality U_1, U_2 (SD over the 171 infant pairs)
        for s in (1, 2):
            us = u[f"ses-{s}"]
            row[f"U_{s} Mean ± SD (171 pairs)"] = f"{us['U_global']:.4f} ± {us['SD_pairs']:.4f}"
            row[f"U_{s} Median [95% CI]"] = f"{us['bootstrap']['median']:.4f} {fmt_ci(us['bootstrap']['ci_95'])}"
            row[f"U_{s} p"] = us["p_global"]
        # Delta U (SD of S_2 - S_1 over the 171 infant pairs)
        row["Delta U Mean ± SD (171 pairs)"] = f"{dev['delta_U_global']:+.4f} ± {np.std(delta_pairs):.4f}"
        row["Delta U Median [95% CI]"] = f"{dev['median_delta']:+.4f} {fmt_ci(dev['ci_95'], '+.4f')}"
        row["Delta U p"] = dev["p_value"]
        # Within-session stability W_1, W_2 and within-session identifiability W - O (SD over the 19 infants);
        # p is the two-sided infant bootstrap test of W - O against 0 (Step 3 item 4)
        for s in (1, 2):
            ws = w[f"ses-{s}"]
            wb = ws["W_bootstrap"]
            db = ws["diff_within_bootstrap"]
            row[f"W_{s} Mean ± SD (19 infants)"] = f"{ws['W_global']:.4f} ± {wb['sd']:.4f}"
            row[f"W_{s} Median [95% CI]"] = f"{wb['median']:.4f} {fmt_ci(wb['ci_95'])}"
            row[f"W_{s} - O_{s} Mean ± SD (19 infants)"] = f"{ws['diff_within_global']:+.4f} ± {db['sd']:.4f}"
            row[f"W_{s} - O_{s} Median [95% CI]"] = f"{db['median']:+.4f} {fmt_ci(db['ci_95'], '+.4f')}"
            row[f"W_{s} - O_{s} p"] = ws["diff_within_p"]
        # Identifiability (SD over the 19 diagonal cells, the 342 off-diagonal cells, and the 19 idiff(a))
        ib = ident["bootstrap"]
        row["I_self Mean ± SD (19 infants)"] = f"{ident['I_self']:.4f} ± {np.nanstd(np.diag(M)):.4f}"
        row["I_self Median [95% CI]"] = f"{ib['self_median']:.4f} {fmt_ci(ib['self_ci_95'])}"
        row["I_others Mean ± SD (342 cells)"] = f"{ident['I_others']:.4f} ± {np.nanstd(M[off]):.4f}"
        row["I_others Median [95% CI]"] = f"{ib['others_median']:.4f} {fmt_ci(ib['others_ci_95'])}"
        row["I_diff Mean ± SD (19 infants)"] = f"{ident['I_diff']:+.4f} ± {np.nanstd(ident['per_infant_idiff']):.4f}"
        row["I_diff Median [95% CI]"] = f"{ib['idiff_median']:+.4f} {fmt_ci(ib['idiff_ci_95'], '+.4f')}"
        row["I_diff p"] = ident["p_idiff"]
        # 1-NN identification accuracy (permutation p; accuracy is not bootstrapped, Step 7.2 item 5)
        row["Acc Forward (%)"] = round(ident["acc_forward"], 1)
        row["Acc Forward p"] = ident["p_acc_forward"]
        row["Acc Backward (%)"] = round(ident["acc_backward"], 1)
        row["Acc Backward p"] = ident["p_acc_backward"]
        row["Acc Mean (%)"] = round(ident["acc_mean"], 1)
        row["Acc Mean p"] = ident["p_acc_mean"]
        # Mantel test: similarity vs age difference, per session, with its infant pair bootstrap CI
        for s in (1, 2):
            mantel = dev[f"mantel_ses{s}"]
            row[f"Mantel rho ses-{s} [95% CI]"] = f"{mantel['rho']:+.4f} {fmt_ci(mantel['ci_95'], '+.4f')}"
            row[f"Mantel ses-{s} p"] = mantel["p"]
        t1_rows.append(row)
    df_t1 = pd.DataFrame(t1_rows)
    # FDR q next to every p column, across the 26 feature sets
    t1_cols = []
    for col in df_t1.columns:
        t1_cols.append(col)
        if col.endswith(" p"):
            q_col = col[:-2] + " q"
            df_t1[q_col] = fdr_across_features(df_t1[col])
            t1_cols.append(q_col)
    df_t1 = df_t1[t1_cols]
    df_t1.to_csv(TABLES_DIR / "Table_T1_Global_Summary.csv", index=False)
    (TABLES_DIR / "Table_T1_Global_Summary.md").write_text(df_to_markdown(df_t1))
    print(f" -> Table T1 saved: {TABLES_DIR / 'Table_T1_Global_Summary.csv'} (.csv & .md)")

    # ------------------------------------------------------------------
    # TABLE T1b: INTERVAL AND AGE CORRELATIONS (26 rows, Step 6 item 2)
    # ------------------------------------------------------------------
    # Spearman rho over the 19 infants with its infant bootstrap CI (Step 7.5), permutation p and FDR q across the
    # 26 feature sets, and the two infants reported individually (Step 4 item 7, Step 6 item 2)
    corr_labels = [("self_vs_interval", "rho(self, interval)"), ("idiff_vs_interval", "rho(idiff, interval)"),
                   ("self_vs_age1", "rho(self, age ses-1)"), ("idiff_vs_age1", "rho(idiff, age ses-1)")]
    t1b_rows = []
    for fid, meta in FEATURE_DEFS.items():
        dev = dev_res[fid]
        row = {"Feature ID": fid, "Feature Name": meta["name"]}
        for key, label in corr_labels:
            corr = dev["interval_corr"][key]
            row[f"{label} [95% CI]"] = f"{corr['rho']:+.4f} {fmt_ci(corr['ci_95'], '+.4f')}"
            row[f"{label} p"] = corr["p"]
            row[f"{label} q"] = corr["q"]
        for subj in SHORT_GAP_IDS:
            ind = dev["individual_infants"][subj]
            row[f"{subj} interval (days)"] = round(ind["gap_days"], 1)
            row[f"{subj} self(a)"] = round(ind["self"], 4)
            row[f"{subj} idiff(a)"] = round(ind["idiff"], 4)
        t1b_rows.append(row)
    df_t1b = pd.DataFrame(t1b_rows)
    df_t1b.to_csv(TABLES_DIR / "Table_T1b_Interval_Correlations.csv", index=False)
    (TABLES_DIR / "Table_T1b_Interval_Correlations.md").write_text(df_to_markdown(df_t1b))
    print(f" -> Table T1b saved: {TABLES_DIR / 'Table_T1b_Interval_Correlations.csv'} (.csv & .md)")

    # ------------------------------------------------------------------
    # TABLE T2: ELEMENT RESULTS (3,230 rows)
    # ------------------------------------------------------------------
    t2_rows = []
    for fid, meta in FEATURE_DEFS.items():
        fname = meta["name"]
        dim = meta["dim"]
        ftype = meta["type"]
        labels = NODE_LABELS if ftype == "node" else EDGE_LABELS
        
        u = u_res[fid]
        w = w_res[fid]
        ident = i_res[fid]
        tax = tax_res[fid]
        shift = u["developmental_shift"]

        for e in range(dim):
            label = labels[e]
            if ftype == "node":
                reg = NODE_REGIONS.get(label, "Unknown")
                hemi = get_node_hemisphere(label)
                wb_flag = "Node"
                e_dist = "N/A"
            else:
                u_node, v_node = label.split("-")
                r1, r2 = NODE_REGIONS.get(u_node, "F"), NODE_REGIONS.get(v_node, "F")
                reg = f"{r1}-{r2}"
                hemi = get_edge_hemisphere_pair(label)
                wb_flag = "Within" if r1 == r2 else "Between"
                e_dist = edge_dists.get(label, "N/A")
                
            row = {
                "Feature ID": fid,
                "Feature Name": fname,
                "Element": label,
                "Region": reg,
                "Hemisphere": hemi,
                "Connection Type": wb_flag,
                "Euclidean Distance (mm)": e_dist,
                "U_1(e)": round(u["ses-1"]["U_elem"][e], 4),
                "p_raw_U_1": round(u["ses-1"]["p_elem"][e], 6),
                "q_fdr_U_1": round(u["ses-1"]["q_elem"][e], 6),
                "U_2(e)": round(u["ses-2"]["U_elem"][e], 4),
                "p_raw_U_2": round(u["ses-2"]["p_elem"][e], 6),
                "q_fdr_U_2": round(u["ses-2"]["q_elem"][e], 6),
                # Delta U(e) = U_2(e) - U_1(e), paired infant bootstrap (Step 6 item 1)
                "Delta U(e)": round(shift["delta_U_elem"][e], 4),
                "p_raw_delta_U": round(shift["delta_U_elem_p"][e], 6),
                "q_fdr_delta_U": round(shift["delta_U_elem_q"][e], 6),
                "W_1(e)": round(w["ses-1"]["W_elem"][e], 4),
                "W_2(e)": round(w["ses-2"]["W_elem"][e], 4),
            }
            # Within-session identifiability W_s(e) - O_s(e), infant bootstrap (Step 3 item 4)
            for s in (1, 2):
                ws = w[f"ses-{s}"]
                row[f"W_{s}(e) - O_{s}(e)"] = round(ws["diff_within_elem"][e], 4)
                row[f"p_raw_W_O_{s}"] = round(ws["diff_within_elem_p"][e], 6)
                row[f"q_fdr_W_O_{s}"] = round(ws["diff_within_elem_q"][e], 6)
            row.update({
                "I_diff(e)": round(ident["I_diff_elem"][e], 4),
                "p_raw_idiff": round(ident["p_elem"][e], 6),
                "q_fdr_idiff": round(ident["q_elem"][e], 6),
                "Taxonomy Class": tax["classes"][e],
                "Undefined (0/0) Frac ses-1": round(u["ses-1"]["undef_frac"][e], 4),
                "Undefined (0/0) Frac ses-2": round(u["ses-2"]["undef_frac"][e], 4)
            })
            t2_rows.append(row)
    df_t2 = pd.DataFrame(t2_rows)
    df_t2.to_csv(TABLES_DIR / "Table_T2_Element_Results.csv", index=False)
    print(f" -> Table T2 saved: {TABLES_DIR / 'Table_T2_Element_Results.csv'} (3,230 element evaluations)")

    # ------------------------------------------------------------------
    # TABLE T2b: REGIONAL UNIVERSALITY (feature x session x region, Step 2 item 4)
    # ------------------------------------------------------------------
    # U_s(g) is the column mean of the pairs x regions matrix of J; SD over the 171 infant pairs, median and
    # 95% CI from the infant pair bootstrap (Step 7.2 item 1)
    region_sizes = df_t2.groupby(["Feature ID", "Region"]).size()
    t2b_rows = []
    for fid, meta in FEATURE_DEFS.items():
        for ses in SESSIONS:
            us = u_res[fid][ses]
            for g in us["regions"]:
                t2b_rows.append({
                    "Feature ID": fid,
                    "Feature Name": meta["name"],
                    "Session": ses,
                    "Region": g,
                    "Elements": int(region_sizes[(fid, g)]),
                    "U_s(g) Mean ± SD (171 pairs)": f"{us['regional_U'][g]:.4f} ± {us['regional_U_SD_pairs'][g]:.4f}",
                    "U_s(g) Median [95% CI]": f"{us['regional_U_median'][g]:.4f} {fmt_ci(us['regional_U_ci_95'][g])}"
                })
    df_t2b = pd.DataFrame(t2b_rows)
    df_t2b.to_csv(TABLES_DIR / "Table_T2b_Regional_Universality.csv", index=False)
    print(f" -> Table T2b saved: {TABLES_DIR / 'Table_T2b_Regional_Universality.csv'}")
    
    # ------------------------------------------------------------------
    # TABLE T3: VARIANCE PARTITION (3,230 rows)
    # ------------------------------------------------------------------
    t3_rows = []
    for fid, meta in FEATURE_DEFS.items():
        vdata = var_res[fid]["elements"]
        for elem_dict in vdata:
            t3_rows.append({
                "Feature ID": fid,
                "Feature Name": meta["name"],
                "Element": elem_dict["element"],
                "Infant (Trait) Prop": round(elem_dict["trait_prop"], 4) if not np.isnan(elem_dict["trait_prop"]) else "NaN",
                "Session Prop": round(elem_dict["session_prop"], 4) if not np.isnan(elem_dict["session_prop"]) else "NaN",
                "Epoch (State) Prop": round(elem_dict["epoch_prop"], 4) if not np.isnan(elem_dict["epoch_prop"]) else "NaN",
                "Age-Adjusted Infant Prop": round(elem_dict["trait_prop_adj"], 4) if not np.isnan(elem_dict["trait_prop_adj"]) else "NaN",
                "Age-Adjusted Session Prop": round(elem_dict["session_prop_adj"], 4) if not np.isnan(elem_dict["session_prop_adj"]) else "NaN",
                "Age-Adjusted Epoch Prop": round(elem_dict["epoch_prop_adj"], 4) if not np.isnan(elem_dict["epoch_prop_adj"]) else "NaN",
                # One flag per model (Phase 5); results saved before that change only have the combined flag
                "Converged (unadjusted)": elem_dict.get("converged_unadj", elem_dict["converged"]),
                "Converged (age-adjusted)": elem_dict.get("converged_adj", elem_dict["converged"]),
                "Skipped Binary Adjacency": elem_dict["skipped"]
            })
    df_t3 = pd.DataFrame(t3_rows)
    df_t3.to_csv(TABLES_DIR / "Table_T3_Variance_Partition.csv", index=False)
    print(f" -> Table T3 saved: {TABLES_DIR / 'Table_T3_Variance_Partition.csv'}")
    
    # ------------------------------------------------------------------
    # TABLES T4 & T4b: ROBUSTNESS & SENSITIVITY (written as CSV by Phase 7)
    # ------------------------------------------------------------------
    for name in ("Table_T4_Robustness", "Table_T4b_Electrode_Distance"):
        t4_csv = TABLES_DIR / f"{name}.csv"
        if t4_csv.exists():
            df_t4 = pd.read_csv(t4_csv)
            (TABLES_DIR / f"{name}.md").write_text(df_to_markdown(df_t4))
            print(f" -> {name} saved: {t4_csv} (.csv & .md)")
        else:
            print(f" -> {name} missing; please run Phase 7 first.")

    # ------------------------------------------------------------------
    # TABLE T5: SUBJECT BOOKKEEPING (19 rows, Step 0 item 6)
    # ------------------------------------------------------------------
    participants = pd.read_csv(DATASET_PATH / "participants.tsv", sep="\t").set_index("participant_id")
    t5_rows = []
    for subj in SUBJECT_IDS:
        ages_days = {}
        annotated, usable = {}, {}
        for ses in SESSIONS:
            scans = pd.read_csv(DATASET_PATH / subj / ses / f"{subj}_{ses}_scans.tsv", sep="\t")
            ages_days[ses] = float(scans["age_acq_time"].iloc[0]) * 365.25
            ann_df = load_annotations(subj, ses)
            annotated[ses] = len(ann_df) if ann_df is not None else 0
            # Usable epochs: annotated windows that lie inside the recording (same rule as segment_eeg_from_annotations)
            edf_path = DATASET_PATH / subj / ses / "eeg" / f"{subj}_{ses}_task-EEG_eeg.edf"
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            sfreq, n_times = float(raw.info["sfreq"]), raw.n_times
            if ann_df is None:
                usable[ses] = 0
            else:
                onsets = ann_df["onset"].to_numpy(dtype=np.float64)
                starts = (onsets * sfreq).astype(int)
                ends = ((onsets + ann_df["duration"].to_numpy(dtype=np.float64)) * sfreq).astype(int)
                usable[ses] = int(np.sum((starts >= 0) & (starts <= ends) & (ends <= n_times)))
        gap_days = ages_days["ses-2"] - ages_days["ses-1"]
        
        if subj in TWIN_IDS:
            sibling = TWIN_IDS[1] if subj == TWIN_IDS[0] else TWIN_IDS[0]
            status = "Twin A" if subj == TWIN_IDS[0] else "Twin B"
            note = f"Twin of {sibling}; flagged in all outputs, excluded in robustness check 4"
        else:
            status = "Singleton"
            note = f"Short inter-session gap ({gap_days:.0f} days), reported individually in Step 6" if subj in SHORT_GAP_IDS else ""
        
        t5_rows.append({
            "Participant ID": subj,
            "Sex": participants.loc[subj, "sex"],
            "Age ses-1 (days)": round(ages_days["ses-1"], 1),
            "Age ses-2 (days)": round(ages_days["ses-2"], 1),
            "Interval (days)": round(gap_days, 1),
            "Annotated epochs ses-1": annotated["ses-1"],
            "Usable epochs ses-1": usable["ses-1"],
            "Annotated epochs ses-2": annotated["ses-2"],
            "Usable epochs ses-2": usable["ses-2"],
            "Epochs used per session": N_EPOCHS,
            "Status": status,
            "Notes": note
        })
    df_t5 = pd.DataFrame(t5_rows)
    df_t5.to_csv(TABLES_DIR / "Table_T5_Bookkeeping.csv", index=False)
    (TABLES_DIR / "Table_T5_Bookkeeping.md").write_text(df_to_markdown(df_t5))
    print(f" -> Table T5 saved: {TABLES_DIR / 'Table_T5_Bookkeeping.csv'} (.csv & .md)")
    
    # ------------------------------------------------------------------
    # TABLE T5b: GCC THRESHOLD BOOKKEEPING (network family x session, Step 0 item 5)
    # ------------------------------------------------------------------
    df_t5b = gcc_threshold_bookkeeping()
    df_t5b.to_csv(TABLES_DIR / "Table_T5b_Threshold_Bookkeeping.csv", index=False)
    (TABLES_DIR / "Table_T5b_Threshold_Bookkeeping.md").write_text(df_to_markdown(df_t5b))
    print(f" -> Table T5b saved: {TABLES_DIR / 'Table_T5b_Threshold_Bookkeeping.csv'} (.csv & .md)")
    
    # ------------------------------------------------------------------
    # TABLE T6: SIBLING / TWIN ANALYSIS (26 rows, all values of the Step 6 sibling analysis)
    # ------------------------------------------------------------------
    def set_str(summary):
        """Median [min, max] of a matched comparison set."""
        if summary is None or summary.get("n", 0) == 0:
            return "n/a"
        return f"{summary['median']:.4f} [{summary['min']:.4f}, {summary['max']:.4f}] (n = {summary['n']})"
    
    def rank_or_na(value, digits=1):
        return "N/A" if value is None or np.isnan(value) else round(value, digits)
    
    t6_rows = []
    for fid, meta in FEATURE_DEFS.items():
        tw = twin_res[fid]
        c = tw["cross_identification"]
        # Which sibling is identified as the other, and in which direction (row: ses-1 -> ses-2, column: ses-2 -> ses-1)
        directions = []
        for who, other in (("86", "87"), ("87", "86")):
            key = f"misidentified_{who}_as_{other}"
            if f"{key}_fwd" in c:
                directions += [f"{who} as {other} ({d})" for d, flag in (("forward", c[f"{key}_fwd"]), ("backward", c[f"{key}_bwd"])) if flag]
            elif c[key]:  # results saved before both directions were checked
                directions.append(f"{who} as {other} (forward)")
        shared_classes = tw.get("shared_element_classes")
        if shared_classes is not None:
            shared_str = ", ".join(f"{label} ({cls})" for label, cls in shared_classes.items()) or "None"
        else:
            shared_str = ", ".join(tw["shared_elements"]) or "None"
            
        t6_rows.append({
            "Feature ID": fid,
            "Feature Name": meta["name"],
            "Twin Similarity ses-1": round(tw["ses1_similarity"], 4),
            "Twin Similarity ses-2": round(tw["ses2_similarity"], 4),
            "Percentile Rank ses-1 (/171)": rank_or_na(tw["rank_among_171_ses1"]),
            "Percentile Rank ses-2 (/171)": rank_or_na(tw["rank_among_171_ses2"]),
            "Matched Infant Rank ses-1": rank_or_na(tw["rank_matched_infant_ses1"]),
            "Matched Infant Set ses-1 Median [Range]": set_str(tw.get("matched_infant_set_ses1")),
            "Matched Infant Rank ses-2": rank_or_na(tw["rank_matched_infant_ses2"]),
            "Matched Infant Set ses-2 Median [Range]": set_str(tw.get("matched_infant_set_ses2")),
            "Matched Age Rank ses-1": rank_or_na(tw.get("rank_matched_age_ses1", np.nan)),
            "Matched Age Set ses-1 Median [Range]": set_str(tw.get("matched_age_set_ses1")),
            "Matched Age Rank ses-2": rank_or_na(tw.get("rank_matched_age_ses2", np.nan)),
            "Matched Age Set ses-2 Median [Range]": set_str(tw.get("matched_age_set_ses2")),
            "M[86, 86] Self": round(c["M_86_86"], 4),
            "M[87, 87] Self": round(c["M_87_87"], 4),
            "M[86, 87] Cross": round(c["M_86_87"], 4),
            "M[87, 86] Cross": round(c["M_87_86"], 4),
            "M[86, 87] Rank in Row 86": c["rank_87_for_86"],
            "M[86, 87] Rank in Column 87": c["rank_87_in_col"],
            "M[87, 86] Rank in Row 87": c["rank_86_for_87"],
            "M[87, 86] Rank in Column 86": c["rank_86_in_col"],
            "Identified as the Other Sibling": "; ".join(directions) or "None",
            "Elements Where Either Is Identified as the Other": tw.get("n_elements_misidentified", "n/a"),
            "Shared P95 Elements Count": tw["n_shared_p95_elements"],
            "Shared P95 Individual Elements": tw.get("n_shared_individual_elements", "n/a"),
            "Shared P95 Elements (Class)": shared_str,
            "Same-Day Control": tw.get("same_day_control", "n/a")
        })
    df_t6 = pd.DataFrame(t6_rows)
    df_t6.to_csv(TABLES_DIR / "Table_T6_Sibling_Analysis.csv", index=False)
    (TABLES_DIR / "Table_T6_Sibling_Analysis.md").write_text(df_to_markdown(df_t6))
    print(f" -> Table T6 saved: {TABLES_DIR / 'Table_T6_Sibling_Analysis.csv'} (.csv & .md)")
    
    print("=" * 80)
    print("ALL TABLES T1 THROUGH T6 SUCCESSFULLY EXPORTED TO tables/ DIRECTORY!")
    print("=" * 80)


if __name__ == "__main__":
    export_all_tables()
