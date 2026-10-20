"""
Phase 8: Export Tables T1 through T6 (CSV & Markdown formats).
Compiles results from all preceding phases into publication-ready tables:
  - Table T1: Global Summary (26 features x 18 metrics)
  - Table T2: Element Results (3,230 element-level evaluations across features)
  - Table T3: Variance Partition (Three-way LMM variance decomposition & age adjustments)
  - Table T4: Robustness & Sensitivity Checks (88 perturbation conditions across 9 checks)
  - Table T5: Cohort Bookkeeping (Subject ages, windows, twin status, and notes)
  - Table T6: Sibling / Twin Analysis (Twin comparisons for sub-NORB00086 & sub-NORB00087)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
    SUBJECT_IDS, SESSIONS, NODE_LABELS, EDGE_LABELS,
    NODE_REGIONS, REGION_PAIRS, FEATURE_DEFS,
    OUTPUT_TENSORS_DIR, TABLES_DIR, DATASET_PATH
)
from .utils import load_annotations, get_node_hemisphere, get_edge_hemisphere_pair

def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as standard GitHub Flavored Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = [str(v) if pd.notna(v) else "" for v in row.values]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def get_electrode_3d_distances_dict():
    """Returns mapping from edge label 'u-v' to Euclidean distance in mm."""
    elec_path = DATASET_PATH / "sub-NORB00001" / "ses-1" / "eeg" / "sub-NORB00001_ses-1_electrodes.tsv"
    df = pd.read_csv(elec_path, sep="\t")
    coords = {}
    for _, row in df.iterrows():
        if row["name"] in NODE_LABELS:
            coords[row["name"]] = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            
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
    with open(OUTPUT_TENSORS_DIR / "universality_results.json") as f:
        u_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "within_session_results.json") as f:
        w_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "identifiability_results.json") as f:
        i_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "taxonomy_results.json") as f:
        tax_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "variance_results.json") as f:
        var_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "developmental_results.json") as f:
        dev_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "twin_results.json") as f:
        twin_res = json.load(f)
        
    # Distance dictionary for Table T2
    edge_dists = get_electrode_3d_distances_dict()

    # ------------------------------------------------------------------
    # TABLE T1: GLOBAL SUMMARY (26 rows)
    # ------------------------------------------------------------------
    t1_rows = []
    for fid, meta in FEATURE_DEFS.items():
        u = u_res[fid]
        w = w_res[fid]
        ident = i_res[fid]
        dev = dev_res[fid]
        
        t1_rows.append({
            "Feature ID": fid,
            "Feature Name": meta["name"],
            "Network": meta["network"],
            "Type": meta["type"],
            "Dim": meta["dim"],
            # Universality U_1
            "U_1 (Mean±SD)": f"{u['ses-1']['U_global']:.4f} ± {u['ses-1']['SD_pairs']:.4f}",
            "U_1 Median [95% CI]": f"{u['ses-1']['bootstrap']['median']:.4f} [{u['ses-1']['bootstrap']['ci_95'][0]:.4f}, {u['ses-1']['bootstrap']['ci_95'][1]:.4f}]",
            "U_1 p-val": u['ses-1']['p_global'],
            # Universality U_2
            "U_2 (Mean±SD)": f"{u['ses-2']['U_global']:.4f} ± {u['ses-2']['SD_pairs']:.4f}",
            "U_2 Median [95% CI]": f"{u['ses-2']['bootstrap']['median']:.4f} [{u['ses-2']['bootstrap']['ci_95'][0]:.4f}, {u['ses-2']['bootstrap']['ci_95'][1]:.4f}]",
            "U_2 p-val": u['ses-2']['p_global'],
            # Delta U
            "Delta U": f"{dev['delta_U_global']:+.4f}",
            "Delta U [95% CI]": f"[{dev['ci_95'][0]:+.4f}, {dev['ci_95'][1]:+.4f}]",
            "Delta U p-val": dev['p_value'],
            # Within-session stability W_1 and W_2
            "W_1": f"{w['ses-1']['W_global']:.4f}",
            "W_1 [95% CI]": f"[{w['ses-1']['W_bootstrap']['ci_95'][0]:.4f}, {w['ses-1']['W_bootstrap']['ci_95'][1]:.4f}]",
            "W_2": f"{w['ses-2']['W_global']:.4f}",
            "W_2 [95% CI]": f"[{w['ses-2']['W_bootstrap']['ci_95'][0]:.4f}, {w['ses-2']['W_bootstrap']['ci_95'][1]:.4f}]",
            # Identifiability I_self, I_others, I_diff
            "I_self": f"{ident['I_self']:.4f}",
            "I_others": f"{ident['I_others']:.4f}",
            "I_diff (%)": f"{ident['I_diff']:+.2f}",
            "I_diff [95% CI]": f"[{ident['bootstrap']['idiff_ci_95'][0]:+.2f}, {ident['bootstrap']['idiff_ci_95'][1]:+.2f}]",
            "I_diff p-val": ident['p_idiff'],
            # 1-NN Accuracy
            "Acc Forward (%)": f"{ident['acc_forward']:.1f}",
            "Acc Backward (%)": f"{ident['acc_backward']:.1f}",
            "Acc Mean (%)": f"{ident['acc_mean']:.1f}",
            # Mantel test rho
            "Mantel rho (ses-1)": f"{dev['mantel_ses1']['rho']:.3f} (p={dev['mantel_ses1']['p']:.4f})",
            "Mantel rho (ses-2)": f"{dev['mantel_ses2']['rho']:.3f} (p={dev['mantel_ses2']['p']:.4f})"
        })
    df_t1 = pd.DataFrame(t1_rows)
    df_t1.to_csv(TABLES_DIR / "Table_T1_Global_Summary.csv", index=False)
    (TABLES_DIR / "Table_T1_Global_Summary.md").write_text(df_to_markdown(df_t1))
    print(f" -> Table T1 saved: {TABLES_DIR / 'Table_T1_Global_Summary.csv'} (.csv & .md)")
    
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
                
            t2_rows.append({
                "Feature ID": fid,
                "Feature Name": fname,
                "Element": label,
                "Region": reg,
                "Hemisphere": hemi,
                "Connection Type": wb_flag,
                "Euclidean Distance (mm)": e_dist,
                "U_1(e)": round(u["ses-1"]["U_elem"][e], 4),
                "U_2(e)": round(u["ses-2"]["U_elem"][e], 4),
                "W_1(e)": round(w["ses-1"]["W_elem"][e], 4),
                "W_2(e)": round(w["ses-2"]["W_elem"][e], 4),
                "I_diff(e) (%)": round(ident["I_diff_elem"][e], 2),
                "p_raw_idiff": round(ident["p_elem"][e], 4),
                "q_fdr_idiff": round(ident["q_elem"][e], 4),
                "Taxonomy Class": tax["classes"][e],
                "Undefined (0/0) Frac ses-1": round(u["ses-1"]["undef_frac"][e], 4),
                "Undefined (0/0) Frac ses-2": round(u["ses-2"]["undef_frac"][e], 4)
            })
    df_t2 = pd.DataFrame(t2_rows)
    df_t2.to_csv(TABLES_DIR / "Table_T2_Element_Results.csv", index=False)
    print(f" -> Table T2 saved: {TABLES_DIR / 'Table_T2_Element_Results.csv'} (3,230 element evaluations)")
    
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
                "Model Converged": elem_dict["converged"],
                "Skipped Binary Adjacency": elem_dict["skipped"]
            })
    df_t3 = pd.DataFrame(t3_rows)
    df_t3.to_csv(TABLES_DIR / "Table_T3_Variance_Partition.csv", index=False)
    print(f" -> Table T3 saved: {TABLES_DIR / 'Table_T3_Variance_Partition.csv'}")
    
    # ------------------------------------------------------------------
    # TABLE T4: ROBUSTNESS & SENSITIVITY SUMMARY (88 rows)
    # ------------------------------------------------------------------
    t4_csv = TABLES_DIR / "Table_T4_Robustness.csv"
    if t4_csv.exists():
        df_t4 = pd.read_csv(t4_csv)
        (TABLES_DIR / "Table_T4_Robustness.md").write_text(df_to_markdown(df_t4))
        print(f" -> Table T4 saved: {t4_csv} (.csv & .md)")
    else:
        print(" -> Table T4 missing; please run Phase 7 first.")

    # ------------------------------------------------------------------
    # TABLE T5: SUBJECT BOOKKEEPING (19 rows)
    # ------------------------------------------------------------------
    t5_rows = []
    for subj in SUBJECT_IDS:
        s1_file = DATASET_PATH / subj / "ses-1" / f"{subj}_ses-1_scans.tsv"
        s2_file = DATASET_PATH / subj / "ses-2" / f"{subj}_ses-2_scans.tsv"
        a1 = float(pd.read_csv(s1_file, sep="\t")["age_acq_time"].iloc[0])
        a2 = float(pd.read_csv(s2_file, sep="\t")["age_acq_time"].iloc[0])
        gap_mo = (a2 - a1) * 12.0
        
        ann1 = len(load_annotations(subj, "ses-1"))
        ann2 = len(load_annotations(subj, "ses-2"))
        
        twin_str = "Twin A" if subj == "sub-NORB00086" else ("Twin B" if subj == "sub-NORB00087" else "Singleton")
        note = "Brother of 00087 (male twin)" if subj == "sub-NORB00086" else ("Brother of 00086 (male twin)" if subj == "sub-NORB00087" else ("Short gap: 4 days" if subj == "sub-NORB00064" else ""))
        
        t5_rows.append({
            "Participant ID": subj,
            "Sex": "Male" if subj in ("sub-NORB00086", "sub-NORB00087") else ("Female" if subj in ("sub-NORB00064", "sub-NORB00069", "sub-NORB00072", "sub-NORB00078", "sub-NORB00081", "sub-NORB00088", "sub-NORB00091", "sub-NORB00092", "sub-NORB00096", "sub-NORB00101", "sub-NORB00102") else "Male"),
            "Age ses-1 (yr)": round(a1, 4),
            "Age ses-2 (yr)": round(a2, 4),
            "Gap (months)": round(gap_mo, 2),
            "Gap (days)": round((a2 - a1) * 365.25, 1),
            "Windows ses-1": ann1,
            "Windows ses-2": ann2,
            "Status": twin_str,
            "Notes": note
        })
    df_t5 = pd.DataFrame(t5_rows)
    df_t5.to_csv(TABLES_DIR / "Table_T5_Bookkeeping.csv", index=False)
    (TABLES_DIR / "Table_T5_Bookkeeping.md").write_text(df_to_markdown(df_t5))
    print(f" -> Table T5 saved: {TABLES_DIR / 'Table_T5_Bookkeeping.csv'} (.csv & .md)")
    
    # ------------------------------------------------------------------
    # TABLE T6: SIBLING / TWIN ANALYSIS (26 rows)
    # ------------------------------------------------------------------
    t6_rows = []
    for fid, meta in FEATURE_DEFS.items():
        tw = twin_res[fid]
        c = tw["cross_identification"]
        mis_str = "None"
        if c["misidentified_86_as_87"] and c["misidentified_87_as_86"]:
            mis_str = "Both"
        elif c["misidentified_86_as_87"]:
            mis_str = "86 -> 87"
        elif c["misidentified_87_as_86"]:
            mis_str = "87 -> 86"
            
        t6_rows.append({
            "Feature ID": fid,
            "Feature Name": meta["name"],
            "Twin Similarity ses-1": round(tw["ses1_similarity"], 4),
            "Twin Similarity ses-2": round(tw["ses2_similarity"], 4),
            "Percentile Rank ses-1 (/171)": round(tw["rank_among_171_ses1"], 1),
            "Percentile Rank ses-2 (/171)": round(tw["rank_among_171_ses2"], 1),
            "Matched Infant Rank ses-1": round(tw["rank_matched_infant_ses1"], 1),
            "Matched Infant Rank ses-2": round(tw["rank_matched_infant_ses2"], 1),
            "Matched Age Rank ses-1": round(tw["rank_matched_age_ses1"], 1) if not np.isnan(tw.get("rank_matched_age_ses1", np.nan)) else "N/A",
            "Matched Age Rank ses-2": round(tw["rank_matched_age_ses2"], 1) if not np.isnan(tw.get("rank_matched_age_ses2", np.nan)) else "N/A",
            "M[86, 86] Self": round(c["M_86_86"], 4),
            "M[87, 87] Self": round(c["M_87_87"], 4),
            "M[86, 87] Cross": round(c["M_86_87"], 4),
            "M[87, 86] Cross": round(c["M_87_86"], 4),
            "86 Cross-Rank in Row": c["rank_87_for_86"],
            "87 Cross-Rank in Row": c["rank_86_for_87"],
            "Misidentified?": mis_str,
            "Shared P95 Elements Count": tw["n_shared_p95_elements"],
            "Shared P95 Elements": ", ".join(tw["shared_elements"]) if tw["shared_elements"] else "None"
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
