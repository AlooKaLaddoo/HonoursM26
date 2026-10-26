"""
Phase 5: Universality-Individuality Taxonomy & Variance Partitioning (Step 5)
Computes:
  - 5a: Four-quadrant classification (Universal Core, Individual Marker, Universal & Individual, Variable/Redundant)
  - 5b: Three-way Variance Partitioning (Infant/Trait, Session/Development, Epoch/State)
        using Linear Mixed-Effects Models on rank-based inverse-normal transformed features.
  - Age-adjusted variance decomposition.
Outputs Table T2 elements and Table T3 variance partitions.
"""
import concurrent.futures
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
        N_EPOCHS,
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
        N_EPOCHS,
        NODE_LABELS,
        OUTPUT_TENSORS_DIR,
        SESSIONS,
        SUBJECT_IDS,
    )


def rank_inverse_normal(values: np.ndarray) -> np.ndarray:
    """Rank-based inverse-normal transformation (Step 5b item 1)."""
    ranks = stats.rankdata(values, method="average")
    n = len(values)
    prob = (ranks - 0.5) / n
    transformed = stats.norm.ppf(prob)
    return transformed


def load_exact_ages():
    """Load exact infant ages (in years and days) for ses-1 and ses-2."""
    ages = {}
    for subj in SUBJECT_IDS:
        ages[subj] = {}
        for ses in SESSIONS:
            scan_file = DATASET_PATH / subj / ses / f"{subj}_{ses}_scans.tsv"
            if scan_file.exists():
                df = pd.read_csv(scan_file, sep="\t")
                age_yr = float(df["age_acq_time"].iloc[0])
                ages[subj][ses] = age_yr * 365.25 # in days
            else:
                ages[subj][ses] = np.nan
    return ages


def fit_element_lmm(args):
    """
    Worker for fitting Model 1 (unadjusted) and Model 2 (age-adjusted) Linear Mixed-Effects Models
    on rank-based inverse-normal transformed element observations.
    """
    _e, raw_vals, base_records, label = args
    warnings.filterwarnings("ignore")
    
    # Check for zero variance or constant raw values
    if np.all(raw_vals == raw_vals[0]) or np.isnan(raw_vals).all():
        return {
            "element": label,
            "trait_prop": np.nan, "session_prop": np.nan, "epoch_prop": np.nan,
            "trait_prop_adj": np.nan, "session_prop_adj": np.nan, "epoch_prop_adj": np.nan,
            "converged": False, "skipped": False
        }
        
    y_trans = rank_inverse_normal(raw_vals)
    
    df_e = pd.DataFrame(base_records)
    df_e["y"] = y_trans
    
    # Model 1: y ~ 1
    try:
        m1 = smf.mixedlm("y ~ 1", df_e, groups=df_e["infant"], re_formula="1", vc_formula={"session": "0 + C(session)"})
        r1 = m1.fit(reml=True, maxiter=200)
        var_infant = float(r1.cov_re.iloc[0, 0])
        var_session = float(r1.vcomp[0]) if len(r1.vcomp) > 0 else 0.0
        var_epoch = float(r1.scale)
        total_var = var_infant + var_session + var_epoch
        p_infant = var_infant / total_var if total_var > 0 else 0.0
        p_session = var_session / total_var if total_var > 0 else 0.0
        p_epoch = var_epoch / total_var if total_var > 0 else 0.0
        conv1 = bool(r1.converged)
    except (np.linalg.LinAlgError, ValueError, IndexError, KeyError, ZeroDivisionError):
        p_infant, p_session, p_epoch, conv1 = np.nan, np.nan, np.nan, False
        
    # Model 2: Age-adjusted: y ~ age_c
    try:
        m2 = smf.mixedlm("y ~ age_c", df_e, groups=df_e["infant"], re_formula="1", vc_formula={"session": "0 + C(session)"})
        r2 = m2.fit(reml=True, maxiter=200)
        var_infant_a = float(r2.cov_re.iloc[0, 0])
        var_session_a = float(r2.vcomp[0]) if len(r2.vcomp) > 0 else 0.0
        var_epoch_a = float(r2.scale)
        total_var_a = var_infant_a + var_session_a + var_epoch_a
        p_infant_a = var_infant_a / total_var_a if total_var_a > 0 else 0.0
        p_session_a = var_session_a / total_var_a if total_var_a > 0 else 0.0
        p_epoch_a = var_epoch_a / total_var_a if total_var_a > 0 else 0.0
        conv2 = bool(r2.converged)
    except (np.linalg.LinAlgError, ValueError, IndexError, KeyError, ZeroDivisionError):
        p_infant_a, p_session_a, p_epoch_a, conv2 = np.nan, np.nan, np.nan, False
        
    return {
        "element": label,
        "trait_prop": p_infant,
        "session_prop": p_session,
        "epoch_prop": p_epoch,
        "trait_prop_adj": p_infant_a,
        "session_prop_adj": p_session_a,
        "epoch_prop_adj": p_epoch_a,
        "converged": bool(conv1 and conv2),
        "skipped": False
    }


def run_taxonomy_and_variance(max_workers: int = 8):
    print("=" * 80)
    print("PHASE 5: TAXONOMY (5a) & STATE-TRAIT VARIANCE DECOMPOSITION (5b)")
    print(f"Cohort: {len(SUBJECT_IDS)} infants | Observations per element: 456 (19x2x12)")
    print("=" * 80)
    
    # Load Phase 2 Universality results and Phase 4 Identifiability results
    u_file = OUTPUT_TENSORS_DIR / "universality_results.json"
    i_file = OUTPUT_TENSORS_DIR / "identifiability_results.json"
    
    if not u_file.exists() or not i_file.exists():
        raise FileNotFoundError("Prerequisite results missing. Please run Phase 2 and Phase 4 first.")
        
    with open(u_file) as f:
        u_results = json.load(f)
    with open(i_file) as f:
        i_results = json.load(f)
        
    ages_dict = load_exact_ages()
    all_ages = [ages_dict[s][ses] for s in SUBJECT_IDS for ses in SESSIONS]
    mean_sample_age = np.nanmean(all_ages)
    
    # Pre-build base observation records
    base_records = []
    for subj in SUBJECT_IDS:
        for ses in SESSIONS:
            age_val = ages_dict[subj][ses]
            age_c = age_val - mean_sample_age
            for ep_idx in range(N_EPOCHS):
                base_records.append({
                    "infant": subj,
                    "session": ses,
                    "epoch": ep_idx + 1,
                    "age_c": age_c
                })
                
    taxonomy_results = {}
    variance_results = {}
    
    for fid_idx, (fid, meta) in enumerate(FEATURE_DEFS.items(), 1):
        fname = meta["name"]
        dim = meta["dim"]
        ftype = meta["type"]
        is_binary_adj = (meta["name"].endswith("_adjacency"))
        labels = NODE_LABELS if ftype == "node" else EDGE_LABELS
        
        u_data = u_results[fid]
        i_data = i_results[fid]
        
        # -------------------------------------------------------------
        # 5a. FOUR-QUADRANT TAXONOMY
        # -------------------------------------------------------------
        q_u1 = np.array(u_data["ses-1"]["q_elem"])
        q_u2 = np.array(u_data["ses-2"]["q_elem"])
        q_idiff = np.array(i_data["q_elem"])
        
        sig_u = (q_u1 < 0.05) & (q_u2 < 0.05)
        sig_idiff = (q_idiff < 0.05)
        
        tax_classes = []
        for e in range(dim):
            if sig_u[e] and sig_idiff[e]:
                tax_classes.append("Universal and Individual")
            elif sig_u[e] and not sig_idiff[e]:
                tax_classes.append("Universal")
            elif not sig_u[e] and sig_idiff[e]:
                tax_classes.append("Individual")
            else:
                tax_classes.append("Neither")
                
        class_counts = {
            "Universal": tax_classes.count("Universal"),
            "Individual": tax_classes.count("Individual"),
            "Universal and Individual": tax_classes.count("Universal and Individual"),
            "Neither": tax_classes.count("Neither")
        }
        
        taxonomy_results[fid] = {
            "fid": fid, "name": fname, "dim": dim,
            "counts": class_counts,
            "classes": tax_classes
        }
        
        # -------------------------------------------------------------
        # 5b. VARIANCE DECOMPOSITION (LINEAR MIXED-EFFECTS MODELS)
        # -------------------------------------------------------------
        tensor_path = OUTPUT_TENSORS_DIR / f"X_{fid}.npy"
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Feature tensor not found: {tensor_path}. Run 01_extract_tensors.py first."
            )
        X = np.load(tensor_path) # (19, 2, 12, dim)
        
        if is_binary_adj:
            # Skip binary adjacency per Step 5b item 1 (0/1 values)
            var_elem_list = []
            for e in range(dim):
                var_elem_list.append({
                    "element": labels[e],
                    "trait_prop": np.nan, "session_prop": np.nan, "epoch_prop": np.nan,
                    "trait_prop_adj": np.nan, "session_prop_adj": np.nan, "epoch_prop_adj": np.nan,
                    "converged": False, "skipped": True
                })
        else:
            tasks = []
            for e in range(dim):
                raw_vals = X[:, :, :, e].flatten()
                tasks.append((e, raw_vals, base_records, labels[e]))
                
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                var_elem_list = list(executor.map(fit_element_lmm, tasks))
                
        variance_results[fid] = {
            "fid": fid, "name": fname,
            "elements": var_elem_list
        }
        
        # Compute summary averages across elements
        valid_traits = [e["trait_prop"] for e in var_elem_list if not np.isnan(e["trait_prop"])]
        valid_sessions = [e["session_prop"] for e in var_elem_list if not np.isnan(e["session_prop"])]
        valid_epochs = [e["epoch_prop"] for e in var_elem_list if not np.isnan(e["epoch_prop"])]
        
        if valid_traits:
            m_trait = np.mean(valid_traits) * 100.0
            m_ses = np.mean(valid_sessions) * 100.0
            m_ep = np.mean(valid_epochs) * 100.0
            var_str = f"Trait={m_trait:4.1f}%, Session={m_ses:4.1f}%, State={m_ep:4.1f}%"
        else:
            var_str = "Skipped (Binary Adjacency)"
            
        print(f"[{fid_idx:02d}/26] {fid} ({fname[:25]:<25}) | Taxonomy: Univ={class_counts['Universal']:<3} Indiv={class_counts['Individual']:<2} Both={class_counts['Universal and Individual']:<2} Neither={class_counts['Neither']:<3} | Var: {var_str}")
        
    # Save outputs
    tax_file = OUTPUT_TENSORS_DIR / "taxonomy_results.json"
    var_file = OUTPUT_TENSORS_DIR / "variance_results.json"
    
    with open(tax_file, "w") as f:
        json.dump(taxonomy_results, f, indent=2)
    with open(var_file, "w") as f:
        json.dump(variance_results, f, indent=2)
        
    print("=" * 80)
    print("Taxonomy and Variance results successfully computed and saved!")
    print(f"Taxonomy: {tax_file}")
    print(f"Variance: {var_file}")
    print("=" * 80)
    return taxonomy_results, variance_results


if __name__ == "__main__":
    run_taxonomy_and_variance()
