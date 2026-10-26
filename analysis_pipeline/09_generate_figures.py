"""
Phase 9: Figure Generation (Figures 1 through 7 + Supplementary Visualizations).
Generates publication-ready vector PDFs and high-resolution 300 DPI PNGs in figures/:
  - Figure 1: Identifiability Matrices M (Key features & All 26 features)
  - Figure 2: Forest Plot of Centrality Comparisons (U_1, U_2, W_1, I_self, I_others with 95% CIs)
  - Figure 3: Universality - Individuality 2D Plane & Quadrant Taxonomy
  - Figure 4: Scalp Topographical Maps & Connectome Fingerprints (Nodes coloured by I_diff, FDR edges)
  - Figure 5: Developmental Shift (Delta U) & Inter-Session Interval Decay Scatter
  - Figure 6: State-Trait Variance Decomposition Stacked Bars (Feature & Anatomical Region Breakdown)
  - Figure 7: Ordered Element Heatmaps (Sorted by |Delta age|, fixed 0-1 scale, Midline 'M') & Sibling Audit
"""
import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Support both direct script execution and package imports
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from analysis_pipeline.config import (
        EDGE_LABELS,
        FEATURE_DEFS,
        FIGURES_DIR,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        REGION_PAIRS,
        SUBJECT_IDS,
        TABLES_DIR,
    )
else:
    from .config import (
        EDGE_LABELS,
        FEATURE_DEFS,
        FIGURES_DIR,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        REGION_PAIRS,
        SUBJECT_IDS,
        TABLES_DIR,
    )

# Consistent publication typography
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "font.sans-serif": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Canonical 10-20 2D electrode coordinates mapped inside unit circle
ELECTRODE_2D = {
    'Fp1': (-0.28, 0.82), 'Fp2': (0.28, 0.82),
    'F7': (-0.75, 0.48),  'F3': (-0.38, 0.42), 'FZ': (0.0, 0.42), 'F4': (0.38, 0.42), 'F8': (0.75, 0.48),
    'T3': (-0.85, 0.0),   'C3': (-0.42, 0.0),  'CZ': (0.0, 0.0),  'C4': (0.42, 0.0),  'T4': (0.85, 0.0),
    'T5': (-0.75, -0.48), 'P3': (-0.38, -0.42), 'PZ': (0.0, -0.42), 'P4': (0.38, -0.42), 'T6': (0.75, -0.48),
    'O1': (-0.28, -0.82), 'O2': (0.28, -0.82)
}


def draw_scalp_outline(ax, title=""):
    """Draws a standardized 2D EEG head outline with nose and ears."""
    head = patches.Circle((0, 0), 1.0, edgecolor="#222222", facecolor="#fcfcfc", linewidth=2.0, zorder=1)
    ax.add_patch(head)
    
    # Nose
    ax.plot([-0.14, 0.0, 0.14], [0.98, 1.14, 0.98], color="#222222", linewidth=2.0, zorder=2)
    
    # Ears
    ear_l = patches.Ellipse((-1.03, 0.0), 0.10, 0.26, edgecolor="#222222", facecolor="#fcfcfc", linewidth=2.0, zorder=1)
    ear_r = patches.Ellipse((1.03, 0.0), 0.10, 0.26, edgecolor="#222222", facecolor="#fcfcfc", linewidth=2.0, zorder=1)
    ax.add_patch(ear_l)
    ax.add_patch(ear_r)
    
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)


def generate_figure_1(i_res, short_subjs):
    """Figure 1: Cross-session identifiability matrix M heatmaps with highlighted diagonals."""
    print("[1/7] Generating Figure 1: Identifiability Matrices M...")
    key_fids = ["f01", "f04", "f06", "f10", "f13", "f24"]
    titles = [
        "Broadband Weighted Correlation (f01)",
        "Broadband Weighted Node Strength (f04)",
        "Broadband Weighted Node Eigenvector (f06)",
        "Broadband Binary Adjacency (f10)",
        "Broadband Binary Node Degree (f13)",
        "Alpha Binary Node Closeness (f24)"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, fid, title in zip(axes.flat, key_fids, titles):
        M = np.load(OUTPUT_TENSORS_DIR / f"M_mat_{fid}.npy")
        im = ax.imshow(M, cmap="hot", aspect="auto")
        ax.set_xticks(range(len(short_subjs)))
        ax.set_yticks(range(len(short_subjs)))
        ax.set_xticklabels(short_subjs, rotation=90, fontsize=8)
        ax.set_yticklabels(short_subjs, fontsize=8)
        ax.set_xlabel("ses-2 Subject ID", fontsize=10)
        ax.set_ylabel("ses-1 Subject ID", fontsize=10)
        
        idiff = i_res[fid]["I_diff"]
        acc = i_res[fid]["acc_mean"]
        pval = i_res[fid]["p_idiff"]
        ax.set_title(f"{title}\n$I_{{diff}} = {idiff:+.2f}\\%$ (p = {pval:.4f}), Acc = {acc:.1f}%", fontsize=10, fontweight="bold")
        
        # Highlight self-pairs (diagonal) with bright cyan boxes
        for i in range(len(short_subjs)):
            rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.8, zorder=5)
            ax.add_patch(rect)
            
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Weighted Jaccard", fontsize=9)
        
    plt.suptitle(r"Figure 1: Longitudinal Infant Identifiability Matrices $M$ ($19 \times 19$ Subjects)" + "\n" + "Rows: ses-1, Columns: ses-2 | Cyan Bounding Boxes: Within-Subject Longitudinal Self-Similarity", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_1_Identifiability_Matrices.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_1_Identifiability_Matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Also generate supplementary full 26-feature grid
    _fig26, axes26 = plt.subplots(5, 6, figsize=(24, 20))
    all_fids = list(FEATURE_DEFS.keys())
    for idx, ax in enumerate(axes26.flat):
        if idx < len(all_fids):
            fid = all_fids[idx]
            M = np.load(OUTPUT_TENSORS_DIR / f"M_mat_{fid}.npy")
            ax.imshow(M, cmap="hot", aspect="auto")
            ax.set_xticks(range(len(short_subjs)))
            ax.set_yticks(range(len(short_subjs)))
            ax.set_xticklabels(short_subjs, rotation=90, fontsize=6)
            ax.set_yticklabels(short_subjs, fontsize=6)
            for i in range(len(short_subjs)):
                rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.2)
                ax.add_patch(rect)
            ax.set_title(f"{fid}: {FEATURE_DEFS[fid]['name']}\n$I_{{diff}} = {i_res[fid]['I_diff']:+.1f}\\%$", fontsize=8)
        else:
            ax.axis("off")
    plt.suptitle(r"Figure 1 (Supplementary): Complete 26-Feature Longitudinal Identifiability Matrices $M$", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_1_Supplementary_All_26_Matrices.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_1_Supplementary_All_26_Matrices.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure_2(u_res, w_res, i_res):
    """Figure 2: Forest plot of Universality (U_1, U_2), Within-session stability (W_1, W_2), and Longitudinal identifiability (I_self, I_others)."""
    print("[2/7] Generating Figure 2: Forest Plot of Centrality Comparisons...")
    all_fids = list(FEATURE_DEFS.keys())
    n_feats = len(all_fids)
    y_pos = np.arange(n_feats)[::-1] # reverse so f01 is at the top
    
    _fig, ax = plt.subplots(figsize=(15, 12))
    
    offset = 0.16
    for i, fid in enumerate(all_fids):
        y = y_pos[i]
        u1_med = u_res[fid]["ses-1"]["bootstrap"]["median"]
        u1_ci = u_res[fid]["ses-1"]["bootstrap"]["ci_95"]
        u2_med = u_res[fid]["ses-2"]["bootstrap"]["median"]
        u2_ci = u_res[fid]["ses-2"]["bootstrap"]["ci_95"]
        w1_med = w_res[fid]["ses-1"]["W_bootstrap"]["median"]
        w1_ci = w_res[fid]["ses-1"]["W_bootstrap"]["ci_95"]
        iself = i_res[fid]["I_self"]
        iself_ci = i_res[fid]["bootstrap"]["self_ci_95"]
        iothers = i_res[fid]["I_others"]
        iothers_ci = i_res[fid]["bootstrap"]["others_ci_95"]
        
        # Plot errorbars with distinct markers
        ax.errorbar(w1_med, y + offset*2, xerr=[[w1_med - w1_ci[0]], [w1_ci[1] - w1_med]],
                    fmt="s", color="#2ca02c", markersize=6, capsize=3, label="Within-Session Stability Ceiling $W_1$" if i == 0 else "")
        ax.errorbar(u1_med, y + offset*1, xerr=[[u1_med - u1_ci[0]], [u1_ci[1] - u1_med]],
                    fmt="o", color="#1f77b4", markersize=6, capsize=3, label="Population Universality $U_1$ (ses-1)" if i == 0 else "")
        ax.errorbar(u2_med, y, xerr=[[u2_med - u2_ci[0]], [u2_ci[1] - u2_med]],
                    fmt="D", color="#17becf", markersize=5, capsize=3, label="Population Universality $U_2$ (ses-2)" if i == 0 else "")
        ax.errorbar(iself, y - offset*1, xerr=[[iself - iself_ci[0]], [iself_ci[1] - iself]],
                    fmt="^", color="#d62728", markersize=6, capsize=3, label=r"Longitudinal Self-Similarity $I_{\mathrm{self}}$" if i == 0 else "")
        ax.errorbar(iothers, y - offset*2, xerr=[[iothers - iothers_ci[0]], [iothers_ci[1] - iothers]],
                    fmt="v", color="#7f7f7f", markersize=6, capsize=3, label=r"Null Cross-Subject Baseline $I_{\mathrm{others}}$" if i == 0 else "")
        
        # Alternating background shading
        if i % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color="#f5f5f5", zorder=0)
            
    # Labels
    feat_labels = [f"{fid}: {FEATURE_DEFS[fid]['name']} ({FEATURE_DEFS[fid]['dim']})" for fid in all_fids]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, fontsize=9)
    ax.set_xlabel("Weighted Jaccard Index (Point Estimate with 95% Infant-Level Bootstrap CI)", fontsize=11, fontweight="bold")
    ax.set_title(r"Figure 2: Architecture of Functional Organization Across 26 Infant EEG Features" + "\n" + r"Comparison Hierarchy: Within-Session Stability Ceiling ($W$) $\geq$ Longitudinal Self-Similarity ($I_{\mathrm{self}}$) > Null Baseline ($I_{\mathrm{others}}$) $\approx$ Universality ($U$)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0.05, 0.95)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_2_Universality_Identifiability_Forest.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_2_Universality_Identifiability_Forest.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure_3(u_res, i_res, tax_res):
    """Figure 3: Universality - Individuality 2D Quadrant Plane."""
    print("[3/7] Generating Figure 3: Universality-Individuality 2D Taxonomy Planes...")
    target_fids = ["f01", "f06", "f10", "f24"]
    titles = [
        "A. Broadband Weighted Correlation (f01)",
        "B. Broadband Weighted Node Eigenvector (f06)",
        "C. Broadband Binary Adjacency (f10)",
        "D. Alpha Binary Node Closeness (f24)"
    ]
    
    _fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    colors = {
        "Universal and Individual": "#9467bd",
        "Universal": "#1f77b4",
        "Individual": "#d62728",
        "Neither": "#7f7f7f"
    }
    
    markers = {
        "Universal and Individual": "D",
        "Universal": "o",
        "Individual": "^",
        "Neither": "."
    }
    
    for ax, fid, title in zip(axes.flat, target_fids, titles):
        dim = FEATURE_DEFS[fid]["dim"]
        ftype = FEATURE_DEFS[fid]["type"]
        labels = NODE_LABELS if ftype == "node" else EDGE_LABELS
        
        u1_e = np.array(u_res[fid]["ses-1"]["U_elem"])
        u2_e = np.array(u_res[fid]["ses-2"]["U_elem"])
        u_mean = (u1_e + u2_e) / 2.0
        idiff_e = np.array(i_res[fid]["I_diff_elem"])
        classes = tax_res[fid]["classes"]
        
        # Quadrant threshold lines
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2, alpha=0.8, zorder=2)
        u_thresh = float(np.nanmedian(u_mean))
        ax.axvline(u_thresh, color="#444444", linestyle="--", linewidth=1.2, alpha=0.8, zorder=2)
        
        # Scatter points by category
        for cat in ["Neither", "Universal", "Individual", "Universal and Individual"]:
            idx = [i for i, c in enumerate(classes) if c == cat]
            if idx:
                s = 80 if cat != "Neither" else 35
                alpha = 0.9 if cat != "Neither" else 0.4
                ax.scatter(u_mean[idx], idiff_e[idx], color=colors[cat], marker=markers[cat],
                           s=s, alpha=alpha, edgecolors="#111111" if cat != "Neither" else "none",
                           linewidth=0.8, label=f"{cat} (n = {len(idx)})", zorder=4)
                
        # Label top significant individual markers
        idiff_sort = np.argsort(idiff_e)[::-1]
        for rank in range(min(4, dim)):
            e_idx = idiff_sort[rank]
            if classes[e_idx] in ["Individual", "Universal and Individual"] and idiff_e[e_idx] > 0:
                ax.annotate(
                    labels[e_idx],
                    (u_mean[e_idx], idiff_e[e_idx]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#111111",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "#ffffff", "edgecolor": "#aaaaaa", "alpha": 0.85},
                )
                
        # Quadrant labels
        ax.text(0.97, 0.97, "Universal & Individual", transform=ax.transAxes, ha="right", va="top", fontsize=9, fontweight="bold", color="#9467bd")
        ax.text(0.97, 0.03, "Universal Core", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold", color="#1f77b4")
        ax.text(0.03, 0.97, "Individual Marker", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color="#d62728")
        ax.text(0.03, 0.03, "Neither", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color="#7f7f7f")
        
        ax.set_xlabel(r"Mean Universality $U(e) = (U_1(e) + U_2(e)) / 2$", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"Differential Identifiability $I_{\mathrm{diff}}(e)$ (%)", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        
    plt.suptitle("Figure 3: Four-Quadrant Functional Taxonomy of Network Constituents\nDecoupling Universal Population Backbone from Subject-Specific Fingerprinting Constituents", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_3_Universality_Individuality_Plane.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_3_Universality_Individuality_Plane.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure_4(i_res):
    """Figure 4: Scalp maps with nodes coloured by I_diff(e) and FDR-significant edge connections."""
    print("[4/7] Generating Figure 4: Scalp Topographical Maps & Hubs...")
    _fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Load Table T2 for FDR values
    t2_file = TABLES_DIR / "Table_T2_Element_Results.csv"
    if not t2_file.exists():
        raise FileNotFoundError(f"Prerequisite file missing: {t2_file}. Run 08_export_tables.py first.")
    t2_df = pd.read_csv(t2_file)
    
    # Panel A: Broadband Weighted Node Eigenvector (f06)
    ax_a = axes[0, 0]
    draw_scalp_outline(ax_a, "A. Node Centrality Identifiability: Node Eigenvector (f06)")
    f06_idiff = np.array(i_res["f06"]["I_diff_elem"])
    norm_a = Normalize(vmin=np.min(f06_idiff), vmax=np.max(f06_idiff))
    cmap_a = plt.cm.hot
    for idx, node in enumerate(NODE_LABELS):
        x, y = ELECTRODE_2D[node]
        val = f06_idiff[idx]
        color = cmap_a(norm_a(val))
        ax_a.scatter([x], [y], s=350, color=color, edgecolors="#111111", linewidth=1.5, zorder=5)
        text_color = "white" if norm_a(val) < 0.6 else "black"
        ax_a.text(x, y, node, ha="center", va="center", fontsize=8, fontweight="bold", color=text_color, zorder=6)
        # Label top hubs
        if node in ["T6", "F8", "F4"]:
            ax_a.annotate(f"{node}\n+{val:.2f}%", (x, y), xytext=(15, 10 if y>=0 else -20),
                          textcoords="offset points", fontsize=8, fontweight="bold",
                          arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1})
    sm_a = ScalarMappable(norm=norm_a, cmap=cmap_a)
    cbar_a = plt.colorbar(sm_a, ax=ax_a, orientation="horizontal", fraction=0.045, pad=0.05)
    cbar_a.set_label(r"Node $I_{\mathrm{diff}}(e)$ (%)", fontsize=9, fontweight="bold")
    
    # Panel B: Alpha Binary Node Closeness (f24)
    ax_b = axes[0, 1]
    draw_scalp_outline(ax_b, "B. Node Topology Identifiability: Alpha Binary Closeness (f24)")
    f24_idiff = np.array(i_res["f24"]["I_diff_elem"])
    norm_b = Normalize(vmin=np.min(f24_idiff), vmax=np.max(f24_idiff))
    cmap_b = plt.cm.hot
    for idx, node in enumerate(NODE_LABELS):
        x, y = ELECTRODE_2D[node]
        val = f24_idiff[idx]
        color = cmap_b(norm_b(val))
        ax_b.scatter([x], [y], s=350, color=color, edgecolors="#111111", linewidth=1.5, zorder=5)
        text_color = "white" if norm_b(val) < 0.6 else "black"
        ax_b.text(x, y, node, ha="center", va="center", fontsize=8, fontweight="bold", color=text_color, zorder=6)
        if node in ["P4", "T6", "F4"]:
            ax_b.annotate(f"{node}\n+{val:.2f}%", (x, y), xytext=(15, 10 if y>=0 else -20),
                          textcoords="offset points", fontsize=8, fontweight="bold",
                          arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1})
    sm_b = ScalarMappable(norm=norm_b, cmap=cmap_b)
    cbar_b = plt.colorbar(sm_b, ax=ax_b, orientation="horizontal", fraction=0.045, pad=0.05)
    cbar_b.set_label(r"Node $I_{\mathrm{diff}}(e)$ (%)", fontsize=9, fontweight="bold")
    
    # Panel C: Delta Binary Adjacency (f20) — FDR Significant Edges
    ax_c = axes[1, 0]
    draw_scalp_outline(ax_c, "C. Significant Functional Edges: Delta Binary Adjacency (f20)")
    for node, (x, y) in ELECTRODE_2D.items():
        ax_c.scatter([x], [y], s=200, color="#e0e0e0", edgecolors="#666666", linewidth=1.0, zorder=4)
        ax_c.text(x, y, node, ha="center", va="center", fontsize=7.5, color="#333333", zorder=5)
        
    f20_sig = t2_df[(t2_df["Feature ID"] == "f20") & (t2_df["q_fdr_idiff"] < 0.05)]
    for elem, idiff_val, q_val in zip(f20_sig["Element"], f20_sig["I_diff(e) (%)"], f20_sig["q_fdr_idiff"]):
        elem_str = str(elem)
        u_node, v_node = elem_str.split("-")
        x1, y1 = ELECTRODE_2D[u_node]
        x2, y2 = ELECTRODE_2D[v_node]
        ax_c.plot([x1, x2], [y1, y2], color="#d62728", linewidth=3.5, zorder=6, solid_capstyle="round")
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax_c.annotate(f"{elem_str}\n$I_{{diff}} = {float(idiff_val):+.1f}\\%$\n(q = {float(q_val):.3f})",
                      (mid_x, mid_y), xytext=(20, 15), textcoords="offset points",
                      fontsize=8, fontweight="bold", color="#900C3F",
                      bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fff0f0", "edgecolor": "#d62728", "lw": 1},
                      arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2})
        
    f20_nom = t2_df[(t2_df["Feature ID"] == "f20") & (t2_df["p_raw_idiff"] < 0.01) & (t2_df["q_fdr_idiff"] >= 0.05)]
    for elem in f20_nom["Element"]:
        u_node, v_node = str(elem).split("-")
        x1, y1 = ELECTRODE_2D[u_node]
        x2, y2 = ELECTRODE_2D[v_node]
        ax_c.plot([x1, x2], [y1, y2], color="#ff7f0e", linewidth=1.8, linestyle="--", alpha=0.8, zorder=5)
        
    ax_c.text(0.5, -0.06, r"Solid Red: FDR $q < 0.05$ | Dashed Orange: Raw $p < 0.01$",
              transform=ax_c.transAxes, ha="center", fontsize=9, style="italic")
    
    # Panel D: Broadband Binary Edge Betweenness (f12) — FDR Significant Edges
    ax_d = axes[1, 1]
    draw_scalp_outline(ax_d, "D. Significant Shortest Path Edges: Binary Edge Betweenness (f12)")
    for node, (x, y) in ELECTRODE_2D.items():
        ax_d.scatter([x], [y], s=200, color="#e0e0e0", edgecolors="#666666", linewidth=1.0, zorder=4)
        ax_d.text(x, y, node, ha="center", va="center", fontsize=7.5, color="#333333", zorder=5)
        
    f12_sig = t2_df[(t2_df["Feature ID"] == "f12") & (t2_df["q_fdr_idiff"] < 0.05)]
    for elem, idiff_val, q_val in zip(f12_sig["Element"], f12_sig["I_diff(e) (%)"], f12_sig["q_fdr_idiff"]):
        elem_str = str(elem)
        u_node, v_node = elem_str.split("-")
        x1, y1 = ELECTRODE_2D[u_node]
        x2, y2 = ELECTRODE_2D[v_node]
        ax_d.plot([x1, x2], [y1, y2], color="#d62728", linewidth=3.5, zorder=6, solid_capstyle="round")
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax_d.annotate(f"{elem_str}\n$I_{{diff}} = {float(idiff_val):+.1f}\\%$\n(q = {float(q_val):.3f})",
                      (mid_x, mid_y), xytext=(-25 if mid_x < 0 else 15, 15), textcoords="offset points",
                      fontsize=8, fontweight="bold", color="#900C3F",
                      bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fff0f0", "edgecolor": "#d62728", "lw": 1},
                      arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2})
        
    f12_nom = t2_df[(t2_df["Feature ID"] == "f12") & (t2_df["p_raw_idiff"] < 0.01) & (t2_df["q_fdr_idiff"] >= 0.05)]
    for elem in f12_nom["Element"]:
        u_node, v_node = str(elem).split("-")
        x1, y1 = ELECTRODE_2D[u_node]
        x2, y2 = ELECTRODE_2D[v_node]
        ax_d.plot([x1, x2], [y1, y2], color="#ff7f0e", linewidth=1.8, linestyle="--", alpha=0.8, zorder=5)
        
    ax_d.text(0.5, -0.06, r"Solid Red: FDR $q < 0.05$ | Dashed Orange: Raw $p < 0.01$",
              transform=ax_d.transAxes, ha="center", fontsize=9, style="italic")
    
    plt.suptitle("Figure 4: Anatomical Topography of Infant Neural Fingerprints\nRight Centroparietal Hubs & Long-Range Fronto-Temporal Edge Specificity", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_4_Topography_Hubs_Fingerprints.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_4_Topography_Hubs_Fingerprints.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure_5(dev_res, i_res):
    """Figure 5: Developmental trajectory and scan interval stability."""
    print("[5/7] Generating Figure 5: Developmental Shifts & Interval Decay...")
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    all_fids = list(FEATURE_DEFS.keys())
    deltas = [dev_res[f]["delta_U_global"] for f in all_fids]
    delta_cis = [dev_res[f]["ci_95"] for f in all_fids]
    y_pos = np.arange(len(all_fids))[::-1]
    
    for i, fid in enumerate(all_fids):
        d = deltas[i]
        ci = delta_cis[i]
        color = "#2ca02c" if d > 0 else "#d62728"
        ax1.errorbar(d, y_pos[i], xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", color=color, markersize=5, capsize=3)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{f}: {FEATURE_DEFS[f]['name']}" for f in all_fids], fontsize=8.5)
    ax1.set_xlabel(r"Developmental Shift $\Delta U = U_2 - U_1$", fontsize=10, fontweight="bold")
    ax1.set_title(r"A. Population Convergence vs Differentiation ($\Delta U$ with 95% Bootstrap CI)" + "\nGreen: Convergence ($U_2 > U_1$) | Red: Differentiation ($U_2 < U_1$)", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    
    # Panel B: Interval decay for broadband correlation (f01)
    dev_mod = importlib.import_module("analysis_pipeline.06_developmental_and_twins")
    _, ages_days = dev_mod.load_exact_subject_ages()
    intervals_days = np.array([ages_days[s]["ses-2"] - ages_days[s]["ses-1"] for s in SUBJECT_IDS])
    
    fid_target = "f01"
    M = np.load(OUTPUT_TENSORS_DIR / f"M_mat_{fid_target}.npy")
    self_vals = np.diag(M)
    
    # Scatter all 19 infants
    ax2.scatter(intervals_days, self_vals, color="#333333", s=70, alpha=0.8, zorder=4, label="Infants (N = 19)")
    
    # Regression line
    slope, intercept = np.polyfit(intervals_days, self_vals, 1)
    x_grid = np.linspace(min(intervals_days) - 2, max(intervals_days) + 2, 100)
    ax2.plot(x_grid, intercept + slope * x_grid, color="#1f77b4", linewidth=2.0, linestyle="-", label="Linear Trend Line")
    
    # Highlight sub-NORB00064 (4 days) and sub-NORB00069 (42 days)
    idx_64 = SUBJECT_IDS.index("sub-NORB00064")
    idx_69 = SUBJECT_IDS.index("sub-NORB00069")
    
    ax2.scatter([intervals_days[idx_64]], [self_vals[idx_64]], color="#d62728", s=130, edgecolors="black", linewidth=1.5, zorder=6)
    ax2.annotate("sub-NORB00064\nGap: 4.0 days\nSelf: 0.697",
                 (intervals_days[idx_64], self_vals[idx_64]),
                 xytext=(15, 5), textcoords="offset points",
                 fontsize=9, fontweight="bold", color="#900C3F",
                 bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff0f0", "edgecolor": "#d62728", "lw": 1.2},
                 arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2})
    
    ax2.scatter([intervals_days[idx_69]], [self_vals[idx_69]], color="#1f77b4", s=130, edgecolors="black", linewidth=1.5, zorder=6)
    ax2.annotate("sub-NORB00069\nGap: 41.9 days\nSelf: 0.640",
                 (intervals_days[idx_69], self_vals[idx_69]),
                 xytext=(15, -20), textcoords="offset points",
                 fontsize=9, fontweight="bold", color="#0b4f6c",
                 bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f0f7ff", "edgecolor": "#1f77b4", "lw": 1.2},
                 arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.2})
    
    corr_info = dev_res[fid_target]["interval_corr"]["self_vs_interval"]
    rho = corr_info["rho"]
    pval = corr_info["p"]
    ax2.set_xlabel(r"Inter-Session Time Gap $\Delta\mathrm{days}$ (Scan Interval)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Longitudinal Self-Similarity $M[a, a]$ (Broadband Correlation)", fontsize=10, fontweight="bold")
    ax2.set_title(r"B. Longitudinal Self-Similarity vs Scan Gap Interval" + "\n" + rf"Spearman $\rho = {rho:+.3f}$ ($p = {pval:.4f}$) | Stable Identification Retained", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    
    plt.suptitle("Figure 5: Developmental Stability and Longitudinal Maintenance Across Time", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_5_Developmental_Interval_Scatter.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_5_Developmental_Interval_Scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure_6(var_res):
    """Figure 6: State-Trait variance decomposition stacked bar charts (per feature and by region)."""
    print("[6/7] Generating Figure 6: Variance Partition Stacked Bars...")
    all_fids = list(FEATURE_DEFS.keys())
    # Exclude pure binary adjacency features from LMM rank models
    valid_fids = [f for f in all_fids if not FEATURE_DEFS[f]["name"].endswith("_adjacency")]
    
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Panel A: Feature-level stacked bars
    trait_props = []
    session_props = []
    epoch_props = []
    names = []
    
    for fid in valid_fids:
        elems = var_res[fid]["elements"]
        t = [e["trait_prop"] for e in elems if not np.isnan(e["trait_prop"])]
        s = [e["session_prop"] for e in elems if not np.isnan(e["session_prop"])]
        ep = [e["epoch_prop"] for e in elems if not np.isnan(e["epoch_prop"])]
        if t:
            trait_props.append(np.mean(t) * 100)
            session_props.append(np.mean(s) * 100)
            epoch_props.append(np.mean(ep) * 100)
            names.append(f"{fid}: {FEATURE_DEFS[fid]['name']}")
            
    y_pos = np.arange(len(names))[::-1]
    ax1.barh(y_pos, trait_props, color="#1f77b4", edgecolor="none", label=r"Trait / Infant Specificity ($\sigma^2_{\mathrm{Subject}}$)")
    ax1.barh(y_pos, session_props, left=trait_props, color="#ff7f0e", edgecolor="none", label=r"Development / Session Shift ($\sigma^2_{\mathrm{Session}}$)")
    left_ep = np.array(trait_props) + np.array(session_props)
    ax1.barh(y_pos, epoch_props, left=left_ep, color="#b0b0b0", edgecolor="none", label=r"State Noise / Epoch Variability ($\sigma^2_{\mathrm{Epoch}}$)")
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8.5)
    ax1.set_xlabel("Variance Proportion (%)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_title("A. Variance Decomposition Across Network Metrics\n(Three-Way Mixed Models: 456 Segments / Element)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    
    # Panel B: Regional breakdown (Broadband Weighted Correlation f01)
    t3_file = TABLES_DIR / "Table_T3_Variance_Partition.csv"
    if not t3_file.exists():
        raise FileNotFoundError(f"Prerequisite file missing: {t3_file}. Run 08_export_tables.py first.")
    t3_df = pd.read_csv(t3_file)
    f01_df = t3_df[t3_df["Feature ID"] == "f01"].copy()
    
    # Regional breakdown using native Python lists and float conversion
    elem_list = [str(e) for e in f01_df["Element"]]
    trait_vals = [float(v) * 100.0 for v in f01_df["Infant (Trait) Prop"]]
    session_vals = [float(v) * 100.0 for v in f01_df["Session Prop"]]
    epoch_vals = [float(v) * 100.0 for v in f01_df["Epoch (State) Prop"]]
    
    reg_trait: dict[str, list[float]] = {r: [] for r in REGION_PAIRS}
    reg_session: dict[str, list[float]] = {r: [] for r in REGION_PAIRS}
    reg_epoch: dict[str, list[float]] = {r: [] for r in REGION_PAIRS}
    
    for elem, t, s, ep in zip(elem_list, trait_vals, session_vals, epoch_vals):
        u_n, v_n = elem.split("-")
        r_pair = f"{NODE_REGIONS[u_n]}-{NODE_REGIONS[v_n]}"
        if r_pair in reg_trait:
            reg_trait[r_pair].append(t)
            reg_session[r_pair].append(s)
            reg_epoch[r_pair].append(ep)
            
    ordered_regs = [r for r in REGION_PAIRS if len(reg_trait[r]) > 0]
    mean_trait = [float(np.mean(reg_trait[r])) for r in ordered_regs]
    mean_session = [float(np.mean(reg_session[r])) for r in ordered_regs]
    mean_epoch = [float(np.mean(reg_epoch[r])) for r in ordered_regs]
    
    y_reg = np.arange(len(ordered_regs))[::-1]
    ax2.barh(y_reg, mean_trait, color="#1f77b4", edgecolor="none", label=r"Trait / Infant ($\sigma^2_{\mathrm{Subject}}$)")
    ax2.barh(y_reg, mean_session, left=mean_trait, color="#ff7f0e", edgecolor="none", label=r"Session / Development ($\sigma^2_{\mathrm{Session}}$)")
    left_reg_ep = np.array(mean_trait) + np.array(mean_session)
    ax2.barh(y_reg, mean_epoch, left=left_reg_ep, color="#b0b0b0", edgecolor="none", label=r"State Noise ($\sigma^2_{\mathrm{Epoch}}$)")
    
    ax2.set_yticks(y_reg)
    ax2.set_yticklabels([f"Region Pair: {r}" for r in ordered_regs], fontsize=9)
    ax2.set_xlabel("Variance Proportion (%)", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_title("B. Anatomical Region Breakdown (Broadband Correlation f01)\nOrdered by Inter-Regional Physical Distance (F-F to O-O)", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    
    plt.suptitle("Figure 6: Three-Way Variance Partitioning of Functional Brain Architecture\nTrait (Infant) vs Developmental (Session) vs State/Noise (Epoch) Variance Components", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_6_Variance_Partition_Stacked.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_6_Variance_Partition_Stacked.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure_7(twin_res):
    """Figure 7: Ordered element heatmaps sorted by |Delta age| and Twin Sibling Analysis."""
    print("[7/7] Generating Figure 7: Ordered Element Heatmaps & Twin Sibling Audit...")
    dev_mod = importlib.import_module("analysis_pipeline.06_developmental_and_twins")
    ages_yr, _ = dev_mod.load_exact_subject_ages()
    
    n_subs = len(SUBJECT_IDS)
    p_idx, q_idx = np.triu_indices(n_subs, k=1)
    
    # Load J_pair for broadband weighted correlation (f01)
    j_ses1 = np.load(OUTPUT_TENSORS_DIR / "J_pair_f01_ses-1.npy") # (171, 171)
    j_ses2 = np.load(OUTPUT_TENSORS_DIR / "J_pair_f01_ses-2.npy") # (171, 171)
    
    # Sort pairs by |Delta age| in ses-1 and ses-2
    diffs_ses1 = np.array([abs(ages_yr[SUBJECT_IDS[i]]["ses-1"] - ages_yr[SUBJECT_IDS[j]]["ses-1"]) for i, j in zip(p_idx, q_idx)])
    diffs_ses2 = np.array([abs(ages_yr[SUBJECT_IDS[i]]["ses-2"] - ages_yr[SUBJECT_IDS[j]]["ses-2"]) for i, j in zip(p_idx, q_idx)])
    
    sort_idx1 = np.argsort(diffs_ses1)
    sort_idx2 = np.argsort(diffs_ses2)
    
    j_sorted1 = j_ses1[sort_idx1, :]
    j_sorted2 = j_ses2[sort_idx2, :]
    
    # Region group boundaries for columns
    boundaries = []
    prev_grp = None
    start = 0
    for idx, edge in enumerate(EDGE_LABELS):
        u, v = edge.split("-")
        r1, r2 = NODE_REGIONS[u], NODE_REGIONS[v]
        grp = f"{r1}-{r2}"
        if prev_grp is None:
            prev_grp = grp
        elif grp != prev_grp:
            boundaries.append((start, idx - 1, prev_grp))
            start = idx
            prev_grp = grp
    boundaries.append((start, len(EDGE_LABELS) - 1, prev_grp))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Panel A: ses-1 Heatmap
    ax1.imshow(j_sorted1, aspect="auto", cmap="hot", vmin=0.0, vmax=1.0, origin="upper")
    ax1.set_title(r"A. Session 1: Pairwise Jaccard (Sorted by $|\Delta\mathrm{age}|$)" + "\n" + "Rows: 171 Infant Pairs | Columns: 171 Functional Edges", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Subject Pairs (Sorted by Chronological Age Gap: Smallest $\to$ Largest)", fontsize=10, fontweight="bold")
    
    # Mark boundaries
    xtick_pos = [(s + e) / 2 for s, e, _ in boundaries]
    xtick_lbl = [grp for _, _, grp in boundaries]
    ax1.set_xticks(xtick_pos)
    ax1.set_xticklabels(xtick_lbl, rotation=90, fontsize=8.5, ha="center")
    ax1.set_xlabel("Edge Groups (Anatomical Region Pairs: F-F to O-O)", fontsize=10, fontweight="bold")
    for s, _e, _grp in boundaries:
        ax1.axvline(s - 0.5, color="cyan", linewidth=0.8, alpha=0.8)
        
    # Mark Male Twin Pair row in ses-1
    twin_i = SUBJECT_IDS.index("sub-NORB00086")
    twin_j = SUBJECT_IDS.index("sub-NORB00087")
    twin_orig_idx = next(k for k, (i, j) in enumerate(zip(p_idx, q_idx)) if (i == twin_i and j == twin_j) or (i == twin_j and j == twin_i))
    twin_sorted_row1 = list(sort_idx1).index(twin_orig_idx)
    ax1.axhline(twin_sorted_row1, color="lime", linewidth=2.0, zorder=5)
    ax1.annotate(f"Male Twin Pair (86 vs 87)\n$|\\Delta\\mathrm{{age}}| = 0.0$ yr (Row {twin_sorted_row1})",
                 xy=(10, twin_sorted_row1), xytext=(25, 25),
                 textcoords="offset points", fontsize=8.5, fontweight="bold", color="white",
                 bbox={"boxstyle": "round,pad=0.25", "facecolor": "#111111", "edgecolor": "lime", "lw": 1.5},
                 arrowprops={"arrowstyle": "->", "color": "lime", "lw": 1.5})
    
    # Panel B: ses-2 Heatmap
    im2 = ax2.imshow(j_sorted2, aspect="auto", cmap="hot", vmin=0.0, vmax=1.0, origin="upper")
    ax2.set_title(r"B. Session 2: Pairwise Jaccard (Sorted by $|\Delta\mathrm{age}|$)" + "\n" + "Rows: 171 Infant Pairs | Columns: 171 Functional Edges", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Subject Pairs (Sorted by Chronological Age Gap: Smallest $\to$ Largest)", fontsize=10, fontweight="bold")
    ax2.set_xticks(xtick_pos)
    ax2.set_xticklabels(xtick_lbl, rotation=90, fontsize=8.5, ha="center")
    ax2.set_xlabel("Edge Groups (Anatomical Region Pairs: F-F to O-O)", fontsize=10, fontweight="bold")
    for s, _e, _grp in boundaries:
        ax2.axvline(s - 0.5, color="cyan", linewidth=0.8, alpha=0.8)
        
    twin_sorted_row2 = list(sort_idx2).index(twin_orig_idx)
    ax2.axhline(twin_sorted_row2, color="lime", linewidth=2.0, zorder=5)
    ax2.annotate(f"Male Twin Pair (86 vs 87)\n$|\\Delta\\mathrm{{age}}| = 0.0$ yr (Row {twin_sorted_row2})",
                 xy=(10, twin_sorted_row2), xytext=(25, 25),
                 textcoords="offset points", fontsize=8.5, fontweight="bold", color="white",
                 bbox={"boxstyle": "round,pad=0.25", "facecolor": "#111111", "edgecolor": "lime", "lw": 1.5},
                 arrowprops={"arrowstyle": "->", "color": "lime", "lw": 1.5})
    
    # Adjust layout before adding cross-axes colorbar to avoid tight_layout warnings
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    # Shared Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="horizontal", fraction=0.04, pad=0.08, shrink=0.6)
    cbar.set_label("Weighted Jaccard Index (Fixed Scale: 0.0 to 1.0)", fontsize=10, fontweight="bold")
    
    plt.suptitle(r"Figure 7: Element-Wise Jaccard Similarity Across All 171 Infant Pairs" + "\n" + r"Rows Ordered by Age Difference $|\Delta\mathrm{age}|$ | Columns Grouped by Anatomical Region (Fixed 0–1 Scale)", fontsize=14, fontweight="bold", y=0.99)
    plt.savefig(FIGURES_DIR / "Figure_7_Ordered_Element_Heatmaps.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_7_Ordered_Element_Heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Also generate Sibling Analysis Figure: Distribution of similarities with twin pair marked
    _fig_sib, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Broadband correlation (f01)
    s_f01 = np.load(OUTPUT_TENSORS_DIR / "S_mat_f01_ses-1.npy")
    s_vals_1 = s_f01[p_idx, q_idx]
    
    twin_s1 = twin_res["f01"]["ses1_similarity"]
    twin_pct1 = twin_res["f01"]["rank_among_171_ses1"]
    
    sns.histplot(s_vals_1, kde=True, ax=ax_s1, color="#1f77b4", stat="density", label="All 171 Infant Pairs")
    ax_s1.axvline(twin_s1, color="red", linewidth=2.5, linestyle="--", label=f"Male Twin Pair: {twin_s1:.3f} ({twin_pct1:.1f}th %ile)")
    ax_s1.set_xlabel("Recording Similarity $S$ (Broadband Correlation)", fontsize=10, fontweight="bold")
    ax_s1.set_title("A. Sibling Pair vs Cohort Distribution (ses-1)", fontsize=11, fontweight="bold")
    ax_s1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    
    # Alpha Binary Closeness (f24)
    # Load alpha closeness S matrix
    s_f24 = np.load(OUTPUT_TENSORS_DIR / "S_mat_f24_ses-1.npy")
    s_vals_f24_1 = s_f24[p_idx, q_idx]
    twin_f24_s1 = twin_res["f24"]["ses1_similarity"]
    twin_f24_pct1 = twin_res["f24"]["rank_among_171_ses1"]
    
    sns.histplot(s_vals_f24_1, kde=True, ax=ax_s2, color="#9467bd", stat="density", label="All 171 Infant Pairs")
    ax_s2.axvline(twin_f24_s1, color="red", linewidth=2.5, linestyle="--", label=f"Male Twin Pair: {twin_f24_s1:.3f} ({twin_f24_pct1:.1f}th %ile)")
    ax_s2.set_xlabel("Recording Similarity $S$ (Alpha Binary Closeness)", fontsize=10, fontweight="bold")
    ax_s2.set_title("B. Sibling Pair High Similarity (Alpha Closeness — 97.7th %ile)", fontsize=11, fontweight="bold")
    ax_s2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    
    plt.suptitle("Figure 7 (Supplementary): Distribution of Pair Similarities with Twin Pair Marked", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_7_Supplementary_Twin_Distributions.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_7_Supplementary_Twin_Distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("PHASE 9: COMPILING & GENERATING PUBLICATION FIGURES (1 THROUGH 7)")
    print("=" * 80)
    
    # Load prerequisite result databases
    with open(OUTPUT_TENSORS_DIR / "identifiability_results.json") as f:
        i_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "universality_results.json") as f:
        u_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "within_session_results.json") as f:
        w_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "taxonomy_results.json") as f:
        tax_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "developmental_results.json") as f:
        dev_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "variance_results.json") as f:
        var_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "twin_results.json") as f:
        twin_res = json.load(f)
        
    short_subjs = [s.replace("sub-NORB00", "") for s in SUBJECT_IDS]
    
    # Generate all figures
    generate_figure_1(i_res, short_subjs)
    generate_figure_2(u_res, w_res, i_res)
    generate_figure_3(u_res, i_res, tax_res)
    generate_figure_4(i_res)
    generate_figure_5(dev_res, i_res)
    generate_figure_6(var_res)
    generate_figure_7(twin_res)
    
    print("=" * 80)
    print("ALL PUBLICATION FIGURES (1 THROUGH 7) SUCCESSFULLY GENERATED!")
    print(f"Output files stored in: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_figures()
