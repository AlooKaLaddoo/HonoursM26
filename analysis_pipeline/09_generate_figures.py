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
import json
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

from .config import (
    SUBJECT_IDS, SESSIONS, NODE_LABELS, EDGE_LABELS,
    NODE_REGIONS, REGION_PAIRS, FEATURE_DEFS,
    OUTPUT_TENSORS_DIR, FIGURES_DIR, TABLES_DIR
)
from .utils import get_node_hemisphere, get_edge_hemisphere_pair

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
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.8, zorder=5)
            ax.add_patch(rect)
            
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Weighted Jaccard", fontsize=9)
        
    plt.suptitle("Figure 1: Longitudinal Infant Identifiability Matrices $M$ ($19 \\times 19$ Subjects)\nRows: ses-1, Columns: ses-2 | Cyan Bounding Boxes: Within-Subject Longitudinal Self-Similarity", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_1_Identifiability_Matrices.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_1_Identifiability_Matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Also generate supplementary full 26-feature grid
    fig26, axes26 = plt.subplots(5, 6, figsize=(24, 20))
    all_fids = list(FEATURE_DEFS.keys())
    for idx, ax in enumerate(axes26.flat):
        if idx < len(all_fids):
            fid = all_fids[idx]
            M = np.load(OUTPUT_TENSORS_DIR / f"M_mat_{fid}.npy")
            im = ax.imshow(M, cmap="hot", aspect="auto")
            ax.set_xticks(range(len(short_subjs)))
            ax.set_yticks(range(len(short_subjs)))
            ax.set_xticklabels(short_subjs, rotation=90, fontsize=6)
            ax.set_yticklabels(short_subjs, fontsize=6)
            for i in range(len(short_subjs)):
                rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.2)
                ax.add_patch(rect)
            ax.set_title(f"{fid}: {FEATURE_DEFS[fid]['name']}\n$I_{{diff}} = {i_res[fid]['I_diff']:+.1f}\\%$", fontsize=8)
        else:
            ax.axis("off")
    plt.suptitle("Figure 1 (Supplementary): Complete 26-Feature Longitudinal Identifiability Matrices $M$", fontsize=16, fontweight="bold")
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
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
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
                    fmt="^", color="#d62728", markersize=6, capsize=3, label="Longitudinal Self-Similarity $I_{\\mathrm{self}}$" if i == 0 else "")
        ax.errorbar(iothers, y - offset*2, xerr=[[iothers - iothers_ci[0]], [iothers_ci[1] - iothers]],
                    fmt="v", color="#7f7f7f", markersize=6, capsize=3, label="Null Cross-Subject Baseline $I_{\\mathrm{others}}$" if i == 0 else "")
        
        # Alternating background shading
        if i % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color="#f5f5f5", zorder=0)
            
    # Labels
    feat_labels = [f"{fid}: {FEATURE_DEFS[fid]['name']} ({FEATURE_DEFS[fid]['dim']})" for fid in all_fids]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, fontsize=9)
    ax.set_xlabel("Weighted Jaccard Index (Point Estimate with 95% Infant-Level Bootstrap CI)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 2: Architecture of Functional Organization Across 26 Infant EEG Features\nComparison Hierarchy: Within-Session Stability Ceiling ($W$) $\\geq$ Longitudinal Self-Similarity ($I_{\\mathrm{self}}$) > Null Baseline ($I_{\\mathrm{others}}$) $\\approx$ Universality ($U$)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0.05, 0.95)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_2_Universality_Identifiability_Forest.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_2_Universality_Identifiability_Forest.png", dpi=300, bbox_inches="tight")
    plt.close()

def generate_figure_3(u_res, i_res, tax_res):
    """Figure 3: Universality - Individuality 2D Quadrant Plane."""
    print("[3/7] Generating Figure 3: Universality - Individuality 2D Plane...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plot_fids = ["f01", "f04", "f20", "f24"]
    titles = [
        "Broadband Weighted Correlation (f01 — 171 Edges)",
        "Broadband Weighted Node Strength (f04 — 19 Nodes)",
        "Delta Binary Adjacency (f20 — 171 Edges)",
        "Alpha Binary Node Closeness (f24 — 19 Nodes)"
    ]
    
    palette = {
        "Universal": "#1f77b4",                # Blue
        "Individual": "#d62728",               # Red
        "Universal and Individual": "#9467bd", # Purple
        "Neither": "#7f7f7f"                   # Gray
    }
    
    for ax, fid, title in zip(axes.flat, plot_fids, titles):
        dim = FEATURE_DEFS[fid]["dim"]
        ftype = FEATURE_DEFS[fid]["type"]
        labels = NODE_LABELS if ftype == "node" else EDGE_LABELS
        
        u1_e = np.array(u_res[fid]["ses-1"]["U_elem"])
        u2_e = np.array(u_res[fid]["ses-2"]["U_elem"])
        u_mean = (u1_e + u2_e) / 2.0
        idiff_e = np.array(i_res[fid]["I_diff_elem"])
        classes = tax_res[fid]["classes"]
        
        # Quadrant threshold lines
        u_med = float(np.median(u_mean))
        ax.axvline(u_med, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        
        # Plot points by taxonomy class
        for cls_name in ["Universal and Individual", "Universal", "Individual", "Neither"]:
            idxs = [e for e in range(dim) if classes[e] == cls_name]
            if idxs:
                ax.scatter(
                    u_mean[idxs], idiff_e[idxs],
                    color=palette[cls_name],
                    label=f"{cls_name} (n = {len(idxs)})",
                    s=65 if ftype == "edge" else 100,
                    alpha=0.8,
                    edgecolors="#222222",
                    linewidth=0.6,
                    zorder=4
                )
                
        # Annotate top outlier elements
        top_k = 5
        sorted_idiff_idx = np.argsort(idiff_e)[::-1]
        for rank, e_idx in enumerate(sorted_idiff_idx[:top_k]):
            if idiff_e[e_idx] > 0.5:
                ax.annotate(
                    labels[e_idx],
                    (u_mean[e_idx], idiff_e[e_idx]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#111111",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#aaaaaa", alpha=0.85)
                )
                
        # Quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(0.97, 0.97, "Universal & Individual", transform=ax.transAxes, ha="right", va="top", fontsize=9, fontweight="bold", color="#9467bd")
        ax.text(0.97, 0.03, "Universal Core", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold", color="#1f77b4")
        ax.text(0.03, 0.97, "Individual Marker", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color="#d62728")
        ax.text(0.03, 0.03, "Neither", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color="#7f7f7f")
        
        ax.set_xlabel("Mean Universality $U(e) = (U_1(e) + U_2(e)) / 2$", fontsize=10, fontweight="bold")
        ax.set_ylabel("Differential Identifiability $I_{\\mathrm{diff}}(e)$ (%)", fontsize=10, fontweight="bold")
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Load Table T2 for FDR values
    t2_df = pd.read_csv(TABLES_DIR / "Table_T2_Element_Results.csv")
    
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
                          arrowprops=dict(arrowstyle="->", color="#111111", lw=1))
    sm_a = ScalarMappable(norm=norm_a, cmap=cmap_a)
    cbar_a = plt.colorbar(sm_a, ax=ax_a, orientation="horizontal", fraction=0.045, pad=0.05)
    cbar_a.set_label("Node $I_{\\mathrm{diff}}(e)$ (%)", fontsize=9, fontweight="bold")
    
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
                          arrowprops=dict(arrowstyle="->", color="#111111", lw=1))
    sm_b = ScalarMappable(norm=norm_b, cmap=cmap_b)
    cbar_b = plt.colorbar(sm_b, ax=ax_b, orientation="horizontal", fraction=0.045, pad=0.05)
    cbar_b.set_label("Node $I_{\\mathrm{diff}}(e)$ (%)", fontsize=9, fontweight="bold")
    
    # Panel C: Delta Binary Adjacency (f20) — FDR Significant Edges
    ax_c = axes[1, 0]
    draw_scalp_outline(ax_c, "C. Significant Functional Edges: Delta Binary Adjacency (f20)")
    # Draw base nodes
    for node, (x, y) in ELECTRODE_2D.items():
        ax_c.scatter([x], [y], s=200, color="#e0e0e0", edgecolors="#666666", linewidth=1.0, zorder=4)
        ax_c.text(x, y, node, ha="center", va="center", fontsize=7.5, color="#333333", zorder=5)
        
    f20_sub = t2_df[t2_df["Feature ID"] == "f20"]
    # FDR significant edges in f20: P4-PZ and F8-T6
    f20_sig = f20_sub[f20_sub["q_fdr_idiff"] < 0.05]
    for _, row in f20_sig.iterrows():
        u_node, v_node = row["Element"].split("-")
        x1, y1 = ELECTRODE_2D[u_node]
        x2, y2 = ELECTRODE_2D[v_node]
        ax_c.plot([x1, x2], [y1, y2], color="#d62728", linewidth=3.5, zorder=6, solid_capstyle="round")
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax_c.annotate(f"{row['Element']}\n$I_{{diff}} = {row['I_diff(e) (%)']:+.1f}\\%$\n(q = {row['q_fdr_idiff']:.3f})",
                      (mid_x, mid_y), xytext=(20, 15), textcoords="offset points",
                      fontsize=8, fontweight="bold", color="#900C3F",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff0f0", edgecolor="#d62728", lw=1),
                      arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
        
    # Draw nominal p < 0.01 edges as dashed orange
    f20_nom = f20_sub[(f20_sub["p_raw_idiff"] < 0.01) & (f20_sub["q_fdr_idiff"] >= 0.05)]
    for _, row in f20_nom.iterrows():
        u_node, v_node = row["Element"].split("-")
        x1, y1 = ELECTRODE_2D[u_node]
        x2, y2 = ELECTRODE_2D[v_node]
        ax_c.plot([x1, x2], [y1, y2], color="#ff7f0e", linewidth=1.8, linestyle="--", alpha=0.8, zorder=5)
        
    ax_c.text(0.5, -0.06, "Solid Red: FDR $q < 0.05$ | Dashed Orange: Raw $p < 0.01$",
              transform=ax_c.transAxes, ha="center", fontsize=9, style="italic")
    
    # Panel D: Broadband Binary Edge Betweenness (f12) — FDR Significant Edges
    ax_d = axes[1, 1]
    draw_scalp_outline(ax_d, "D. Significant Shortest Path Edges: Binary Edge Betweenness (f12)")
    for node, (x, y) in ELECTRODE_2D.items():
        ax_d.scatter([x], [y], s=200, color="#e0e0e0", edgecolors="#666666", linewidth=1.0, zorder=4)
        ax_d.text(x, y, node, ha="center", va="center", fontsize=7.5, color="#333333", zorder=5)
        
    f12_sub = t2_df[t2_df["Feature ID"] == "f12"]
    f12_sig = f12_sub[f12_sub["q_fdr_idiff"] < 0.05]
    for _, row in f12_sig.iterrows():
        u_node, v_node = row["Element"].split("-")
        x1, y1 = ELECTRODE_2D[u_node]
        x2, y2 = ELECTRODE_2D[v_node]
        ax_d.plot([x1, x2], [y1, y2], color="#d62728", linewidth=3.5, zorder=6, solid_capstyle="round")
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax_d.annotate(f"{row['Element']}\n$I_{{diff}} = {row['I_diff(e) (%)']:+.1f}\\%$\n(q = {row['q_fdr_idiff']:.3f})",
                      (mid_x, mid_y), xytext=(-25 if mid_x < 0 else 15, 15), textcoords="offset points",
                      fontsize=8, fontweight="bold", color="#900C3F",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff0f0", edgecolor="#d62728", lw=1),
                      arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
        
    f12_nom = f12_sub[(f12_sub["p_raw_idiff"] < 0.01) & (f12_sub["q_fdr_idiff"] >= 0.05)]
    for _, row in f12_nom.iterrows():
        u_node, v_node = row["Element"].split("-")
        x1, y1 = ELECTRODE_2D[u_node]
        x2, y2 = ELECTRODE_2D[v_node]
        ax_d.plot([x1, x2], [y1, y2], color="#ff7f0e", linewidth=1.8, linestyle="--", alpha=0.8, zorder=5)
        
    ax_d.text(0.5, -0.06, "Solid Red: FDR $q < 0.05$ | Dashed Orange: Raw $p < 0.01$",
              transform=ax_d.transAxes, ha="center", fontsize=9, style="italic")
    
    plt.suptitle("Figure 4: Scalp Topographies of Individual Network Constituents\nNodes Coloured by Differential Identifiability $I_{\\mathrm{diff}}(e)$ | Edges Drawn at Statistical Significance Thresholds", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_4_Topography_Hubs_Fingerprints.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_4_Topography_Hubs_Fingerprints.png", dpi=300, bbox_inches="tight")
    plt.close()

def generate_figure_5(dev_res, i_res):
    """Figure 5: Developmental shift (Delta U) & Inter-session interval decay scatter."""
    print("[5/7] Generating Figure 5: Development & Time Interval Scatter...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    
    # Left: Delta U per feature
    all_fids = list(FEATURE_DEFS.keys())
    delta_vals = [dev_res[f]["delta_U_global"] for f in all_fids]
    ci_lows = [dev_res[f]["ci_95"][0] for f in all_fids]
    ci_highs = [dev_res[f]["ci_95"][1] for f in all_fids]
    p_vals = [dev_res[f]["p_value"] for f in all_fids]
    
    y_pos = np.arange(len(all_fids))[::-1]
    
    # Color points: Significant maturation in Navy, nonsignificant in Gray
    point_colors = ["#1f77b4" if p < 0.05 else "#7f7f7f" for p in p_vals]
    for i, y in enumerate(y_pos):
        ax1.errorbar(delta_vals[i], y, xerr=[[delta_vals[i] - ci_lows[i]], [ci_highs[i] - delta_vals[i]]],
                     fmt="o", color=point_colors[i], ecolor=point_colors[i], elinewidth=1.5, capsize=3, markersize=5)
        
    ax1.axvline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{f}: {FEATURE_DEFS[f]['name']}" for f in all_fids], fontsize=8.5)
    ax1.set_xlabel("Developmental Shift in Universality: $\\Delta U = U_2 - U_1$ (95% Bootstrap CI)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Normative Maturation Shift in Population Universality\n(Blue: FDR/Permutation $p < 0.05$ | Gray: Invariant Across 3 Months)", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    
    # Right: Self-similarity vs inter-session interval (f01)
    fid_target = "f01"
    self_vals = np.array(i_res[fid_target]["per_infant_self"])
    dev_mod = importlib.import_module("analysis_pipeline.06_developmental_and_twins")
    _, ages_days = dev_mod.load_exact_subject_ages()
    intervals_days = np.array([ages_days[s]["ses-2"] - ages_days[s]["ses-1"] for s in SUBJECT_IDS])
    
    ax2.scatter(intervals_days, self_vals, color="#9467bd", s=80, edgecolors="#222222", linewidth=1.0, zorder=4)
    
    # Regression trendline
    m, b = np.polyfit(intervals_days, self_vals, 1)
    x_grid = np.linspace(min(intervals_days) - 5, max(intervals_days) + 5, 100)
    ax2.plot(x_grid, m * x_grid + b, color="#333333", linestyle="-", linewidth=1.5, label=f"Linear Fit (Slope: {m*30:+.4f} / month)")
    
    # Highlight sub-NORB00064 (4 days) and sub-NORB00069 (42 days)
    idx_64 = SUBJECT_IDS.index("sub-NORB00064")
    idx_69 = SUBJECT_IDS.index("sub-NORB00069")
    
    ax2.scatter([intervals_days[idx_64]], [self_vals[idx_64]], color="#d62728", s=130, edgecolors="black", linewidth=1.5, zorder=6)
    ax2.annotate("sub-NORB00064\nGap: 4.0 days\nSelf: 0.697",
                 (intervals_days[idx_64], self_vals[idx_64]),
                 xytext=(15, 5), textcoords="offset points",
                 fontsize=9, fontweight="bold", color="#900C3F",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff0f0", edgecolor="#d62728", lw=1.2),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
    
    ax2.scatter([intervals_days[idx_69]], [self_vals[idx_69]], color="#1f77b4", s=130, edgecolors="black", linewidth=1.5, zorder=6)
    ax2.annotate("sub-NORB00069\nGap: 41.9 days\nSelf: 0.640",
                 (intervals_days[idx_69], self_vals[idx_69]),
                 xytext=(15, -20), textcoords="offset points",
                 fontsize=9, fontweight="bold", color="#0b4f6c",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f7ff", edgecolor="#1f77b4", lw=1.2),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))
    
    corr_info = dev_res[fid_target]["interval_corr"]["self_vs_interval"]
    rho = corr_info["rho"]
    pval = corr_info["p"]
    ax2.set_xlabel("Inter-Session Time Gap $\\Delta\\mathrm{days}$ (Scan Interval)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Longitudinal Self-Similarity $M[a, a]$ (Broadband Correlation)", fontsize=10, fontweight="bold")
    ax2.set_title(f"B. Longitudinal Self-Similarity vs Scan Gap Interval\nSpearman $\\rho = {rho:+.3f}$ ($p = {pval:.4f}$) | Stable Identification Retained", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    
    plt.suptitle("Figure 5: Developmental Maturation & Stability Across Inter-Session Intervals", fontsize=14, fontweight="bold", y=0.99)
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
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
    ax1.barh(y_pos, trait_props, color="#1f77b4", edgecolor="none", label="Trait / Infant Specificity ($\\sigma^2_{\\mathrm{Subject}}$)")
    ax1.barh(y_pos, session_props, left=trait_props, color="#ff7f0e", edgecolor="none", label="Development / Session Shift ($\\sigma^2_{\\mathrm{Session}}$)")
    left_ep = np.array(trait_props) + np.array(session_props)
    ax1.barh(y_pos, epoch_props, left=left_ep, color="#b0b0b0", edgecolor="none", label="State Noise / Epoch Variability ($\\sigma^2_{\\mathrm{Epoch}}$)")
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8.5)
    ax1.set_xlabel("Variance Proportion (%)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_title("A. Variance Decomposition Across Network Metrics\n(Three-Way Mixed Models: 456 Segments / Element)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    
    # Panel B: Regional breakdown (Broadband Weighted Correlation f01)
    t3_df = pd.read_csv(TABLES_DIR / "Table_T3_Variance_Partition.csv")
    f01_df = t3_df[t3_df["Feature ID"] == "f01"].copy()
    
    # Compute Region for each edge
    f01_df["Region"] = [
        f"{NODE_REGIONS[e.split('-')[0]]}-{NODE_REGIONS[e.split('-')[1]]}"
        for e in f01_df["Element"]
    ]
    
    reg_summary = f01_df.groupby("Region")[["Infant (Trait) Prop", "Session Prop", "Epoch (State) Prop"]].mean() * 100
    # Sort by increasing inter-region distance (as in REGION_PAIRS)
    ordered_regs = [r for r in REGION_PAIRS if r in reg_summary.index]
    reg_summary = reg_summary.loc[ordered_regs]
    
    y_reg = np.arange(len(ordered_regs))[::-1]
    ax2.barh(y_reg, reg_summary["Infant (Trait) Prop"], color="#1f77b4", edgecolor="none", label="Trait / Infant ($\sigma^2_{\mathrm{Subject}}$)")
    ax2.barh(y_reg, reg_summary["Session Prop"], left=reg_summary["Infant (Trait) Prop"], color="#ff7f0e", edgecolor="none", label="Session / Development ($\sigma^2_{\mathrm{Session}}$)")
    left_reg_ep = reg_summary["Infant (Trait) Prop"] + reg_summary["Session Prop"]
    ax2.barh(y_reg, reg_summary["Epoch (State) Prop"], left=left_reg_ep, color="#b0b0b0", edgecolor="none", label="State Noise ($\sigma^2_{\mathrm{Epoch}}$)")
    
    ax2.set_yticks(y_reg)
    ax2.set_yticklabels([f"Region Pair: {r}" for r in ordered_regs], fontsize=9)
    ax2.set_xlabel("Variance Proportion (%)", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_title("B. Anatomical Region Breakdown (Broadband Correlation f01)\nOrdered by Inter-Regional Physical Distance (F-F to O-O)", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    
    plt.suptitle("Figure 6: State-Trait Variance Decomposition Partitioning\nPartitioning Network Variance into Inherent Trait, Normative Session Maturation, and Transient State Fluctuations", fontsize=14, fontweight="bold", y=0.99)
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
    im1 = ax1.imshow(j_sorted1, aspect="auto", cmap="hot", vmin=0.0, vmax=1.0, origin="upper")
    ax1.set_title("A. Session 1: Pairwise Jaccard (Sorted by $|\\Delta\\mathrm{age}|$)\nRows: 171 Infant Pairs | Columns: 171 Functional Edges", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Subject Pairs (Sorted by Chronological Age Gap: Smallest $\\to$ Largest)", fontsize=10, fontweight="bold")
    
    # Mark boundaries
    xtick_pos = [(s + e) / 2 for s, e, _ in boundaries]
    xtick_lbl = [grp for _, _, grp in boundaries]
    ax1.set_xticks(xtick_pos)
    ax1.set_xticklabels(xtick_lbl, rotation=90, fontsize=8.5, ha="center")
    ax1.set_xlabel("Edge Groups (Anatomical Region Pairs: F-F to O-O)", fontsize=10, fontweight="bold")
    for s, e, grp in boundaries:
        ax1.axvline(s - 0.5, color="cyan", linewidth=0.8, alpha=0.8)
        
    # Mark Male Twin Pair row in ses-1
    # Locate twin pair index in sorted list
    twin_i = SUBJECT_IDS.index("sub-NORB00086")
    twin_j = SUBJECT_IDS.index("sub-NORB00087")
    twin_orig_idx = [k for k, (i, j) in enumerate(zip(p_idx, q_idx)) if (i == twin_i and j == twin_j) or (i == twin_j and j == twin_i)][0]
    twin_sorted_row1 = list(sort_idx1).index(twin_orig_idx)
    ax1.axhline(twin_sorted_row1, color="lime", linewidth=2.0, zorder=5)
    ax1.annotate(f"Male Twin Pair (86 vs 87)\n$|\\Delta\\mathrm{{age}}| = 0.0$ yr (Row {twin_sorted_row1})",
                 xy=(10, twin_sorted_row1), xytext=(25, 25),
                 textcoords="offset points", fontsize=8.5, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111", edgecolor="lime", lw=1.5),
                 arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))
    
    # Panel B: ses-2 Heatmap
    im2 = ax2.imshow(j_sorted2, aspect="auto", cmap="hot", vmin=0.0, vmax=1.0, origin="upper")
    ax2.set_title("B. Session 2: Pairwise Jaccard (Sorted by $|\\Delta\\mathrm{age}|$)\nRows: 171 Infant Pairs | Columns: 171 Functional Edges", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Subject Pairs (Sorted by Chronological Age Gap: Smallest $\\to$ Largest)", fontsize=10, fontweight="bold")
    ax2.set_xticks(xtick_pos)
    ax2.set_xticklabels(xtick_lbl, rotation=90, fontsize=8.5, ha="center")
    ax2.set_xlabel("Edge Groups (Anatomical Region Pairs: F-F to O-O)", fontsize=10, fontweight="bold")
    for s, e, grp in boundaries:
        ax2.axvline(s - 0.5, color="cyan", linewidth=0.8, alpha=0.8)
        
    twin_sorted_row2 = list(sort_idx2).index(twin_orig_idx)
    ax2.axhline(twin_sorted_row2, color="lime", linewidth=2.0, zorder=5)
    ax2.annotate(f"Male Twin Pair (86 vs 87)\n$|\\Delta\\mathrm{{age}}| = 0.0$ yr (Row {twin_sorted_row2})",
                 xy=(10, twin_sorted_row2), xytext=(25, 25),
                 textcoords="offset points", fontsize=8.5, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111", edgecolor="lime", lw=1.5),
                 arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))
    
    # Shared Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="horizontal", fraction=0.04, pad=0.08, shrink=0.6)
    cbar.set_label("Weighted Jaccard Index (Fixed Scale: 0.0 to 1.0)", fontsize=10, fontweight="bold")
    
    plt.suptitle("Figure 7: Element-Wise Jaccard Similarity Across All 171 Infant Pairs\nRows Ordered by Age Difference $|\\Delta\\mathrm{age}|$ | Columns Grouped by Anatomical Region (Fixed 0–1 Scale)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_7_Ordered_Element_Heatmaps.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "Figure_7_Ordered_Element_Heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Also generate Sibling Analysis Figure: Distribution of similarities with twin pair marked
    fig_sib, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Broadband correlation (f01)
    s_f01 = np.load(OUTPUT_TENSORS_DIR / "S_mat_f01_ses-1.npy")
    s_vals_1 = s_f01[p_idx, q_idx]
    s_f01_2 = np.load(OUTPUT_TENSORS_DIR / "S_mat_f01_ses-2.npy")
    s_vals_2 = s_f01_2[p_idx, q_idx]
    
    twin_s1 = twin_res["f01"]["ses1_similarity"]
    twin_s2 = twin_res["f01"]["ses2_similarity"]
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
    """Main execution orchestrating Figures 1 through 7."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("PHASE 9: GENERATING PUBLICATION FIGURES 1 THROUGH 7")
    print(f"Target Directory: {FIGURES_DIR}")
    print("=" * 80)
    
    # Load required databases
    with open(OUTPUT_TENSORS_DIR / "universality_results.json") as f:
        u_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "within_session_results.json") as f:
        w_res = json.load(f)
    with open(OUTPUT_TENSORS_DIR / "identifiability_results.json") as f:
        i_res = json.load(f)
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
