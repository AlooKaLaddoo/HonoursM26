"""
Phase 9: Figure Generation (Figures 1 through 7, the sibling figure, and per-feature supplements).
Generates vector PDFs and 300 DPI PNGs in figures/, following the furtherAnalysis "Figures" list:
  - Figure 1: Identifiability matrix M per feature (rows ses-1, columns ses-2, one colour scale and colour bar each)
  - Figure 2: U_1, U_2, W_1, W_2, I_self and I_others with 95% bootstrap CIs, side by side per feature
  - Figure 3: Universality-individuality plane per feature, coloured by region (nodes) or region pair (edges),
    marker shape for within- vs between-region, Step 5a classes from the FDR-corrected tests
  - Figure 4: Scalp maps: nodes coloured by I_diff(e); edges drawn only where I_diff(e) is FDR-significant
  - Figure 5: Delta U per feature with CI; self(a) against inter-session interval, NORB00064 and NORB00069 labelled
  - Figure 6: Variance partition stacked bars per feature, averaged over elements and by region
  - Figure 7: Element heatmaps for ses-1 and ses-2, rows sorted by |Delta age|, colour scale fixed at 0-1,
    hemispheric category shown with midline labelled "M"
  - Sibling figure: per feature and session, the distribution of pair similarities with the sibling pair and the
    matched pairs marked
  - Threshold bars (not in the plan's list; as in the legacy notebooks): per session, the percentage of element
    similarities J above 0.30, 0.50, 0.75 and 0.90 per region (nodes) or region pair (edges), stacked by hemisphere
Every number written on a figure is read from the results; nothing is typed in by hand. Figures shown for a few
key features have a supplement covering all 26 features (a grid or a multi-page PDF).
"""
import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.multitest import multipletests

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
        SESSIONS,
        SUBJECT_IDS,
    )
    from analysis_pipeline.utils import get_edge_hemisphere_pair, get_node_hemisphere
else:
    from .config import (
        EDGE_LABELS,
        FEATURE_DEFS,
        FIGURES_DIR,
        NODE_LABELS,
        NODE_REGIONS,
        OUTPUT_TENSORS_DIR,
        REGION_PAIRS,
        SESSIONS,
        SUBJECT_IDS,
    )
    from .utils import get_edge_hemisphere_pair, get_node_hemisphere

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

REGIONS = ["F", "C", "T", "P", "O"]
GROUP_COLORS = {
    "node": dict(zip(REGIONS, [plt.get_cmap("tab10")(i) for i in range(5)])),
    "edge": dict(zip(REGION_PAIRS, [plt.get_cmap("tab20")(i) for i in range(15)])),
}
HEMI_CATEGORIES = {"node": ["L", "R", "M"], "edge": ["L-L", "R-R", "L-R", "L-M", "R-M", "M-M"]}
TWIN_IDS = ("sub-NORB00086", "sub-NORB00087")
SHORT_GAP_IDS = ("sub-NORB00064", "sub-NORB00069")
KEY_FIDS = ["f01", "f04", "f06", "f10", "f13", "f24"]
IDIFF_LABEL = r"$I_{\mathrm{diff}}$"
IDIFF_E_LABEL = r"$I_{\mathrm{diff}}(e)$"


def feature_title(fid: str) -> str:
    return f"{fid}: {FEATURE_DEFS[fid]['name'].replace('_', ' ')}"


def save_figure(fig, stem: str):
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def global_q(results: dict, p_key: str, q_key: str) -> dict:
    """FDR q of a global test across the feature sets: the saved q if present, else BH of the saved p (Step 7.4 ii)."""
    fids = list(FEATURE_DEFS)
    if all(q_key in results[f] for f in fids):
        return {f: results[f][q_key] for f in fids}
    q = multipletests([results[f][p_key] for f in fids], method="fdr_bh")[1]
    return dict(zip(fids, q))


def element_info(fid: str):
    """Labels, region (nodes) or region pair (edges), within-region flag and group order of a feature's elements."""
    ftype = FEATURE_DEFS[fid]["type"]
    if ftype == "node":
        return NODE_LABELS, np.array([NODE_REGIONS[n] for n in NODE_LABELS]), np.ones(len(NODE_LABELS), bool), REGIONS
    groups, within = [], []
    for e in EDGE_LABELS:
        r1, r2 = sorted((NODE_REGIONS[n] for n in e.split("-")), key=REGIONS.index)
        groups.append(f"{r1}-{r2}")
        within.append(r1 == r2)
    return EDGE_LABELS, np.array(groups), np.array(within), REGION_PAIRS


def hemi_category(fid: str) -> list:
    if FEATURE_DEFS[fid]["type"] == "node":
        return [get_node_hemisphere(n) for n in NODE_LABELS]
    return [get_edge_hemisphere_pair(e) for e in EDGE_LABELS]


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


# ----------------------------------------------------------------------------------------------
# Figure 1: identifiability matrices
# ----------------------------------------------------------------------------------------------

def matrix_panel(fig, ax, fid, i_res, q_idiff, short_subjs, fontsize):
    M = np.load(OUTPUT_TENSORS_DIR / f"M_mat_{fid}.npy")
    im = ax.imshow(M, cmap="hot", aspect="auto")
    ax.set_xticks(range(len(short_subjs)))
    ax.set_yticks(range(len(short_subjs)))
    ax.set_xticklabels(short_subjs, rotation=90, fontsize=fontsize - 2)
    ax.set_yticklabels(short_subjs, fontsize=fontsize - 2)
    ax.set_xlabel("ses-2 infant", fontsize=fontsize)
    ax.set_ylabel("ses-1 infant", fontsize=fontsize)
    for i in range(len(short_subjs)):
        ax.add_patch(patches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.2, zorder=5))
    r = i_res[fid]
    ax.set_title(f"{feature_title(fid)}\n" + IDIFF_LABEL + f" = {r['I_diff']:+.4f} (q = {q_idiff[fid]:.3f}), accuracy = {r['acc_mean']:.1f}%",
                 fontsize=fontsize, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    cbar.set_label("Weighted Jaccard S", fontsize=fontsize - 1)


def generate_figure_1(i_res, short_subjs):
    """Figure 1: Cross-session identifiability matrix M per feature, with its own colour scale and colour bar."""
    print("[1/9] Generating Figure 1: Identifiability Matrices M...")
    q_idiff = global_q(i_res, "p_idiff", "q_idiff")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, fid in zip(axes.flat, KEY_FIDS):
        matrix_panel(fig, ax, fid, i_res, q_idiff, short_subjs, fontsize=10)
    fig.suptitle(r"Figure 1: Identifiability Matrices $M$ (rows: ses-1, columns: ses-2, same infant order)" + "\n"
                 + "Cyan boxes: the same infant at both sessions (diagonal)", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_figure(fig, "Figure_1_Identifiability_Matrices")

    fig26, axes26 = plt.subplots(5, 6, figsize=(30, 24))
    all_fids = list(FEATURE_DEFS)
    for idx, ax in enumerate(axes26.flat):
        if idx < len(all_fids):
            matrix_panel(fig26, ax, all_fids[idx], i_res, q_idiff, short_subjs, fontsize=8)
        else:
            ax.axis("off")
    fig26.suptitle(r"Figure 1 (Supplementary): Identifiability Matrices $M$ for All 26 Feature Sets", fontsize=16, fontweight="bold")
    fig26.tight_layout()
    save_figure(fig26, "Figure_1_Supplementary_All_26_Matrices")


# ----------------------------------------------------------------------------------------------
# Figure 2: U, W, I_self, I_others with CIs
# ----------------------------------------------------------------------------------------------

def generate_figure_2(u_res, w_res, i_res):
    """Figure 2: U_1, U_2, W_1, W_2, I_self, I_others per feature, each with its 95% infant bootstrap CI."""
    print("[2/9] Generating Figure 2: U, W, I_self and I_others with 95% CIs...")
    all_fids = list(FEATURE_DEFS)
    y_pos = np.arange(len(all_fids))[::-1]  # f01 at the top
    series = [
        ("W_1 (ses-1)", "s", "#2ca02c", lambda f: (w_res[f]["ses-1"]["W_global"], w_res[f]["ses-1"]["W_bootstrap"]["ci_95"])),
        ("W_2 (ses-2)", "s", "#98df8a", lambda f: (w_res[f]["ses-2"]["W_global"], w_res[f]["ses-2"]["W_bootstrap"]["ci_95"])),
        ("U_1 (ses-1)", "o", "#1f77b4", lambda f: (u_res[f]["ses-1"]["U_global"], u_res[f]["ses-1"]["bootstrap"]["ci_95"])),
        ("U_2 (ses-2)", "D", "#17becf", lambda f: (u_res[f]["ses-2"]["U_global"], u_res[f]["ses-2"]["bootstrap"]["ci_95"])),
        ("I_self", "^", "#d62728", lambda f: (i_res[f]["I_self"], i_res[f]["bootstrap"]["self_ci_95"])),
        ("I_others", "v", "#7f7f7f", lambda f: (i_res[f]["I_others"], i_res[f]["bootstrap"]["others_ci_95"])),
    ]
    offsets = np.linspace(0.3, -0.3, len(series))
    fig, ax = plt.subplots(figsize=(15, 14))
    lo, hi = np.inf, -np.inf
    for i, fid in enumerate(all_fids):
        for (label, marker, color, get), off in zip(series, offsets):
            est, ci = get(fid)
            lo, hi = min(lo, ci[0]), max(hi, ci[1])
            ax.errorbar(est, y_pos[i] + off, xerr=[[est - ci[0]], [ci[1] - est]], fmt=marker, color=color,
                        markersize=5, capsize=2, label=label if i == 0 else "")
        if i % 2 == 1:
            ax.axhspan(y_pos[i] - 0.5, y_pos[i] + 0.5, color="#f5f5f5", zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{feature_title(f)} ({FEATURE_DEFS[f]['dim']})" for f in all_fids], fontsize=9)
    ax.set_ylim(-0.6, len(all_fids) - 0.4)
    pad = 0.03 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("Weighted Jaccard S (estimate with 95% infant bootstrap CI)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 2: Within-session (W), between-infant (U) and cross-session (I_self, I_others) similarity per feature",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "Figure_2_Universality_Identifiability_Forest")


# ----------------------------------------------------------------------------------------------
# Figure 3: universality-individuality plane
# ----------------------------------------------------------------------------------------------

def plane_panel(ax, fid, u_res, i_res, tax_res, fontsize=10, marker_size=30):
    """
    One point per element at x = U(e) = (U_1(e) + U_2(e)) / 2, y = I_diff(e) (Step 5a). Colour: region (nodes) or
    region pair (edges); marker: circle within-region, triangle between-region. Filled: U(e) significant at both
    sessions; black ring: I_diff(e) significant (q < 0.05), as in the Step 5a classes.
    """
    ftype = FEATURE_DEFS[fid]["type"]
    _labels, groups, within, order = element_info(fid)
    U = (np.array(u_res[fid]["ses-1"]["U_elem"], float) + np.array(u_res[fid]["ses-2"]["U_elem"], float)) / 2.0
    I = np.array(i_res[fid]["I_diff_elem"], float)
    classes = tax_res[fid]["classes"]
    sig_u = np.array([c in ("Universal", "Universal and Individual") for c in classes])
    sig_i = np.array([c in ("Individual", "Universal and Individual") for c in classes])
    for g in order:
        color = GROUP_COLORS[ftype][g]
        for is_within, marker in ((True, "o"), (False, "^")):
            m = (groups == g) & (within == is_within)
            ax.scatter(U[m & sig_u], I[m & sig_u], marker=marker, s=marker_size, color=color, zorder=3)
            ax.scatter(U[m & ~sig_u], I[m & ~sig_u], marker=marker, s=marker_size, facecolors="none", edgecolors=color, zorder=3)
    ax.scatter(U[sig_i], I[sig_i], s=marker_size * 4, facecolors="none", edgecolors="black", linewidths=1.4, zorder=4)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, zorder=2)
    c = tax_res[fid]["counts"]
    ax.set_title(f"{feature_title(fid)}\nUniversal {c['Universal']} | Individual {c['Individual']} | "
                 f"Both {c['Universal and Individual']} | Neither {c['Neither']}", fontsize=fontsize, fontweight="bold")
    ax.set_xlabel(r"$U(e) = (U_1(e) + U_2(e)) / 2$", fontsize=fontsize)
    ax.set_ylabel(IDIFF_E_LABEL, fontsize=fontsize)
    ax.grid(True, linestyle=":", alpha=0.5)


def plane_legend_handles():
    region_handles = [Line2D([], [], marker="o", linestyle="", color=GROUP_COLORS["node"][r], label=f"Node region {r}") for r in REGIONS]
    pair_handles = [Line2D([], [], marker="s", linestyle="", color=GROUP_COLORS["edge"][rp], label=rp) for rp in REGION_PAIRS]
    style_handles = [
        Line2D([], [], marker="o", linestyle="", color="#555555", label="Node, or within-region edge"),
        Line2D([], [], marker="^", linestyle="", color="#555555", label="Between-region edge"),
        Line2D([], [], marker="o", linestyle="", color="#555555", label="Filled: U(e) significant at both sessions"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#555555", label="Open: U(e) not significant"),
        Line2D([], [], marker="o", linestyle="", markersize=12, markerfacecolor="none", markeredgecolor="black", label="Black ring: I_diff(e) significant"),
    ]
    return region_handles, pair_handles, style_handles


def add_plane_legends(fig):
    region_handles, pair_handles, style_handles = plane_legend_handles()
    fig.legend(handles=region_handles, loc="lower left", bbox_to_anchor=(0.02, -0.10), ncol=1, fontsize=9, title="Node features: region")
    fig.legend(handles=pair_handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=8, fontsize=9, title="Edge features: region pair")
    fig.legend(handles=style_handles, loc="lower right", bbox_to_anchor=(0.98, -0.10), ncol=1, fontsize=9, title="Marker")


def generate_figure_3(u_res, i_res, tax_res):
    """Figure 3: Universality-individuality plane (Step 5a), key features and all 26 features."""
    print("[3/9] Generating Figure 3: Universality-Individuality Planes...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, fid in zip(axes.flat, ["f01", "f06", "f10", "f24"]):
        plane_panel(ax, fid, u_res, i_res, tax_res)
    fig.suptitle("Figure 3: Universality-Individuality Plane (Step 5a)\nClasses from the FDR-corrected tests, not from the position in the plane",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    add_plane_legends(fig)
    save_figure(fig, "Figure_3_Universality_Individuality_Plane")

    fig26, axes26 = plt.subplots(5, 6, figsize=(32, 26))
    all_fids = list(FEATURE_DEFS)
    for idx, ax in enumerate(axes26.flat):
        if idx < len(all_fids):
            plane_panel(ax, all_fids[idx], u_res, i_res, tax_res, fontsize=8, marker_size=14)
        else:
            ax.axis("off")
    fig26.suptitle("Figure 3 (Supplementary): Universality-Individuality Plane for All 26 Feature Sets", fontsize=16, fontweight="bold")
    fig26.tight_layout()
    add_plane_legends(fig26)
    save_figure(fig26, "Figure_3_Supplementary_All_26_Planes")


# ----------------------------------------------------------------------------------------------
# Figure 4: scalp maps
# ----------------------------------------------------------------------------------------------

def scalp_panel(fig, ax, fid, i_res, fontsize=10):
    """
    Node features: every electrode coloured by I_diff(e), the three largest values labelled.
    Edge features: only edges with FDR q < 0.05 for I_diff(e) are drawn, coloured by I_diff(e).
    The colour scale is centred on 0, because I_diff(e) can be negative.
    """
    values = np.array(i_res[fid]["I_diff_elem"], float)
    q = np.array(i_res[fid]["q_elem"], float)
    draw_scalp_outline(ax, feature_title(fid))
    vmax = float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    if FEATURE_DEFS[fid]["type"] == "node":
        for idx, node in enumerate(NODE_LABELS):
            x, y = ELECTRODE_2D[node]
            val = values[idx]
            sig = q[idx] < 0.05
            ax.scatter([x], [y], s=350, color=cmap(norm(val)) if np.isfinite(val) else "#dddddd",
                       edgecolors="black", linewidth=3.0 if sig else 1.0, zorder=5)
            ax.text(x, y, node, ha="center", va="center", fontsize=fontsize - 3, fontweight="bold", zorder=6)
        for idx in np.argsort(-np.nan_to_num(values, nan=-np.inf))[:3]:
            x, y = ELECTRODE_2D[NODE_LABELS[idx]]
            ax.annotate(f"{values[idx]:+.3f}", (x, y), xytext=(14, 10 if y >= 0 else -16), textcoords="offset points",
                        fontsize=fontsize - 2, fontweight="bold", arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1})
        note = "Thick ring: FDR q < 0.05 | labels: three largest values"
    else:
        for node, (x, y) in ELECTRODE_2D.items():
            ax.scatter([x], [y], s=200, color="#e0e0e0", edgecolors="#666666", linewidth=1.0, zorder=4)
            ax.text(x, y, node, ha="center", va="center", fontsize=fontsize - 3, color="#333333", zorder=5)
        sig_idx = np.where(q < 0.05)[0]
        for idx in sig_idx:
            u_node, v_node = EDGE_LABELS[idx].split("-")
            (x1, y1), (x2, y2) = ELECTRODE_2D[u_node], ELECTRODE_2D[v_node]
            # Black outline keeps values near 0 (drawn almost white) visible; edges stay under the electrode labels
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=6.0, zorder=2, solid_capstyle="round")
            ax.plot([x1, x2], [y1, y2], color=cmap(norm(values[idx])), linewidth=4.0, zorder=3, solid_capstyle="round")
            if len(sig_idx) <= 6:
                ax.annotate(f"{EDGE_LABELS[idx]}\n{values[idx]:+.3f} (q = {q[idx]:.3f})", ((x1 + x2) / 2, (y1 + y2) / 2),
                            xytext=(15, 12), textcoords="offset points", fontsize=fontsize - 3, fontweight="bold",
                            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#888888", "lw": 1},
                            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1})
        note = f"Edges drawn: FDR q < 0.05 only ({len(sig_idx)} of {len(EDGE_LABELS)})"
    ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", fontsize=fontsize - 2, style="italic")
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", fraction=0.045, pad=0.06)
    cbar.set_label(IDIFF_E_LABEL, fontsize=fontsize - 1)


def generate_figure_4(i_res):
    """Figure 4: Scalp maps of I_diff(e), key features and all 26 features."""
    print("[4/9] Generating Figure 4: Scalp Maps of I_diff(e)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 15))
    for ax, fid in zip(axes.flat, ["f06", "f04", "f20", "f12"]):
        scalp_panel(fig, ax, fid, i_res)
    fig.suptitle("Figure 4: Scalp Maps of Individuality " + IDIFF_E_LABEL + "\nNodes coloured by value; edges drawn only where FDR-significant",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_figure(fig, "Figure_4_Topography_Hubs_Fingerprints")

    fig26, axes26 = plt.subplots(5, 6, figsize=(30, 30))
    all_fids = list(FEATURE_DEFS)
    for idx, ax in enumerate(axes26.flat):
        if idx < len(all_fids):
            scalp_panel(fig26, ax, all_fids[idx], i_res, fontsize=9)
        else:
            ax.axis("off")
    fig26.suptitle("Figure 4 (Supplementary): Scalp Maps of " + IDIFF_E_LABEL + " for All 26 Feature Sets", fontsize=16, fontweight="bold")
    fig26.tight_layout()
    save_figure(fig26, "Figure_4_Supplementary_All_26_Scalp_Maps")


# ----------------------------------------------------------------------------------------------
# Figure 5: development
# ----------------------------------------------------------------------------------------------

def generate_figure_5(dev_res, ages_days):
    """Figure 5: Delta U per feature with CI, and self(a) against the inter-session interval."""
    print("[5/9] Generating Figure 5: Delta U and Self-Similarity vs Interval...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    all_fids = list(FEATURE_DEFS)
    q_delta = global_q(dev_res, "p_value", "q_value")
    y_pos = np.arange(len(all_fids))[::-1]
    for i, fid in enumerate(all_fids):
        d = dev_res[fid]["delta_U_global"]
        ci = dev_res[fid]["ci_95"]
        color = "#7f7f7f" if q_delta[fid] >= 0.05 else ("#2ca02c" if d > 0 else "#d62728")
        ax1.errorbar(d, y_pos[i], xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", color=color, markersize=5, capsize=3)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([feature_title(f) for f in all_fids], fontsize=8.5)
    ax1.set_xlabel(r"$\Delta U = U_2 - U_1$ (with 95% infant bootstrap CI)", fontsize=10, fontweight="bold")
    ax1.set_title(r"A. Change in universality $\Delta U$", fontsize=11, fontweight="bold")
    ax1.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#2ca02c", label=r"$\Delta U > 0$, q < 0.05"),
        Line2D([], [], marker="o", linestyle="", color="#d62728", label=r"$\Delta U < 0$, q < 0.05"),
        Line2D([], [], marker="o", linestyle="", color="#7f7f7f", label="q ≥ 0.05 (FDR across features)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # Panel B: self(a) against the inter-session interval, broadband correlation (f01)
    intervals_days = np.array([ages_days[s]["ses-2"] - ages_days[s]["ses-1"] for s in SUBJECT_IDS])
    fid_target = "f01"
    self_vals = np.diag(np.load(OUTPUT_TENSORS_DIR / f"M_mat_{fid_target}.npy"))
    ax2.scatter(intervals_days, self_vals, color="#333333", s=70, alpha=0.8, zorder=4, label=f"Infants (N = {len(SUBJECT_IDS)})")
    slope, intercept = np.polyfit(intervals_days, self_vals, 1)
    x_grid = np.linspace(intervals_days.min() - 2, intervals_days.max() + 2, 100)
    ax2.plot(x_grid, intercept + slope * x_grid, color="#1f77b4", linewidth=2.0, label="Least-squares line")
    for subj, color, offset in zip(SHORT_GAP_IDS, ("#d62728", "#1f77b4"), ((15, 5), (15, -30))):
        idx = SUBJECT_IDS.index(subj)
        ax2.scatter([intervals_days[idx]], [self_vals[idx]], color=color, s=130, edgecolors="black", linewidth=1.5, zorder=6)
        ax2.annotate(f"{subj}\nInterval: {intervals_days[idx]:.1f} days\nself(a): {self_vals[idx]:.3f}",
                     (intervals_days[idx], self_vals[idx]), xytext=offset, textcoords="offset points", fontsize=9, fontweight="bold",
                     bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": color, "lw": 1.2},
                     arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
    corr = dev_res[fid_target]["interval_corr"]["self_vs_interval"]
    q_text = f", q = {corr['q']:.4f}" if "q" in corr else ""
    ci_text = f" [{corr['ci_95'][0]:+.3f}, {corr['ci_95'][1]:+.3f}]" if "ci_95" in corr else ""
    ax2.set_xlabel("Inter-session interval (days)", fontsize=10, fontweight="bold")
    ax2.set_ylabel(f"self(a) = M[a, a] ({feature_title(fid_target)})", fontsize=10, fontweight="bold")
    ax2.set_title("B. Self-similarity against the inter-session interval\n"
                  + rf"Spearman $\rho = {corr['rho']:+.3f}${ci_text} (permutation p = {corr['p']:.4f}{q_text})", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    fig.suptitle("Figure 5: Development Between Sessions", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_figure(fig, "Figure_5_Developmental_Interval_Scatter")


# ----------------------------------------------------------------------------------------------
# Figure 6: variance partition
# ----------------------------------------------------------------------------------------------

def variance_means(elems, keep):
    """Mean trait, session and epoch proportion (%) over the elements in keep, ignoring non-converged (NaN) fits."""
    vals = np.array([[e["trait_prop"], e["session_prop"], e["epoch_prop"]] for e, k in zip(elems, keep) if k], float)
    ok = ~np.isnan(vals).any(axis=1)
    return (vals[ok].mean(axis=0) * 100.0 if ok.any() else np.full(3, np.nan)), int(ok.sum())


def stacked_bars(ax, names, props, fontsize: float = 9):
    props = np.array(props, float)
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for k, (label, color) in enumerate([(r"Infant / trait ($\sigma^2_{\mathrm{infant}}$)", "#1f77b4"),
                                        (r"Session ($\sigma^2_{\mathrm{session}}$)", "#ff7f0e"),
                                        (r"Epoch / state ($\sigma^2_{\mathrm{epoch}}$)", "#b0b0b0")]):
        ax.barh(y, props[:, k], left=left, color=color, edgecolor="none", label=label)
        left = left + np.nan_to_num(props[:, k])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=fontsize)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Variance proportion (%)", fontsize=fontsize + 1, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def region_breakdown(fid, var_res):
    """Mean proportions per region (nodes) or region pair (edges) for one feature."""
    elems = var_res[fid]["elements"]
    _labels, groups, _within, order = element_info(fid)
    names, props = [], []
    for g in order:
        mean, n = variance_means(elems, groups == g)
        if n:
            names.append(f"{g} (n = {n})")
            props.append(mean)
    return names, props


def generate_figure_6(var_res):
    """Figure 6: Variance partition stacked bars per feature, averaged over elements and by region (Step 5b)."""
    print("[6/9] Generating Figure 6: Variance Partition Stacked Bars...")
    valid_fids = [f for f in FEATURE_DEFS if not var_res[f]["elements"][0]["skipped"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    names, props = [], []
    for fid in valid_fids:
        mean, n = variance_means(var_res[fid]["elements"], [True] * len(var_res[fid]["elements"]))
        names.append(f"{feature_title(fid)} ({n}/{len(var_res[fid]['elements'])} fits)")
        props.append(mean)
    stacked_bars(ax1, names, props, fontsize=8.5)
    ax1.set_title("A. Mean over the elements of each feature\n(converged fits only; binary adjacency skipped)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    reg_names, reg_props = region_breakdown("f01", var_res)
    stacked_bars(ax2, reg_names, reg_props)
    ax2.set_title(f"B. By region pair: {feature_title('f01')}\n(converged fits only)", fontsize=11, fontweight="bold")

    fig.suptitle("Figure 6: Variance Partition: Infant (Trait) vs Session vs Epoch (State)", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_figure(fig, "Figure_6_Variance_Partition_Stacked")

    n_cols = 4
    n_rows = int(np.ceil(len(valid_fids) / n_cols))
    fig_r, axes_r = plt.subplots(n_rows, n_cols, figsize=(26, 5 * n_rows))
    for idx, ax in enumerate(axes_r.flat):
        if idx < len(valid_fids):
            fid = valid_fids[idx]
            reg_names, reg_props = region_breakdown(fid, var_res)
            stacked_bars(ax, reg_names, reg_props, fontsize=8)
            ax.set_title(feature_title(fid), fontsize=10, fontweight="bold")
        else:
            ax.axis("off")
    handles, labels = axes_r.flat[0].get_legend_handles_labels()
    fig_r.legend(handles, labels, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig_r.suptitle("Figure 6 (Supplementary): Variance Partition by Region (Nodes) or Region Pair (Edges), Converged Fits Only",
                   fontsize=16, fontweight="bold")
    fig_r.tight_layout(rect=(0, 0.02, 1, 0.98))
    save_figure(fig_r, "Figure_6_Supplementary_Regions_All_Features")


# ----------------------------------------------------------------------------------------------
# Figure 7: element heatmaps sorted by |Delta age|
# ----------------------------------------------------------------------------------------------

def heatmap_figure(fid, ages_days):
    """
    Element heatmaps of J for every infant pair at ses-1 and ses-2, rows sorted by |Delta age| (smallest first),
    colour scale fixed at 0-1. A strip above each heatmap gives each column's hemispheric category; "M" is midline.
    """
    ftype = FEATURE_DEFS[fid]["type"]
    labels, groups, _within, _order = element_info(fid)
    categories = HEMI_CATEGORIES[ftype]
    cat_idx = np.array([categories.index(c) for c in hemi_category(fid)])
    cat_cmap = ListedColormap([plt.get_cmap("Set2")(i) for i in range(len(categories))])
    p_idx, q_idx = np.triu_indices(len(SUBJECT_IDS), k=1)
    twin_pair = next(k for k, (i, j) in enumerate(zip(p_idx, q_idx))
                     if {SUBJECT_IDS[i], SUBJECT_IDS[j]} == set(TWIN_IDS))

    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    images = []
    for ax, ses in zip(axes, SESSIONS):
        J = np.load(OUTPUT_TENSORS_DIR / f"J_pair_{fid}_{ses}.npy")
        dage = np.array([abs(ages_days[SUBJECT_IDS[i]][ses] - ages_days[SUBJECT_IDS[j]][ses]) for i, j in zip(p_idx, q_idx)])
        order_rows = np.argsort(dage, kind="stable")
        images.append(ax.imshow(J[order_rows], aspect="auto", cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest"))
        ax.set_ylabel(r"Infant pairs, sorted by $|\Delta\mathrm{age}|$ (smallest at the top)", fontsize=10, fontweight="bold")
        if ftype == "node":
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=9)
            ax.set_xlabel("Electrode", fontsize=10, fontweight="bold")
        else:
            starts = [i for i in range(len(groups)) if i == 0 or groups[i] != groups[i - 1]]
            ends = starts[1:] + [len(groups)]
            ax.set_xticks([(s + e - 1) / 2 for s, e in zip(starts, ends)])
            ax.set_xticklabels([groups[s] for s in starts], rotation=90, fontsize=8.5)
            for s in starts[1:]:
                ax.axvline(s - 0.5, color="cyan", linewidth=0.8, alpha=0.8)
            ax.set_xlabel("Edges, grouped by region pair", fontsize=10, fontweight="bold")
        twin_row = int(np.where(order_rows == twin_pair)[0][0])
        ax.axhline(twin_row, color="lime", linewidth=2.0, zorder=5)
        ax.annotate(f"Twins (86, 87): |Δage| = {dage[twin_pair]:.1f} days, row {twin_row + 1}",
                    xy=(0, twin_row), xytext=(60, -60), textcoords="offset points", fontsize=8.5, fontweight="bold", color="white",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "#111111", "edgecolor": "lime", "lw": 1.5},
                    arrowprops={"arrowstyle": "->", "color": "lime", "lw": 1.5})
        # Hemispheric category strip above the heatmap
        strip = make_axes_locatable(ax).append_axes("top", size="3%", pad=0.05)
        strip.imshow(cat_idx[None, :], aspect="auto", cmap=cat_cmap, vmin=-0.5, vmax=len(categories) - 0.5, interpolation="nearest")
        strip.set_xticks([])
        strip.set_yticks([])
        if ftype == "node":
            for x, c in enumerate(hemi_category(fid)):
                strip.text(x, 0, c, ha="center", va="center", fontsize=8, fontweight="bold")
        strip.set_title(f"{ses}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.legend(handles=[patches.Patch(color=cat_cmap(i), label=c) for i, c in enumerate(categories)],
               loc="lower left", bbox_to_anchor=(0.01, -0.04), ncol=len(categories), fontsize=10,
               title="Strip above each heatmap: hemispheric category of the column (L left, R right, M midline)")
    cbar = fig.colorbar(images[-1], ax=list(axes), orientation="horizontal", fraction=0.04, pad=0.08, shrink=0.6)
    cbar.set_label("Weighted Jaccard J (fixed scale 0 to 1)", fontsize=10, fontweight="bold")
    fig.suptitle(f"Figure 7: Element Similarity J for Every Infant Pair, {feature_title(fid)}\n"
                 r"Rows sorted by age difference $|\Delta\mathrm{age}|$ at that session", fontsize=14, fontweight="bold", y=0.99)
    return fig


def generate_figure_7(ages_days):
    """Figure 7: Element heatmaps sorted by |Delta age|; f01 as the main figure, all 26 features in a multi-page PDF."""
    print("[7/9] Generating Figure 7: Element Heatmaps Sorted by |Delta age|...")
    save_figure(heatmap_figure("f01", ages_days), "Figure_7_Ordered_Element_Heatmaps")
    with PdfPages(FIGURES_DIR / "Figure_7_Supplementary_All_26_Heatmaps.pdf") as pdf:
        for fid in FEATURE_DEFS:
            fig = heatmap_figure(fid, ages_days)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Sibling figure
# ----------------------------------------------------------------------------------------------

def sibling_figure(fid, twin_res, ages_days):
    """
    Distribution of the pair similarities S at each session, with the sibling pair and the two matched comparison
    sets (Step 6 sibling item 2) marked: pairs involving infant 86 or 87, and non-sibling pairs with |Delta age| <= 7 days.
    """
    n = len(SUBJECT_IDS)
    p_idx, q_idx = np.triu_indices(n, k=1)
    i86, i87 = SUBJECT_IDS.index(TWIN_IDS[0]), SUBJECT_IDS.index(TWIN_IDS[1])
    is_twin = np.array([{i, j} == {i86, i87} for i, j in zip(p_idx, q_idx)])
    involves_twin = np.array([bool({i, j} & {i86, i87}) for i, j in zip(p_idx, q_idx)]) & ~is_twin

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for s, (ax, ses) in enumerate(zip(axes, SESSIONS), 1):
        S = np.load(OUTPUT_TENSORS_DIR / f"S_mat_{fid}_{ses}.npy")
        vals = S[p_idx, q_idx]
        dage = np.array([abs(ages_days[SUBJECT_IDS[i]][ses] - ages_days[SUBJECT_IDS[j]][ses]) for i, j in zip(p_idx, q_idx)])
        age_matched = (dage <= 7.0) & ~is_twin
        twin_val = float(vals[is_twin][0])
        rank = twin_res[fid][f"rank_among_171_ses{s}"]
        sns.histplot(vals, kde=True, ax=ax, color="#1f77b4", stat="density", label=f"All {len(vals)} infant pairs")
        ax.plot(vals[involves_twin], np.full(involves_twin.sum(), 0.02), "|", color="#ff7f0e", markersize=18, mew=1.5,
                transform=ax.get_xaxis_transform(), label=f"Pairs with infant 86 or 87 (n = {involves_twin.sum()}, median {np.median(vals[involves_twin]):.3f})")
        ax.plot(vals[age_matched], np.full(age_matched.sum(), 0.07), "|", color="#2ca02c", markersize=18, mew=1.5,
                transform=ax.get_xaxis_transform(), label=f"Non-sibling pairs, |Δage| ≤ 7 days (n = {age_matched.sum()}, median {np.median(vals[age_matched]):.3f})")
        ax.axvline(twin_val, color="red", linewidth=2.5, linestyle="--", label=f"Sibling pair: {twin_val:.3f} (percentile rank {rank:.1f})")
        ax.set_xlabel(f"Recording similarity S ({feature_title(fid)})", fontsize=10, fontweight="bold")
        ax.set_title(f"{ses}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
    fig.suptitle(f"Sibling Figure: Pair Similarities with the Sibling Pair and Matched Pairs Marked, {feature_title(fid)}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def generate_sibling_figure(twin_res, ages_days):
    """Sibling figure (Step 6): f01 as the main figure, all 26 features in a multi-page PDF."""
    print("[8/9] Generating the Sibling Figure: Pair Similarity Distributions...")
    save_figure(sibling_figure("f01", twin_res, ages_days), "Figure_7_Supplementary_Twin_Distributions")
    with PdfPages(FIGURES_DIR / "Figure_7_Supplementary_Twin_Distributions_All_26.pdf") as pdf:
        for fid in FEATURE_DEFS:
            fig = sibling_figure(fid, twin_res, ages_days)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Threshold bars: share of element similarities J above a threshold, by region and hemisphere
# ----------------------------------------------------------------------------------------------

THRESHOLD_PAIRS = ((0.75, 0.90), (0.50, 0.30))  # (left bar, right bar) per row, as in the legacy notebooks


def threshold_percentages(J, groups, order, hemi, categories, threshold):
    """
    Per group: percentage of the group's defined J values (all infant pairs x its elements) above the threshold,
    split by hemispheric category, so the stacked segments add up to the group's total. 0/0 (NaN) values are left out.
    """
    pct = np.full((len(order), len(categories)), np.nan)
    for gi, g in enumerate(order):
        in_g = groups == g
        n_defined = np.sum(~np.isnan(J[:, in_g]))
        if n_defined:
            for ci, c in enumerate(categories):
                pct[gi, ci] = 100.0 * np.sum(J[:, in_g & (hemi == c)] > threshold) / n_defined
    return pct


def threshold_panel(ax, fid, J, thresholds, fontsize=9, label_min=2.0, vertical=False):
    """
    Grouped stacked bars: one pair of bars per region (nodes) or region pair (edges), stacked by hemispheric category.
    Segments above label_min (%) are labelled. vertical=True writes the numbers and group labels rotated, for narrow bars.
    """
    ftype = FEATURE_DEFS[fid]["type"]
    _labels, groups, _within, order = element_info(fid)
    categories = HEMI_CATEGORIES[ftype]
    hemi = np.array(hemi_category(fid))
    x = np.arange(len(order))
    width = 0.38
    rotation = 90 if vertical else 0
    for offset, threshold in zip((-width / 2, width / 2), thresholds):
        pct = threshold_percentages(J, groups, order, hemi, categories, threshold)
        bottom = np.zeros(len(order))
        for ci in range(len(categories)):
            vals = np.nan_to_num(pct[:, ci])
            bars = ax.bar(x + offset, vals, width, bottom=bottom, color=plt.get_cmap("Set2")(ci), edgecolor="white", linewidth=0.5)
            ax.bar_label(bars, labels=[f"{v:.1f}" if v > label_min else "" for v in vals], label_type="center",
                         fontsize=fontsize - 3, rotation=rotation)
            bottom += vals
        for xi, total in zip(x + offset, bottom):
            ax.text(xi, total + 0.8, f"{total:.1f}", ha="center", va="bottom", fontsize=fontsize - 3, fontweight="bold", rotation=rotation)
    ax.set_xticks(x)
    if vertical:
        ax.set_xticklabels([f"{g} (n = {np.sum(groups == g)})" for g in order], fontsize=fontsize - 1, rotation=90)
    else:
        ax.set_xticklabels([f"{g}\n(n = {np.sum(groups == g)})" for g in order], fontsize=fontsize - 1)
    ax.set_ylim(0, 124 if vertical else 105)  # room for the rotated totals above a full bar
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylabel("Percentage of defined J values (%)", fontsize=fontsize, fontweight="bold")
    ax.set_xlabel("Region" if ftype == "node" else "Region pair", fontsize=fontsize, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def threshold_figure(fid):
    """
    For each session, the percentage of the element similarities J (every infant pair x element of a region or region
    pair) above 0.75 and 0.90 (top) and above 0.50 and 0.30 (bottom), stacked by hemispheric category.
    """
    categories = HEMI_CATEGORIES[FEATURE_DEFS[fid]["type"]]
    fig, axes = plt.subplots(2, 2, figsize=(28, 14), sharey=True)
    for col, ses in enumerate(SESSIONS):
        J = np.load(OUTPUT_TENSORS_DIR / f"J_pair_{fid}_{ses}.npy")
        n_undefined = int(np.isnan(J).sum())
        undefined_text = f"; {n_undefined} of {J.size} values 0/0, left out" if n_undefined else ""
        for row, thresholds in enumerate(THRESHOLD_PAIRS):
            ax = axes[row, col]
            threshold_panel(ax, fid, J, thresholds)
            ax.set_title(f"{ses}: J > {thresholds[0]:.2f} (left bar) and J > {thresholds[1]:.2f} (right bar){undefined_text}",
                         fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.legend(handles=[patches.Patch(color=plt.get_cmap("Set2")(i), label=c) for i, c in enumerate(categories)],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(categories), fontsize=10,
               title="Hemispheric category (L left, R right, M midline); segments add up to the bar total written on top")
    fig.suptitle(f"Figure 7 (Supplementary): Share of Element Similarities J Above a Threshold, {feature_title(fid)}\n"
                 f"Each bar: all {len(SUBJECT_IDS) * (len(SUBJECT_IDS) - 1) // 2} infant pairs x the elements of that "
                 f"{'region' if FEATURE_DEFS[fid]['type'] == 'node' else 'region pair'} (n elements below each label)",
                 fontsize=14, fontweight="bold", y=0.99)
    return fig


WEIGHTED_NODE_FIDS = [f for f, d in FEATURE_DEFS.items() if d["network"] == "weighted" and d["type"] == "node"]
# The same four measures computed on edges (f03, f05, f07, f09): "..._node_<measure>" -> "..._edge_<measure>"
WEIGHTED_EDGE_FIDS = [f for f, d in FEATURE_DEFS.items()
                      if d["name"] in {FEATURE_DEFS[n]["name"].replace("_node_", "_edge_") for n in WEIGHTED_NODE_FIDS}]


def threshold_grid_figure(fids, thresholds):
    """
    Threshold bars for one threshold pair (left and right bar) of several features of one element type side by side,
    one column per feature, with ses-1 on top and ses-2 below. Text is larger than on the per-feature pages.
    """
    ftype = FEATURE_DEFS[fids[0]]["type"]
    vertical = ftype == "edge"  # 15 region pairs per panel: narrow bars, so numbers and labels are rotated
    panel_w, panel_h = (10.0, 6.8) if vertical else (10.0, 5.2)
    label_min = 12.0 if vertical else 6.0
    categories = HEMI_CATEGORIES[ftype]
    fig = plt.figure(figsize=(panel_w * len(fids), panel_h * 2 + 3.5), layout="constrained")
    ses_figs = fig.subfigures(3, 1, height_ratios=(1, 1, 0.18), hspace=0.03)  # last row: the legend, clear of the axes
    for ses_fig, ses in zip(ses_figs, SESSIONS):
        ses_fig.suptitle(ses, fontsize=30, fontweight="bold")
        for col, (ax, fid) in enumerate(zip(ses_fig.subplots(1, len(fids)), fids)):
            J = np.load(OUTPUT_TENSORS_DIR / f"J_pair_{fid}_{ses}.npy")
            n_undefined = int(np.isnan(J).sum())
            threshold_panel(ax, fid, J, thresholds, fontsize=16, label_min=label_min, vertical=vertical)
            title = feature_title(fid)
            if n_undefined:
                title += f"\n{n_undefined} of {J.size} values 0/0, left out"
            ax.set_title(title, fontsize=20, fontweight="bold")
            if col:
                ax.set_ylabel("")
    ses_figs[2].legend(handles=[patches.Patch(color=plt.get_cmap("Set2")(i), label=c) for i, c in enumerate(categories)],
                       loc="center", ncol=len(categories), fontsize=20, title_fontsize=20,
                       title="Hemispheric category (L left, R right, M midline); segments add up to the bar total written on top")
    noun = "region" if ftype == "node" else "region pair"
    fig.suptitle(f"Threshold Bars, Weighted {ftype.capitalize()} Features: Share of Element Similarities "
                 f"J > {thresholds[0]:.2f} (left bar) and J > {thresholds[1]:.2f} (right bar)\n"
                 f"Each bar: all {len(SUBJECT_IDS) * (len(SUBJECT_IDS) - 1) // 2} infant pairs x the elements of that {noun} "
                 f"(n elements in each label); segments of {label_min:.0f}% or less are not labelled",
                 fontsize=30, fontweight="bold")
    return fig


def generate_threshold_figure():
    """
    Threshold bars (as in the legacy notebooks): f01 as the main figure, all 26 features in a multi-page PDF, and the
    weighted node features and their edge versions side by side (one page per threshold pair, also as PNGs).
    """
    print("[9/9] Generating the Threshold Bars: Share of J Above 0.30, 0.50, 0.75 and 0.90...")
    save_figure(threshold_figure("f01"), "Figure_7_Supplementary_Threshold_Bars")
    with PdfPages(FIGURES_DIR / "Figure_7_Supplementary_Threshold_Bars_All_26.pdf") as pdf:
        for fid in FEATURE_DEFS:
            fig = threshold_figure(fid)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    generate_threshold_grid_figure()


def generate_threshold_grid_figure():
    """
    Weighted node features (pages 1-2) and the same measures on edges (pages 3-4), one column per feature:
    J > 0.75 and 0.90 on one page, J > 0.50 and 0.30 on the next, each with ses-1 above ses-2.
    """
    with PdfPages(FIGURES_DIR / "Figure_7_Supplementary_Threshold_Bars_Weighted_Grid.pdf") as pdf:
        for kind, fids in (("Nodes", WEIGHTED_NODE_FIDS), ("Edges", WEIGHTED_EDGE_FIDS)):
            for thresholds in THRESHOLD_PAIRS:
                fig = threshold_grid_figure(fids, thresholds)
                pdf.savefig(fig, bbox_inches="tight")
                tag = "_".join(f"J{round(t * 100):03d}" for t in thresholds)  # e.g. J075_J090
                fig.savefig(FIGURES_DIR / f"Figure_7_Supplementary_Threshold_Bars_Weighted_{kind}_{tag}.png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)


def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("PHASE 9: COMPILING & GENERATING PUBLICATION FIGURES (1 THROUGH 7, THE SIBLING FIGURE AND THE THRESHOLD BARS)")
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
    dev_mod = importlib.import_module("analysis_pipeline.06_developmental_and_twins")
    _, ages_days = dev_mod.load_exact_subject_ages()

    short_subjs = [s.replace("sub-NORB00", "") for s in SUBJECT_IDS]

    # Generate all figures
    generate_figure_1(i_res, short_subjs)
    generate_figure_2(u_res, w_res, i_res)
    generate_figure_3(u_res, i_res, tax_res)
    generate_figure_4(i_res)
    generate_figure_5(dev_res, ages_days)
    generate_figure_6(var_res)
    generate_figure_7(ages_days)
    generate_sibling_figure(twin_res, ages_days)
    generate_threshold_figure()

    print("=" * 80)
    print("ALL PUBLICATION FIGURES SUCCESSFULLY GENERATED!")
    print(f"Output files stored in: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_figures()
