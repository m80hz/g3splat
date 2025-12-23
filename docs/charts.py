import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "static/charts"  
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Data (LaTeX tables)
# -------------------------
POSE_PNP_RANSAC = {
  "CoPoNeRF":   {"RE10K": [0.161, 0.362, 0.575], "ScanNet": [np.nan]*3,            "ACID": [0.078, 0.216, 0.398]},
  "DUSt3R":    {"RE10K": [0.301, 0.495, 0.657], "ScanNet": [0.085, 0.210, 0.398], "ACID": [0.166, 0.304, 0.437]},
  "MASt3R":    {"RE10K": [0.372, 0.561, 0.709], "ScanNet": [0.083, 0.200, 0.381], "ACID": [0.234, 0.396, 0.541]},
  "RoMa":      {"RE10K": [0.546, 0.698, 0.797], "ScanNet": [0.168, 0.361, 0.575], "ACID": [0.463, 0.588, 0.689]},
  "Splatt3R":   {"RE10K": [0.158, 0.325, 0.504], "ScanNet": [0.011, 0.042, 0.119], "ACID": [0.044, 0.121, 0.260]},
  "SelfSplat":   {"RE10K": [0.030, 0.083, 0.180], "ScanNet": [0.030, 0.098, 0.254], "ACID": [0.064, 0.153, 0.283]},
  "NoPoSplat":  {"RE10K": [0.572, 0.728, 0.833], "ScanNet": [0.078, 0.198, 0.394], "ACID": [0.337, 0.497, 0.646]},
  "G3Splat":     {"RE10K": [0.629, 0.770, 0.858], "ScanNet": [0.124, 0.282, 0.493], "ACID": [0.404, 0.560, 0.689]},
}

POSE_REFINEMENT = {
  "SelfSplat":  {"RE10K": [0.031, 0.086, 0.182], "ScanNet": [0.033, 0.112, 0.274], "ACID": [0.069, 0.156, 0.285]},
  "NoPoSplat":  {"RE10K": [0.672, 0.791, 0.868], "ScanNet": [0.109, 0.256, 0.463], "ACID": [0.456, 0.593, 0.705]},
  "G3Splat":    {"RE10K": [0.684, 0.801, 0.875], "ScanNet": [0.148, 0.326, 0.540], "ACID": [0.466, 0.598, 0.713]},
}

SCANNET_DEPTH = {
  "pixelSplat": {"AbsRel": (0.299, 0.288), "d110": (0.552, 0.553), "d125": (0.818, 0.820)},
  "MVSplat":    {"AbsRel": (0.189, 0.132), "d110": (0.412, 0.641), "d125": (0.745, 0.891)},
  "FreeSplat":  {"AbsRel": (0.126, 0.124), "d110": (0.556, 0.556), "d125": (0.831, 0.833)},
  "DepthSplat": {"AbsRel": (0.135, 0.105), "d110": (0.578, 0.722), "d125": (0.864, 0.914)},
  "Splatt3R":   {"AbsRel": (0.148, np.nan), "d110": (0.546, np.nan), "d125": (0.806, np.nan)},
  "SelfSplat":  {"AbsRel": (0.160, 0.155), "d110": (0.502, 0.460), "d125": (0.810, 0.801)},
  "NoPoSplat":  {"AbsRel": (0.131, 0.121), "d110": (0.554, 0.662), "d125": (0.851, 0.869)},
  "G3Splat":    {"AbsRel": (0.090, 0.082), "d110": (0.713, 0.740), "d125": (0.916, 0.928)},
}

MESH_SCANNET = {
    "DUSt3R": {"No Prior": (0.266, 0.514, 0.390), "G3Splat": (0.255, 0.498, 0.377)},
    "VGGT":   {"No Prior": (0.150, 0.362, 0.256), "G3Splat": (0.139, 0.349, 0.244)},
}

# -------------------------
# Helpers
# -------------------------
def save_svg(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}", format="svg", bbox_inches="tight")
    plt.close(fig)

def heatmap_pose(pose_dict, title, filename, highlight="G3Splat"):
    datasets = ["RE10K", "ScanNet", "ACID"]
    thresholds = ["AUC@5°", "AUC@10°", "AUC@20°"]

    methods = list(pose_dict.keys())
    cols = []
    col_labels = []
    for ds in datasets:
        for th_i, th in enumerate(thresholds):
            col_labels.append(f"{ds}\n{th}")
            cols.append([pose_dict[m][ds][th_i] for m in methods])

    M = np.array(cols).T  # (methods, cols)
    mask = np.isnan(M)

    fig = plt.figure(figsize=(12, 0.45*len(methods) + 1.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.ma.array(M, mask=mask), aspect="auto")

    ax.set_title(title, pad=12)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    if highlight:
        for tick in ax.get_yticklabels():
            if tick.get_text() == highlight:
                tick.set_fontweight("bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                is_hi_row = highlight and (methods[i] == highlight)
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    # fontweight=("bold" if is_hi_row else "normal"),
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("AUC (higher is better)")

    ax.set_xlabel("Dataset / threshold")
    ax.set_ylabel("Pose-Free Method")
    ax.grid(False)

    save_svg(fig, filename)

def depth_scatter(depth_dict, which="novel", filename="scannet_depth_novel_scatter.svg", highlight="G3Splat"):
    idx = 0 if which == "novel" else 1

    methods = list(depth_dict.keys())
    x = np.array([depth_dict[m]["AbsRel"][idx] for m in methods], dtype=float)
    y = np.array([depth_dict[m]["d125"][idx] for m in methods], dtype=float)

    valid = ~np.isnan(x) & ~np.isnan(y)
    methods_v = [m for m, ok in zip(methods, valid) if ok]
    x, y = x[valid], y[valid]

    # Split into "highlight" vs others
    other_idx = [i for i, m in enumerate(methods_v) if m != highlight]
    hi_idx = [i for i, m in enumerate(methods_v) if m == highlight]

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)

    # Others (default styling)
    ax.scatter(x[other_idx], y[other_idx])

    # Highlight (different color + slightly larger)
    if hi_idx:
        ax.scatter(x[hi_idx], y[hi_idx], s=90, c="crimson", zorder=3)

    # Labels
    for m, xi, yi in zip(methods_v, x, y):
        if m == highlight:
            ax.text(xi, yi, f"  {m}", fontsize=11, fontweight="bold", va="center")
        else:
            ax.text(xi, yi, f"  {m}", fontsize=9, va="center")

    ax.set_title(f"ScanNet depth ({which} view): AbsRel vs δ1<1.25")
    ax.set_xlabel("Abs Rel ↓")
    ax.set_ylabel("δ1 < 1.25 ↑")
    ax.invert_xaxis()  # so "better" is up-right
    ax.grid(True, alpha=0.3)

    save_svg(fig, filename)


def dumbbell_mesh_metric(
    mesh_dict,
    *,
    metric_name: str,
    metric_idx: int,
    xlabel: str,
    filename: str,
):
    """Dumbbell chart for a single mesh metric.
    - No Prior: circle marker
    - G3Splat: diamond marker
    - Bold numeric label for G3Splat
    - Invert x-axis (lower is better)
    """

    backbones = list(mesh_dict.keys())
    y = list(range(len(backbones)))  # 0..n-1

    no_prior = [float(mesh_dict[b]["No Prior"][metric_idx]) for b in backbones]
    g3splat = [float(mesh_dict[b]["G3Splat"][metric_idx]) for b in backbones]

    fig = plt.figure(figsize=(7.2, 3.2))
    ax = fig.add_subplot(111)

    for i, backbone in enumerate(backbones):
        a = no_prior[i]
        b = g3splat[i]
        if np.isnan(a) or np.isnan(b):
            continue

        ax.plot([a, b], [i, i], linewidth=2.0)
        ax.scatter([a], [i], s=70, marker="o")
        ax.scatter([b], [i], s=90, marker="D")

        # Numeric annotations
        ax.annotate(
            f"{a:.3f}",
            (a, i),
            xytext=(-6, 2),
            textcoords="offset points",
            ha="right",
            fontsize=9,
        )
        ax.annotate(
            f"{b:.3f}",
            (b, i),
            xytext=(6, 2),
            textcoords="offset points",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(backbones)
    ax.set_title(f"ScanNet mesh reconstruction (TSDF-Fusion) — {metric_name}")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.6)

    # Because lower is better, invert x-axis so “better” goes to the right visually.
    xmin = min(min(no_prior), min(g3splat)) * 0.95
    xmax = max(max(no_prior), max(g3splat)) * 1.05
    ax.set_xlim(xmax, xmin)

    ax.text(
        0.99,
        0.05,
        "○ No Prior    ◆ G3Splat",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()
    save_svg(fig, filename)

# -------------------------
# Generate charts
# -------------------------
heatmap_pose(POSE_PNP_RANSAC, "Relative Pose AUC (PnP+RANSAC)", "pose_auc_pnp_ransac_heatmap.svg", highlight="G3Splat")
heatmap_pose(POSE_REFINEMENT, "Relative Pose AUC (with refinement)", "pose_auc_refinement_heatmap.svg", highlight="G3Splat")

depth_scatter(SCANNET_DEPTH, which="novel",  filename="scannet_depth_novel_scatter.svg",  highlight="G3Splat")
depth_scatter(SCANNET_DEPTH, which="source", filename="scannet_depth_source_scatter.svg", highlight="G3Splat")

dumbbell_mesh_metric(
    MESH_SCANNET,
    metric_name="Accuracy",
    metric_idx=0,
    xlabel="Acc ↓ (lower is better)",
    filename="mesh_scannet_accuracy_dumbbell.svg",
)
dumbbell_mesh_metric(
    MESH_SCANNET,
    metric_name="Completeness",
    metric_idx=1,
    xlabel="Comp ↓ (lower is better)",
    filename="mesh_scannet_completeness_dumbbell.svg",
)
dumbbell_mesh_metric(
    MESH_SCANNET,
    metric_name="Chamfer",
    metric_idx=2,
    xlabel="Chamfer ↓ (lower is better)",
    filename="mesh_scannet_chamfer_dumbbell.svg",
)

print("Done. SVGs written to:", OUT_DIR)
