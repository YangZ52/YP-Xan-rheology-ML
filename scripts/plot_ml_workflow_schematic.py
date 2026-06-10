from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


ROOT = Path("/Users/zhiy/Documents/Rheology ML")
ONEDRIVE_ROOT = Path("/Users/zhiy/Library/CloudStorage/OneDrive-Personal/GPR new")
RUN_ID = os.environ.get("ML_WORKFLOW_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"ml_workflow_schematic_{RUN_ID}"
DPI = 450


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def box(ax, xy, w, h, title, body, fc, ec="#27313A", title_color="#101820"):
    x, y = xy
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.013,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=ec,
        facecolor=fc,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=8.4,
        color="#24303A",
        linespacing=1.18,
    )
    return patch


def label(ax, xy, text, color="#3B4652", size=8.8, weight="normal"):
    ax.text(*xy, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)


def arrow(ax, start, end, color="#40505E", rad=0.0, lw=1.65, ms=15, z=2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=z,
        )
    )


def pill(ax, xy, w, h, text, fc, ec, color="#101820", size=8.4):
    x, y = xy
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight="bold", zorder=5)


def mini_grid(ax, origin, w, h):
    x0, y0 = origin
    for i, yp in enumerate([0.08, 0.25, 0.43, 0.60, 0.78]):
        for j, xg in enumerate([0.18, 0.42, 0.68]):
            color = ["#5AA9A4", "#F2B84B", "#D76F5B"][(i + j) % 3]
            ax.add_patch(Circle((x0 + w * yp, y0 + h * xg), 0.0055, facecolor=color, edgecolor="white", linewidth=0.55, zorder=5))


def mini_curve(ax, origin, w, h, color="#427AA1"):
    x0, y0 = origin
    verts = []
    codes = []
    for i in range(36):
        t = i / 35
        x = x0 + w * t
        y = y0 + h * (0.76 - 0.52 * t + 0.06 * (1 - t) * t)
        verts.append((x, y))
        codes.append(MplPath.MOVETO if i == 0 else MplPath.LINETO)
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none", edgecolor=color, linewidth=1.8, zorder=5))
    ax.plot([x0, x0, x0 + w], [y0, y0 + h, y0], color="#75808A", lw=0.8, zorder=5)


def mini_heatmap(ax, origin, w, h):
    x0, y0 = origin
    colors = ["#F7F3D4", "#DDEEC3", "#9FD0B2", "#5FA8A3", "#3B6E8F"]
    for i in range(6):
        for j in range(4):
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + w * i / 6, y0 + h * j / 4),
                    w / 6,
                    h / 4,
                    boxstyle="square,pad=0",
                    facecolor=colors[(i + j) % len(colors)],
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=5,
                )
            )


def mini_target(ax, center, r):
    cx, cy = center
    for frac, color in [(1.0, "#DDE9F6"), (0.66, "#FFFFFF"), (0.33, "#D65F5F")]:
        ax.add_patch(Circle((cx, cy), r * frac, facecolor=color, edgecolor="#5A6470", linewidth=0.65, zorder=5))
    ax.add_patch(Circle((cx + r * 0.16, cy + r * 0.10), r * 0.085, facecolor="#1F2933", edgecolor="none", zorder=6))


def add_panel_title(ax, xy, text, color):
    x, y = xy
    ax.plot([x - 0.065, x + 0.065], [y - 0.022, y - 0.022], color=color, lw=2.1, solid_capstyle="round")
    ax.text(x, y, text, ha="center", va="center", fontsize=9.7, fontweight="bold", color=color)


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    fig, ax = plt.subplots(figsize=(14.2, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    colors = {
        "input": "#E8F2F1",
        "data": "#F6EADB",
        "features": "#EAF2DD",
        "model": "#ECE7F5",
        "prediction": "#E4EEF8",
        "inverse": "#FFF1C9",
        "validation": "#F7E4DF",
        "output": "#F4F6F7",
    }

    ax.text(
        0.045,
        0.948,
        "Machine-learning workflow for rheology-guided prediction and inverse formulation design",
        fontsize=15.6,
        fontweight="bold",
        ha="left",
        color="#101820",
    )
    ax.text(
        0.045,
        0.905,
        "Composition-property measurements are converted into scalar rheology descriptors, modeled with uncertainty-aware small-data ML, and searched to identify candidate formulations.",
        fontsize=9.4,
        ha="left",
        color="#46525D",
    )

    y_top = 0.635
    w = 0.155
    h = 0.185
    xs = [0.055, 0.265, 0.475, 0.685]

    box(ax, (xs[0], y_top), w, h, "Formulation space", "yeast protein (%)\nxanthan gum (%)\ncomposition grid", colors["input"])

    box(ax, (xs[1], y_top), w, h, "Rheology data", "flow curves\nfrequency sweeps\nstrain sweeps", colors["data"])

    box(
        ax,
        (xs[2], y_top),
        w,
        h,
        "Target extraction",
        r"$\eta_{50}$, $G'$ at 1 Hz"
        + "\n"
        + r"$\sigma_\mathrm{break}$, $\gamma_\mathrm{break}$"
        + "\nstandardized targets",
        colors["features"],
    )

    box(
        ax,
        (xs[3], y_top),
        w,
        h,
        "Surrogate models",
        "GPR-Matern-ARD\nKRR, SVR, trees\nbalanced CV + external test",
        colors["model"],
    )

    for i in range(3):
        arrow(ax, (xs[i] + w + 0.005, y_top + h * 0.52), (xs[i + 1] - 0.010, y_top + h * 0.52))

    add_panel_title(ax, (0.305, 0.522), "Forward prediction", "#2F6F9F")
    add_panel_title(ax, (0.690, 0.522), "Inverse design", "#9B6A00")

    y_mid = 0.300
    pred1 = box(
        ax,
        (0.115, y_mid),
        0.175,
        0.170,
        "Property prediction",
        "predict rheology for\nnew compositions\nwith uncertainty",
        colors["prediction"],
        ec="#2F6F9F",
    )

    pred2 = box(
        ax,
        (0.350, y_mid),
        0.175,
        0.170,
        "Model checking",
        "parity plots\nresidual locations\nlength-scale relevance",
        colors["validation"],
        ec="#A65045",
    )

    inv1 = box(
        ax,
        (0.585, y_mid),
        0.175,
        0.170,
        "Target windows",
        r"$\eta_{50}$ window"
        + "\n"
        + r"$G'$ window"
        + "\napplication constraints",
        colors["inverse"],
        ec="#9B6A00",
    )

    inv2 = box(
        ax,
        (0.820, y_mid),
        0.145,
        0.170,
        "Candidate ranking",
        "posterior feasibility\nknowledge support\nlow-risk shortlist",
        colors["output"],
        ec="#69737E",
    )

    arrow(ax, (xs[3] + w * 0.30, y_top), (0.205, y_mid + 0.170), color="#2F6F9F", rad=0.16)
    arrow(ax, (0.290, y_mid + 0.085), (0.350, y_mid + 0.085), color="#2F6F9F")
    arrow(ax, (xs[3] + w * 0.70, y_top), (0.672, y_mid + 0.170), color="#9B6A00", rad=-0.12)
    arrow(ax, (0.760, y_mid + 0.085), (0.820, y_mid + 0.085), color="#9B6A00")

    y_bot = 0.085
    box(
        ax,
        (0.290, y_bot),
        0.200,
        0.145,
        "Publication outputs",
        "property maps, parity plots,\ninverse-design maps,\nranked formulation table",
        colors["output"],
        ec="#66717C",
    )
    box(
        ax,
        (0.585, y_bot),
        0.205,
        0.145,
        "Experimental validation",
        "prepare top formulations\nmeasure target rheology\nupdate training set",
        "#E6F0FF",
        ec="#4979B8",
    )

    arrow(ax, (0.438, y_mid), (0.400, y_bot + 0.145), color="#65717C", rad=0.08)
    arrow(ax, (0.892, y_mid), (0.735, y_bot + 0.145), color="#65717C", rad=-0.12)
    arrow(ax, (0.585, y_bot + 0.072), (0.490, y_bot + 0.072), color="#65717C")

    arrow(ax, (0.690, y_bot + 0.145), (0.755, y_top - 0.010), color="#4979B8", rad=-0.42, lw=1.45, ms=13)
    label(ax, (0.835, 0.205), "active-learning loop", color="#4979B8", size=8.1, weight="bold")

    pill(ax, (0.065, 0.545), 0.095, 0.042, "inputs", "#FFFFFF", "#89A9A6", color="#2E5E5A")
    pill(ax, (0.276, 0.545), 0.095, 0.042, "measurements", "#FFFFFF", "#CBA06E", color="#7A5528")
    pill(ax, (0.486, 0.545), 0.095, 0.042, "descriptors", "#FFFFFF", "#95B879", color="#4C6E31")
    pill(ax, (0.696, 0.545), 0.095, 0.042, "ML model", "#FFFFFF", "#A396C5", color="#5A4B83")

    metric_x = 0.047
    metric_y = 0.186
    for i, (txt, col) in enumerate(
        [
            (r"$\eta_{50}$", "#B45A4D"),
            (r"$G'$ 1 Hz", "#3E75A5"),
            (r"$\sigma_\mathrm{break}$", "#7A4E9D"),
            (r"$\gamma_\mathrm{break}$", "#2F875A"),
        ]
    ):
        pill(ax, (metric_x, metric_y - i * 0.052), 0.118, 0.038, txt, "#FFFFFF", col, color=col, size=8.8)

    ax.text(0.047, 0.245, "Modeled targets", ha="left", va="center", fontsize=8.4, fontweight="bold", color="#46525D")

    for letter, xy in zip(["a", "b", "c"], [(0.040, 0.840), (0.100, 0.485), (0.570, 0.485)]):
        ax.text(*xy, letter, ha="center", va="center", fontsize=12.0, fontweight="bold", color="white", bbox=dict(boxstyle="circle,pad=0.18", fc="#25313B", ec="none"))

    fig.savefig(OUT_DIR / "Figure_ML_workflow_inverse_design_schematic.png", dpi=DPI)
    fig.savefig(OUT_DIR / "Figure_ML_workflow_inverse_design_schematic.pdf")
    fig.savefig(OUT_DIR / "Figure_ML_workflow_inverse_design_schematic.svg")
    fig.savefig(OUT_DIR / "Figure_ML_workflow_inverse_design_schematic.tiff", dpi=DPI)
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)

    if os.environ.get("COPY_TO_ONEDRIVE", "0") == "1":
        onedrive_dir = ONEDRIVE_ROOT / "time_lapse" / OUT_DIR.name
        if onedrive_dir.exists():
            shutil.rmtree(onedrive_dir)
        shutil.copytree(OUT_DIR, onedrive_dir)
        print(onedrive_dir)

    print(OUT_DIR)


if __name__ == "__main__":
    main()
