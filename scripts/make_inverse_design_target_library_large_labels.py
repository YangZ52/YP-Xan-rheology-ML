from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MPLCONFIGDIR = Path("/private/tmp/rheology_ml_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

ROOT = Path("/Users/zhiy/Documents/Rheology ML")
SOURCE_DIR = ROOT / "outputs" / "bayesian_inverse_design_no_new_experiments_20260531_145221"
DATA_DIR = ROOT / "outputs" / "ml_ready_xanthan_positive_20260529"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"inverse_design_target_library_large_labels_{RUN_ID}"
DPI = 600

SCENARIOS = [
    {
        "id": "viscosity_only_eta50_200",
        "title": r"$\eta_{50}$ = 200 mPa s",
        "candidate_file": "viscosity_only_eta50_200_top_candidates.csv",
    },
    {
        "id": "viscosity_only_eta50_500",
        "title": r"$\eta_{50}$ = 500 mPa s",
        "candidate_file": "viscosity_only_eta50_500_top_candidates.csv",
    },
    {
        "id": "elasticity_only_Gp1Hz_100",
        "title": r"$G'$ 1 Hz = 100 Pa",
        "candidate_file": "elasticity_only_Gp1Hz_100_top_candidates.csv",
    },
    {
        "id": "elasticity_only_Gp1Hz_500",
        "title": r"$G'$ 1 Hz = 500 Pa",
        "candidate_file": "elasticity_only_Gp1Hz_500_top_candidates.csv",
    },
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 1.0,
            "savefig.bbox": "tight",
        }
    )


def load_measured_points() -> pd.DataFrame:
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    return rep.groupby(["split", "formulation_std"], as_index=False).mean(numeric_only=True)


def overlay_points(ax: plt.Axes, measured: pd.DataFrame) -> None:
    train = measured[measured["split"] == "train"]
    external = measured[measured["split"] == "predict"]
    ax.scatter(
        train["xanthan_pct"],
        train["yp_pct"],
        s=42,
        marker="o",
        facecolors="#6E6E6E",
        edgecolors="white",
        linewidths=0.75,
        alpha=0.92,
        zorder=5,
    )
    ax.scatter(
        external["xanthan_pct"],
        external["yp_pct"],
        s=96,
        marker="D",
        facecolors="#2E7CB8",
        edgecolors="white",
        linewidths=1.0,
        alpha=0.98,
        zorder=6,
    )


def add_candidates(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    if candidates.empty:
        return
    top = candidates.head(12)
    ax.scatter(
        top["xanthan_pct"],
        top["yp_pct"],
        s=128,
        facecolors="#F5D1A3",
        edgecolors="white",
        linewidths=1.1,
        alpha=0.78,
        zorder=7,
    )
    ax.scatter(
        top["xanthan_pct"],
        top["yp_pct"],
        s=128,
        facecolors="none",
        edgecolors="#2B2B2B",
        linewidths=0.55,
        alpha=0.70,
        zorder=8,
    )
    best = candidates.iloc[0]
    ax.scatter(
        [best["xanthan_pct"]],
        [best["yp_pct"]],
        marker="*",
        s=360,
        facecolors="#D62728",
        edgecolors="white",
        linewidths=1.2,
        zorder=9,
    )


def format_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.225, 1.025)
    ax.set_ylim(-1.0, 31.0)
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Xanthan gum (wt%)", fontsize=18, fontweight="bold", labelpad=8)
    ax.set_ylabel("Yeast protein (wt%)", fontsize=18, fontweight="bold", labelpad=8)
    ax.tick_params(axis="both", labelsize=14, width=1.0, length=4)
    ax.grid(color="white", linewidth=0.48, alpha=0.24)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_panel(
    ax: plt.Axes,
    grid: pd.DataFrame,
    measured: pd.DataFrame,
    scenario: dict,
    letter: str,
) -> None:
    score_col = f"scenario_{scenario['id']}_score"
    pivot = grid.pivot_table(index="yp_pct", columns="xanthan_pct", values=score_col, aggfunc="mean").sort_index()
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)

    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="magma",
    )
    finite = z[np.isfinite(z)]
    if finite.size:
        level = float(np.nanmax(finite) * 0.72)
        if np.nanmin(finite) < level < np.nanmax(finite):
            ax.contour(x, y, z, levels=[level], colors="white", linewidths=2.4, alpha=0.96, zorder=4)
            ax.contour(x, y, z, levels=[level], colors="#151515", linewidths=0.75, alpha=0.92, zorder=4)

    overlay_points(ax, measured)
    candidates = pd.read_csv(SOURCE_DIR / scenario["candidate_file"])
    add_candidates(ax, candidates)
    format_axes(ax)
    ax.set_title(scenario["title"], fontsize=20, fontweight="bold", pad=12)
    ax.text(
        -0.16,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=27,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )
    return im


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    grid = pd.read_csv(SOURCE_DIR / "bayesian_inverse_design_grid.csv")
    measured = load_measured_points()

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 12.6), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.94, bottom=0.075, top=0.945, wspace=0.30, hspace=0.34)

    letters = ["A", "B", "C", "D"]
    for ax, scenario, letter in zip(axes.ravel(), SCENARIOS, letters):
        im = plot_panel(ax, grid, measured, scenario, letter)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.032)
        cbar.set_label("Scenario score", fontsize=16, fontweight="bold", labelpad=9)
        cbar.ax.tick_params(labelsize=13, width=0.9, length=3)

    out_base = OUT_DIR / "Figure_inverse_design_target_library_large_labels_600dpi"
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI)
    fig.savefig(out_base.with_suffix(".tiff"), dpi=DPI)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    summary = {
        "source_dir": str(SOURCE_DIR),
        "outputs": [str(out_base.with_suffix(ext)) for ext in [".png", ".tiff", ".pdf"]],
        "dpi": DPI,
        "figure_size_inches": [14.8, 12.6],
        "changes": [
            "larger title, axis, tick, panel-letter, and colorbar fonts",
            "axis labels changed to Xanthan gum (wt%) and Yeast protein (wt%)",
            "increased horizontal and vertical spacing between panels",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
