from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "ml_ready_xanthan_positive_20260529"
ARCHIVE_ROOT = Path(os.environ.get("RHEOLOGY_ARCHIVE_ROOT", ROOT / "outputs"))
RUN_ID = os.environ.get("FORMULATION_PHASE_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"formulation_phase_diagram_{RUN_ID}"
DPI = 400

def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 13.6,
            "axes.titlesize": 14.6,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.4,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.09, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")


def finish_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.20, 1.05)
    ax.set_ylim(-1.5, 31.5)
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Xanthan gum (wt%)", fontweight="bold")
    ax.set_ylabel("Yeast protein (wt%)", fontweight="bold", fontsize=14.8)
    ax.grid(color="#D9D9D9", linewidth=0.65, alpha=0.65)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_property_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    external: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    cmap: str,
    letter: str,
) -> None:
    sc = ax.scatter(
        df["xanthan_pct"],
        df["yp_pct"],
        c=df[value_col],
        s=72,
        marker="o",
        cmap=cmap,
        edgecolors="white",
        linewidths=0.75,
        zorder=3,
    )
    ax.scatter(
        external["xanthan_pct"],
        external["yp_pct"],
        s=120,
        marker="D",
        facecolors="none",
        edgecolors="#2F80C1",
        linewidths=1.5,
        label="external test",
        zorder=4,
    )
    ax.set_title(title, fontweight="bold")
    finish_axes(ax)
    panel_label(ax, letter)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.044, pad=0.045)
    cbar.set_label(cbar_label)


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    df = rep.groupby(["split", "formulation_std"], as_index=False).mean(numeric_only=True)
    df["log10_break_stress_Pa"] = np.log10(df["break_stress_Pa"])
    train = df[df["split"] == "train"].copy()
    external = df[df["split"] == "predict"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.93, bottom=0.16, wspace=0.42, hspace=0.42)
    axes = axes.ravel()

    ax = axes[0]
    ax.scatter(
        train["xanthan_pct"],
        train["yp_pct"],
        s=58,
        marker="o",
        facecolors="#4A4A4A",
        edgecolors="white",
        linewidths=0.7,
        label="training/internal",
        zorder=3,
    )
    ax.scatter(
        external["xanthan_pct"],
        external["yp_pct"],
        s=84,
        marker="D",
        facecolors="#2F80C1",
        edgecolors="white",
        linewidths=0.9,
        label="external test",
        zorder=4,
    )
    ax.set_title("Formulation space and data split", fontweight="bold")
    finish_axes(ax)
    panel_label(ax, "A")

    plot_property_panel(
        axes[1],
        df,
        external,
        "log10_eta_50",
        r"Measured viscosity range",
        r"log$_{10}$ $\eta_{50}$ (mPa s)",
        "magma",
        "B",
    )
    plot_property_panel(
        axes[2],
        df,
        external,
        "log10_Gp_1Hz",
        r"Measured $G'$ 1 Hz range",
        r"log$_{10}$ $G'$ 1 Hz (Pa)",
        "viridis",
        "C",
    )
    plot_property_panel(
        axes[3],
        df,
        external,
        "log10_break_stress_Pa",
        r"Measured breaking stress range",
        r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
        "plasma",
        "D",
    )
    plot_property_panel(
        axes[4],
        df,
        external,
        "break_strain_pct",
        r"Measured breaking strain range",
        r"$\gamma_\mathrm{break}$ (%)",
        "cividis",
        "E",
    )
    axes[5].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=True,
        fontsize=10.8,
        markerscale=1.2,
        handletextpad=0.8,
        columnspacing=1.6,
    )

    fig.savefig(OUT_DIR / "Figure_formulation_phase_diagram_2x3_rheology_landscape.png", dpi=DPI)
    fig.savefig(OUT_DIR / "Figure_formulation_phase_diagram_2x3_rheology_landscape.pdf")
    fig.savefig(OUT_DIR / "Figure_formulation_phase_diagram_2x3_rheology_landscape.tiff", dpi=DPI)

    df.to_csv(OUT_DIR / "formulation_phase_diagram_points.csv", index=False)
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)

    archive_dir = ARCHIVE_ROOT / "time_lapse" / OUT_DIR.name
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    shutil.copytree(OUT_DIR, archive_dir)
    print(OUT_DIR)
    print(archive_dir)


if __name__ == "__main__":
    main()
