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
RUN_ID = os.environ.get("RHEOLOGY_RANGE_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"rheology_data_range_examples_{RUN_ID}"
DPI = 400

SELECTED = [
    {
        "formulation": "10%YP+0.25%XG",
        "label": "low: 10 wt% YP + 0.25 wt% XG",
        "color": "#2F80C1",
    },
    {
        "formulation": "17.5%YP+0.5%XG",
        "label": "medium: 17.5 wt% YP + 0.50 wt% XG",
        "color": "#D97706",
    },
    {
        "formulation": "30%YP+1%XG",
        "label": "high: 30 wt% YP + 1.00 wt% XG",
        "color": "#7A3E9D",
    },
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 13.2,
            "axes.titlesize": 14.4,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 13.5,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def mean_min_max(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, as_index=False)[value_col]
        .agg(mean="mean", low="min", high="max")
        .sort_values(group_col)
    )


def median_quantile(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, as_index=False)[value_col]
        .agg(median="median", low=lambda x: x.quantile(0.25), high=lambda x: x.quantile(0.75))
        .sort_values(group_col)
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.09, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")


def plot_flow(ax: plt.Axes, viscosity: pd.DataFrame) -> None:
    for item in SELECTED:
        d = viscosity[viscosity["formulation_std"] == item["formulation"]].copy()
        stats = mean_min_max(d, "shear_rate", "viscosity")
        ax.plot(stats["shear_rate"], stats["mean"], color=item["color"], linewidth=2.2, label=item["label"])
        ax.fill_between(stats["shear_rate"], stats["low"], stats["high"], color=item["color"], alpha=0.16, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Shear rate (s$^{-1}$)", fontweight="bold")
    ax.set_ylabel("Apparent viscosity (mPa s)", fontweight="bold")
    ax.set_title("Flow curves", fontweight="bold")
    panel_label(ax, "A")


def plot_frequency(ax: plt.Axes, frequency: pd.DataFrame) -> None:
    frequency = frequency[(frequency["frequency_Hz"] >= 0.01) & (frequency["frequency_Hz"] <= 10)].copy()
    for item in SELECTED:
        d = frequency[frequency["formulation_std"] == item["formulation"]].copy()
        gp = median_quantile(d, "frequency_Hz", "Gp_Pa")
        gpp = median_quantile(d, "frequency_Hz", "Gpp_Pa")
        ax.plot(gp["frequency_Hz"], gp["median"], color=item["color"], linewidth=2.2, label=item["label"])
        ax.plot(gpp["frequency_Hz"], gpp["median"], color=item["color"], linewidth=1.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax.set_ylabel(r"$G'$ solid, $G''$ dashed (Pa)", fontweight="bold")
    ax.set_title("Frequency sweeps", fontweight="bold")
    panel_label(ax, "B")


def plot_strain(ax: plt.Axes, strain: pd.DataFrame) -> None:
    for item in SELECTED:
        d = strain[strain["formulation_std"] == item["formulation"]].copy()
        gp = median_quantile(d, "strain_pct", "Gp_Pa")
        gpp = median_quantile(d, "strain_pct", "Gpp_Pa")
        ax.plot(gp["strain_pct"], gp["median"], color=item["color"], linewidth=2.2, label=item["label"])
        ax.plot(gpp["strain_pct"], gpp["median"], color=item["color"], linewidth=1.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Strain (%)", fontweight="bold")
    ax.set_ylabel(r"$G'$ solid, $G''$ dashed (Pa)", fontweight="bold")
    ax.set_title("Strain sweeps", fontweight="bold")
    panel_label(ax, "C")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    viscosity = pd.read_csv(DATA_DIR / "viscosity_long.csv")
    frequency = pd.read_csv(DATA_DIR / "frequency_long.csv")
    strain = pd.read_csv(DATA_DIR / "strain_long.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), constrained_layout=True)
    plot_flow(axes[0], viscosity)
    plot_frequency(axes[1], frequency)
    plot_strain(axes[2], strain)

    for ax in axes:
        ax.grid(color="#D9D9D9", linewidth=0.55, alpha=0.55, which="both")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        frameon=True,
        fontsize=13.5,
        handlelength=2.5,
        columnspacing=1.4,
    )
    fig.savefig(OUT_DIR / "Figure_rheology_data_range_examples.png", dpi=DPI)
    fig.savefig(OUT_DIR / "Figure_rheology_data_range_examples.pdf")
    fig.savefig(OUT_DIR / "Figure_rheology_data_range_examples.tiff", dpi=DPI)

    summary_rows = []
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    for item in SELECTED:
        d = rep[rep["formulation_std"] == item["formulation"]]
        summary_rows.append(
            {
                "formulation_std": item["formulation"],
                "label": item["label"],
                "yp_pct": float(d["yp_pct"].mean()),
                "xanthan_pct": float(d["xanthan_pct"].mean()),
                "eta_50_mPa_s": float(d["eta_50_Pa_s"].mean()),
                "Gp_1Hz_Pa": float(d["Gp_1Hz_Pa"].mean()),
                "break_strain_pct": float(d["break_strain_pct"].mean()),
                "break_stress_Pa": float(d["break_stress_Pa"].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "rheology_data_range_selected_formulations.csv", index=False)
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)

    archive_dir = None
    if os.environ.get("COPY_TO_ARCHIVE", "0") == "1":
        archive_dir = ARCHIVE_ROOT / "time_lapse" / OUT_DIR.name
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        shutil.copytree(OUT_DIR, archive_dir)
    print(OUT_DIR)
    if archive_dir is not None:
        print(archive_dir)


if __name__ == "__main__":
    main()
