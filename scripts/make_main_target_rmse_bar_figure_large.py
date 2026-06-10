from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

MPLCONFIGDIR = Path("/private/tmp/rheology_ml_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/zhiy/Documents/Rheology ML")
METRICS_PATH = ROOT / "outputs" / "reviewer_performance_figures_20260530_205058" / "main_target_internal_external_metrics.csv"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"main_target_rmse_bar_large_labels_{RUN_ID}"
DPI = 600

INTERNAL_LABEL = "Internal 5-fold CV"
EXTERNAL_LABEL = "External validation"
VALIDATION_COLORS = {INTERNAL_LABEL: "#6E8FB2", EXTERNAL_LABEL: "#C58B57"}

MODEL_ORDER = [
    "GPR_Matern_ARD",
    "KernelRidge_RBF",
    "SVR_RBF",
    "Ridge",
    "XGBoost",
    "ExtraTrees",
    "RandomForest",
    "GradientBoosting",
]

MODEL_LABELS = {
    "GPR_Matern_ARD": "GPR-Matern-ARD",
    "KernelRidge_RBF": "KRR-RBF",
    "SVR_RBF": "SVR-RBF",
    "Ridge": "Ridge",
    "XGBoost": "XGBoost",
    "ExtraTrees": "Extra Trees",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
}

TARGETS = [
    ("Gp1Hz", r"$G'$ at 1 Hz"),
    ("eta50", r"$\eta_{50}$"),
    ("break_stress", "Breaking stress"),
    ("break_strain", "Breaking strain"),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.1,
            "axes.labelsize": 24,
            "axes.titlesize": 23,
            "xtick.labelsize": 17,
            "ytick.labelsize": 18,
            "legend.fontsize": 22,
            "savefig.bbox": "tight",
        }
    )


def fmt_rmse(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def draw_panel(ax: plt.Axes, metrics: pd.DataFrame, target_id: str, title: str, letter: str) -> None:
    models = [m for m in MODEL_ORDER if m in set(metrics["model"])]
    x = np.arange(len(models))
    width = 0.38
    panel = metrics[metrics["target_id"] == target_id].copy()
    values_by_validation = {}

    for offset, validation in [(-width / 2, INTERNAL_LABEL), (width / 2, EXTERNAL_LABEL)]:
        values = []
        for model in models:
            sub = panel[(panel["validation"] == validation) & (panel["model"] == model)]
            values.append(float(sub["rmse"].iloc[0]) if len(sub) else np.nan)
        values_by_validation[validation] = values
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=VALIDATION_COLORS[validation],
            edgecolor="white",
            linewidth=0.9,
            label=validation,
        )
        y_top = max([v for vals in values_by_validation.values() for v in vals if np.isfinite(v)] or [1])
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + y_top * 0.025,
                fmt_rmse(value),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=15,
                clip_on=False,
            )

    finite_vals = [v for vals in values_by_validation.values() for v in vals if np.isfinite(v)]
    ax.set_ylim(0, max(finite_vals) * 1.25 if finite_vals else 1)
    ax.set_ylabel("RMSE", fontsize=24, fontweight="bold", labelpad=12)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=38, ha="right", fontsize=16.5)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.18, 1.11, letter, transform=ax.transAxes, fontsize=30, fontweight="bold", ha="left", va="top", clip_on=False)


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    metrics = pd.read_csv(METRICS_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(17, 14.5), constrained_layout=True)
    for ax, (target_id, title), letter in zip(axes.ravel(), TARGETS, ["A", "B", "C", "D"]):
        draw_panel(ax, metrics, target_id, title, letter)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.055),
        prop={"size": 23, "weight": "bold"},
        handlelength=1.6,
        columnspacing=2.0,
    )

    out_base = OUT_DIR / "Figure_main_targets_RMSE_internal_external_large_labels_600dpi"
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI)
    fig.savefig(out_base.with_suffix(".tiff"), dpi=DPI)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    summary = {
        "source_metrics": str(METRICS_PATH),
        "outputs": [str(out_base.with_suffix(ext)) for ext in [".png", ".tiff", ".pdf"]],
        "dpi": DPI,
        "figure_size_inches": [17, 14.5],
        "changes": ["larger font sizes", "removed black best-bar outlines", "vertical RMSE value labels"],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
