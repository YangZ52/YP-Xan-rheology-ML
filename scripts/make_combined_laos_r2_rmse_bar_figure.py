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
SOURCE_RUN = ROOT / "outputs" / "publication_formulation_laos_model_suite_20260531_130806"
METRICS_PATH = SOURCE_RUN / "all_model_metrics_internal_external.csv"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"combined_laos_r2_rmse_bar_figure_{RUN_ID}"
DPI = 600

INTERNAL_LABEL = "Internal balanced 5-fold CV"
EXTERNAL_LABEL = "External validation"

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

FEATURE_SETS = [
    ("formulation", "formulation only"),
    ("formulation_Gp1Hz", r"formulation + $G'$ 1 Hz"),
    ("formulation_eta50", r"formulation + $\eta_{50}$"),
    ("formulation_Gp1Hz_eta50", r"formulation + $G'$ 1 Hz + $\eta_{50}$"),
]

TARGETS = [
    ("break_stress", r"$\sigma_\mathrm{break}$"),
    ("break_strain", r"$\gamma_\mathrm{break}$"),
]

VALIDATION_COLORS = {
    INTERNAL_LABEL: "#6E8FB2",
    EXTERNAL_LABEL: "#C58B57",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "axes.labelsize": 23,
            "axes.titlesize": 23,
            "xtick.labelsize": 17,
            "ytick.labelsize": 18,
            "legend.fontsize": 23,
            "savefig.bbox": "tight",
        }
    )


def fmt(value: float, metric: str) -> str:
    if not np.isfinite(value):
        return ""
    if metric == "r2":
        return f"{value:.2f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def draw_metric_panel(
    axes: np.ndarray,
    metrics: pd.DataFrame,
    metric: str,
    panel_label: str,
) -> None:
    models = [m for m in MODEL_ORDER if m in set(metrics["model"])]
    x = np.arange(len(models))
    width = 0.38

    for r, (target_id, target_label) in enumerate(TARGETS):
        for c, (feature_id, feature_label) in enumerate(FEATURE_SETS):
            ax = axes[r, c]
            panel = metrics[
                (metrics["family"] == "laos")
                & (metrics["target_id"] == target_id)
                & (metrics["feature_set_id"] == feature_id)
            ].copy()

            values_by_validation: dict[str, list[float]] = {}
            for validation in [INTERNAL_LABEL, EXTERNAL_LABEL]:
                values = []
                for model in models:
                    sub = panel[(panel["validation"] == validation) & (panel["model"] == model)]
                    values.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
                values_by_validation[validation] = values

            for offset, validation in [(-width / 2, INTERNAL_LABEL), (width / 2, EXTERNAL_LABEL)]:
                raw = values_by_validation[validation]
                draw = [max(v, 0) if metric == "r2" and np.isfinite(v) else v for v in raw]
                bars = ax.bar(
                    x + offset,
                    draw,
                    width=width,
                    color=VALIDATION_COLORS[validation],
                    edgecolor="white",
                    linewidth=0.9,
                    label=validation,
                )
                y_top = ax.get_ylim()[1]
                for bar, value, draw_value in zip(bars, raw, draw):
                    if not np.isfinite(value):
                        continue
                    if metric == "r2":
                        y = (draw_value if np.isfinite(draw_value) else 0) + 0.018
                        rotation = 90
                    else:
                        y = (draw_value if np.isfinite(draw_value) else 0) + y_top * 0.025
                        rotation = 90
                    if metric == "r2" and value < 0:
                        y = 0.04
                        rotation = 90
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y,
                        fmt(value, metric),
                        ha="center",
                        va="bottom",
                        fontsize=14.5,
                        rotation=rotation,
                        clip_on=False,
                    )

            if metric == "r2":
                ax.set_ylim(0, 1.18)
                ax.axhline(0.8, color="#555555", linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.70)
                metric_label = r"$R^2$"
            else:
                finite_vals = [
                    v
                    for values in values_by_validation.values()
                    for v in values
                    if np.isfinite(v)
                ]
                ax.set_ylim(0, max(finite_vals) * 1.30 if finite_vals else 1)
                metric_label = "RMSE"

            if c == 0:
                ylabel = f"{target_label}\n\n{metric_label}" if metric == "r2" else f"{target_label}\n{metric_label}"
                ax.set_ylabel(ylabel, fontsize=25, fontweight="bold", labelpad=14, linespacing=1.15)
            if r == 0:
                ax.set_title(feature_label, fontsize=24, fontweight="bold", pad=14)

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=38, ha="right", fontsize=16.5)
            ax.tick_params(axis="y", labelsize=18)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.72)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[0, 0].text(
        -0.24,
        1.28,
        panel_label,
        transform=axes[0, 0].transAxes,
        fontsize=42,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    metrics = pd.read_csv(METRICS_PATH)

    fig = plt.figure(figsize=(24, 20), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, hspace=0.12)

    axes_a = subfigs[0].subplots(2, 4, squeeze=False)
    axes_b = subfigs[1].subplots(2, 4, squeeze=False)

    draw_metric_panel(axes_a, metrics, "r2", "A")
    draw_metric_panel(axes_b, metrics, "rmse", "B")

    handles, labels = axes_a[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.045),
        prop={"size": 24, "weight": "bold"},
        handlelength=1.6,
        columnspacing=2.0,
    )

    out_base = OUT_DIR / "Figure_laos_combined_R2_RMSE_by_feature_combination_panel_AB_600dpi"
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI)
    fig.savefig(out_base.with_suffix(".tiff"), dpi=DPI)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    summary = {
        "source_metrics": str(METRICS_PATH),
        "outputs": [str(out_base.with_suffix(ext)) for ext in [".png", ".tiff", ".pdf"]],
        "dpi": DPI,
        "figure_size_inches": [24, 20],
        "panels": {"A": "R2", "B": "RMSE"},
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
