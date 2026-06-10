from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/zhiy/Documents/Rheology ML")
OUTPUTS = ROOT / "outputs"
ONEDRIVE_ROOT = Path("/Users/zhiy/Library/CloudStorage/OneDrive-Personal/GPR new")
RUN_ID = os.environ.get("RHEOLOGY_MAIN_TARGET_FIG_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUTPUTS / f"main_target_publication_figures_{RUN_ID}"

DPI = 600

MODEL_LABELS = {
    "GPR_Matern_ARD": "GPR-Matern-ARD",
    "KernelRidge_RBF": "Kernel ridge-RBF",
    "SVR_RBF": "SVR-RBF",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
}

MODEL_ORDER = [
    "GPR-Matern-ARD",
    "Kernel ridge-RBF",
    "SVR-RBF",
    "Ridge",
    "XGBoost",
    "ExtraTrees",
    "Random forest",
    "Gradient boosting",
]

MAIN_TARGETS = [
    {
        "id": "eta50",
        "task": "viscosity_scalar",
        "target": "log10_eta_50",
        "label": r"$\eta_{50}$",
        "metric_label": r"log$_{10}$ $\eta_{50}$",
        "unit": "cP = mPa·s",
    },
    {
        "id": "Gp1Hz",
        "task": "saos_scalar",
        "target": "log10_Gp_1Hz",
        "label": r"$G'$ at 1 Hz",
        "metric_label": r"log$_{10}$ $G'$",
        "unit": "Pa",
    },
    {
        "id": "break_stress",
        "task": "strain_from_formulation",
        "target": "log10_break_stress_Pa",
        "label": r"breaking stress",
        "metric_label": r"log$_{10}$ $\sigma_\mathrm{break}$",
        "unit": "Pa",
    },
    {
        "id": "break_strain",
        "task": "strain_from_formulation",
        "target": "break_strain_pct",
        "label": r"breaking strain",
        "metric_label": r"$\gamma_\mathrm{break}$",
        "unit": "%",
    },
]


def latest_benchmark() -> Path:
    candidates = sorted(OUTPUTS.glob("ML_results_xanthan_positive_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No ML_results_xanthan_positive_* output folders found")
    return candidates[-1]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.0,
            "savefig.bbox": "tight",
        }
    )


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def load_main_metrics(benchmark: Path) -> pd.DataFrame:
    external = pd.read_csv(benchmark / "all_external_model_metrics.csv")
    external["validation"] = "External"
    internal = pd.read_csv(benchmark / "all_internal_group_cv_mean_metrics.csv")
    internal["validation"] = "Internal 5-fold CV"
    df = pd.concat([external, internal], ignore_index=True)
    df["model_label"] = df["model"].map(model_label)
    rows = []
    for spec in MAIN_TARGETS:
        d = df[(df["task"] == spec["task"]) & (df["target"] == spec["target"])].copy()
        d["target_id"] = spec["id"]
        d["target_label"] = spec["label"]
        d["target_metric_label"] = spec["metric_label"]
        d["unit"] = spec["unit"]
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["model_label"] = pd.Categorical(out["model_label"], categories=[m for m in MODEL_ORDER if m in set(out["model_label"])], ordered=True)
    return out.sort_values(["target_id", "validation", "model_label"]).reset_index(drop=True)


def format_metric(value: float, metric: str) -> str:
    if pd.isna(value):
        return ""
    if metric == "r2":
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot_summary_heatmap(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10.8, 16.8), constrained_layout=True)
    column_keys = [
        ("External", "r2"),
        ("External", "mae"),
        ("External", "rmse"),
        ("Internal 5-fold CV", "r2"),
        ("Internal 5-fold CV", "mae"),
        ("Internal 5-fold CV", "rmse"),
    ]
    column_labels = ["Ext R²", "Ext MAE", "Ext RMSE", "Int R²", "Int MAE", "Int RMSE"]
    for ax, spec in zip(axes, MAIN_TARGETS):
        d = metrics[metrics["target_id"] == spec["id"]].copy()
        models = [m for m in MODEL_ORDER if m in set(d["model_label"])]
        vals_text = []
        vals_color = []
        for model in models:
            row_text, row_color = [], []
            for validation, metric in column_keys:
                one = d[(d["validation"] == validation) & (d["model_label"] == model)]
                value = float(one[metric].iloc[0]) if len(one) else np.nan
                row_text.append(format_metric(value, metric))
                if metric == "r2":
                    color_val = value
                else:
                    series = d[d["validation"] == validation][metric].astype(float)
                    lo, hi = series.min(), series.max()
                    color_val = 1 - (value - lo) / (hi - lo) if hi > lo else 0.5
                row_color.append(color_val)
            vals_text.append(row_text)
            vals_color.append(row_color)
        vals_color = np.asarray(vals_color, dtype=float)
        im = ax.imshow(vals_color, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)
        ax.set_xticks(np.arange(len(column_labels)))
        ax.set_xticklabels(column_labels, rotation=0)
        ax.set_ylabel(f"{spec['label']}\n({spec['metric_label']}, {spec['unit']})", rotation=0, ha="right", va="center", labelpad=58)
        for i in range(vals_color.shape[0]):
            for j in range(vals_color.shape[1]):
                ax.text(j, i, vals_text[i][j], ha="center", va="center", fontsize=10.5, color="black")
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015)
    cbar.set_label("relative performance within validation metric")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def best_three_by_target(metrics: pd.DataFrame) -> pd.DataFrame:
    external = metrics[metrics["validation"] == "External"].copy()
    external["rank_score"] = external.groupby("target_id")["r2"].rank(ascending=False, method="min") + external.groupby("target_id")[
        "rmse"
    ].rank(ascending=True, method="min")
    return external.sort_values(["target_id", "rank_score", "rmse"]).groupby("target_id", as_index=False).head(3)


def plot_best_three(metrics: pd.DataFrame, path: Path) -> None:
    best = best_three_by_target(metrics)
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 14.4), constrained_layout=True)
    colors = {"r2": "#2C7FB8", "mae": "#41AB5D", "rmse": "#D95F0E"}
    for r, spec in enumerate(MAIN_TARGETS):
        d = best[best["target_id"] == spec["id"]].copy()
        for c in range(3):
            ax = axes[r, c]
            if c >= len(d):
                ax.axis("off")
                continue
            row = d.iloc[c]
            values = [row["r2"], row["mae"], row["rmse"]]
            labels = ["R²", "MAE", "RMSE"]
            scaled = [row["r2"], min(row["mae"] / max(d["mae"].max(), 1e-9), 1), min(row["rmse"] / max(d["rmse"].max(), 1e-9), 1)]
            ax.barh(labels, scaled, color=[colors["r2"], colors["mae"], colors["rmse"]], alpha=0.9)
            ax.set_xlim(0, 1.08)
            ax.invert_yaxis()
            ax.set_yticklabels(labels, fontsize=12)
            ax.set_xticks([])
            ax.grid(axis="x", alpha=0.18)
            ax.text(0.02, 1.08, str(row["model_label"]), transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")
            ax.text(
                0.98,
                1.08,
                f"{spec['label']}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=12,
            )
            for i, (metric, value) in enumerate(zip(labels, values)):
                ax.text(
                    min(scaled[i] + 0.03, 1.03),
                    i,
                    format_metric(value, "r2" if metric == "R²" else metric.lower()),
                    va="center",
                    ha="left",
                    fontsize=12,
                )
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_lengthscales(benchmark: Path, path: Path) -> None:
    scalar = pd.read_csv(benchmark / "gpr_lengthscales_scalar_targets.csv")
    full = pd.read_csv(benchmark / "gpr_lengthscales_full_viscosity_curve.csv")
    df = pd.concat([scalar, full], ignore_index=True)
    rows = []
    for spec in MAIN_TARGETS:
        d = df[(df["task"] == spec["task"]) & (df["target"] == spec["target"])].copy()
        d["target_display"] = f"{spec['label']}\n{spec['metric_label']}"
        d["target_order"] = MAIN_TARGETS.index(spec)
        rows.append(d)
    length = pd.concat(rows, ignore_index=True)
    pivot = length.pivot_table(index="target_display", columns="feature", values="relative_relevance", aggfunc="mean").fillna(0)
    order = (
        length[["target_display", "target_order"]]
        .drop_duplicates()
        .sort_values("target_order")["target_display"]
        .tolist()
    )
    pivot = pivot.reindex(order)
    feature_order = [f for f in ["yp_pct", "xanthan_pct"] if f in pivot.columns]
    pivot = pivot[feature_order]
    feature_labels = {"yp_pct": "yeast protein (%)", "xanthan_pct": "xanthan gum (%)"}
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    vals = pivot.to_numpy(dtype=float)
    im = ax.imshow(vals, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.01, vals.max()))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([feature_labels.get(c, c) for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("relative relevance (1 / ARD length scale)")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    benchmark = latest_benchmark()
    metrics = load_main_metrics(benchmark)
    metrics.to_csv(OUT_DIR / "main_targets_internal_external_metrics.csv", index=False)
    best_three_by_target(metrics).to_csv(OUT_DIR / "main_targets_best_three_external_models.csv", index=False)
    plot_summary_heatmap(metrics, OUT_DIR / "figure_1_main_targets_internal_external_summary_600dpi.png")
    plot_best_three(metrics, OUT_DIR / "figure_2_main_targets_best_three_models_600dpi.png")
    plot_lengthscales(benchmark, OUT_DIR / "figure_3_main_targets_gpr_lengthscales_600dpi.png")
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)
    summary = {
        "run_id": RUN_ID,
        "benchmark_source": str(benchmark),
        "dpi": DPI,
        "targets": [{k: v for k, v in spec.items() if k in {"id", "task", "target", "label", "unit"}} for spec in MAIN_TARGETS],
        "outputs": [
            "figure_1_main_targets_internal_external_summary_600dpi.png",
            "figure_2_main_targets_best_three_models_600dpi.png",
            "figure_3_main_targets_gpr_lengthscales_600dpi.png",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
