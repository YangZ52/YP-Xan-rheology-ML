from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ARCHIVE_ROOT = Path(os.environ.get("RHEOLOGY_ARCHIVE_ROOT", ROOT / "outputs"))
BENCHMARK = OUTPUTS / "ML_results_xanthan_positive_20260530_131336"
DATA_DIR = OUTPUTS / "ml_ready_xanthan_positive_20260529"
METRIC_WORKBOOK = (
    OUTPUTS
    / "publication_metric_table_20260530_151410"
    / "rheology_ml_publication_metric_table.xlsx"
)

RUN_ID = os.environ.get("RHEOLOGY_REVIEWER_FIG_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUTPUTS / f"reviewer_performance_figures_{RUN_ID}"
DPI = 600
INTERNAL_VALIDATION_LABEL = "Internal 5-fold CV"
EXTERNAL_VALIDATION_LABEL = "External validation"
_BALANCED_INTERNAL_CACHE = None

MODEL_LABELS = {
    "GPR_Matern_ARD": "GPR-Matern-ARD",
    "KernelRidge_RBF": "KRR-RBF",
    "SVR_RBF": "SVR-RBF",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
    "ExtraTrees": "Extra Trees",
    "XGBoost": "XGBoost",
    "Ridge": "Ridge",
}

MODEL_ORDER = [
    "GPR-Matern-ARD",
    "KRR-RBF",
    "SVR-RBF",
    "Ridge",
    "XGBoost",
    "Extra Trees",
    "Random forest",
    "Gradient boosting",
]

TARGETS = [
    {
        "id": "Gp1Hz",
        "task": "saos_scalar",
        "target": "log10_Gp_1Hz",
        "short": r"$G'$ 1 Hz",
        "bar_title": r"$G'$ at 1 Hz",
        "axis": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
        "unit": "log10 Pa",
    },
    {
        "id": "eta50",
        "task": "viscosity_scalar",
        "target": "log10_eta_50",
        "short": r"$\eta_{50}$",
        "bar_title": r"$\eta_{50}$",
        "axis": r"log$_{10}$ $\eta_{50}$ (mPa s)",
        "unit": "log10 mPa s",
    },
    {
        "id": "break_stress",
        "task": "strain_from_formulation",
        "target": "log10_break_stress_Pa",
        "short": r"$\sigma_\mathrm{break}$",
        "bar_title": "Breaking stress",
        "axis": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
        "unit": "log10 Pa",
    },
    {
        "id": "break_strain",
        "task": "strain_from_formulation",
        "target": "break_strain_pct",
        "short": r"$\gamma_\mathrm{break}$",
        "bar_title": "Breaking strain",
        "axis": r"$\gamma_\mathrm{break}$ (%)",
        "unit": "%",
    },
]


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def make_gpr(n_features: int) -> Pipeline:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e3), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    )
    return Pipeline(
        [
            ("x_scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=8,
                    random_state=42,
                ),
            ),
        ]
    )


def make_models(n_features: int) -> dict:
    models = {
        "Ridge": Pipeline([("x_scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "SVR_RBF": Pipeline([("x_scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale"))]),
        "KernelRidge_RBF": Pipeline([("x_scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf", alpha=0.05, gamma=None))]),
        "GPR_Matern_ARD": make_gpr(n_features),
        "RandomForest": RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=2,
            random_state=42,
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=350,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=4.0,
            objective="reg:squarederror",
            random_state=42,
        )
    return models


def formulation_level_master() -> pd.DataFrame:
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    rep["log10_break_stress_Pa"] = np.where(rep["break_stress_Pa"] > 0, np.log10(rep["break_stress_Pa"]), np.nan)
    numeric_cols = rep.select_dtypes(include=[np.number]).columns.tolist()
    return rep.groupby(["split", "formulation_std"], as_index=False)[numeric_cols].mean()


def balanced_grid_splits(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Latin-style formulation folds for the regular xanthan x protein training grid."""
    xg_levels = sorted(train["xanthan_pct"].unique())
    yp_levels = sorted(train["yp_pct"].unique())
    fold_by_index = {}
    complete_grid = True
    for i, xg in enumerate(xg_levels):
        for j, yp in enumerate(yp_levels):
            matches = train.index[(train["xanthan_pct"] == xg) & (train["yp_pct"] == yp)].tolist()
            if len(matches) != 1:
                complete_grid = False
                break
            fold_by_index[matches[0]] = (i + j) % 5
        if not complete_grid:
            break
    if not complete_grid:
        raise ValueError("Balanced grid CV requires one training formulation at every xanthan x protein grid point.")
    splits = []
    all_idx = np.asarray(train.index.tolist())
    for fold in range(5):
        va_idx = np.asarray([idx for idx in all_idx if fold_by_index[idx] == fold])
        tr_idx = np.asarray([idx for idx in all_idx if fold_by_index[idx] != fold])
        splits.append((tr_idx, va_idx))
    return splits


def load_metrics() -> pd.DataFrame:
    """Use benchmark CSVs for exact IDs, while keeping the workbook as the table source."""
    if not METRIC_WORKBOOK.exists():
        raise FileNotFoundError(METRIC_WORKBOOK)
    # Read once so the script is explicitly tied to the publication workbook requested.
    pd.read_excel(METRIC_WORKBOOK, sheet_name="External validation", header=None, nrows=5)

    external = pd.read_csv(BENCHMARK / "all_external_model_metrics.csv")
    external["validation"] = "External validation"
    internal, _, _ = balanced_internal_cv_results()
    df = pd.concat([internal, external], ignore_index=True)
    df["model_label"] = df["model"].map(model_label)
    rows = []
    for order, spec in enumerate(TARGETS):
        d = df[(df["task"] == spec["task"]) & (df["target"] == spec["target"])].copy()
        d["target_id"] = spec["id"]
        d["target_order"] = order
        d["target_label"] = spec["bar_title"]
        d["axis_label"] = spec["axis"]
        d["unit"] = spec["unit"]
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["model_label"] = pd.Categorical(out["model_label"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["target_order", "validation", "model_label"]).reset_index(drop=True)


def balanced_internal_cv_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Xanthan-stratified formulation CV, so every fold spans the formulation space."""
    global _BALANCED_INTERNAL_CACHE
    if _BALANCED_INTERNAL_CACHE is not None:
        return _BALANCED_INTERNAL_CACHE

    data = formulation_level_master()
    features = ["yp_pct", "xanthan_pct"]
    model_dict = make_models(len(features))
    metric_rows = []
    pred_rows = []
    fold_rows = []

    for spec in TARGETS:
        needed = ["split", "formulation_std", spec["target"], *features]
        train = data.loc[data["split"] == "train", needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
        train = train.sort_values(["xanthan_pct", "yp_pct"]).reset_index(drop=True)
        splits = balanced_grid_splits(train)

        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            for role, idxs in [("training", tr_idx), ("validation", va_idx)]:
                for _, row in train.iloc[idxs].iterrows():
                    fold_rows.append(
                        {
                            "task": spec["task"],
                            "target": spec["target"],
                            "feature_set": ",".join(features),
                            "fold": fold,
                            "role": role,
                            "formulation_std": row["formulation_std"],
                            "yp_pct": row["yp_pct"],
                            "xanthan_pct": row["xanthan_pct"],
                        }
                    )

        for model_name, base_model in model_dict.items():
            y_true_all = []
            y_pred_all = []
            for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
                model = clone(base_model)
                model.fit(train.iloc[tr_idx][features], train.iloc[tr_idx][spec["target"]])
                pred = np.asarray(model.predict(train.iloc[va_idx][features]), dtype=float)
                y_true = train.iloc[va_idx][spec["target"]].to_numpy(dtype=float)
                y_true_all.extend(y_true.tolist())
                y_pred_all.extend(pred.tolist())
                for idx, p in zip(va_idx, pred):
                    source = train.iloc[idx]
                    pred_rows.append(
                        {
                            "task": spec["task"],
                            "target": spec["target"],
                            "model": model_name,
                            "split": "internal_xg_stratified_cv",
                            "fold": fold,
                            "formulation_std": source["formulation_std"],
                            "y_true": source[spec["target"]],
                            "y_pred": float(p),
                            "yp_pct": source["yp_pct"],
                            "xanthan_pct": source["xanthan_pct"],
                        }
                    )
            metric_rows.append(
                {
                    "task": spec["task"],
                    "target": spec["target"],
                    "feature_set": ",".join(features),
                    "model": model_name,
                    "mae": mean_absolute_error(y_true_all, y_pred_all),
                    "rmse": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
                    "r2": r2_score(y_true_all, y_pred_all),
                    "validation": INTERNAL_VALIDATION_LABEL,
                    "split": "internal_xg_stratified_cv",
                }
            )

    _BALANCED_INTERNAL_CACHE = (pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(fold_rows))
    return _BALANCED_INTERNAL_CACHE


def fmt_value(value: float, metric: str) -> str:
    if pd.isna(value):
        return ""
    if metric == "r2":
        return f"{value:.2f}"
    av = abs(value)
    if av >= 100:
        return f"{value:.0f}"
    if av >= 10:
        return f"{value:.1f}"
    if av >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def metric_label(metric: str) -> str:
    return {"r2": r"$R^2$", "mae": "MAE", "rmse": "RMSE"}[metric]


def plot_grouped_metric_bars(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.89, wspace=0.20, hspace=0.43)
    axes = axes.ravel()
    colors = {INTERNAL_VALIDATION_LABEL: "#6E8FB2", EXTERNAL_VALIDATION_LABEL: "#C58B57"}
    width = 0.38
    panel_letters = list("ABCD")
    for ax, spec, letter in zip(axes, TARGETS, panel_letters):
        d = metrics[metrics["target_id"] == spec["id"]]
        models = [m for m in MODEL_ORDER if m in set(d["model_label"].astype(str))]
        x = np.arange(len(models))
        vals = {}
        for val_name in [INTERNAL_VALIDATION_LABEL, EXTERNAL_VALIDATION_LABEL]:
            vals[val_name] = [
                float(
                    d[(d["validation"] == val_name) & (d["model_label"].astype(str) == model)][
                        metric
                    ].iloc[0]
                )
                for model in models
            ]
        draw_vals_int = [max(v, 0) if metric == "r2" else v for v in vals[INTERNAL_VALIDATION_LABEL]]
        draw_vals_ext = [max(v, 0) if metric == "r2" else v for v in vals[EXTERNAL_VALIDATION_LABEL]]
        bars_int = ax.bar(
            x - width / 2,
            draw_vals_int,
            width=width,
            label="Internal 5-fold CV",
            color=colors[INTERNAL_VALIDATION_LABEL],
            edgecolor="white",
            linewidth=0.7,
        )
        bars_ext = ax.bar(
            x + width / 2,
            draw_vals_ext,
            width=width,
            label=EXTERNAL_VALIDATION_LABEL,
            color=colors[EXTERNAL_VALIDATION_LABEL],
            edgecolor="white",
            linewidth=0.7,
        )
        if metric == "r2":
            ax.set_ylim(0, 1.05)
            ax.axhline(0.8, color="#555555", linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.8)
        else:
            ymax = max(max(vals[INTERNAL_VALIDATION_LABEL]), max(vals[EXTERNAL_VALIDATION_LABEL]))
            ax.set_ylim(0, ymax * 1.23 if ymax > 0 else 1)

        best_int = int(np.argmax(vals[INTERNAL_VALIDATION_LABEL]) if metric == "r2" else np.argmin(vals[INTERNAL_VALIDATION_LABEL]))
        best_ext = int(np.argmax(vals[EXTERNAL_VALIDATION_LABEL]) if metric == "r2" else np.argmin(vals[EXTERNAL_VALIDATION_LABEL]))
        bars_int[best_int].set_edgecolor("black")
        bars_int[best_int].set_linewidth(1.4)
        bars_ext[best_ext].set_edgecolor("black")
        bars_ext[best_ext].set_linewidth(1.4)
        y_text = ax.get_ylim()[1]
        for bars, values, draw_values in [
            (bars_int, vals[INTERNAL_VALIDATION_LABEL], draw_vals_int),
            (bars_ext, vals[EXTERNAL_VALIDATION_LABEL], draw_vals_ext),
        ]:
            for bar, value, draw_value in zip(bars, values, draw_values):
                offset = 0.014 if metric == "r2" else y_text * 0.018
                label_y = draw_value + offset
                if metric == "r2" and value < 0:
                    label_y = 0.03
                    ax.scatter(
                        bar.get_x() + bar.get_width() / 2,
                        0.012,
                        marker="v",
                        s=22,
                        color="#4A4A4A",
                        zorder=4,
                        clip_on=False,
                    )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    fmt_value(value, metric),
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    rotation=90 if metric != "r2" or value < 0 else 0,
                    clip_on=False,
                )
        ax.text(
            -0.12,
            1.05,
            letter,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.set_title(spec["bar_title"], fontsize=11, pad=8)
        ax.set_ylabel(metric_label(metric), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=38, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def normalized_by_column(values: pd.DataFrame, higher_better: bool) -> pd.DataFrame:
    norm = values.copy().astype(float)
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        if hi == lo:
            norm[col] = 0.5
        elif higher_better:
            norm[col] = (norm[col] - lo) / (hi - lo)
        else:
            norm[col] = 1 - (norm[col] - lo) / (hi - lo)
    return norm


def plot_external_heatmap(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    d = metrics[metrics["validation"] == "External validation"].copy()
    values = (
        d.pivot(index="model_label", columns="target_id", values=metric)
        .reindex(index=MODEL_ORDER, columns=[t["id"] for t in TARGETS])
    )
    norm = normalized_by_column(values, higher_better=(metric == "r2"))
    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    im = ax.imshow(norm.values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels([t["short"] for t in TARGETS], fontsize=10)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=10)
    ax.set_xlabel("Target variable", fontsize=12)
    ax.set_ylabel("ML model", fontsize=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, fmt_value(values.iloc[i, j], metric), ha="center", va="center", fontsize=9.2)
    ax.set_xticks(np.arange(-0.5, len(TARGETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.025)
    cbar.set_label("normalized external performance", fontsize=11)
    note = "Higher values indicate better performance." if metric == "r2" else "Lower values indicate better performance; colors are normalized within each target."
    ax.text(0.0, -0.18, note, transform=ax.transAxes, fontsize=9.5, ha="left", va="top")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def best_three_external(metrics: pd.DataFrame) -> pd.DataFrame:
    d = metrics[metrics["validation"] == "External validation"].copy()
    d = d.sort_values(["target_order", "r2", "rmse"], ascending=[True, False, True])
    d["rank"] = d.groupby("target_id").cumcount() + 1
    return d[d["rank"] <= 3].copy()


def panel_limits(values: pd.DataFrame) -> tuple[float, float]:
    lo = float(np.nanmin([values["y_true"].min(), values["y_pred"].min()]))
    hi = float(np.nanmax([values["y_true"].max(), values["y_pred"].max()]))
    pad = (hi - lo) * 0.08 if hi > lo else max(abs(hi) * 0.08, 1)
    return lo - pad, hi + pad


def annotation_position(d: pd.DataFrame) -> tuple[float, float, str, str]:
    corr = np.corrcoef(d["y_true"], d["y_pred"])[0, 1] if len(d) > 2 else 0
    if corr >= 0:
        return 0.04, 0.96, "left", "top"
    return 0.96, 0.04, "right", "bottom"


def plot_external_top3_parity(metrics: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    best = best_three_external(metrics)
    best.to_csv(out_dir / "Figure5_top3_external_models_summary.csv", index=False)

    row_colors = ["#4C78A8", "#7C6A9B", "#5F8D6A", "#A66B55"]
    fig, axes = plt.subplots(4, 3, figsize=(11.2, 13.8), constrained_layout=True)
    letters = list("ABCDEFGHIJKL")
    for r, spec in enumerate(TARGETS):
        target_best = best[best["target_id"] == spec["id"]].sort_values("rank")
        row_pred = predictions[
            (predictions["split"] == "external_predict")
            & (predictions["task"] == spec["task"])
            & (predictions["target"] == spec["target"])
            & (predictions["model"].isin(target_best["model"]))
        ].copy()
        lims = panel_limits(row_pred)
        for c in range(3):
            ax = axes[r, c]
            rank_row = target_best.iloc[c]
            model = rank_row["model"]
            model_name = rank_row["model_label"]
            d = row_pred[row_pred["model"] == model].copy()
            ax.scatter(
                d["y_true"],
                d["y_pred"],
                s=62,
                c=row_colors[r],
                edgecolors="black",
                linewidths=0.6,
                alpha=0.82,
                zorder=3,
            )
            ax.plot(lims, lims, color="black", linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
            if len(d) >= 2 and d["y_true"].nunique() > 1:
                slope, intercept = np.polyfit(d["y_true"], d["y_pred"], 1)
                xfit = np.asarray(lims)
                yfit = slope * xfit + intercept
                ax.plot(
                    xfit,
                    yfit,
                    color="#B45A4D",
                    linestyle="-",
                    linewidth=1.25,
                    zorder=2,
                )
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{spec['short']} — Rank {int(rank_row['rank'])}: {model_name}", fontsize=10.5, pad=7)
            ax.set_xlabel(f"Measured {spec['axis']}", fontsize=10.5)
            ax.set_ylabel(f"Predicted {spec['axis']}", fontsize=10.5)
            xann, yann, ha, va = annotation_position(d)
            ax.text(
                xann,
                yann,
                f"$R^2$ = {rank_row['r2']:.2f}\nMAE = {fmt_value(rank_row['mae'], 'mae')}\nRMSE = {fmt_value(rank_row['rmse'], 'rmse')}",
                transform=ax.transAxes,
                ha=ha,
                va=va,
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, boxstyle="round,pad=0.24"),
            )
            ax.text(
                -0.16,
                1.08,
                letters[r * 3 + c],
                transform=ax.transAxes,
                fontsize=15,
                fontweight="bold",
                va="top",
                ha="left",
            )
            ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.65)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    base = out_dir / "Figure5_external_parity_top3_models"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=DPI)
    fig.savefig(base.with_suffix(".tiff"), dpi=DPI)
    plt.close(fig)
    return best


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    metrics = load_metrics()
    balanced_internal_metrics, balanced_internal_predictions, balanced_internal_folds = balanced_internal_cv_results()
    predictions = pd.read_csv(BENCHMARK / "scalar_model_predictions.csv")

    metrics.to_csv(OUT_DIR / "main_target_internal_external_metrics.csv", index=False)
    balanced_internal_metrics.to_csv(OUT_DIR / "balanced_xg_stratified_internal_cv_metrics.csv", index=False)
    balanced_internal_predictions.to_csv(OUT_DIR / "balanced_xg_stratified_internal_cv_predictions.csv", index=False)
    balanced_internal_folds.to_csv(OUT_DIR / "balanced_xg_stratified_internal_cv_fold_assignments.csv", index=False)
    for metric in ["r2", "mae", "rmse"]:
        plot_grouped_metric_bars(
            metrics,
            metric,
            OUT_DIR / f"PartA_grouped_bar_{metric.upper()}_internal_external_600dpi.png",
        )
        plot_external_heatmap(
            metrics,
            metric,
            OUT_DIR / f"PartB_external_heatmap_{metric.upper()}_600dpi.png",
        )

    best = plot_external_top3_parity(metrics, predictions, OUT_DIR)

    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)
    summary = {
        "run_id": RUN_ID,
        "benchmark": str(BENCHMARK),
        "metric_workbook": str(METRIC_WORKBOOK),
        "dpi": DPI,
        "targets": TARGETS,
        "model_order": MODEL_ORDER,
        "internal_cv_rule": "5-fold formulation-level balanced Latin-grid CV over the complete xanthan_pct x yp_pct training grid; each validation fold contains all six protein levels and spans all five xanthan levels.",
        "exports": sorted(p.name for p in OUT_DIR.iterdir()),
        "top3_models": best[["target_id", "rank", "model_label", "r2", "mae", "rmse"]].to_dict("records"),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
