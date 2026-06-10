from __future__ import annotations

import json
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

MPLCONFIGDIR = Path("/private/tmp/rheology_ml_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "ml_ready_xanthan_positive_20260529"
RUN_ID = os.environ.get("RHEOLOGY_PAPER_STYLE_PARITY_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"paper_style_parity_no_LVR_{RUN_ID}"
DPI = 600

FEATURE_SETS = [
    {
        "id": "formulation_Gp1Hz",
        "label": r"formulation + $G'$ 1 Hz",
        "short": r"$G'$ 1 Hz",
        "features": ["yp_pct", "xanthan_pct", "log10_Gp_1Hz"],
    },
    {
        "id": "formulation_eta50",
        "label": r"formulation + $\eta_{50}$",
        "short": r"$\eta_{50}$",
        "features": ["yp_pct", "xanthan_pct", "log10_eta_50"],
    },
    {
        "id": "formulation_Gp1Hz_eta50",
        "label": r"formulation + $G'$ 1 Hz + $\eta_{50}$",
        "short": r"$G'$ 1 Hz + $\eta_{50}$",
        "features": ["yp_pct", "xanthan_pct", "log10_Gp_1Hz", "log10_eta_50"],
    },
]

LAOS_TARGETS = [
    {
        "id": "break_stress",
        "target": "log10_break_stress_Pa",
        "symbol": r"$\sigma_\mathrm{break}$",
        "axis": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
        "color": "#5F8D6A",
    },
    {
        "id": "break_strain",
        "target": "break_strain_pct",
        "symbol": r"$\gamma_\mathrm{break}$",
        "axis": r"$\gamma_\mathrm{break}$ (%)",
        "color": "#A66B55",
    },
]

DESCRIPTOR_TARGETS = [
    {
        "id": "Gp1Hz",
        "target": "log10_Gp_1Hz",
        "symbol": r"$G'$ 1 Hz",
        "axis": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
        "features": ["yp_pct", "xanthan_pct"],
        "feature_set_id": "formulation",
        "feature_set_label": "formulation",
        "color": "#4C78A8",
    },
    {
        "id": "eta50",
        "target": "log10_eta_50",
        "symbol": r"$\eta_{50}$",
        "axis": r"log$_{10}$ $\eta_{50}$ (Pa s)",
        "features": ["yp_pct", "xanthan_pct"],
        "feature_set_id": "formulation",
        "feature_set_label": "formulation",
        "color": "#7C6A9B",
    },
]

MODEL_LABELS = {
    "GPR_Matern_ARD": "GPR-Matern-ARD",
    "KernelRidge_RBF": "KRR-RBF",
    "SVR_RBF": "SVR-RBF",
    "Ridge": "Ridge",
    "XGBoost": "XGBoost",
    "ExtraTrees": "Extra Trees",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
}

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


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 8.8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "savefig.bbox": "tight",
        }
    )


def make_gpr(n_features: int) -> Pipeline:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
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
        "GPR_Matern_ARD": make_gpr(n_features),
        "KernelRidge_RBF": Pipeline([("x_scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf", alpha=0.05))]),
        "SVR_RBF": Pipeline([("x_scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale"))]),
        "Ridge": Pipeline([("x_scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
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
    return {k: models[k] for k in MODEL_ORDER if k in models}


def formulation_level_master() -> pd.DataFrame:
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    rep["log10_break_stress_Pa"] = np.where(rep["break_stress_Pa"] > 0, np.log10(rep["break_stress_Pa"]), np.nan)
    numeric_cols = rep.select_dtypes(include=[np.number]).columns.tolist()
    return rep.groupby(["split", "formulation_std"], as_index=False)[numeric_cols].mean()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def run_one_spec(data: pd.DataFrame, spec: dict, feature_set: dict) -> tuple[list[dict], list[dict]]:
    features = feature_set["features"]
    target = spec["target"]
    needed = ["split", "formulation_std", target, *features]
    d = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train = d[d["split"] == "train"].reset_index(drop=True)
    test = d[d["split"] == "predict"].reset_index(drop=True)
    metric_rows = []
    pred_rows = []
    for model_name, base_model in make_models(len(features)).items():
        cv = GroupKFold(n_splits=min(5, train["formulation_std"].nunique()))
        y_true_all, y_pred_all = [], []
        for fold, (tr_idx, va_idx) in enumerate(cv.split(train[features], train[target], train["formulation_std"]), start=1):
            model = clone(base_model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train.iloc[tr_idx][features], train.iloc[tr_idx][target])
            pred = np.asarray(model.predict(train.iloc[va_idx][features]), dtype=float)
            y_true = train.iloc[va_idx][target].to_numpy(dtype=float)
            y_true_all.extend(y_true.tolist())
            y_pred_all.extend(pred.tolist())
        metric_rows.append(
            {
                "target_id": spec["id"],
                "target": target,
                "symbol": spec["symbol"],
                "axis": spec["axis"],
                "feature_set_id": feature_set["id"],
                "feature_set_label": feature_set["label"],
                "features": ",".join(features),
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "validation": "Internal 5-fold CV",
                **metric_row(np.asarray(y_true_all), np.asarray(y_pred_all)),
            }
        )
        final = clone(base_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final.fit(train[features], train[target])
        pred = np.asarray(final.predict(test[features]), dtype=float)
        metric_rows.append(
            {
                "target_id": spec["id"],
                "target": target,
                "symbol": spec["symbol"],
                "axis": spec["axis"],
                "feature_set_id": feature_set["id"],
                "feature_set_label": feature_set["label"],
                "features": ",".join(features),
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "validation": "External validation",
                **metric_row(test[target].to_numpy(dtype=float), pred),
            }
        )
        for idx, y_hat in enumerate(pred):
            pred_rows.append(
                {
                    "target_id": spec["id"],
                    "target": target,
                    "symbol": spec["symbol"],
                    "axis": spec["axis"],
                    "color": spec["color"],
                    "feature_set_id": feature_set["id"],
                    "feature_set_label": feature_set["label"],
                    "features": ",".join(features),
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "validation": "External validation",
                    "formulation_std": test.loc[idx, "formulation_std"],
                    "y_true": test.loc[idx, target],
                    "y_pred": float(y_hat),
                }
            )
    return metric_rows, pred_rows


def run_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    for spec in DESCRIPTOR_TARGETS:
        m, p = run_one_spec(
            data,
            spec,
            {"id": spec["feature_set_id"], "label": spec["feature_set_label"], "features": spec["features"]},
        )
        metric_rows.extend(m)
        pred_rows.extend(p)
    for spec in LAOS_TARGETS:
        for feature_set in FEATURE_SETS:
            m, p = run_one_spec(data, spec, feature_set)
            metric_rows.extend(m)
            pred_rows.extend(p)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def rank_rows(metrics: pd.DataFrame, target_ids: list[str]) -> pd.DataFrame:
    d = metrics[(metrics["validation"] == "External validation") & (metrics["target_id"].isin(target_ids))].copy()
    d = d.sort_values(["target_id", "r2", "rmse", "mae"], ascending=[True, False, True, True])
    d["rank"] = d.groupby("target_id").cumcount() + 1
    return d[d["rank"] <= 3].copy()


def fmt(value: float, metric: str) -> str:
    if metric == "r2":
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def panel_limits(d: pd.DataFrame) -> tuple[float, float]:
    lo = float(np.nanmin([d["y_true"].min(), d["y_pred"].min()]))
    hi = float(np.nanmax([d["y_true"].max(), d["y_pred"].max()]))
    pad = (hi - lo) * 0.08 if hi > lo else 1.0
    return lo - pad, hi + pad


def draw_panel(ax: plt.Axes, d: pd.DataFrame, metric: pd.Series, letter: str) -> None:
    color = d["color"].iloc[0]
    lo, hi = panel_limits(d)
    ax.scatter(d["y_true"], d["y_pred"], s=32, color=color, edgecolor="#303030", linewidth=0.35, alpha=0.88, zorder=3)
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=(0, (4, 3)), linewidth=0.85, zorder=1)
    if len(d) >= 2:
        slope, intercept = np.polyfit(d["y_true"], d["y_pred"], 1)
        xx = np.array([lo, hi])
        ax.plot(xx, slope * xx + intercept, color="#C4453C", linewidth=0.85, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.28, 1.16, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top")
    ax.text(
        0.04,
        0.94,
        f"$R^2$ = {fmt(metric['r2'], 'r2')}\nMAE = {fmt(metric['mae'], 'mae')}\nRMSE = {fmt(metric['rmse'], 'rmse')}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
    )
    ax.set_xlabel(f"Measured {metric['axis']}", labelpad=2)
    ax.set_ylabel(f"Predicted {metric['axis']}", labelpad=2)
    ax.set_title(f"{metric['symbol']} — Rank {int(metric['rank'])}: {metric['model_label']}", pad=5)


def save_all(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.tiff", dpi=DPI)
    plt.close(fig)


def plot_main_best3(metrics: pd.DataFrame, preds: pd.DataFrame) -> None:
    desc_best = rank_rows(metrics, ["Gp1Hz", "eta50"])
    laos_best = rank_rows(metrics, ["break_stress", "break_strain"])
    best = pd.concat(
        [
            desc_best[desc_best["target_id"] == "Gp1Hz"],
            desc_best[desc_best["target_id"] == "eta50"],
            laos_best[laos_best["target_id"] == "break_stress"],
            laos_best[laos_best["target_id"] == "break_strain"],
        ],
        ignore_index=True,
    )
    best.to_csv(OUT_DIR / "paper_style_best_three_external_models.csv", index=False)
    fig, axes = plt.subplots(4, 3, figsize=(7.1, 9.9))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.055, top=0.975, wspace=0.36, hspace=0.58)
    for idx, (ax, (_, row)) in enumerate(zip(axes.ravel(), best.iterrows())):
        d = preds[
            (preds["validation"] == "External validation")
            & (preds["target_id"] == row["target_id"])
            & (preds["feature_set_id"] == row["feature_set_id"])
            & (preds["model"] == row["model"])
        ]
        metric = row.copy()
        metric["rank"] = (idx % 3) + 1
        draw_panel(ax, d, metric, chr(ord("A") + idx))
    save_all(fig, "Figure_paper_style_best3_parity_no_LVR")


def plot_laos_feature_set_model_panels(metrics: pd.DataFrame, preds: pd.DataFrame) -> None:
    out_dir = OUT_DIR / "LAOS_parity_by_model_no_LVR"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in MODEL_ORDER:
        if model not in set(metrics["model"]):
            continue
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.095, top=0.90, wspace=0.38, hspace=0.54)
        panel = 0
        for r, target in enumerate(LAOS_TARGETS):
            for c, fset in enumerate(FEATURE_SETS):
                ax = axes[r, c]
                m = metrics[
                    (metrics["validation"] == "External validation")
                    & (metrics["target_id"] == target["id"])
                    & (metrics["feature_set_id"] == fset["id"])
                    & (metrics["model"] == model)
                ].iloc[0].copy()
                m["rank"] = c + 1
                d = preds[
                    (preds["validation"] == "External validation")
                    & (preds["target_id"] == target["id"])
                    & (preds["feature_set_id"] == fset["id"])
                    & (preds["model"] == model)
                ]
                draw_panel(ax, d, m, chr(ord("A") + panel))
                ax.set_title(f"{target['symbol']} — {fset['short']}", pad=5)
                panel += 1
        fig.suptitle(f"External parity: {MODEL_LABELS[model]}", fontsize=11, y=0.985)
        for ext in ["png", "pdf", "tiff"]:
            fig.savefig(out_dir / f"parity_external_{model}_no_LVR.{ext}", dpi=DPI if ext != "pdf" else None)
        plt.close(fig)


def plot_laos_best3_by_feature_set(metrics: pd.DataFrame, preds: pd.DataFrame) -> None:
    external = metrics[
        (metrics["validation"] == "External validation") & (metrics["target_id"].isin([t["id"] for t in LAOS_TARGETS]))
    ].copy()
    external = external.sort_values(
        ["target_id", "feature_set_id", "r2", "rmse", "mae"],
        ascending=[True, True, False, True, True],
    )
    external["rank"] = external.groupby(["target_id", "feature_set_id"]).cumcount() + 1
    best = external[external["rank"] <= 3].copy()
    best.to_csv(OUT_DIR / "LAOS_best_three_models_by_feature_set.csv", index=False)

    target_order = ["break_stress", "break_strain"]
    feature_order = ["formulation_eta50", "formulation_Gp1Hz", "formulation_Gp1Hz_eta50"]
    fig, axes = plt.subplots(6, 3, figsize=(7.15, 14.3))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.045, top=0.955, wspace=0.38, hspace=0.64)
    panel = 0
    for target_idx, target_id in enumerate(target_order):
        for feature_idx, feature_id in enumerate(feature_order):
            row_idx = target_idx * len(feature_order) + feature_idx
            rows = best[(best["target_id"] == target_id) & (best["feature_set_id"] == feature_id)].sort_values("rank")
            for col_idx, (_, row) in enumerate(rows.iterrows()):
                ax = axes[row_idx, col_idx]
                d = preds[
                    (preds["validation"] == "External validation")
                    & (preds["target_id"] == row["target_id"])
                    & (preds["feature_set_id"] == row["feature_set_id"])
                    & (preds["model"] == row["model"])
                ]
                draw_panel(ax, d, row, chr(ord("A") + panel))
                ax.set_title(f"{row['symbol']} — {row['feature_set_label']}\nRank {int(row['rank'])}: {row['model_label']}", pad=5)
                panel += 1
    save_all(fig, "Figure_LAOS_best3_by_feature_set_no_LVR")

    for target_id in target_order:
        target_rows = best[best["target_id"] == target_id].copy()
        fig, axes = plt.subplots(3, 3, figsize=(7.15, 7.2))
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.065, top=0.935, wspace=0.38, hspace=0.64)
        panel = 0
        for row_idx, feature_id in enumerate(feature_order):
            rows = target_rows[target_rows["feature_set_id"] == feature_id].sort_values("rank")
            for col_idx, (_, row) in enumerate(rows.iterrows()):
                ax = axes[row_idx, col_idx]
                d = preds[
                    (preds["validation"] == "External validation")
                    & (preds["target_id"] == row["target_id"])
                    & (preds["feature_set_id"] == row["feature_set_id"])
                    & (preds["model"] == row["model"])
                ]
                draw_panel(ax, d, row, chr(ord("A") + panel))
                ax.set_title(f"{row['feature_set_label']}\nRank {int(row['rank'])}: {row['model_label']}", pad=5)
                panel += 1
        save_all(fig, f"Figure_LAOS_best3_by_feature_set_{target_id}_no_LVR")


def p_text(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def plot_correlation_figure(data: pd.DataFrame) -> None:
    train = data[data["split"] == "train"].copy()
    pairs = [
        {
            "x": "log10_Gp_1Hz",
            "y": "log10_break_stress_Pa",
            "x_label": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
            "y_label": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
            "title": r"$\sigma_\mathrm{break}$ vs $G'$ 1 Hz",
            "color": "#5F8D6A",
        },
        {
            "x": "log10_eta_50",
            "y": "log10_break_stress_Pa",
            "x_label": r"log$_{10}$ $\eta_{50}$ (Pa s)",
            "y_label": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
            "title": r"$\sigma_\mathrm{break}$ vs $\eta_{50}$",
            "color": "#5F8D6A",
        },
        {
            "x": "log10_Gp_1Hz",
            "y": "break_strain_pct",
            "x_label": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
            "y_label": r"$\gamma_\mathrm{break}$ (%)",
            "title": r"$\gamma_\mathrm{break}$ vs $G'$ 1 Hz",
            "color": "#A66B55",
        },
        {
            "x": "log10_eta_50",
            "y": "break_strain_pct",
            "x_label": r"log$_{10}$ $\eta_{50}$ (Pa s)",
            "y_label": r"$\gamma_\mathrm{break}$ (%)",
            "title": r"$\gamma_\mathrm{break}$ vs $\eta_{50}$",
            "color": "#A66B55",
        },
    ]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.085, top=0.93, wspace=0.34, hspace=0.48)
    for idx, (ax, spec) in enumerate(zip(axes.ravel(), pairs)):
        d = train[[spec["x"], spec["y"]]].replace([np.inf, -np.inf], np.nan).dropna()
        x = d[spec["x"]].to_numpy(dtype=float)
        y = d[spec["y"]].to_numpy(dtype=float)
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        y_fit = slope * x + intercept
        residual = y - y_fit
        dof = max(len(x) - 2, 1)
        s_err = np.sqrt(np.sum(residual**2) / dof)
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        ci = 1.96 * s_err * np.sqrt(1 / len(x) + (x_line - x_mean) ** 2 / max(sxx, 1e-12))
        ax.scatter(x, y, s=34, color=spec["color"], edgecolor="#303030", linewidth=0.35, alpha=0.9, zorder=3)
        ax.plot(x_line, y_line, color="#C4453C", linewidth=0.9, zorder=2)
        ax.fill_between(x_line, y_line - ci, y_line + ci, color=spec["color"], alpha=0.14, linewidth=0)
        if spec["y"] == "break_strain_pct":
            y_top = max(float(y.max()) * 1.08, 1)
            ax.set_ylim(0, y_top)
        ax.grid(color="#D9D9D9", linewidth=0.45, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(spec["x_label"], labelpad=2)
        ax.set_ylabel(spec["y_label"], labelpad=2)
        ax.set_title(spec["title"], pad=5)
        ax.text(-0.24, 1.13, chr(ord("A") + idx), transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top")
        stat_y = 0.94 if spec["y"] != "break_strain_pct" else 0.06
        stat_va = "top" if spec["y"] != "break_strain_pct" else "bottom"
        ax.text(
            0.04,
            stat_y,
            f"Pearson r = {pearson_r:.2f}\n{p_text(pearson_p)}\nSpearman rho = {spearman_rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va=stat_va,
            fontsize=7.1,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
        )
        rows.append(
            {
                "x": spec["x"],
                "y": spec["y"],
                "n": len(d),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "linear_slope": slope,
                "linear_intercept": intercept,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "breaking_laos_descriptor_correlations.csv", index=False)
    save_all(fig, "Figure_correlation_breaking_targets_vs_Gp1Hz_eta50")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    data = formulation_level_master()
    metrics, preds = run_models(data)
    metrics.to_csv(OUT_DIR / "paper_style_model_metrics_no_LVR.csv", index=False)
    preds.to_csv(OUT_DIR / "paper_style_model_predictions_no_LVR.csv", index=False)
    plot_main_best3(metrics, preds)
    plot_laos_feature_set_model_panels(metrics, preds)
    plot_laos_best3_by_feature_set(metrics, preds)
    plot_correlation_figure(data)
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)
    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "note": "Paper-style parity figures. LVR removed; LAOS targets include only breaking stress and breaking strain.",
        "outputs": [
            "Figure_paper_style_best3_parity_no_LVR.png/pdf/tiff",
            "Figure_LAOS_best3_by_feature_set_no_LVR.png/pdf/tiff",
            "LAOS_parity_by_model_no_LVR/parity_external_<model>_no_LVR.png/pdf/tiff",
            "Figure_correlation_breaking_targets_vs_Gp1Hz_eta50.png/pdf/tiff",
            "paper_style_model_metrics_no_LVR.csv",
            "paper_style_model_predictions_no_LVR.csv",
            "breaking_laos_descriptor_correlations.csv",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
