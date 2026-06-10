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
RUN_ID = os.environ.get("RHEOLOGY_LAOS_FEATURE_PARITY_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"LAOS_feature_set_parity_comparison_{RUN_ID}"
DPI = 600

FEATURE_SETS = [
    {
        "id": "formulation_Gp1Hz",
        "label": r"formulation + $G'$ 1 Hz",
        "features": ["yp_pct", "xanthan_pct", "log10_Gp_1Hz"],
    },
    {
        "id": "formulation_eta50",
        "label": r"formulation + $\eta_{50}$",
        "features": ["yp_pct", "xanthan_pct", "log10_eta_50"],
    },
    {
        "id": "formulation_Gp1Hz_eta50",
        "label": r"formulation + $G'$ 1 Hz + $\eta_{50}$",
        "features": ["yp_pct", "xanthan_pct", "log10_Gp_1Hz", "log10_eta_50"],
    },
]

TARGETS = [
    {
        "id": "break_strain",
        "target": "break_strain_pct",
        "title": "Breaking strain",
        "axis": r"$\gamma_\mathrm{break}$ (%)",
    },
    {
        "id": "break_stress",
        "target": "log10_break_stress_Pa",
        "title": "Breaking stress",
        "axis": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
    },
    {
        "id": "LVR",
        "target": "LVR_pct",
        "title": "LVR",
        "axis": "LVR (%)",
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

MODEL_COLORS = {
    "GPR_Matern_ARD": "#4C78A8",
    "KernelRidge_RBF": "#5F8D6A",
    "SVR_RBF": "#7C6A9B",
    "Ridge": "#8A8A8A",
    "XGBoost": "#B279A2",
    "ExtraTrees": "#E17C05",
    "RandomForest": "#54A24B",
    "GradientBoosting": "#A66B55",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.85,
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


def run_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    for fset in FEATURE_SETS:
        features = fset["features"]
        models = make_models(len(features))
        for target_spec in TARGETS:
            target = target_spec["target"]
            needed = ["split", "formulation_std", target, *features]
            d = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
            train = d[d["split"] == "train"].reset_index(drop=True)
            test = d[d["split"] == "predict"].reset_index(drop=True)
            for model_name, base_model in models.items():
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
                    for local_idx, y_hat in zip(va_idx, pred):
                        pred_rows.append(
                            {
                                "feature_set_id": fset["id"],
                                "feature_set_label": fset["label"],
                                "features": ",".join(features),
                                "target_id": target_spec["id"],
                                "target": target,
                                "model": model_name,
                                "model_label": MODEL_LABELS[model_name],
                                "validation": "Internal 5-fold CV",
                                "fold": fold,
                                "formulation_std": train.loc[local_idx, "formulation_std"],
                                "y_true": train.loc[local_idx, target],
                                "y_pred": float(y_hat),
                            }
                        )
                metric_rows.append(
                    {
                        "feature_set_id": fset["id"],
                        "feature_set_label": fset["label"],
                        "features": ",".join(features),
                        "target_id": target_spec["id"],
                        "target": target,
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
                        "feature_set_id": fset["id"],
                        "feature_set_label": fset["label"],
                        "features": ",".join(features),
                        "target_id": target_spec["id"],
                        "target": target,
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "validation": "External validation",
                        **metric_row(test[target].to_numpy(dtype=float), pred),
                    }
                )
                for idx, y_hat in enumerate(pred):
                    pred_rows.append(
                        {
                            "feature_set_id": fset["id"],
                            "feature_set_label": fset["label"],
                            "features": ",".join(features),
                            "target_id": target_spec["id"],
                            "target": target,
                            "model": model_name,
                            "model_label": MODEL_LABELS[model_name],
                            "validation": "External validation",
                            "fold": np.nan,
                            "formulation_std": test.loc[idx, "formulation_std"],
                            "y_true": test.loc[idx, target],
                            "y_pred": float(y_hat),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def fmt(value: float, metric: str) -> str:
    if pd.isna(value):
        return ""
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


def save_all(fig: plt.Figure, subdir: Path, stem: str) -> None:
    subdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(subdir / f"{stem}.png", dpi=DPI)
    fig.savefig(subdir / f"{stem}.pdf")
    fig.savefig(subdir / f"{stem}.tiff", dpi=DPI)
    plt.close(fig)


def annotate(ax: plt.Axes, row: pd.Series) -> None:
    ax.text(
        0.04,
        0.96,
        f"$R^2$ = {row['r2']:.2f}\nMAE = {fmt(row['mae'], 'mae')}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#C9C9C9", "alpha": 0.9, "pad": 3},
    )


def plot_model_parity(metrics: pd.DataFrame, preds: pd.DataFrame) -> None:
    out = OUT_DIR / "parity_by_model"
    for model_name in MODEL_ORDER:
        if model_name not in set(preds["model"]):
            continue
        fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.6), constrained_layout=True)
        color = MODEL_COLORS[model_name]
        for r, target_spec in enumerate(TARGETS):
            for c, fset in enumerate(FEATURE_SETS):
                ax = axes[r, c]
                d = preds[
                    (preds["validation"] == "External validation")
                    & (preds["model"] == model_name)
                    & (preds["feature_set_id"] == fset["id"])
                    & (preds["target_id"] == target_spec["id"])
                ]
                m = metrics[
                    (metrics["validation"] == "External validation")
                    & (metrics["model"] == model_name)
                    & (metrics["feature_set_id"] == fset["id"])
                    & (metrics["target_id"] == target_spec["id"])
                ].iloc[0]
                lo, hi = panel_limits(d)
                ax.scatter(d["y_true"], d["y_pred"], s=46, color=color, edgecolor="white", linewidth=0.6, alpha=0.9)
                ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
                if len(d) >= 2:
                    slope, intercept = np.polyfit(d["y_true"], d["y_pred"], 1)
                    xx = np.array([lo, hi])
                    ax.plot(xx, slope * xx + intercept, color="#444444", linestyle=(0, (4, 3)), linewidth=0.9)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.grid(color="#D9D9D9", linewidth=0.65, alpha=0.6)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                annotate(ax, m)
                if r == 0:
                    ax.set_title(fset["label"])
                if c == 0:
                    ax.set_ylabel(f"Predicted {target_spec['axis']}")
                if r == 2:
                    ax.set_xlabel(f"Measured {target_spec['axis']}")
                else:
                    ax.set_xlabel("")
                ax.text(0.98, 0.05, target_spec["title"], transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
        fig.suptitle(f"External parity: {MODEL_LABELS[model_name]}", fontsize=14)
        save_all(fig, out, f"parity_external_{model_name}")


def best_three(metrics: pd.DataFrame) -> pd.DataFrame:
    d = metrics[metrics["validation"] == "External validation"].copy()
    d["rank_score"] = d.groupby("target_id")["r2"].rank(ascending=False, method="min") + d.groupby("target_id")[
        "rmse"
    ].rank(ascending=True, method="min")
    d = d.sort_values(["target_id", "rank_score", "rmse", "mae"])
    d["rank"] = d.groupby("target_id").cumcount() + 1
    return d[d["rank"] <= 3].copy()


def plot_best_three(metrics: pd.DataFrame, preds: pd.DataFrame) -> None:
    best = best_three(metrics)
    best.to_csv(OUT_DIR / "best_three_external_model_feature_sets_by_target.csv", index=False)
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.8), constrained_layout=True)
    for r, target_spec in enumerate(TARGETS):
        rows = best[best["target_id"] == target_spec["id"]].sort_values("rank")
        for c in range(3):
            ax = axes[r, c]
            row = rows.iloc[c]
            d = preds[
                (preds["validation"] == "External validation")
                & (preds["model"] == row["model"])
                & (preds["feature_set_id"] == row["feature_set_id"])
                & (preds["target_id"] == row["target_id"])
            ]
            lo, hi = panel_limits(d)
            ax.scatter(
                d["y_true"],
                d["y_pred"],
                s=52,
                color=MODEL_COLORS[row["model"]],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.92,
            )
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
            if len(d) >= 2:
                slope, intercept = np.polyfit(d["y_true"], d["y_pred"], 1)
                xx = np.array([lo, hi])
                ax.plot(xx, slope * xx + intercept, color="#444444", linestyle=(0, (4, 3)), linewidth=0.9)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(color="#D9D9D9", linewidth=0.65, alpha=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f"#{int(row['rank'])} {row['model_label']}\n{row['feature_set_label']}", fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"Predicted {target_spec['axis']}")
            if r == 2:
                ax.set_xlabel(f"Measured {target_spec['axis']}")
            annotate(ax, row)
    fig.suptitle("Best three external parity plots for each LAOS target", fontsize=14)
    save_all(fig, OUT_DIR, "Figure_best_three_external_parity_by_LAOS_target")


def plot_feature_set_summary(metrics: pd.DataFrame) -> None:
    external = metrics[metrics["validation"] == "External validation"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    for ax, target_spec in zip(axes, TARGETS):
        d = external[external["target_id"] == target_spec["id"]].copy()
        models = [MODEL_LABELS[m] for m in MODEL_ORDER if m in set(d["model"])]
        fsets = [f["id"] for f in FEATURE_SETS]
        values = d.pivot_table(index="model_label", columns="feature_set_id", values="r2").reindex(index=models, columns=fsets)
        im = ax.imshow(values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_title(target_spec["title"])
        ax.set_xticks(np.arange(len(fsets)))
        ax.set_xticklabels([f["label"] for f in FEATURE_SETS], rotation=28, ha="right")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values.iloc[i, j]
                ax.text(j, i, fmt(val, "r2"), ha="center", va="center", fontsize=8.0, color="white" if val >= 0.68 else "black")
        ax.set_xticks(np.arange(-0.5, len(fsets), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.012)
    cbar.set_label(r"External $R^2$")
    save_all(fig, OUT_DIR, "Figure_external_R2_feature_set_summary")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    data = formulation_level_master()
    metrics, preds = run_models(data)
    metrics.to_csv(OUT_DIR / "laos_feature_set_model_metrics.csv", index=False)
    preds.to_csv(OUT_DIR / "laos_feature_set_model_predictions.csv", index=False)
    plot_model_parity(metrics, preds)
    plot_best_three(metrics, preds)
    plot_feature_set_summary(metrics)
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)
    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "feature_sets": FEATURE_SETS,
        "targets": TARGETS,
        "models": [MODEL_LABELS[m] for m in MODEL_ORDER if m in set(metrics["model"])],
        "outputs": [
            "parity_by_model/parity_external_<model>.png/pdf/tiff",
            "Figure_best_three_external_parity_by_LAOS_target.png/pdf/tiff",
            "Figure_external_R2_feature_set_summary.png/pdf/tiff",
            "laos_feature_set_model_metrics.csv",
            "laos_feature_set_model_predictions.csv",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
