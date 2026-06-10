from __future__ import annotations

import json
import math
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


ROOT = Path("/Users/zhiy/Documents/Rheology ML")
DATA_DIR = ROOT / "outputs" / "ml_ready_xanthan_positive_20260529"
ONEDRIVE_ROOT = Path("/Users/zhiy/Library/CloudStorage/OneDrive-Personal/GPR new")
RUN_ID = os.environ.get("RHEOLOGY_FORMULATION_LAOS_SUITE_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = Path(os.environ.get("RHEOLOGY_OUTPUT_ROOT", ONEDRIVE_ROOT if ONEDRIVE_ROOT.exists() else ROOT / "outputs"))
OUT_DIR = OUTPUT_ROOT / "time_lapse" / f"publication_formulation_laos_model_suite_{RUN_ID}"
DPI = 600
INTERNAL_LABEL = "Internal balanced 5-fold CV"
EXTERNAL_LABEL = "External validation"

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
    "Ridge": "#7F7F7F",
    "XGBoost": "#B279A2",
    "ExtraTrees": "#E17C05",
    "RandomForest": "#54A24B",
    "GradientBoosting": "#A66B55",
}

FORMULATION_FEATURES = ["yp_pct", "xanthan_pct"]

FORMULATION_TARGETS = [
    {
        "id": "Gp1Hz",
        "target": "log10_Gp_1Hz",
        "title": r"$G'$ at 1 Hz",
        "short": r"$G'$ 1 Hz",
        "axis": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
    },
    {
        "id": "eta50",
        "target": "log10_eta_50",
        "title": r"$\eta_{50}$",
        "short": r"$\eta_{50}$",
        "axis": r"log$_{10}$ $\eta_{50}$ (Pa s)",
    },
    {
        "id": "break_stress",
        "target": "log10_break_stress_Pa",
        "title": "Breaking stress",
        "short": r"$\sigma_\mathrm{break}$",
        "axis": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
    },
    {
        "id": "break_strain",
        "target": "break_strain_pct",
        "title": "Breaking strain",
        "short": r"$\gamma_\mathrm{break}$",
        "axis": r"$\gamma_\mathrm{break}$ (%)",
    },
]

LAOS_FEATURE_SETS = [
    {
        "id": "formulation",
        "label": "formulation only",
        "features": ["yp_pct", "xanthan_pct"],
    },
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

LAOS_BEST_THREE_FEATURE_SETS = [fset for fset in LAOS_FEATURE_SETS if fset["id"] != "formulation"]

LAOS_TARGETS = [
    {
        "id": "break_stress",
        "target": "log10_break_stress_Pa",
        "title": "LAOS breaking stress",
        "short": r"$\sigma_\mathrm{break}$",
        "axis": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
    },
    {
        "id": "break_strain",
        "target": "break_strain_pct",
        "title": "LAOS breaking strain",
        "short": r"$\gamma_\mathrm{break}$",
        "axis": r"$\gamma_\mathrm{break}$ (%)",
    },
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 11.4,
            "axes.titlesize": 11.2,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 10.2,
            "axes.linewidth": 0.85,
            "savefig.bbox": "tight",
        }
    )


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def fmt(value: float, metric: str) -> str:
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
            verbosity=0,
        )
    return {name: models[name] for name in MODEL_ORDER if name in models}


def formulation_level_master() -> pd.DataFrame:
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    for freq in ["0.1", "1", "6.31"]:
        gp = f"Gp_{freq}Hz_Pa"
        gpp = f"Gpp_{freq}Hz_Pa"
        if gp in rep.columns:
            rep[f"log10_Gp_{freq}Hz"] = np.where(rep[gp] > 0, np.log10(rep[gp]), np.nan)
        if gpp in rep.columns:
            rep[f"log10_Gpp_{freq}Hz"] = np.where(rep[gpp] > 0, np.log10(rep[gpp]), np.nan)
    rep["log10_break_stress_Pa"] = np.where(rep["break_stress_Pa"] > 0, np.log10(rep["break_stress_Pa"]), np.nan)
    numeric_cols = rep.select_dtypes(include=[np.number]).columns.tolist()
    return rep.groupby(["split", "formulation_std"], as_index=False)[numeric_cols].mean()


def balanced_grid_splits(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
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
        raise ValueError("Balanced CV expects one training formulation at every xanthan x protein grid point.")
    all_idx = np.asarray(train.index.tolist())
    return [
        (
            np.asarray([idx for idx in all_idx if fold_by_index[idx] != fold]),
            np.asarray([idx for idx in all_idx if fold_by_index[idx] == fold]),
        )
        for fold in range(5)
    ]


def run_task_family(
    data: pd.DataFrame,
    family: str,
    target_specs: list[dict],
    feature_sets: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    fold_rows = []

    for fset in feature_sets:
        features = fset["features"]
        models = make_models(len(features))
        for spec in target_specs:
            target = spec["target"]
            grid_cols = [c for c in ["yp_pct", "xanthan_pct"] if c not in features]
            needed = ["split", "formulation_std", target, *features, *grid_cols]
            d = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
            train = d[d["split"] == "train"].sort_values(["xanthan_pct", "yp_pct"]).reset_index(drop=True)
            test = d[d["split"] == "predict"].sort_values(["xanthan_pct", "yp_pct"]).reset_index(drop=True)
            splits = balanced_grid_splits(train)

            for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
                for role, idxs in [("training", tr_idx), ("validation", va_idx)]:
                    for _, row in train.iloc[idxs].iterrows():
                        fold_rows.append(
                            {
                                "family": family,
                                "feature_set_id": fset["id"],
                                "feature_set_label": fset["label"],
                                "features": ",".join(features),
                                "target_id": spec["id"],
                                "target": target,
                                "fold": fold,
                                "role": role,
                                "formulation_std": row["formulation_std"],
                                "yp_pct": row["yp_pct"],
                                "xanthan_pct": row["xanthan_pct"],
                            }
                        )

            for model_name, base_model in models.items():
                y_cv_true = []
                y_cv_pred = []
                for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
                    model = clone(base_model)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(train.iloc[tr_idx][features], train.iloc[tr_idx][target])
                    pred = np.asarray(model.predict(train.iloc[va_idx][features]), dtype=float)
                    y_true = train.iloc[va_idx][target].to_numpy(dtype=float)
                    y_cv_true.extend(y_true.tolist())
                    y_cv_pred.extend(pred.tolist())
                    for local_idx, y_hat in zip(va_idx, pred):
                        row = train.loc[local_idx]
                        pred_rows.append(
                            {
                                "family": family,
                                "feature_set_id": fset["id"],
                                "feature_set_label": fset["label"],
                                "features": ",".join(features),
                                "target_id": spec["id"],
                                "target": target,
                                "target_title": spec["title"],
                                "model": model_name,
                                "model_label": model_label(model_name),
                                "validation": INTERNAL_LABEL,
                                "split": "internal_balanced_cv",
                                "fold": fold,
                                "formulation_std": row["formulation_std"],
                                "y_true": row[target],
                                "y_pred": float(y_hat),
                                "yp_pct": row["yp_pct"],
                                "xanthan_pct": row["xanthan_pct"],
                            }
                        )
                metric_rows.append(
                    {
                        "family": family,
                        "feature_set_id": fset["id"],
                        "feature_set_label": fset["label"],
                        "features": ",".join(features),
                        "target_id": spec["id"],
                        "target": target,
                        "target_title": spec["title"],
                        "model": model_name,
                        "model_label": model_label(model_name),
                        "validation": INTERNAL_LABEL,
                        **metric_row(np.asarray(y_cv_true), np.asarray(y_cv_pred)),
                    }
                )

                final = clone(base_model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    final.fit(train[features], train[target])
                external_pred = np.asarray(final.predict(test[features]), dtype=float)
                metric_rows.append(
                    {
                        "family": family,
                        "feature_set_id": fset["id"],
                        "feature_set_label": fset["label"],
                        "features": ",".join(features),
                        "target_id": spec["id"],
                        "target": target,
                        "target_title": spec["title"],
                        "model": model_name,
                        "model_label": model_label(model_name),
                        "validation": EXTERNAL_LABEL,
                        **metric_row(test[target].to_numpy(dtype=float), external_pred),
                    }
                )
                for local_idx, y_hat in enumerate(external_pred):
                    row = test.loc[local_idx]
                    pred_rows.append(
                        {
                            "family": family,
                            "feature_set_id": fset["id"],
                            "feature_set_label": fset["label"],
                            "features": ",".join(features),
                            "target_id": spec["id"],
                            "target": target,
                            "target_title": spec["title"],
                            "model": model_name,
                            "model_label": model_label(model_name),
                            "validation": EXTERNAL_LABEL,
                            "split": "external_validation",
                            "fold": np.nan,
                            "formulation_std": row["formulation_std"],
                            "y_true": row[target],
                            "y_pred": float(y_hat),
                            "yp_pct": row["yp_pct"],
                            "xanthan_pct": row["xanthan_pct"],
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def panel_limits(d: pd.DataFrame) -> tuple[float, float]:
    lo = float(np.nanmin([d["y_true"].min(), d["y_pred"].min()]))
    hi = float(np.nanmax([d["y_true"].max(), d["y_pred"].max()]))
    pad = (hi - lo) * 0.08 if hi > lo else max(abs(hi) * 0.08, 1.0)
    return lo - pad, hi + pad


def annotate_parity(ax: plt.Axes, d: pd.DataFrame, m: pd.Series | None, fontsize: float = 16.2, pad: float = 3) -> None:
    if len(d) >= 2 and d["y_true"].nunique() > 1:
        slope, intercept = np.polyfit(d["y_true"].astype(float), d["y_pred"].astype(float), 1)
        text = f"fit: y = {slope:.2f}x + {intercept:.2f}"
    else:
        text = "fit: n/a"
    if m is not None:
        text += f"\n$R^2$ = {m['r2']:.2f}\nRMSE = {fmt(m['rmse'], 'rmse')}\nMAE = {fmt(m['mae'], 'mae')}"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "#C9C9C9", "alpha": 0.9, "pad": pad},
    )


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.13,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=25,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def draw_parity_panel(
    ax: plt.Axes,
    d: pd.DataFrame,
    m: pd.Series | None,
    color: str,
    axis_label: str,
    annotation_fontsize: float = 16.2,
    annotation_pad: float = 3,
) -> None:
    lo, hi = panel_limits(d)
    ax.scatter(d["y_true"], d["y_pred"], s=108, color=color, edgecolor="white", linewidth=1.0, alpha=0.92, zorder=3)
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=(0, (4, 3)), linewidth=1.45, zorder=2)
    if len(d) >= 2 and d["y_true"].nunique() > 1:
        slope, intercept = np.polyfit(d["y_true"].astype(float), d["y_pred"].astype(float), 1)
        xx = np.asarray([lo, hi])
        ax.plot(xx, slope * xx + intercept, color="#B45A4D", linewidth=1.65, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Measured {axis_label}", fontsize=16.0, fontweight="bold")
    ax.set_ylabel(f"Predicted {axis_label}", fontsize=16.0, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14.5)
    annotate_parity(ax, d, m, fontsize=annotation_fontsize, pad=annotation_pad)
    ax.grid(color="#D9D9D9", linewidth=0.65, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_bland_altman_panel(ax: plt.Axes, d: pd.DataFrame, m: pd.Series | None, color: str, axis_label: str) -> None:
    mean = (d["y_true"].to_numpy(dtype=float) + d["y_pred"].to_numpy(dtype=float)) / 2
    diff = d["y_pred"].to_numpy(dtype=float) - d["y_true"].to_numpy(dtype=float)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    ax.scatter(mean, diff, s=108, color=color, edgecolor="white", linewidth=1.0, alpha=0.92, zorder=3)
    ax.axhline(0, color="black", linewidth=1.2, linestyle=":", label="zero difference")
    ax.axhline(bias, color="#B45A4D", linewidth=1.5, label="bias")
    ax.axhline(loa_low, color="#555555", linewidth=1.2, linestyle=(0, (4, 3)), label="95% limits")
    ax.axhline(loa_high, color="#555555", linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_xlabel(f"Mean measured/predicted {axis_label}", fontsize=16.0, fontweight="bold")
    ax.set_ylabel("Predicted - measured", fontsize=16.0, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14.5)
    text = f"bias = {fmt(bias, 'rmse')}\n95% LoA: {fmt(loa_low, 'rmse')} to {fmt(loa_high, 'rmse')}"
    if m is not None:
        text += f"\nRMSE = {fmt(m['rmse'], 'rmse')}"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16.2,
        bbox={"facecolor": "white", "edgecolor": "#C9C9C9", "alpha": 0.9, "pad": 3},
    )
    ax.legend(loc="lower right", frameon=False, fontsize=14.5)
    ax.grid(color="#D9D9D9", linewidth=0.65, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.tiff", dpi=DPI)
    plt.close(fig)


def metric_for_panel(metrics: pd.DataFrame, selector: dict) -> pd.Series | None:
    d = metrics.copy()
    for key, value in selector.items():
        d = d[d[key] == value]
    if d.empty:
        return None
    return d.iloc[0]


def plot_all_model_figures(
    metrics: pd.DataFrame,
    preds: pd.DataFrame,
    family: str,
    target_specs: list[dict],
    feature_sets: list[dict],
) -> None:
    validations = [INTERNAL_LABEL, EXTERNAL_LABEL]
    for model in [m for m in MODEL_ORDER if m in set(preds["model"])]:
        for plot_kind in ["parity", "bland_altman"]:
            nrows, ncols = 4, 4
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(18.4, 15.2),
                squeeze=False,
                constrained_layout=True,
            )
            axes_flat = axes.ravel()
            panel_idx = 0
            letters = list("ABCDEFGHIJKLMNOP")
            for validation in validations:
                for spec in target_specs:
                    for fset in feature_sets:
                        ax = axes_flat[panel_idx]
                        d = preds[
                            (preds["family"] == family)
                            & (preds["validation"] == validation)
                            & (preds["model"] == model)
                            & (preds["target_id"] == spec["id"])
                            & (preds["feature_set_id"] == fset["id"])
                        ].copy()
                        if d.empty:
                            ax.axis("off")
                            panel_idx += 1
                            continue
                        m = metric_for_panel(
                            metrics,
                            {
                                "family": family,
                                "validation": validation,
                                "model": model,
                                "target_id": spec["id"],
                                "feature_set_id": fset["id"],
                            },
                        )
                        color = MODEL_COLORS.get(model, "#4C78A8")
                        if plot_kind == "parity":
                            draw_parity_panel(ax, d, m, color, spec["axis"])
                        else:
                            draw_bland_altman_panel(ax, d, m, color, spec["axis"])
                        add_panel_letter(ax, letters[panel_idx])
                        if panel_idx < ncols:
                            ax.set_title(fset["label"], fontsize=10.4)
                        panel_idx += 1
            for ax in axes_flat[panel_idx:]:
                ax.axis("off")
            fig.suptitle(
                f"{family.capitalize()} {plot_kind.replace('_', '-')} plots: {model_label(model)}",
                fontsize=14.5,
            )
            subdir = OUT_DIR / family / f"all_model_{plot_kind}"
            save_all(fig, subdir, f"{family}_internal_external_{plot_kind}_{model}")


def plot_formulation_by_target_all_models(
    metrics: pd.DataFrame,
    preds: pd.DataFrame,
    target_specs: list[dict],
    feature_set_id: str = "formulation",
) -> None:
    models = [m for m in MODEL_ORDER if m in set(preds[(preds["family"] == "formulation")]["model"])]
    validations = [INTERNAL_LABEL, EXTERNAL_LABEL]
    for plot_kind in ["parity", "bland_altman"]:
        out_dir = OUT_DIR / "formulation" / f"{plot_kind}_by_target_all_models"
        for spec in target_specs:
            fig, axes = plt.subplots(4, 4, figsize=(26.0, 21.5), squeeze=False, constrained_layout=True)
            axes_flat = axes.ravel()
            panel_idx = 0
            letters = list("ABCDEFGHIJKLMNOP")
            for validation in validations:
                for model in models:
                    ax = axes_flat[panel_idx]
                    d = preds[
                        (preds["family"] == "formulation")
                        & (preds["validation"] == validation)
                        & (preds["model"] == model)
                        & (preds["target_id"] == spec["id"])
                        & (preds["feature_set_id"] == feature_set_id)
                    ].copy()
                    if d.empty:
                        ax.axis("off")
                        panel_idx += 1
                        continue
                    m = metric_for_panel(
                        metrics,
                        {
                            "family": "formulation",
                            "validation": validation,
                            "model": model,
                            "target_id": spec["id"],
                            "feature_set_id": feature_set_id,
                        },
                    )
                    if plot_kind == "parity":
                        draw_parity_panel(ax, d, m, MODEL_COLORS.get(model, "#4C78A8"), spec["axis"])
                    else:
                        draw_bland_altman_panel(ax, d, m, MODEL_COLORS.get(model, "#4C78A8"), spec["axis"])
                    add_panel_letter(ax, letters[panel_idx])
                    ax.set_title(model_label(model), fontsize=17.0, fontweight="bold")
                    panel_idx += 1
            fig.suptitle(f"Formulation-only {plot_kind.replace('_', '-')} plots: {spec['title']} ({spec['short']})", fontsize=25.0, fontweight="bold")
            save_all(fig, out_dir, f"formulation_internal_external_{plot_kind}_all_models_{spec['id']}")


def plot_best_three_parity(
    metrics: pd.DataFrame,
    preds: pd.DataFrame,
    family: str,
    target_specs: list[dict],
    validation: str = EXTERNAL_LABEL,
) -> pd.DataFrame:
    d = metrics[(metrics["family"] == family) & (metrics["validation"] == validation)].copy()
    d = d.sort_values(["target_id", "r2", "rmse"], ascending=[True, False, True])
    d["rank"] = d.groupby("target_id").cumcount() + 1
    best = d[d["rank"] <= 3].copy()
    best.to_csv(OUT_DIR / family / f"{family}_best_three_{validation.replace(' ', '_').lower()}.csv", index=False)

    fig, axes = plt.subplots(len(target_specs), 3, figsize=(16.2, 5.35 * len(target_specs)), squeeze=False, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.055, top=0.965, wspace=0.28, hspace=0.48)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for r, spec in enumerate(target_specs):
        rows = best[best["target_id"] == spec["id"]].sort_values("rank")
        for c in range(3):
            ax = axes[r, c]
            row = rows.iloc[c]
            p = preds[
                (preds["family"] == family)
                & (preds["validation"] == validation)
                & (preds["target_id"] == row["target_id"])
                & (preds["feature_set_id"] == row["feature_set_id"])
                & (preds["model"] == row["model"])
            ].copy()
            draw_parity_panel(
                ax,
                p,
                row,
                MODEL_COLORS.get(row["model"], "#4C78A8"),
                spec["axis"],
                annotation_fontsize=13.0,
                annotation_pad=2,
            )
            ax.set_title(f"Rank {int(row['rank'])}: {row['model_label']}\n{row['feature_set_label']}", fontsize=15.0, fontweight="bold")
            add_panel_letter(ax, letters[r * 3 + c])
    save_all(fig, OUT_DIR / family, f"Figure_{family}_best_three_{validation.replace(' ', '_').lower()}_parity")
    return best


def plot_best_three_by_feature_combination_parity(
    metrics: pd.DataFrame,
    preds: pd.DataFrame,
    family: str,
    target_specs: list[dict],
    feature_sets: list[dict],
    validation: str = EXTERNAL_LABEL,
) -> pd.DataFrame:
    d = metrics[(metrics["family"] == family) & (metrics["validation"] == validation)].copy()
    included_feature_sets = {fset["id"] for fset in feature_sets}
    d = d[d["feature_set_id"].isin(included_feature_sets)].copy()
    d = d.sort_values(["feature_set_id", "target_id", "r2", "rmse"], ascending=[True, True, False, True])
    d["rank"] = d.groupby(["feature_set_id", "target_id"]).cumcount() + 1
    best = d[d["rank"] <= 3].copy()
    validation_stem = validation.replace(" ", "_").lower()
    best.to_csv(OUT_DIR / family / f"{family}_best_three_by_feature_combination_{validation_stem}.csv", index=False)

    def draw_best_grid(specs: list[dict], stem_suffix: str) -> None:
        combos = [(fset, spec) for spec in specs for fset in feature_sets]
        fig, axes = plt.subplots(len(combos), 3, figsize=(12.3, 3.75 * len(combos)), squeeze=False, constrained_layout=True)
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for r, (fset, spec) in enumerate(combos):
            rows = best[(best["feature_set_id"] == fset["id"]) & (best["target_id"] == spec["id"])].sort_values("rank")
            for c in range(3):
                ax = axes[r, c]
                row = rows.iloc[c]
                p = preds[
                    (preds["family"] == family)
                    & (preds["validation"] == validation)
                    & (preds["target_id"] == row["target_id"])
                    & (preds["feature_set_id"] == row["feature_set_id"])
                    & (preds["model"] == row["model"])
                ].copy()
                draw_parity_panel(ax, p, row, MODEL_COLORS.get(row["model"], "#4C78A8"), spec["axis"])
                ax.set_title(f"{fset['label']}\n{spec['short']} rank {int(row['rank'])}: {row['model_label']}", fontsize=10.2)
                ax.text(
                    -0.13,
                    1.08,
                    letters[r * 3 + c],
                    transform=ax.transAxes,
                    fontsize=15,
                    fontweight="bold",
                    va="top",
                    ha="left",
                )
        save_all(fig, OUT_DIR / family, f"Figure_{family}_best_three_by_feature_combination_{validation_stem}_{stem_suffix}_parity")

    if validation == EXTERNAL_LABEL and len(target_specs) > 1:
        for spec in target_specs:
            draw_best_grid([spec], spec["id"])
    else:
        draw_best_grid(target_specs, "all_targets")
    return best


def select_best_feature_per_model(metrics: pd.DataFrame, family: str) -> pd.DataFrame:
    d = metrics[metrics["family"] == family].copy()
    d = d.sort_values(["validation", "target_id", "model", "r2", "rmse"], ascending=[True, True, True, False, True])
    return d.groupby(["validation", "target_id", "model"], as_index=False).head(1)


def plot_grouped_r2_rmse_bars(
    metrics: pd.DataFrame,
    family: str,
    target_specs: list[dict],
    use_best_feature_set: bool,
) -> pd.DataFrame:
    d = select_best_feature_per_model(metrics, family) if use_best_feature_set else metrics[metrics["family"] == family].copy()
    d.to_csv(OUT_DIR / family / f"{family}_bar_chart_source_metrics.csv", index=False)
    fig, axes = plt.subplots(2, len(target_specs), figsize=(4.25 * len(target_specs), 8.0), squeeze=False, constrained_layout=True)
    validation_colors = {INTERNAL_LABEL: "#6E8FB2", EXTERNAL_LABEL: "#C58B57"}
    width = 0.38
    for col, spec in enumerate(target_specs):
        target_data = d[d["target_id"] == spec["id"]].copy()
        models = [m for m in MODEL_ORDER if m in set(target_data["model"])]
        x = np.arange(len(models))
        for row_idx, metric in enumerate(["r2", "rmse"]):
            ax = axes[row_idx, col]
            values_by_validation = {}
            for validation in [INTERNAL_LABEL, EXTERNAL_LABEL]:
                vals = []
                for model in models:
                    sub = target_data[(target_data["validation"] == validation) & (target_data["model"] == model)]
                    vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
                values_by_validation[validation] = vals
            for offset, validation in [(-width / 2, INTERNAL_LABEL), (width / 2, EXTERNAL_LABEL)]:
                raw = values_by_validation[validation]
                draw = [max(v, 0) if metric == "r2" and np.isfinite(v) else v for v in raw]
                bars = ax.bar(
                    x + offset,
                    draw,
                    width=width,
                    label=validation,
                    color=validation_colors[validation],
                    edgecolor="white",
                    linewidth=0.7,
                )
                if any(np.isfinite(raw)):
                    best_idx = int(np.nanargmax(raw) if metric == "r2" else np.nanargmin(raw))
                    bars[best_idx].set_edgecolor("black")
                    bars[best_idx].set_linewidth(1.35)
                ymax = ax.get_ylim()[1]
                for bar, value, draw_value in zip(bars, raw, draw):
                    if not np.isfinite(value):
                        continue
                    label_y = (draw_value if np.isfinite(draw_value) else 0) + (0.015 if metric == "r2" else ymax * 0.018)
                    if metric == "r2" and value < 0:
                        label_y = 0.03
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_y,
                        fmt(value, metric),
                        ha="center",
                        va="bottom",
                        fontsize=7.4,
                        rotation=90 if metric == "rmse" or value < 0 else 0,
                        clip_on=False,
                    )
            if metric == "r2":
                ax.set_ylim(0, 1.05)
                ax.axhline(0.8, color="#555555", linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.75)
                ax.set_ylabel(r"$R^2$")
            else:
                finite_vals = [v for vals in values_by_validation.values() for v in vals if np.isfinite(v)]
                ax.set_ylim(0, max(finite_vals) * 1.25 if finite_vals else 1)
                ax.set_ylabel("RMSE")
            ax.set_title(spec["title"] if row_idx == 0 else "")
            ax.set_xticks(x)
            ax.set_xticklabels([model_label(m) for m in models], rotation=38, ha="right")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.65)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    note = "LAOS bars use each model's best feature set per target and validation." if use_best_feature_set else "Formulation bars use formulation-only predictors."
    fig.text(0.01, -0.005, note, ha="left", va="top", fontsize=9)
    save_all(fig, OUT_DIR / family, f"Figure_{family}_grouped_bar_R2_RMSE")
    return d


def plot_grouped_metric_bars_by_feature_set(
    metrics: pd.DataFrame,
    family: str,
    target_specs: list[dict],
    feature_sets: list[dict],
) -> pd.DataFrame:
    d = metrics[metrics["family"] == family].copy()
    d.to_csv(OUT_DIR / family / f"{family}_feature_combination_bar_chart_source_metrics.csv", index=False)
    validation_colors = {INTERNAL_LABEL: "#6E8FB2", EXTERNAL_LABEL: "#C58B57"}
    width = 0.38
    models = [m for m in MODEL_ORDER if m in set(d["model"])]
    x = np.arange(len(models))

    for metric in ["r2", "rmse"]:
        fig, axes = plt.subplots(
            len(target_specs),
            len(feature_sets),
            figsize=(6.15 * len(feature_sets), 5.45 * len(target_specs)),
            squeeze=False,
            constrained_layout=True,
        )
        for r, spec in enumerate(target_specs):
            for c, fset in enumerate(feature_sets):
                ax = axes[r, c]
                panel = d[(d["target_id"] == spec["id"]) & (d["feature_set_id"] == fset["id"])].copy()
                values_by_validation = {}
                for validation in [INTERNAL_LABEL, EXTERNAL_LABEL]:
                    vals = []
                    for model in models:
                        sub = panel[(panel["validation"] == validation) & (panel["model"] == model)]
                        vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
                    values_by_validation[validation] = vals

                for offset, validation in [(-width / 2, INTERNAL_LABEL), (width / 2, EXTERNAL_LABEL)]:
                    raw = values_by_validation[validation]
                    draw = [max(v, 0) if metric == "r2" and np.isfinite(v) else v for v in raw]
                    bars = ax.bar(
                        x + offset,
                        draw,
                        width=width,
                        label=validation,
                        color=validation_colors[validation],
                        edgecolor="white",
                        linewidth=0.7,
                    )
                    if any(np.isfinite(raw)):
                        best_idx = int(np.nanargmax(raw) if metric == "r2" else np.nanargmin(raw))
                        bars[best_idx].set_edgecolor("black")
                        bars[best_idx].set_linewidth(1.35)
                    ymax = ax.get_ylim()[1]
                    for bar, value, draw_value in zip(bars, raw, draw):
                        if not np.isfinite(value):
                            continue
                        label_y = (draw_value if np.isfinite(draw_value) else 0) + (0.018 if metric == "r2" else ymax * 0.020)
                        if metric == "r2" and value < 0:
                            label_y = 0.035
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            label_y,
                            fmt(value, metric),
                            ha="center",
                            va="bottom",
                            fontsize=12.5,
                            rotation=90 if metric == "rmse" or value < 0 else 0,
                            clip_on=False,
                        )

                if metric == "r2":
                    ax.set_ylim(0, 1.08)
                    ax.axhline(0.8, color="#555555", linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.75)
                    ax.set_ylabel(f"{spec['short']}\n" + r"$R^2$" if c == 0 else "", fontsize=17.0, fontweight="bold")
                else:
                    finite_vals = [v for vals in values_by_validation.values() for v in vals if np.isfinite(v)]
                    ax.set_ylim(0, max(finite_vals) * 1.28 if finite_vals else 1)
                    ax.set_ylabel(f"{spec['short']}\nRMSE" if c == 0 else "", fontsize=17.0, fontweight="bold")
                if r == 0:
                    ax.set_title(fset["label"], fontsize=16.0, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels([model_label(m) for m in models], rotation=38, ha="right", fontsize=13.5)
                ax.tick_params(axis="y", labelsize=13.5)
                ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.65)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.055),
            prop={"size": 16.0, "weight": "bold"},
        )
        save_all(fig, OUT_DIR / family, f"Figure_{family}_grouped_bar_{metric.upper()}_by_feature_combination")
    return d


def plot_laos_feature_set_heatmap(metrics: pd.DataFrame) -> None:
    d = metrics[(metrics["family"] == "laos") & (metrics["validation"] == EXTERNAL_LABEL)].copy()
    rows = []
    for target in [t["id"] for t in LAOS_TARGETS]:
        for fset in [f["id"] for f in LAOS_FEATURE_SETS]:
            sub = d[(d["target_id"] == target) & (d["feature_set_id"] == fset)]
            best = sub.sort_values(["r2", "rmse"], ascending=[False, True]).head(1)
            if len(best):
                row = best.iloc[0].to_dict()
                rows.append(row)
    best = pd.DataFrame(rows)
    best.to_csv(OUT_DIR / "laos" / "laos_external_best_model_by_target_feature_set.csv", index=False)
    values = best.pivot(index="feature_set_label", columns="target_id", values="r2").reindex(
        index=[f["label"] for f in LAOS_FEATURE_SETS],
        columns=[t["id"] for t in LAOS_TARGETS],
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    im = ax.imshow(values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(LAOS_TARGETS)))
    ax.set_xticklabels([t["short"] for t in LAOS_TARGETS])
    ax.set_yticks(np.arange(len(values.index)))
    ax.set_yticklabels(values.index)
    ax.set_xlabel("LAOS target")
    ax.set_ylabel("Feature set")
    ax.set_title(r"External LAOS $R^2$: best model within each feature set")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9.2, color="white" if val >= 0.68 else "black")
    ax.set_xticks(np.arange(-0.5, len(LAOS_TARGETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(values.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.025)
    cbar.set_label(r"External $R^2$")
    save_all(fig, OUT_DIR / "laos", "Figure_laos_feature_set_external_R2_heatmap")


def write_leakage_report(data: pd.DataFrame) -> None:
    train_forms = set(data.loc[data["split"] == "train", "formulation_std"])
    test_forms = set(data.loc[data["split"] == "predict", "formulation_std"])
    overlap = sorted(train_forms & test_forms)
    report = {
        "internal_validation": "Balanced 5-fold CV on the xanthan_pct x yp_pct formulation grid; each fold follows fold=(xanthan_level_index + protein_level_index) mod 5.",
        "external_validation": "Final models are trained on split == train and evaluated on split == predict.",
        "train_formulations": len(train_forms),
        "external_formulations": len(test_forms),
        "train_external_overlap_count": len(overlap),
        "train_external_overlap_formulations": overlap,
        "leakage_status": "PASS" if not overlap else "CHECK_OVERLAP",
    }
    (OUT_DIR / "validation_design_and_leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUT_DIR / "formulation").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "laos").mkdir(parents=True, exist_ok=True)
    set_style()
    data = formulation_level_master()
    write_leakage_report(data)

    formulation_feature_sets = [{"id": "formulation", "label": "formulation only", "features": FORMULATION_FEATURES}]
    formulation_metrics, formulation_preds, formulation_folds = run_task_family(
        data,
        "formulation",
        FORMULATION_TARGETS,
        formulation_feature_sets,
    )
    laos_metrics, laos_preds, laos_folds = run_task_family(data, "laos", LAOS_TARGETS, LAOS_FEATURE_SETS)
    metrics = pd.concat([formulation_metrics, laos_metrics], ignore_index=True)
    predictions = pd.concat([formulation_preds, laos_preds], ignore_index=True)
    folds = pd.concat([formulation_folds, laos_folds], ignore_index=True)

    metrics.to_csv(OUT_DIR / "all_model_metrics_internal_external.csv", index=False)
    predictions.to_csv(OUT_DIR / "all_model_predictions_internal_external.csv", index=False)
    folds.to_csv(OUT_DIR / "balanced_internal_cv_fold_assignments.csv", index=False)
    formulation_metrics.to_csv(OUT_DIR / "formulation" / "formulation_model_metrics.csv", index=False)
    formulation_preds.to_csv(OUT_DIR / "formulation" / "formulation_model_predictions.csv", index=False)
    laos_metrics.to_csv(OUT_DIR / "laos" / "laos_model_metrics.csv", index=False)
    laos_preds.to_csv(OUT_DIR / "laos" / "laos_model_predictions.csv", index=False)

    plot_grouped_r2_rmse_bars(metrics, "formulation", FORMULATION_TARGETS, use_best_feature_set=False)
    plot_grouped_metric_bars_by_feature_set(metrics, "formulation", FORMULATION_TARGETS, formulation_feature_sets)
    plot_grouped_metric_bars_by_feature_set(metrics, "laos", LAOS_TARGETS, LAOS_FEATURE_SETS)
    plot_laos_feature_set_heatmap(metrics)

    plot_best_three_parity(metrics, predictions, "formulation", FORMULATION_TARGETS, validation=EXTERNAL_LABEL)
    plot_best_three_parity(metrics, predictions, "formulation", FORMULATION_TARGETS, validation=INTERNAL_LABEL)
    plot_best_three_by_feature_combination_parity(
        metrics,
        predictions,
        "laos",
        LAOS_TARGETS,
        LAOS_BEST_THREE_FEATURE_SETS,
        validation=EXTERNAL_LABEL,
    )
    plot_best_three_by_feature_combination_parity(
        metrics,
        predictions,
        "laos",
        LAOS_TARGETS,
        LAOS_BEST_THREE_FEATURE_SETS,
        validation=INTERNAL_LABEL,
    )

    plot_all_model_figures(metrics, predictions, "formulation", FORMULATION_TARGETS, formulation_feature_sets)
    plot_formulation_by_target_all_models(metrics, predictions, FORMULATION_TARGETS)
    plot_all_model_figures(metrics, predictions, "laos", LAOS_TARGETS, LAOS_FEATURE_SETS)

    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)
    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "models": [model_label(m) for m in MODEL_ORDER if m in set(metrics["model"])],
        "formulation_targets": FORMULATION_TARGETS,
        "laos_targets": LAOS_TARGETS,
        "laos_feature_sets": LAOS_FEATURE_SETS,
        "validation": {
            "internal": INTERNAL_LABEL,
            "external": EXTERNAL_LABEL,
        },
        "key_outputs": [
            "all_model_metrics_internal_external.csv",
            "all_model_predictions_internal_external.csv",
            "balanced_internal_cv_fold_assignments.csv",
            "formulation/Figure_formulation_grouped_bar_R2_RMSE.png/pdf/tiff",
            "laos/Figure_laos_grouped_bar_R2_by_feature_combination.png/pdf/tiff",
            "laos/Figure_laos_grouped_bar_RMSE_by_feature_combination.png/pdf/tiff",
            "formulation/Figure_formulation_best_three_external_validation_parity.png/pdf/tiff",
            "formulation/parity_by_target_all_models/*.png/pdf/tiff",
            "formulation/bland_altman_by_target_all_models/*.png/pdf/tiff",
            "laos/Figure_laos_best_three_by_feature_combination_external_validation_break_stress_parity.png/pdf/tiff",
            "laos/Figure_laos_best_three_by_feature_combination_external_validation_break_strain_parity.png/pdf/tiff",
            "formulation/all_model_parity/*.png/pdf/tiff",
            "formulation/all_model_bland_altman/*.png/pdf/tiff",
            "laos/all_model_parity/*.png/pdf/tiff",
            "laos/all_model_bland_altman/*.png/pdf/tiff",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
