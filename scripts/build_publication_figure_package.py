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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DATA_DIR = OUTPUTS / "ml_ready_xanthan_positive_20260529"
ARCHIVE_ROOT = Path(os.environ.get("RHEOLOGY_ARCHIVE_ROOT", ROOT / "outputs"))
RUN_ID = os.environ.get("RHEOLOGY_PUBLICATION_PACKAGE_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUTPUTS / f"publication_package_{RUN_ID}"

SELECTED_FREQS = [0.1, 1.0, 6.31]
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
MODEL_NAME_MAP = {
    "GPR_Matern_ARD": "GPR-Matern-ARD",
    "KernelRidge_RBF": "Kernel ridge-RBF",
    "SVR_RBF": "SVR-RBF",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
}
FEATURES = ["yp_pct", "xanthan_pct"]
TARGET_LABELS = {
    "log10_Gp_0p1Hz": r"log$_{10}$ $G'$ at 0.1 Hz (Pa)",
    "log10_Gpp_0p1Hz": r"log$_{10}$ $G''$ at 0.1 Hz (Pa)",
    "tan_delta_0p1Hz": "tan δ at 0.1 Hz",
    "log10_Gp_1Hz": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
    "log10_Gpp_1Hz": r"log$_{10}$ $G''$ at 1 Hz (Pa)",
    "tan_delta_1Hz": "tan δ at 1 Hz",
    "log10_Gp_6p31Hz": r"log$_{10}$ $G'$ at 6.31 Hz (Pa)",
    "log10_Gpp_6p31Hz": r"log$_{10}$ $G''$ at 6.31 Hz (Pa)",
    "tan_delta_6p31Hz": "tan δ at 6.31 Hz",
}

PUBLICATION_TARGET_LABELS = {
    "log_viscosity": r"log$_{10}$ $\eta$ (cP = mPa·s)",
    "log10_eta_1": r"log$_{10}$ $\eta_{1}$ (cP = mPa·s)",
    "log10_eta_50": r"log$_{10}$ $\eta_{50}$ (cP = mPa·s)",
    "log10_eta_100": r"log$_{10}$ $\eta_{100}$ (cP = mPa·s)",
    "viscosity_shear_thinning_slope_1to100": "shear-thinning slope (1-100 s$^{-1}$)",
    "log10_Gp_0.1Hz": r"log$_{10}$ $G'$ at 0.1 Hz (Pa)",
    "log10_Gpp_0.1Hz": r"log$_{10}$ $G''$ at 0.1 Hz (Pa)",
    "tan_delta_0.1Hz": "tan δ at 0.1 Hz",
    "log10_Gp_1Hz": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
    "log10_Gpp_1Hz": r"log$_{10}$ $G''$ at 1 Hz (Pa)",
    "tan_delta_1Hz": "tan δ at 1 Hz",
    "log10_Gp_6.31Hz": r"log$_{10}$ $G'$ at 6.31 Hz (Pa)",
    "log10_Gpp_6.31Hz": r"log$_{10}$ $G''$ at 6.31 Hz (Pa)",
    "tan_delta_6.31Hz": "tan δ at 6.31 Hz",
    "break_strain_pct": "break strain (%)",
    "log10_break_stress_Pa": "log10 break stress (Pa)",
    "LVR_pct": "LVR (%)",
}

FEATURE_LABELS = {
    "yp_pct": "yeast protein (%)",
    "xanthan_pct": "xanthan gum (%)",
    "log_shear_rate": "log10 shear rate (s$^{-1}$)",
    "log10_eta_1": r"log$_{10}$ $\eta_{1}$ (cP = mPa·s)",
    "log10_eta_50": r"log$_{10}$ $\eta_{50}$ (cP = mPa·s)",
    "log10_eta_100": r"log$_{10}$ $\eta_{100}$ (cP = mPa·s)",
    "viscosity_shear_thinning_slope_1to100": "shear-thinning slope",
    "log10_Gp_0.1Hz": r"log$_{10}$ $G'$ at 0.1 Hz",
    "log10_Gpp_0.1Hz": r"log$_{10}$ $G''$ at 0.1 Hz",
    "tan_delta_0.1Hz": "tan δ at 0.1 Hz",
    "log10_Gp_1Hz": r"log$_{10}$ $G'$ at 1 Hz",
    "log10_Gpp_1Hz": r"log$_{10}$ $G''$ at 1 Hz",
    "tan_delta_1Hz": "tan δ at 1 Hz",
    "log10_Gp_6.31Hz": r"log$_{10}$ $G'$ at 6.31 Hz",
    "log10_Gpp_6.31Hz": r"log$_{10}$ $G''$ at 6.31 Hz",
    "tan_delta_6.31Hz": "tan δ at 6.31 Hz",
    "Gp_frequency_slope_0p1to6p31": r"$G'$ slope (0.1-6.31 Hz)",
    "Gpp_frequency_slope_0p1to6p31": r"$G''$ slope (0.1-6.31 Hz)",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "figure.titlesize": 18,
            "axes.linewidth": 1.05,
            "savefig.bbox": "tight",
        }
    )


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metrics(y_true, y_pred) -> dict:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def clean_target_label(target: str) -> str:
    return PUBLICATION_TARGET_LABELS.get(target, TARGET_LABELS.get(target, target))


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("′", "prime")
        .replace("″", "doubleprime")
    )


def make_gpr(n_features: int) -> Pipeline:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )
    return Pipeline(
        [
            ("scale", StandardScaler()),
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


def make_models() -> dict:
    models = {
        "GPR-Matern-ARD": make_gpr(len(FEATURES)),
        "Kernel ridge-RBF": Pipeline([("scale", StandardScaler()), ("krr", KernelRidge(kernel="rbf", alpha=0.05))]),
        "SVR-RBF": Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale"))]),
        "Ridge": Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, random_state=42),
        "Random forest": RandomForestRegressor(n_estimators=700, min_samples_leaf=2, random_state=42),
        "Gradient boosting": GradientBoostingRegressor(
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


def target_name(freq: float, kind: str) -> str:
    fs = f"{freq:g}".replace(".", "p")
    return f"{kind}_{fs}Hz"


def build_selected_saos_table() -> pd.DataFrame:
    freq = pd.read_csv(DATA_DIR / "frequency_long.csv")
    freq = freq.replace([np.inf, -np.inf], np.nan)
    keep = freq[freq["frequency_Hz"].isin(SELECTED_FREQS)].copy()
    keep = keep.dropna(subset=["Gp_Pa", "Gpp_Pa", "tan_delta"])
    mean = (
        keep.groupby(["split", "formulation_std", "yp_pct", "xanthan_pct", "frequency_Hz"], as_index=False)
        .agg(Gp_Pa=("Gp_Pa", "mean"), Gpp_Pa=("Gpp_Pa", "mean"), tan_delta=("tan_delta", "mean"), n_points=("Gp_Pa", "size"))
        .sort_values(["split", "yp_pct", "xanthan_pct", "frequency_Hz"])
    )
    wide_base = mean[["split", "formulation_std", "yp_pct", "xanthan_pct"]].drop_duplicates().reset_index(drop=True)
    wide = wide_base.copy()
    for f in SELECTED_FREQS:
        d = mean[mean["frequency_Hz"] == f][["formulation_std", "Gp_Pa", "Gpp_Pa", "tan_delta"]].copy()
        d[target_name(f, "log10_Gp")] = np.log10(d["Gp_Pa"])
        d[target_name(f, "log10_Gpp")] = np.log10(d["Gpp_Pa"])
        d[target_name(f, "tan_delta")] = d["tan_delta"]
        cols = ["formulation_std", target_name(f, "log10_Gp"), target_name(f, "log10_Gpp"), target_name(f, "tan_delta")]
        wide = wide.merge(d[cols], on="formulation_std", how="left")
    return wide


def add_pred_rows(rows: list[dict], data: pd.DataFrame, pred, *, target: str, model: str, split: str, fold=None) -> None:
    for (_, r), p in zip(data.iterrows(), pred):
        rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "model": model,
                "split": split,
                "fold": fold,
                "formulation_std": r["formulation_std"],
                "yp_pct": r["yp_pct"],
                "xanthan_pct": r["xanthan_pct"],
                "y_true": r[target],
                "y_pred": float(p),
            }
        )


def run_selected_saos_models(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = list(TARGET_LABELS)
    train = wide[wide["split"] == "train"].copy()
    test = wide[wide["split"] == "predict"].copy()
    folds = list(GroupKFold(n_splits=5).split(train[FEATURES], groups=train["formulation_std"]))
    rows_m, rows_p = [], []
    for target in targets:
        tr = train.dropna(subset=FEATURES + [target]).copy()
        te = test.dropna(subset=FEATURES + [target]).copy()
        models = make_models()
        for model_name, base_model in models.items():
            for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
                model = clone(base_model)
                model.fit(tr.iloc[tr_idx][FEATURES], tr.iloc[tr_idx][target])
                pred = model.predict(tr.iloc[va_idx][FEATURES])
                rows_m.append(
                    {
                        "task": "selected_frequency_SAOS",
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "model": model_name,
                        "split": "internal_group_cv",
                        "fold": fold,
                        **metrics(tr.iloc[va_idx][target], pred),
                    }
                )
                add_pred_rows(rows_p, tr.iloc[va_idx], pred, target=target, model=model_name, split="internal_group_cv", fold=fold)
            model = clone(base_model)
            model.fit(tr[FEATURES], tr[target])
            pred = model.predict(te[FEATURES])
            rows_m.append(
                {
                    "task": "selected_frequency_SAOS",
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "model": model_name,
                    "split": "external_predict",
                    "fold": np.nan,
                    **metrics(te[target], pred),
                }
            )
            add_pred_rows(rows_p, te, pred, target=target, model=model_name, split="external_predict")
    return pd.DataFrame(rows_m), pd.DataFrame(rows_p)


def plot_selected_saos_heatmaps(metrics_df: pd.DataFrame, split: str) -> None:
    d = metrics_df[metrics_df["split"] == split].copy()
    d["model"] = pd.Categorical(d["model"], categories=[m for m in MODEL_ORDER if m in set(d["model"])], ordered=True)
    d["target_label"] = pd.Categorical(d["target_label"], categories=[TARGET_LABELS[t] for t in TARGET_LABELS], ordered=True)
    for metric in ["r2", "mae", "rmse"]:
        pivot = d.pivot_table(index="target_label", columns="model", values=metric, aggfunc="mean", observed=False)
        pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
        vals = pivot.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(12.8, 7.1))
        cmap = "RdYlGn" if metric == "r2" else "RdYlGn_r"
        vmin, vmax = (-0.25, 1.0) if metric == "r2" else (np.nanmin(vals), np.nanmax(vals))
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        title_metric = "R$^2$" if metric == "r2" else metric.upper()
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    text = f"{vals[i, j]:.2f}" if metric == "r2" else f"{vals[i, j]:.3g}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(title_metric)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "figures" / "selected_saos" / f"selected_saos_heatmap_{split}_{metric}.png", dpi=300)
        plt.close(fig)


def plot_external_parity_for_target(pred_df: pd.DataFrame, target: str) -> None:
    d0 = pred_df[(pred_df["split"] == "external_predict") & (pred_df["target"] == target)].copy()
    if d0.empty:
        return
    models = [m for m in MODEL_ORDER if m in set(d0["model"])]
    y_all = pd.concat([d0["y_true"], d0["y_pred"]]).astype(float)
    lo, hi = y_all.min(), y_all.max()
    pad = 0.07 * (hi - lo) if hi > lo else 1
    fig, axes = plt.subplots(4, 2, figsize=(11.6, 16.3), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        d = d0[d0["model"] == model]
        ax.scatter(d["y_true"], d["y_pred"], s=42, alpha=0.8, color="#2E6F9E", edgecolor="white", linewidth=0.4)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.0)
        slope, intercept = np.polyfit(d["y_true"], d["y_pred"], 1)
        xx = np.array([lo - pad, hi + pad])
        ax.plot(xx, slope * xx + intercept, color="#C43B40", lw=1.35, ls="--")
        m = metrics(d["y_true"], d["y_pred"])
        ax.text(
            0.04,
            0.96,
            f"y = {slope:.2f}x + {intercept:.2f}\nR$^2$ = {m['r2']:.3f}\nMAE = {m['mae']:.3g}\nRMSE = {m['rmse']:.3g}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )
        ax.set_title(model)
        ax.set_xlabel(f"Measured {TARGET_LABELS[target]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[target]}")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "selected_saos" / f"selected_saos_parity_external_{target}.png", dpi=300)
    plt.close(fig)


def plot_parity_and_bland_altman_grids(pred_df: pd.DataFrame, label: str, base_path: Path) -> list[str]:
    d0 = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"]).copy()
    if d0.empty:
        return []
    d0["model_label"] = d0["model"].map(lambda x: MODEL_NAME_MAP.get(x, x))
    models = [m for m in MODEL_ORDER if m in set(d0["model_label"])]
    if not models:
        models = sorted(d0["model_label"].unique())
    y_all = pd.concat([d0["y_true"], d0["y_pred"]]).astype(float)
    lo, hi = y_all.min(), y_all.max()
    pad = 0.07 * (hi - lo) if hi > lo else 1.0
    diffs = d0["y_pred"].to_numpy(dtype=float) - d0["y_true"].to_numpy(dtype=float)
    diff_abs = np.nanmax(np.abs(diffs)) if len(diffs) else 1.0
    diff_lim = max(diff_abs * 1.25, 0.05)

    parity_path = base_path.with_name(base_path.stem + "_parity.png")
    bland_altman_path = base_path.with_name(base_path.stem + "_bland_altman.png")
    parity_path.parent.mkdir(parents=True, exist_ok=True)

    fig_p, axes_p = plt.subplots(4, 2, figsize=(11.6, 16.2), squeeze=False)
    fig_b, axes_b = plt.subplots(4, 2, figsize=(11.6, 16.2), squeeze=False)
    for i, model in enumerate(models[:8]):
        ax_p = axes_p.ravel()[i]
        ax_b = axes_b.ravel()[i]
        d = d0[d0["model_label"] == model]
        y_true = d["y_true"].astype(float).to_numpy()
        y_pred = d["y_pred"].astype(float).to_numpy()
        ax_p.scatter(y_true, y_pred, s=42, alpha=0.82, color="#2E6F9E", edgecolor="white", linewidth=0.35)
        ax_p.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.1)
        if len(d) >= 2:
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            xx = np.array([lo - pad, hi + pad])
            ax_p.plot(xx, slope * xx + intercept, color="#C43B40", lw=1.5, ls="--")
            m = metrics(y_true, y_pred)
            stats = f"{model}\ny = {slope:.2f}x + {intercept:.2f}\nR$^2$ = {m['r2']:.3f}\nMAE = {m['mae']:.3g}\nRMSE = {m['rmse']:.3g}"
        else:
            stats = f"{model}\nR$^2$ = n/a"
        ax_p.text(
            0.04,
            0.96,
            stats,
            transform=ax_p.transAxes,
            va="top",
            ha="left",
            fontsize=11.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )
        ax_p.set_xlabel(f"Measured {label}")
        ax_p.set_ylabel(f"Predicted {label}")
        ax_p.set_xlim(lo - pad, hi + pad)
        ax_p.set_ylim(lo - pad, hi + pad)
        ax_p.grid(alpha=0.25)

        mean_vals = (y_true + y_pred) / 2
        diff_vals = y_pred - y_true
        bias = float(np.mean(diff_vals))
        sd = float(np.std(diff_vals, ddof=1)) if len(diff_vals) > 1 else 0.0
        loa_low = bias - 1.96 * sd
        loa_high = bias + 1.96 * sd
        ax_b.scatter(mean_vals, diff_vals, s=42, alpha=0.82, color="#4B8F77", edgecolor="white", linewidth=0.35)
        ax_b.axhline(bias, color="#C43B40", lw=1.5)
        ax_b.axhline(loa_low, color="black", lw=1.05, ls="--")
        ax_b.axhline(loa_high, color="black", lw=1.05, ls="--")
        ax_b.axhline(0, color="#777777", lw=0.9, alpha=0.7)
        ax_b.text(
            0.04,
            0.96,
            f"{model}\nbias = {bias:.3g}\n95% LoA = [{loa_low:.3g}, {loa_high:.3g}]",
            transform=ax_b.transAxes,
            va="top",
            ha="left",
            fontsize=11.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )
        ax_b.set_xlabel(f"Mean {label}")
        ax_b.set_ylabel("Predicted - measured")
        ax_b.set_ylim(-diff_lim, diff_lim)
        ax_b.grid(alpha=0.25)
    for ax in axes_p.ravel()[len(models[:8]) :]:
        ax.axis("off")
    for ax in axes_b.ravel()[len(models[:8]) :]:
        ax.axis("off")
    fig_p.tight_layout()
    fig_b.tight_layout()
    fig_p.savefig(parity_path, dpi=300)
    fig_b.savefig(bland_altman_path, dpi=300)
    plt.close(fig_p)
    plt.close(fig_b)
    return [str(parity_path), str(bland_altman_path)]


def generate_publication_parity_bland_altman(main: Path | None, selected_pred_df: pd.DataFrame) -> list[str]:
    written = []
    out_root = OUT_DIR / "figures" / "parity_bland_altman"
    if main is not None:
        sources = [
            (main / "scalar_model_predictions.csv", "scalar_targets"),
            (main / "full_viscosity_curve_predictions.csv", "full_viscosity_curve"),
        ]
        for source, group_name in sources:
            if not source.exists():
                continue
            df = pd.read_csv(source)
            if "target" not in df.columns:
                continue
            for (split, task, target), d in df.groupby(["split", "task", "target"], dropna=False):
                label = clean_target_label(str(target))
                filename = f"{safe_name(group_name)}_{safe_name(split)}_{safe_name(task)}_{safe_name(target)}"
                path = out_root / safe_name(split) / filename
                written.extend(plot_parity_and_bland_altman_grids(d, label, path))
    if not selected_pred_df.empty:
        df = selected_pred_df.copy()
        df["task"] = "selected_frequency_SAOS"
        for (split, target), d in df.groupby(["split", "target"], dropna=False):
            label = TARGET_LABELS.get(str(target), str(target))
            filename = f"selected_saos_{safe_name(split)}_{safe_name(target)}"
            path = out_root / safe_name(split) / filename
            written.extend(plot_parity_and_bland_altman_grids(d, label, path))
    return written


def latest_dir(pattern: str) -> Path | None:
    candidates = sorted(OUTPUTS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def copy_selected_assets(src: Path | None, dst: Path, names: list[str]) -> list[str]:
    copied = []
    if src is None:
        return copied
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
            copied.append(str(dst / name))
    return copied


def plot_clean_lengthscales(main: Path | None) -> list[str]:
    if main is None:
        return []
    frames = []
    for name in ["gpr_lengthscales_scalar_targets.csv", "gpr_lengthscales_full_viscosity_curve.csv"]:
        p = main / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return []

    out_dir = OUT_DIR / "figures" / "combined_publication"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["task"] != "full_frequency_spectra"].copy()
    task_names = {
        "full_viscosity_curve": "viscosity curve",
        "viscosity_scalar": "viscosity scalar",
        "saos_scalar": "SAOS scalar",
        "strain_from_formulation": "LAOS from formulation",
        "strain_from_viscosity": "LAOS from viscosity",
        "strain_from_saos": "LAOS from SAOS",
        "strain_from_visc_saos": "LAOS from viscosity + SAOS",
    }
    df["target_label"] = df["task"].map(lambda x: task_names.get(x, x)) + " | " + df["target"].map(
        lambda x: PUBLICATION_TARGET_LABELS.get(x, x)
    )
    df["feature_label"] = df["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    form = df[df["feature"].isin(["yp_pct", "xanthan_pct", "log_shear_rate"])].copy()
    multi = df[~df.index.isin(form.index)].copy()
    form_order = [
        r"viscosity curve | log$_{10}$ $\eta$ (cP = mPa·s)",
        r"viscosity scalar | log$_{10}$ $\eta_{1}$ (cP = mPa·s)",
        r"viscosity scalar | log$_{10}$ $\eta_{50}$ (cP = mPa·s)",
        r"viscosity scalar | log$_{10}$ $\eta_{100}$ (cP = mPa·s)",
        "viscosity scalar | shear-thinning slope (1-100 s$^{-1}$)",
        r"SAOS scalar | log$_{10}$ $G'$ at 0.1 Hz (Pa)",
        r"SAOS scalar | log$_{10}$ $G''$ at 0.1 Hz (Pa)",
        "SAOS scalar | tan δ at 0.1 Hz",
        r"SAOS scalar | log$_{10}$ $G'$ at 1 Hz (Pa)",
        r"SAOS scalar | log$_{10}$ $G''$ at 1 Hz (Pa)",
        "SAOS scalar | tan δ at 1 Hz",
        r"SAOS scalar | log$_{10}$ $G'$ at 6.31 Hz (Pa)",
        r"SAOS scalar | log$_{10}$ $G''$ at 6.31 Hz (Pa)",
        "SAOS scalar | tan δ at 6.31 Hz",
        "LAOS from formulation | break strain (%)",
        "LAOS from formulation | log10 break stress (Pa)",
        "LAOS from formulation | LVR (%)",
    ]
    form["target_label"] = pd.Categorical(
        form["target_label"],
        categories=[x for x in form_order if x in set(form["target_label"])],
        ordered=True,
    )
    form = form.dropna(subset=["target_label"]).sort_values("target_label")
    written = []
    for sub, filename in [
        (form, "combined_lengthscales_formulation_and_curve_inputs_no_spectra.png"),
        (multi, "combined_lengthscales_multimodal_LAOS_inputs_no_spectra.png"),
    ]:
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="target_label",
            columns="feature_label",
            values="relative_relevance",
            aggfunc="mean",
            observed=False,
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(pivot.columns) + 4), max(4.5, 0.52 * len(pivot.index) + 2)))
        vals = pivot.to_numpy(dtype=float)
        im = ax.imshow(vals, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.01, np.nanmax(vals)))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if vals[i, j] > 0:
                    ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("relative relevance (1 / ARD length scale)")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)
        written.append(str(out_dir / filename))
    df.to_csv(OUT_DIR / "tables" / "gpr_lengthscales_no_full_spectra.csv", index=False)
    return written


def copy_prior_publication_assets() -> dict:
    copied = {}
    main = latest_dir("ML_results_xanthan_positive_*")
    iddsi = latest_dir("IDDSI_inverse_design_scenarios_*")

    copied["main_benchmark"] = copy_selected_assets(
        main,
        OUT_DIR / "figures" / "main_benchmark",
        [
            "heatmap_external_r2.png",
            "heatmap_external_mae.png",
            "heatmap_external_rmse.png",
            "heatmap_internal_group_cv_r2.png",
            "heatmap_internal_group_cv_mae.png",
            "heatmap_internal_group_cv_rmse.png",
            "parity_external_full_viscosity_curve.png",
            "parity_internal_group_cv_full_viscosity_curve.png",
            "external_full_viscosity_mean_curves_gpr.png",
            "external_full_viscosity_mean_curves_extratrees.png",
            "parity_external_viscosity_scalar_log10_eta_50.png",
            "parity_external_strain_from_visc_saos_break_strain_pct.png",
            "parity_external_strain_from_visc_saos_log10_break_stress_Pa.png",
            "parity_external_strain_from_visc_saos_LVR_pct.png",
            "inverse_design_pred_eta50_map.png",
            "inverse_design_probability_of_success_map.png",
            "inverse_design_exploitation_score_map.png",
            "bayesian_exploration_score_map.png",
            "inverse_design_top_exploitation_candidates.csv",
            "bayesian_top_exploration_candidates.csv",
            "all_external_model_metrics.csv",
            "all_internal_group_cv_mean_metrics.csv",
            "scalar_model_predictions.csv",
            "scalar_internal_group_cv_predictions.csv",
            "scalar_internal_group_cv_fold_assignments.csv",
            "full_viscosity_curve_predictions.csv",
            "full_viscosity_curve_internal_group_cv_predictions.csv",
            "full_viscosity_curve_internal_group_cv_fold_assignments.csv",
            "gpr_lengthscales_scalar_targets.csv",
            "gpr_lengthscales_full_viscosity_curve.csv",
            "run_summary.json",
        ],
    )
    copied["iddsi_inverse_design"] = copy_selected_assets(
        iddsi,
        OUT_DIR / "figures" / "iddsi_inverse_design",
        [
            "level3_moderately_thick_high_protein_eta50_contour.png",
            "level3_moderately_thick_high_protein_probability_contour.png",
            "level3_moderately_thick_high_protein_scenario_score_contour.png",
            "level3_lower_xanthan_high_protein_eta50_contour.png",
            "level3_lower_xanthan_high_protein_probability_contour.png",
            "level3_lower_xanthan_high_protein_scenario_score_contour.png",
            "level4_extremely_thick_high_protein_eta50_contour.png",
            "level4_extremely_thick_high_protein_probability_contour.png",
            "level4_extremely_thick_high_protein_scenario_score_contour.png",
            "iddsi_recommended_next_experiments.csv",
            "iddsi_inverse_design_scenario_config.csv",
            "measured_formulations_overlay.csv",
            "literature_and_unit_notes.json",
        ],
    )
    copied["combined_publication_figures"] = plot_clean_lengthscales(main)
    copied["combined_publication_figures"] += copy_selected_assets(
        main,
        OUT_DIR / "figures" / "combined_publication",
        [
            "heatmap_external_r2.png",
            "heatmap_external_mae.png",
            "heatmap_external_rmse.png",
        ],
    )
    return copied


def copy_code_snapshot() -> None:
    code_dir = OUT_DIR / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for script in [
        "build_publication_figure_package.py",
        "run_xanthan_positive_ml_benchmark.py",
        "run_iddsi_inverse_design_scenarios.py",
        "prepare_rheology_ml_data.py",
        "filter_xanthan_positive_ml_data.py",
        "build_ml_ready_workbook_xanthan_positive.mjs",
    ]:
        p = ROOT / "scripts" / script
        if p.exists():
            shutil.copy2(p, code_dir / script)


def write_readme(summary: dict) -> None:
    lines = [
        "# Rheology ML Publication Figure Package",
        "",
        f"Run ID: `{RUN_ID}`",
        "",
        "This folder gathers publication-ready outputs into one timestamped package.",
        "",
        "Key new analysis in this run:",
        "- selected-frequency SAOS scalar prediction at 0.1, 1, and 6.31 Hz",
        "- targets: log10 G′, log10 G″, and tan δ at each selected frequency",
        "- internal validation: 5-fold GroupKFold by formulation",
        "- external validation: held-out prediction formulations",
        "- paired parity and Bland-Altman panels for external validation and internal 5-fold formulation CV",
        "",
        "Important note: full frequency-sweep spectra are intentionally excluded from this cleaned package.",
        "",
        "Main subfolders:",
        "- `figures/selected_saos`: new selected-frequency SAOS heatmaps and parity plots",
        "- `figures/main_benchmark`: main ML benchmark, viscosity, LAOS, and original inverse-design outputs",
        "- `figures/iddsi_inverse_design`: IDDSI/NDD-style inverse-design scenarios with measured overlays",
        "- `figures/combined_publication`: combined heatmaps and GPR length-scale summaries with full spectra removed",
        "- `figures/parity_bland_altman`: title-free parity and Bland-Altman panels for publication assembly",
        "- `tables`: new selected-frequency SAOS data, metrics, and predictions",
        "- `code`: code snapshots used to generate the package",
        "",
        "Best external selected-SAOS models by RMSE:",
    ]
    for row in summary["best_selected_saos_external"]:
        lines.append(
            f"- {row['target_label']}: {row['model']}, R2={row['r2']:.3f}, MAE={row['mae']:.3g}, RMSE={row['rmse']:.3g}"
        )
    (OUT_DIR / "README_publication_package.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_style()
    for sub in ["figures/selected_saos", "tables", "code"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    selected = build_selected_saos_table()
    selected.to_csv(OUT_DIR / "tables" / "selected_frequency_saos_formulation_mean_targets.csv", index=False)
    metrics_df, pred_df = run_selected_saos_models(selected)
    metrics_df.to_csv(OUT_DIR / "tables" / "selected_frequency_saos_model_metrics.csv", index=False)
    pred_df.to_csv(OUT_DIR / "tables" / "selected_frequency_saos_predictions.csv", index=False)
    (
        metrics_df[metrics_df["split"] == "external_predict"]
        .sort_values(["target", "rmse"])
        .to_csv(OUT_DIR / "tables" / "selected_frequency_saos_external_ranked_metrics.csv", index=False)
    )
    for split in ["external_predict", "internal_group_cv"]:
        plot_selected_saos_heatmaps(metrics_df, split)
    for target in TARGET_LABELS:
        plot_external_parity_for_target(pred_df, target)

    copied = copy_prior_publication_assets()
    main = latest_dir("ML_results_xanthan_positive_*")
    copied["parity_bland_altman"] = generate_publication_parity_bland_altman(main, pred_df)
    copy_code_snapshot()
    best = (
        metrics_df[metrics_df["split"] == "external_predict"]
        .sort_values(["target", "rmse"])
        .groupby("target", as_index=False)
        .head(1)
        .to_dict("records")
    )
    summary = {
        "run_id": RUN_ID,
        "output_dir": str(OUT_DIR),
        "selected_frequencies_Hz": SELECTED_FREQS,
        "selected_saos_features": ["log10 G′ (Pa)", "log10 G″ (Pa)", "tan δ"],
        "validation": {
            "internal": "5-fold GroupKFold by formulation",
            "external": "held-out prediction formulations",
        },
        "best_selected_saos_external": best,
        "copied_assets": copied,
        "recommendation": "Use selected-frequency SAOS scalar prediction in the main paper; full frequency-spectra prediction is excluded from this package.",
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
