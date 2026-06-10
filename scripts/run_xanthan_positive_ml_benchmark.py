from __future__ import annotations

import json
import math
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
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


DATA_DIR = Path("/Users/zhiy/Documents/Rheology ML/outputs/ml_ready_xanthan_positive_20260529")
RUN_ID = os.environ.get("RHEOLOGY_ML_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"/Users/zhiy/Documents/Rheology ML/outputs/ML_results_xanthan_positive_{RUN_ID}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
INCLUDE_UNCERTAINTY_OUTPUTS = False
INCLUDE_BAGGED_GPR = False

TARGET_LABELS = {
    "log_viscosity": "log10(viscosity [Pa s])",
    "log10_eta_1": "log10(eta at 1 s^-1 [Pa s])",
    "log10_eta_50": "log10(eta at 50 s^-1 [Pa s])",
    "log10_eta_100": "log10(eta at 100 s^-1 [Pa s])",
    "viscosity_shear_thinning_slope_1to100": "shear-thinning slope, 1-100 s^-1 [dimensionless]",
    "log10_Gp_0.1Hz": "log10(G' at 0.1 Hz [Pa])",
    "log10_Gpp_0.1Hz": "log10(G'' at 0.1 Hz [Pa])",
    "tan_delta_0.1Hz": "tan delta at 0.1 Hz [dimensionless]",
    "log10_Gp_1Hz": "log10(G' at 1 Hz [Pa])",
    "log10_Gpp_1Hz": "log10(G'' at 1 Hz [Pa])",
    "tan_delta_1Hz": "tan delta at 1 Hz [dimensionless]",
    "log10_Gp_6.31Hz": "log10(G' at 6.31 Hz [Pa])",
    "log10_Gpp_6.31Hz": "log10(G'' at 6.31 Hz [Pa])",
    "tan_delta_6.31Hz": "tan delta at 6.31 Hz [dimensionless]",
    "break_strain_pct": "break strain [%]",
    "log10_break_stress_Pa": "log10(break stress [Pa])",
    "LVR_pct": "linear viscoelastic region, LVR [%]",
}

FEATURE_LABELS = {
    "yp_pct": "yeast protein [%]",
    "xanthan_pct": "xanthan gum [%]",
    "log_shear_rate": "log10(shear rate [s^-1])",
    "log10_eta_1": "log10(eta at 1 s^-1 [Pa s])",
    "log10_eta_50": "log10(eta at 50 s^-1 [Pa s])",
    "log10_eta_100": "log10(eta at 100 s^-1 [Pa s])",
    "viscosity_shear_thinning_slope_1to100": "shear-thinning slope, 1-100 s^-1",
    "log10_Gp_0.1Hz": "log10(G' at 0.1 Hz [Pa])",
    "log10_Gpp_0.1Hz": "log10(G'' at 0.1 Hz [Pa])",
    "tan_delta_0.1Hz": "tan delta at 0.1 Hz",
    "log10_Gp_1Hz": "log10(G' at 1 Hz [Pa])",
    "log10_Gpp_1Hz": "log10(G'' at 1 Hz [Pa])",
    "tan_delta_1Hz": "tan delta at 1 Hz",
    "log10_Gp_6.31Hz": "log10(G' at 6.31 Hz [Pa])",
    "log10_Gpp_6.31Hz": "log10(G'' at 6.31 Hz [Pa])",
    "tan_delta_6.31Hz": "tan delta at 6.31 Hz",
    "Gp_frequency_slope_0p1to6p31": "G' frequency slope, 0.1-6.31 Hz",
    "Gpp_frequency_slope_0p1to6p31": "G'' frequency slope, 0.1-6.31 Hz",
}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def make_gpr(n_features, n_restarts_optimizer=10, random_state=42):
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
                    n_restarts_optimizer=n_restarts_optimizer,
                    random_state=random_state,
                ),
            ),
        ]
    )


class BaggedGPR(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=7, max_samples=0.8, n_restarts_optimizer=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        sample_size = max(5, int(round(self.max_samples * n)))
        self.estimators_ = []
        for i in range(self.n_estimators):
            idx = rng.choice(np.arange(n), size=sample_size, replace=True)
            model = make_gpr(X.shape[1], n_restarts_optimizer=self.n_restarts_optimizer, random_state=self.random_state + i + 1)
            model.fit(X[idx], y[idx])
            self.estimators_.append(model)
        return self

    def predict(self, X, return_std=False):
        X = pd.DataFrame(X).to_numpy(dtype=float)
        means = []
        variances = []
        for model in self.estimators_:
            mean, std = predict_with_uncertainty(model, X)
            means.append(mean)
            variances.append(std ** 2)
        means = np.vstack(means)
        variances = np.vstack(variances)
        mean = means.mean(axis=0)
        total_var = variances.mean(axis=0) + means.var(axis=0)
        std = np.sqrt(np.maximum(total_var, 0))
        if return_std:
            return mean, std
        return mean


def make_models(n_features):
    models = {
        "Ridge": Pipeline(
            [
                ("x_scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        ),
        "SVR_RBF": Pipeline(
            [
                ("x_scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale")),
            ]
        ),
        "KernelRidge_RBF": Pipeline(
            [
                ("x_scaler", StandardScaler()),
                ("krr", KernelRidge(kernel="rbf", alpha=0.05, gamma=None)),
            ]
        ),
        "GPR_Matern_ARD": make_gpr(n_features),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=42,
        ),
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


def predict_with_uncertainty(model, X):
    if isinstance(model, BaggedGPR):
        return model.predict(X, return_std=True)
    if isinstance(model, Pipeline) and "gpr" in model.named_steps:
        scaler = model.named_steps["x_scaler"]
        gpr = model.named_steps["gpr"]
        return gpr.predict(scaler.transform(X), return_std=True)
    pred = model.predict(X)
    return np.asarray(pred, dtype=float), np.full(len(pred), np.nan)


def extract_gpr_lengthscales(model, features):
    if not isinstance(model, Pipeline) or "gpr" not in model.named_steps:
        return []
    kernel = model.named_steps["gpr"].kernel_
    matern = None

    def walk(k):
        nonlocal matern
        if isinstance(k, Matern):
            matern = k
        for attr in ("k1", "k2"):
            if hasattr(k, attr):
                walk(getattr(k, attr))

    walk(kernel)
    if matern is None:
        return []
    length_scales = np.atleast_1d(matern.length_scale).astype(float)
    relevance = 1 / length_scales
    relevance = relevance / relevance.sum()
    return [
        {
            "feature": f,
            "length_scale_standardized": float(ls),
            "relative_relevance": float(rel),
        }
        for f, ls, rel in zip(features, length_scales, relevance)
    ]


def pretty_target(target):
    return TARGET_LABELS.get(target, target)


def pretty_feature(feature):
    return FEATURE_LABELS.get(feature, feature)


def plot_parity(df, title, path, target):
    if df.empty:
        return
    preferred = ["GPR_Matern_ARD", "KernelRidge_RBF", "SVR_RBF", "Ridge", "XGBoost", "ExtraTrees", "RandomForest", "GradientBoosting"]
    models = [m for m in preferred if m in set(df["model"])] + [m for m in df["model"].unique() if m not in preferred]
    n = len(models)
    if n == 8:
        nrows, ncols = 4, 2
    else:
        ncols = min(3, n)
        nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    y_all = pd.concat([df["y_true"], df["y_pred"]]).astype(float)
    lo, hi = y_all.min(), y_all.max()
    pad = (hi - lo) * 0.07 if hi > lo else 1
    for ax, model in zip(axes_flat, models):
        d = df[df["model"] == model]
        ax.scatter(d["y_true"], d["y_pred"], s=42, alpha=0.8)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1)
        if len(d) >= 2:
            slope, intercept = np.polyfit(d["y_true"].astype(float), d["y_pred"].astype(float), 1)
            xx = np.array([lo - pad, hi + pad])
            ax.plot(xx, slope * xx + intercept, color="#C44E52", lw=1.6, linestyle="--")
            m = metrics(d["y_true"], d["y_pred"])
            txt = (
                f"y = {slope:.2f}x + {intercept:.2f}\n"
                f"R2 = {m['r2']:.3f}\n"
                f"MAE = {m['mae']:.3g}\n"
                f"RMSE = {m['rmse']:.3g}"
            )
        else:
            txt = "R2 = n/a"
        ax.text(
            0.04,
            0.96,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
        )
        ax.set_title(model)
        label = pretty_target(target)
        ax.set_xlabel(f"Measured {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(models):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_full_curve_uncertainty(pred_df, path):
    d = pred_df[(pred_df["model"] == "GPR_Matern_ARD") & (pred_df["split"] == "external_predict")].copy()
    if d.empty or "log_shear_rate" not in d.columns or "y_std" not in d.columns:
        return
    forms = sorted(d["formulation_std"].unique())
    ncols = 2
    nrows = int(math.ceil(len(forms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.4 * nrows), squeeze=False)
    for ax, form in zip(axes.ravel(), forms):
        g = d[d["formulation_std"] == form].sort_values("log_shear_rate")
        x = 10 ** g["log_shear_rate"].to_numpy(dtype=float)
        mean = g["y_pred"].to_numpy(dtype=float)
        std = g["y_std"].to_numpy(dtype=float)
        y_true = g["y_true"].to_numpy(dtype=float)
        ax.plot(x, y_true, "o", color="black", ms=4, label="measured")
        ax.plot(x, mean, color="#4C78A8", lw=1.8, label="GPR mean")
        ax.fill_between(x, mean - 1.96 * std, mean + 1.96 * std, color="#4C78A8", alpha=0.22, label="95% interval")
        ax.set_xscale("log")
        ax.set_title(form)
        ax.set_xlabel("shear rate [s^-1]")
        ax.set_ylabel("log10(viscosity [Pa s])")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(forms):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("External full-viscosity-curve GPR prediction with 95% uncertainty bands", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_external_viscosity_mean_curves(raw_df, pred_df, path, model_name="GPR_Matern_ARD"):
    d_pred = pred_df[(pred_df["split"] == "external_predict") & (pred_df["model"] == model_name)].copy()
    if d_pred.empty:
        return
    forms = sorted(d_pred["formulation_std"].unique(), key=lambda s: (float(s.split("%YP")[0]), float(s.split("+")[1].split("%XG")[0])))
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 17.2), squeeze=False)
    for ax, form in zip(axes.ravel(), forms):
        raw_f = raw_df[(raw_df["split"] == "predict") & (raw_df["formulation_std"] == form)].copy()
        pred_f = d_pred[d_pred["formulation_std"] == form].copy()
        first_rep = sorted(raw_f["replicate"].unique())[0] if len(raw_f) else None
        for rep, rep_df in raw_f.groupby("replicate"):
            rep_df = rep_df.sort_values("shear_rate")
            ax.plot(
                rep_df["shear_rate"],
                10 ** rep_df["log_viscosity"],
                "o",
                ms=3.0,
                alpha=0.34,
                color="#4C78A8",
                label="measured replicates" if rep == first_rep else None,
            )
        measured_mean = raw_f.groupby("shear_rate", as_index=False)["log_viscosity"].mean().sort_values("shear_rate")
        predicted_mean = pred_f.groupby("log_shear_rate", as_index=False)["y_pred"].mean().sort_values("log_shear_rate")
        ax.plot(
            measured_mean["shear_rate"],
            10 ** measured_mean["log_viscosity"],
            "o-",
            ms=4.4,
            lw=1.2,
            color="#1B4F8A",
            label="measured mean",
        )
        ax.plot(
            10 ** predicted_mean["log_shear_rate"],
            10 ** predicted_mean["y_pred"],
            "-",
            lw=2.2,
            color="#D95F02",
            label=f"predicted mean ({model_name.replace('_', '-')})",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(form)
        ax.set_xlabel("shear rate (s$^{-1}$)")
        ax.set_ylabel("viscosity (cP = mPa·s)")
        ax.grid(alpha=0.25, which="both")
    for ax in axes.ravel()[len(forms):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("External validation: measured and predicted viscosity curves", y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_lengthscales(length_df, title, path, top_n=12):
    if length_df.empty:
        return
    d = length_df.sort_values("relative_relevance", ascending=False).head(top_n).copy()
    d["feature_label"] = d["feature"].map(pretty_feature)
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.38 * len(d))))
    ax.barh(d["feature_label"][::-1], d["relative_relevance"][::-1], color="#4C78A8")
    ax.set_xlabel("Relative relevance from inverse ARD length scale")
    ax.set_title(title)
    ax.text(
        0,
        -0.18,
        "Length scales are fitted in standardized input-feature space; log10-transformed inputs retain their log-scale meaning.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.grid(axis="x", alpha=0.25)
    for i, (_, row) in enumerate(d.iloc[::-1].iterrows()):
        ax.text(row["relative_relevance"] + 0.005, i, f"l={row['length_scale_standardized']:.2g}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metric_heatmap(metric_df, metric, title, path):
    if metric_df.empty:
        return
    df = metric_df.copy()
    df["prediction_target"] = df["task"] + " / " + df["target"].map(pretty_target)
    pivot = df.pivot_table(index="prediction_target", columns="model", values=metric, aggfunc="mean")
    preferred = ["GPR_Matern_ARD", "KernelRidge_RBF", "SVR_RBF", "Ridge", "XGBoost", "ExtraTrees", "RandomForest", "GradientBoosting"]
    cols = [c for c in preferred if c in pivot.columns] + [c for c in pivot.columns if c not in preferred]
    pivot = pivot[cols]

    cmap = "RdYlGn" if metric == "r2" else "RdYlGn_r"
    fig_w = max(7.5, 1.35 * len(pivot.columns) + 4.5)
    fig_h = max(5.5, 0.42 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    values = pivot.to_numpy(dtype=float)
    if metric == "r2":
        vmin, vmax = -0.2, 1.0
    else:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                text = f"{val:.2f}" if metric == "r2" else f"{val:.3g}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label(metric.upper() if metric != "r2" else "R2")
    ax.text(
        0,
        -0.16,
        "MAE/RMSE are reported in the modeled target scale. Targets labeled log10 are evaluated in log10 units.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def write_leakage_report(name, train_groups, test_groups, path):
    train_groups = set(train_groups)
    test_groups = set(test_groups)
    overlap = sorted(train_groups & test_groups)
    report = {
        "dataset": name,
        "train_formulations": len(train_groups),
        "external_predict_formulations": len(test_groups),
        "overlap_count": len(overlap),
        "overlap_formulations": overlap,
        "leakage_status": "PASS" if not overlap else "CHECK_OVERLAP",
        "validation_rule": "All internal validation uses GroupKFold by formulation_std; external validation uses formulations from split == predict only.",
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def calibration_table(pred_df):
    d = pred_df[pred_df["y_std"].notna() & (pred_df["y_std"] > 0)].copy()
    rows = []
    for keys, g in d.groupby(["task", "target", "model", "split"]):
        task, target, model, split = keys
        err = np.abs(g["y_true"].to_numpy(dtype=float) - g["y_pred"].to_numpy(dtype=float))
        std = g["y_std"].to_numpy(dtype=float)
        rows.append(
            {
                "task": task,
                "target": target,
                "model": model,
                "split": split,
                "n": len(g),
                "mean_pred_std": float(np.mean(std)),
                "mae": float(np.mean(err)),
                "coverage_68pct_interval": float(np.mean(err <= std)),
                "coverage_95pct_interval": float(np.mean(err <= 1.96 * std)),
            }
        )
    return pd.DataFrame(rows)


def plot_uncertainty_calibration(calib, path):
    if calib.empty:
        return
    d = calib.copy()
    d["label"] = d["task"] + " / " + d["target"].map(pretty_target) + " / " + d["split"]
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.34 * len(d))))
    y = np.arange(len(d))
    ax.barh(y - 0.18, d["coverage_68pct_interval"], height=0.34, label="within +/- 1 sigma")
    ax.barh(y + 0.18, d["coverage_95pct_interval"], height=0.34, label="within +/- 1.96 sigma")
    ax.axvline(0.68, color="black", lw=1, linestyle=":", label="nominal 68%")
    ax.axvline(0.95, color="black", lw=1, linestyle="--", label="nominal 95%")
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Observed coverage")
    ax.set_title("GPR uncertainty calibration")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def formulation_level_master():
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    for freq in ["0.1", "1", "6.31"]:
        gp = f"Gp_{freq}Hz_Pa"
        gpp = f"Gpp_{freq}Hz_Pa"
        if gp in rep.columns:
            rep[f"log10_Gp_{freq}Hz"] = np.where(rep[gp] > 0, np.log10(rep[gp]), np.nan)
        if gpp in rep.columns:
            rep[f"log10_Gpp_{freq}Hz"] = np.where(rep[gpp] > 0, np.log10(rep[gpp]), np.nan)
    rep["log10_Gpp_1Hz"] = np.where(rep["Gpp_1Hz_Pa"] > 0, np.log10(rep["Gpp_1Hz_Pa"]), np.nan)
    rep["log10_break_stress_Pa"] = np.where(rep["break_stress_Pa"] > 0, np.log10(rep["break_stress_Pa"]), np.nan)
    numeric_cols = rep.select_dtypes(include=[np.number]).columns.tolist()
    agg = rep.groupby(["split", "formulation_std"], as_index=False)[numeric_cols].mean()
    return agg


def run_tabular_target(task_name, df, target, features, group_col="formulation_std"):
    needed = [target, group_col, "split"] + features
    data = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "predict"].copy()
    if not (OUT_DIR / f"leakage_check_{task_name}.json").exists():
        write_leakage_report(task_name, train[group_col].unique(), test[group_col].unique(), OUT_DIR / f"leakage_check_{task_name}.json")
    models = make_models(len(features))
    metric_rows = []
    pred_rows = []
    length_rows = []
    fold_assignment_rows = []

    if train[group_col].nunique() >= 5:
        n_splits = min(5, train[group_col].nunique())
        cv = GroupKFold(n_splits=n_splits)
        cv_splits = list(cv.split(train[features], train[target], groups=train[group_col]))
        for fold, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
            train_forms = sorted(train.iloc[tr_idx][group_col].unique())
            val_forms = sorted(train.iloc[va_idx][group_col].unique())
            for form in val_forms:
                fold_assignment_rows.append(
                    {
                        "task": task_name,
                        "target": target,
                        "feature_set": ",".join(features),
                        "fold": fold,
                        "role": "validation",
                        "formulation_std": form,
                        "n_train_formulations": len(train_forms),
                        "n_validation_formulations": len(val_forms),
                    }
                )
            for form in train_forms:
                fold_assignment_rows.append(
                    {
                        "task": task_name,
                        "target": target,
                        "feature_set": ",".join(features),
                        "fold": fold,
                        "role": "training",
                        "formulation_std": form,
                        "n_train_formulations": len(train_forms),
                        "n_validation_formulations": len(val_forms),
                    }
                )
        for model_name, base_model in models.items():
            for fold, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
                model = clone(base_model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train.iloc[tr_idx][features], train.iloc[tr_idx][target])
                pred, std = predict_with_uncertainty(model, train.iloc[va_idx][features])
                m = metrics(train.iloc[va_idx][target], pred)
                metric_rows.append({"task": task_name, "target": target, "feature_set": ",".join(features), "model": model_name, "split": "internal_group_cv", "fold": fold, **m})
                for idx, p, s in zip(train.index[va_idx], pred, std):
                    row = {"task": task_name, "target": target, "model": model_name, "split": "internal_group_cv", "fold": fold, "formulation_std": train.loc[idx, group_col], "y_true": train.loc[idx, target], "y_pred": float(p), "y_std": float(s) if np.isfinite(s) else np.nan}
                    for feat in features:
                        row[feat] = train.loc[idx, feat]
                    pred_rows.append(row)

    for model_name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[features], train[target])
        if len(test):
            pred, std = predict_with_uncertainty(model, test[features])
            m = metrics(test[target], pred)
            metric_rows.append({"task": task_name, "target": target, "feature_set": ",".join(features), "model": model_name, "split": "external_predict", "fold": np.nan, **m})
            for idx, p, s in zip(test.index, pred, std):
                row = {"task": task_name, "target": target, "model": model_name, "split": "external_predict", "fold": np.nan, "formulation_std": test.loc[idx, group_col], "y_true": test.loc[idx, target], "y_pred": float(p), "y_std": float(s) if np.isfinite(s) else np.nan}
                for feat in features:
                    row[feat] = test.loc[idx, feat]
                pred_rows.append(row)
        for row in extract_gpr_lengthscales(model, features):
            length_rows.append({"task": task_name, "target": target, "feature_set": ",".join(features), **row})

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(length_rows), pd.DataFrame(fold_assignment_rows)


def run_scalar_benchmarks():
    df = formulation_level_master()
    formulation_features = ["yp_pct", "xanthan_pct"]
    viscosity_features = formulation_features + [
        "log10_eta_1",
        "log10_eta_50",
        "log10_eta_100",
        "viscosity_shear_thinning_slope_1to100",
    ]
    selected_saos_features = [
        "log10_Gp_0.1Hz",
        "log10_Gpp_0.1Hz",
        "tan_delta_0.1Hz",
        "log10_Gp_1Hz",
        "log10_Gpp_1Hz",
        "tan_delta_1Hz",
        "log10_Gp_6.31Hz",
        "log10_Gpp_6.31Hz",
        "tan_delta_6.31Hz",
        "Gp_frequency_slope_0p1to6p31",
        "Gpp_frequency_slope_0p1to6p31",
    ]
    saos_features = formulation_features + selected_saos_features
    compact_rheology_features = formulation_features + ["log10_eta_50", "log10_Gp_1Hz"]

    jobs = [
        ("viscosity_scalar", "log10_eta_1", formulation_features),
        ("viscosity_scalar", "log10_eta_50", formulation_features),
        ("viscosity_scalar", "log10_eta_100", formulation_features),
        ("viscosity_scalar", "viscosity_shear_thinning_slope_1to100", formulation_features),
        ("saos_scalar", "log10_Gp_0.1Hz", formulation_features),
        ("saos_scalar", "log10_Gpp_0.1Hz", formulation_features),
        ("saos_scalar", "tan_delta_0.1Hz", formulation_features),
        ("saos_scalar", "log10_Gp_1Hz", formulation_features),
        ("saos_scalar", "log10_Gpp_1Hz", formulation_features),
        ("saos_scalar", "tan_delta_1Hz", formulation_features),
        ("saos_scalar", "log10_Gp_6.31Hz", formulation_features),
        ("saos_scalar", "log10_Gpp_6.31Hz", formulation_features),
        ("saos_scalar", "tan_delta_6.31Hz", formulation_features),
        ("strain_from_formulation", "break_strain_pct", formulation_features),
        ("strain_from_formulation", "log10_break_stress_Pa", formulation_features),
        ("strain_from_formulation", "LVR_pct", formulation_features),
        ("strain_from_viscosity", "break_strain_pct", viscosity_features),
        ("strain_from_viscosity", "log10_break_stress_Pa", viscosity_features),
        ("strain_from_viscosity", "LVR_pct", viscosity_features),
        ("strain_from_saos", "break_strain_pct", saos_features),
        ("strain_from_saos", "log10_break_stress_Pa", saos_features),
        ("strain_from_saos", "LVR_pct", saos_features),
        ("strain_from_visc_saos", "break_strain_pct", compact_rheology_features),
        ("strain_from_visc_saos", "log10_break_stress_Pa", compact_rheology_features),
        ("strain_from_visc_saos", "LVR_pct", compact_rheology_features),
    ]

    all_metrics, all_preds, all_lengths, all_folds = [], [], [], []
    for task, target, features in jobs:
        metric, pred, length, folds = run_tabular_target(task, df, target, features)
        all_metrics.append(metric)
        all_preds.append(pred)
        all_lengths.append(length)
        all_folds.append(folds)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)
    lengths_df = pd.concat(all_lengths, ignore_index=True)
    folds_df = pd.concat(all_folds, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "scalar_model_metrics.csv", index=False)
    metrics_df[metrics_df["split"] == "external_predict"].to_csv(OUT_DIR / "scalar_external_metrics.csv", index=False)
    metrics_df[metrics_df["split"] == "internal_group_cv"].groupby(
        ["task", "target", "feature_set", "model"], as_index=False
    )[["mae", "rmse", "r2"]].mean().to_csv(OUT_DIR / "scalar_internal_group_cv_mean_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / "scalar_model_predictions.csv", index=False)
    preds_df[preds_df["split"] == "internal_group_cv"].to_csv(OUT_DIR / "scalar_internal_group_cv_predictions.csv", index=False)
    if INCLUDE_UNCERTAINTY_OUTPUTS:
        calib = calibration_table(preds_df)
        calib.to_csv(OUT_DIR / "scalar_gpr_uncertainty_calibration.csv", index=False)
        plot_uncertainty_calibration(calib[calib["model"] == "GPR_Matern_ARD"], OUT_DIR / "calibration_scalar_gpr.png")
    folds_df.to_csv(OUT_DIR / "scalar_internal_group_cv_fold_assignments.csv", index=False)
    lengths_df.to_csv(OUT_DIR / "gpr_lengthscales_scalar_targets.csv", index=False)

    for (task, target), d in lengths_df.groupby(["task", "target"]):
        safe = f"{task}_{target}".replace("/", "_")
        plot_lengthscales(d, f"GPR ARD length scales: {task} / {target}", OUT_DIR / f"lengthscales_{safe}.png")

    for (task, target), d in preds_df[preds_df["split"] == "external_predict"].groupby(["task", "target"]):
        safe = f"{task}_{target}".replace("/", "_")
        plot_parity(d, f"{task}: {pretty_target(target)} external prediction", OUT_DIR / f"parity_external_{safe}.png", target)
    for (task, target), d in preds_df[preds_df["split"] == "internal_group_cv"].groupby(["task", "target"]):
        safe = f"{task}_{target}".replace("/", "_")
        plot_parity(d, f"{task}: {pretty_target(target)} internal formulation-grouped CV", OUT_DIR / f"parity_internal_group_cv_{safe}.png", target)


def run_full_viscosity_curve():
    df = pd.read_csv(DATA_DIR / "viscosity_long.csv")
    features = ["yp_pct", "xanthan_pct", "log_shear_rate"]
    target = "log_viscosity"
    metric, pred, length, folds = run_tabular_target("full_viscosity_curve", df, target, features, group_col="formulation_std")
    metric.to_csv(OUT_DIR / "full_viscosity_curve_metrics.csv", index=False)
    metric[metric["split"] == "external_predict"].to_csv(OUT_DIR / "full_viscosity_curve_external_metrics.csv", index=False)
    metric[metric["split"] == "internal_group_cv"].groupby(
        ["task", "target", "feature_set", "model"], as_index=False
    )[["mae", "rmse", "r2"]].mean().to_csv(OUT_DIR / "full_viscosity_curve_internal_group_cv_mean_metrics.csv", index=False)
    pred.to_csv(OUT_DIR / "full_viscosity_curve_predictions.csv", index=False)
    pred[pred["split"] == "internal_group_cv"].to_csv(OUT_DIR / "full_viscosity_curve_internal_group_cv_predictions.csv", index=False)
    if INCLUDE_UNCERTAINTY_OUTPUTS:
        calib = calibration_table(pred)
        calib.to_csv(OUT_DIR / "full_viscosity_curve_gpr_uncertainty_calibration.csv", index=False)
        plot_uncertainty_calibration(calib[calib["model"] == "GPR_Matern_ARD"], OUT_DIR / "calibration_full_viscosity_curve_gpr.png")
    folds.to_csv(OUT_DIR / "full_viscosity_curve_internal_group_cv_fold_assignments.csv", index=False)
    length.to_csv(OUT_DIR / "gpr_lengthscales_full_viscosity_curve.csv", index=False)
    plot_lengthscales(
        length,
        "GPR ARD length scales: full viscosity curve",
        OUT_DIR / "lengthscales_full_viscosity_curve.png",
    )
    plot_parity(
        pred[pred["split"] == "external_predict"],
        "Full viscosity curve: external prediction",
        OUT_DIR / "parity_external_full_viscosity_curve.png",
        target,
    )
    plot_parity(
        pred[pred["split"] == "internal_group_cv"],
        "Full viscosity curve: internal formulation-grouped CV",
        OUT_DIR / "parity_internal_group_cv_full_viscosity_curve.png",
        target,
    )
    plot_external_viscosity_mean_curves(
        df,
        pred,
        OUT_DIR / "external_full_viscosity_mean_curves_gpr.png",
        model_name="GPR_Matern_ARD",
    )
    plot_external_viscosity_mean_curves(
        df,
        pred,
        OUT_DIR / "external_full_viscosity_mean_curves_extratrees.png",
        model_name="ExtraTrees",
    )
    if INCLUDE_UNCERTAINTY_OUTPUTS:
        plot_full_curve_uncertainty(pred, OUT_DIR / "gpr_external_full_viscosity_uncertainty_bands.png")


def run_bagged_gpr_scalar_check():
    df = formulation_level_master()
    formulation_features = ["yp_pct", "xanthan_pct"]
    viscosity_features = formulation_features + [
        "log10_eta_1",
        "log10_eta_50",
        "log10_eta_100",
        "viscosity_shear_thinning_slope_1to100",
    ]
    compact_rheology_features = formulation_features + ["log10_eta_50", "log10_Gp_1Hz"]
    jobs = [
        ("viscosity_scalar", "log10_eta_50", formulation_features),
        ("saos_scalar", "log10_Gp_1Hz", formulation_features),
        ("saos_scalar", "log10_Gpp_1Hz", formulation_features),
        ("strain_from_visc_saos", "break_strain_pct", compact_rheology_features),
        ("strain_from_viscosity", "log10_break_stress_Pa", viscosity_features),
        ("strain_from_visc_saos", "LVR_pct", compact_rheology_features),
    ]
    metric_rows = []
    pred_rows = []
    for task, target, features in jobs:
        data = df[[target, "formulation_std", "split"] + features].replace([np.inf, -np.inf], np.nan).dropna().copy()
        train = data[data["split"] == "train"].copy()
        test = data[data["split"] == "predict"].copy()
        model_name = "BaggedGPR_Matern_ARD"
        if train["formulation_std"].nunique() >= 5:
            cv = GroupKFold(n_splits=min(5, train["formulation_std"].nunique()))
            for fold, (tr_idx, va_idx) in enumerate(cv.split(train[features], train[target], groups=train["formulation_std"]), start=1):
                model = BaggedGPR(n_estimators=5, max_samples=0.85, n_restarts_optimizer=1, random_state=100 + fold)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train.iloc[tr_idx][features], train.iloc[tr_idx][target])
                pred, std = model.predict(train.iloc[va_idx][features], return_std=True)
                m = metrics(train.iloc[va_idx][target], pred)
                metric_rows.append({"task": task, "target": target, "feature_set": ",".join(features), "model": model_name, "split": "internal_group_cv", "fold": fold, **m})
                for idx, p, s in zip(train.index[va_idx], pred, std):
                    pred_rows.append({"task": task, "target": target, "model": model_name, "split": "internal_group_cv", "fold": fold, "formulation_std": train.loc[idx, "formulation_std"], "y_true": train.loc[idx, target], "y_pred": float(p), "y_std": float(s)})
        model = BaggedGPR(n_estimators=7, max_samples=0.85, n_restarts_optimizer=1, random_state=177)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[features], train[target])
        if len(test):
            pred, std = model.predict(test[features], return_std=True)
            m = metrics(test[target], pred)
            metric_rows.append({"task": task, "target": target, "feature_set": ",".join(features), "model": model_name, "split": "external_predict", "fold": np.nan, **m})
            for idx, p, s in zip(test.index, pred, std):
                pred_rows.append({"task": task, "target": target, "model": model_name, "split": "external_predict", "fold": np.nan, "formulation_std": test.loc[idx, "formulation_std"], "y_true": test.loc[idx, target], "y_pred": float(p), "y_std": float(s)})
    metric_df = pd.DataFrame(metric_rows)
    pred_df = pd.DataFrame(pred_rows)
    metric_df.to_csv(OUT_DIR / "bagged_gpr_scalar_metrics.csv", index=False)
    pred_df.to_csv(OUT_DIR / "bagged_gpr_scalar_predictions.csv", index=False)
    calib = calibration_table(pred_df)
    calib.to_csv(OUT_DIR / "bagged_gpr_scalar_uncertainty_calibration.csv", index=False)
    plot_uncertainty_calibration(calib, OUT_DIR / "calibration_bagged_gpr_scalar.png")
    for (task, target), d in pred_df[pred_df["split"] == "external_predict"].groupby(["task", "target"]):
        safe = f"{task}_{target}".replace("/", "_")
        plot_parity(d, f"Bagged GPR: {task} / {pretty_target(target)} external prediction", OUT_DIR / f"bagged_gpr_parity_external_{safe}.png", target)


def train_gpr_on_all(df, target, features, n_restarts=10):
    data = df[[target] + features].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model = make_gpr(len(features), n_restarts_optimizer=n_restarts, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data[features], data[target])
    return model, data


def predict_mean_std(model, X):
    mean, std = predict_with_uncertainty(model, X)
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def prob_between(mean, std, lo, hi):
    std = np.maximum(std, 1e-8)
    return norm.cdf((hi - mean) / std) - norm.cdf((lo - mean) / std)


def prob_above(mean, std, lo):
    std = np.maximum(std, 1e-8)
    return 1 - norm.cdf((lo - mean) / std)


def minmax01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def diverse_top(df, score_col, n=12, distance_cols=("yp_pct", "xanthan_pct"), min_dist=0.08):
    chosen = []
    work = df.sort_values(score_col, ascending=False).copy()
    mins = work[list(distance_cols)].min()
    spans = work[list(distance_cols)].max() - mins
    spans = spans.replace(0, 1)
    for _, row in work.iterrows():
        point = ((row[list(distance_cols)] - mins) / spans).to_numpy(dtype=float)
        if not chosen:
            chosen.append(row)
        else:
            prev = np.vstack([((r[list(distance_cols)] - mins) / spans).to_numpy(dtype=float) for r in chosen])
            if np.sqrt(((prev - point) ** 2).sum(axis=1)).min() >= min_dist:
                chosen.append(row)
        if len(chosen) >= n:
            break
    return pd.DataFrame(chosen)


def plot_design_map(grid, value_col, title, path, candidates=None):
    pivot = grid.pivot_table(index="yp_pct", columns="xanthan_pct", values=value_col, aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        cmap="viridis",
    )
    ax.set_xlabel("xanthan gum [%]")
    ax.set_ylabel("yeast protein [%]")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_col)
    if candidates is not None and not candidates.empty:
        ax.scatter(candidates["xanthan_pct"], candidates["yp_pct"], s=58, facecolors="none", edgecolors="white", linewidths=1.4, label="recommended")
        ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def run_inverse_design_and_bayesian():
    measured = formulation_level_master()
    full = measured.copy()
    form_features = ["yp_pct", "xanthan_pct"]
    viscosity_feature_targets = ["log10_eta_50"]
    saos_feature_targets = ["log10_Gp_1Hz"]
    primary_targets = viscosity_feature_targets + saos_feature_targets
    config = pd.DataFrame(
        [
            {"item": "grid_yp_pct_min", "value": 0.0, "notes": "bounded by measured formulation design"},
            {"item": "grid_yp_pct_max", "value": 30.0, "notes": "bounded by measured formulation design"},
            {"item": "grid_xanthan_pct_min", "value": 0.25, "notes": "xanthan-positive only"},
            {"item": "grid_xanthan_pct_max", "value": 1.0, "notes": "bounded by measured formulation design"},
            {"item": "target_log10_eta50_low", "value": 2.0, "notes": "eta50 about 100 Pa s; editable design window"},
            {"item": "target_log10_eta50_high", "value": 3.0, "notes": "eta50 about 1000 Pa s; editable design window"},
            {"item": "target_shear_thinning_slope_low", "value": -0.85, "notes": "moderate/high shear thinning"},
            {"item": "target_shear_thinning_slope_high", "value": -0.65, "notes": "moderate/high shear thinning"},
            {"item": "target_tan_delta_1Hz_low", "value": 0.25, "notes": "avoid too elastic or too viscous, data-informed"},
            {"item": "target_tan_delta_1Hz_high", "value": 0.50, "notes": "avoid too elastic or too viscous, data-informed"},
            {"item": "target_break_strain_pct_min", "value": 40.0, "notes": "retain nonlinear deformation tolerance"},
            {"item": "target_break_stress_Pa_min", "value": 5.0, "notes": "avoid weak structures; editable"},
            {"item": "objective", "value": "maximize feasibility, maximize protein, minimize xanthan", "notes": "Bayesian acquisition ranks exploitation and exploration candidates"},
        ]
    )
    config.to_csv(OUT_DIR / "inverse_design_config.csv", index=False)

    yp_grid = np.round(np.arange(0, 30.0001, 0.25), 4)
    xg_grid = np.round(np.arange(0.25, 1.0001, 0.01), 4)
    grid = pd.DataFrame([(yp, xg) for yp in yp_grid for xg in xg_grid], columns=form_features)

    model_rows = []
    for target in primary_targets:
        model, data = train_gpr_on_all(full, target, form_features)
        mean, std = predict_mean_std(model, grid[form_features])
        grid[f"pred_{target}"] = mean
        grid[f"std_{target}"] = std
        model_rows.append({"target": target, "features": ",".join(form_features), "n_train": len(data)})

    compact_rheology_features = form_features + viscosity_feature_targets + saos_feature_targets
    laos_specs = [
        ("break_strain_pct", compact_rheology_features),
        ("log10_break_stress_Pa", compact_rheology_features),
        ("LVR_pct", compact_rheology_features),
    ]
    for target, features in laos_specs:
        train_df = full.copy()
        model, data = train_gpr_on_all(train_df, target, features)
        X_grid = grid[form_features].copy()
        for feat in features:
            if feat not in X_grid.columns:
                X_grid[feat] = grid[f"pred_{feat}"]
        mean, std = predict_mean_std(model, X_grid[features])
        grid[f"pred_{target}"] = mean
        grid[f"std_{target}"] = std
        model_rows.append({"target": target, "features": ",".join(features), "n_train": len(data)})
    pd.DataFrame(model_rows).to_csv(OUT_DIR / "inverse_design_surrogate_models.csv", index=False)

    eta_p = prob_between(grid["pred_log10_eta_50"], grid["std_log10_eta_50"], 2.0, 3.0)
    strain_p = prob_above(grid["pred_break_strain_pct"], grid["std_break_strain_pct"], 40.0)
    stress_p = prob_above(grid["pred_log10_break_stress_Pa"], grid["std_log10_break_stress_Pa"], math.log10(5.0))
    grid["prob_eta50_in_window"] = eta_p
    grid["prob_break_strain_ok"] = strain_p
    grid["prob_break_stress_ok"] = stress_p
    component_cols = [
        "prob_eta50_in_window",
        "prob_break_strain_ok",
        "prob_break_stress_ok",
    ]
    grid["probability_of_success"] = np.exp(np.log(np.clip(grid[component_cols], 1e-8, 1)).mean(axis=1))
    grid["protein_desirability"] = grid["yp_pct"] / grid["yp_pct"].max()
    grid["xanthan_sparing"] = 1 - (grid["xanthan_pct"] - grid["xanthan_pct"].min()) / (grid["xanthan_pct"].max() - grid["xanthan_pct"].min())
    uncertainty_cols = [c for c in grid.columns if c.startswith("std_log10_eta_50") or c.startswith("std_log10_Gp_1Hz") or c.startswith("std_break_strain_pct") or c.startswith("std_log10_break_stress_Pa")]
    grid["uncertainty_index"] = np.mean([minmax01(grid[c]) for c in uncertainty_cols], axis=0)
    grid["exploitation_score"] = grid["probability_of_success"] * (0.65 * grid["protein_desirability"] + 0.35 * grid["xanthan_sparing"])
    grid["bayesian_exploration_score"] = grid["exploitation_score"] * (1 + 0.35 * grid["uncertainty_index"])
    grid["pred_eta_50_Pa_s"] = 10 ** grid["pred_log10_eta_50"]
    grid["pred_break_stress_Pa"] = 10 ** grid["pred_log10_break_stress_Pa"]

    grid.to_csv(OUT_DIR / "inverse_design_full_grid.csv", index=False)
    exploit = diverse_top(grid, "exploitation_score", n=15)
    explore = diverse_top(grid, "bayesian_exploration_score", n=15)
    feasible = grid[grid["probability_of_success"] >= 0.55].copy().sort_values("exploitation_score", ascending=False).head(100)
    exploit.to_csv(OUT_DIR / "inverse_design_top_exploitation_candidates.csv", index=False)
    explore.to_csv(OUT_DIR / "bayesian_top_exploration_candidates.csv", index=False)
    feasible.to_csv(OUT_DIR / "inverse_design_feasible_region_top100.csv", index=False)
    recommended = pd.concat(
        [
            exploit.head(6).assign(recommendation_type="high_confidence_exploitation"),
            explore.head(6).assign(recommendation_type="bayesian_exploration"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["yp_pct", "xanthan_pct"], keep="first")
    recommended.to_csv(OUT_DIR / "recommended_next_experiments.csv", index=False)

    plot_design_map(grid, "probability_of_success", "Inverse design: probability of satisfying rheology constraints", OUT_DIR / "inverse_design_probability_of_success_map.png", exploit.head(10))
    plot_design_map(grid, "exploitation_score", "Inverse design exploitation score", OUT_DIR / "inverse_design_exploitation_score_map.png", exploit.head(10))
    plot_design_map(grid, "bayesian_exploration_score", "Bayesian exploration score", OUT_DIR / "bayesian_exploration_score_map.png", explore.head(10))
    plot_design_map(grid, "pred_eta_50_Pa_s", "Predicted eta at 50 s^-1 [Pa s]", OUT_DIR / "inverse_design_pred_eta50_map.png", recommended)
    plot_design_map(grid, "pred_break_stress_Pa", "Predicted break stress [Pa]", OUT_DIR / "inverse_design_pred_break_stress_map.png", recommended)


def copy_code_snapshot():
    code_dir = OUT_DIR / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path("/Users/zhiy/Documents/Rheology ML/scripts")
    for name in [
        "run_xanthan_positive_ml_benchmark.py",
        "prepare_rheology_ml_data.py",
        "filter_xanthan_positive_ml_data.py",
        "build_ml_ready_workbook_xanthan_positive.mjs",
    ]:
        src = script_dir / name
        if src.exists():
            shutil.copy2(src, code_dir / name)


def make_summary_heatmaps():
    scalar = pd.read_csv(OUT_DIR / "scalar_model_metrics.csv")
    full = pd.read_csv(OUT_DIR / "full_viscosity_curve_metrics.csv")
    all_metrics = pd.concat([scalar, full], ignore_index=True)
    external = all_metrics[all_metrics["split"] == "external_predict"].copy()
    internal_mean = all_metrics[all_metrics["split"] == "internal_group_cv"].groupby(
        ["task", "target", "feature_set", "model"], as_index=False
    )[["mae", "rmse", "r2"]].mean()
    external.to_csv(OUT_DIR / "all_external_model_metrics.csv", index=False)
    internal_mean.to_csv(OUT_DIR / "all_internal_group_cv_mean_metrics.csv", index=False)
    for metric in ["r2", "mae", "rmse"]:
        plot_metric_heatmap(
            external,
            metric,
            f"External validation model comparison: {metric.upper() if metric != 'r2' else 'R2'}",
            OUT_DIR / f"heatmap_external_{metric}.png",
        )
        plot_metric_heatmap(
            internal_mean,
            metric,
            f"Internal grouped-CV model comparison: {metric.upper() if metric != 'r2' else 'R2'}",
            OUT_DIR / f"heatmap_internal_group_cv_{metric}.png",
        )


def main():
    copy_code_snapshot()
    run_scalar_benchmarks()
    run_full_viscosity_curve()
    if INCLUDE_BAGGED_GPR:
        run_bagged_gpr_scalar_check()
    run_inverse_design_and_bayesian()
    make_summary_heatmaps()
    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "target_scale_note": "Dynamic-range rheology targets are modeled on log10 scale where target names start with log/log10. GPR ARD length scales are estimated after StandardScaler normalization of input features; log10 input features remain log transformed before standardization.",
        "validation_note": "Internal validation uses GroupKFold by formulation_std. External validation uses split == predict formulations. Leakage check JSON files report train/predict formulation overlap.",
        "inverse_design_note": "Inverse design trains final GPR surrogates on all available xanthan-positive measured formulations after validation. GPR uncertainty is used internally for Bayesian acquisition/probability scoring, but uncertainty plots are disabled in the main benchmark outputs.",
        "main_outputs": [
            "scalar_model_metrics.csv",
            "scalar_external_metrics.csv",
            "scalar_internal_group_cv_mean_metrics.csv",
            "scalar_internal_group_cv_predictions.csv",
            "scalar_internal_group_cv_fold_assignments.csv",
            "scalar_model_predictions.csv",
            "gpr_lengthscales_scalar_targets.csv",
            "full_viscosity_curve_metrics.csv",
            "full_viscosity_curve_external_metrics.csv",
            "full_viscosity_curve_internal_group_cv_mean_metrics.csv",
            "full_viscosity_curve_internal_group_cv_predictions.csv",
            "full_viscosity_curve_internal_group_cv_fold_assignments.csv",
            "full_viscosity_curve_predictions.csv",
            "gpr_lengthscales_full_viscosity_curve.csv",
            "all_external_model_metrics.csv",
            "all_internal_group_cv_mean_metrics.csv",
            "heatmap_external_r2.png",
            "heatmap_external_mae.png",
            "heatmap_external_rmse.png",
            "inverse_design_full_grid.csv",
            "inverse_design_top_exploitation_candidates.csv",
            "bayesian_top_exploration_candidates.csv",
            "recommended_next_experiments.csv",
            "inverse_design_probability_of_success_map.png",
            "bayesian_exploration_score_map.png",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
