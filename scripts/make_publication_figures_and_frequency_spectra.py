from __future__ import annotations

import json
import math
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


DATA_DIR = Path(__file__).resolve().parents[1] / "outputs" / "ml_ready_xanthan_positive_20260529"
RUN_ID = os.environ.get("RHEOLOGY_PUBFIG_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / f"publication_figures_frequency_spectra_{RUN_ID}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

MODEL_RENAME = {
    "GPR_Matern_ARD": "GPR-Matern-ARD",
    "KernelRidge_RBF": "Kernel ridge-RBF",
    "SVR_RBF": "SVR-RBF",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
}

TARGET_LABELS = {
    "log10_Gp_Pa": "log10 G′ (Pa)",
    "log10_Gpp_Pa": "log10 G″ (Pa)",
    "log_viscosity": "log10 η (cP = mPa·s)",
    "log10_eta_50": "log10 η50 (cP = mPa·s)",
    "viscosity_shear_thinning_slope_1to100": "shear-thinning slope (1-100 s$^{-1}$)",
    "log10_Gp_1Hz": "log10 G′ at 1 Hz (Pa)",
    "log10_Gpp_1Hz": "log10 G″ at 1 Hz (Pa)",
    "tan_delta_1Hz": "tan δ at 1 Hz",
    "break_strain_pct": "break strain (%)",
    "log10_break_stress_Pa": "log10 break stress (Pa)",
    "LVR_pct": "LVR (%)",
}

FEATURE_LABELS = {
    "yp_pct": "yeast protein (%)",
    "xanthan_pct": "xanthan gum (%)",
    "log_shear_rate": "log10 shear rate (s$^{-1}$)",
    "log10_frequency_Hz": "log10 frequency (Hz)",
    "log10_eta_1": "log10 η1 (cP = mPa·s)",
    "log10_eta_50": "log10 η50 (cP = mPa·s)",
    "log10_eta_100": "log10 η100 (cP = mPa·s)",
    "viscosity_shear_thinning_slope_1to100": "shear-thinning slope",
    "log10_Gp_1Hz": "log10 G′ at 1 Hz",
    "log10_Gpp_1Hz": "log10 G″ at 1 Hz",
    "tan_delta_1Hz": "tan δ at 1 Hz",
    "Gp_frequency_slope_0p1to6p31": "G′ slope (0.1-6.31 Hz)",
    "Gpp_frequency_slope_0p1to6p31": "G″ slope (0.1-6.31 Hz)",
}


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.titlesize": 20,
            "axes.linewidth": 1.1,
            "savefig.bbox": "tight",
        }
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def make_gpr(n_features, restarts=8):
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=restarts, random_state=42)),
        ]
    )


def make_models(n_features):
    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "SVR-RBF": Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale"))]),
        "Kernel ridge-RBF": Pipeline([("scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf", alpha=0.05))]),
        "GPR-Matern-ARD": make_gpr(n_features),
        "Random forest": RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=350, learning_rate=0.035, max_depth=2, min_samples_leaf=2, random_state=42),
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


def extract_gpr_lengthscales(model, features):
    if not isinstance(model, Pipeline) or "gpr" not in model.named_steps:
        return pd.DataFrame()
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
        return pd.DataFrame()
    ls = np.atleast_1d(matern.length_scale).astype(float)
    rel = (1 / ls) / (1 / ls).sum()
    return pd.DataFrame({"feature": features, "length_scale_standardized": ls, "relative_relevance": rel})


def plot_parity_grid(preds, target, split, path):
    d0 = preds[preds["split"] == split].copy()
    if d0.empty:
        return
    d0["model"] = d0["model"].replace(MODEL_RENAME)
    models = [m for m in MODEL_ORDER if m in set(d0["model"])]
    y_all = pd.concat([d0["y_true"], d0["y_pred"]]).astype(float)
    lo, hi = y_all.min(), y_all.max()
    pad = 0.07 * (hi - lo) if hi > lo else 1
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 17.5), squeeze=False)
    axes_flat = axes.ravel()
    for ax, model in zip(axes_flat, models):
        d = d0[d0["model"] == model]
        ax.scatter(d["y_true"], d["y_pred"], s=24, alpha=0.72, edgecolor="none")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.1)
        slope, intercept = np.polyfit(d["y_true"], d["y_pred"], 1)
        xx = np.array([lo - pad, hi + pad])
        ax.plot(xx, slope * xx + intercept, color="#C44E52", lw=1.4, ls="--")
        m = metrics(d["y_true"], d["y_pred"])
        ax.text(
            0.04,
            0.96,
            f"y = {slope:.2f}x + {intercept:.2f}\nR$^2$ = {m['r2']:.3f}\nMAE = {m['mae']:.3g}\nRMSE = {m['rmse']:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )
        ax.set_title(model)
        ax.set_xlabel(f"Measured {TARGET_LABELS[target]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[target]}")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(models):]:
        ax.axis("off")
    fig.suptitle(f"Frequency-sweep spectra prediction: {TARGET_LABELS[target]} ({split.replace('_', ' ')})", y=0.997)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_frequency_spectra_prediction():
    freq = pd.read_csv(DATA_DIR / "frequency_long.csv")
    freq = freq.replace([np.inf, -np.inf], np.nan).dropna(subset=["log10_Gp_Pa", "log10_Gpp_Pa"])
    freq = freq[(freq["frequency_Hz"] >= 0.01) & (freq["frequency_Hz"] <= 6.31)].copy()
    features = ["yp_pct", "xanthan_pct", "log10_frequency_Hz"]
    targets = ["log10_Gp_Pa", "log10_Gpp_Pa"]
    all_metrics = []
    all_preds = []
    length_rows = []
    for target in targets:
        data = freq[["split", "formulation_std", "sample_id_std", "frequency_Hz"] + features + [target]].dropna().copy()
        train = data[data["split"] == "train"]
        test = data[data["split"] == "predict"]
        models = make_models(len(features))
        splits = list(GroupKFold(n_splits=5).split(train[features], train[target], groups=train["formulation_std"]))
        for model_name, base_model in models.items():
            for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
                model = clone(base_model)
                model.fit(train.iloc[tr_idx][features], train.iloc[tr_idx][target])
                pred = model.predict(train.iloc[va_idx][features])
                all_metrics.append({"task": "full_frequency_spectra", "target": target, "model": model_name, "split": "internal_group_cv", "fold": fold, **metrics(train.iloc[va_idx][target], pred)})
                for idx, p in zip(train.index[va_idx], pred):
                    row = train.loc[idx, ["formulation_std", "sample_id_std", "frequency_Hz"] + features].to_dict()
                    row.update({"task": "full_frequency_spectra", "target": target, "model": model_name, "split": "internal_group_cv", "fold": fold, "y_true": train.loc[idx, target], "y_pred": float(p)})
                    all_preds.append(row)
            model = clone(base_model)
            model.fit(train[features], train[target])
            pred = model.predict(test[features])
            all_metrics.append({"task": "full_frequency_spectra", "target": target, "model": model_name, "split": "external_predict", "fold": np.nan, **metrics(test[target], pred)})
            for idx, p in zip(test.index, pred):
                row = test.loc[idx, ["formulation_std", "sample_id_std", "frequency_Hz"] + features].to_dict()
                row.update({"task": "full_frequency_spectra", "target": target, "model": model_name, "split": "external_predict", "fold": np.nan, "y_true": test.loc[idx, target], "y_pred": float(p)})
                all_preds.append(row)
            if model_name == "GPR-Matern-ARD":
                ls = extract_gpr_lengthscales(model, features)
                if not ls.empty:
                    ls["target"] = target
                    length_rows.append(ls)
    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.DataFrame(all_preds)
    metrics_df.to_csv(OUT_DIR / "full_frequency_spectra_model_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / "full_frequency_spectra_predictions.csv", index=False)
    if length_rows:
        pd.concat(length_rows, ignore_index=True).to_csv(OUT_DIR / "gpr_lengthscales_full_frequency_spectra.csv", index=False)
    for target in targets:
        plot_parity_grid(preds_df[preds_df["target"] == target], target, "external_predict", OUT_DIR / f"parity_external_full_frequency_{target}.png")
        plot_parity_grid(preds_df[preds_df["target"] == target], target, "internal_group_cv", OUT_DIR / f"parity_internal_group_cv_full_frequency_{target}.png")
    plot_frequency_curves(preds_df)
    return metrics_df, preds_df


def plot_frequency_curves(preds):
    gpr = preds[(preds["model"] == "GPR-Matern-ARD") & (preds["split"] == "external_predict")].copy()
    if gpr.empty:
        return
    forms = sorted(gpr["formulation_std"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 17.2), squeeze=False)
    colors = {"log10_Gp_Pa": "#4C78A8", "log10_Gpp_Pa": "#F58518"}
    labels = {"log10_Gp_Pa": "G′", "log10_Gpp_Pa": "G″"}
    for ax, form in zip(axes.ravel(), forms):
        for target in ["log10_Gp_Pa", "log10_Gpp_Pa"]:
            d = gpr[(gpr["formulation_std"] == form) & (gpr["target"] == target)].sort_values("frequency_Hz")
            f = d["frequency_Hz"].to_numpy(dtype=float)
            measured = 10 ** d["y_true"].to_numpy(dtype=float)
            predicted = 10 ** d["y_pred"].to_numpy(dtype=float)
            ax.plot(f, measured, "o", ms=4, color=colors[target], alpha=0.85, label=f"measured {labels[target]}")
            ax.plot(f, predicted, "-", lw=1.8, color=colors[target], label=f"predicted {labels[target]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(form)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("modulus (Pa)")
        ax.grid(alpha=0.25, which="both")
    handles, labels0 = axes.ravel()[0].get_legend_handles_labels()
    by_label = dict(zip(labels0, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle("External validation: measured and GPR-predicted frequency spectra (0.01-6.31 Hz)", y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    fig.savefig(OUT_DIR / "external_frequency_spectra_curves_gpr_loglog.png", dpi=300)
    plt.close(fig)


def latest_ml_results_dir():
    candidates = sorted((Path(__file__).resolve().parents[1] / "outputs").glob("ML_results_xanthan_positive_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def plot_metric_heatmap_from_latest():
    src = latest_ml_results_dir()
    if src is None:
        return
    metric_path = src / "all_external_model_metrics.csv"
    if not metric_path.exists():
        return
    df = pd.read_csv(metric_path)
    df["model"] = df["model"].replace(MODEL_RENAME)
    df["target_label"] = df["task"] + " / " + df["target"].map(lambda x: TARGET_LABELS.get(x, x))
    for metric in ["r2", "mae", "rmse"]:
        pivot = df.pivot_table(index="target_label", columns="model", values=metric, aggfunc="mean")
        cols = [m for m in MODEL_ORDER if m in pivot.columns]
        pivot = pivot[cols]
        fig, ax = plt.subplots(figsize=(13.8, 8.8))
        cmap = "RdYlGn" if metric == "r2" else "RdYlGn_r"
        vals = pivot.to_numpy(dtype=float)
        vmin, vmax = (-0.2, 1.0) if metric == "r2" else (np.nanmin(vals), np.nanmax(vals))
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"External validation model comparison: {metric.upper() if metric != 'r2' else 'R$^2$'}")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                val = vals[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}" if metric == "r2" else f"{val:.3g}", ha="center", va="center", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(metric.upper() if metric != "r2" else "R$^2$")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"publication_heatmap_external_{metric}.png", dpi=300)
        plt.close(fig)


def load_lengthscales():
    src = latest_ml_results_dir()
    frames = []
    if src is None:
        return pd.DataFrame()
    for name in ["gpr_lengthscales_scalar_targets.csv", "gpr_lengthscales_full_viscosity_curve.csv"]:
        p = src / name
        if p.exists():
            frames.append(pd.read_csv(p))
    freq_p = OUT_DIR / "gpr_lengthscales_full_frequency_spectra.csv"
    if freq_p.exists():
        f = pd.read_csv(freq_p)
        f["task"] = "full_frequency_spectra"
        f["feature_set"] = "yp_pct,xanthan_pct,log10_frequency_Hz"
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def plot_combined_lengthscales():
    df = load_lengthscales()
    if df.empty:
        return
    task_names = {
        "full_frequency_spectra": "frequency spectra",
        "full_viscosity_curve": "viscosity curve",
        "viscosity_scalar": "viscosity scalar",
        "saos_scalar": "SAOS scalar",
        "strain_from_formulation": "LAOS from formulation",
        "strain_from_viscosity": "LAOS from viscosity",
        "strain_from_visc_saos": "LAOS from viscosity + SAOS",
    }
    df["target_label"] = df["task"].fillna("").map(lambda x: task_names.get(x, x)) + " | " + df["target"].map(lambda x: TARGET_LABELS.get(x, x))
    df["feature_label"] = df["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    form = df[df["feature"].isin(["yp_pct", "xanthan_pct", "log_shear_rate", "log10_frequency_Hz"])].copy()
    multi = df[~df.index.isin(form.index)].copy()
    form_order = [
        "frequency spectra | log10 G′ (Pa)",
        "frequency spectra | log10 G″ (Pa)",
        "viscosity curve | log10 η (cP = mPa·s)",
        "viscosity scalar | log10 η50 (cP = mPa·s)",
        "viscosity scalar | shear-thinning slope (1-100 s$^{-1}$)",
        "SAOS scalar | log10 G′ at 1 Hz (Pa)",
        "SAOS scalar | log10 G″ at 1 Hz (Pa)",
        "SAOS scalar | tan δ at 1 Hz",
        "LAOS from formulation | break strain (%)",
        "LAOS from formulation | log10 break stress (Pa)",
        "LAOS from formulation | LVR (%)",
    ]
    form["target_label"] = pd.Categorical(form["target_label"], categories=[x for x in form_order if x in set(form["target_label"])], ordered=True)
    form = form.dropna(subset=["target_label"]).sort_values("target_label")
    for sub, filename, title in [
        (form, "combined_lengthscales_formulation_and_curve_inputs.png", "GPR ARD relevance: formulation and curve-coordinate inputs"),
        (multi, "combined_lengthscales_multimodal_LAOS_inputs.png", "GPR ARD relevance: multimodal LAOS prediction inputs"),
    ]:
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="target_label", columns="feature_label", values="relative_relevance", aggfunc="mean").fillna(0)
        fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(pivot.columns) + 4), max(4.5, 0.52 * len(pivot.index) + 2)))
        vals = pivot.to_numpy(dtype=float)
        im = ax.imshow(vals, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.01, np.nanmax(vals)))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if vals[i, j] > 0:
                    ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("relative relevance (1 / ARD length scale)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=300)
        plt.close(fig)


def copy_code():
    code_dir = OUT_DIR / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)


def main():
    set_style()
    freq_metrics, _ = run_frequency_spectra_prediction()
    plot_metric_heatmap_from_latest()
    plot_combined_lengthscales()
    copy_code()
    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "frequency_spectra_targets": ["log10 G′ (Pa)", "log10 G″ (Pa)"],
        "frequency_range_Hz": "0.01-6.31",
        "frequency_spectra_features": ["yeast protein (%)", "xanthan gum (%)", "log10 frequency (Hz)"],
        "frequency_external_best": freq_metrics[freq_metrics["split"] == "external_predict"].sort_values(["target", "rmse"]).groupby("target").head(1).to_dict("records"),
        "outputs": [
            "parity_external_full_frequency_log10_Gp_Pa.png",
            "parity_external_full_frequency_log10_Gpp_Pa.png",
            "external_frequency_spectra_curves_gpr_loglog.png",
            "combined_lengthscales_formulation_and_curve_inputs.png",
            "combined_lengthscales_multimodal_LAOS_inputs.png",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
