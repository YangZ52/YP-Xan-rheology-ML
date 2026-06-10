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
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import gpytorch
    import torch
except Exception:  # pragma: no cover
    gpytorch = None
    torch = None


DATA_DIR = Path("/Users/zhiy/Documents/Rheology ML/outputs/ml_ready_xanthan_positive_20260529")
RUN_ID = os.environ.get("RHEOLOGY_ML_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"/Users/zhiy/Documents/Rheology ML/outputs/single_vs_multioutput_{RUN_ID}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["yp_pct", "xanthan_pct"]
TARGETS = [
    "log10_eta_50",
    "log10_Gp_1Hz",
    "log10_Gpp_1Hz",
    "tan_delta_1Hz",
    "log10_break_stress_Pa",
    "break_strain_pct",
]
TARGET_LABELS = {
    "log10_eta_50": "log10 eta50",
    "log10_Gp_1Hz": "log10 G' 1 Hz",
    "log10_Gpp_1Hz": "log10 G'' 1 Hz",
    "tan_delta_1Hz": "tan delta 1 Hz",
    "log10_break_stress_Pa": "log10 break stress",
    "break_strain_pct": "break strain %",
}

MODEL_ORDER = [
    "single_GPR_Matern_ARD",
    "multi_GPyTorch_true_multitask_GP",
    "multi_GPR_independent",
    "single_KernelRidge_RBF",
    "multi_KernelRidge_independent",
    "single_SVR_RBF",
    "multi_SVR_independent",
    "single_Ridge",
    "multi_Ridge_native",
    "single_ExtraTrees",
    "multi_ExtraTrees_native",
    "single_RandomForest",
    "multi_RandomForest_native",
]
MODEL_LABELS = {
    "single_GPR_Matern_ARD": "Single GPR",
    "multi_GPyTorch_true_multitask_GP": "True multitask GP",
    "multi_GPR_independent": "Multi wrapper GPR",
    "single_KernelRidge_RBF": "Single KRR",
    "multi_KernelRidge_independent": "Multi wrapper KRR",
    "single_SVR_RBF": "Single SVR",
    "multi_SVR_independent": "Multi wrapper SVR",
    "single_Ridge": "Single Ridge",
    "multi_Ridge_native": "Native multi Ridge",
    "single_ExtraTrees": "Single ExtraTrees",
    "multi_ExtraTrees_native": "Native multi ExtraTrees",
    "single_RandomForest": "Single RandomForest",
    "multi_RandomForest_native": "Native multi RandomForest",
}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def ordered_models(models):
    present = list(dict.fromkeys(models))
    return [m for m in MODEL_ORDER if m in present] + [m for m in present if m not in MODEL_ORDER]


def short_model_name(model):
    return MODEL_LABELS.get(model, model)


def make_gpr(n_features, n_restarts_optimizer=5, random_state=42):
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


def load_formulation_level():
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


def single_output_models():
    return {
        "single_Ridge": Pipeline([("x_scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "single_SVR_RBF": Pipeline([("x_scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale"))]),
        "single_KernelRidge_RBF": Pipeline([("x_scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf", alpha=0.05))]),
        "single_GPR_Matern_ARD": make_gpr(len(FEATURES)),
        "single_RandomForest": RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
        "single_ExtraTrees": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
    }


def multioutput_models():
    return {
        "multi_Ridge_native": Pipeline([("x_scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "multi_RandomForest_native": RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
        "multi_ExtraTrees_native": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, random_state=42),
        "multi_SVR_independent": MultiOutputRegressor(
            Pipeline([("x_scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale"))])
        ),
        "multi_KernelRidge_independent": MultiOutputRegressor(
            Pipeline([("x_scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf", alpha=0.05))])
        ),
        "multi_GPR_independent": MultiOutputRegressor(make_gpr(len(FEATURES), n_restarts_optimizer=2)),
    }


class MultitaskGPModel(gpytorch.models.ExactGP if gpytorch is not None else object):
    def __init__(self, train_x, train_y, likelihood, n_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=n_tasks)
        base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1])
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel,
            num_tasks=n_tasks,
            rank=min(2, n_tasks),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPyTorchMultitaskRegressor:
    def __init__(self, training_iter=300, lr=0.08, random_state=42):
        self.training_iter = training_iter
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, Y):
        if gpytorch is None or torch is None:
            raise RuntimeError("GPyTorch is not installed. Install torch and gpytorch to run the true multitask GP.")
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0, ddof=0)
        self.x_scale_[self.x_scale_ == 0] = 1
        self.y_mean_ = Y.mean(axis=0)
        self.y_scale_ = Y.std(axis=0, ddof=0)
        self.y_scale_[self.y_scale_ == 0] = 1
        Xs = (X - self.x_mean_) / self.x_scale_
        Ys = (Y - self.y_mean_) / self.y_scale_
        self.train_x_ = torch.tensor(Xs, dtype=torch.float32)
        self.train_y_ = torch.tensor(Ys, dtype=torch.float32)
        self.likelihood_ = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y.shape[1])
        self.model_ = MultitaskGPModel(self.train_x_, self.train_y_, self.likelihood_, Y.shape[1])
        self.model_.train()
        self.likelihood_.train()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_, self.model_)
        for _ in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model_(self.train_x_)
            loss = -mll(output, self.train_y_)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.x_mean_) / self.x_scale_
        test_x = torch.tensor(Xs, dtype=torch.float32)
        self.model_.eval()
        self.likelihood_.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood_(self.model_(test_x))
        mean = pred.mean.detach().cpu().numpy()
        std = pred.stddev.detach().cpu().numpy()
        return mean * self.y_scale_ + self.y_mean_, std * self.y_scale_


def append_prediction_rows(rows, model_name, validation_type, fold, form_index, y_true, y_pred, y_std=None):
    y_std = np.full_like(y_pred, np.nan, dtype=float) if y_std is None else y_std
    for row_i, form in enumerate(form_index):
        for col_i, target in enumerate(TARGETS):
            rows.append(
                {
                    "model": model_name,
                    "validation": validation_type,
                    "fold": fold,
                    "formulation_std": form,
                    "target": target,
                    "y_true": float(y_true[row_i, col_i]),
                    "y_pred": float(y_pred[row_i, col_i]),
                    "y_std": float(y_std[row_i, col_i]) if np.isfinite(y_std[row_i, col_i]) else np.nan,
                }
            )


def evaluate_prediction_rows(pred_df, target_scales):
    rows = []
    for keys, g in pred_df.groupby(["model", "validation", "fold", "target"], dropna=False):
        model, validation, fold, target = keys
        m = metric_dict(g["y_true"], g["y_pred"])
        scale = target_scales.get(target, np.nan)
        m["nrmse_train_sd"] = float(m["rmse"] / scale) if np.isfinite(scale) and scale > 0 else np.nan
        rows.append({"model": model, "validation": validation, "fold": fold, "target": target, **m})
    metrics_df = pd.DataFrame(rows)
    avg_rows = []
    for keys, g in metrics_df.groupby(["model", "validation", "fold"], dropna=False):
        model, validation, fold = keys
        avg_rows.append(
            {
                "model": model,
                "validation": validation,
                "fold": fold,
                "target": "mean_across_targets",
                "n": int(g["n"].sum()),
                "mae": float(g["mae"].mean()),
                "rmse": float(g["rmse"].mean()),
                "nrmse_train_sd": float(g["nrmse_train_sd"].mean()),
                "r2": float(g["r2"].mean()),
            }
        )
    return pd.concat([metrics_df, pd.DataFrame(avg_rows)], ignore_index=True)


def run_single_output(train, test, cv_splits):
    pred_rows = []
    for model_name, base_model in single_output_models().items():
        for target_i, target in enumerate(TARGETS):
            for fold, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
                model = clone(base_model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train.iloc[tr_idx][FEATURES], train.iloc[tr_idx][target])
                pred = model.predict(train.iloc[va_idx][FEATURES])
                y_pred = np.full((len(va_idx), len(TARGETS)), np.nan)
                y_true = np.full((len(va_idx), len(TARGETS)), np.nan)
                y_pred[:, target_i] = pred
                y_true[:, target_i] = train.iloc[va_idx][target].to_numpy(dtype=float)
                append_prediction_rows(
                    pred_rows,
                    model_name,
                    "internal_group_cv",
                    fold,
                    train.iloc[va_idx]["formulation_std"].to_numpy(),
                    y_true,
                    y_pred,
                )
            model = clone(base_model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train[FEATURES], train[target])
            pred = model.predict(test[FEATURES])
            y_pred = np.full((len(test), len(TARGETS)), np.nan)
            y_true = np.full((len(test), len(TARGETS)), np.nan)
            y_pred[:, target_i] = pred
            y_true[:, target_i] = test[target].to_numpy(dtype=float)
            append_prediction_rows(
                pred_rows,
                model_name,
                "external_predict",
                np.nan,
                test["formulation_std"].to_numpy(),
                y_true,
                y_pred,
            )
    return pd.DataFrame(pred_rows).dropna(subset=["y_true", "y_pred"])


def run_multioutput(train, test, cv_splits):
    pred_rows = []
    for model_name, base_model in multioutput_models().items():
        for fold, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
            model = clone(base_model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train.iloc[tr_idx][FEATURES], train.iloc[tr_idx][TARGETS])
            pred = np.asarray(model.predict(train.iloc[va_idx][FEATURES]), dtype=float)
            append_prediction_rows(
                pred_rows,
                model_name,
                "internal_group_cv",
                fold,
                train.iloc[va_idx]["formulation_std"].to_numpy(),
                train.iloc[va_idx][TARGETS].to_numpy(dtype=float),
                pred,
            )
        model = clone(base_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[FEATURES], train[TARGETS])
        pred = np.asarray(model.predict(test[FEATURES]), dtype=float)
        append_prediction_rows(
            pred_rows,
            model_name,
            "external_predict",
            np.nan,
            test["formulation_std"].to_numpy(),
            test[TARGETS].to_numpy(dtype=float),
            pred,
        )
    return pd.DataFrame(pred_rows)


def run_multitask_gp(train, test, cv_splits):
    if gpytorch is None or torch is None:
        return pd.DataFrame()
    pred_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
        model = GPyTorchMultitaskRegressor(training_iter=300, lr=0.08, random_state=42 + fold)
        model.fit(train.iloc[tr_idx][FEATURES], train.iloc[tr_idx][TARGETS])
        pred, std = model.predict(train.iloc[va_idx][FEATURES])
        append_prediction_rows(
            pred_rows,
            "multi_GPyTorch_true_multitask_GP",
            "internal_group_cv",
            fold,
            train.iloc[va_idx]["formulation_std"].to_numpy(),
            train.iloc[va_idx][TARGETS].to_numpy(dtype=float),
            pred,
            std,
        )
    model = GPyTorchMultitaskRegressor(training_iter=500, lr=0.08, random_state=42)
    model.fit(train[FEATURES], train[TARGETS])
    pred, std = model.predict(test[FEATURES])
    append_prediction_rows(
        pred_rows,
        "multi_GPyTorch_true_multitask_GP",
        "external_predict",
        np.nan,
        test["formulation_std"].to_numpy(),
        test[TARGETS].to_numpy(dtype=float),
        pred,
        std,
    )
    return pd.DataFrame(pred_rows)


def plot_summary(metrics_df):
    d = metrics_df[(metrics_df["target"] == "mean_across_targets") & (metrics_df["validation"] == "external_predict")].copy()
    if d.empty:
        return
    d = d.sort_values("nrmse_train_sd")
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.36 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["nrmse_train_sd"], color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean normalized RMSE across targets")
    ax.set_title("External validation: single-output vs multi-output models")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "single_vs_multioutput_external_mean_normalized_rmse.png", dpi=240)
    plt.close(fig)


def plot_grouped_metric_bars(metrics_df, validation, metric, path):
    d = metrics_df[(metrics_df["target"] == "mean_across_targets") & (metrics_df["validation"] == validation)].copy()
    if d.empty:
        return
    d = d.groupby("model", as_index=False)[["mae", "rmse", "nrmse_train_sd", "r2"]].mean()
    d["model_label"] = d["model"].map(short_model_name)
    d = d.set_index("model").reindex(ordered_models(d["model"])).reset_index()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(d))
    colors = np.where(d["model"].str.startswith("single_"), "#4C78A8", "#F58518")
    colors = np.where(d["model"].eq("multi_GPyTorch_true_multitask_GP"), "#54A24B", colors)
    ax.bar(x, d[metric], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(d["model_label"], rotation=35, ha="right")
    ylabel = {
        "nrmse_train_sd": "Mean normalized RMSE",
        "r2": "Mean R2",
        "mae": "Mean MAE",
        "rmse": "Mean RMSE",
    }.get(metric, metric)
    ax.set_ylabel(ylabel)
    title_validation = "External validation" if validation == "external_predict" else "Internal formulation-grouped CV"
    ax.set_title(f"{title_validation}: single-output vs multi-output {ylabel}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_metric_heatmap(metrics_df, validation, metric, path):
    d = metrics_df[(metrics_df["validation"] == validation) & (metrics_df["target"] != "mean_across_targets")].copy()
    if d.empty:
        return
    pivot = d.pivot_table(index="model", columns="target", values=metric, aggfunc="mean")
    pivot = pivot.reindex(ordered_models(pivot.index)).reindex(columns=TARGETS)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.8, max(5.2, 0.34 * len(pivot))))
    cmap = "viridis_r" if metric in {"mae", "rmse", "nrmse_train_sd"} else "viridis"
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([short_model_name(m) for m in pivot.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white")
    title_validation = "External validation" if validation == "external_predict" else "Internal formulation-grouped CV"
    label = "normalized RMSE" if metric == "nrmse_train_sd" else metric
    ax.set_title(f"{title_validation}: target-level {label}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_parity_grid(pred_df, validation, target, path):
    d = pred_df[(pred_df["validation"] == validation) & (pred_df["target"] == target)].copy()
    if d.empty:
        return
    models = ordered_models(d["model"].unique())
    ncols = 3
    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), squeeze=False)
    y_all = pd.concat([d["y_true"], d["y_pred"]]).astype(float)
    lo, hi = y_all.min(), y_all.max()
    pad = (hi - lo) * 0.07 if hi > lo else 1.0
    for ax, model in zip(axes.ravel(), models):
        g = d[d["model"] == model]
        ax.scatter(g["y_true"], g["y_pred"], s=38, alpha=0.82, color="#4C78A8")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1)
        if len(g) >= 2:
            m = metric_dict(g["y_true"], g["y_pred"])
            txt = f"R2={m['r2']:.2f}\nRMSE={m['rmse']:.2g}"
        else:
            txt = "R2=n/a"
        ax.text(
            0.04,
            0.96,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
        )
        ax.set_title(short_model_name(model), fontsize=10)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    label = TARGET_LABELS.get(target, target)
    title_validation = "External validation" if validation == "external_predict" else "Internal formulation-grouped CV"
    fig.supxlabel(f"Measured {label}")
    fig.supylabel(f"Predicted {label}")
    fig.suptitle(f"{title_validation} parity: {label}", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_all_figures(pred_df, metrics_df):
    plot_summary(metrics_df)
    for validation in ["external_predict", "internal_group_cv"]:
        suffix = "external" if validation == "external_predict" else "internal_group_cv"
        plot_grouped_metric_bars(metrics_df, validation, "nrmse_train_sd", OUT_DIR / f"bar_{suffix}_mean_normalized_rmse.png")
        plot_grouped_metric_bars(metrics_df, validation, "r2", OUT_DIR / f"bar_{suffix}_mean_r2.png")
        plot_metric_heatmap(metrics_df, validation, "nrmse_train_sd", OUT_DIR / f"heatmap_{suffix}_target_normalized_rmse.png")
        plot_metric_heatmap(metrics_df, validation, "r2", OUT_DIR / f"heatmap_{suffix}_target_r2.png")
        for target in TARGETS:
            safe_target = target.replace("/", "_")
            plot_parity_grid(pred_df, validation, target, OUT_DIR / f"parity_{suffix}_{safe_target}.png")


def copy_code_snapshot():
    code_dir = OUT_DIR / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    req = Path("/Users/zhiy/Documents/Rheology ML/requirements-multioutput-gp.txt")
    if req.exists():
        shutil.copy2(req, code_dir / req.name)


def main():
    copy_code_snapshot()
    df = load_formulation_level()
    data = df[["split", "formulation_std", *FEATURES, *TARGETS]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train = data[data["split"] == "train"].reset_index(drop=True)
    test = data[data["split"] == "predict"].reset_index(drop=True)
    target_scales = train[TARGETS].std(ddof=0).to_dict()
    cv = GroupKFold(n_splits=min(5, train["formulation_std"].nunique()))
    cv_splits = list(cv.split(train[FEATURES], train[TARGETS], groups=train["formulation_std"]))

    pred_parts = [run_single_output(train, test, cv_splits), run_multioutput(train, test, cv_splits)]
    gp_pred = run_multitask_gp(train, test, cv_splits)
    if not gp_pred.empty:
        pred_parts.append(gp_pred)
    pred_df = pd.concat(pred_parts, ignore_index=True)
    metrics_df = evaluate_prediction_rows(pred_df, target_scales)
    internal_mean = (
        metrics_df[metrics_df["validation"] == "internal_group_cv"]
        .groupby(["model", "validation", "target"], as_index=False)[["mae", "rmse", "nrmse_train_sd", "r2"]]
        .mean()
    )
    external = metrics_df[metrics_df["validation"] == "external_predict"].copy()

    pred_df.to_csv(OUT_DIR / "single_vs_multioutput_predictions.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "single_vs_multioutput_metrics_by_fold.csv", index=False)
    internal_mean.to_csv(OUT_DIR / "single_vs_multioutput_internal_group_cv_mean_metrics.csv", index=False)
    external.to_csv(OUT_DIR / "single_vs_multioutput_external_metrics.csv", index=False)
    plot_all_figures(pred_df, metrics_df)

    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "features": FEATURES,
        "targets": TARGETS,
        "target_scales_train_sd": target_scales,
        "n_train": int(len(train)),
        "n_external": int(len(test)),
        "validation_note": "Internal validation uses formulation-grouped CV. External validation uses split == predict.",
        "single_output_note": "Single-output models are trained independently for each target.",
        "multioutput_note": "Sklearn multi-output models predict all targets from one estimator when supported; independent wrappers fit one estimator per target under a shared API.",
        "true_multitask_gp_note": "GPyTorch MultitaskKernel model is included only when torch/gpytorch are installed.",
        "gpytorch_available": bool(gpytorch is not None and torch is not None),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
