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
RUN_ID = os.environ.get("RHEOLOGY_COMPACT_DESCRIPTOR_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"compact_descriptor_laos_figures_{RUN_ID}"
DPI = 600

MODEL_LABELS = {
    "GPR_Matern_ARD": "GPR-Matern-ARD",
    "KernelRidge_RBF": "KRR-RBF",
    "SVR_RBF": "SVR-RBF",
    "RandomForest": "Random forest",
    "ExtraTrees": "Extra Trees",
    "GradientBoosting": "Gradient boosting",
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

TASKS = [
    {
        "id": "eta50",
        "stage": "descriptor",
        "task": "viscosity_descriptor",
        "target": "log10_eta_50",
        "features": ["yp_pct", "xanthan_pct"],
        "short": r"$\eta_{50}$",
        "title": r"Viscosity descriptor, $\eta_{50}$",
        "axis": r"log$_{10}$ $\eta_{50}$ (Pa s)",
    },
    {
        "id": "Gp1Hz",
        "stage": "descriptor",
        "task": "saos_descriptor",
        "target": "log10_Gp_1Hz",
        "features": ["yp_pct", "xanthan_pct"],
        "short": r"$G'$ 1 Hz",
        "title": r"SAOS descriptor, $G'$ at 1 Hz",
        "axis": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
    },
    {
        "id": "break_strain",
        "stage": "laos",
        "task": "laos_from_compact_descriptors",
        "target": "break_strain_pct",
        "features": ["log10_eta_50", "log10_Gp_1Hz"],
        "short": r"$\gamma_\mathrm{break}$",
        "title": "LAOS breaking strain",
        "axis": r"$\gamma_\mathrm{break}$ (%)",
    },
    {
        "id": "break_stress",
        "stage": "laos",
        "task": "laos_from_compact_descriptors",
        "target": "log10_break_stress_Pa",
        "features": ["log10_eta_50", "log10_Gp_1Hz"],
        "short": r"$\sigma_\mathrm{break}$",
        "title": "LAOS breaking stress",
        "axis": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
    },
    {
        "id": "LVR",
        "stage": "laos",
        "task": "laos_from_compact_descriptors",
        "target": "LVR_pct",
        "features": ["log10_eta_50", "log10_Gp_1Hz"],
        "short": "LVR",
        "title": "LAOS LVR",
        "axis": "LVR (%)",
    },
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


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
    return models


def formulation_level_master() -> pd.DataFrame:
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    rep["log10_break_stress_Pa"] = np.where(rep["break_stress_Pa"] > 0, np.log10(rep["break_stress_Pa"]), np.nan)
    numeric_cols = rep.select_dtypes(include=[np.number]).columns.tolist()
    return rep.groupby(["split", "formulation_std"], as_index=False)[numeric_cols].mean()


def extract_gpr_lengthscales(model: Pipeline, features: list[str]) -> list[dict]:
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
    relevance = 1 / np.maximum(length_scales, 1e-12)
    relevance = relevance / relevance.sum()
    return [
        {"feature": feat, "length_scale_standardized": float(ls), "relative_relevance": float(rel)}
        for feat, ls, rel in zip(features, length_scales, relevance)
    ]


def run_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    length_rows = []
    for spec in TASKS:
        features = spec["features"]
        needed = ["split", "formulation_std", spec["target"], *features]
        d = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
        train = d[d["split"] == "train"].reset_index(drop=True)
        test = d[d["split"] == "predict"].reset_index(drop=True)
        models = make_models(len(features))

        for model_name, base_model in models.items():
            cv = GroupKFold(n_splits=min(5, train["formulation_std"].nunique()))
            all_true, all_pred = [], []
            for fold, (tr_idx, va_idx) in enumerate(cv.split(train[features], train[spec["target"]], train["formulation_std"]), start=1):
                model = clone(base_model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train.iloc[tr_idx][features], train.iloc[tr_idx][spec["target"]])
                pred = np.asarray(model.predict(train.iloc[va_idx][features]), dtype=float)
                y_true = train.iloc[va_idx][spec["target"]].to_numpy(dtype=float)
                all_true.extend(y_true.tolist())
                all_pred.extend(pred.tolist())
                for local_idx, y_hat in zip(va_idx, pred):
                    row = {
                        **{k: spec[k] for k in ["id", "stage", "task", "target"]},
                        "feature_set": ",".join(features),
                        "model": model_name,
                        "model_label": model_label(model_name),
                        "split": "internal_group_cv",
                        "fold": fold,
                        "formulation_std": train.loc[local_idx, "formulation_std"],
                        "y_true": train.loc[local_idx, spec["target"]],
                        "y_pred": float(y_hat),
                    }
                    for feat in features:
                        row[feat] = train.loc[local_idx, feat]
                    pred_rows.append(row)
            metric_rows.append(
                {
                    **{k: spec[k] for k in ["id", "stage", "task", "target"]},
                    "feature_set": ",".join(features),
                    "model": model_name,
                    "model_label": model_label(model_name),
                    "validation": "Internal 5-fold CV",
                    **metric_row(np.asarray(all_true), np.asarray(all_pred)),
                }
            )

            final = clone(base_model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final.fit(train[features], train[spec["target"]])
            if model_name == "GPR_Matern_ARD":
                for row in extract_gpr_lengthscales(final, features):
                    length_rows.append({**{k: spec[k] for k in ["id", "stage", "task", "target"]}, **row})
            if len(test):
                pred = np.asarray(final.predict(test[features]), dtype=float)
                metric_rows.append(
                    {
                        **{k: spec[k] for k in ["id", "stage", "task", "target"]},
                        "feature_set": ",".join(features),
                        "model": model_name,
                        "model_label": model_label(model_name),
                        "validation": "External validation",
                        **metric_row(test[spec["target"]].to_numpy(dtype=float), pred),
                    }
                )
                for idx, y_hat in enumerate(pred):
                    row = {
                        **{k: spec[k] for k in ["id", "stage", "task", "target"]},
                        "feature_set": ",".join(features),
                        "model": model_name,
                        "model_label": model_label(model_name),
                        "split": "external_validation",
                        "fold": np.nan,
                        "formulation_std": test.loc[idx, "formulation_std"],
                        "y_true": test.loc[idx, spec["target"]],
                        "y_pred": float(y_hat),
                    }
                    for feat in features:
                        row[feat] = test.loc[idx, feat]
                    pred_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(length_rows)


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


def save_all(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.tiff", dpi=DPI)
    plt.close(fig)


def normalized(values: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = values.astype(float).copy()
    for col in out.columns:
        lo, hi = out[col].min(), out[col].max()
        if hi <= lo:
            out[col] = 0.5
        elif metric == "r2":
            out[col] = (out[col] - lo) / (hi - lo)
        else:
            out[col] = 1 - (out[col] - lo) / (hi - lo)
    return out


def plot_performance_heatmap(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.9), constrained_layout=True)
    validation = "External validation"
    for ax, metric in zip(axes, ["r2", "mae", "rmse"]):
        d = metrics[metrics["validation"] == validation].copy()
        values = d.pivot_table(index="model_label", columns="id", values=metric).reindex(
            index=[m for m in MODEL_ORDER if m in set(d["model_label"])],
            columns=[t["id"] for t in TASKS],
        )
        color = normalized(values, metric)
        im = ax.imshow(color, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_title({"r2": r"External $R^2$", "mae": "External MAE", "rmse": "External RMSE"}[metric])
        ax.set_xticks(np.arange(len(TASKS)))
        ax.set_xticklabels([t["short"] for t in TASKS], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(values.index)))
        ax.set_yticklabels(values.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                text_color = "white" if color.iloc[i, j] >= 0.68 else "black"
                ax.text(j, i, fmt(values.iloc[i, j], metric), ha="center", va="center", fontsize=8.6, color=text_color)
        ax.set_xticks(np.arange(-0.5, len(TASKS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(values.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label("normalized within each target")
    save_all(fig, "Figure1_compact_descriptor_external_performance_heatmap")


def best_external_models(metrics: pd.DataFrame) -> pd.DataFrame:
    d = metrics[metrics["validation"] == "External validation"].copy()
    d = d.sort_values(["id", "r2", "rmse"], ascending=[True, False, True])
    return d.groupby("id", as_index=False).head(1)


def panel_limits(d: pd.DataFrame) -> tuple[float, float]:
    lo = float(np.nanmin([d["y_true"].min(), d["y_pred"].min()]))
    hi = float(np.nanmax([d["y_true"].max(), d["y_pred"].max()]))
    pad = (hi - lo) * 0.08 if hi > lo else 1.0
    return lo - pad, hi + pad


def plot_best_parity(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    best = best_external_models(metrics)
    best.to_csv(OUT_DIR / "best_external_model_by_target.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.8), constrained_layout=True)
    axes = axes.ravel()
    colors = {"descriptor": "#4C78A8", "laos": "#A66B55"}
    for ax, spec in zip(axes, TASKS):
        model = best.loc[best["id"] == spec["id"], "model"].iloc[0]
        d = predictions[(predictions["split"] == "external_validation") & (predictions["id"] == spec["id"]) & (predictions["model"] == model)]
        m = metrics[(metrics["validation"] == "External validation") & (metrics["id"] == spec["id"]) & (metrics["model"] == model)].iloc[0]
        lo, hi = panel_limits(d)
        ax.scatter(d["y_true"], d["y_pred"], s=56, color=colors[spec["stage"]], edgecolor="white", linewidth=0.7, alpha=0.92)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.1)
        if len(d) >= 2:
            slope, intercept = np.polyfit(d["y_true"], d["y_pred"], 1)
            xx = np.array([lo, hi])
            ax.plot(xx, slope * xx + intercept, color="#333333", linestyle=(0, (4, 3)), linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(spec["title"])
        ax.set_xlabel(f"Measured {spec['axis']}")
        ax.set_ylabel(f"Predicted {spec['axis']}")
        ax.text(
            0.04,
            0.96,
            f"{model_label(model)}\n$R^2$ = {m['r2']:.2f}\nMAE = {fmt(m['mae'], 'mae')}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "#C9C9C9", "alpha": 0.92, "pad": 4},
        )
        ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].axis("off")
    save_all(fig, "Figure2_compact_descriptor_best_external_parity")


def plot_lengthscales(lengths: pd.DataFrame) -> None:
    feature_labels = {
        "yp_pct": "yeast protein (%)",
        "xanthan_pct": "xanthan gum (%)",
        "log10_eta_50": r"log$_{10}$ $\eta_{50}$",
        "log10_Gp_1Hz": r"log$_{10}$ $G'$ 1 Hz",
    }
    target_labels = {t["id"]: t["short"] for t in TASKS}
    pivot = lengths.pivot_table(index="id", columns="feature", values="relative_relevance", aggfunc="mean")
    pivot = pivot.reindex(index=[t["id"] for t in TASKS])
    pivot = pivot[[c for c in ["yp_pct", "xanthan_pct", "log10_eta_50", "log10_Gp_1Hz"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    vals = pivot.to_numpy(dtype=float)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=0, vmax=max(np.nanmax(vals), 0.01), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([feature_labels.get(c, c) for c in pivot.columns], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([target_labels.get(i, i) for i in pivot.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                text_color = "white" if vals[i, j] >= 0.68 else "black"
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=10, color=text_color)
    ax.set_title("GPR-ARD relative feature relevance")
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("relative relevance")
    save_all(fig, "Figure3_compact_descriptor_gpr_lengthscales")


def plot_descriptor_space(data: pd.DataFrame) -> None:
    train = data[data["split"] == "train"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    specs = [
        ("break_strain_pct", r"$\gamma_\mathrm{break}$ (%)", "viridis"),
        ("log10_break_stress_Pa", r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)", "magma"),
        ("LVR_pct", "LVR (%)", "cividis"),
    ]
    for ax, (target, label, cmap) in zip(axes, specs):
        sc = ax.scatter(
            train["log10_eta_50"],
            train["log10_Gp_1Hz"],
            c=train[target],
            cmap=cmap,
            s=74,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_xlabel(r"log$_{10}$ $\eta_{50}$ (Pa s)")
        ax.set_ylabel(r"log$_{10}$ $G'$ at 1 Hz (Pa)")
        ax.set_title(label)
        ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.025)
        cbar.set_label(label)
    save_all(fig, "Figure4_laos_response_in_compact_descriptor_space")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    data = formulation_level_master()
    metrics, predictions, lengths = run_models(data)
    metrics.to_csv(OUT_DIR / "compact_descriptor_model_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "compact_descriptor_model_predictions.csv", index=False)
    lengths.to_csv(OUT_DIR / "compact_descriptor_gpr_lengthscales.csv", index=False)
    plot_performance_heatmap(metrics)
    plot_best_parity(metrics, predictions)
    plot_lengthscales(lengths)
    plot_descriptor_space(data)
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)
    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "modeling_note": "Compact rheology descriptor workflow: formulation predicts log10_eta_50 and log10_Gp_1Hz; LAOS targets are modeled using only log10_eta_50 and log10_Gp_1Hz.",
        "outputs": [
            "Figure1_compact_descriptor_external_performance_heatmap.png",
            "Figure2_compact_descriptor_best_external_parity.png",
            "Figure3_compact_descriptor_gpr_lengthscales.png",
            "Figure4_laos_response_in_compact_descriptor_space.png",
            "compact_descriptor_model_metrics.csv",
            "compact_descriptor_model_predictions.csv",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
