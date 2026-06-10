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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "ml_ready_xanthan_positive_20260529"
RUN_ID = os.environ.get("RHEOLOGY_FREQ_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"improved_frequency_spectra_{RUN_ID}"
ARCHIVE_ROOT = Path(os.environ.get("RHEOLOGY_ARCHIVE_ROOT", ROOT / "outputs"))

FREQ_MIN = 0.01
FREQ_MAX = 6.31
FEATURES_FORM = ["yp_pct", "xanthan_pct"]
FEATURES_DIRECT = ["yp_pct", "xanthan_pct", "log10_frequency_Hz"]
TARGETS = ["log10_Gp_Pa", "log10_Gpp_Pa"]
TARGET_SYMBOLS = {"log10_Gp_Pa": "G′", "log10_Gpp_Pa": "G″"}
COLORS = {"log10_Gp_Pa": "#2566A5", "log10_Gpp_Pa": "#E56B1F"}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 19,
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


def make_gpr(n_features: int, restarts: int = 8) -> Pipeline:
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
                    n_restarts_optimizer=restarts,
                    random_state=42,
                ),
            ),
        ]
    )


def make_models(n_features: int) -> dict:
    return {
        "GPR-Matern-ARD": make_gpr(n_features),
        "Kernel ridge-RBF": Pipeline(
            [("scale", StandardScaler()), ("krr", KernelRidge(kernel="rbf", alpha=0.04, gamma=None))]
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=800,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
        ),
    }


def load_frequency_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA_DIR / "frequency_long.csv")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw[(raw["frequency_Hz"] >= FREQ_MIN) & (raw["frequency_Hz"] <= FREQ_MAX)].copy()
    raw = raw.dropna(subset=["Gp_Pa", "Gpp_Pa", "log10_Gp_Pa", "log10_Gpp_Pa"])

    group_cols = ["split", "formulation_std", "yp_pct", "xanthan_pct", "frequency_Hz", "log10_frequency_Hz"]
    mean_rows = (
        raw.groupby(group_cols, as_index=False)
        .agg(
            Gp_Pa_mean=("Gp_Pa", "mean"),
            Gpp_Pa_mean=("Gpp_Pa", "mean"),
            Gp_Pa_sd=("Gp_Pa", "std"),
            Gpp_Pa_sd=("Gpp_Pa", "std"),
            n_replicate_points=("Gp_Pa", "size"),
        )
        .sort_values(["split", "yp_pct", "xanthan_pct", "frequency_Hz"])
    )
    mean_rows["log10_Gp_Pa"] = np.log10(mean_rows["Gp_Pa_mean"])
    mean_rows["log10_Gpp_Pa"] = np.log10(mean_rows["Gpp_Pa_mean"])
    return raw, mean_rows


def fit_power_law_parameters(mean_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, form, yp, xg), d in mean_rows.groupby(["split", "formulation_std", "yp_pct", "xanthan_pct"]):
        d = d.sort_values("frequency_Hz")
        x = d[["log10_frequency_Hz"]].to_numpy()
        row = {"split": split, "formulation_std": form, "yp_pct": yp, "xanthan_pct": xg}
        for target in TARGETS:
            lr = LinearRegression().fit(x, d[target].to_numpy(dtype=float))
            pred = lr.predict(x)
            short = "Gp" if target == "log10_Gp_Pa" else "Gpp"
            row[f"log10A_{short}"] = float(lr.intercept_)
            row[f"n_{short}"] = float(lr.coef_[0])
            row[f"fit_r2_{short}"] = float(r2_score(d[target], pred))
            row[f"fit_rmse_{short}"] = rmse(d[target], pred)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "yp_pct", "xanthan_pct"])


def add_prediction_rows(store: list[dict], data: pd.DataFrame, preds: np.ndarray, *, approach: str, target: str, model: str, split: str, fold=None) -> None:
    for (_, row), pred in zip(data.iterrows(), preds):
        store.append(
            {
                "approach": approach,
                "target": target,
                "model": model,
                "split": split,
                "fold": fold,
                "formulation_std": row["formulation_std"],
                "yp_pct": row["yp_pct"],
                "xanthan_pct": row["xanthan_pct"],
                "frequency_Hz": row["frequency_Hz"],
                "log10_frequency_Hz": row["log10_frequency_Hz"],
                "y_true": row[target],
                "y_pred": float(pred),
            }
        )


def run_direct_mean_spectrum(mean_rows: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    all_metrics = []
    all_preds = []
    train = mean_rows[mean_rows["split"] == "train"].copy()
    test = mean_rows[mean_rows["split"] == "predict"].copy()
    folds = list(GroupKFold(n_splits=5).split(train[FEATURES_DIRECT], groups=train["formulation_std"]))
    for target in TARGETS:
        models = make_models(len(FEATURES_DIRECT))
        for model_name, base_model in models.items():
            for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
                model = clone(base_model)
                model.fit(train.iloc[tr_idx][FEATURES_DIRECT], train.iloc[tr_idx][target])
                pred = model.predict(train.iloc[va_idx][FEATURES_DIRECT])
                all_metrics.append(
                    {
                        "approach": "direct mean-spectrum",
                        "target": target,
                        "model": model_name,
                        "split": "internal_group_cv",
                        "fold": fold,
                        **metrics(train.iloc[va_idx][target], pred),
                    }
                )
                add_prediction_rows(
                    all_preds,
                    train.iloc[va_idx],
                    pred,
                    approach="direct mean-spectrum",
                    target=target,
                    model=model_name,
                    split="internal_group_cv",
                    fold=fold,
                )
            model = clone(base_model)
            model.fit(train[FEATURES_DIRECT], train[target])
            pred = model.predict(test[FEATURES_DIRECT])
            all_metrics.append(
                {
                    "approach": "direct mean-spectrum",
                    "target": target,
                    "model": model_name,
                    "split": "external_predict",
                    "fold": np.nan,
                    **metrics(test[target], pred),
                }
            )
            add_prediction_rows(
                all_preds,
                test,
                pred,
                approach="direct mean-spectrum",
                target=target,
                model=model_name,
                split="external_predict",
            )
    return all_metrics, all_preds


def predict_params(train_params: pd.DataFrame, test_params: pd.DataFrame, param: str, model) -> np.ndarray:
    fitted = clone(model)
    fitted.fit(train_params[FEATURES_FORM], train_params[param])
    return fitted.predict(test_params[FEATURES_FORM])


def reconstruct_from_params(rows: pd.DataFrame, param_predictions: pd.DataFrame, target: str) -> np.ndarray:
    short = "Gp" if target == "log10_Gp_Pa" else "Gpp"
    merged = rows.merge(
        param_predictions[["formulation_std", f"log10A_{short}_pred", f"n_{short}_pred"]],
        on="formulation_std",
        how="left",
    )
    return merged[f"log10A_{short}_pred"].to_numpy(dtype=float) + merged[f"n_{short}_pred"].to_numpy(dtype=float) * merged[
        "log10_frequency_Hz"
    ].to_numpy(dtype=float)


def run_power_law_parameter_models(mean_rows: pd.DataFrame, params: pd.DataFrame, residual: bool) -> tuple[list[dict], list[dict]]:
    all_metrics = []
    all_preds = []
    train_rows = mean_rows[mean_rows["split"] == "train"].copy()
    test_rows = mean_rows[mean_rows["split"] == "predict"].copy()
    train_params = params[params["split"] == "train"].copy()
    test_params = params[params["split"] == "predict"].copy()
    folds = list(GroupKFold(n_splits=5).split(train_params[FEATURES_FORM], groups=train_params["formulation_std"]))
    approach = "power-law parameter" if not residual else "power-law + residual GPR"
    models = make_models(len(FEATURES_FORM))

    for model_name, base_model in models.items():
        fold_param_rows = []
        for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
            tr_params = train_params.iloc[tr_idx]
            va_params = train_params.iloc[va_idx]
            pred_params = va_params[["formulation_std"] + FEATURES_FORM].copy()
            for param in ["log10A_Gp", "n_Gp", "log10A_Gpp", "n_Gpp"]:
                pred_params[f"{param}_pred"] = predict_params(tr_params, va_params, param, base_model)

            va_rows = train_rows[train_rows["formulation_std"].isin(va_params["formulation_std"])].copy()
            tr_rows = train_rows[train_rows["formulation_std"].isin(tr_params["formulation_std"])].copy()
            if residual:
                train_true_params = tr_params.copy()
                for param in ["log10A_Gp", "n_Gp", "log10A_Gpp", "n_Gpp"]:
                    train_true_params[f"{param}_pred"] = train_true_params[param]
            for target in TARGETS:
                pred = reconstruct_from_params(va_rows, pred_params, target)
                if residual:
                    short = "Gp" if target == "log10_Gp_Pa" else "Gpp"
                    backbone_train = reconstruct_from_params(tr_rows, train_true_params, target)
                    residual_train = tr_rows[target].to_numpy(dtype=float) - backbone_train
                    residual_model = make_gpr(len(FEATURES_DIRECT), restarts=4)
                    residual_model.fit(tr_rows[FEATURES_DIRECT], residual_train)
                    pred = pred + residual_model.predict(va_rows[FEATURES_DIRECT])
                    _ = short
                all_metrics.append(
                    {
                        "approach": approach,
                        "target": target,
                        "model": model_name,
                        "split": "internal_group_cv",
                        "fold": fold,
                        **metrics(va_rows[target], pred),
                    }
                )
                add_prediction_rows(all_preds, va_rows, pred, approach=approach, target=target, model=model_name, split="internal_group_cv", fold=fold)
            fold_param_rows.append(pred_params.assign(fold=fold, model=model_name, approach=approach))

        pred_params = test_params[["formulation_std"] + FEATURES_FORM].copy()
        for param in ["log10A_Gp", "n_Gp", "log10A_Gpp", "n_Gpp"]:
            pred_params[f"{param}_pred"] = predict_params(train_params, test_params, param, base_model)
        for target in TARGETS:
            pred = reconstruct_from_params(test_rows, pred_params, target)
            if residual:
                train_true_params = train_params.copy()
                for param in ["log10A_Gp", "n_Gp", "log10A_Gpp", "n_Gpp"]:
                    train_true_params[f"{param}_pred"] = train_true_params[param]
                backbone_train = reconstruct_from_params(train_rows, train_true_params, target)
                residual_train = train_rows[target].to_numpy(dtype=float) - backbone_train
                residual_model = make_gpr(len(FEATURES_DIRECT), restarts=6)
                residual_model.fit(train_rows[FEATURES_DIRECT], residual_train)
                pred = pred + residual_model.predict(test_rows[FEATURES_DIRECT])
            all_metrics.append(
                {
                    "approach": approach,
                    "target": target,
                    "model": model_name,
                    "split": "external_predict",
                    "fold": np.nan,
                    **metrics(test_rows[target], pred),
                }
            )
            add_prediction_rows(all_preds, test_rows, pred, approach=approach, target=target, model=model_name, split="external_predict")
        pred_params.assign(model=model_name, approach=approach).to_csv(
            OUT_DIR / f"{approach.replace(' ', '_').replace('+', 'plus')}_{model_name.replace(' ', '_')}_external_parameter_predictions.csv",
            index=False,
        )
    return all_metrics, all_preds


def plot_curve_grid(raw: pd.DataFrame, mean_rows: pd.DataFrame, preds: pd.DataFrame, approach: str, model: str, filename: str) -> None:
    pred = preds[(preds["approach"] == approach) & (preds["model"] == model) & (preds["split"] == "external_predict")].copy()
    forms = sorted(pred["formulation_std"].unique(), key=lambda s: (float(s.split("%YP")[0]), float(s.split("+")[1].split("%XG")[0])))
    fig, axes = plt.subplots(4, 2, figsize=(12.8, 17.2), squeeze=False)
    for ax, form in zip(axes.ravel(), forms):
        raw_f = raw[(raw["split"] == "predict") & (raw["formulation_std"] == form)]
        mean_f = mean_rows[(mean_rows["split"] == "predict") & (mean_rows["formulation_std"] == form)].sort_values("frequency_Hz")
        pred_f = pred[pred["formulation_std"] == form].sort_values("frequency_Hz")
        for target in TARGETS:
            symbol = TARGET_SYMBOLS[target]
            color = COLORS[target]
            raw_col = "Gp_Pa" if target == "log10_Gp_Pa" else "Gpp_Pa"
            mean_col = "Gp_Pa_mean" if target == "log10_Gp_Pa" else "Gpp_Pa_mean"
            d_raw = raw_f.sort_values("frequency_Hz")
            ax.scatter(
                d_raw["frequency_Hz"],
                d_raw[raw_col],
                s=14,
                color=color,
                alpha=0.22,
                edgecolor="none",
            )
            ax.plot(
                mean_f["frequency_Hz"],
                mean_f[mean_col],
                "o",
                ms=5.0,
                color=color,
                alpha=0.95,
                label=f"measured mean {symbol}",
            )
            d_pred = pred_f[pred_f["target"] == target].sort_values("frequency_Hz")
            ax.plot(
                d_pred["frequency_Hz"],
                10 ** d_pred["y_pred"],
                "-",
                lw=2.1,
                color=color,
                label=f"predicted {symbol}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(form, pad=5)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("modulus (Pa)")
        ax.grid(alpha=0.25, which="both")
    for ax in axes.ravel()[len(forms) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"External validation mean frequency spectra: {approach}, {model}", y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    fig.savefig(OUT_DIR / filename, dpi=300)
    plt.close(fig)


def plot_metric_heatmap(metrics_df: pd.DataFrame) -> None:
    ext = metrics_df[metrics_df["split"] == "external_predict"].copy()
    ext["series"] = ext["approach"] + "\n" + ext["model"]
    ext["target_label"] = ext["target"].map({"log10_Gp_Pa": "log10 G′ (Pa)", "log10_Gpp_Pa": "log10 G″ (Pa)"})
    for metric in ["r2", "mae", "rmse"]:
        pivot = ext.pivot_table(index="target_label", columns="series", values=metric, aggfunc="mean")
        vals = pivot.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(14.5, 4.7))
        cmap = "RdYlGn" if metric == "r2" else "RdYlGn_r"
        vmin, vmax = (-0.2, 1.0) if metric == "r2" else (np.nanmin(vals), np.nanmax(vals))
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"External validation of frequency spectra approaches: {metric.upper() if metric != 'r2' else 'R$^2$'}")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, f"{vals[i, j]:.3f}" if metric == "r2" else f"{vals[i, j]:.3g}", ha="center", va="center", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label(metric.upper() if metric != "r2" else "R$^2$")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"frequency_spectra_approach_heatmap_external_{metric}.png", dpi=300)
        plt.close(fig)


def plot_parity(preds: pd.DataFrame, approach: str, model: str) -> None:
    d0 = preds[(preds["approach"] == approach) & (preds["model"] == model) & (preds["split"] == "external_predict")].copy()
    if d0.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3), squeeze=False)
    for ax, target in zip(axes.ravel(), TARGETS):
        d = d0[d0["target"] == target]
        y = pd.concat([d["y_true"], d["y_pred"]]).to_numpy(dtype=float)
        lo, hi = np.nanmin(y), np.nanmax(y)
        pad = 0.06 * (hi - lo)
        ax.scatter(d["y_true"], d["y_pred"], s=36, color=COLORS[target], alpha=0.78, edgecolor="white", linewidth=0.3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.0)
        slope, intercept = np.polyfit(d["y_true"], d["y_pred"], 1)
        xx = np.array([lo - pad, hi + pad])
        ax.plot(xx, slope * xx + intercept, color="#C43B40", lw=1.4, ls="--")
        m = metrics(d["y_true"], d["y_pred"])
        ax.text(
            0.04,
            0.96,
            f"y = {slope:.2f}x + {intercept:.2f}\nR$^2$ = {m['r2']:.3f}\nMAE = {m['mae']:.3f}\nRMSE = {m['rmse']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )
        ax.set_title(f"log10 {TARGET_SYMBOLS[target]} (Pa)")
        ax.set_xlabel(f"measured log10 {TARGET_SYMBOLS[target]} (Pa)")
        ax.set_ylabel(f"predicted log10 {TARGET_SYMBOLS[target]} (Pa)")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
    fig.suptitle(f"External parity: {approach}, {model}", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"parity_external_{approach.replace(' ', '_').replace('+', 'plus')}_{model.replace(' ', '_')}.png", dpi=300)
    plt.close(fig)


def copy_code_snapshot() -> None:
    code_dir = OUT_DIR / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    raw, mean_rows = load_frequency_data()
    params = fit_power_law_parameters(mean_rows)
    raw.to_csv(OUT_DIR / "frequency_replicate_points_used_0p01_to_6p31Hz.csv", index=False)
    mean_rows.to_csv(OUT_DIR / "frequency_formulation_mean_spectra.csv", index=False)
    params.to_csv(OUT_DIR / "frequency_power_law_fit_parameters_by_formulation.csv", index=False)

    metrics_rows = []
    pred_rows = []
    m, p = run_direct_mean_spectrum(mean_rows)
    metrics_rows.extend(m)
    pred_rows.extend(p)
    m, p = run_power_law_parameter_models(mean_rows, params, residual=False)
    metrics_rows.extend(m)
    pred_rows.extend(p)
    m, p = run_power_law_parameter_models(mean_rows, params, residual=True)
    metrics_rows.extend(m)
    pred_rows.extend(p)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.DataFrame(pred_rows)
    metrics_df.to_csv(OUT_DIR / "frequency_spectra_improved_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / "frequency_spectra_improved_predictions.csv", index=False)
    (
        metrics_df[metrics_df["split"] == "external_predict"]
        .sort_values(["target", "rmse"])
        .to_csv(OUT_DIR / "frequency_spectra_external_ranked_metrics.csv", index=False)
    )

    plot_metric_heatmap(metrics_df)
    for approach, stem in [
        ("direct mean-spectrum", "direct_mean_spectrum"),
        ("power-law parameter", "power_law_parameter"),
        ("power-law + residual GPR", "power_law_plus_residual_gpr"),
    ]:
        for model in ["GPR-Matern-ARD", "ExtraTrees"]:
            plot_curve_grid(raw, mean_rows, preds_df, approach, model, f"external_frequency_mean_spectra_{stem}_{model.replace(' ', '_')}_loglog.png")
            plot_parity(preds_df, approach, model)

    best = (
        metrics_df[metrics_df["split"] == "external_predict"]
        .sort_values(["target", "rmse"])
        .groupby("target", as_index=False)
        .head(3)
    )
    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUT_DIR),
        "frequency_range_Hz": [FREQ_MIN, FREQ_MAX],
        "approaches": ["direct mean-spectrum", "power-law parameter", "power-law + residual GPR"],
        "models": ["GPR-Matern-ARD", "Kernel ridge-RBF", "ExtraTrees"],
        "validation": "External validation uses prediction formulations; internal validation is 5-fold GroupKFold by formulation.",
        "best_external_by_log_target": best.to_dict("records"),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    copy_code_snapshot()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
