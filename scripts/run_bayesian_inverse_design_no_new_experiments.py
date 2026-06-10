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
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/zhiy/Documents/Rheology ML")
DATA_DIR = ROOT / "outputs" / "ml_ready_xanthan_positive_20260529"
ONEDRIVE_ROOT = Path("/Users/zhiy/Library/CloudStorage/OneDrive-Personal/GPR new")
RUN_ID = os.environ.get("RHEOLOGY_INVERSE_DESIGN_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"bayesian_inverse_design_no_new_experiments_{RUN_ID}"
DPI = 400

FEATURES = ["yp_pct", "xanthan_pct"]
PRIMARY_TARGETS = [
    {
        "id": "eta50",
        "column": "log10_eta_50",
        "label": r"log$_{10}$ $\eta_{50}$",
        "unit_label": r"log$_{10}$ $\eta_{50}$ (Pa s)",
        "display_target": r"$\eta_{50}$ around 500 mPa s",
        "goal": "between",
        "low": math.log10(400.0),
        "high": math.log10(600.0),
        "target_value": math.log10(500.0),
        "target_display_value": "500 mPa s",
        "weight": 1.0,
    },
    {
        "id": "Gp1Hz",
        "column": "log10_Gp_1Hz",
        "label": r"log$_{10}$ $G'$ 1 Hz",
        "unit_label": r"log$_{10}$ $G'$ at 1 Hz (Pa)",
        "display_target": r"$G'$ 1 Hz around 100 Pa",
        "goal": "between",
        "low": math.log10(75.0),
        "high": math.log10(150.0),
        "target_value": math.log10(100.0),
        "target_display_value": "100 Pa",
        "weight": 1.0,
    },
]

SECONDARY_TARGETS = [
    {
        "id": "break_stress",
        "column": "log10_break_stress_Pa",
        "label": r"log$_{10}$ $\sigma_\mathrm{break}$",
        "unit_label": r"log$_{10}$ $\sigma_\mathrm{break}$ (Pa)",
        "display_target": "secondary check",
        "goal": "above",
        "low": math.log10(5.0),
        "weight": 0.0,
    },
    {
        "id": "break_strain",
        "column": "break_strain_pct",
        "label": r"$\gamma_\mathrm{break}$",
        "unit_label": r"$\gamma_\mathrm{break}$ (%)",
        "display_target": "secondary check",
        "goal": "above",
        "low": 40.0,
        "weight": 0.0,
    },
]

ALL_TARGETS = PRIMARY_TARGETS + SECONDARY_TARGETS
TARGET_BY_ID = {spec["id"]: spec for spec in ALL_TARGETS}

DESIGN_SCENARIOS = [
    {
        "id": "viscosity_only_eta50_200",
        "label": r"$\eta_{50}$ = 200 mPa s",
        "targets": [
            {"target_id": "eta50", "low": math.log10(170.0), "high": math.log10(240.0), "target_value": math.log10(200.0), "weight": 1.0},
        ],
    },
    {
        "id": "viscosity_only_eta50_500",
        "label": r"$\eta_{50}$ = 500 mPa s",
        "targets": [
            {"target_id": "eta50", "low": math.log10(400.0), "high": math.log10(600.0), "target_value": math.log10(500.0), "weight": 1.0},
        ],
    },
    {
        "id": "elasticity_only_Gp1Hz_100",
        "label": r"$G'$ 1 Hz = 100 Pa",
        "targets": [
            {"target_id": "Gp1Hz", "low": math.log10(75.0), "high": math.log10(150.0), "target_value": math.log10(100.0), "weight": 1.0},
        ],
    },
    {
        "id": "elasticity_only_Gp1Hz_500",
        "label": r"$G'$ 1 Hz = 500 Pa",
        "targets": [
            {"target_id": "Gp1Hz", "low": math.log10(400.0), "high": math.log10(650.0), "target_value": math.log10(500.0), "weight": 1.0},
        ],
    },
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.4,
            "axes.linewidth": 0.85,
            "savefig.bbox": "tight",
        }
    )


def make_gpr(n_features: int, random_state: int = 42) -> Pipeline:
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
                    n_restarts_optimizer=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def predict_mean_std(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = model.named_steps["x_scaler"]
    gpr = model.named_steps["gpr"]
    mean, std = gpr.predict(scaler.transform(x), return_std=True)
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def load_formulation_data() -> pd.DataFrame:
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    rep["log10_break_stress_Pa"] = np.where(rep["break_stress_Pa"] > 0, np.log10(rep["break_stress_Pa"]), np.nan)
    df = rep.groupby(["split", "formulation_std"], as_index=False).mean(numeric_only=True)
    return df


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def probability(mean: np.ndarray, std: np.ndarray, spec: dict) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    mean = np.asarray(mean, dtype=float)
    if spec["goal"] == "between":
        return norm.cdf((spec["high"] - mean) / std) - norm.cdf((spec["low"] - mean) / std)
    if spec["goal"] == "above":
        return 1 - norm.cdf((spec["low"] - mean) / std)
    raise ValueError(spec["goal"])


def minmax01(x: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def diverse_top(df: pd.DataFrame, score_col: str, n: int = 20, min_dist: float = 0.055) -> pd.DataFrame:
    chosen = []
    work = df.sort_values(score_col, ascending=False).copy()
    mins = work[FEATURES].min()
    spans = (work[FEATURES].max() - mins).replace(0, 1)
    for _, row in work.iterrows():
        point = ((row[FEATURES] - mins) / spans).to_numpy(dtype=float)
        if chosen:
            prev = np.vstack([((r[FEATURES] - mins) / spans).to_numpy(dtype=float) for r in chosen])
            if np.sqrt(((prev - point) ** 2).sum(axis=1)).min() < min_dist:
                continue
        chosen.append(row)
        if len(chosen) >= n:
            break
    return pd.DataFrame(chosen)


def validate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"].copy()
    external = df[df["split"] == "predict"].copy()
    rows = []
    preds = []
    for spec in ALL_TARGETS:
        target = spec["column"]
        train_target = train.dropna(subset=FEATURES + [target]).copy()
        external_target = external.dropna(subset=FEATURES + [target]).copy()

        cv_pred = np.full(len(train_target), np.nan)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train_target), start=1):
            model = make_gpr(len(FEATURES), random_state=40 + fold)
            model.fit(train_target.iloc[tr_idx][FEATURES], train_target.iloc[tr_idx][target])
            mean, std = predict_mean_std(model, train_target.iloc[va_idx][FEATURES])
            cv_pred[va_idx] = mean
            for local_idx, m, s in zip(va_idx, mean, std):
                r = train_target.iloc[local_idx]
                preds.append(
                    {
                        "target_id": spec["id"],
                        "target": target,
                        "split_eval": "internal_5fold_cv",
                        "formulation_std": r["formulation_std"],
                        "yp_pct": r["yp_pct"],
                        "xanthan_pct": r["xanthan_pct"],
                        "measured": r[target],
                        "predicted": float(m),
                        "std": float(s),
                        "residual": float(m - r[target]),
                    }
                )
        rows.append({"target_id": spec["id"], "target": target, "split_eval": "internal_5fold_cv", **metric_row(train_target[target], cv_pred)})

        model = make_gpr(len(FEATURES), random_state=42)
        model.fit(train_target[FEATURES], train_target[target])
        mean_train, std_train = predict_mean_std(model, train_target[FEATURES])
        rows.append({"target_id": spec["id"], "target": target, "split_eval": "train_refit", **metric_row(train_target[target], mean_train)})
        for _, r in train_target.iterrows():
            pass
        if len(external_target):
            mean_ext, std_ext = predict_mean_std(model, external_target[FEATURES])
            rows.append({"target_id": spec["id"], "target": target, "split_eval": "external_validation", **metric_row(external_target[target], mean_ext)})
            for (_, r), m, s in zip(external_target.iterrows(), mean_ext, std_ext):
                preds.append(
                    {
                        "target_id": spec["id"],
                        "target": target,
                        "split_eval": "external_validation",
                        "formulation_std": r["formulation_std"],
                        "yp_pct": r["yp_pct"],
                        "xanthan_pct": r["xanthan_pct"],
                        "measured": r[target],
                        "predicted": float(m),
                        "std": float(s),
                        "residual": float(m - r[target]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(preds)


def build_design_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    yp_grid = np.round(np.arange(0, 30.0001, 0.25), 4)
    xg_grid = np.round(np.arange(0.25, 1.0001, 0.005), 4)
    grid = pd.DataFrame([(yp, xg) for yp in yp_grid for xg in xg_grid], columns=FEATURES)
    models = {}
    component_cols = []
    uncertainty_cols = []
    weights = []
    for spec in ALL_TARGETS:
        target = spec["column"]
        d = df.dropna(subset=FEATURES + [target]).copy()
        model = make_gpr(len(FEATURES), random_state=42)
        model.fit(d[FEATURES], d[target])
        mean, std = predict_mean_std(model, grid[FEATURES])
        models[spec["id"]] = model
        grid[f"pred_{target}"] = mean
        grid[f"std_{target}"] = std
        uncertainty_cols.append(f"std_{target}")
        if spec in PRIMARY_TARGETS:
            grid[f"p_{spec['id']}"] = probability(mean, std, spec)
            grid[f"closeness_{spec['id']}"] = np.exp(-0.5 * ((mean - spec["target_value"]) / np.maximum(std, 1e-8)) ** 2)
            component_cols.append(f"p_{spec['id']}")
            weights.append(spec["weight"])

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    probs = np.clip(grid[component_cols].to_numpy(dtype=float), 1e-8, 1.0)
    grid["posterior_feasibility"] = np.exp((np.log(probs) * weights).sum(axis=1))
    closeness = np.clip(np.column_stack([grid[f"closeness_{spec['id']}"] for spec in PRIMARY_TARGETS]), 1e-8, 1.0)
    grid["target_closeness"] = np.exp(np.log(closeness).mean(axis=1))
    grid["uncertainty_index"] = np.mean([minmax01(grid[c]) for c in uncertainty_cols], axis=0)
    grid["protein_desirability"] = grid["yp_pct"] / grid["yp_pct"].max()
    grid["xanthan_sparing"] = 1 - (grid["xanthan_pct"] - grid["xanthan_pct"].min()) / (grid["xanthan_pct"].max() - grid["xanthan_pct"].min())
    grid["posterior_design_score"] = grid["posterior_feasibility"] * (
        0.78 + 0.14 * grid["protein_desirability"] + 0.08 * grid["xanthan_sparing"]
    )
    grid["knowledge_supported_score"] = grid["posterior_design_score"] * (1 - 0.25 * grid["uncertainty_index"])
    add_design_scenarios(grid)
    return grid, models


def add_design_scenarios(grid: pd.DataFrame) -> None:
    score_cols = []
    for scenario in DESIGN_SCENARIOS:
        p_cols = []
        c_cols = []
        weights = []
        for item in scenario["targets"]:
            spec = TARGET_BY_ID[item["target_id"]]
            mean = grid[f"pred_{spec['column']}"].to_numpy(dtype=float)
            std = grid[f"std_{spec['column']}"].to_numpy(dtype=float)
            local_spec = {**spec, **item}
            p_col = f"scenario_{scenario['id']}_p_{spec['id']}"
            c_col = f"scenario_{scenario['id']}_closeness_{spec['id']}"
            grid[p_col] = probability(mean, std, local_spec)
            grid[c_col] = np.exp(-0.5 * ((mean - item["target_value"]) / np.maximum(std, 1e-8)) ** 2)
            p_cols.append(p_col)
            c_cols.append(c_col)
            weights.append(float(item.get("weight", 1.0)))
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        p = np.clip(grid[p_cols].to_numpy(dtype=float), 1e-8, 1.0)
        c = np.clip(grid[c_cols].to_numpy(dtype=float), 1e-8, 1.0)
        grid[f"scenario_{scenario['id']}_feasibility"] = np.exp((np.log(p) * weights).sum(axis=1))
        grid[f"scenario_{scenario['id']}_closeness"] = np.exp((np.log(c) * weights).sum(axis=1))
        grid[f"scenario_{scenario['id']}_score"] = grid[f"scenario_{scenario['id']}_feasibility"] * (
            0.80 + 0.12 * grid["protein_desirability"] + 0.08 * grid["xanthan_sparing"]
        ) * (1 - 0.20 * grid["uncertainty_index"])
        score_cols.append(f"scenario_{scenario['id']}_score")
    grid["target_library_score"] = grid[score_cols].max(axis=1)


def posterior_samples(grid: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    w = np.clip(grid["posterior_design_score"].to_numpy(dtype=float), 0, None)
    w = w / w.sum()
    idx = rng.choice(np.arange(len(grid)), size=n, replace=True, p=w)
    return grid.iloc[idx].reset_index(drop=True)


def overlay_points(ax: plt.Axes, df: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    external = df[df["split"] == "predict"]
    ax.scatter(
        train["xanthan_pct"],
        train["yp_pct"],
        s=30,
        marker="o",
        facecolors="#3F3F3F",
        edgecolors="white",
        linewidths=0.55,
        alpha=0.82,
        label="internal/train",
        zorder=5,
    )
    ax.scatter(
        external["xanthan_pct"],
        external["yp_pct"],
        marker="D",
        s=58,
        facecolors="#2F80C1",
        edgecolors="white",
        linewidths=0.85,
        alpha=0.98,
        label="external",
        zorder=6,
    )


def finish_design_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.225, 1.025)
    ax.set_ylim(-1.0, 31.0)
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.grid(color="white", linewidth=0.35, alpha=0.25)


def add_candidates_and_boundary(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    candidates: pd.DataFrame | None,
    boundary_fraction: float = 0.72,
    add_label: bool = True,
) -> None:
    finite = z[np.isfinite(z)]
    if finite.size:
        level = float(np.nanmax(finite) * boundary_fraction)
        if np.nanmin(finite) < level < np.nanmax(finite):
            ax.contour(x, y, z, levels=[level], colors="white", linewidths=2.4, alpha=0.95, zorder=4)
            ax.contour(x, y, z, levels=[level], colors="#1A1A1A", linewidths=0.8, alpha=0.95, zorder=4)
    if candidates is None or candidates.empty:
        return
    top = candidates.head(12)
    ax.scatter(
        top["xanthan_pct"],
        top["yp_pct"],
        s=92,
        facecolors="none",
        edgecolors="white",
        linewidths=1.7,
        label="candidate region" if add_label else None,
        zorder=7,
    )
    ax.scatter(top["xanthan_pct"], top["yp_pct"], s=118, facecolors="none", edgecolors="#202020", linewidths=0.55, zorder=7)
    best = candidates.iloc[0]
    ax.scatter(
        [best["xanthan_pct"]],
        [best["yp_pct"]],
        marker="*",
        s=280,
        facecolors="#D62728",
        edgecolors="white",
        linewidths=1.0,
        label="best candidate" if add_label else None,
        zorder=8,
    )


def plot_heatmap(grid: pd.DataFrame, df: pd.DataFrame, value_col: str, title: str, filename: str, candidates: pd.DataFrame | None = None) -> None:
    pivot = grid.pivot_table(index="yp_pct", columns="xanthan_pct", values=value_col, aggfunc="mean").sort_index()
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="magma",
    )
    overlay_points(ax, df)
    add_candidates_and_boundary(ax, x, y, z, candidates)
    ax.set_xlabel("xanthan gum (%)", fontweight="bold")
    ax.set_ylabel("yeast protein (%)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    finish_design_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(value_col)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        frameon=True,
        fontsize=7.4,
        handlelength=1.1,
        columnspacing=1.0,
    )
    fig.savefig(OUT_DIR / filename, dpi=DPI)
    fig.savefig(OUT_DIR / filename.replace(".png", ".pdf"))
    plt.close(fig)


def plot_scenario_summary(grid: pd.DataFrame, df: pd.DataFrame, scenario_candidates: dict[str, pd.DataFrame]) -> None:
    ncols = 2
    nrows = int(math.ceil(len(DESIGN_SCENARIOS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 8.6), constrained_layout=True)
    axes = np.ravel(axes)
    for i, (ax, scenario) in enumerate(zip(axes, DESIGN_SCENARIOS)):
        value_col = f"scenario_{scenario['id']}_score"
        pivot = grid.pivot_table(index="yp_pct", columns="xanthan_pct", values=value_col, aggfunc="mean").sort_index()
        x = pivot.columns.to_numpy(dtype=float)
        y = pivot.index.to_numpy(dtype=float)
        z = pivot.to_numpy(dtype=float)
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="magma",
        )
        overlay_points(ax, df)
        candidates = scenario_candidates[scenario["id"]].head(10)
        add_candidates_and_boundary(ax, x, y, z, candidates, add_label=False)
        ax.text(
            -0.12,
            1.06,
            chr(ord("A") + i),
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.set_xlabel("xanthan gum (%)", fontweight="bold")
        ax.set_ylabel("yeast protein (%)", fontweight="bold")
        ax.set_title(scenario["label"], fontweight="bold")
        finish_design_axes(ax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("scenario score")
    for ax in axes[len(DESIGN_SCENARIOS):]:
        ax.axis("off")
    fig.savefig(OUT_DIR / "Figure_inverse_design_target_library.png", dpi=DPI)
    fig.savefig(OUT_DIR / "Figure_inverse_design_target_library.pdf")
    plt.close(fig)


def plot_target_space(grid: pd.DataFrame, df: pd.DataFrame, scenario_candidates: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    sc = ax.scatter(
        10 ** grid["pred_log10_eta_50"],
        10 ** grid["pred_log10_Gp_1Hz"],
        c=grid["target_library_score"],
        s=10,
        cmap="magma",
        alpha=0.45,
        linewidths=0,
        label="predicted formulation grid",
        zorder=1,
    )
    train = df[df["split"] == "train"]
    external = df[df["split"] == "predict"]
    ax.scatter(train["eta_50_Pa_s"], train["Gp_1Hz_Pa"], s=38, marker="o", facecolors="#3F3F3F", edgecolors="white", linewidths=0.55, label="internal/train", zorder=3)
    ax.scatter(external["eta_50_Pa_s"], external["Gp_1Hz_Pa"], s=70, marker="D", facecolors="#2F80C1", edgecolors="white", linewidths=0.85, label="external", zorder=4)
    for scenario in DESIGN_SCENARIOS:
        best = scenario_candidates[scenario["id"]].iloc[0]
        ax.scatter(
            [10 ** best["pred_log10_eta_50"]],
            [10 ** best["pred_log10_Gp_1Hz"]],
            marker="*",
            s=180,
            edgecolors="white",
            linewidths=0.8,
            label=scenario["label"],
            zorder=6,
        )
    for low, high, target in [(170, 240, 200), (400, 600, 500)]:
        ax.axvspan(low, high, color="#F4B183", alpha=0.14)
        ax.axvline(target, color="#B45A4D", linewidth=1.2)
    for low, high, target in [(75, 150, 100), (400, 650, 500)]:
        ax.axhspan(low, high, color="#8FB8DE", alpha=0.12)
        ax.axhline(target, color="#4C78A8", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta_{50}$ (mPa s)", fontweight="bold")
    ax.set_ylabel(r"$G'$ at 1 Hz (Pa)", fontweight="bold")
    ax.set_title(r"Target-space view: independent viscosity and elasticity targets", fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("best target-library score")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
        fontsize=7.2,
        handlelength=1.2,
        columnspacing=1.0,
    )
    ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.55, which="both")
    fig.savefig(OUT_DIR / "Figure_inverse_design_target_space.png", dpi=DPI)
    fig.savefig(OUT_DIR / "Figure_inverse_design_target_space.pdf")
    plt.close(fig)


def plot_target_library_contours(grid: pd.DataFrame, df: pd.DataFrame, scenario_candidates: dict[str, pd.DataFrame]) -> None:
    pivot = grid.pivot_table(index="yp_pct", columns="xanthan_pct", values="target_library_score", aggfunc="mean").sort_index()
    eta = grid.pivot_table(index="yp_pct", columns="xanthan_pct", values="pred_log10_eta_50", aggfunc="mean").reindex(index=pivot.index, columns=pivot.columns).to_numpy()
    gp = grid.pivot_table(index="yp_pct", columns="xanthan_pct", values="pred_log10_Gp_1Hz", aggfunc="mean").reindex(index=pivot.index, columns=pivot.columns).to_numpy()
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    im = ax.imshow(z, origin="lower", aspect="auto", extent=[x.min(), x.max(), y.min(), y.max()], cmap="magma")
    overlay_points(ax, df)
    ax.contour(x, y, eta, levels=[math.log10(200), math.log10(500)], colors=["#C25B33", "#7F3C22"], linewidths=2.0, zorder=6)
    ax.contour(x, y, gp, levels=[math.log10(100), math.log10(500)], colors=["#4C78A8", "#1F4E79"], linewidths=2.0, zorder=6)
    for scenario in DESIGN_SCENARIOS:
        candidates = scenario_candidates[scenario["id"]].head(5)
        ax.scatter(
            candidates["xanthan_pct"],
            candidates["yp_pct"],
            s=78,
            facecolors="none",
            edgecolors="white",
            linewidths=1.4,
            zorder=7,
        )
        best = candidates.iloc[0]
        ax.scatter(
            [best["xanthan_pct"]],
            [best["yp_pct"]],
            marker="*",
            s=210,
            edgecolors="white",
            linewidths=0.8,
            label=scenario["label"],
            zorder=8,
        )
    ax.set_xlabel("xanthan gum (%)", fontweight="bold")
    ax.set_ylabel("yeast protein (%)", fontweight="bold")
    ax.set_title(r"Inverse-design target library: viscosity and $G'$ contours", fontweight="bold")
    finish_design_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("best target-library score")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=True,
        fontsize=7.6,
        handlelength=1.1,
        columnspacing=1.0,
    )
    fig.savefig(OUT_DIR / "Figure_inverse_design_target_library_contours.png", dpi=DPI)
    fig.savefig(OUT_DIR / "Figure_inverse_design_target_library_contours.pdf")
    plt.close(fig)


def plot_validation_parity(preds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), constrained_layout=True)
    axes = axes.ravel()
    letters = list("ABCD")
    for ax, spec, letter in zip(axes, ALL_TARGETS, letters):
        d = preds[preds["target_id"] == spec["id"]].copy()
        colors = {"internal_5fold_cv": "#555555", "external_validation": "#1f77b4"}
        for split_eval, sub in d.groupby("split_eval"):
            ax.scatter(sub["measured"], sub["predicted"], s=44, color=colors.get(split_eval, "#999999"), edgecolor="white", linewidth=0.5, alpha=0.9, label=split_eval.replace("_", " "))
        lo = float(np.nanmin([d["measured"].min(), d["predicted"].min()]))
        hi = float(np.nanmax([d["measured"].max(), d["predicted"].max()]))
        pad = (hi - lo) * 0.08 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle=(0, (4, 3)), linewidth=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Measured {spec['unit_label']}", fontweight="bold")
        ax.set_ylabel(f"Predicted {spec['unit_label']}", fontweight="bold")
        ax.set_title(spec["label"], fontweight="bold")
        ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", clip_on=False)
        ax.grid(color="#D9D9D9", linewidth=0.65, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", frameon=True)
    fig.savefig(OUT_DIR / "validation_internal_external_parity.png", dpi=DPI)
    fig.savefig(OUT_DIR / "validation_internal_external_parity.pdf")
    plt.close(fig)


def plot_external_residual_map(df: pd.DataFrame, preds: pd.DataFrame) -> None:
    external = preds[preds["split_eval"] == "external_validation"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.6), constrained_layout=True)
    for ax, spec in zip(axes.ravel(), ALL_TARGETS):
        d = external[external["target_id"] == spec["id"]]
        overlay_points(ax, df)
        if len(d):
            scale = np.nanmax(np.abs(d["residual"])) or 1
            sc = ax.scatter(d["xanthan_pct"], d["yp_pct"], c=d["residual"], cmap="coolwarm", vmin=-scale, vmax=scale, marker="s", s=92, edgecolor="black", linewidth=0.5, zorder=8)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("prediction - measured")
        ax.set_xlabel("xanthan gum (%)", fontweight="bold")
        ax.set_ylabel("yeast protein (%)", fontweight="bold")
        ax.set_title(f"External residuals: {spec['label']}", fontweight="bold")
        ax.grid(color="#D9D9D9", linewidth=0.55, alpha=0.55)
    fig.savefig(OUT_DIR / "external_validation_residual_locations.png", dpi=DPI)
    fig.savefig(OUT_DIR / "external_validation_residual_locations.pdf")
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, grid: pd.DataFrame, candidates: pd.DataFrame, samples: pd.DataFrame) -> None:
    target_rows = []
    for spec in PRIMARY_TARGETS:
        target_rows.append(
            {
                "target_id": spec["id"],
                "target_column": spec["column"],
                "goal": spec["goal"],
                "target_value": spec.get("target_value", np.nan),
                "target_display_value": spec.get("target_display_value", ""),
                "low": spec.get("low"),
                "high": spec.get("high", np.nan),
                "weight": spec["weight"],
            }
        )
    pd.DataFrame(target_rows).to_csv(OUT_DIR / "primary_inverse_design_target_windows.csv", index=False)
    scenario_rows = []
    for scenario in DESIGN_SCENARIOS:
        for item in scenario["targets"]:
            spec = TARGET_BY_ID[item["target_id"]]
            scenario_rows.append(
                {
                    "scenario_id": scenario["id"],
                    "scenario_label": scenario["label"],
                    "target_id": spec["id"],
                    "target_column": spec["column"],
                    "target_value": item["target_value"],
                    "low": item["low"],
                    "high": item["high"],
                    "weight": item.get("weight", 1.0),
                }
            )
    pd.DataFrame(scenario_rows).to_csv(OUT_DIR / "inverse_design_scenario_target_windows.csv", index=False)
    metrics.to_csv(OUT_DIR / "surrogate_validation_metrics.csv", index=False)
    candidates.to_csv(OUT_DIR / "posterior_top_design_candidates.csv", index=False)
    samples.to_csv(OUT_DIR / "posterior_design_samples.csv", index=False)
    summary = {
        "run_id": RUN_ID,
        "method": "Small-data GPR inverse design for two primary paper-facing targets with retrospective internal/external validation overlays and optional experimental validation candidates. Candidate ranking uses the true scenario score without a measured-point penalty.",
        "paper_inspired_elements": [
            "Gaussian-process surrogates with uncertainty",
            "Bayesian inverse-design posterior over formulation grid",
            "multiple feasible pathways via posterior samples rather than a single optimum",
            "breaking stress and breaking strain retained as secondary mechanical sanity checks",
        ],
        "validation_display": [
            "internal 5-fold CV parity",
            "external validation parity",
            "external residual location map",
            "all internal/train and external points overlaid on inverse-design heatmaps",
        ],
        "outputs": [
            "posterior_feasibility_map.png/pdf",
            "posterior_design_score_map.png/pdf",
            "Figure_inverse_design_target_library.png/pdf",
            "Figure_inverse_design_target_space.png/pdf",
            "Figure_inverse_design_target_library_contours.png/pdf",
            "scenario_<id>_score_map.png/pdf",
            "knowledge_supported_score_map.png/pdf",
            "uncertainty_index_map.png/pdf",
            "validation_internal_external_parity.png/pdf",
            "external_validation_residual_locations.png/pdf",
            "posterior_top_design_candidates.csv",
            "recommended_experimental_validation_candidates.csv",
            "posterior_design_samples.csv",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    df = load_formulation_data()
    metrics, preds = validate_models(df)
    preds.to_csv(OUT_DIR / "surrogate_validation_predictions.csv", index=False)
    grid, _ = build_design_grid(df)
    grid.to_csv(OUT_DIR / "bayesian_inverse_design_grid.csv", index=False)
    candidates = diverse_top(grid, "knowledge_supported_score", n=30)
    samples = posterior_samples(grid, n=5000)
    scenario_candidates = {}
    for scenario in DESIGN_SCENARIOS:
        score_col = f"scenario_{scenario['id']}_score"
        scenario_candidates[scenario["id"]] = diverse_top(grid, score_col, n=30)
        scenario_candidates[scenario["id"]].to_csv(OUT_DIR / f"{scenario['id']}_top_candidates.csv", index=False)
        plot_heatmap(grid, df, score_col, scenario["label"], f"{scenario['id']}_score_map.png", scenario_candidates[scenario["id"]])
        plot_heatmap(grid, df, f"scenario_{scenario['id']}_feasibility", f"Feasibility: {scenario['label']}", f"{scenario['id']}_feasibility_map.png", scenario_candidates[scenario["id"]])
    plot_scenario_summary(grid, df, scenario_candidates)
    plot_target_space(grid, df, scenario_candidates)
    plot_target_library_contours(grid, df, scenario_candidates)

    plot_heatmap(grid, df, "posterior_feasibility", "Posterior feasibility from GPR uncertainty", "posterior_feasibility_map.png", candidates)
    plot_heatmap(grid, df, "posterior_design_score", "Bayesian inverse-design posterior score", "posterior_design_score_map.png", candidates)
    plot_heatmap(grid, df, "target_closeness", "Closeness to eta50 and G' targets", "target_closeness_map.png", candidates)
    plot_heatmap(grid, df, "knowledge_supported_score", "Knowledge-supported design score", "knowledge_supported_score_map.png", candidates)
    plot_heatmap(grid, df, "uncertainty_index", "GPR uncertainty index", "uncertainty_index_map.png", candidates)
    for spec in ALL_TARGETS:
        plot_heatmap(grid, df, f"pred_{spec['column']}", f"Predicted {spec['label']}", f"predicted_{spec['id']}_map.png", candidates)
        if spec in PRIMARY_TARGETS:
            plot_heatmap(grid, df, f"p_{spec['id']}", f"Probability target is satisfied: {spec['display_target']}", f"probability_{spec['id']}_map.png", candidates)
            plot_heatmap(grid, df, f"closeness_{spec['id']}", f"Target closeness: {spec['display_target']}", f"target_closeness_{spec['id']}_map.png", candidates)

    plot_validation_parity(preds)
    plot_external_residual_map(df, preds)
    recommended_rows = []
    for scenario in DESIGN_SCENARIOS:
        top = scenario_candidates[scenario["id"]].head(5).copy()
        top.insert(0, "scenario_label", scenario["label"])
        top.insert(0, "scenario_id", scenario["id"])
        recommended_rows.append(top)
    recommended = pd.concat(recommended_rows, ignore_index=True)
    recommended["pred_eta50_mPa_s"] = 10 ** recommended["pred_log10_eta_50"]
    recommended["pred_Gp1Hz_Pa"] = 10 ** recommended["pred_log10_Gp_1Hz"]
    recommended["pred_break_stress_Pa"] = 10 ** recommended["pred_log10_break_stress_Pa"]
    recommended.to_csv(OUT_DIR / "recommended_experimental_validation_candidates.csv", index=False)
    write_summary(metrics, grid, candidates, samples)
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)

    onedrive_dir = ONEDRIVE_ROOT / "time_lapse" / OUT_DIR.name
    if onedrive_dir.exists():
        shutil.rmtree(onedrive_dir)
    shutil.copytree(OUT_DIR, onedrive_dir)
    print(OUT_DIR)
    print(onedrive_dir)


if __name__ == "__main__":
    main()
