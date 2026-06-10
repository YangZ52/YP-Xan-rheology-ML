from __future__ import annotations

import json
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ARCHIVE_ROOT = Path(os.environ.get("RHEOLOGY_ARCHIVE_ROOT", ROOT / "outputs"))
BENCHMARK = OUTPUTS / "ML_results_xanthan_positive_20260530_131336"
DATA_DIR = OUTPUTS / "ml_ready_xanthan_positive_20260529"
RUN_ID = os.environ.get("RHEOLOGY_LENGTHSCALE_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUTPUTS / f"main_target_gpr_lengthscales_{RUN_ID}"
DPI = 600

TARGETS = [
    ("saos_scalar", "log10_Gp_1Hz", r"$G'$ at 1 Hz"),
    ("viscosity_scalar", "log10_eta_50", r"$\eta_{50}$"),
    ("strain_from_formulation", "log10_break_stress_Pa", r"$\sigma_\mathrm{break}$"),
    ("strain_from_formulation", "break_strain_pct", r"$\gamma_\mathrm{break}$"),
]

FEATURE_LABELS = {
    "yp_pct": "Yeast protein (wt%)",
    "xanthan_pct": "Xanthan gum (wt%)",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 19,
            "axes.titlesize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def formulation_level_master() -> pd.DataFrame:
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    rep["log10_break_stress_Pa"] = np.where(rep["break_stress_Pa"] > 0, np.log10(rep["break_stress_Pa"]), np.nan)
    numeric_cols = rep.select_dtypes(include=[np.number]).columns.tolist()
    return rep.groupby(["split", "formulation_std"], as_index=False)[numeric_cols].mean()


def balanced_grid_splits(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    xg_levels = sorted(train["xanthan_pct"].unique())
    yp_levels = sorted(train["yp_pct"].unique())
    fold_by_index = {}
    for i, xg in enumerate(xg_levels):
        for j, yp in enumerate(yp_levels):
            matches = train.index[(train["xanthan_pct"] == xg) & (train["yp_pct"] == yp)].tolist()
            if len(matches) != 1:
                raise ValueError("Balanced grid CV requires one training formulation at every xanthan x protein grid point.")
            fold_by_index[matches[0]] = (i + j) % 5

    all_idx = np.asarray(train.index.tolist())
    return [
        (
            np.asarray([idx for idx in all_idx if fold_by_index[idx] != fold]),
            np.asarray([idx for idx in all_idx if fold_by_index[idx] == fold]),
        )
        for fold in range(5)
    ]


def make_gpr(n_features: int, random_state: int) -> Pipeline:
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
    relevance = 1 / length_scales
    relevance = relevance / relevance.sum()
    return [
        {
            "feature": feat,
            "length_scale_standardized": float(length_scale),
            "relative_relevance": float(rel),
        }
        for feat, length_scale, rel in zip(features, length_scales, relevance)
    ]


def compute_balanced_cv_lengthscales() -> pd.DataFrame:
    data = formulation_level_master()
    features = ["yp_pct", "xanthan_pct"]
    rows = []
    for order, (task, target, label) in enumerate(TARGETS):
        needed = ["split", "formulation_std", target, *features]
        train = data.loc[data["split"] == "train", needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
        train = train.sort_values(["xanthan_pct", "yp_pct"]).reset_index(drop=True)
        for fold, (tr_idx, _va_idx) in enumerate(balanced_grid_splits(train), start=1):
            model = make_gpr(len(features), random_state=42 + fold)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train.iloc[tr_idx][features], train.iloc[tr_idx][target])
            for length_row in extract_gpr_lengthscales(model, features):
                rows.append(
                    {
                        "task": task,
                        "target": target,
                        "feature_set": ",".join(features),
                        "fold": fold,
                        "target_order": order,
                        "target_label": label,
                        "feature_label": FEATURE_LABELS[length_row["feature"]],
                        **length_row,
                    }
                )

    return pd.DataFrame(rows)


def summarize_lengths(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["task", "target", "feature_set", "target_order", "target_label", "feature", "feature_label"]
    out = (
        fold_df.groupby(group_cols, as_index=False)
        .agg(
            length_scale_standardized=("length_scale_standardized", "mean"),
            length_scale_sd=("length_scale_standardized", "std"),
            length_scale_sem=("length_scale_standardized", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
            relative_relevance=("relative_relevance", "mean"),
            relative_relevance_sd=("relative_relevance", "std"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["target_order", "feature"])
    )
    out["relative_relevance_pct"] = 100 * out["relative_relevance"]
    out["relative_relevance_sd_pct"] = 100 * out["relative_relevance_sd"]
    return out


def fmt_num(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot_lengthscale_figure(df: pd.DataFrame, out_dir: Path) -> None:
    targets = [label for _, _, label in TARGETS]
    features = ["yp_pct", "xanthan_pct"]
    feature_labels = [FEATURE_LABELS[f] for f in features]
    heatmap_feature_labels = ["Yeast protein\n(wt%)", "Xanthan gum\n(wt%)"]

    length = df.pivot(index="target_label", columns="feature", values="length_scale_standardized").reindex(targets)[features]
    length_sd = df.pivot(index="target_label", columns="feature", values="length_scale_sd").reindex(targets)[features]
    rel = df.pivot(index="target_label", columns="feature", values="relative_relevance_pct").reindex(targets)[features]

    fig = plt.figure(figsize=(13.2, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.36, left=0.12, right=0.98, bottom=0.14, top=0.86)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    y = np.arange(len(targets))
    colors = {"yp_pct": "#52789F", "xanthan_pct": "#C58B57"}
    offsets = [-0.17, 0.17]
    height = 0.28

    for off, feat in zip(offsets, features):
        vals = length[feat].to_numpy(dtype=float)
        err = length_sd[feat].to_numpy(dtype=float)
        xerr_lower = np.minimum(err, vals * 0.88)
        xerr_upper = err
        bars = ax0.barh(
            y + off,
            vals,
            xerr=np.vstack([xerr_lower, xerr_upper]),
            height=height,
            color=colors[feat],
            edgecolor="white",
            linewidth=0.8,
            error_kw={"ecolor": "#222222", "elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
            label=FEATURE_LABELS[feat],
        )
        for bar, val, upper in zip(bars, vals, xerr_upper):
            ax0.text((val + upper) * 1.10, bar.get_y() + bar.get_height() / 2, fmt_num(val), va="center", ha="left", fontsize=17)

    ax0.set_xscale("log")
    ax0.set_yticks(y)
    ax0.set_yticklabels(targets)
    ax0.invert_yaxis()
    ax0.set_xlabel("Standardized GPR length scale")
    ax0.set_ylabel("Target variable")
    ax0.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax0.text(-0.18, 1.04, "A", transform=ax0.transAxes, fontsize=24, fontweight="bold", va="top")

    heatmap_cmap = LinearSegmentedColormap.from_list(
        "paper_gold_green",
        ["#FFF7D6", "#F2E890", "#BDD77B", "#74B77A", "#2F7F73"],
    )
    im = ax1.imshow(rel.to_numpy(dtype=float), cmap=heatmap_cmap, vmin=0, vmax=100, aspect="auto")
    ax1.set_xticks(np.arange(len(features)))
    ax1.set_xticklabels(heatmap_feature_labels)
    ax1.set_yticks(np.arange(len(targets)))
    ax1.set_yticklabels(targets)
    ax1.set_xlabel("Formulation input")
    ax1.set_ylabel("")
    ax1.text(-0.18, 1.04, "B", transform=ax1.transAxes, fontsize=24, fontweight="bold", va="top")
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            value = rel.iloc[i, j]
            text_color = "white" if value >= 70 else "black"
            ax1.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=19, color=text_color)
    ax1.set_xticks(np.arange(-0.5, len(features), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
    ax1.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax1.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.035)
    cbar.set_label("relative relevance (%)")
    cbar.ax.tick_params(labelsize=17)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.31, 0.985))

    for ax in [ax0, ax1]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    base = out_dir / "Figure_gpr_lengthscales_four_main_targets"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=DPI)
    fig.savefig(base.with_suffix(".tiff"), dpi=DPI)
    plt.close(fig)


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    fold_df = compute_balanced_cv_lengthscales()
    fold_df.to_csv(OUT_DIR / "gpr_lengthscales_four_main_targets_balanced_cv_folds.csv", index=False)
    df = summarize_lengths(fold_df)
    df.to_csv(OUT_DIR / "gpr_lengthscales_four_main_targets.csv", index=False)
    plot_lengthscale_figure(df, OUT_DIR)
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)
    summary = {
        "run_id": RUN_ID,
        "benchmark": str(BENCHMARK),
        "data_dir": str(DATA_DIR),
        "dpi": DPI,
        "note": "GPR ARD length scales are mean values from balanced 5-fold formulation-grid CV models. Error bars in panel A indicate SD across folds. Inputs were standardized before GPR fitting. External validation was used separately for predictive generalization.",
        "exports": sorted(p.name for p in OUT_DIR.iterdir()),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
