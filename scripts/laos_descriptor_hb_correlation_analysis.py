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
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "ml_ready_xanthan_positive_20260529"
OUT_ROOT = ROOT / "outputs"
ARCHIVE_ROOT = Path(os.environ.get("RHEOLOGY_ARCHIVE_ROOT", ROOT / "outputs"))
RUN_ID = os.environ.get("RHEOLOGY_LAOS_DESCRIPTOR_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUT_ROOT / f"LAOS_descriptor_HB_correlation_{RUN_ID}"
DPI = 600

FLOW_RATES = [0.1, 50.0, 100.0]
SAOS_FREQS = [0.01, 1.0, 6.31]
LAOS_TARGETS = [
    ("break_stress_Pa_mean", r"$\sigma_\mathrm{break}$ (Pa)", True),
    ("break_strain_pct_mean", r"$\gamma_\mathrm{break}$ (%)", False),
    ("LVR_pct_mean", "LVR (%)", False),
]

DESCRIPTOR_LABELS = {
    "log10_eta_0p1_Pa_s": r"log$_{10}$ $\eta_{0.1}$",
    "log10_eta_50_Pa_s": r"log$_{10}$ $\eta_{50}$",
    "log10_eta_100_Pa_s": r"log$_{10}$ $\eta_{100}$",
    "log10_tau0_Pa": r"log$_{10}$ $\tau_0$",
    "log10_K_Pa_sn": r"log$_{10}$ $K$",
    "HB_n": r"$n$",
    "log10_Gp_0p01Hz_Pa": r"log$_{10}$ $G'_{0.01}$",
    "log10_Gp_1Hz_Pa": r"log$_{10}$ $G'_{1}$",
    "log10_Gp_6p31Hz_Pa": r"log$_{10}$ $G'_{6.31}$",
    "log10_Gpp_0p01Hz_Pa": r"log$_{10}$ $G''_{0.01}$",
    "log10_Gpp_1Hz_Pa": r"log$_{10}$ $G''_{1}$",
    "log10_Gpp_6p31Hz_Pa": r"log$_{10}$ $G''_{6.31}$",
    "tan_delta_0p01Hz": r"tan $\delta_{0.01}$",
    "tan_delta_1Hz": r"tan $\delta_{1}$",
    "tan_delta_6p31Hz": r"tan $\delta_{6.31}$",
}

FLOW_DESCRIPTOR_COLS = [
    "log10_eta_0p1_Pa_s",
    "log10_eta_50_Pa_s",
    "log10_eta_100_Pa_s",
    "log10_tau0_Pa",
    "log10_K_Pa_sn",
    "HB_n",
]

SAOS_DESCRIPTOR_COLS = [
    "log10_Gp_0p01Hz_Pa",
    "log10_Gp_1Hz_Pa",
    "log10_Gp_6p31Hz_Pa",
    "log10_Gpp_0p01Hz_Pa",
    "log10_Gpp_1Hz_Pa",
    "log10_Gpp_6p31Hz_Pa",
    "tan_delta_0p01Hz",
    "tan_delta_1Hz",
    "tan_delta_6p31Hz",
]

DESCRIPTOR_COLS = FLOW_DESCRIPTOR_COLS + SAOS_DESCRIPTOR_COLS


def numeric_key(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def hb_model(gdot: np.ndarray, tau0: float, K: float, n: float) -> np.ndarray:
    return tau0 + K * np.power(gdot, n)


def fit_hb_curve(group: pd.DataFrame) -> dict:
    gdot = group["shear_rate"].to_numpy(dtype=float)
    stress = group["shear_stress"].to_numpy(dtype=float)
    order = np.argsort(gdot)
    gdot = gdot[order]
    stress = stress[order]
    tau_guess = max(0.0, np.percentile(stress, 5) * 0.5)
    n_guess = 0.45
    K_guess = max((np.percentile(stress, 75) - tau_guess) / np.power(np.percentile(gdot, 75), n_guess), 1e-6)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                hb_model,
                gdot,
                stress,
                p0=[tau_guess, K_guess, n_guess],
                bounds=([0.0, 1e-12, 0.01], [np.inf, np.inf, 2.0]),
                maxfev=40000,
            )
        pred = hb_model(gdot, *popt)
        ss_res = float(np.sum((stress - pred) ** 2))
        ss_tot = float(np.sum((stress - stress.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean((stress - pred) ** 2)))
        status = "ok"
    except Exception:
        popt = [np.nan, np.nan, np.nan]
        r2 = np.nan
        rmse = np.nan
        status = "failed"
    first = group.iloc[0]
    return {
        "split": first["split"],
        "sample_id_std": first["sample_id_std"],
        "formulation_std": first["formulation_std"],
        "yp_pct": first["yp_pct"],
        "xanthan_pct": first["xanthan_pct"],
        "replicate": first["replicate"],
        "HB_tau0_Pa": float(popt[0]),
        "HB_K_Pa_sn": float(popt[1]),
        "HB_n": float(popt[2]),
        "HB_R2": r2,
        "HB_RMSE_Pa": rmse,
        "HB_status": status,
        "n_points": len(group),
    }


def nearest_value(df: pd.DataFrame, value_col: str, x_col: str, target_x: float) -> float:
    d = df.iloc[(df[x_col] - target_x).abs().argsort()[:1]]
    if d.empty:
        return np.nan
    return float(d[value_col].iloc[0])


def extract_flow_descriptors() -> tuple[pd.DataFrame, pd.DataFrame]:
    visc = pd.read_csv(DATA_DIR / "viscosity_long.csv")
    hb = pd.DataFrame([fit_hb_curve(g) for _, g in visc.groupby("sample_id_std", sort=False)])
    rows = []
    for sample, g in visc.groupby("sample_id_std", sort=False):
        first = g.iloc[0]
        row = {
            "split": first["split"],
            "sample_id_std": first["sample_id_std"],
            "formulation_std": first["formulation_std"],
            "yp_pct": first["yp_pct"],
            "xanthan_pct": first["xanthan_pct"],
            "replicate": first["replicate"],
        }
        for rate in FLOW_RATES:
            key = numeric_key(rate)
            eta = nearest_value(g, "viscosity", "shear_rate", rate)
            stress = nearest_value(g, "shear_stress", "shear_rate", rate)
            row[f"eta_{key}_Pa_s"] = eta
            row[f"log10_eta_{key}_Pa_s"] = np.log10(eta) if eta > 0 else np.nan
            row[f"stress_{key}_Pa"] = stress
        rows.append(row)
    flow = pd.DataFrame(rows).merge(hb, on=["split", "sample_id_std", "formulation_std", "yp_pct", "xanthan_pct", "replicate"], how="left")
    flow["log10_tau0_Pa"] = np.where(flow["HB_tau0_Pa"] > 0, np.log10(flow["HB_tau0_Pa"]), np.nan)
    flow["log10_K_Pa_sn"] = np.where(flow["HB_K_Pa_sn"] > 0, np.log10(flow["HB_K_Pa_sn"]), np.nan)
    return flow, hb


def extract_saos_descriptors() -> pd.DataFrame:
    freq = pd.read_csv(DATA_DIR / "frequency_long.csv")
    rows = []
    for sample, g in freq.groupby("sample_id_std", sort=False):
        first = g.iloc[0]
        row = {
            "split": first["split"],
            "sample_id_std": first["sample_id_std"],
            "formulation_std": first["formulation_std"],
            "yp_pct": first["yp_pct"],
            "xanthan_pct": first["xanthan_pct"],
            "replicate": first["replicate"],
        }
        for hz in SAOS_FREQS:
            key = numeric_key(hz)
            gp = nearest_value(g, "Gp_Pa", "frequency_Hz", hz)
            gpp = nearest_value(g, "Gpp_Pa", "frequency_Hz", hz)
            tan = nearest_value(g, "tan_delta", "frequency_Hz", hz)
            row[f"Gp_{key}Hz_Pa"] = gp
            row[f"Gpp_{key}Hz_Pa"] = gpp
            row[f"tan_delta_{key}Hz"] = tan
            row[f"log10_Gp_{key}Hz_Pa"] = np.log10(gp) if gp > 0 else np.nan
            row[f"log10_Gpp_{key}Hz_Pa"] = np.log10(gpp) if gpp > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def formulation_mean(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = ["split", "formulation_std"]
    return df.groupby(keep, as_index=False)[numeric].mean()


def build_master_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flow_rep, hb_rep = extract_flow_descriptors()
    saos_rep = extract_saos_descriptors()
    strain = pd.read_csv(DATA_DIR / "strain_summary_formulation.csv")

    rep_master = flow_rep.merge(
        saos_rep,
        on=["split", "sample_id_std", "formulation_std", "yp_pct", "xanthan_pct", "replicate"],
        how="outer",
    )

    flow_form = formulation_mean(flow_rep)
    saos_form = formulation_mean(saos_rep)
    form_master = flow_form.merge(saos_form, on=["split", "formulation_std", "yp_pct", "xanthan_pct", "replicate"], how="outer")
    # formulation_mean keeps replicate as a numeric average; drop it for formulation-level analysis.
    if "replicate" in form_master.columns:
        form_master = form_master.drop(columns=["replicate"])
    form_master = form_master.merge(strain, on=["formulation_std", "yp_pct", "xanthan_pct"], how="inner")
    form_master["log10_break_stress_Pa_mean"] = np.where(
        form_master["break_stress_Pa_mean"] > 0,
        np.log10(form_master["break_stress_Pa_mean"]),
        np.nan,
    )
    return rep_master, form_master, hb_rep, strain


def correlation_table(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, target_label, log_target in LAOS_TARGETS:
        y = np.log10(master[target]) if log_target else master[target]
        for feature in DESCRIPTOR_COLS:
            d = pd.DataFrame({"x": master[feature], "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 4 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(d["x"], d["y"])
            rows.append(
                {
                    "LAOS_target": target,
                    "LAOS_target_label": target_label,
                    "feature": feature,
                    "feature_label": DESCRIPTOR_LABELS.get(feature, feature),
                    "pearson_r": r,
                    "p_value": p,
                    "n": len(d),
                    "abs_r": abs(r) if np.isfinite(r) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def feature_selection_tables(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    X = master[DESCRIPTOR_COLS].replace([np.inf, -np.inf], np.nan)
    for target, target_label, log_target in LAOS_TARGETS:
        y = np.log10(master[target]) if log_target else master[target]
        d = pd.concat([X, y.rename("target")], axis=1).dropna()
        if len(d) < 8:
            continue
        x = d[DESCRIPTOR_COLS].to_numpy(dtype=float)
        yv = d["target"].to_numpy(dtype=float)
        xs = StandardScaler().fit_transform(x)
        mi = mutual_info_regression(xs, yv, random_state=42)
        rf = RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42)
        rf.fit(xs, yv)
        for feat, mi_val, rf_val in zip(DESCRIPTOR_COLS, mi, rf.feature_importances_):
            rows.append(
                {
                    "LAOS_target": target,
                    "LAOS_target_label": target_label,
                    "feature": feat,
                    "feature_label": DESCRIPTOR_LABELS.get(feat, feat),
                    "mutual_information": float(mi_val),
                    "random_forest_importance": float(rf_val),
                    "n": len(d),
                }
            )
    return pd.DataFrame(rows)


def plot_hb_fit_qc(hb: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    axes[0].hist(hb["HB_R2"].dropna(), bins=np.linspace(0.9, 1.0, 21), color="#5F7FA3", edgecolor="white")
    axes[0].set_xlabel("HB fit $R^2$")
    axes[0].set_ylabel("Number of replicates")
    axes[0].text(-0.18, 1.05, "A", transform=axes[0].transAxes, fontsize=15, fontweight="bold", va="top")

    axes[1].scatter(hb["HB_R2"], hb["HB_RMSE_Pa"], s=38, color="#C58B57", edgecolor="black", linewidth=0.4, alpha=0.85)
    axes[1].set_xlabel("HB fit $R^2$")
    axes[1].set_ylabel("HB fit RMSE (Pa)")
    axes[1].text(-0.18, 1.05, "B", transform=axes[1].transAxes, fontsize=15, fontweight="bold", va="top")
    for ax in axes:
        ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(path.with_suffix(".png"), dpi=DPI)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".tiff"), dpi=DPI)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    pivot = corr.pivot(index="LAOS_target_label", columns="feature", values="pearson_r")
    target_order = [x[1] for x in LAOS_TARGETS]
    pivot = pivot.reindex(target_order)[DESCRIPTOR_COLS]
    fig, ax = plt.subplots(figsize=(12.8, 4.6), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(target_order)))
    ax.set_yticklabels(target_order)
    ax.set_xticks(np.arange(len(DESCRIPTOR_COLS)))
    ax.set_xticklabels([DESCRIPTOR_LABELS[c] for c in DESCRIPTOR_COLS], rotation=45, ha="right")
    ax.set_xlabel("Flow and SAOS descriptors")
    ax.set_ylabel("LAOS/fracture descriptor")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            txt = "" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.8, color="black")
    ax.set_xticks(np.arange(-0.5, len(DESCRIPTOR_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Pearson $r$")
    fig.savefig(path.with_suffix(".png"), dpi=DPI)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".tiff"), dpi=DPI)
    plt.close(fig)


def plot_top_scatter(master: pd.DataFrame, corr: pd.DataFrame, path: Path, n_panels: int = 6) -> pd.DataFrame:
    selected = (
        corr.dropna(subset=["pearson_r"])
        .sort_values(["abs_r", "p_value"], ascending=[False, True])
        .groupby("LAOS_target", group_keys=False)
        .head(2)
        .sort_values(["LAOS_target", "abs_r"], ascending=[True, False])
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.4), constrained_layout=True)
    axes = axes.ravel()
    colors = {"break_stress_Pa_mean": "#4C78A8", "break_strain_pct_mean": "#6A8F5D", "LVR_pct_mean": "#B8795B"}
    letters = list("ABCDEF")
    for ax, (_, row), letter in zip(axes, selected.iterrows(), letters):
        target = row["LAOS_target"]
        log_target = dict((t, log) for t, _, log in LAOS_TARGETS)[target]
        x = master[row["feature"]].to_numpy(dtype=float)
        y = np.log10(master[target].to_numpy(dtype=float)) if log_target else master[target].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        ax.scatter(x, y, s=48, color=colors[target], edgecolor="black", linewidth=0.5, alpha=0.85)
        if len(x) > 2:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 200)
            yy = coef[0] * xx + coef[1]
            yhat = coef[0] * x + coef[1]
            resid = y - yhat
            se = np.sqrt(np.sum(resid**2) / max(len(x) - 2, 1))
            xbar = x.mean()
            sxx = np.sum((x - xbar) ** 2)
            ci = 1.96 * se * np.sqrt(1 / len(x) + (xx - xbar) ** 2 / sxx) if sxx > 0 else 0
            ax.plot(xx, yy, color="black", linewidth=1.2)
            ax.fill_between(xx, yy - ci, yy + ci, color="black", alpha=0.12, linewidth=0)
        ylabel = row["LAOS_target_label"]
        if log_target:
            ylabel = r"log$_{10}$ " + ylabel
        ax.set_xlabel(row["feature_label"])
        ax.set_ylabel(ylabel)
        ax.text(
            0.04,
            0.96,
            f"$r$ = {row['pearson_r']:.2f}\n$p$ = {row['p_value']:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "boxstyle": "round,pad=0.22"},
        )
        ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
        ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.savefig(path.with_suffix(".png"), dpi=DPI)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".tiff"), dpi=DPI)
    plt.close(fig)
    return selected


def plot_feature_ranking(feature_rank: pd.DataFrame, corr: pd.DataFrame, path: Path) -> pd.DataFrame:
    merged = feature_rank.merge(corr[["LAOS_target", "feature", "abs_r"]], on=["LAOS_target", "feature"], how="left")
    merged["mi_rank"] = merged.groupby("LAOS_target")["mutual_information"].rank(ascending=False, method="min")
    merged["rf_rank"] = merged.groupby("LAOS_target")["random_forest_importance"].rank(ascending=False, method="min")
    merged["corr_rank"] = merged.groupby("LAOS_target")["abs_r"].rank(ascending=False, method="min")
    merged["consensus_score"] = merged[["mi_rank", "rf_rank", "corr_rank"]].mean(axis=1)
    top = merged.sort_values(["LAOS_target", "consensus_score"]).groupby("LAOS_target", as_index=False).head(6)

    targets = [x[0] for x in LAOS_TARGETS]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), constrained_layout=True)
    for ax, target, letter in zip(axes, targets, "ABC"):
        d = top[top["LAOS_target"] == target].sort_values("consensus_score", ascending=True)
        y = np.arange(len(d))
        ax.barh(y, d["consensus_score"], color="#5F7FA3", edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(d["feature_label"])
        ax.invert_yaxis()
        ax.set_xlabel("Mean rank\n(lower is better)")
        label = d["LAOS_target_label"].iloc[0] if len(d) else target
        ax.set_title(label)
        ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(path.with_suffix(".png"), dpi=DPI)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".tiff"), dpi=DPI)
    plt.close(fig)
    return merged


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    rep_master, form_master, hb_rep, _ = build_master_tables()
    corr = correlation_table(form_master)
    feature_rank = feature_selection_tables(form_master)

    rep_master.to_csv(OUT_DIR / "LAOS_descriptor_master_replicate.csv", index=False)
    form_master.to_csv(OUT_DIR / "LAOS_descriptor_master_formulation.csv", index=False)
    hb_rep.to_csv(OUT_DIR / "HB_fit_parameters_replicate.csv", index=False)
    hb_form = formulation_mean(hb_rep)
    if "replicate" in hb_form.columns:
        hb_form = hb_form.drop(columns=["replicate"])
    hb_form.to_csv(OUT_DIR / "HB_fit_parameters_formulation_mean.csv", index=False)
    corr.to_csv(OUT_DIR / "LAOS_descriptor_pearson_correlations.csv", index=False)
    feature_rank.to_csv(OUT_DIR / "LAOS_descriptor_feature_selection_MI_RF.csv", index=False)

    plot_hb_fit_qc(hb_rep, OUT_DIR / "Figure_A_HB_fit_quality")
    plot_correlation_heatmap(corr, OUT_DIR / "Figure_B_LAOS_descriptor_correlation_heatmap")
    top_scatter = plot_top_scatter(form_master, corr, OUT_DIR / "Figure_C_top_descriptor_LAOS_scatter")
    top_scatter.to_csv(OUT_DIR / "Figure_C_top_scatter_relationships.csv", index=False)
    consensus = plot_feature_ranking(feature_rank, corr, OUT_DIR / "Figure_D_LAOS_feature_selection_ranking")
    consensus.to_csv(OUT_DIR / "LAOS_descriptor_consensus_feature_ranking.csv", index=False)

    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)
    summary = {
        "run_id": RUN_ID,
        "data_dir": str(DATA_DIR),
        "n_replicate_rows": int(len(rep_master)),
        "n_formulations_with_laos": int(len(form_master)),
        "hb_fit_success_rate": float((hb_rep["HB_status"] == "ok").mean()),
        "hb_median_r2": float(hb_rep["HB_R2"].median()),
        "descriptors": {
            "flow": FLOW_DESCRIPTOR_COLS,
            "saos": SAOS_DESCRIPTOR_COLS,
            "laos_targets": [x[0] for x in LAOS_TARGETS],
        },
        "note": "Positive descriptors were log10 transformed for correlation/feature analysis where appropriate. HB was fit to shear stress vs shear rate for each replicate curve.",
        "exports": sorted(p.name for p in OUT_DIR.iterdir()),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
