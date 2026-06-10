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


ROOT = Path(__file__).resolve().parents[1]
BASE_RESULTS = Path(os.environ.get("RHEOLOGY_BASE_RESULTS", ROOT / "outputs" / "ML_results_xanthan_positive_20260529_221920"))
DATA_DIR = ROOT / "outputs" / "ml_ready_xanthan_positive_20260529"
RUN_ID = os.environ.get("RHEOLOGY_IDDSI_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / f"IDDSI_inverse_design_scenarios_{RUN_ID}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def prob_between(mean, std, lo, hi):
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    mean = np.asarray(mean, dtype=float)
    return norm.cdf((hi - mean) / std) - norm.cdf((lo - mean) / std)


def prob_above(mean, std, lo):
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    mean = np.asarray(mean, dtype=float)
    return 1 - norm.cdf((lo - mean) / std)


def prob_below(mean, std, hi):
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    mean = np.asarray(mean, dtype=float)
    return norm.cdf((hi - mean) / std)


def minmax01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def diverse_top(df, score_col, n=12, min_dist=0.075):
    chosen = []
    work = df.sort_values(score_col, ascending=False).copy()
    mins = work[["yp_pct", "xanthan_pct"]].min()
    spans = (work[["yp_pct", "xanthan_pct"]].max() - mins).replace(0, 1)
    for _, row in work.iterrows():
        point = ((row[["yp_pct", "xanthan_pct"]] - mins) / spans).to_numpy(dtype=float)
        if not chosen:
            chosen.append(row)
        else:
            prev = np.vstack([((r[["yp_pct", "xanthan_pct"]] - mins) / spans).to_numpy(dtype=float) for r in chosen])
            if np.sqrt(((prev - point) ** 2).sum(axis=1)).min() >= min_dist:
                chosen.append(row)
        if len(chosen) >= n:
            break
    return pd.DataFrame(chosen)


def score_scenario(grid, scenario):
    g = grid.copy()
    eta_col = "pred_eta_50_Pa_s"
    eta_std_col = "std_log10_eta_50"
    eta_lo_log = math.log10(scenario["eta50_cP_low"])
    eta_hi_log = math.log10(scenario["eta50_cP_high"])
    eta_center_log = math.log10(scenario["eta50_cP_target"])

    g["p_eta50"] = prob_between(g["pred_log10_eta_50"], g[eta_std_col], eta_lo_log, eta_hi_log)
    g["p_eta50_centered"] = np.exp(-0.5 * ((g["pred_log10_eta_50"] - eta_center_log) / scenario["eta50_log_tolerance"]) ** 2)
    g["p_shear_thinning"] = prob_between(
        g["pred_viscosity_shear_thinning_slope_1to100"],
        g["std_viscosity_shear_thinning_slope_1to100"],
        scenario["slope_low"],
        scenario["slope_high"],
    )
    g["p_tan_delta"] = prob_between(g["pred_tan_delta_1Hz"], g["std_tan_delta_1Hz"], scenario["tan_delta_low"], scenario["tan_delta_high"])
    g["p_break_strain"] = prob_above(g["pred_break_strain_pct"], g["std_break_strain_pct"], scenario["break_strain_min"])
    g["p_break_stress"] = prob_above(g["pred_log10_break_stress_Pa"], g["std_log10_break_stress_Pa"], math.log10(scenario["break_stress_min_Pa"]))
    g["p_eta50_not_too_high"] = prob_below(g["pred_log10_eta_50"], g[eta_std_col], eta_hi_log)

    components = ["p_eta50", "p_shear_thinning", "p_tan_delta", "p_break_strain", "p_break_stress"]
    weights = np.array([scenario["w_eta"], scenario["w_slope"], scenario["w_tan_delta"], scenario["w_break_strain"], scenario["w_break_stress"]], dtype=float)
    weights = weights / weights.sum()
    vals = np.clip(g[components].to_numpy(dtype=float), 1e-8, 1)
    g["probability_of_success"] = np.exp((np.log(vals) * weights).sum(axis=1))
    g["protein_desirability"] = g["yp_pct"] / g["yp_pct"].max()
    g["xanthan_sparing"] = 1 - (g["xanthan_pct"] - g["xanthan_pct"].min()) / (g["xanthan_pct"].max() - g["xanthan_pct"].min())
    g["target_closeness"] = g["p_eta50_centered"]
    g["scenario_score"] = (
        scenario["w_success"] * g["probability_of_success"]
        + scenario["w_protein"] * g["protein_desirability"]
        + scenario["w_xanthan_sparing"] * g["xanthan_sparing"]
        + scenario["w_target_closeness"] * g["target_closeness"]
    )
    g["scenario_score"] = g["scenario_score"] / (
        scenario["w_success"] + scenario["w_protein"] + scenario["w_xanthan_sparing"] + scenario["w_target_closeness"]
    )
    g["eta50_cP_or_mPa_s"] = g[eta_col]
    g["break_stress_Pa"] = g["pred_break_stress_Pa"]
    g["scenario"] = scenario["name"]
    g["scenario_label"] = scenario["label"]
    return g


def load_measured_formulations():
    rep = pd.read_csv(DATA_DIR / "replicate_master.csv")
    measured = rep.groupby(["split", "formulation_std"], as_index=False).mean(numeric_only=True)
    measured["eta50_cP_or_mPa_s_measured"] = measured["eta_50_Pa_s"]
    measured["break_stress_Pa_measured"] = measured["break_stress_Pa"]
    return measured


def overlay_measured(ax, measured):
    train = measured[measured["split"] == "train"]
    predict = measured[measured["split"] == "predict"]
    ax.scatter(
        train["xanthan_pct"],
        train["yp_pct"],
        marker="o",
        s=28,
        facecolors="#222222",
        edgecolors="white",
        linewidths=0.5,
        alpha=0.80,
        label="measured train",
        zorder=5,
    )
    ax.scatter(
        predict["xanthan_pct"],
        predict["yp_pct"],
        marker="s",
        s=46,
        facecolors="#1f77b4",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.95,
        label="measured external",
        zorder=6,
    )


def plot_contour(g, scenario, value_col, filename, candidates, measured):
    pivot = g.pivot_table(index="yp_pct", columns="xanthan_pct", values=value_col, aggfunc="mean").sort_index()
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    levels = 18
    cf = ax.contourf(x, y, z, levels=levels, cmap="viridis")
    cs = ax.contour(x, y, z, levels=8, colors="white", linewidths=0.45, alpha=0.65)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
    best = candidates.iloc[0]
    overlay_measured(ax, measured)
    ax.scatter(candidates["xanthan_pct"], candidates["yp_pct"], s=68, facecolors="none", edgecolors="white", linewidths=1.5, label="recommended", zorder=7)
    ax.scatter(candidates["xanthan_pct"], candidates["yp_pct"], s=94, facecolors="none", edgecolors="black", linewidths=0.45, zorder=7)
    ax.scatter([best["xanthan_pct"]], [best["yp_pct"]], marker="*", s=280, color="#D62728", edgecolor="black", linewidth=0.7, label="best candidate", zorder=8)
    ax.set_xlabel("xanthan gum [%]")
    ax.set_ylabel("yeast protein [%]")
    ax.set_title(f"{scenario['label']}: {value_col}")
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_col)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=260)
    plt.close(fig)


def main():
    grid = pd.read_csv(BASE_RESULTS / "inverse_design_full_grid.csv")
    measured = load_measured_formulations()
    scenarios = [
        {
            "name": "level3_moderately_thick_high_protein",
            "label": "IDDSI/NDD Level 3-style moderately thick",
            "eta50_cP_low": 351,
            "eta50_cP_high": 1750,
            "eta50_cP_target": 800,
            "eta50_log_tolerance": 0.22,
            "slope_low": -0.88,
            "slope_high": -0.62,
            "tan_delta_low": 0.25,
            "tan_delta_high": 0.50,
            "break_strain_min": 40,
            "break_stress_min_Pa": 5,
            "w_eta": 2.0,
            "w_slope": 1.0,
            "w_tan_delta": 1.0,
            "w_break_strain": 1.2,
            "w_break_stress": 1.2,
            "w_success": 0.55,
            "w_protein": 0.25,
            "w_xanthan_sparing": 0.15,
            "w_target_closeness": 0.05,
        },
        {
            "name": "level3_lower_xanthan_high_protein",
            "label": "Level 3-style low-xanthan high-protein",
            "eta50_cP_low": 351,
            "eta50_cP_high": 1200,
            "eta50_cP_target": 650,
            "eta50_log_tolerance": 0.20,
            "slope_low": -0.86,
            "slope_high": -0.62,
            "tan_delta_low": 0.25,
            "tan_delta_high": 0.52,
            "break_strain_min": 40,
            "break_stress_min_Pa": 5,
            "w_eta": 2.0,
            "w_slope": 1.0,
            "w_tan_delta": 0.8,
            "w_break_strain": 1.2,
            "w_break_stress": 1.0,
            "w_success": 0.50,
            "w_protein": 0.25,
            "w_xanthan_sparing": 0.22,
            "w_target_closeness": 0.03,
        },
        {
            "name": "level4_extremely_thick_high_protein",
            "label": "IDDSI/NDD Level 4-style extremely thick",
            "eta50_cP_low": 1751,
            "eta50_cP_high": 3780,
            "eta50_cP_target": 2400,
            "eta50_log_tolerance": 0.20,
            "slope_low": -0.90,
            "slope_high": -0.65,
            "tan_delta_low": 0.22,
            "tan_delta_high": 0.45,
            "break_strain_min": 30,
            "break_stress_min_Pa": 10,
            "w_eta": 2.2,
            "w_slope": 0.8,
            "w_tan_delta": 0.8,
            "w_break_strain": 0.8,
            "w_break_stress": 1.5,
            "w_success": 0.58,
            "w_protein": 0.25,
            "w_xanthan_sparing": 0.12,
            "w_target_closeness": 0.05,
        },
    ]
    pd.DataFrame(scenarios).to_csv(OUT_DIR / "iddsi_inverse_design_scenario_config.csv", index=False)

    all_scored = []
    all_recs = []
    for scenario in scenarios:
        scored = score_scenario(grid, scenario)
        all_scored.append(scored)
        top = diverse_top(scored, "scenario_score", n=15)
        top.to_csv(OUT_DIR / f"{scenario['name']}_top_candidates.csv", index=False)
        all_recs.append(top.head(8).copy())
        plot_contour(scored, scenario, "scenario_score", f"{scenario['name']}_scenario_score_contour.png", top.head(10), measured)
        plot_contour(scored, scenario, "probability_of_success", f"{scenario['name']}_probability_contour.png", top.head(10), measured)
        plot_contour(scored, scenario, "eta50_cP_or_mPa_s", f"{scenario['name']}_eta50_contour.png", top.head(10), measured)
    scored_all = pd.concat(all_scored, ignore_index=True)
    recs = pd.concat(all_recs, ignore_index=True)
    scored_all.to_csv(OUT_DIR / "iddsi_inverse_design_all_scenarios_grid.csv", index=False)
    recs.to_csv(OUT_DIR / "iddsi_recommended_next_experiments.csv", index=False)
    measured.to_csv(OUT_DIR / "measured_formulations_overlay.csv", index=False)

    notes = {
        "unit_note": "The viscosity values in the workbook are treated as cP/mPa.s for dysphagia-level targeting. This matches their magnitude and the NDD literature ranges at 50 s^-1. Existing column names from earlier processing may say Pa_s, but the scenario outputs use eta50_cP_or_mPa_s.",
        "literature_basis": [
            "NDD viscosity at 50 s^-1: thin 1-50 cP, nectar-like 51-350 cP, honey-like 351-1750 cP, spoon-thick >1750 cP.",
            "IDDSI uses flow testing rather than viscosity alone; IDDSI levels 3 and 4 correspond to moderately and extremely thick liquids/foods. Viscosity at 50 s^-1 remains commonly used in rheology studies for comparison.",
            "Reported IDDSI level 4 dysphagia-oriented products can fall around 0.64-3.78 Pa.s at 50 s^-1, i.e. roughly 640-3780 cP.",
        ],
        "base_results": str(BASE_RESULTS),
        "run_id": RUN_ID,
        "overlay_note": "Contour maps include measured train formulations as black circles, measured external validation formulations as blue squares, recommended candidates as open white/black circles, and the top candidate as a red star.",
    }
    (OUT_DIR / "literature_and_unit_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    code_dir = OUT_DIR / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)


if __name__ == "__main__":
    main()
