import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(os.environ.get("RHEOLOGY_DATA_ROOT", ROOT / "data" / "raw"))
OUT_DIR = ROOT / "outputs" / "ml_ready_20260529"

FILES = {
    "viscosity_train": "Viscosity_modeling_ready_parsed_20260411.xlsx",
    "viscosity_predict": "Viscosity_predicting data_ready_parsed_20260416.xlsx",
    "frequency_train": "Frequency sweep_training data_ready_parsed_20260411.xlsx",
    "frequency_predict": "Frequency sweep_predicting data_ready_parsed_20260416.xlsx",
    "strain": "Strain sweep_YP+XG_all data_0522.xlsx",
}


def compact_num(x):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return (f"{x:.4f}").rstrip("0").rstrip(".")


def formulation_label(yp, xg):
    return f"{compact_num(yp)}%YP+{compact_num(xg)}%XG"


def sample_id(yp, xg, replicate):
    return f"YP{compact_num(yp)}_XG{compact_num(xg)}_R{int(replicate)}"


def parse_formulation(text):
    if pd.isna(text):
        return (np.nan, np.nan)
    s = str(text).strip().replace(" ", "")
    s = s.replace("_RPT", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)%YP\+([0-9]+(?:\.[0-9]+)?)%XG", s, flags=re.I)
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))


def normalize_common(df, split, source_file, measurement):
    df = df.copy()
    df["split"] = split
    df["measurement"] = measurement
    df["source_file"] = source_file
    df["yp_pct"] = pd.to_numeric(df["yp_pct"], errors="coerce")
    df["xanthan_pct"] = pd.to_numeric(df["xanthan_pct"], errors="coerce")
    df["replicate"] = pd.to_numeric(df["replicate"], errors="coerce").astype("Int64")
    df["formulation_std"] = [formulation_label(y, x) for y, x in zip(df["yp_pct"], df["xanthan_pct"])]
    df["sample_id_std"] = [sample_id(y, x, r) for y, x, r in zip(df["yp_pct"], df["xanthan_pct"], df["replicate"])]
    return df


def log_interp(x, y, target):
    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[(clean["x"] > 0) & (clean["y"] > 0)].sort_values("x")
    if clean.empty or target < clean["x"].min() or target > clean["x"].max():
        return np.nan
    lx = np.log10(clean["x"].to_numpy(dtype=float))
    ly = np.log10(clean["y"].to_numpy(dtype=float))
    return float(10 ** np.interp(math.log10(target), lx, ly))


def log_slope(x, y, lo, hi):
    y_lo = log_interp(x, y, lo)
    y_hi = log_interp(x, y, hi)
    if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_lo <= 0 or y_hi <= 0:
        return np.nan
    return float((math.log10(y_hi) - math.log10(y_lo)) / (math.log10(hi) - math.log10(lo)))


def viscosity_scalars(visc):
    rows = []
    for keys, g in visc.groupby(["split", "sample_id_std", "formulation_std", "yp_pct", "xanthan_pct", "replicate"]):
        split, sid, form, yp, xg, rep = keys
        row = {
            "split": split,
            "sample_id_std": sid,
            "formulation_std": form,
            "yp_pct": yp,
            "xanthan_pct": xg,
            "replicate": int(rep),
            "n_points_viscosity": int(g["viscosity"].notna().sum()),
        }
        for target in [1, 10, 50, 100]:
            eta = log_interp(g["shear_rate"], g["viscosity"], target)
            tau = log_interp(g["shear_rate"], g["shear_stress"], target)
            row[f"eta_{target:g}_Pa_s"] = eta
            row[f"log10_eta_{target:g}"] = math.log10(eta) if np.isfinite(eta) and eta > 0 else np.nan
            row[f"stress_at_{target:g}s_inv_Pa"] = tau
        row["viscosity_shear_thinning_slope_1to100"] = log_slope(g["shear_rate"], g["viscosity"], 1, 100)
        rows.append(row)
    return pd.DataFrame(rows)


def frequency_scalars(freq):
    rows = []
    for keys, g in freq.groupby(["split", "sample_id_std", "formulation_std", "yp_pct", "xanthan_pct", "replicate"]):
        split, sid, form, yp, xg, rep = keys
        row = {
            "split": split,
            "sample_id_std": sid,
            "formulation_std": form,
            "yp_pct": yp,
            "xanthan_pct": xg,
            "replicate": int(rep),
            "n_points_frequency": int(g[["Gp_Pa", "Gpp_Pa"]].dropna(how="all").shape[0]),
        }
        for target in [0.1, 1, 6.31]:
            gp = log_interp(g["frequency_Hz"], g["Gp_Pa"], target)
            gpp = log_interp(g["frequency_Hz"], g["Gpp_Pa"], target)
            row[f"Gp_{target:g}Hz_Pa"] = gp
            row[f"Gpp_{target:g}Hz_Pa"] = gpp
            row[f"tan_delta_{target:g}Hz"] = gpp / gp if np.isfinite(gp) and gp > 0 and np.isfinite(gpp) else np.nan
            row[f"log10_Gp_{target:g}Hz"] = math.log10(gp) if np.isfinite(gp) and gp > 0 else np.nan
        row["Gp_frequency_slope_0p1to6p31"] = log_slope(g["frequency_Hz"], g["Gp_Pa"], 0.1, 6.31)
        row["Gpp_frequency_slope_0p1to6p31"] = log_slope(g["frequency_Hz"], g["Gpp_Pa"], 0.1, 6.31)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_strain_raw(path):
    raw = pd.read_excel(path, sheet_name="Strain sweep Raw data", header=None)
    records = []
    header_rows = raw.index[raw.apply(lambda row: row.astype(str).str.contains("Strain sweep", case=False, regex=False).any(), axis=1)]
    for header_row in header_rows:
        sample_row = header_row - 1
        if sample_row < 0:
            continue
        for col in range(2, raw.shape[1], 4):
            name = raw.iat[sample_row, col] if col < raw.shape[1] else np.nan
            yp, xg = parse_formulation(name)
            if not np.isfinite(yp) or not np.isfinite(xg):
                continue
            row_idx = header_row + 1
            while row_idx < raw.shape[0] and pd.notna(raw.iat[row_idx, 1]):
                strain = pd.to_numeric(raw.iat[row_idx, 1], errors="coerce")
                if pd.isna(strain):
                    break
                vals = {
                    1: (raw.iat[row_idx, col], raw.iat[row_idx, col + 1] if col + 1 < raw.shape[1] else np.nan),
                    2: (raw.iat[row_idx, col + 2] if col + 2 < raw.shape[1] else np.nan,
                        raw.iat[row_idx, col + 3] if col + 3 < raw.shape[1] else np.nan),
                }
                for rep, (gp, gpp) in vals.items():
                    records.append({
                        "split": "strain_only",
                        "measurement": "strain_sweep",
                        "sample_id_std": sample_id(yp, xg, rep),
                        "formulation_std": formulation_label(yp, xg),
                        "yp_pct": yp,
                        "xanthan_pct": xg,
                        "replicate": rep,
                        "strain_pct": float(strain),
                        "Gp_Pa": pd.to_numeric(gp, errors="coerce"),
                        "Gpp_Pa": pd.to_numeric(gpp, errors="coerce"),
                        "source_file": path.name,
                    })
                row_idx += 1
    out = pd.DataFrame(records)
    out["tan_delta"] = out["Gpp_Pa"] / out["Gp_Pa"]
    return out


def parse_strain_summary(path):
    df = pd.read_excel(path, sheet_name="Strain sweep Summary", header=None)
    metrics = {"Break strain (%)": "break_strain_pct", "Break stress (Pa)": "break_stress_Pa", "LVR (%)": "LVR_pct"}
    current_metric = None
    rep_records = []
    summary_records = []
    for r in range(df.shape[0] - 3):
        row = df.iloc[r]
        text_cells = [str(x).strip() for x in row.dropna().tolist() if isinstance(x, str)]
        for label, metric in metrics.items():
            if any(label.lower() == t.lower() for t in text_cells):
                current_metric = metric
        formulation_cols = []
        for c, val in row.items():
            yp, xg = parse_formulation(val)
            if np.isfinite(yp) and np.isfinite(xg):
                formulation_cols.append((c, val, yp, xg, "_RPT" in str(val)))
        if current_metric is None or not formulation_cols:
            continue
        next_numeric = pd.to_numeric(df.iloc[r + 1, [c for c, *_ in formulation_cols]], errors="coerce").notna().sum()
        if next_numeric == 0:
            continue
        for c, val, yp, xg, is_rpt in formulation_cols:
            replicate = 2 if is_rpt else 1
            rep_val = pd.to_numeric(df.iat[r + 1, c], errors="coerce")
            if pd.notna(rep_val):
                rep_records.append({
                    "sample_id_std": sample_id(yp, xg, replicate),
                    "formulation_std": formulation_label(yp, xg),
                    "yp_pct": yp,
                    "xanthan_pct": xg,
                    "replicate": replicate,
                    "parameter": current_metric,
                    "value": float(rep_val),
                    "source_file": path.name,
                })
            if not is_rpt:
                mean_val = pd.to_numeric(df.iat[r + 2, c], errors="coerce")
                sd_val = pd.to_numeric(df.iat[r + 3, c], errors="coerce")
                if pd.notna(mean_val):
                    summary_records.append({
                        "formulation_std": formulation_label(yp, xg),
                        "yp_pct": yp,
                        "xanthan_pct": xg,
                        "parameter": current_metric,
                        "mean": float(mean_val),
                        "sd": float(sd_val) if pd.notna(sd_val) else np.nan,
                        "source_file": path.name,
                    })
    rep = pd.DataFrame(rep_records).drop_duplicates()
    wide_rep = rep.pivot_table(
        index=["sample_id_std", "formulation_std", "yp_pct", "xanthan_pct", "replicate", "source_file"],
        columns="parameter",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide_rep.columns.name = None
    summary = pd.DataFrame(summary_records).drop_duplicates()
    wide_summary = summary.pivot_table(
        index=["formulation_std", "yp_pct", "xanthan_pct", "source_file"],
        columns="parameter",
        values=["mean", "sd"],
        aggfunc="first",
    )
    wide_summary.columns = [f"{metric}_{stat}" for stat, metric in wide_summary.columns]
    return wide_rep.reset_index(drop=True), wide_summary.reset_index()


def clean_long_tables():
    visc_train = normalize_common(
        pd.read_excel(SOURCE_DIR / FILES["viscosity_train"]),
        "train",
        FILES["viscosity_train"],
        "viscosity",
    )
    visc_predict = normalize_common(
        pd.read_excel(SOURCE_DIR / FILES["viscosity_predict"]),
        "predict",
        FILES["viscosity_predict"],
        "viscosity",
    )
    visc = pd.concat([visc_train, visc_predict], ignore_index=True)
    visc = visc[[
        "split", "measurement", "sample_id_std", "sample_id", "formulation_std", "formulation",
        "yp_pct", "xanthan_pct", "replicate", "shear_rate", "viscosity", "shear_stress",
        "log_shear_rate", "log_viscosity", "log_shear_stress", "source_file",
    ]]

    freq_train = normalize_common(
        pd.read_excel(SOURCE_DIR / FILES["frequency_train"]),
        "train",
        FILES["frequency_train"],
        "frequency_sweep",
    )
    freq_predict = normalize_common(
        pd.read_excel(SOURCE_DIR / FILES["frequency_predict"]),
        "predict",
        FILES["frequency_predict"],
        "frequency_sweep",
    )
    freq = pd.concat([freq_train, freq_predict], ignore_index=True)
    freq["tan_delta"] = freq["Gpp_Pa"] / freq["Gp_Pa"]
    freq["log10_frequency_Hz"] = np.log10(freq["frequency_Hz"])
    freq["log10_Gp_Pa"] = np.where(freq["Gp_Pa"] > 0, np.log10(freq["Gp_Pa"]), np.nan)
    freq["log10_Gpp_Pa"] = np.where(freq["Gpp_Pa"] > 0, np.log10(freq["Gpp_Pa"]), np.nan)
    freq = freq[[
        "split", "measurement", "sample_id_std", "sample_id", "formulation_std", "formulation",
        "yp_pct", "xanthan_pct", "replicate", "frequency_Hz", "Gp_Pa", "Gpp_Pa",
        "tan_delta", "log10_frequency_Hz", "log10_Gp_Pa", "log10_Gpp_Pa",
        "fit_cutoff_Hz_optional", "source_file",
    ]]

    strain_raw = parse_strain_raw(SOURCE_DIR / FILES["strain"])
    strain_rep, strain_form = parse_strain_summary(SOURCE_DIR / FILES["strain"])
    return visc, freq, strain_raw, strain_rep, strain_form


def build_outputs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    visc, freq, strain_raw, strain_rep, strain_form = clean_long_tables()

    visc_scalar = viscosity_scalars(visc)
    freq_scalar = frequency_scalars(freq)
    replicate_master = pd.merge(
        visc_scalar,
        freq_scalar,
        on=["split", "sample_id_std", "formulation_std", "yp_pct", "xanthan_pct", "replicate"],
        how="outer",
    )
    replicate_master = pd.merge(
        replicate_master,
        strain_rep.drop(columns=["source_file"], errors="ignore"),
        on=["sample_id_std", "formulation_std", "yp_pct", "xanthan_pct", "replicate"],
        how="outer",
    )

    formulation_flags = []
    all_forms = sorted(set(visc["formulation_std"]) | set(freq["formulation_std"]) | set(strain_raw["formulation_std"]))
    for form in all_forms:
        yp, xg = parse_formulation(form)
        formulation_flags.append({
            "formulation_std": form,
            "yp_pct": yp,
            "xanthan_pct": xg,
            "in_viscosity_train": bool(((visc["formulation_std"] == form) & (visc["split"] == "train")).any()),
            "in_viscosity_predict": bool(((visc["formulation_std"] == form) & (visc["split"] == "predict")).any()),
            "in_frequency_train": bool(((freq["formulation_std"] == form) & (freq["split"] == "train")).any()),
            "in_frequency_predict": bool(((freq["formulation_std"] == form) & (freq["split"] == "predict")).any()),
            "in_strain_sweep": bool((strain_raw["formulation_std"] == form).any()),
        })
    formulation_master = pd.DataFrame(formulation_flags).sort_values(["yp_pct", "xanthan_pct"])
    formulation_master = pd.merge(formulation_master, strain_form.drop(columns=["source_file"], errors="ignore"), on=["formulation_std", "yp_pct", "xanthan_pct"], how="left")

    source_inventory = []
    for name, filename in FILES.items():
        path = SOURCE_DIR / filename
        xl = pd.ExcelFile(path)
        source_inventory.append({
            "source_key": name,
            "source_file": filename,
            "sheet_names": "; ".join(xl.sheet_names),
            "original_path": str(path),
        })
    source_inventory = pd.DataFrame(source_inventory)

    qc = pd.DataFrame([
        {"check": "viscosity_long_rows", "value": len(visc), "notes": "train + prediction rows"},
        {"check": "frequency_long_rows", "value": len(freq), "notes": "train + prediction rows"},
        {"check": "strain_long_rows", "value": len(strain_raw), "notes": "parsed from grouped raw strain sheet"},
        {"check": "viscosity_unique_formulations", "value": visc["formulation_std"].nunique(), "notes": ""},
        {"check": "frequency_unique_formulations", "value": freq["formulation_std"].nunique(), "notes": ""},
        {"check": "strain_unique_formulations", "value": strain_raw["formulation_std"].nunique(), "notes": ""},
        {"check": "replicate_master_rows", "value": len(replicate_master), "notes": "one row per replicate/formulation where available"},
        {"check": "formulation_master_rows", "value": len(formulation_master), "notes": "one row per unique formulation"},
        {"check": "frequency_missing_Gp_rows", "value": int(freq["Gp_Pa"].isna().sum()), "notes": "training file includes 0% XG blank frequency rows"},
        {"check": "frequency_missing_Gpp_rows", "value": int(freq["Gpp_Pa"].isna().sum()), "notes": "training file includes 0% XG blank frequency rows"},
    ])

    tables = {
        "README": pd.DataFrame([
            {"item": "purpose", "description": "ML-ready rheology tables for yeast protein particle + xanthan gum dysphagia formulations."},
            {"item": "safety", "description": "Original source data files were read only. Cleaned outputs are written in the local Rheology ML workspace."},
            {"item": "key", "description": "Use formulation_std, yp_pct, xanthan_pct, replicate, and split to join tables."},
            {"item": "split", "description": "train = original modeling/training files; predict = external prediction/holdout files; strain_only = strain workbook."},
            {"item": "recommended_GPR_targets", "description": "Start with log10_eta_50, log10_Gp_1Hz, tan_delta_1Hz, break_strain_pct, break_stress_Pa, and LVR_pct."},
        ]),
        "source_inventory": source_inventory,
        "qc_summary": qc,
        "formulation_master": formulation_master,
        "replicate_master": replicate_master,
        "viscosity_long": visc,
        "frequency_long": freq,
        "strain_long": strain_raw,
        "strain_summary_replicate": strain_rep,
        "strain_summary_formulation": strain_form,
    }

    for sheet, table in tables.items():
        table.to_csv(OUT_DIR / f"{sheet}.csv", index=False)
        table.replace({np.nan: None}).to_json(OUT_DIR / f"{sheet}.json", orient="records", indent=2)

    manifest = {sheet: {"rows": len(table), "columns": list(table.columns)} for sheet, table in tables.items()}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
