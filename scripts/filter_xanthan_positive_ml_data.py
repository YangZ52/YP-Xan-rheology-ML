import json
from pathlib import Path

import pandas as pd


SRC = Path("/Users/zhiy/Documents/Rheology ML/outputs/ml_ready_20260529")
OUT = Path("/Users/zhiy/Documents/Rheology ML/outputs/ml_ready_xanthan_positive_20260529")
OUT.mkdir(parents=True, exist_ok=True)

TABLES = [
    "README",
    "source_inventory",
    "qc_summary",
    "formulation_master",
    "replicate_master",
    "viscosity_long",
    "frequency_long",
    "strain_long",
    "strain_summary_replicate",
    "strain_summary_formulation",
]


def filter_table(name, df):
    if name == "README":
        extra = pd.DataFrame([
            {
                "item": "xanthan_positive_filter",
                "description": "This workbook keeps only rows/formulations with xanthan_pct > 0 for ML because pure YP / 0% XG is outside the modeling scope.",
            }
        ])
        return pd.concat([df, extra], ignore_index=True)
    if "xanthan_pct" in df.columns:
        return df[pd.to_numeric(df["xanthan_pct"], errors="coerce") > 0].reset_index(drop=True)
    return df


def main():
    manifest = {}
    for name in TABLES:
        df = pd.read_csv(SRC / f"{name}.csv")
        filtered = filter_table(name, df)
        if name == "qc_summary":
            filtered = pd.DataFrame([
                {"check": "filter_rule", "value": "xanthan_pct > 0", "notes": "Pure YP / 0% XG excluded for ML"},
                {"check": "formulation_master_rows", "value": len(pd.read_csv(SRC / "formulation_master.csv").query("xanthan_pct > 0")), "notes": "xanthan-positive formulations"},
                {"check": "replicate_master_rows", "value": len(pd.read_csv(SRC / "replicate_master.csv").query("xanthan_pct > 0")), "notes": "xanthan-positive replicate/formulation rows"},
                {"check": "viscosity_long_rows", "value": len(pd.read_csv(SRC / "viscosity_long.csv").query("xanthan_pct > 0")), "notes": "xanthan-positive viscosity rows"},
                {"check": "frequency_long_rows", "value": len(pd.read_csv(SRC / "frequency_long.csv").query("xanthan_pct > 0")), "notes": "xanthan-positive frequency rows"},
                {"check": "strain_long_rows", "value": len(pd.read_csv(SRC / "strain_long.csv").query("xanthan_pct > 0")), "notes": "xanthan-positive strain rows"},
            ])
        filtered.to_csv(OUT / f"{name}.csv", index=False)
        filtered.where(pd.notna(filtered), None).to_json(OUT / f"{name}.json", orient="records", indent=2)
        manifest[name] = {"rows": len(filtered), "columns": list(filtered.columns)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
