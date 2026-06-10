# Reproducibility Workflow

This document summarizes the analysis order for the YP-Xan rheology ML manuscript.

## 1. Build ML-Ready Rheology Tables

```bash
python scripts/prepare_rheology_ml_data.py
```

This script parses the original viscosity, frequency-sweep, and strain-sweep workbooks into standardized tables:

- `viscosity_long.csv`
- `frequency_long.csv`
- `strain_long.csv`
- `replicate_master.csv`
- `formulation_master.csv`
- `strain_summary_replicate.csv`
- `strain_summary_formulation.csv`

The raw input file paths are local absolute paths and should be updated before sharing the repository publicly.

## 2. Restrict To Xanthan-Positive Formulations

```bash
python scripts/filter_xanthan_positive_ml_data.py
```

Pure 0 wt% xanthan formulations are excluded from the ML design scope. The resulting tables are written to the xanthan-positive ML-ready output folder.

## 3. Main ML Benchmark

```bash
python scripts/run_xanthan_positive_ml_benchmark.py
```

This benchmark evaluates:

- scalar viscosity targets such as `log10_eta_50`
- SAOS targets such as `log10_Gp_1Hz`
- full viscosity-curve prediction using formulation plus `log10(shear rate)`
- strain-sweep targets including breaking strain, breaking stress, and LVR

Models include ridge regression, SVR-RBF, kernel ridge regression, GPR-Matern-ARD, random forest, extra trees, gradient boosting, and optional XGBoost.

## 4. Publication Model Suite

```bash
python scripts/run_publication_formulation_laos_model_suite.py
```

This script produces the formulation and LAOS model comparisons used in the manuscript. It uses balanced formulation-grid internal validation and external validation.

## 5. Inverse Design

```bash
python scripts/run_bayesian_inverse_design_no_new_experiments.py
```

This script fits GPR surrogates, generates formulation-space maps, computes posterior feasibility/scenario scores, and ranks candidate formulations for validation.

## 6. Polished Publication Figures

Large-label figure scripts were added for final manuscript assembly:

```bash
python scripts/make_main_target_r2_bar_figure_large.py
python scripts/make_main_target_rmse_bar_figure_large.py
python scripts/make_combined_laos_r2_rmse_bar_figure.py
python scripts/make_inverse_design_target_library_large_labels.py
```

These scripts read existing output tables and regenerate high-resolution PNG/TIFF/PDF figures without rerunning the full model training.

