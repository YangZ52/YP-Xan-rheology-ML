# YP-Xan Rheology Machine Learning

Code accompanying a manuscript on machine-learning-guided rheology prediction and inverse design of yeast protein particle-xanthan gum (YP-Xan) structured fluids.

The workflow converts rheology measurements into ML-ready tables, benchmarks regression models for viscosity, small-amplitude oscillatory shear (SAOS), and strain-sweep targets, and performs Gaussian-process-based inverse design to propose new YP-Xan formulations for experimental validation.

## What This Repository Contains

```text
scripts/      Data preparation, model benchmarking, inverse design, and figure generation
docs/         Reproducibility notes, data dictionary, and manuscript figure guide
data/         Placeholder for source data; raw workbooks are not committed
figures/      Selected paper-ready figures plus a placeholder for generated exports
results/      Compact review-facing result tables
outputs/      Generated analysis outputs, ignored by git
```

## Peer-Review Package

For peer review, this repository includes enough material to evaluate the computational workflow without requiring the full raw workbook archive:

- analysis and figure-generation code in `scripts/`
- methodological notes in `docs/`
- selected result tables in `results/peer_review/`
- selected manuscript figures in `figures/peer_review/`

The raw rheology files and full intermediate output folders are excluded from git. They can be deposited separately in a journal-approved repository before final publication.

## Main Analyses

- **Data preparation:** parse raw viscosity, frequency-sweep, and strain-sweep files into standardized long-format and replicate/formulation-level tables.
- **Modeling:** compare ridge regression, SVR-RBF, kernel ridge regression, Gaussian process regression with Matern ARD kernel, random forest, extra trees, gradient boosting, and XGBoost.
- **Validation:** use formulation-level internal validation and independent external validation to prevent replicate-level or shear-rate-level leakage.
- **Inverse design:** use GPR surrogates and posterior scoring to identify candidate YP-Xan formulations targeting viscosity and elasticity windows.
- **Figures:** generate publication-ready parity plots, bar charts, phase diagrams, length-scale summaries, and inverse-design maps.

## Installation

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate yp-xan-rheology-ml
```

`xgboost` is optional in the scripts. If unavailable, the benchmark scripts continue with the remaining models.

## Reproducible Workflow

The typical workflow is:

```bash
python scripts/prepare_rheology_ml_data.py
python scripts/filter_xanthan_positive_ml_data.py
python scripts/run_xanthan_positive_ml_benchmark.py
python scripts/run_publication_formulation_laos_model_suite.py
python scripts/run_bayesian_inverse_design_no_new_experiments.py
```

Publication-specific figure polishing scripts include:

```bash
python scripts/make_main_target_r2_bar_figure_large.py
python scripts/make_main_target_rmse_bar_figure_large.py
python scripts/make_combined_laos_r2_rmse_bar_figure.py
python scripts/make_inverse_design_target_library_large_labels.py
```

See [docs/workflow.md](docs/workflow.md) for a fuller execution order and [docs/figures.md](docs/figures.md) for paper figure provenance.

## Data Availability

The raw experimental workbooks are not committed because the full data package is expected to be deposited separately. To reproduce the workflow from raw data, either:

1. deposit raw or ML-ready data in a journal-approved repository, or
2. provide anonymized ML-ready CSV/XLSX tables and update the input paths in the scripts.

See [docs/data_dictionary.md](docs/data_dictionary.md) for the key tables and variables.

The compact tables in [results/peer_review](results/peer_review) are included only as review-facing outputs. They are not a substitute for the final raw-data deposition.

## Validation Design

Internal validation is formulation-based. Entire formulations are held out as unseen validation cases, so all replicates and all shear-rate/frequency observations for a formulation remain together. This prevents leakage from replicate measurements or curve points being split across training and validation folds. External validation uses independent holdout formulations.

## Inverse-Design Validation Candidates

The inverse-design analysis selected the following formulations for experimental validation:

| Target property | Predicted value | Yeast protein (wt%) | Xanthan gum (wt%) | Experimental value | Relative error (%) |
|---|---:|---:|---:|---:|---:|
| eta50 | 200 mPa s | 15.0 | 0.335 | 150.1 +/- 1.1 | -24.9 |
| eta50 | 200 mPa s | 10.25 | 0.465 | 216.7 +/- 8.8 | 8.4 |
| eta50 | 500 mPa s | 24.5 | 0.260 | 404.5 +/- 17.2 | -19.1 |
| G' at 1 Hz | 100 Pa | 24.75 | 0.295 | 70.6 +/- 4.7 | -29.4 |
| G' at 1 Hz | 500 Pa | 26.5 | 0.385 | 421.9 +/- 69.7 | -15.6 |
