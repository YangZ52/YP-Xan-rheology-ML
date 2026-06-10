# Figure Provenance

This document maps common manuscript figures to their generating scripts.

## Main Model Performance

| Figure type | Script | Notes |
|---|---|---|
| Main-target R2 grouped bars | `scripts/make_main_target_r2_bar_figure_large.py` | Reads reviewer/main-target metrics; exports PNG/TIFF/PDF |
| Main-target RMSE grouped bars | `scripts/make_main_target_rmse_bar_figure_large.py` | Same metrics source as R2 figure |
| Formulation/LAOS model suite | `scripts/run_publication_formulation_laos_model_suite.py` | Generates model metrics, parity plots, Bland-Altman plots, and grouped bars |
| Combined LAOS R2/RMSE panel | `scripts/make_combined_laos_r2_rmse_bar_figure.py` | Large-label combined Panel A/B figure |

## Inverse Design

| Figure type | Script | Notes |
|---|---|---|
| Target-library inverse-design maps | `scripts/make_inverse_design_target_library_large_labels.py` | Large-label map with wt% axes |
| Bayesian inverse-design outputs | `scripts/run_bayesian_inverse_design_no_new_experiments.py` | Produces candidate tables, feasibility maps, uncertainty maps, target-space plots |

## Data/Workflow Schematics

| Figure type | Script |
|---|---|
| ML workflow schematic | `scripts/plot_ml_workflow_schematic.py` |
| Formulation phase diagram | `scripts/plot_formulation_phase_diagram.py` |
| Rheology data range examples | `scripts/plot_rheology_data_range_examples.py` |

Generated figures are written under `outputs/` and are ignored by git. Copy final selected figures into `figures/` only if the journal or repository deposit requires them.

