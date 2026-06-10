# Data Dictionary

The repository uses standardized ML-ready tables created from raw rheology workbooks. Raw files are not committed.

## Core Identifiers

| Column | Description |
|---|---|
| `formulation_std` | Standardized formulation label, e.g. `15%YP+0.5%XG` |
| `sample_id_std` | Standardized sample/replicate identifier |
| `yp_pct` | Yeast protein concentration |
| `xanthan_pct` | Xanthan gum concentration |
| `replicate` | Replicate number |
| `split` | Dataset split: `train`, `predict`, or `strain_only` depending on table |

## Viscosity Variables

| Column | Description |
|---|---|
| `shear_rate` | Shear rate for flow-curve measurements |
| `log_shear_rate` | `log10(shear_rate)` |
| `viscosity` | Apparent viscosity in the units exported from the rheometer source file |
| `log_viscosity` | `log10(viscosity)` |
| `eta_1_Pa_s`, `eta_10_Pa_s`, `eta_50_Pa_s`, `eta_100_Pa_s` | Log-log interpolated viscosity descriptors |
| `log10_eta_1`, `log10_eta_50`, `log10_eta_100` | Log-transformed viscosity descriptors |
| `viscosity_shear_thinning_slope_1to100` | Log-log viscosity slope between 1 and 100 s^-1 |

## Frequency-Sweep Variables

| Column | Description |
|---|---|
| `frequency_Hz` | Frequency for SAOS measurement |
| `Gp_Pa` | Storage modulus |
| `Gpp_Pa` | Loss modulus |
| `tan_delta` | `Gpp_Pa / Gp_Pa` |
| `Gp_0.1Hz_Pa`, `Gp_1Hz_Pa`, `Gp_6.31Hz_Pa` | Log-log interpolated storage modulus descriptors |
| `Gpp_0.1Hz_Pa`, `Gpp_1Hz_Pa`, `Gpp_6.31Hz_Pa` | Log-log interpolated loss modulus descriptors |
| `log10_Gp_1Hz` | Primary elasticity descriptor used in several models |
| `Gp_frequency_slope_0p1to6p31` | Log-log G' slope between 0.1 and 6.31 Hz |
| `Gpp_frequency_slope_0p1to6p31` | Log-log G'' slope between 0.1 and 6.31 Hz |

## Strain-Sweep Variables

| Column | Description |
|---|---|
| `strain_pct` | Strain amplitude |
| `LVR_pct` | Linear viscoelastic region limit |
| `break_strain_pct` | Breaking strain |
| `break_stress_Pa` | Breaking stress |
| `log10_break_stress_Pa` | Log-transformed breaking stress |

## Modeling Notes

- Dynamic-range rheology targets are modeled on the log10 scale where indicated.
- Input features for ridge, SVR, kernel ridge, and GPR models are standardized inside pipelines.
- GPR ARD length scales are interpreted after feature standardization.
- Missing or non-finite values are removed independently for each target and feature set.

