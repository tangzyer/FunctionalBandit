# LCL figure / table / artifact provenance

Each entry: artifact → script that produced it → key input(s).
Paths are relative to `simulation/numeric_code/bandit_Experiments/`.

## Figures (PDF)

| Artifact | Script | Key inputs |
| --- | --- | --- |
| `forest_5methods_q03_gp_test_95.pdf` | `plot_forest_5methods_q03.py` | `std_xnorm_q20_80_logY_ci_5methods_q03_gp_test.csv`. Main body of `fig:lcl-forest` in `ci/anisotropic_v2.tex` (subsection `subsec:sim-real-data`). 5 methods at q=0.3 default; supersedes the 6-method legacy figure below. |
| `forest_5methods_q03_legend.pdf` | `plot_forest_5methods_q03.py` (legend strip from `render_legend_only`) | Same script; emitted alongside the main forest PDF. Stacked on top of `fig:lcl-forest` via a `subcaption` subfigure to match the legend-strip pattern of Figures 1-4. |
| `forest_6methods_gp_test_95.pdf` | `plot_forest_6methods.py` | `std_xnorm_q20_80_logY_ci_6methods_gp_test.csv`. Legacy 6-method version (Trunc.Aniso q∈{0.3,0.4}, Adapt.Trunc q=0.2); kept for reproducibility but no longer cited from manuscript. |
| `std_betahat_6methods.pdf` | **no saved script — generated ad-hoc / in REPL** (Apr 2026); β̂(t) for 6 methods on Std/logY/centered data, Haar K=32 r=1.5 | `std_weekly_profile_clean_xnorm_q20_80_logY_centered.csv` |
| `lcl_6methods_bootstrap_C0.005.pdf` | `plot_bootstrap_results.py` | `lcl_6methods_results_bootstrap_C0.005.npz` |
| `tou_weekly_sample50.pdf` | `plot_weekly_sample.py` | `tou_weekly_profile_centered.csv` |

## Tables (Markdown)

| Artifact | Script | Inputs / notes |
| --- | --- | --- |
| `kernel_selection_wide_haar_table_std_weekly_profile_clean_xnorm_q20_80_logY.md` | `select_kernel_wide.py std_weekly_profile_clean_xnorm_q20_80_logY_centered.csv` | wide kernel CV sweep; winner = Haar K=32, r=1.5 |
| `lcl_6methods_table*.md` (many variants, e.g. `_bootstrap_C0.005*`, `_log_centered_imputed_5fold_r1.5*`, `_fourier_weekday_weekend_log*`) | `run_lcl_6methods.py` and `run_lcl_6methods_bootstrap.py` | CLI args encode suffix: dataset stem, n_folds, r, basis, K |

## NPZ result caches

| Artifact | Script | Notes |
| --- | --- | --- |
| `lcl_6methods_results*.npz` (same variants as above) | `run_lcl_6methods.py` / `run_lcl_6methods_bootstrap.py` | same CLI as the matching `_table*.md` |
| `lcl_foldwise_*`, `lcl_10fold_*`, `lcl_10cluster_*` | `run_lcl_grid_foldwise.py`, `run_lcl_grid_foldwise_ypos.py`, `run_lcl_10fold_ypos.py`, `run_lcl_10cluster_xonly.py`, `run_lcl_10fold_subcluster.py` | filename suffix matches script name |

## Derived CSVs

| Artifact | Script | Notes |
| --- | --- | --- |
| `std_weekly_profile.csv` | `build_std_dataset.py` | raw feature/target for Std households |
| `std_weekly_profile_centered.csv`, `_relY*.csv`, `_logY*.csv` | `center_std_dataset.py`, `build_std_relY_dataset.py`, `build_std_logY_dataset.py` | downstream transforms of the raw Std file |
| `*_clean_xnorm_q20_80*.csv` | `build_clean_raw_xnorm.py` | x-norm trimming at q20–q80 quantiles |
| `std_xnorm_q20_80_logY_gp_test_functions_n100.csv` | `generate_gp_test_functions.py` | 100 GP draws, N(0, Σ_train); seed=0. NOT held-out real households — see memory `project_lcl_std_gp_test_file.md` |
| `std_xnorm_q20_80_logY_ci_6methods_gp_test.csv` | `run_std_6methods_on_gp_test.py` | long-format CI output, 100 tests × 6 methods × 3 conf levels = 1800 rows. Legacy (Adapt.Trunc q=0.2). |
| `std_xnorm_q20_80_logY_ci_5methods_q03_gp_test.csv` | `run_std_5methods_q03_on_gp_test.py` | long-format CI output, 100 tests × 5 methods × 3 conf levels = 1500 rows. Trunc.Aniso and Adapt.Trunc share J=n^0.3; consumed by the manuscript figure. |
| `tou_weekly_profile*.csv` | `build_tou_dataset.py` and friends | ToU (treatment) analogues of the Std files |

## Manuscript cross-reference

| Manuscript label | LaTeX file | Subsection | PDF here |
| --- | --- | --- | --- |
| `fig:lcl-forest` | `ci/anisotropic_v2.tex` | `subsec:sim-real-data` | `forest_6methods_gp_test_95.pdf` |

## Conventions worth remembering

- "Std" = LCL control-group households (tariff stayed standard); "ToU" = treatment.
- The chosen basis for the Std/logY real-data analysis is Haar K=32, r=1.5 (winner of `select_kernel_wide.py`).
- Lepski constant for Std/logY runs: C = 0.005. (Earlier ToU runs used C = 0.05 — don't mix.)
- Dataset stems encode transforms: `_centered`, `_imputed`, `_logY`, `_clean_xnorm_q20_80`, `_relY`.
