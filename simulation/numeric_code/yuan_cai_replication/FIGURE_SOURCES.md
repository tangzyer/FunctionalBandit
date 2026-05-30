# Simulation figure provenance (yuan_cai_replication)

Each entry: output PDF (under `results/`) → plot script (under `plots/`) →
upstream simulation script (under `simulations/`, which writes the NPZ the
plot consumes). Auto-detected from `savefig(...)` calls and NPZ load paths.

Pipeline: `simulations/run_*.py` → `results/*.npz` → `plots/plot_*.py` → `results/*.pdf`.

## Figures published on 2026-04-19 (the 6 "3col_6m" panels)

These all consume the same family of NPZ caches produced by
`simulations/run_alpha_sweep_v13*.py` (filename template
`alpha_sweep_{case}_n{n}_Ca{Ca}_Ci{Ci}_v13.npz`). The `_3col` plot scripts
arrange three panels (cases or n values) in one row.

| PDF (`results/`) | Plot script (`plots/`) | Upstream NPZ producer (`simulations/`) |
| --- | --- | --- |
| `inference_normal_3col_6m.pdf` | `plot_r2_3_normal_3col.py` | `run_alpha_sweep_v13.py` (normal case, r=2) |
| `inference_sparse2_3col_6m.pdf` | `plot_r2_3_sparse2_3col.py` | `run_alpha_sweep_v13.py` (sparse2 case) |
| `inference_beta4_3col_6m.pdf` | `plot_r2_3_beta4_3col.py` | `run_alpha_sweep_v13.py` (beta4 case) |
| `inference_normal_r2_1p5_3col_6m.pdf` | `plot_r2_1p5_normal_3col.py` | `run_r2_1p5_3col.py` (normal, r=1.5) |
| `inference_sparse2_r2_1p5_3col_6m.pdf` | `plot_r2_1p5_sparse2_3col.py` | `run_r2_1p5_3col.py` (sparse2, r=1.5) |
| `inference_beta4_r2_1p5_3col_6m.pdf` | `plot_r2_1p5_beta4_3col.py` | `run_r2_1p5_3col.py` (beta4, r=1.5) |

Naming convention: `plot_<r-setting>_<case>_3col.py` → `inference_<case>[_<r-setting>]_3col_6m.pdf`.
When `r2_3` is the default, it is omitted from the PDF name.

## Figures published on 2026-05-28 (the 3 `r2_1` panels, BM test functions — superseded)

Initial $r_2=1$ triple using Brownian-motion test inputs (the default
`test_func_kind='bm'`). Superseded by the `_inputtest` triple below
(2026-05-29), which is what `ci/anisotropic_v2.tex` actually includes; the
BM-test PDFs are kept on disk for reproducibility but no longer referenced.

| PDF (`results/`) | Plot script (`plots/`) | Upstream NPZ producer (`simulations/`) |
| --- | --- | --- |
| `inference_normal_r2_1_3col_6m.pdf` | `plot_r2_1_normal_3col.py` | `run_r2_1_3col_parallel.py` → `run_alpha_sweep_v13.py` (normal, r=1, BM) |
| `inference_beta4_r2_1_3col_6m.pdf` | `plot_r2_1_beta4_3col.py` | `run_r2_1_3col_parallel.py` → `run_alpha_sweep_v13.py` (beta4, r=1, BM) |
| `inference_sparse2_r2_1_3col_6m.pdf` | `plot_r2_1_sparse2_3col.py` | `run_r2_1_3col_parallel.py` → `run_alpha_sweep_v13.py` (sparse2, r=1, BM) |

## Figures published on 2026-05-29 (the 3 `r2_1` panels, INPUT-distribution test functions — superseded by grid figures below)

Per-slope triple using input-distribution test inputs (`test_func_kind='input'`).
Three slope-specific figures, one per (D1 regime). Superseded the same day by
the `_grid_*_inputtest` pair that LaTeX now includes; the per-slope PDFs are
kept for cross-reference but no longer referenced from `anisotropic_v2.tex`.

| PDF (`results/`) | Plot script (`plots/`) | Upstream NPZ producer (`simulations/`) |
| --- | --- | --- |
| `inference_normal_r2_1_3col_6m_inputtest.pdf` | `plot_r2_1_normal_3col_inputtest.py` | `run_r2_1_3col_inputtest_parallel.py` (normal, r=1, input) |
| `inference_beta4_r2_1_3col_6m_inputtest.pdf` | `plot_r2_1_beta4_3col_inputtest.py` | `run_r2_1_3col_inputtest_parallel.py` (beta4, r=1, input) |
| `inference_sparse2_r2_1_3col_6m_inputtest.pdf` | `plot_r2_1_sparse2_3col_inputtest.py` | `run_r2_1_3col_inputtest_parallel.py` (sparse2, r=1, input) |

## Figures published on 2026-05-29 (the 2 `r2_1` 3×3 grid figures, INPUT-distribution test functions)

Original $r_2=1$ grid figures (rows = slope, cols = design). Generalised on
2026-05-30 into the multi-$r_2$ family below; the script (now in
`plot_r2_1_grid_inputtest.py`) still emits these two PDFs and writes the
shared legend at the neutral name `inference_grid_legend_inputtest.pdf`.

| PDF (`results/`) | Plot script (`plots/`) | Upstream NPZ producer (`simulations/`) |
| --- | --- | --- |
| `inference_r2_1_grid_coverage_inputtest.pdf` | `plot_r2_1_grid_inputtest.py` | `run_r2_1_3col_inputtest_parallel.py` (9 cases, r=1, input) + `run_adaptq04_sparse2_r2_1_inputtest_parallel.py` |
| `inference_r2_1_grid_width_inputtest.pdf` | `plot_r2_1_grid_inputtest.py` | same |
| `inference_grid_legend_inputtest.pdf` | `plot_r2_1_grid_inputtest.py` (legend strip) | shared across all r2-figures |

## Figures published on 2026-05-30 (multi-$r_2$ grid family, INPUT-distribution test functions)

Two additional grid figures emitted by the same generalised script, mirroring
the $r_2=1$ layout at the rougher ($r_2=0.5$) and smoother ($r_2=2$) covariance
regimes. Currently referenced from `ci/anisotropic_v2.tex` as
`fig:sim-grid-r2-0p5` and `fig:sim-grid-r2-2`.

| PDF (`results/`) | Plot script (`plots/`) | Upstream NPZ producer (`simulations/`) |
| --- | --- | --- |
| `inference_r2_0p5_grid_coverage_inputtest.pdf` | `plot_r2_1_grid_inputtest.py` | `run_r2_0p5_and_2_grid_inputtest_parallel.py` |
| `inference_r2_0p5_grid_width_inputtest.pdf` | `plot_r2_1_grid_inputtest.py` | same |
| `inference_r2_2_grid_coverage_inputtest.pdf` | `plot_r2_1_grid_inputtest.py` | same |
| `inference_r2_2_grid_width_inputtest.pdf` | `plot_r2_1_grid_inputtest.py` | same |

## Figures published on 2026-05-30 (truncation-level sensitivity, beta4 slope)

Comparison of $\varphi^{\anitr}$ and $\varphi^{\adptr}$ at three common truncation
levels $J\in\{\lfloor n^{0.2}\rfloor,\lfloor n^{0.3}\rfloor,\lfloor n^{0.4}\rfloor\}$,
fixed to the $b_k\propto k^{-4}$ slope. Grid layout: rows = $r_2\in\{0.5,1,2\}$,
cols = covariance design. Referenced as `fig:sim-grid-trunc`.

| PDF (`results/`) | Plot script (`plots/`) | Upstream NPZ producer (`simulations/`) |
| --- | --- | --- |
| `inference_trunc_sensitivity_legend_inputtest.pdf` | `plot_trunc_sensitivity_beta4_inputtest.py` (legend strip) | -- |
| `inference_trunc_sensitivity_coverage_inputtest.pdf` | `plot_trunc_sensitivity_beta4_inputtest.py` | `run_trunc_sensitivity_beta4_inputtest_parallel.py` |
| `inference_trunc_sensitivity_width_inputtest.pdf` | `plot_trunc_sensitivity_beta4_inputtest.py` | same |

## Other "3col / 6m" / width figures

| PDF | Plot script | Upstream |
| --- | --- | --- |
| `inference_r2_3_normal_6m.pdf` | `plot_r2_3_normal_6m.py` | `run_alpha_sweep_v13.py` |
| `inference_r2_3_sparse2_6m.pdf` | `plot_r2_3_sparse2_6m.py` | `run_alpha_sweep_v13.py` |
| `inference_r2_3_beta4_6m.pdf` | `plot_r2_3_beta4_6m.py` | `run_alpha_sweep_v13.py` |
| `width_noFPCA_C1_3col.pdf` | `plot_width_noFPCA_3col.py` | v13 family |
| `width_r2_3_beta4_n{n}_noFPCA.pdf` | `plot_r2_3_beta4_n20k_width.py` | `run_alpha_sweep_v13.py` (n=20000 variant) |
| `inference_trunc_5x.pdf` | `plot_trunc_5x.py` | `run_trunc_5x.py` |
| `inference_adapt_r3_q02_sparse.pdf` | `plot_adapt_r3_q02.py` | `run_adapt_r3_q02.py` |

## `plot_alpha_sweep_v13*` family (Ca/Ci sweeps)

These share output path template `inference_alpha_sweep_v13_..._Ca{C_a}_Ci{C_i}.pdf`.
Upstream NPZs: `run_alpha_sweep_v13*.py` (and `rerun_*` helpers).

| Plot script | Output pattern |
| --- | --- |
| `plot_alpha_sweep_v13.py` | `inference_alpha_sweep_v13_7methods_Ca{Ca}_Ci{Ci}.pdf` |
| `plot_alpha_sweep_v13_r2_3.py` | `..._v13_r2_3_7methods_Ca{Ca}_Ci{Ci}.pdf` |
| `plot_alpha_sweep_v13_r2_1p5.py` | `..._v13_r2_1p5_7methods_Ca{Ca}_Ci{Ci}.pdf` |
| `plot_alpha_sweep_v13_beta4.py` | `..._v13_beta4_7methods_Ca{Ca}_Ci{Ci}.pdf` |
| `plot_alpha_sweep_v13_sparse.py` | `..._v13_sparse_7methods_Ca{Ca}_Ci{Ci}.pdf` |
| `plot_alpha_sweep_v13_r2_3_6m.py` | `..._v13_r2_3_6m_Ca{Ca}_Ci{Ci}.pdf` |
| `plot_alpha_sweep_v13_r2_3_sparse_6m.py` | `..._v13_r2_3_sparse_6m_Ca{Ca}_Ci{Ci}.pdf` |
| `plot_alpha_sweep_v13_r2_3_beta4_6m.py` | `..._v13_r2_3_beta4_6m_Ca{Ca}_Ci{Ci}.pdf` |

## Earlier `alpha_sweep` families (vN = version number)

Every `plot_alpha_sweep_vN.py` writes `inference_alpha_sweep_vN_3cases.pdf` (unless
a variant suffix like `_lepski`, `_aniso_kappa`, `_aligned`, `_onesided`, `_predrisk`
is present). Producers are the matching `run_alpha_sweep_vN.py` scripts under
`simulations/`. Only v5, v6, v7, v8, v9, v10, v11, v12, v14 have saved `run_*`
counterparts in `simulations/`; later versions (v15–v45) apparently reused v13
NPZs with updated plotting only.

Notable non-`_3cases` variants:
- `plot_alpha_sweep_v6.py` → `inference_alpha_sweep_v6_onesided.pdf`
- `plot_alpha_sweep_v7.py` → `inference_alpha_sweep_v7_predrisk.pdf`
- `plot_alpha_sweep_v8.py` → `inference_alpha_sweep_v8_aligned.pdf`
- `plot_alpha_sweep_v8_haar.py` → `inference_alpha_sweep_v8_haar.pdf`
- `plot_alpha_sweep_v41.py` → `inference_alpha_sweep_v41_lepski.pdf`
- `plot_alpha_sweep_v42.py` / `v43.py` / `v44.py` → `..._aniso_kappa.pdf`
- `plot_alpha_sweep_v45.py` → `inference_alpha_sweep_v45_aniso.pdf`
- `plot_alpha_sweep_predrisk.py` → `inference_alpha_sweep_predrisk_3cases.pdf`
- `plot_alpha_sweep_r2_2.py` / `_r2_2_t1000.py` → `inference_alpha_sweep_r2_2[_t1000].pdf`

## Figures for the paper

| PDF | Plot script | Upstream |
| --- | --- | --- |
| `figure2.pdf` | `plot_figure2.py` | `run_figure2.py` |
| `figure3.pdf` | `plot_figure3.py` | `run_figure3_top.py`, `run_figure3_bottom.py` |

## Quick-preview / ad hoc

- `quick_preview_sparse2.pdf` — no saved plot script (likely REPL).

## Maintenance note

This file was auto-built on 2026-04-21 from `savefig` calls across
`plots/*.py`. If you add a new plot script, append its entry here (or rerun
the one-liner: `for f in plots/*.py; do grep -oE "'[^']*\.pdf'|\"[^\"]*\.pdf\"" "$f" | head -1; done`).
