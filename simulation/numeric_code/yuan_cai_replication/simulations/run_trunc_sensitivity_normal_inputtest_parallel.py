"""Parallel runner for the truncation-level sensitivity figure, NORMAL slope.

Companion to run_trunc_sensitivity_beta4_inputtest_parallel.py, but for the
D1-FAILS slope b_k ∝ k^{-2} (the default `true_beta_coeffs` power=2), whose
case names carry no `_beta4` suffix.

Generates the missing NPZs needed for:
  Trunc.Aniso q in {0.2, 0.3, 0.4}  x  Adapt.Trunc q in {0.2, 0.3, 0.4}
across r_2 in {0.5, 1, 2} and designs (Aligned, Haar), with INPUT-distribution
test functions.

Sources of q variants (same as the beta4 runner):
  - Trunc.Aniso q=0.3 and q=0.4 already live in the base v13 NPZ (m=3, m=4).
  - Trunc.Aniso q=0.2 -> truncq0p20_v13_inputtest.npz  (run_trunc_aniso_only).
  - Adapt.Trunc q=0.X -> adaptq0pX_v13_inputtest.npz   (run_adapt_q03_only).

Each completed (kind, q, case, n) writes its own NPZ; the worker SKIPs any file
that already exists, so the run is checkpointed and resumable.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# args: ('trunc' | 'adapt', q_val, case, n, results_dir)
def _worker(args):
    kind, q_val, case, n, results_dir = args
    suffix_int = int(round(q_val * 100)) if kind == 'trunc' else int(round(q_val * 10))
    suffix = (f'truncq0p{suffix_int:02d}' if kind == 'trunc'
              else f'adaptq0p{suffix_int}')
    fname = f'alpha_sweep_{case}_n{n}_{suffix}_v13_inputtest.npz'
    if os.path.exists(os.path.join(results_dir, fname)):
        print(f'  SKIP {fname}', flush=True)
        return
    print(f'  START [{kind} q={q_val}] {case} n={n}', flush=True)
    if kind == 'trunc':
        from run_trunc_aniso_only import run
        run(case, n, results_dir, q_trunc=q_val, C_aniso=0.005,
            test_func_kind='input')
    else:
        from run_adapt_q03_only import run
        run(case, n, results_dir, q_adapt=q_val, test_func_kind='input')
    print(f'  DONE  [{kind} q={q_val}] {case} n={n}', flush=True)


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    # Normal-slope (b_k ∝ k^{-2}) cases; only Aligned/Haar are used by the
    # figure (2 rows x 3 r_2 columns).
    normal_cases = [
        'aligned_r2_0p5', 'haar_r2_0p5',
        'aligned_r2_1',   'haar_r2_1',
        'aligned_r2_2',   'haar_r2_2',
    ]

    jobs = []
    for n in [1000, 2000, 4000]:
        # Trunc.Aniso at q=0.2 -> truncq0p20 family.
        for case in normal_cases:
            jobs.append(('trunc', 0.2, case, n, results_dir))
        # Adapt.Trunc at q in {0.2, 0.3, 0.4}; existing files are skipped.
        for q_val in (0.2, 0.3, 0.4):
            for case in normal_cases:
                jobs.append(('adapt', q_val, case, n, results_dir))

    n_workers = 4
    print(f'Dispatching {len(jobs)} configs across {n_workers} workers', flush=True)
    with mp.get_context('spawn').Pool(n_workers) as pool:
        pool.map(_worker, jobs)
    print('All done.', flush=True)
