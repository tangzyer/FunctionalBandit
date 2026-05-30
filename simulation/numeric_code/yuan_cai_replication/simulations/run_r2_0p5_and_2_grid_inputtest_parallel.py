"""Parallel runner: produce the missing base v13 AND adaptq0p4 inputtest
NPZs needed for the r2=0.5 and r2=2 grid figures (5 methods, slope x design).

Plan:
  r2=0.5: 6 base + 6 adaptq0p4   (only the sparse2 aligned/haar cases)
  r2=2:   12 base + 12 adaptq0p4 (normal aligned/haar + sparse2 aligned/haar)
Total: 36 fresh NPZs (skips anything already on disk).
"""
import os
# Pin BLAS to 1 thread per worker BEFORE numpy is imported in any child.
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# kind in {'base', 'adapt'}
def _worker(args):
    kind, case, n, results_dir = args
    if kind == 'base':
        fname = (f'alpha_sweep_{case}_n{n}_Ca0.005_Ci0.005'
                 f'_v13_inputtest.npz')
    else:  # adapt
        fname = f'alpha_sweep_{case}_n{n}_adaptq0p4_v13_inputtest.npz'
    if os.path.exists(os.path.join(results_dir, fname)):
        print(f'  SKIP {fname}', flush=True)
        return
    print(f'  START [{kind}] {case} n={n}', flush=True)
    if kind == 'base':
        from run_alpha_sweep_v13 import run
        run(case, n, results_dir, C_aniso=0.005, C_iso=0.005,
            test_func_kind='input')
    else:
        from run_adapt_q03_only import run
        run(case, n, results_dir, q_adapt=0.4, test_func_kind='input')
    print(f'  DONE  [{kind}] {case} n={n}', flush=True)


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # All 9 (slope, design) case names at r2=0.5 and r2=2.
    cases_per_r2 = {
        0.5: [
            'aligned_r2_0p5', 'haar_r2_0p5', 'shifted',
            'aligned_r2_0p5_beta4', 'haar_r2_0p5_beta4', 'shifted_beta4',
            'aligned_r2_0p5_sparse2', 'haar_r2_0p5_sparse2', 'shifted_sparse2',
        ],
        2.0: [
            'aligned_r2_2', 'haar_r2_2', 'shifted',
            'aligned_r2_2_beta4', 'haar_r2_2_beta4', 'shifted_beta4',
            'aligned_r2_2_sparse2', 'haar_r2_2_sparse2', 'shifted_sparse2',
        ],
    }

    jobs = []
    for cases in cases_per_r2.values():
        for n in [1000, 2000, 4000]:
            for case in cases:
                jobs.append(('base', case, n, results_dir))
                jobs.append(('adapt', case, n, results_dir))

    n_workers = 6
    print(f'Dispatching {len(jobs)} configs across {n_workers} workers '
          '(SKIPs are cheap)', flush=True)
    with mp.get_context('spawn').Pool(n_workers) as pool:
        pool.map(_worker, jobs)
    print('All done.', flush=True)
