"""Parallel runner: r2=1 normal/beta4/sparse2 3col panels with INPUT test functions.

Mirrors run_r2_1_3col_parallel.py but passes test_func_kind='input' so that
test covariates are drawn from the same distribution as the training inputs
(producing the *_inputtest.npz output files). Only the sparse2 cases at r2=1
are missing; everything else skips via filename checkpoint.
"""
import os
# Pin BLAS to 1 thread per worker BEFORE numpy is imported in any child.
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _worker(args):
    case, n, results_dir, C_a, C_i = args
    fname = f'alpha_sweep_{case}_n{n}_Ca{C_a}_Ci{C_i}_v13_inputtest.npz'
    if os.path.exists(os.path.join(results_dir, fname)):
        print(f'  SKIP {fname}', flush=True)
        return
    print(f'  START {case} n={n} (inputtest)', flush=True)
    from run_alpha_sweep_v13 import run
    run(case, n, results_dir, C_aniso=C_a, C_iso=C_i, test_func_kind='input')
    print(f'  DONE  {case} n={n} (inputtest)', flush=True)


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    cases = [
        'aligned_r2_1', 'haar_r2_1', 'shifted',
        'aligned_r2_1_beta4', 'haar_r2_1_beta4', 'shifted_beta4',
        'aligned_r2_1_sparse2', 'haar_r2_1_sparse2', 'shifted_sparse2',
    ]
    C_a, C_i = 0.005, 0.005
    jobs = []
    for n in [1000, 2000, 4000]:
        for case in cases:
            jobs.append((case, n, results_dir, C_a, C_i))

    n_workers = 6
    print(f'Running {len(jobs)} configs across {n_workers} workers', flush=True)
    with mp.get_context('spawn').Pool(n_workers) as pool:
        pool.map(_worker, jobs)
    print('All done.', flush=True)
