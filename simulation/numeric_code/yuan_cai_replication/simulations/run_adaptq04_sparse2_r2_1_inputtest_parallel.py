"""Parallel runner: Adapt.Trunc q=0.4 with INPUT test functions, for the
sparse2 cases at r2=1 that are not yet on disk.

Pairs with the existing Trunc.Aniso q=0.4 column already in the base
alpha_sweep_*_v13_inputtest.npz so the grid figures can compare both methods
at a common truncation level. Other slope/design pairs (normal × all, beta4 × all)
already have adaptq0p4 inputtest NPZs; only sparse2 is missing.
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
    case, n, results_dir = args
    fname = f'alpha_sweep_{case}_n{n}_adaptq0p4_v13_inputtest.npz'
    if os.path.exists(os.path.join(results_dir, fname)):
        print(f'  SKIP {fname}', flush=True)
        return
    print(f'  START {case} n={n} (adaptq0p4 inputtest)', flush=True)
    from run_adapt_q03_only import run
    run(case, n, results_dir, q_adapt=0.4, test_func_kind='input')
    print(f'  DONE  {case} n={n} (adaptq0p4 inputtest)', flush=True)


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    # Only the sparse2 cases at r2=1 are missing adaptq0p4_inputtest.
    cases = [
        'aligned_r2_1_sparse2', 'haar_r2_1_sparse2', 'shifted_sparse2',
    ]
    jobs = []
    for n in [1000, 2000, 4000]:
        for case in cases:
            jobs.append((case, n, results_dir))

    n_workers = 6
    print(f'Running {len(jobs)} configs across {n_workers} workers', flush=True)
    with mp.get_context('spawn').Pool(n_workers) as pool:
        pool.map(_worker, jobs)
    print('All done.', flush=True)
