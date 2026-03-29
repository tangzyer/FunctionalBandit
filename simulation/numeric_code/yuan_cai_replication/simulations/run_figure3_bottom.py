"""Simulation for Figure 3 bottom: Haar eigenbasis (basis misalignment)."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.covariance import figure3_bottom_specs
from src.data_generation import generate_data_haar_basis
from src.estimators import oracle_roughness_reg, oracle_fpca


def run_figure3_bottom(
    K=200,
    M=200,
    n_values=(32, 64, 128, 256, 512, 1024),
    r2_values=(1.0, 1.5, 2.0, 2.5),
    n_rep=200,
    sigma=0.5,
    seed=456,
    save_path=None,
):
    """Run Figure 3 bottom row simulations.

    Returns
    -------
    results : dict with keys:
        'n_values', 'r2_values',
        'risk_reg' : shape (len(r2_values), len(n_values), n_rep)
        'risk_fpca': shape (len(r2_values), len(n_values), n_rep)
    """
    specs = figure3_bottom_specs(K, r2_values)
    n_values = list(n_values)
    r2_values = list(r2_values)

    risk_reg = np.zeros((len(r2_values), len(n_values), n_rep))
    risk_fpca = np.zeros((len(r2_values), len(n_values), n_rep))

    rng = np.random.default_rng(seed)

    total = len(r2_values) * len(n_values) * n_rep
    count = 0

    for i, spec in enumerate(specs):
        for j, n in enumerate(n_values):
            for rep in range(n_rep):
                Z, Y, b_true, cov_Z = generate_data_haar_basis(
                    n, spec, K, M=M, sigma=sigma, rng=rng
                )

                _, r_reg, _ = oracle_roughness_reg(Z, Y, b_true, K, cov_Z)
                _, r_fpca, _ = oracle_fpca(Z, Y, b_true, K, cov_Z)

                risk_reg[i, j, rep] = r_reg
                risk_fpca[i, j, rep] = r_fpca

                count += 1
                if count % 500 == 0:
                    print(f"  Figure 3 bottom: {count}/{total} done")

    results = {
        'n_values': np.array(n_values),
        'r2_values': np.array(r2_values),
        'risk_reg': risk_reg,
        'risk_fpca': risk_fpca,
    }

    if save_path is not None:
        np.savez(save_path, **results)
        print(f"  Saved Figure 3 bottom results to {save_path}")

    return results


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run_figure3_bottom(save_path=os.path.join(results_dir, 'figure3_bottom.npz'))
