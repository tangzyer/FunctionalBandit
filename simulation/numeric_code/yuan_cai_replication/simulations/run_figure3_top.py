"""Simulation for Figure 3 top: shifted location (k₀) misalignment."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.covariance import figure3_top_specs
from src.data_generation import generate_data_cosine_basis
from src.estimators import oracle_roughness_reg, oracle_fpca


def run_figure3_top(
    K=200,
    n_values=(32, 64, 128, 256, 512, 1024),
    k0_values=(5, 10, 15, 20),
    n_rep=200,
    sigma=0.5,
    seed=123,
    save_path=None,
):
    """Run Figure 3 top row simulations.

    Returns
    -------
    results : dict with keys:
        'n_values', 'k0_values',
        'risk_reg' : shape (len(k0_values), len(n_values), n_rep)
        'risk_fpca': shape (len(k0_values), len(n_values), n_rep)
    """
    specs = figure3_top_specs(K, k0_values)
    n_values = list(n_values)
    k0_values = list(k0_values)

    risk_reg = np.zeros((len(k0_values), len(n_values), n_rep))
    risk_fpca = np.zeros((len(k0_values), len(n_values), n_rep))

    rng = np.random.default_rng(seed)

    total = len(k0_values) * len(n_values) * n_rep
    count = 0

    for i, spec in enumerate(specs):
        for j, n in enumerate(n_values):
            for rep in range(n_rep):
                Z, Y, b_true, cov_Z = generate_data_cosine_basis(
                    n, spec, K, sigma=sigma, rng=rng
                )

                _, r_reg, _ = oracle_roughness_reg(Z, Y, b_true, K, cov_Z)
                _, r_fpca, _ = oracle_fpca(Z, Y, b_true, K, cov_Z)

                risk_reg[i, j, rep] = r_reg
                risk_fpca[i, j, rep] = r_fpca

                count += 1
                if count % 500 == 0:
                    print(f"  Figure 3 top: {count}/{total} done")

    results = {
        'n_values': np.array(n_values),
        'k0_values': np.array(k0_values),
        'risk_reg': risk_reg,
        'risk_fpca': risk_fpca,
    }

    if save_path is not None:
        np.savez(save_path, **results)
        print(f"  Saved Figure 3 top results to {save_path}")

    return results


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run_figure3_top(save_path=os.path.join(results_dir, 'figure3_top.npz'))
