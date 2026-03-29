"""Simulation for Figure 2: aligned cosine eigenbasis."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.covariance import figure2_specs
from src.data_generation import generate_data_cosine_basis
from src.estimators import oracle_roughness_reg, oracle_fpca


def run_figure2(
    K=200,
    n_values=(32, 64, 128, 256, 512, 1024),
    r2_values=(1.0, 1.5, 2.0, 2.5, 3.0),
    n_rep=200,
    sigma=0.5,
    seed=42,
    save_path=None,
):
    """Run Figure 2 simulations.

    Returns
    -------
    results : dict with keys:
        'n_values', 'r2_values',
        'risk_reg' : shape (len(r2_values), len(n_values), n_rep)
        'risk_fpca': shape (len(r2_values), len(n_values), n_rep)
    """
    specs = figure2_specs(K, r2_values)
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
                Z, Y, b_true, cov_Z = generate_data_cosine_basis(
                    n, spec, K, sigma=sigma, rng=rng
                )

                _, r_reg, _ = oracle_roughness_reg(Z, Y, b_true, K, cov_Z)
                _, r_fpca, _ = oracle_fpca(Z, Y, b_true, K, cov_Z)

                risk_reg[i, j, rep] = r_reg
                risk_fpca[i, j, rep] = r_fpca

                count += 1
                if count % 500 == 0:
                    print(f"  Figure 2: {count}/{total} done")

    results = {
        'n_values': np.array(n_values),
        'r2_values': np.array(r2_values),
        'risk_reg': risk_reg,
        'risk_fpca': risk_fpca,
    }

    if save_path is not None:
        np.savez(save_path, **results)
        print(f"  Saved Figure 2 results to {save_path}")

    return results


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run_figure2(save_path=os.path.join(results_dir, 'figure2.npz'))
