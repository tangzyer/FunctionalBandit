"""Run only Truncated Aniso with J=5*n^0.3 and J=5*n^0.4.

2 methods:
  m=0: Trunc.Aniso J=5*n^0.3
  m=1: Trunc.Aniso J=5*n^0.4

Usage: python run_trunc_5x.py <case_name> <n_val>
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.covariance import figure2_specs, figure3_top_specs, figure3_bottom_specs
from src.data_generation import generate_data_cosine_basis, generate_data_haar_basis, true_beta_coeffs


def run(case_name, n_val, results_dir):
    K = 200
    sigma = 0.5
    n_datasets = 1000
    n_test = 500

    if case_name == 'aligned_r2_2':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
    elif case_name == 'shifted':
        cov_spec = figure3_top_specs(K, k0_values=[50])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
    elif case_name == 'haar_r2_2':
        cov_spec = figure3_bottom_specs(K, r2_values=[2.0])[0]
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
    else:
        raise ValueError(f'Unknown case: {case_name}')

    b_true = true_beta_coeffs(K)

    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** 4
    sqrt_mu = np.sqrt(mu_k)

    # BM test functions
    rng_test = np.random.default_rng(999)
    n_grid_pts = 200
    t_grid = np.linspace(0, 1, n_grid_pts + 1)[1:]
    dt = 1.0 / n_grid_pts
    dW = rng_test.normal(size=(n_test, n_grid_pts)) * np.sqrt(dt)
    bm_paths = np.cumsum(dW, axis=1)
    phi_grid = np.sqrt(2) * np.cos(np.outer(ks, np.pi * t_grid))
    x_test = bm_paths @ phi_grid.T * dt
    true_vals = x_test @ b_true
    mx_test = x_test * sqrt_mu

    n_methods = 2  # 0: J=5*n^0.3, 1: J=5*n^0.4
    all_yhat = np.zeros((n_methods, n_datasets, n_test))
    all_se = np.zeros((n_methods, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)
    powers = [0.3, 0.4]
    J_vals = [min(int(5 * n_val ** p), K) for p in powers]
    print(f'  {case_name} n={n_val}: J values = {J_vals}')

    for ds in range(n_datasets):
        if ds % 100 == 0:
            print(f'  {case_name} n={n_val}: dataset {ds}/{n_datasets}', flush=True)

        rng = np.random.default_rng(42 + ds)
        Z, Y, _, cov_Z = gen_func(n_val, rng)
        n = Z.shape[0]

        # Aniso eigendecomposition
        Z_tilde = Z * sqrt_mu
        T_n = Z_tilde.T @ Z_tilde / n
        eigvals_T, U_T = np.linalg.eigh(T_n)
        order = np.argsort(eigvals_T)[::-1]
        d_eig = np.maximum(eigvals_T[order], 0)
        U_T = U_T[:, order]
        c_est = U_T.T @ (Z_tilde.T @ Y / n)
        c_new = mx_test @ U_T
        c_new_sq = c_new ** 2

        for mi, power in enumerate(powers):
            J_tr = J_vals[mi]
            gamma_tr = np.zeros(K)
            valid_tr = d_eig[:J_tr] > 1e-15
            gamma_tr[:J_tr] = np.where(valid_tr,
                                       c_est[:J_tr] / d_eig[:J_tr], 0.0)
            yhat_tr = c_new @ gamma_tr
            b_tr = (U_T @ gamma_tr) * sqrt_mu
            resid_tr = Y - Z @ b_tr
            sig2_tr = np.sum(resid_tr ** 2) / max(n - J_tr, 1)
            wv_tr = np.zeros(K)
            wv_tr[:J_tr] = np.where(valid_tr, 1.0 / d_eig[:J_tr], 0.0)
            se_tr = np.sqrt(sig2_tr / n * (c_new_sq @ wv_tr))
            all_yhat[mi, ds] = yhat_tr
            all_se[mi, ds] = se_tr

    fname = f'trunc_5x_{case_name}_n{n_val}.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas,
             J_vals=J_vals, powers=powers)
    print(f'Saved {fname}')


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run(case_name, n_val, results_dir)
