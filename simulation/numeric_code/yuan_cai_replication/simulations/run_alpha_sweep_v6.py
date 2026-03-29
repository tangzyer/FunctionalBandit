"""Alpha sweep with fixed lambda = 1/n.

Supports aligned_r2_2, haar_r2_2, shifted cases.
Aniso divisor: (logn)^4
Iso divisor: logn

Usage: python run_alpha_sweep_v6.py <case_name> <n_val>
  case_name: aligned_r2_2, haar_r2_2, shifted
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.covariance import figure2_specs, figure3_top_specs, figure3_bottom_specs
from src.data_generation import (generate_data_cosine_basis,
                                  generate_data_haar_basis, true_beta_coeffs)
from src.estimators import rkhs_penalty_matrix


def run(case_name, n_val, results_dir):
    K = 50
    sigma = 0.5
    n_datasets = 1000
    n_test = 1000
    n_grid = 200
    lam_grid = np.logspace(-12, 0, n_grid)

    if case_name == 'aligned_r2_2':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
    elif case_name == 'shifted':
        cov_spec = figure3_top_specs(K, k0_values=[10])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
    elif case_name == 'haar_r2_2':
        cov_spec = figure3_bottom_specs(K, r2_values=[2.0])[0]
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
    else:
        raise ValueError(f'Unknown case: {case_name}')

    b_true = true_beta_coeffs(K)
    D = rkhs_penalty_matrix(K)

    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** 4
    sqrt_mu = np.sqrt(mu_k)

    logn = np.log(n_val)
    loglogn = np.log(np.log(n_val))
    div_aniso = 10000.0
    div_iso = 10000.0

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

    n_methods = 5
    all_yhat = np.zeros((n_methods, n_datasets, n_test))
    all_se = np.zeros((n_methods, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)

    for ds in range(n_datasets):
        if ds % 100 == 0:
            print(f'  {case_name} n={n_val}: dataset {ds}/{n_datasets}',
                  flush=True)

        rng = np.random.default_rng(42 + ds)
        Z, Y, _, cov_Z = gen_func(n_val, rng)
        n = Z.shape[0]
        ZtZ = Z.T @ Z / n
        ZtY = Z.T @ Y / n

        # --- Fixed lambda = 1/n ---
        best_lam_iso = 1.0 / n_val ** 1.1
        best_lam_aniso = 1.0 / n_val ** 1.1

        # Full-data aniso eigendecomposition
        Z_tilde = Z * sqrt_mu
        T_n = Z_tilde.T @ Z_tilde / n
        eigvals_T, U_T = np.linalg.eigh(T_n)
        order = np.argsort(eigvals_T)[::-1]
        d_eig = np.maximum(eigvals_T[order], 0)
        U_T = U_T[:, order]
        c_est = U_T.T @ (Z_tilde.T @ Y / n)

        # --- Aniso oracle (m=0) ---
        lam0 = best_lam_aniso
        nu0 = np.minimum(lam0, np.sqrt(d_eig * lam0))
        den0 = d_eig + nu0
        gam0 = np.where(den0 > 1e-15, c_est / den0, 0.0)
        b0 = sqrt_mu * (U_T @ gam0)
        sig2_0 = np.sum((Y - Z @ b0) ** 2) / n
        c_new = mx_test @ U_T
        wv0 = np.where(den0 > 1e-15, d_eig / den0 ** 2, 0.0)
        all_yhat[0, ds] = x_test @ b0
        all_se[0, ds] = np.sqrt(sig2_0 / n * (c_new ** 2 @ wv0))

        # --- Aniso divided (m=1) ---
        lam1 = best_lam_aniso / div_aniso
        nu1 = np.minimum(lam1, np.sqrt(d_eig * lam1))
        den1 = d_eig + nu1
        gam1 = np.where(den1 > 1e-15, c_est / den1, 0.0)
        b1 = sqrt_mu * (U_T @ gam1)
        sig2_1 = np.sum((Y - Z @ b1) ** 2) / n
        wv1 = np.where(den1 > 1e-15, d_eig / den1 ** 2, 0.0)
        all_yhat[1, ds] = x_test @ b1
        all_se[1, ds] = np.sqrt(sig2_1 / n * (c_new ** 2 @ wv1))

        # --- Iso oracle (m=2) ---
        lam2 = best_lam_iso
        A2 = ZtZ + lam2 * np.diag(D)
        A2_inv = np.linalg.inv(A2)
        b2 = A2_inv @ ZtY
        sig2_2 = np.sum((Y - Z @ b2) ** 2) / n
        xAi2 = x_test @ A2_inv
        all_yhat[2, ds] = x_test @ b2
        all_se[2, ds] = np.sqrt(sig2_2 / n * np.sum(xAi2 * x_test, axis=1))

        # --- Iso divided (m=3) ---
        lam3 = best_lam_iso / div_iso
        A3 = ZtZ + lam3 * np.diag(D)
        A3_inv = np.linalg.inv(A3)
        b3 = A3_inv @ ZtY
        sig2_3 = np.sum((Y - Z @ b3) ** 2) / n
        xAi3 = x_test @ A3_inv
        all_yhat[3, ds] = x_test @ b3
        all_se[3, ds] = np.sqrt(sig2_3 / n * np.sum(xAi3 * x_test, axis=1))

        # --- FPCA (m=4) ---
        S = Z.T @ Z / n
        eigvals_S, eigvecs_S = np.linalg.eigh(S)
        order_S = np.argsort(eigvals_S)[::-1]
        eigvecs_S = eigvecs_S[:, order_S]
        J = int(np.sqrt(n_val))
        W = Z @ eigvecs_S[:, :J]
        WtW = W.T @ W
        Wi = np.linalg.inv(WtW)
        gf = Wi @ (W.T @ Y)
        rf = Y - W @ gf
        sig2_f = np.sum(rf ** 2) / (n_val - J)
        w_new = x_test @ eigvecs_S[:, :J]
        wWi = w_new @ Wi
        all_yhat[4, ds] = w_new @ gf
        all_se[4, ds] = np.sqrt(sig2_f * np.sum(wWi * w_new, axis=1))

    fname = f'alpha_sweep_{case_name}_n{n_val}_v6.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas)
    print(f'Saved {fname}')


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run(case_name, n_val, results_dir)
