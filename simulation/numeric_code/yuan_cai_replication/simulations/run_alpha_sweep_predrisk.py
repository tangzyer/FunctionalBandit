"""Alpha sweep with prediction-risk tuning of lambda per test function.

Instead of finding one oracle lambda per dataset that minimizes
  excess_risk = (b̂ - b_true)^T C_Z (b̂ - b_true),
we find a separate oracle lambda for each test function x minimizing
  pred_risk(x) = (x^T (b̂(λ) - b_true))^2.

Then inference_lambda(x) = oracle_lambda(x) / divisor.

Usage: python run_alpha_sweep_predrisk.py <case_name> <n_val>
  case_name: aligned_r2_2, shifted, haar_r2_2
  n_val: 64, 256, 1024
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.covariance import figure2_specs, figure3_top_specs, figure3_bottom_specs
from src.data_generation import (generate_data_cosine_basis,
                                  generate_data_haar_basis, true_beta_coeffs)
from src.estimators import rkhs_penalty_matrix


def run_case(case_name, n_val, results_dir):
    K = 50
    sigma = 0.5
    n_datasets = 1000
    n_test = 1000
    n_grid = 200
    lam_grid = np.logspace(-12, 2, n_grid)

    # Covariance spec
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

    # RKHS kernel eigenvalues
    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** 4
    sqrt_mu = np.sqrt(mu_k)

    # Divisors
    logn = np.log(n_val)
    loglogn = np.log(np.log(n_val))
    div_aniso = logn ** 3
    div_iso = loglogn

    # Generate BM test functions (same seed as before)
    rng_test = np.random.default_rng(999)
    n_grid_pts = 200
    t_grid = np.linspace(0, 1, n_grid_pts + 1)[1:]
    dt = 1.0 / n_grid_pts
    dW = rng_test.normal(size=(n_test, n_grid_pts)) * np.sqrt(dt)
    bm_paths = np.cumsum(dW, axis=1)
    phi_grid = np.sqrt(2) * np.cos(np.outer(ks, np.pi * t_grid))  # (K, n_grid_pts)
    x_test = bm_paths @ phi_grid.T * dt  # (n_test, K)
    true_vals = x_test @ b_true  # (n_test,)

    # Precompute transformed test functions for aniso
    mx_test = x_test * sqrt_mu  # (n_test, K)

    # Output: 5 methods x n_datasets x n_test
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

        # ---- Precompute b_hat(λ) for all λ: isotropic ----
        b_hats_iso = np.zeros((n_grid, K))
        for li, lam in enumerate(lam_grid):
            A = ZtZ + lam * np.diag(D)
            b_hats_iso[li] = np.linalg.solve(A, ZtY)

        # ---- Precompute b_hat(λ) for all λ: anisotropic ----
        Z_tilde = Z * sqrt_mu
        T_n = Z_tilde.T @ Z_tilde / n
        eigvals_T, U_T = np.linalg.eigh(T_n)
        order = np.argsort(eigvals_T)[::-1]
        d_eig = np.maximum(eigvals_T[order], 0)
        U_T = U_T[:, order]
        c_est = U_T.T @ (Z_tilde.T @ Y / n)

        b_hats_aniso = np.zeros((n_grid, K))
        for li, lam in enumerate(lam_grid):
            nu = np.minimum(lam, np.sqrt(d_eig * lam))
            denom = d_eig + nu
            gamma = np.where(denom > 1e-15, c_est / denom, 0.0)
            b_tilde = U_T @ gamma
            b_hats_aniso[li] = sqrt_mu * b_tilde

        # ---- Predictions at all (test_fn, λ) pairs ----
        preds_iso = x_test @ b_hats_iso.T      # (n_test, n_grid)
        preds_aniso = x_test @ b_hats_aniso.T  # (n_test, n_grid)

        # ---- Prediction squared error ----
        pred_err_iso = (preds_iso - true_vals[:, None]) ** 2
        pred_err_aniso = (preds_aniso - true_vals[:, None]) ** 2

        # ---- Best λ per test function ----
        best_idx_iso = np.argmin(pred_err_iso, axis=1)
        best_idx_aniso = np.argmin(pred_err_aniso, axis=1)

        # ---- Anisotropic: oracle (m=0) and divided (m=1) ----
        # Precompute projected test functions in T_n eigenbasis
        c_new_all = mx_test @ U_T  # (n_test, K)

        for u_idx in np.unique(best_idx_aniso):
            mask = (best_idx_aniso == u_idx)
            x_sub = x_test[mask]
            c_sub = c_new_all[mask]
            lam_oracle = lam_grid[u_idx]

            # Oracle (m=0): inference at lam_oracle
            nu_o = np.minimum(lam_oracle, np.sqrt(d_eig * lam_oracle))
            den_o = d_eig + nu_o
            gam_o = np.where(den_o > 1e-15, c_est / den_o, 0.0)
            b_hat_o = sqrt_mu * (U_T @ gam_o)
            sig2_o = np.sum((Y - Z @ b_hat_o) ** 2) / n
            wv_o = np.where(den_o > 1e-15, d_eig / den_o ** 2, 0.0)
            all_yhat[0, ds, mask] = x_sub @ b_hat_o
            all_se[0, ds, mask] = np.sqrt(sig2_o / n * (c_sub ** 2 @ wv_o))

            # Divided (m=1): inference at lam_oracle / div_aniso
            lam_inf = lam_oracle / div_aniso
            nu_i = np.minimum(lam_inf, np.sqrt(d_eig * lam_inf))
            den_i = d_eig + nu_i
            gam_i = np.where(den_i > 1e-15, c_est / den_i, 0.0)
            b_hat_i = sqrt_mu * (U_T @ gam_i)
            sig2_i = np.sum((Y - Z @ b_hat_i) ** 2) / n
            wv_i = np.where(den_i > 1e-15, d_eig / den_i ** 2, 0.0)
            all_yhat[1, ds, mask] = x_sub @ b_hat_i
            all_se[1, ds, mask] = np.sqrt(sig2_i / n * (c_sub ** 2 @ wv_i))

        # ---- Isotropic: oracle (m=2) and divided (m=3) ----
        for u_idx in np.unique(best_idx_iso):
            mask = (best_idx_iso == u_idx)
            x_sub = x_test[mask]
            lam_oracle = lam_grid[u_idx]

            # Oracle (m=2): inference at lam_oracle
            A_o = ZtZ + lam_oracle * np.diag(D)
            A_inv_o = np.linalg.inv(A_o)
            b_hat_o = A_inv_o @ ZtY
            sig2_o = np.sum((Y - Z @ b_hat_o) ** 2) / n
            xAi_o = x_sub @ A_inv_o
            all_yhat[2, ds, mask] = x_sub @ b_hat_o
            all_se[2, ds, mask] = np.sqrt(
                sig2_o / n * np.sum(xAi_o * x_sub, axis=1))

            # Divided (m=3): inference at lam_oracle / div_iso
            lam_inf = lam_oracle / div_iso
            A_i = ZtZ + lam_inf * np.diag(D)
            A_inv_i = np.linalg.inv(A_i)
            b_hat_i = A_inv_i @ ZtY
            sig2_i = np.sum((Y - Z @ b_hat_i) ** 2) / n
            xAi_i = x_sub @ A_inv_i
            all_yhat[3, ds, mask] = x_sub @ b_hat_i
            all_se[3, ds, mask] = np.sqrt(
                sig2_i / n * np.sum(xAi_i * x_sub, axis=1))

        # ---- FPCA (m=4): J = sqrt(n), unchanged ----
        S = Z.T @ Z / n
        eigvals_S, eigvecs_S = np.linalg.eigh(S)
        order_S = np.argsort(eigvals_S)[::-1]
        eigvecs_S = eigvecs_S[:, order_S]

        J = int(np.sqrt(n_val))
        W = Z @ eigvecs_S[:, :J]
        WtW = W.T @ W
        WtW_inv = np.linalg.inv(WtW)
        gamma_f = WtW_inv @ (W.T @ Y)
        resid_f = Y - W @ gamma_f
        sig2_f = np.sum(resid_f ** 2) / (n_val - J)

        w_new = x_test @ eigvecs_S[:, :J]
        wWi = w_new @ WtW_inv
        all_yhat[4, ds, :] = w_new @ gamma_f
        all_se[4, ds, :] = np.sqrt(sig2_f * np.sum(wWi * w_new, axis=1))

    # Save
    fname = f'alpha_sweep_{case_name}_n{n_val}_v4.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas)
    print(f'Saved {fname}')


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run_case(case_name, n_val, results_dir)
