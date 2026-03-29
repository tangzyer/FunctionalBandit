"""Lepski-type adaptive lambda selection for inference.

For both aniso and iso methods, sweep over a lambda grid and use
an anticoncentration-based criterion to select the largest lambda
where the bias is undetectable relative to the standard error.

Criterion: for each test point, find the largest lambda such that
  |yhat(lambda) - yhat(lambda_ref)| <= kappa * SE(lambda)
where lambda_ref is the smallest lambda (most undersmoothed, ~OLS).

Methods:
  m=0,1,2,3: Aniso Lepski with kappa=0.01, 0.03, 0.1, 0.3
  m=4,5,6,7: Iso Lepski with kappa=0.01, 0.03, 0.1, 0.3
  m=8: FPCA

Usage: python run_alpha_sweep_v7.py <case_name> <n_val>
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


def lepski_select(yhat_grid, se_grid, kappa):
    """Select lambda index per test point using Lepski criterion.

    For each test point, find the largest lambda (highest grid index)
    such that |yhat(lambda) - yhat(lambda_ref)| <= kappa * SE(lambda),
    where lambda_ref is the smallest lambda (index 0).

    Args:
        yhat_grid: (n_grid, n_test) predictions at each lambda
        se_grid: (n_grid, n_test) standard errors at each lambda
        kappa: threshold constant

    Returns:
        best_idx: (n_test,) selected lambda index per test point
    """
    n_grid, n_test = yhat_grid.shape
    # Reference: smallest lambda (index 0, most undersmoothed ~ OLS)
    yhat_ref = yhat_grid[0]

    # Bias estimate: |yhat(lambda) - yhat(lambda_ref)|
    bias_est = np.abs(yhat_grid - yhat_ref[None, :])

    # Accept if bias <= kappa * SE
    accepted = bias_est <= kappa * se_grid

    # For each test point, find the largest accepted lambda index
    indices = np.arange(n_grid)[:, None]
    masked = np.where(accepted, indices, -1)
    best_idx = masked.max(axis=0)

    # Fallback to index 0 if none accepted (shouldn't happen)
    best_idx = np.maximum(best_idx, 0)
    return best_idx


def run(case_name, n_val, results_dir):
    K = 50
    sigma = 0.5
    n_datasets = 1000
    n_test = 1000
    n_grid = 1000
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

    kappas = [0.001, 0.003, 0.01]
    n_kappas = len(kappas)
    # m=0,1,2: Aniso Lepski (kappa=0.001, 0.003, 0.01)
    # m=3,4,5: Iso Lepski (kappa=0.001, 0.003, 0.01)
    # m=6: FPCA
    n_methods = 2 * n_kappas + 1
    all_yhat = np.zeros((n_methods, n_datasets, n_test))
    all_se = np.zeros((n_methods, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)

    selected_lam_aniso = np.zeros((n_kappas, n_datasets, n_test))
    selected_lam_iso = np.zeros((n_kappas, n_datasets, n_test))

    for ds in range(n_datasets):
        if ds % 100 == 0:
            print(f'  {case_name} n={n_val}: dataset {ds}/{n_datasets}',
                  flush=True)

        rng = np.random.default_rng(42 + ds)
        Z, Y, _, cov_Z = gen_func(n_val, rng)
        n = Z.shape[0]
        ZtZ = Z.T @ Z / n
        ZtY = Z.T @ Y / n

        # === Aniso: vectorized sweep over lambda grid ===
        Z_tilde = Z * sqrt_mu
        T_n = Z_tilde.T @ Z_tilde / n
        eigvals_T, U_T = np.linalg.eigh(T_n)
        order = np.argsort(eigvals_T)[::-1]
        d_eig = np.maximum(eigvals_T[order], 0)
        U_T = U_T[:, order]
        c_est = U_T.T @ (Z_tilde.T @ Y / n)
        c_new = mx_test @ U_T          # (n_test, K)
        c_new_sq = c_new ** 2           # (n_test, K)

        lam_arr = lam_grid[:, None]     # (n_grid, 1)
        d_arr = d_eig[None, :]          # (1, K)

        nu_all = np.minimum(lam_arr, np.sqrt(d_arr * lam_arr))   # (n_grid, K)
        den_all = d_arr + nu_all                                  # (n_grid, K)
        gamma_all = np.where(den_all > 1e-15,
                             c_est[None, :] / den_all, 0.0)      # (n_grid, K)

        # Predictions: (n_grid, n_test)
        yhat_aniso = gamma_all @ c_new.T

        # Coefficients b for each lambda -> residuals -> sigma^2
        b_all = (U_T @ gamma_all.T).T * sqrt_mu[None, :]  # (n_grid, K)
        resid_all = Y[:, None] - Z @ b_all.T               # (n, n_grid)
        sig2_all = np.sum(resid_all ** 2, axis=0) / n       # (n_grid,)

        # Variance weights and SE
        wv_all = np.where(den_all > 1e-15,
                          d_arr / den_all ** 2, 0.0)         # (n_grid, K)
        se_aniso = np.sqrt((sig2_all[:, None] / n) *
                           (wv_all @ c_new_sq.T))            # (n_grid, n_test)

        # Lepski selection for each kappa
        for ki, kappa in enumerate(kappas):
            best_idx = lepski_select(yhat_aniso, se_aniso, kappa)
            idx_range = np.arange(n_test)
            all_yhat[ki, ds] = yhat_aniso[best_idx, idx_range]
            all_se[ki, ds] = se_aniso[best_idx, idx_range]
            selected_lam_aniso[ki, ds] = lam_grid[best_idx]

        # === Iso: loop over lambda grid ===
        yhat_iso = np.zeros((n_grid, n_test))
        se_iso = np.zeros((n_grid, n_test))

        for gi, lam in enumerate(lam_grid):
            A = ZtZ + lam * np.diag(D)
            A_inv = np.linalg.inv(A)
            b = A_inv @ ZtY
            sig2 = np.sum((Y - Z @ b) ** 2) / n
            xAi = x_test @ A_inv
            yhat_iso[gi] = x_test @ b
            se_iso[gi] = np.sqrt(sig2 / n * np.sum(xAi * x_test, axis=1))

        # Lepski selection for each kappa
        for ki, kappa in enumerate(kappas):
            best_idx = lepski_select(yhat_iso, se_iso, kappa)
            mi = n_kappas + ki
            idx_range = np.arange(n_test)
            all_yhat[mi, ds] = yhat_iso[best_idx, idx_range]
            all_se[mi, ds] = se_iso[best_idx, idx_range]
            selected_lam_iso[ki, ds] = lam_grid[best_idx]

        # === FPCA (m=6) ===
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
        all_yhat[2 * n_kappas, ds] = w_new @ gf
        all_se[2 * n_kappas, ds] = np.sqrt(sig2_f * np.sum(wWi * w_new, axis=1))

    fname = f'alpha_sweep_{case_name}_n{n_val}_v10.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas,
             kappas=np.array(kappas),
             selected_lam_aniso=selected_lam_aniso,
             selected_lam_iso=selected_lam_iso,
             lam_grid=lam_grid)
    print(f'Saved {fname}')


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run(case_name, n_val, results_dir)
