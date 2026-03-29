"""Lepski-type adaptive lambda selection with dense lambda grid (2000 pts).

Correct Lepski criterion: for each test point, find the largest lambda_j
such that for ALL i < j:
  |yhat(lambda_j) - yhat(lambda_i)| <= kappa * SE(lambda_i)

SE is evaluated at the less regularized point (lambda_i), which has
higher variance and serves as the resolution for detecting bias.

Methods:
  m=0,1,2: Aniso Lepski with kappa=0.001, 0.003, 0.01
  m=3,4,5: Iso Lepski with kappa=0.001, 0.003, 0.01
  m=6: FPCA

Usage: python run_alpha_sweep_v10.py <case_name> <n_val>
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
    """Select lambda index per test point using correct Lepski criterion.

    For each test point, find the largest lambda_j (highest grid index)
    such that for ALL i < j:
        |yhat(lambda_j) - yhat(lambda_i)| <= kappa * SE(lambda_i)

    Equivalent to: yhat(lambda_j) in [lower, upper] where
        lower = max_{i<j} (yhat_i - kappa * SE_i)
        upper = min_{i<j} (yhat_i + kappa * SE_i)

    This runs in O(n_grid * n_test) by tracking running bounds.

    Args:
        yhat_grid: (n_grid, n_test) predictions at each lambda
        se_grid: (n_grid, n_test) standard errors at each lambda
        kappa: threshold constant

    Returns:
        best_idx: (n_test,) selected lambda index per test point
    """
    n_grid, n_test = yhat_grid.shape
    best_idx = np.zeros(n_test, dtype=int)

    # Running bounds from i=0
    lower = yhat_grid[0] - kappa * se_grid[0]  # (n_test,)
    upper = yhat_grid[0] + kappa * se_grid[0]  # (n_test,)

    for j in range(1, n_grid):
        # Check: yhat[j] within bounds from all i=0,...,j-1
        valid = (yhat_grid[j] >= lower) & (yhat_grid[j] <= upper)
        best_idx = np.where(valid, j, best_idx)

        # Update bounds to include i=j
        lower = np.maximum(lower, yhat_grid[j] - kappa * se_grid[j])
        upper = np.minimum(upper, yhat_grid[j] + kappa * se_grid[j])

    return best_idx


def run(case_name, n_val, results_dir):
    K = 50
    sigma = 0.5
    n_datasets = 1000
    n_test = 1000
    n_grid = 2000
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

    kappas = [0.03, 0.1, 0.3]
    n_kappas = len(kappas)
    n_methods = 2 * n_kappas + 1
    all_yhat = np.zeros((n_methods, n_datasets, n_test))
    all_se = np.zeros((n_methods, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)

    selected_lam_aniso = np.zeros((n_kappas, n_datasets, n_test))
    selected_lam_iso = np.zeros((n_kappas, n_datasets, n_test))

    # Pre-compute for vectorized Iso: eigendecompose D^{-1/2} ZtZ D^{-1/2}
    T_iso = 1.0 / np.sqrt(D)  # D^{-1/2}, (K,)

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
        b_all_aniso = (U_T @ gamma_all.T).T * sqrt_mu[None, :]  # (n_grid, K)
        resid_all = Y[:, None] - Z @ b_all_aniso.T               # (n, n_grid)
        sig2_all = np.sum(resid_all ** 2, axis=0) / n             # (n_grid,)

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

        # === Iso: vectorized via eigendecomposition of D^{-1/2} ZtZ D^{-1/2} ===
        B = ZtZ * np.outer(T_iso, T_iso)  # (K, K)
        eigvals_B, Q = np.linalg.eigh(B)
        order_B = np.argsort(eigvals_B)[::-1]
        eigvals_B = np.maximum(eigvals_B[order_B], 0)
        Q = Q[:, order_B]

        # Pre-compute projections
        c_iso = Q.T @ (T_iso * ZtY)                  # (K,)
        v_test = Q.T @ (T_iso[:, None] * x_test.T)   # (K, n_test)
        v_test_sq = v_test ** 2                        # (K, n_test)

        # For all lambdas: 1/(eigval + lam)
        inv_diag = 1.0 / (eigvals_B[None, :] + lam_grid[:, None])  # (n_grid, K)

        # b(lam) = T_iso * Q @ (c_iso / (eigvals_B + lam))
        coeff_all = c_iso[None, :] * inv_diag           # (n_grid, K)
        b_all_iso = (coeff_all @ Q.T) * T_iso[None, :]  # (n_grid, K)

        # Predictions
        yhat_iso = b_all_iso @ x_test.T  # (n_grid, n_test)

        # Residual variance
        resid_iso = Y[:, None] - Z @ b_all_iso.T  # (n, n_grid)
        sig2_iso = np.sum(resid_iso ** 2, axis=0) / n  # (n_grid,)

        # SE: sqrt(sig2/n * x^T A^{-1} x)
        # x^T A^{-1} x = sum_j v_j^2 / (eigval_j + lam)
        quad_form = inv_diag @ v_test_sq  # (n_grid, n_test)
        se_iso = np.sqrt(sig2_iso[:, None] / n * quad_form)  # (n_grid, n_test)

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
