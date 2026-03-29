"""Lepski with Gaussian multiplier bootstrap kappa + FPCA baseline.

v14: Bootstrap kappa following Chernozhukov et al. (2014) Algorithm 2.
  - STUDENTIZED bootstrap: each pair (lambda, lambda') is normalized by
    sigma_diff(x, lambda, lambda') = SD of pairwise difference.
  - Lepski criterion: sqrt(n)|yhat(l)-yhat(l')|/sigma_diff <= q * c_tilde
  - c_tilde = (1-gamma_n) quantile of sup of STUDENTIZED bootstrap process.
  - gamma_n = 0.1 * n^{-0.5}, q = 1.1.

Methods:
  m=0: Aniso Lepski, bootstrap kappa
  m=1: Iso Lepski, bootstrap kappa
  m=2: FPCA

Usage: python run_alpha_sweep_v14.py <case_name> <n_val>
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.covariance import figure2_specs, figure3_top_specs, figure3_bottom_specs
from src.data_generation import (generate_data_cosine_basis,
                                  generate_data_haar_basis, true_beta_coeffs)
from src.estimators import rkhs_penalty_matrix


def studentized_bootstrap_lepski(W_proj, Y, resid_pilot, inv_factors_full,
                                 c_new, yhat_full, se_full, lam_grid, n,
                                 n_grid_sub=50, B=200, gamma=0.05, q=1.1,
                                 rng=None):
    """Studentized bootstrap Lepski following Chernozhukov Algorithm 2.

    Adapted for regression: uses pilot residuals (not raw Y) so that
    sigma_diff captures noise-driven variability, not signal variance.

    The bootstrap process for pair (a, b):
        G_tilde(x, a, b) = (1/(sqrt(n) * sigma_diff)) * sum_i xi_i * eps_i * d_i(x)
    where d_i(x) = psi_i(x,a) - psi_i(x,b) is the influence difference.

    Lepski criterion:
        sqrt(n) |yhat(a,x) - yhat(b,x)| / sigma_diff(a,b,x) <= q * c_tilde(x)

    Returns:
        best_idx: (n_test,) selected full-grid index per test point
        c_tilde: (n_test,) bootstrap critical value per test point
    """
    n_grid = len(lam_grid)
    n_test = c_new.shape[0]
    K = c_new.shape[1]

    # Subsample grid
    sub_idx = np.round(np.linspace(0, n_grid - 1, n_grid_sub)).astype(int)
    n_sub = len(sub_idx)
    inv_sub = inv_factors_full[sub_idx]  # (n_sub, K)
    yhat_sub = yhat_full[sub_idx]        # (n_sub, n_test)

    # Precompute: R_bk = sum_i xi_i * eps_pilot_i * W_ki
    xi_all = rng.normal(size=(B, n))               # (B, n)
    xi_eps = xi_all * resid_pilot[None, :]         # (B, n)
    R = xi_eps @ W_proj.T                          # (B, K)

    # For sigma_diff: eps_pilot^2 weighted
    eps2 = resid_pilot ** 2                        # (n,)
    eps2_W = eps2[:, None] * (W_proj.T ** 2)       # (n, K): eps_i^2 * W_ki^2
    # Won't use this directly; compute d then sigma_diff per pair

    # Precompute psi on subgrid: psi_l_i(x) = sum_k c_new_k(x) * W_ki * inv_k(l)
    # Using factored form: for pair (a,b), d = psi_a - psi_b via delta_k
    eps_W = resid_pilot[:, None] * W_proj.T        # (n, K): eps_i * W_ki

    # Bootstrap: iterate over pairs
    max_stat = np.full((B, n_test), -np.inf)

    # Studentized statistic for Lepski: max over b > a
    stud_max = np.zeros((n_sub, n_test))

    for a in range(n_sub - 1):
        inv_a = inv_sub[a]
        for b in range(a + 1, n_sub):
            inv_b = inv_sub[b]
            delta_k = inv_a - inv_b  # (K,)

            # d_i(x) = sum_k c_new_k(x) * W_ki * delta_k
            # Phi_i(x) = eps_pilot_i * d_i(x)
            # = sum_k c_new_k(x) * (eps_i * W_ki) * delta_k
            Phi = (eps_W * delta_k[None, :]) @ c_new.T  # (n, n_test)

            # sigma_diff(x) = sqrt((1/n) sum_i Phi_i^2)
            sigma_diff = np.sqrt(np.maximum(
                (Phi ** 2).mean(axis=0), 1e-30))  # (n_test,)

            # Bootstrap: G_tilde = (1/(sqrt(n)*sigma_diff)) * (R * delta) @ c_new.T
            G_raw = (R * delta_k[None, :]) @ c_new.T  # (B, n_test)
            G_tilde = G_raw / (np.sqrt(n) * sigma_diff[None, :])

            max_stat = np.maximum(max_stat, np.abs(G_tilde))

            # Studentized statistic for Lepski
            delta_yhat = yhat_sub[a] - yhat_sub[b]
            stud_val = np.sqrt(n) * np.abs(delta_yhat) / sigma_diff
            stud_max[a] = np.maximum(stud_max[a], stud_val)

    # c_tilde(x) = (1-gamma) quantile of max_stat
    c_tilde = np.percentile(max_stat, 100 * (1 - gamma), axis=0)

    # Lepski: find smallest a s.t. for all b > a, stud_max[a,x] <= q*c_tilde(x)
    threshold = q * c_tilde
    best_sub_idx = np.full(n_test, n_sub - 1, dtype=int)
    for a in range(n_sub):
        passes = stud_max[a] <= threshold
        first_pass = passes & (best_sub_idx == n_sub - 1)
        best_sub_idx = np.where(first_pass, a, best_sub_idx)

    best_full_idx = sub_idx[best_sub_idx]
    return best_full_idx, c_tilde


def run(case_name, n_val, results_dir):
    K = 50
    sigma = 0.5
    n_datasets = 1000
    n_test = 200
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

    # Bootstrap parameters
    q_lepski = 1.1
    gamma_n = 0.1 * n_val ** (-0.5)
    B_boot = 200
    n_grid_sub = 50
    print(f'  Bootstrap params: q={q_lepski}, gamma_n={gamma_n:.4f}, '
          f'B={B_boot}, n_grid_sub={n_grid_sub}')

    # Methods: 0=Aniso-boot, 1=Iso-boot, 2=FPCA
    n_methods = 3
    all_yhat = np.zeros((n_methods, n_datasets, n_test))
    all_se = np.zeros((n_methods, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)

    selected_lam = np.zeros((2, n_datasets, n_test))  # Aniso + Iso
    ctilde_aniso = np.zeros((n_datasets, n_test))
    ctilde_iso = np.zeros((n_datasets, n_test))

    # Pre-compute for vectorized Iso
    T_iso = 1.0 / np.sqrt(D)  # D^{-1/2}, (K,)

    for ds in range(n_datasets):
        if ds % 100 == 0:
            print(f'  {case_name} n={n_val}: dataset {ds}/{n_datasets}',
                  flush=True)

        rng = np.random.default_rng(42 + ds)
        rng_boot_a = np.random.default_rng(10000 + ds)
        rng_boot_i = np.random.default_rng(20000 + ds)
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

        yhat_aniso = gamma_all @ c_new.T  # (n_grid, n_test)

        b_all_aniso = (U_T @ gamma_all.T).T * sqrt_mu[None, :]  # (n_grid, K)
        resid_aniso = Y[:, None] - Z @ b_all_aniso.T             # (n, n_grid)
        sig2_aniso = np.sum(resid_aniso ** 2, axis=0) / n        # (n_grid,)

        wv_all = np.where(den_all > 1e-15,
                          d_arr / den_all ** 2, 0.0)         # (n_grid, K)
        se_aniso = np.sqrt((sig2_aniso[:, None] / n) *
                           (wv_all @ c_new_sq.T))            # (n_grid, n_test)

        # Studentized bootstrap Lepski for Aniso
        W_aniso = U_T.T @ Z_tilde.T  # (K, n)
        inv_factors_aniso = np.where(den_all > 1e-15,
                                     1.0 / den_all, 0.0)  # (n_grid, K)

        # Pilot residuals at median subgrid lambda
        sub_idx_temp = np.round(np.linspace(0, n_grid - 1, n_grid_sub)).astype(int)
        pilot_grid_idx = sub_idx_temp[n_grid_sub // 2]
        resid_pilot_aniso = resid_aniso[:, pilot_grid_idx]  # (n,)

        best_idx_ab, ct_a = studentized_bootstrap_lepski(
            W_aniso, Y, resid_pilot_aniso, inv_factors_aniso, c_new,
            yhat_aniso, se_aniso, lam_grid, n,
            n_grid_sub=n_grid_sub, B=B_boot, gamma=gamma_n,
            q=q_lepski, rng=rng_boot_a)
        ctilde_aniso[ds] = ct_a

        idx_range = np.arange(n_test)
        all_yhat[0, ds] = yhat_aniso[best_idx_ab, idx_range]
        all_se[0, ds] = se_aniso[best_idx_ab, idx_range]
        selected_lam[0, ds] = lam_grid[best_idx_ab]

        # === Iso: vectorized via eigendecomposition ===
        B_mat = ZtZ * np.outer(T_iso, T_iso)
        eigvals_B, Q = np.linalg.eigh(B_mat)
        order_B = np.argsort(eigvals_B)[::-1]
        eigvals_B = np.maximum(eigvals_B[order_B], 0)
        Q = Q[:, order_B]

        c_iso = Q.T @ (T_iso * ZtY)                  # (K,)
        v_test = Q.T @ (T_iso[:, None] * x_test.T)   # (K, n_test)
        v_test_sq = v_test ** 2                        # (K, n_test)

        inv_diag = 1.0 / (eigvals_B[None, :] + lam_grid[:, None])  # (n_grid, K)

        coeff_all = c_iso[None, :] * inv_diag           # (n_grid, K)
        b_all_iso = (coeff_all @ Q.T) * T_iso[None, :]  # (n_grid, K)

        yhat_iso = b_all_iso @ x_test.T  # (n_grid, n_test)

        resid_iso = Y[:, None] - Z @ b_all_iso.T  # (n, n_grid)
        sig2_iso = np.sum(resid_iso ** 2, axis=0) / n  # (n_grid,)

        quad_form = inv_diag @ v_test_sq  # (n_grid, n_test)
        se_iso = np.sqrt(sig2_iso[:, None] / n * quad_form)  # (n_grid, n_test)

        # Studentized bootstrap Lepski for Iso
        Q_proj_iso = Q.T @ (T_iso[:, None] * Z.T)  # (K, n)

        # Pilot residuals at median subgrid lambda
        resid_pilot_iso = resid_iso[:, pilot_grid_idx]  # (n,)

        # For Iso: c_new analog is v_test.T (n_test, K)
        best_idx_ib, ct_i = studentized_bootstrap_lepski(
            Q_proj_iso, Y, resid_pilot_iso, inv_diag, v_test.T,
            yhat_iso, se_iso, lam_grid, n,
            n_grid_sub=n_grid_sub, B=B_boot, gamma=gamma_n,
            q=q_lepski, rng=rng_boot_i)
        ctilde_iso[ds] = ct_i

        all_yhat[1, ds] = yhat_iso[best_idx_ib, idx_range]
        all_se[1, ds] = se_iso[best_idx_ib, idx_range]
        selected_lam[1, ds] = lam_grid[best_idx_ib]

        # === FPCA (m=2) ===
        S = Z.T @ Z / n
        eigvals_S, eigvecs_S = np.linalg.eigh(S)
        order_S = np.argsort(eigvals_S)[::-1]
        eigvecs_S = eigvecs_S[:, order_S]
        J = int(np.sqrt(n_val))
        W_fpca = Z @ eigvecs_S[:, :J]
        WtW = W_fpca.T @ W_fpca
        Wi = np.linalg.inv(WtW)
        gf = Wi @ (W_fpca.T @ Y)
        rf = Y - W_fpca @ gf
        sig2_f = np.sum(rf ** 2) / (n_val - J)
        w_new = x_test @ eigvecs_S[:, :J]
        wWi = w_new @ Wi
        all_yhat[2, ds] = w_new @ gf
        all_se[2, ds] = np.sqrt(sig2_f * np.sum(wWi * w_new, axis=1))

        if ds % 100 == 0:
            print(f'    c_tilde: aniso={ct_a.mean():.4f}, '
                  f'iso={ct_i.mean():.4f}', flush=True)

    fname = f'alpha_sweep_{case_name}_n{n_val}_v14.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas,
             kappa_boot_aniso=ctilde_aniso,
             kappa_boot_iso=ctilde_iso,
             selected_lam=selected_lam,
             lam_grid=lam_grid)
    print(f'Saved {fname}')
    print(f'  c_tilde (aniso): mean={ctilde_aniso.mean():.4f}, '
          f'std={ctilde_aniso.mean(axis=1).std():.4f}')
    print(f'  c_tilde (iso): mean={ctilde_iso.mean():.4f}, '
          f'std={ctilde_iso.mean(axis=1).std():.4f}')


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run(case_name, n_val, results_dir)
