"""Lepski with bootstrap kappa (Option A) and theoretical kappa (Option B).

Option A: Gaussian multiplier bootstrap to estimate the (1-gamma)-quantile of
  max_{j>i} max_x |G*(j,x) - G*(i,x)| / SE(i,x), then multiply by q=1.1.

Option B: kappa = 0.05 * 1.1 * sqrt(2*log(L) + 2*log(n))
  where L = lambda grid size, n = sample size.

Methods:
  m=0: Aniso Lepski, bootstrap kappa
  m=1: Aniso Lepski, theoretical kappa
  m=2: Iso Lepski, bootstrap kappa
  m=3: Iso Lepski, theoretical kappa
  m=4: FPCA

Usage: python run_alpha_sweep_v11.py <case_name> <n_val>
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

    Uses running bounds: O(n_grid * n_test).
    """
    n_grid, n_test = yhat_grid.shape
    best_idx = np.zeros(n_test, dtype=int)
    lower = yhat_grid[0] - kappa * se_grid[0]
    upper = yhat_grid[0] + kappa * se_grid[0]
    for j in range(1, n_grid):
        valid = (yhat_grid[j] >= lower) & (yhat_grid[j] <= upper)
        best_idx = np.where(valid, j, best_idx)
        lower = np.maximum(lower, yhat_grid[j] - kappa * se_grid[j])
        upper = np.minimum(upper, yhat_grid[j] + kappa * se_grid[j])
    return best_idx


def lepski_select_per_test(yhat_grid, se_grid, kappas_per_test):
    """Lepski selection with a different kappa per test point.

    kappas_per_test: (n_test,) array of kappa values.
    """
    n_grid, n_test = yhat_grid.shape
    best_idx = np.zeros(n_test, dtype=int)
    lower = yhat_grid[0] - kappas_per_test * se_grid[0]
    upper = yhat_grid[0] + kappas_per_test * se_grid[0]
    for j in range(1, n_grid):
        valid = (yhat_grid[j] >= lower) & (yhat_grid[j] <= upper)
        best_idx = np.where(valid, j, best_idx)
        lower = np.maximum(lower, yhat_grid[j] - kappas_per_test * se_grid[j])
        upper = np.minimum(upper, yhat_grid[j] + kappas_per_test * se_grid[j])
    return best_idx


def bootstrap_kappa(G_proj, resid, se_grid, inv_factors, pred_proj, n,
                    n_grid_sub=50, B=200, gamma=0.05, q=1.1, rng=None):
    """Compute Lepski kappa via Gaussian multiplier bootstrap.

    Uses suffix-max trick for O(B * n_grid_sub * n_test) computation.

    Args:
        G_proj: (K, n) projection matrix (W_aniso or Q_proj_iso)
        resid: (n, n_grid) residuals at each lambda
        se_grid: (n_grid, n_test) standard errors
        inv_factors: (n_grid, K) — 1/(d_k+nu_k) for aniso or 1/(eigval_k+l) for iso
        pred_proj: (K, n_test) — c_new.T for aniso or v_test for iso
        n: sample size
        n_grid_sub: number of subsampled lambda points
        B: number of bootstrap replications
        gamma: probability level for quantile (default 0.05)
        q: inflation constant (default 1.1)
        rng: numpy random generator

    Returns:
        kappa: scalar bootstrap kappa
    """
    n_grid = resid.shape[1]
    n_test = pred_proj.shape[1]
    K = G_proj.shape[0]

    # Subsample the lambda grid
    sub_idx = np.round(np.linspace(0, n_grid - 1, n_grid_sub)).astype(int)
    se_sub = se_grid[sub_idx]  # (n_grid_sub, n_test)

    # Generate bootstrap weights
    e_all = rng.normal(size=(B, n))  # (B, n)

    # Compute bootstrap predictions on subsampled grid
    G_boot = np.zeros((B, n_grid_sub, n_test))
    for li in range(n_grid_sub):
        gi = sub_idx[li]
        e_eps = e_all * resid[:, gi]           # (B, n)
        r_l = e_eps @ G_proj.T                 # (B, K)
        coeff_l = r_l * inv_factors[gi]        # (B, K)
        G_boot[:, li, :] = coeff_l @ pred_proj / n  # (B, n_test)

    # Suffix max/min for efficient violation computation
    # suffix_max[i] = max(G[i+1], ..., G[L-1])
    suffix_max = np.full_like(G_boot, -np.inf)
    suffix_min = np.full_like(G_boot, np.inf)
    for j in range(n_grid_sub - 2, -1, -1):
        suffix_max[:, j, :] = np.maximum(G_boot[:, j + 1, :], suffix_max[:, j + 1, :])
        suffix_min[:, j, :] = np.minimum(G_boot[:, j + 1, :], suffix_min[:, j + 1, :])

    # max_{j>i} |G(j) - G(i)| / SE(i) = max_i max(suffix_max[i]-G[i], G[i]-suffix_min[i]) / SE(i)
    max_pos = suffix_max - G_boot     # (B, n_grid_sub, n_test)
    max_neg = G_boot - suffix_min
    violation = np.maximum(max_pos, max_neg) / np.maximum(se_sub[None, :, :], 1e-15)
    violation[:, -1, :] = 0  # no j > last index

    # Take max over grid points and test points for each bootstrap rep
    max_stat = violation.max(axis=(1, 2))  # (B,)

    kappa = q * np.percentile(max_stat, 100 * (1 - gamma))
    return kappa


def run(case_name, n_val, results_dir):
    K = 50
    sigma = 0.5
    n_datasets = 1000
    n_test = 200
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

    # Theoretical kappa: 0.05 * q * sqrt(2*log(L) + 2*log(n))
    q_lepski = 1.1
    kappa_theory = 0.05 * q_lepski * np.sqrt(2 * np.log(n_grid) + 2 * np.log(n_val))
    print(f'  Theoretical kappa for n={n_val}, L={n_grid}: {kappa_theory:.4f}')

    # Methods: 0=Aniso-boot, 1=Aniso-theory, 2=Iso-boot, 3=Iso-theory, 4=FPCA
    n_methods = 5
    all_yhat = np.zeros((n_methods, n_datasets, n_test))
    all_se = np.zeros((n_methods, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)

    selected_lam = np.zeros((n_methods - 1, n_datasets, n_test))
    kappa_boot_aniso = np.zeros(n_datasets)
    kappa_boot_iso = np.zeros(n_datasets)

    # Pre-compute for vectorized Iso
    T_iso = 1.0 / np.sqrt(D)  # D^{-1/2}, (K,)

    # Bootstrap parameters
    B_boot = 200
    n_grid_sub = 50
    gamma_boot = 0.05

    for ds in range(n_datasets):
        if ds % 100 == 0:
            print(f'  {case_name} n={n_val}: dataset {ds}/{n_datasets}',
                  flush=True)

        rng = np.random.default_rng(42 + ds)
        rng_boot = np.random.default_rng(10000 + ds)
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

        # Coefficients -> residuals -> sigma^2
        b_all_aniso = (U_T @ gamma_all.T).T * sqrt_mu[None, :]  # (n_grid, K)
        resid_aniso = Y[:, None] - Z @ b_all_aniso.T             # (n, n_grid)
        sig2_aniso = np.sum(resid_aniso ** 2, axis=0) / n        # (n_grid,)

        # Variance weights and SE
        wv_all = np.where(den_all > 1e-15,
                          d_arr / den_all ** 2, 0.0)         # (n_grid, K)
        se_aniso = np.sqrt((sig2_aniso[:, None] / n) *
                           (wv_all @ c_new_sq.T))            # (n_grid, n_test)

        # Aniso inv_factors for bootstrap
        inv_factors_aniso = np.where(den_all > 1e-15, 1.0 / den_all, 0.0)  # (n_grid, K)

        # Aniso projection matrices for bootstrap
        W_aniso = U_T.T @ Z_tilde.T  # (K, n)

        # Bootstrap kappa for Aniso
        kappa_ba = bootstrap_kappa(
            W_aniso, resid_aniso, se_aniso, inv_factors_aniso,
            c_new.T,  # (K, n_test)
            n, n_grid_sub=n_grid_sub, B=B_boot, gamma=gamma_boot,
            q=q_lepski, rng=rng_boot)
        kappa_boot_aniso[ds] = kappa_ba

        # Lepski with bootstrap kappa (m=0)
        best_idx_ab = lepski_select(yhat_aniso, se_aniso, kappa_ba)
        idx_range = np.arange(n_test)
        all_yhat[0, ds] = yhat_aniso[best_idx_ab, idx_range]
        all_se[0, ds] = se_aniso[best_idx_ab, idx_range]
        selected_lam[0, ds] = lam_grid[best_idx_ab]

        # Lepski with theoretical kappa (m=1)
        best_idx_at = lepski_select(yhat_aniso, se_aniso, kappa_theory)
        all_yhat[1, ds] = yhat_aniso[best_idx_at, idx_range]
        all_se[1, ds] = se_aniso[best_idx_at, idx_range]
        selected_lam[1, ds] = lam_grid[best_idx_at]

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

        # Iso projection matrix for bootstrap
        Q_proj_iso = Q.T @ (T_iso[:, None] * Z.T)  # (K, n)

        # Bootstrap kappa for Iso
        rng_boot_iso = np.random.default_rng(20000 + ds)
        kappa_bi = bootstrap_kappa(
            Q_proj_iso, resid_iso, se_iso, inv_diag,
            v_test,  # (K, n_test)
            n, n_grid_sub=n_grid_sub, B=B_boot, gamma=gamma_boot,
            q=q_lepski, rng=rng_boot_iso)
        kappa_boot_iso[ds] = kappa_bi

        # Lepski with bootstrap kappa (m=2)
        best_idx_ib = lepski_select(yhat_iso, se_iso, kappa_bi)
        all_yhat[2, ds] = yhat_iso[best_idx_ib, idx_range]
        all_se[2, ds] = se_iso[best_idx_ib, idx_range]
        selected_lam[2, ds] = lam_grid[best_idx_ib]

        # Lepski with theoretical kappa (m=3)
        best_idx_it = lepski_select(yhat_iso, se_iso, kappa_theory)
        all_yhat[3, ds] = yhat_iso[best_idx_it, idx_range]
        all_se[3, ds] = se_iso[best_idx_it, idx_range]
        selected_lam[3, ds] = lam_grid[best_idx_it]

        # === FPCA (m=4) ===
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
        all_yhat[4, ds] = w_new @ gf
        all_se[4, ds] = np.sqrt(sig2_f * np.sum(wWi * w_new, axis=1))

        if ds % 100 == 0:
            print(f'    kappa_boot: aniso={kappa_ba:.4f}, iso={kappa_bi:.4f}, '
                  f'theory={kappa_theory:.4f}', flush=True)

    fname = f'alpha_sweep_{case_name}_n{n_val}_v11.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas,
             kappa_theory=kappa_theory,
             kappa_boot_aniso=kappa_boot_aniso,
             kappa_boot_iso=kappa_boot_iso,
             selected_lam=selected_lam,
             lam_grid=lam_grid)
    print(f'Saved {fname}')
    print(f'  Bootstrap kappa (aniso): mean={kappa_boot_aniso.mean():.4f}, '
          f'std={kappa_boot_aniso.std():.4f}')
    print(f'  Bootstrap kappa (iso): mean={kappa_boot_iso.mean():.4f}, '
          f'std={kappa_boot_iso.std():.4f}')
    print(f'  Theoretical kappa: {kappa_theory:.4f}')


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run(case_name, n_val, results_dir)
