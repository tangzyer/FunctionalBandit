"""Lepski with hybrid kappa + FPCA baseline + truncated/adaptive.

v13 — 6 methods:
  m=0: Aniso Lepski, hybrid kappa
  m=1: Iso Lepski, hybrid kappa
  m=2: FPCA (sqrt(n))
  m=3: Trunc.Aniso J=5*n^0.3
  m=4: Trunc.Aniso J=5*n^0.4
  m=5: Adapt.Trunc J=5*n^0.2 (10-fold CV, mu_n/(log n)^2)

Usage: python run_alpha_sweep_v13.py <case_name> <n_val> [C_aniso] [C_iso]
  case_name: aligned_r2_2, aligned_r2_3, haar_r2_2, shifted, etc.
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


def run(case_name, n_val, results_dir, C_aniso=0.05, C_iso=0.05):
    K = 200
    sigma = 0.5
    n_datasets = 1000
    n_test = 500
    n_grid = n_val // 2
    lam_grid = np.logspace(-3 * np.log10(n_val), -1 * np.log10(n_val), n_grid)

    # mu_n grid for adaptive method: n^{-4} to 1, L=100
    n_grid_adapt = 100
    mu_grid = np.logspace(-4 * np.log10(n_val), 0, n_grid_adapt)

    if case_name == 'aligned_r2_2':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
    elif case_name == 'shifted':
        cov_spec = figure3_top_specs(K, k0_values=[50])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
    elif case_name == 'aligned_r2_1p5':
        cov_spec = figure2_specs(K, r2_values=[1.5])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
    elif case_name == 'aligned_r2_3':
        cov_spec = figure2_specs(K, r2_values=[3.0])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
    elif case_name == 'aligned_r2_2_beta4':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=4)
        beta_power = 4
    elif case_name == 'aligned_r2_2_sparse':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        _bvec = np.zeros(K)
        _bvec[0], _bvec[2], _bvec[4] = 4.0, -2.0, 1.0
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_vec=_bvec)
        custom_beta = _bvec
    elif case_name == 'aligned_r2_3_beta4':
        cov_spec = figure2_specs(K, r2_values=[3.0])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=4)
        beta_power = 4
    elif case_name == 'aligned_r2_3_sparse':
        cov_spec = figure2_specs(K, r2_values=[3.0])[0]
        _bvec = np.zeros(K)
        _bvec[0], _bvec[2], _bvec[4] = 4.0, -2.0, 1.0
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_vec=_bvec)
        custom_beta = _bvec
    elif case_name == 'aligned_r2_3_sparse2':
        cov_spec = figure2_specs(K, r2_values=[3.0])[0]
        _bvec = np.zeros(K)
        _bvec[0], _bvec[1] = 4.0, -2.0
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_vec=_bvec)
        custom_beta = _bvec
    elif case_name == 'haar_r2_2':
        cov_spec = figure3_bottom_specs(K, r2_values=[2.0])[0]
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
    elif case_name == 'haar_r2_1p5':
        cov_spec = figure3_bottom_specs(K, r2_values=[1.5])[0]
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
    elif case_name == 'haar_r2_3':
        cov_spec = figure3_bottom_specs(K, r2_values=[3.0])[0]
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
    elif case_name == 'haar_r2_3_beta4':
        cov_spec = figure3_bottom_specs(K, r2_values=[3.0])[0]
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_power=4)
        beta_power = 4
    elif case_name == 'haar_r2_3_sparse2':
        cov_spec = figure3_bottom_specs(K, r2_values=[3.0])[0]
        _bvec = np.zeros(K)
        _bvec[0], _bvec[1] = 4.0, -2.0
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_vec=_bvec)
        custom_beta = _bvec
    elif case_name == 'shifted_beta4':
        cov_spec = figure3_top_specs(K, k0_values=[50])[0]
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=4)
        beta_power = 4
    elif case_name == 'shifted_sparse2':
        cov_spec = figure3_top_specs(K, k0_values=[50])[0]
        _bvec = np.zeros(K)
        _bvec[0], _bvec[1] = 4.0, -2.0
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_vec=_bvec)
        custom_beta = _bvec
    else:
        raise ValueError(f'Unknown case: {case_name}')

    if 'custom_beta' in dir() or 'custom_beta' in locals():
        b_true = custom_beta
    else:
        beta_power = locals().get('beta_power', 2)
        b_true = true_beta_coeffs(K, power=beta_power)
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

    # Lepski kappa: C * sqrt(log(L) + log(n)), separate for Aniso/Iso
    log_term = np.sqrt(np.log(n_grid) + np.log(n_val))
    kappa_aniso = C_aniso * log_term
    kappa_iso = C_iso * log_term
    print(f'  Lepski kappa for n={n_val}: aniso={kappa_aniso:.4f}, iso={kappa_iso:.4f} '
          f'(C_aniso={C_aniso}, C_iso={C_iso})')

    # Methods: 0=Aniso-theory, 1=Iso-theory, 2=FPCA,
    #          3=TruncAniso n^0.3, 4=TruncAniso n^0.4,
    #          5=AdaptTruncAniso n^0.2
    n_methods = 6
    all_yhat = np.zeros((n_methods, n_datasets, n_test))
    all_se = np.zeros((n_methods, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)

    selected_lam = np.zeros((2, n_datasets, n_test))  # Aniso + Iso

    # Pre-compute for vectorized Iso
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

        # Coefficients -> residuals -> sigma^2
        b_all_aniso = (U_T @ gamma_all.T).T * sqrt_mu[None, :]  # (n_grid, K)
        resid_aniso = Y[:, None] - Z @ b_all_aniso.T             # (n, n_grid)
        sig2_aniso = np.sum(resid_aniso ** 2, axis=0) / n        # (n_grid,)

        # Variance weights and SE: sandwich form d_k/(d_k + ν_k)²
        wv_all = np.where(den_all > 1e-15,
                          d_arr / den_all ** 2, 0.0)         # (n_grid, K)
        se_aniso = np.sqrt((sig2_aniso[:, None] / n) *
                           (wv_all @ c_new_sq.T))            # (n_grid, n_test)

        # Lepski with hybrid kappa (m=0)
        idx_range = np.arange(n_test)
        best_idx_at = lepski_select(yhat_aniso, se_aniso, kappa_aniso)
        all_yhat[0, ds] = yhat_aniso[best_idx_at, idx_range]
        all_se[0, ds] = se_aniso[best_idx_at, idx_range]
        selected_lam[0, ds] = lam_grid[best_idx_at]

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

        # Lepski with hybrid kappa (m=1)
        best_idx_it = lepski_select(yhat_iso, se_iso, kappa_iso)
        all_yhat[1, ds] = yhat_iso[best_idx_it, idx_range]
        all_se[1, ds] = se_iso[best_idx_it, idx_range]
        selected_lam[1, ds] = lam_grid[best_idx_it]

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

        # === Truncated Aniso (m=3: J=n^0.3, m=4: J=n^0.4) ===
        trunc_sig2 = {}  # cache sigma^2 for reuse by adaptive
        for mi_off, power in enumerate([0.3, 0.4]):
            J_tr = min(int(n_val ** power), K)
            gamma_tr = np.zeros(K)
            valid_tr = d_eig[:J_tr] > 1e-15
            gamma_tr[:J_tr] = np.where(valid_tr,
                                       c_est[:J_tr] / d_eig[:J_tr], 0.0)
            yhat_tr = c_new @ gamma_tr
            b_tr = (U_T @ gamma_tr) * sqrt_mu
            resid_tr = Y - Z @ b_tr
            sig2_tr = np.sum(resid_tr ** 2) / max(n - J_tr, 1)
            trunc_sig2[mi_off] = sig2_tr
            wv_tr = np.zeros(K)
            wv_tr[:J_tr] = np.where(valid_tr, 1.0 / d_eig[:J_tr], 0.0)
            se_tr = np.sqrt(sig2_tr / n * (c_new_sq @ wv_tr))
            all_yhat[3 + mi_off, ds] = yhat_tr
            all_se[3 + mi_off, ds] = se_tr

        # === Adaptive Truncated Aniso (m=5: J=n^0.2) ===
        # Section 7 of anisotropic_v2.pdf, Eq. (17).
        # 10-fold CV to select mu_n, then shrink by (log n)^2.
        J_adapt = min(int(n_val ** 0.2), K)
        n_folds = 10
        fold_ids = np.arange(n) % n_folds
        rng_fold = np.random.default_rng(12345 + ds)
        rng_fold.shuffle(fold_ids)

        cv_mse = np.zeros(n_grid_adapt)
        for fold in range(n_folds):
            val_mask = fold_ids == fold
            tr_mask = ~val_mask
            Z_tr_f, Y_tr_f = Z[tr_mask], Y[tr_mask]
            Z_val, Y_val = Z[val_mask], Y[val_mask]
            n_tr = Z_tr_f.shape[0]

            Z_tilde_tr = Z_tr_f * sqrt_mu
            T_tr = Z_tilde_tr.T @ Z_tilde_tr / n_tr
            eig_tr, U_tr = np.linalg.eigh(T_tr)
            ord_tr = np.argsort(eig_tr)[::-1]
            d_tr = np.maximum(eig_tr[ord_tr], 0)
            U_tr = U_tr[:, ord_tr]
            c_tr = U_tr.T @ (Z_tilde_tr.T @ Y_tr_f / n_tr)

            mx_val = Z_val * sqrt_mu
            c_val = mx_val @ U_tr
            c_val_sq = c_val ** 2

            s_tr_a = d_tr[:J_adapt]
            zp = c_val[:, :J_adapt]
            zp_sq = c_val_sq[:, :J_adapt]
            c_tr_J = c_tr[:J_adapt]
            abs_zp = np.maximum(np.abs(zp), 1e-15)
            b1_base_cv = s_tr_a[None, :] / abs_zp
            cond_cv = zp_sq >= 2.0 * s_tr_a[None, :]

            for gi, mu_val in enumerate(mu_grid):
                sqrt_mu_val = np.sqrt(mu_val)
                lam_b1 = b1_base_cv * sqrt_mu_val
                lam_b2 = np.minimum(mu_val,
                                    np.sqrt(s_tr_a * mu_val))[None, :]
                lam = np.where(cond_cv, lam_b1, lam_b2)
                den = s_tr_a[None, :] + lam
                gamma_cv = np.where(den > 1e-15,
                                    c_tr_J[None, :] / den, 0.0)
                yhat_val = np.sum(gamma_cv * zp, axis=1)
                cv_mse[gi] += np.sum((Y_val - yhat_val) ** 2)

        best_gi = np.argmin(cv_mse)
        mu_cv = mu_grid[best_gi] / np.log(n_val) ** 2

        # Compute estimate + SE on full data
        s_hat = d_eig[:J_adapt]
        z_proj = c_new[:, :J_adapt]
        z_proj_sq = c_new_sq[:, :J_adapt]
        c_est_J = c_est[:J_adapt]

        # Use sigma^2 from m=3 truncated (closest truncation level)
        gamma_tr_a = np.zeros(K)
        valid_tr_a = d_eig[:J_adapt] > 1e-15
        gamma_tr_a[:J_adapt] = np.where(valid_tr_a,
                                        c_est[:J_adapt] / d_eig[:J_adapt], 0.0)
        b_tr_a = (U_T @ gamma_tr_a) * sqrt_mu
        resid_tr_a = Y - Z @ b_tr_a
        sig2_adapt = np.sum(resid_tr_a ** 2) / max(n - J_adapt, 1)

        sqrt_mu_sel = np.sqrt(mu_cv)
        cond = z_proj_sq >= 2.0 * s_hat[None, :]
        abs_z = np.maximum(np.abs(z_proj), 1e-15)
        lam_b1 = s_hat[None, :] * sqrt_mu_sel / abs_z
        lam_b2 = np.minimum(mu_cv,
                            np.sqrt(s_hat * mu_cv))[None, :]
        lam = np.where(cond, lam_b1, lam_b2)
        den = s_hat[None, :] + lam
        gamma_a = np.where(den > 1e-15,
                           c_est_J[None, :] / den, 0.0)

        all_yhat[5, ds] = np.sum(gamma_a * z_proj, axis=1)

        vw = np.where(den > 1e-15,
                      s_hat[None, :] / den ** 2, 0.0)
        all_se[5, ds] = np.sqrt(
            sig2_adapt / n * np.sum(z_proj_sq * vw, axis=1))

        if ds % 100 == 0:
            print(f'    kappa_aniso={kappa_aniso:.4f}, kappa_iso={kappa_iso:.4f}',
                  flush=True)

    fname = f'alpha_sweep_{case_name}_n{n_val}_Ca{C_aniso}_Ci{C_iso}_v13.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas,
             kappa_aniso=kappa_aniso, kappa_iso=kappa_iso,
             selected_lam=selected_lam,
             lam_grid=lam_grid)
    print(f'Saved {fname}')
    print(f'  kappa_aniso={kappa_aniso:.4f}, kappa_iso={kappa_iso:.4f}')


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    C_a = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    C_i = float(sys.argv[4]) if len(sys.argv) > 4 else C_a
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run(case_name, n_val, results_dir, C_aniso=C_a, C_iso=C_i)
