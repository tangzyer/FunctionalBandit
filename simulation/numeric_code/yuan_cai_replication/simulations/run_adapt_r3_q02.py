"""Focused experiment: Adaptive Truncated Aniso with RKHS r=3, q=0.2.

Compares:
  m=0: Trunc.Aniso (non-adaptive, J=n^0.2)
  m=1: Adapt.Trunc (adaptive, J=n^0.2, 10-fold CV for mu_n)

Uses sparse beta = (4, 0, -2, 0, 1, 0, ...) to satisfy (C1).
RKHS kernel eigenvalues: mu_k = 2/(k*pi)^{2r} with r=3.
Covariance eigenvalues: theta_k = k^{-2*r2} with r2=2.

Usage: python run_adapt_r3_q02.py <n_val>
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.covariance import figure2_specs
from src.data_generation import generate_data_cosine_basis


def run(n_val, results_dir):
    K = 50
    sigma = 0.5
    n_datasets = 1000
    n_test = 500
    rkhs_r = 3        # RKHS smoothness
    q = 0.2            # truncation exponent
    r2 = 2.0           # covariance eigenvalue decay

    # mu_n grid for adaptive method: n^{-2} to 1, L=100
    n_grid_adapt = 100
    mu_grid = np.logspace(-4 * np.log10(n_val), 0, n_grid_adapt)

    # Sparse beta (C1 satisfied)
    b_true = np.zeros(K)
    b_true[0], b_true[2], b_true[4] = 4.0, -2.0, 1.0

    cov_spec = figure2_specs(K, r2_values=[r2])[0]
    gen_func = lambda n, rng: generate_data_cosine_basis(
        n, cov_spec, K, sigma=sigma, rng=rng, beta_vec=b_true)

    # RKHS kernel eigenvalues for r=3: mu_k = 2/(k*pi)^{2r}
    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** (2 * rkhs_r)
    sqrt_mu = np.sqrt(mu_k)

    J_tr = min(int(5 * n_val ** q), K)
    print(f'  r={rkhs_r}, q={q}, J=n^{q}={J_tr}, K={K}, n={n_val}')

    # BM test functions (same seed as main script)
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

    # Methods: 0=TruncAniso (non-adaptive), 1=AdaptTruncAniso (adaptive)
    n_methods = 2
    all_yhat = np.zeros((n_methods, n_datasets, n_test))
    all_se = np.zeros((n_methods, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)

    for ds in range(n_datasets):
        if ds % 100 == 0:
            print(f'  n={n_val}: dataset {ds}/{n_datasets}', flush=True)

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
        c_new = mx_test @ U_T          # (n_test, K)
        c_new_sq = c_new ** 2           # (n_test, K)

        # === m=0: Truncated Aniso (non-adaptive, J=n^q) ===
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
        all_yhat[0, ds] = yhat_tr
        all_se[0, ds] = se_tr

        # === m=1: Adaptive Truncated Aniso (J=n^q, 10-fold CV for mu_n) ===
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

            s_tr = d_tr[:J_tr]
            zp = c_val[:, :J_tr]
            zp_sq = c_val_sq[:, :J_tr]
            c_tr_J = c_tr[:J_tr]
            abs_zp = np.maximum(np.abs(zp), 1e-15)
            b1_base_cv = s_tr[None, :] / abs_zp
            cond_cv = zp_sq >= 2.0 * s_tr[None, :]

            for gi, mu_val in enumerate(mu_grid):
                sqrt_mu_val = np.sqrt(mu_val)
                lam_b1 = b1_base_cv * sqrt_mu_val
                lam_b2 = np.minimum(mu_val,
                                    np.sqrt(s_tr * mu_val))[None, :]
                lam = np.where(cond_cv, lam_b1, lam_b2)
                den = s_tr[None, :] + lam
                gamma_cv = np.where(den > 1e-15,
                                    c_tr_J[None, :] / den, 0.0)
                yhat_val = np.sum(gamma_cv * zp, axis=1)
                cv_mse[gi] += np.sum((Y_val - yhat_val) ** 2)

        best_gi = np.argmin(cv_mse)
        mu_cv = mu_grid[best_gi] / np.log(n_val) ** 2

        # Compute estimate + SE on full data
        s_hat = d_eig[:J_tr]
        z_proj = c_new[:, :J_tr]
        z_proj_sq = c_new_sq[:, :J_tr]
        c_est_J = c_est[:J_tr]

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

        all_yhat[1, ds] = np.sum(gamma_a * z_proj, axis=1)

        vw = np.where(den > 1e-15,
                      s_hat[None, :] / den ** 2, 0.0)
        all_se[1, ds] = np.sqrt(
            sig2_tr / n * np.sum(z_proj_sq * vw, axis=1))

    fname = f'adapt_r3_q02_sparse_n{n_val}.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas,
             rkhs_r=rkhs_r, q=q, J_tr=J_tr)
    print(f'Saved {fname}')


if __name__ == '__main__':
    n_val = int(sys.argv[1])
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run(n_val, results_dir)
