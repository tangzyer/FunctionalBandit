"""Run only m=5&6 (Adaptive Truncated Aniso) with 10-fold CV for mu_n.

Loads existing .npz for m=0-4 results, recomputes m=5&6, saves back.

Usage: python run_adapt_only_v13.py <case_name> <n_val> [C_a] [C_i]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.covariance import figure2_specs, figure3_top_specs, figure3_bottom_specs
from src.data_generation import (generate_data_cosine_basis,
                                  generate_data_haar_basis, true_beta_coeffs)
from src.estimators import rkhs_penalty_matrix


def run_adapt_only(case_name, n_val, results_dir, C_aniso=0.05, C_iso=0.05):
    K = 50
    sigma = 0.5
    n_datasets = 1000
    n_test = 500

    # mu_n grid for adaptive method: n^{-2} to 1, L=100
    n_grid_adapt = 100
    mu_grid = np.logspace(-2 * np.log10(n_val), 0, n_grid_adapt)

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
    elif case_name == 'aligned_r2_2_sparse':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        _bvec = np.zeros(K)
        _bvec[0], _bvec[2], _bvec[4] = 4.0, -2.0, 1.0
        gen_func = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_vec=_bvec)
    elif case_name == 'haar_r2_2':
        cov_spec = figure3_bottom_specs(K, r2_values=[2.0])[0]
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
    elif case_name == 'haar_r2_1p5':
        cov_spec = figure3_bottom_specs(K, r2_values=[1.5])[0]
        gen_func = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
    else:
        raise ValueError(f'Unknown case: {case_name}')

    # Load existing results for m=0-4
    fname = f'alpha_sweep_{case_name}_n{n_val}_Ca{C_aniso}_Ci{C_iso}_v13.npz'
    fpath = os.path.join(results_dir, fname)
    existing = np.load(fpath, allow_pickle=True)
    all_yhat = existing['all_yhat'].copy()   # (7, n_datasets, n_test)
    all_se = existing['all_se'].copy()
    true_vals = existing['true_vals']
    alphas = existing['alphas']
    kappa_aniso = float(existing['kappa_aniso'])
    kappa_iso = float(existing['kappa_iso'])
    selected_lam = existing['selected_lam']
    lam_grid = existing['lam_grid']

    b_true = true_beta_coeffs(K)
    D = rkhs_penalty_matrix(K)

    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** 4
    sqrt_mu = np.sqrt(mu_k)

    # BM test functions (same seed as main script)
    rng_test = np.random.default_rng(999)
    n_grid_pts = 200
    t_grid = np.linspace(0, 1, n_grid_pts + 1)[1:]
    dt = 1.0 / n_grid_pts
    dW = rng_test.normal(size=(n_test, n_grid_pts)) * np.sqrt(dt)
    bm_paths = np.cumsum(dW, axis=1)
    phi_grid = np.sqrt(2) * np.cos(np.outer(ks, np.pi * t_grid))
    x_test = bm_paths @ phi_grid.T * dt
    mx_test = x_test * sqrt_mu

    for ds in range(n_datasets):
        if ds % 100 == 0:
            print(f'  {case_name} n={n_val}: dataset {ds}/{n_datasets}',
                  flush=True)

        rng = np.random.default_rng(42 + ds)
        Z, Y, _, cov_Z = gen_func(n_val, rng)
        n = Z.shape[0]

        # Aniso eigendecomposition (needed for d_eig, U_T, c_est, c_new)
        Z_tilde = Z * sqrt_mu
        T_n = Z_tilde.T @ Z_tilde / n
        eigvals_T, U_T = np.linalg.eigh(T_n)
        order = np.argsort(eigvals_T)[::-1]
        d_eig = np.maximum(eigvals_T[order], 0)
        U_T = U_T[:, order]
        c_est = U_T.T @ (Z_tilde.T @ Y / n)
        c_new = mx_test @ U_T          # (n_test, K)
        c_new_sq = c_new ** 2           # (n_test, K)

        # Truncated aniso sigma^2 (needed for SE)
        trunc_sig2 = {}
        for mi_off, power in enumerate([0.3, 0.4]):
            J_tr = min(int(n_val ** power), K)
            gamma_tr = np.zeros(K)
            valid_tr = d_eig[:J_tr] > 1e-15
            gamma_tr[:J_tr] = np.where(valid_tr,
                                       c_est[:J_tr] / d_eig[:J_tr], 0.0)
            b_tr = (U_T @ gamma_tr) * sqrt_mu
            resid_tr = Y - Z @ b_tr
            sig2_tr = np.sum(resid_tr ** 2) / max(n - J_tr, 1)
            trunc_sig2[mi_off] = sig2_tr

        # === Adaptive Truncated Aniso (m=5: J=n^0.3, m=6: J=n^0.4) ===
        # 10-fold CV to select mu_n, then shrink by log(n).
        n_folds = 10
        fold_ids = np.arange(n) % n_folds
        rng_fold = np.random.default_rng(12345 + ds)
        rng_fold.shuffle(fold_ids)

        for mi_off, power in enumerate([0.3, 0.4]):
            J_tr = min(int(n_val ** power), K)

            # --- 10-fold CV over mu_grid ---
            cv_mse = np.zeros(n_grid_adapt)
            for fold in range(n_folds):
                val_mask = fold_ids == fold
                tr_mask = ~val_mask
                Z_tr, Y_tr = Z[tr_mask], Y[tr_mask]
                Z_val, Y_val = Z[val_mask], Y[val_mask]
                n_tr = Z_tr.shape[0]

                # Eigen-decomposition on training fold
                Z_tilde_tr = Z_tr * sqrt_mu
                T_tr = Z_tilde_tr.T @ Z_tilde_tr / n_tr
                eig_tr, U_tr = np.linalg.eigh(T_tr)
                ord_tr = np.argsort(eig_tr)[::-1]
                d_tr = np.maximum(eig_tr[ord_tr], 0)
                U_tr = U_tr[:, ord_tr]
                c_tr = U_tr.T @ (Z_tilde_tr.T @ Y_tr / n_tr)

                # Validation projections in eigenbasis
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

            # Best mu by CV, then shrink by log(n)
            best_gi = np.argmin(cv_mse)
            mu_cv = mu_grid[best_gi] / np.log(n_val) ** 2

            # --- Compute estimate + SE on full data with selected mu ---
            s_hat = d_eig[:J_tr]
            z_proj = c_new[:, :J_tr]
            z_proj_sq = c_new_sq[:, :J_tr]
            c_est_J = c_est[:J_tr]
            sig2_a = trunc_sig2[mi_off]

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

            all_yhat[5 + mi_off, ds] = np.sum(gamma_a * z_proj, axis=1)

            vw = np.where(den > 1e-15,
                          s_hat[None, :] / den ** 2, 0.0)
            all_se[5 + mi_off, ds] = np.sqrt(
                sig2_a / n * np.sum(z_proj_sq * vw, axis=1))

    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas,
             kappa_aniso=kappa_aniso, kappa_iso=kappa_iso,
             selected_lam=selected_lam,
             lam_grid=lam_grid)
    print(f'Saved {fname} (m=5&6 updated)')


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    C_a = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    C_i = float(sys.argv[4]) if len(sys.argv) > 4 else C_a
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    run_adapt_only(case_name, n_val, results_dir, C_aniso=C_a, C_iso=C_i)
