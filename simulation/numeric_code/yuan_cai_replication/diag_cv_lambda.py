"""Diagnostic: check typical CV-selected lambda values."""
import sys, os
sys.path.insert(0, '.')
import numpy as np
from src.covariance import figure2_specs, figure3_top_specs, figure3_bottom_specs
from src.data_generation import generate_data_cosine_basis, generate_data_haar_basis, true_beta_coeffs
from src.estimators import rkhs_penalty_matrix

K = 50; sigma = 0.5
n_grid = 200
lam_grid = np.logspace(-12, 0, n_grid)
D = rkhs_penalty_matrix(K)
b_true = true_beta_coeffs(K)
ks = np.arange(1, K+1, dtype=float)
mu_k = 2.0 / (ks * np.pi)**4
sqrt_mu = np.sqrt(mu_k)

cases = {
    'aligned_r2_2': (figure2_specs(K, r2_values=[2.0])[0], 'cosine'),
    'shifted': (figure3_top_specs(K, k0_values=[10])[0], 'cosine'),
    'haar_r2_2': (figure3_bottom_specs(K, r2_values=[2.0])[0], 'haar'),
}

n_values = [64, 256, 1024]
n_check = 100

for case_name, (cov_spec, basis) in cases.items():
    for n_val in n_values:
        lams_iso = []
        lams_aniso = []
        for ds in range(n_check):
            rng = np.random.default_rng(42 + ds)
            if basis == 'cosine':
                Z, Y, _, cov_Z = generate_data_cosine_basis(n_val, cov_spec, K, sigma=sigma, rng=rng)
            else:
                Z, Y, _, cov_Z = generate_data_haar_basis(n_val, cov_spec, K, M=K, sigma=sigma, rng=rng)
            n = Z.shape[0]

            n_folds = 5
            cv_idx = np.arange(n)
            rng_cv = np.random.default_rng(42 + ds + 10000)
            rng_cv.shuffle(cv_idx)
            folds = np.array_split(cv_idx, n_folds)

            cv_errors_iso = np.zeros(n_grid)
            cv_errors_aniso = np.zeros(n_grid)

            for fold in range(n_folds):
                val_idx = folds[fold]
                train_idx = np.concatenate([folds[f] for f in range(n_folds) if f != fold])
                Z_tr, Y_tr = Z[train_idx], Y[train_idx]
                Z_va, Y_va = Z[val_idx], Y[val_idx]
                n_tr = len(train_idx); n_va = len(val_idx)
                ZtZ_tr = Z_tr.T @ Z_tr / n_tr
                ZtY_tr = Z_tr.T @ Y_tr / n_tr

                Z_tilde_tr = Z_tr * sqrt_mu
                T_n_tr = Z_tilde_tr.T @ Z_tilde_tr / n_tr
                eigvals_tr, U_tr = np.linalg.eigh(T_n_tr)
                order_tr = np.argsort(eigvals_tr)[::-1]
                d_eig_tr = np.maximum(eigvals_tr[order_tr], 0)
                U_tr = U_tr[:, order_tr]
                c_est_tr = U_tr.T @ (Z_tilde_tr.T @ Y_tr / n_tr)

                for li, lam in enumerate(lam_grid):
                    A_tr = ZtZ_tr + lam * np.diag(D)
                    b_hat_tr = np.linalg.solve(A_tr, ZtY_tr)
                    cv_errors_iso[li] += np.sum((Y_va - Z_va @ b_hat_tr)**2) / n_va

                    nu_tr = np.minimum(lam, np.sqrt(d_eig_tr * lam))
                    den_tr = d_eig_tr + nu_tr
                    gam_tr = np.where(den_tr > 1e-15, c_est_tr / den_tr, 0.0)
                    b_hat_a = sqrt_mu * (U_tr @ gam_tr)
                    cv_errors_aniso[li] += np.sum((Y_va - Z_va @ b_hat_a)**2) / n_va

            lams_iso.append(lam_grid[np.argmin(cv_errors_iso)])
            lams_aniso.append(lam_grid[np.argmin(cv_errors_aniso)])

        li = np.array(lams_iso)
        la = np.array(lams_aniso)
        lo_i = np.sum(li == lam_grid[0])
        hi_i = np.sum(li == lam_grid[-1])
        lo_a = np.sum(la == lam_grid[0])
        hi_a = np.sum(la == lam_grid[-1])
        print(f'{case_name:15s} n={n_val:4d} | ISO: median={np.median(li):.2e} [{np.min(li):.2e}, {np.max(li):.2e}] lo={lo_i} hi={hi_i} | ANISO: median={np.median(la):.2e} [{np.min(la):.2e}, {np.max(la):.2e}] lo={lo_a} hi={hi_a}')
