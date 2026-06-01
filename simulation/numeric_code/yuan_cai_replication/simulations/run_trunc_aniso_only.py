"""Slim runner: Trunc.Aniso at a single arbitrary q.

Same data-gen seeds and Lepski/undersmoothing as run_alpha_sweep_v13.py.
Saves (1, n_datasets, n_test)-shaped all_yhat / all_se to a separate npz
so the plot script can stack methods.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.covariance import figure2_specs, figure3_top_specs, figure3_bottom_specs
from src.data_generation import (generate_data_cosine_basis,
                                  generate_data_haar_basis, true_beta_coeffs)


def lepski_select(yhat_grid, se_grid, kappa):
    n_grid, n_test = yhat_grid.shape
    best_idx = np.ones(n_test, dtype=int)
    lower = yhat_grid[0] - kappa * se_grid[0]
    upper = yhat_grid[0] + kappa * se_grid[0]
    for j in range(1, n_grid):
        valid = (yhat_grid[j] >= lower) & (yhat_grid[j] <= upper)
        best_idx = np.where(valid, j, best_idx)
        lower = np.maximum(lower, yhat_grid[j] - kappa * se_grid[j])
        upper = np.minimum(upper, yhat_grid[j] + kappa * se_grid[j])
    best_idx = np.minimum(best_idx, n_grid - 2)
    return best_idx


def _resolve_case(case_name, sigma, K):
    """Map case name to (cov_spec, gen_func, b_true)."""
    # Pull out r2_val from suffix; supports e.g. r2_0p5, r2_1, r2_3.
    if case_name == 'aligned_r2_0p5_beta4':
        cov_spec = figure2_specs(K, r2_values=[0.5])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=4)
        b_true = true_beta_coeffs(K, power=4)
    elif case_name == 'haar_r2_0p5_beta4':
        cov_spec = figure3_bottom_specs(K, r2_values=[0.5])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_power=4)
        b_true = true_beta_coeffs(K, power=4)
    elif case_name == 'aligned_r2_1_beta4':
        cov_spec = figure2_specs(K, r2_values=[1.0])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=4)
        b_true = true_beta_coeffs(K, power=4)
    elif case_name == 'haar_r2_1_beta4':
        cov_spec = figure3_bottom_specs(K, r2_values=[1.0])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_power=4)
        b_true = true_beta_coeffs(K, power=4)
    elif case_name == 'aligned_r2_0p5_beta2p75':
        cov_spec = figure2_specs(K, r2_values=[0.5])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=2.75)
        b_true = true_beta_coeffs(K, power=2.75)
    elif case_name == 'aligned_r2_1_beta2p75':
        cov_spec = figure2_specs(K, r2_values=[1.0])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=2.75)
        b_true = true_beta_coeffs(K, power=2.75)
    elif case_name == 'haar_r2_0p5_beta2p75':
        cov_spec = figure3_bottom_specs(K, r2_values=[0.5])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_power=2.75)
        b_true = true_beta_coeffs(K, power=2.75)
    elif case_name == 'haar_r2_1_beta4':
        cov_spec = figure3_bottom_specs(K, r2_values=[1.0])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_power=4)
        b_true = true_beta_coeffs(K, power=4)
    elif case_name == 'haar_r2_1_beta2p75':
        cov_spec = figure3_bottom_specs(K, r2_values=[1.0])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_power=2.75)
        b_true = true_beta_coeffs(K, power=2.75)
    elif case_name == 'aligned_r2_0p5':
        cov_spec = figure2_specs(K, r2_values=[0.5])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
        b_true = true_beta_coeffs(K, power=2)
    elif case_name == 'aligned_r2_1':
        cov_spec = figure2_specs(K, r2_values=[1.0])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
        b_true = true_beta_coeffs(K, power=2)
    elif case_name == 'haar_r2_0p5':
        cov_spec = figure3_bottom_specs(K, r2_values=[0.5])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
        b_true = true_beta_coeffs(K, power=2)
    elif case_name == 'haar_r2_1':
        cov_spec = figure3_bottom_specs(K, r2_values=[1.0])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
        b_true = true_beta_coeffs(K, power=2)
    elif case_name == 'aligned_r2_2':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
        b_true = true_beta_coeffs(K, power=2)
    elif case_name == 'haar_r2_2':
        cov_spec = figure3_bottom_specs(K, r2_values=[2.0])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng)
        b_true = true_beta_coeffs(K, power=2)
    elif case_name == 'shifted':
        cov_spec = figure3_top_specs(K, k0_values=[25])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng)
        b_true = true_beta_coeffs(K, power=2)
    elif case_name == 'aligned_r2_2_beta4':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=4)
        b_true = true_beta_coeffs(K, power=4)
    elif case_name == 'haar_r2_2_beta4':
        cov_spec = figure3_bottom_specs(K, r2_values=[2.0])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_power=4)
        b_true = true_beta_coeffs(K, power=4)
    elif case_name == 'shifted_beta4':
        cov_spec = figure3_top_specs(K, k0_values=[25])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=4)
        b_true = true_beta_coeffs(K, power=4)
    elif case_name == 'aligned_r2_2_beta3p5':
        cov_spec = figure2_specs(K, r2_values=[2.0])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=3.5)
        b_true = true_beta_coeffs(K, power=3.5)
    elif case_name == 'haar_r2_2_beta3p5':
        cov_spec = figure3_bottom_specs(K, r2_values=[2.0])[0]
        gen = lambda n, rng: generate_data_haar_basis(
            n, cov_spec, K, M=K, sigma=sigma, rng=rng, beta_power=3.5)
        b_true = true_beta_coeffs(K, power=3.5)
    elif case_name == 'shifted_beta3p5':
        cov_spec = figure3_top_specs(K, k0_values=[25])[0]
        gen = lambda n, rng: generate_data_cosine_basis(
            n, cov_spec, K, sigma=sigma, rng=rng, beta_power=3.5)
        b_true = true_beta_coeffs(K, power=3.5)
    else:
        raise ValueError(f'Unknown case: {case_name}')
    return cov_spec, gen, b_true


def run(case_name, n_val, results_dir, q_trunc=0.2, C_aniso=0.005,
        test_func_kind='bm'):
    K = 200
    sigma = 0.5
    n_datasets = 500
    n_test = 500
    n_grid = 2 * n_val
    lam_grid = np.logspace(-3 * np.log10(n_val), -1 * np.log10(n_val), n_grid)

    cov_spec, gen_func, b_true = _resolve_case(case_name, sigma, K)

    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** 4
    sqrt_mu = np.sqrt(mu_k)

    rng_test = np.random.default_rng(999)
    if test_func_kind == 'bm':
        n_grid_pts = 200
        t_grid = np.linspace(0, 1, n_grid_pts + 1)[1:]
        dt = 1.0 / n_grid_pts
        dW = rng_test.normal(size=(n_test, n_grid_pts)) * np.sqrt(dt)
        bm_paths = np.cumsum(dW, axis=1)
        phi_grid = np.sqrt(2) * np.cos(np.outer(ks, np.pi * t_grid))
        x_test = bm_paths @ phi_grid.T * dt
    elif test_func_kind == 'input':
        x_test, _, _, _ = gen_func(n_test, rng_test)
    else:
        raise ValueError(f'Unknown test_func_kind: {test_func_kind}')
    true_vals = x_test @ b_true
    mx_test = x_test * sqrt_mu

    log_term = np.sqrt(np.log(n_grid) + np.log(n_val))
    kappa_aniso = C_aniso * log_term
    logn2 = np.log(n_val) ** 2
    us_shift = int(round(-n_grid * np.log(logn2) / (2 * np.log(n_val))))
    J_tr = min(int(n_val ** q_trunc), K)
    print(f'  {case_name} n={n_val} q_trunc={q_trunc} J={J_tr} '
          f'kappa={kappa_aniso:.4f}', flush=True)

    all_yhat = np.zeros((1, n_datasets, n_test))
    all_se = np.zeros((1, n_datasets, n_test))
    alphas = np.linspace(0.01, 0.30, 30)

    suffix = f'truncq0p{int(q_trunc * 100)}'
    _tfx_suf = '' if test_func_kind == 'bm' else f'_{test_func_kind}test'
    partial_fname = (f'alpha_sweep_{case_name}_n{n_val}_{suffix}'
                     f'_v13{_tfx_suf}.partial.npz')
    partial_path = os.path.join(results_dir, partial_fname)
    start_ds = 0
    if os.path.exists(partial_path):
        try:
            _cp = np.load(partial_path)
            if (_cp['all_yhat'].shape == all_yhat.shape and
                int(_cp['last_ds']) + 1 < n_datasets):
                all_yhat[:] = _cp['all_yhat']
                all_se[:] = _cp['all_se']
                start_ds = int(_cp['last_ds']) + 1
                print(f'  resumed from ds={start_ds}', flush=True)
        except Exception as _e:
            print(f'  could not resume ({_e}); starting fresh', flush=True)

    CHECKPOINT_EVERY = 50

    mask_J = np.zeros(K)
    mask_J[:J_tr] = 1.0

    for ds in range(start_ds, n_datasets):
        if ds % 100 == 0:
            print(f'  {case_name} n={n_val}: dataset {ds}/{n_datasets}',
                  flush=True)

        rng = np.random.default_rng(42 + ds)
        Z, Y, _, _ = gen_func(n_val, rng)
        n = Z.shape[0]

        Z_tilde = Z * sqrt_mu
        T_n = Z_tilde.T @ Z_tilde / n
        eigvals_T, U_T = np.linalg.eigh(T_n)
        order = np.argsort(eigvals_T)[::-1]
        d_eig = np.maximum(eigvals_T[order], 0)
        U_T = U_T[:, order]
        c_est = U_T.T @ (Z_tilde.T @ Y / n)
        c_new = mx_test @ U_T
        c_new_sq = c_new ** 2

        lam_arr = lam_grid[:, None]
        d_arr = d_eig[None, :]
        nu_all = np.sqrt(d_arr * lam_arr)
        den_all = d_arr + nu_all
        gamma_all = np.where(den_all > 1e-15, c_est[None, :] / den_all, 0.0)
        wv_all = np.where(den_all > 1e-15, d_arr / den_all ** 2, 0.0)

        # Trunc.Aniso at q_trunc: mask top-J directions, Lepski over lam
        gamma_all_tr = gamma_all * mask_J[None, :]
        yhat_tr_all = gamma_all_tr @ c_new.T
        b_all_tr = (U_T @ gamma_all_tr.T).T * sqrt_mu[None, :]
        resid_tr_all = Y[:, None] - Z @ b_all_tr.T
        sig2_tr_all = np.sum(resid_tr_all ** 2, axis=0) / n
        wv_tr_all = wv_all * mask_J[None, :]
        se_tr_all = np.sqrt((sig2_tr_all[:, None] / n) *
                            (wv_tr_all @ c_new_sq.T))

        # Recompute analytically at λ/log²n (per test point) — grid is only
        # for Lepski selection, NOT for evaluating the inference quantities.
        idx_range = np.arange(n_test)
        best_idx_tr = lepski_select(yhat_tr_all, se_tr_all, kappa_aniso)
        logn2 = np.log(n_val) ** 2
        lam_used_tr = lam_grid[best_idx_tr] / logn2  # (n_test,)
        nu_used_tr = np.sqrt(d_eig[None, :] * lam_used_tr[:, None])
        den_used_tr = d_eig[None, :] + nu_used_tr
        gamma_used_tr_full = np.where(den_used_tr > 1e-15,
                                       c_est[None, :] / den_used_tr, 0.0)
        gamma_used_tr = gamma_used_tr_full * mask_J[None, :]
        yhat_used_tr = np.sum(gamma_used_tr * c_new, axis=1)
        b_used_tr = (U_T @ gamma_used_tr.T).T * sqrt_mu[None, :]
        resid_used_tr = Y[:, None] - Z @ b_used_tr.T
        sig2_used_tr = np.sum(resid_used_tr ** 2, axis=0) / n
        wv_used_tr = (np.where(den_used_tr > 1e-15,
                                d_eig[None, :] / den_used_tr ** 2, 0.0)
                      * mask_J[None, :])
        se_used_tr = np.sqrt(sig2_used_tr / n *
                             np.sum(c_new_sq * wv_used_tr, axis=1))
        all_yhat[0, ds] = yhat_used_tr
        all_se[0, ds] = se_used_tr

        if (ds + 1) % CHECKPOINT_EVERY == 0 or ds + 1 == n_datasets:
            tmp_path = partial_path + '.tmp.npz'
            np.savez(tmp_path,
                     all_yhat=all_yhat, all_se=all_se, last_ds=ds)
            os.replace(tmp_path, partial_path)

    fname = f'alpha_sweep_{case_name}_n{n_val}_{suffix}_v13{_tfx_suf}.npz'
    np.savez(os.path.join(results_dir, fname),
             all_yhat=all_yhat, all_se=all_se,
             true_vals=true_vals, alphas=alphas,
             q_trunc=q_trunc, J_trunc=J_tr,
             kappa_aniso=kappa_aniso)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f'Saved {fname}', flush=True)


if __name__ == '__main__':
    case_name = sys.argv[1]
    n_val = int(sys.argv[2])
    q_trunc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run(case_name, n_val, results_dir, q_trunc=q_trunc)
