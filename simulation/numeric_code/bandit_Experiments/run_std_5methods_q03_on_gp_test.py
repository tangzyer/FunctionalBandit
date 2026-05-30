"""Fit the 5 inference methods (q=0.3-aligned) on the full Std-logY training
set and output CIs for eta(x) at each of the 100 GP-synthetic test functions.

Methods: Aniso (Lepski), Iso (Lepski), FPCA J=sqrt(n),
         Trunc.Aniso J=n^0.3 (Lepski), Adapt.Trunc J=n^0.3.
Figure `fig:sim-grid-trunc` of `anisotropic_v2.tex` shows q=0.3 is the smallest
truncation level whose coverage stays honest at all simulated r_2 in {0.5,1,2},
so it is the natural default for real-data use where the true r_2 is unknown.

- Training: std_weekly_profile_clean_xnorm_q20_80_logY_centered.csv (n=2640, p=336)
- Test:     std_xnorm_q20_80_logY_gp_test_functions_n100.csv (100 x 336)
- Basis:    Haar K=32, r=1.5   (winner from kernel_selection_wide, C=0.005)
- Interval: CI for eta(x): yhat +/- q * se.  (NOT PI for Y_new -- see
  feedback_ci_formula memory.)

Output CSV (long format): one row per (test_idx, method, conf_level).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

BASE = Path("/Users/zhiyuantang/Dropbox/FunctionalBandit/simulation/numeric_code/bandit_Experiments")
IN_TR = BASE / "std_weekly_profile_clean_xnorm_q20_80_logY_centered.csv"
IN_TE = BASE / "std_xnorm_q20_80_logY_gp_test_functions_n100.csv"
OUT = BASE / "std_xnorm_q20_80_logY_ci_5methods_q03_gp_test.csv"

BASIS = "haar"
K_BASIS = 32
R_SMOOTH = 1.5
C_LEPSKI = 0.005                 # matches kernel_selection_wide run
CONF_LEVELS = [0.75, 0.85, 0.95]
SEED = 0


def build_haar(p, K, r):
    t = (np.arange(p) + 0.5) / p
    Phi = np.empty((K, p)); mu = np.empty(K)
    Phi[0] = 1.0; mu[0] = 1.0
    idx = 1; j = 0
    while idx < K:
        for k in range(2 ** j):
            if idx >= K:
                break
            arg = (2 ** j) * t - k
            plus = (arg >= 0) & (arg < 0.5)
            minus = (arg >= 0.5) & (arg < 1.0)
            Phi[idx] = (2 ** (j / 2.0)) * (plus.astype(float) - minus.astype(float))
            mu[idx] = 2.0 ** (-(2 * r + 1) * (j + 1))
            idx += 1
        j += 1
    return Phi, mu


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


def fit_all_methods(Z, Y_tr, Z_te, sqrt_mu, D, C_lepski, seed=0):
    """Port of run_lcl_6methods.fit_all_methods. Returns dict per method index."""
    n, K = Z.shape
    n_te = Z_te.shape[0]
    n_grid = 2 * n
    lam_grid = np.logspace(-3 * np.log10(n), -1 * np.log10(n), n_grid)
    kappa = C_lepski * np.sqrt(np.log(n_grid) + np.log(n))

    out = {}

    # Aniso sweep
    Z_tilde = Z * sqrt_mu
    T_n = Z_tilde.T @ Z_tilde / n
    eig_T, U_T = np.linalg.eigh(T_n)
    o = np.argsort(eig_T)[::-1]
    d_eig = np.maximum(eig_T[o], 0); U_T = U_T[:, o]
    c_est = U_T.T @ (Z_tilde.T @ Y_tr / n)
    c_new = (Z_te * sqrt_mu) @ U_T
    c_new_sq = c_new ** 2
    d_arr = d_eig[None, :]; lam_arr = lam_grid[:, None]
    nu_all = np.sqrt(d_arr * lam_arr)
    den_all = d_arr + nu_all
    gamma_all = np.where(den_all > 1e-15, c_est[None, :] / den_all, 0.0)
    yhat_aniso = gamma_all @ c_new.T
    b_all = (U_T @ gamma_all.T).T * sqrt_mu[None, :]
    resid = Y_tr[:, None] - Z @ b_all.T
    sig2_arr = np.sum(resid ** 2, axis=0) / n
    wv = np.where(den_all > 1e-15, d_arr / den_all ** 2, 0.0)
    se_aniso = np.sqrt((sig2_arr[:, None] / n) * (wv @ c_new_sq.T))

    idx_range = np.arange(n_te)
    sel = lepski_select(yhat_aniso, se_aniso, kappa)
    out[0] = dict(yhat=yhat_aniso[sel, idx_range],
                  se=se_aniso[sel, idx_range],
                  sig2=sig2_arr[sel], sel=sel)

    # Iso sweep
    T_iso = 1.0 / np.sqrt(D)
    ZtZ = Z.T @ Z / n; ZtY = Z.T @ Y_tr / n
    B_mat = ZtZ * np.outer(T_iso, T_iso)
    eig_B, Q = np.linalg.eigh(B_mat)
    oB = np.argsort(eig_B)[::-1]
    eig_B = np.maximum(eig_B[oB], 0); Q = Q[:, oB]
    c_iso_coef = Q.T @ (T_iso * ZtY)
    v_te = Q.T @ (T_iso[:, None] * Z_te.T)
    v_te_sq = v_te ** 2
    inv_diag = 1.0 / (eig_B[None, :] + lam_grid[:, None])
    coeff_iso = c_iso_coef[None, :] * inv_diag
    b_iso = (coeff_iso @ Q.T) * T_iso[None, :]
    yhat_iso = b_iso @ Z_te.T
    resid_i = Y_tr[:, None] - Z @ b_iso.T
    sig2_i_arr = np.sum(resid_i ** 2, axis=0) / n
    quad = inv_diag @ v_te_sq
    se_iso = np.sqrt(sig2_i_arr[:, None] / n * quad)
    sel = lepski_select(yhat_iso, se_iso, kappa)
    out[1] = dict(yhat=yhat_iso[sel, idx_range],
                  se=se_iso[sel, idx_range],
                  sig2=sig2_i_arr[sel], sel=sel)

    # FPCA J=sqrt(n)
    S = Z.T @ Z / n
    eig_S, V_S = np.linalg.eigh(S)
    oS = np.argsort(eig_S)[::-1]; V_S = V_S[:, oS]
    J_fpca = int(np.sqrt(n))
    W = Z @ V_S[:, :J_fpca]
    WtW = W.T @ W
    Wi = np.linalg.pinv(WtW)
    gf = Wi @ (W.T @ Y_tr)
    rf = Y_tr - W @ gf
    sig2_f = float(np.sum(rf ** 2) / max(n - J_fpca, 1))
    w_new = Z_te @ V_S[:, :J_fpca]
    wWi = w_new @ Wi
    yhat_f = w_new @ gf
    se_f = np.sqrt(sig2_f * np.sum(wWi * w_new, axis=1))
    out[2] = dict(yhat=yhat_f, se=se_f, sig2=sig2_f, J=J_fpca)

    # Truncated Aniso n^0.3 (only — q=0.3 default)
    power = 0.3
    J_tr = min(int(n ** power), K)
    mask = np.zeros(K); mask[:J_tr] = 1.0
    gamma_tr = gamma_all * mask[None, :]
    yhat_tr = gamma_tr @ c_new.T
    b_tr = (U_T @ gamma_tr.T).T * sqrt_mu[None, :]
    resid_t = Y_tr[:, None] - Z @ b_tr.T
    sig2_t = np.sum(resid_t ** 2, axis=0) / n
    wv_tr = wv * mask[None, :]
    se_tr = np.sqrt((sig2_t[:, None] / n) * (wv_tr @ c_new_sq.T))
    sel = lepski_select(yhat_tr, se_tr, kappa)
    out[3] = dict(yhat=yhat_tr[sel, idx_range],
                  se=se_tr[sel, idx_range],
                  sig2=sig2_t[sel], J=J_tr, sel=sel)

    # Adaptive Truncated Aniso at q=0.3 (default, matching Trunc.Aniso above).
    J_ad = min(int(n ** 0.3), K)
    mu_grid = np.logspace(-4 * np.log10(n), 0, 100)
    n_folds = 10
    fold_ids = np.arange(n) % n_folds
    rng_fold = np.random.default_rng(12345 + seed)
    rng_fold.shuffle(fold_ids)
    cv_mse = np.zeros(mu_grid.size)
    for f in range(n_folds):
        val_mask = fold_ids == f
        tr_mask = ~val_mask
        Z_tr_f, Y_tr_f = Z[tr_mask], Y_tr[tr_mask]
        Z_val, Y_val = Z[val_mask], Y_tr[val_mask]
        n_tr = Z_tr_f.shape[0]
        Z_tilde_tr = Z_tr_f * sqrt_mu
        T_tr = Z_tilde_tr.T @ Z_tilde_tr / n_tr
        eig_tr, U_tr = np.linalg.eigh(T_tr)
        o_tr = np.argsort(eig_tr)[::-1]
        d_tr = np.maximum(eig_tr[o_tr], 0); U_tr = U_tr[:, o_tr]
        c_tr = U_tr.T @ (Z_tilde_tr.T @ Y_tr_f / n_tr)
        mx_val = Z_val * sqrt_mu
        c_val = mx_val @ U_tr
        c_val_sq = c_val ** 2
        s_tr_a = d_tr[:J_ad]
        zp = c_val[:, :J_ad]; zp_sq = c_val_sq[:, :J_ad]
        c_tr_J = c_tr[:J_ad]
        abs_zp = np.maximum(np.abs(zp), 1e-15)
        b1_base_cv = s_tr_a[None, :] / abs_zp
        cond_cv = zp_sq >= 2.0 * s_tr_a[None, :]
        for gi, mu_val in enumerate(mu_grid):
            sq_mu_v = np.sqrt(mu_val)
            lam_b1 = b1_base_cv * sq_mu_v
            lam_b2 = np.minimum(mu_val, np.sqrt(s_tr_a * mu_val))[None, :]
            lam = np.where(cond_cv, lam_b1, lam_b2)
            den = s_tr_a[None, :] + lam
            gamma_cv = np.where(den > 1e-15, c_tr_J[None, :] / den, 0.0)
            yhat_val = np.sum(gamma_cv * zp, axis=1)
            cv_mse[gi] += np.sum((Y_val - yhat_val) ** 2)
    mu_cv = mu_grid[int(np.argmin(cv_mse))] / np.log(n) ** 2

    s_hat = d_eig[:J_ad]
    z_proj = c_new[:, :J_ad]; z_proj_sq = c_new_sq[:, :J_ad]
    c_est_J = c_est[:J_ad]
    gamma_tr_a = np.zeros(K)
    valid = s_hat > 1e-15
    gamma_tr_a[:J_ad] = np.where(valid, c_est[:J_ad] / s_hat, 0.0)
    b_tr_a = (U_T @ gamma_tr_a) * sqrt_mu
    resid_tr_a = Y_tr - Z @ b_tr_a
    sig2_ad = float(np.sum(resid_tr_a ** 2) / max(n - J_ad, 1))
    sq_mu_sel = np.sqrt(mu_cv)
    cond = z_proj_sq >= 2.0 * s_hat[None, :]
    abs_z = np.maximum(np.abs(z_proj), 1e-15)
    lam_b1 = s_hat[None, :] * sq_mu_sel / abs_z
    lam_b2 = np.minimum(mu_cv, np.sqrt(s_hat * mu_cv))[None, :]
    lam = np.where(cond, lam_b1, lam_b2)
    den = s_hat[None, :] + lam
    gamma_a = np.where(den > 1e-15, c_est_J[None, :] / den, 0.0)
    yhat_a = np.sum(gamma_a * z_proj, axis=1)
    vw = np.where(den > 1e-15, s_hat[None, :] / den ** 2, 0.0)
    se_a = np.sqrt(sig2_ad / n * np.sum(z_proj_sq * vw, axis=1))
    out[4] = dict(yhat=yhat_a, se=se_a, sig2=sig2_ad, J=J_ad, mu_cv=mu_cv)
    return out


def main():
    df_tr = pd.read_csv(IN_TR)
    feat = [c for c in df_tr.columns if c not in ("LCLid", "TARGET")]
    X_tr = df_tr[feat].to_numpy()
    Y_tr = df_tr["TARGET"].to_numpy()
    n, p = X_tr.shape
    print(f"train: n={n}, p={p}")

    df_te = pd.read_csv(IN_TE)
    assert list(df_te.columns) == feat, "test columns must match training feature columns"
    X_te = df_te.to_numpy()
    n_te = X_te.shape[0]
    print(f"test:  n_te={n_te}, p={X_te.shape[1]}")

    Phi, mu = build_haar(p, K_BASIS, R_SMOOTH)
    sqrt_mu = np.sqrt(mu); D = 1.0 / mu
    Z_tr = X_tr @ Phi.T / p
    Z_te = X_te @ Phi.T / p
    print(f"basis=Haar K={K_BASIS} r={R_SMOOTH}  C_lepski={C_LEPSKI}")

    res = fit_all_methods(Z_tr, Y_tr, Z_te, sqrt_mu, D, C_LEPSKI, seed=SEED)

    method_names = [
        "Aniso (Lepski)",
        "Iso (Lepski)",
        "FPCA J=sqrt(n)",
        "Trunc.Aniso J=n^0.3 (Lepski)",
        "Adapt.Trunc J=n^0.3",
    ]

    rows = []
    for m, name in enumerate(method_names):
        yhat = np.asarray(res[m]["yhat"])
        se = np.asarray(res[m]["se"])
        sig2 = np.atleast_1d(res[m]["sig2"]).astype(float)
        if sig2.size == 1:
            sig2 = np.repeat(sig2, yhat.size)
        for cl in CONF_LEVELS:
            alpha = 1 - cl
            if m == 2:
                J = int(np.sqrt(n))
                q = t_dist.ppf(1 - alpha / 2, df=n - J)
            elif m == 4:
                J = min(int(n ** 0.3), K_BASIS)
                q = t_dist.ppf(1 - alpha / 2, df=n - J)
            else:
                q = norm.ppf(1 - alpha / 2)
            half = q * se
            for t_idx in range(n_te):
                rows.append(dict(
                    test_idx=t_idx,
                    method=name,
                    conf_level=cl,
                    yhat=float(yhat[t_idx]),
                    se=float(se[t_idx]),
                    sig2_hat=float(sig2[t_idx]),
                    q=float(q),
                    half_width=float(half[t_idx]),
                    lower=float(yhat[t_idx] - half[t_idx]),
                    upper=float(yhat[t_idx] + half[t_idx]),
                ))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  shape={out_df.shape}")

    # quick summary
    print("\nMean half-width per method at each conf_level:")
    summ = out_df.groupby(["method", "conf_level"])["half_width"].mean().unstack("conf_level")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
