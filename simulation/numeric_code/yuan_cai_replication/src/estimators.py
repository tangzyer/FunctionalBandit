"""Estimators: roughness regularization and FPCA."""

import numpy as np
from .metrics import excess_risk


def predict(b_hat, Z_new):
    """Predict ⟨β̂, X⟩ for new observations.

    Parameters
    ----------
    b_hat : array, shape (K,)
        Estimated coefficients in the cosine basis.
    Z_new : array, shape (n_new, K) or (K,)
        Cosine basis scores of new predictor(s).

    Returns
    -------
    y_hat : array, shape (n_new,) or scalar
        Predicted values: Z_new @ b_hat.
    """
    return Z_new @ b_hat


def rkhs_penalty_matrix(K):
    """Diagonal penalty matrix D = diag(1/μ_k) for the cosine RKHS.

    μ_k = 2/(kπ)^4, so 1/μ_k = (kπ)^4 / 2.

    Parameters
    ----------
    K : int
        Number of basis functions.

    Returns
    -------
    D : array, shape (K,)
        Diagonal entries of the penalty matrix.
    """
    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** 4
    return 1.0 / mu_k


def roughness_regularization(Z, Y, lam, K):
    """Roughness regularization estimator.

    Solves: (Z^T Z / n + λ D) b = Z^T Y / n.

    Parameters
    ----------
    Z : array, shape (n, K)
        Score matrix.
    Y : array, shape (n,)
        Response.
    lam : float
        Regularization parameter.
    K : int
        Number of basis functions.

    Returns
    -------
    b_hat : array, shape (K,)
        Estimated coefficients.
    """
    n = Z.shape[0]
    D = rkhs_penalty_matrix(K)
    ZtZ = Z.T @ Z / n
    ZtY = Z.T @ Y / n
    A = ZtZ + lam * np.diag(D)
    b_hat = np.linalg.solve(A, ZtY)
    return b_hat


def fpca_estimator(Z, Y, J):
    """FPCA estimator: truncated spectral approach.

    Eigendecompose the sample covariance Z^T Z / n, keep top J components,
    and regress Y on those components.

    Parameters
    ----------
    Z : array, shape (n, K)
        Score matrix.
    Y : array, shape (n,)
        Response.
    J : int
        Number of principal components to retain.

    Returns
    -------
    b_hat : array, shape (K,)
        Estimated coefficients.
    """
    n, K = Z.shape
    if J <= 0:
        return np.zeros(K)

    # Sample covariance
    S = Z.T @ Z / n  # (K, K)

    # Eigendecomposition (sorted ascending by numpy)
    eigvals, eigvecs = np.linalg.eigh(S)

    # Take top J eigenvectors (largest eigenvalues)
    idx = np.argsort(eigvals)[::-1][:J]
    V = eigvecs[:, idx]       # (K, J)
    d = eigvals[idx]          # (J,)

    # Skip near-zero eigenvalues
    mask = d > 1e-12
    if not np.any(mask):
        return np.zeros(K)
    V = V[:, mask]
    d = d[mask]

    # Project and regress: b_hat = V diag(1/d) V^T (Z^T Y / n)
    ZtY = Z.T @ Y / n
    coeffs = V.T @ ZtY  # (J,)
    coeffs = coeffs / d  # (J,)
    b_hat = V @ coeffs   # (K,)

    return b_hat


def fpca_inference(Z, Y, J, z_new, alpha=0.05):
    """Inference for ⟨β, X_new⟩ via FPCA + OLS.

    Truncates to J principal components, treats it as finite-dimensional
    OLS, and constructs confidence intervals using the t-distribution.

    Parameters
    ----------
    Z : array, shape (n, K)
        Training score matrix.
    Y : array, shape (n,)
        Training response.
    J : int
        Number of principal components.
    z_new : array, shape (K,)
        Cosine basis scores of the new predictor.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    dict with keys:
        'prediction' : float
            Point estimate ŷ = ⟨β̂, X_new⟩.
        'left' : (float, float)
            Left-sided (1-α) CI: (-∞, upper).
        'right' : (float, float)
            Right-sided (1-α) CI: (lower, ∞).
        'two_sided' : (float, float)
            Two-sided (1-α) CI: (lower, upper).
    """
    from scipy.stats import t as t_dist

    n, K = Z.shape

    # Eigendecompose sample covariance
    S = Z.T @ Z / n
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order[:J]]   # (K, J)

    # PC scores
    W = Z @ V  # (n, J)

    # OLS in PC space: γ̂ = (W^T W)^{-1} W^T Y
    WtW = W.T @ W
    WtW_inv = np.linalg.inv(WtW)
    gamma_hat = WtW_inv @ (W.T @ Y)

    # Residual variance estimate
    residuals = Y - W @ gamma_hat
    sigma2_hat = np.sum(residuals ** 2) / (n - J)

    # Prediction at new point
    w_new = V.T @ z_new  # (J,)
    y_hat = w_new @ gamma_hat

    # Standard error of prediction
    var_pred = sigma2_hat * (w_new @ WtW_inv @ w_new)
    se_pred = np.sqrt(var_pred)

    # t-quantiles
    df = n - J
    t_two = t_dist.ppf(1 - alpha / 2, df)
    t_one = t_dist.ppf(1 - alpha, df)

    return {
        'prediction': y_hat,
        'left': (-np.inf, y_hat + t_one * se_pred),
        'right': (y_hat - t_one * se_pred, np.inf),
        'two_sided': (y_hat - t_two * se_pred, y_hat + t_two * se_pred),
    }


def roughness_reg_inference(Z, Y, lam, K, x_new, alpha=0.05):
    """Inference for ⟨β, X_new⟩ via roughness regularization.

    Uses the variance proxy:
        σ̂_n(x)^iso = (σ̂_ε / √n) √(x^T (Z^TZ/n + λD)^{-1} x)

    Parameters
    ----------
    Z : array, shape (n, K)
        Training score matrix (cosine basis).
    Y : array, shape (n,)
        Training response.
    lam : float
        Regularization parameter λ_n.
    K : int
        Number of basis functions.
    x_new : array, shape (K,)
        Cosine basis scores of the new predictor.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    dict with keys:
        'prediction' : float
        'left' : (float, float) — (-∞, upper)
        'right' : (float, float) — (lower, ∞)
        'two_sided' : (float, float) — (lower, upper)
    """
    from scipy.stats import norm

    n = Z.shape[0]
    D = rkhs_penalty_matrix(K)

    # Estimation
    A = Z.T @ Z / n + lam * np.diag(D)
    A_inv = np.linalg.inv(A)
    b_hat = A_inv @ (Z.T @ Y / n)

    # Residual variance estimate
    residuals = Y - Z @ b_hat
    sigma2_hat = np.sum(residuals ** 2) / n

    # Prediction
    y_hat = x_new @ b_hat

    # Variance proxy: (σ̂_ε² / n) * x^T A^{-1} x
    var_pred = sigma2_hat / n * (x_new @ A_inv @ x_new)
    se_pred = np.sqrt(var_pred)

    # Normal quantiles
    z_two = norm.ppf(1 - alpha / 2)
    z_one = norm.ppf(1 - alpha)

    return {
        'prediction': y_hat,
        'left': (-np.inf, y_hat + z_one * se_pred),
        'right': (y_hat - z_one * se_pred, np.inf),
        'two_sided': (y_hat - z_two * se_pred, y_hat + z_two * se_pred),
    }


def oracle_roughness_reg(Z, Y, b_true, K, cov_Z, lam_grid=None):
    """Oracle-tuned roughness regularization: pick λ minimizing excess risk.

    Parameters
    ----------
    Z : array, shape (n, K)
    Y : array, shape (n,)
    b_true : array, shape (K,)
    K : int
    cov_Z : array, shape (K, K)
        Population covariance of scores in cosine basis.
    lam_grid : array or None
        Grid of λ values to search over.

    Returns
    -------
    b_hat_best : array, shape (K,)
    best_risk : float
    best_lam : float
    """
    if lam_grid is None:
        lam_grid = np.logspace(-12, 2, 200)

    best_risk = np.inf
    b_hat_best = None
    best_lam = None

    for lam in lam_grid:
        b_hat = roughness_regularization(Z, Y, lam, K)
        risk = excess_risk(b_hat, b_true, cov_Z)
        if risk < best_risk:
            best_risk = risk
            b_hat_best = b_hat
            best_lam = lam

    return b_hat_best, best_risk, best_lam


def anisotropic_roughness_reg(Z, Y, lam, K):
    """Anisotropic roughness regularization estimator.

    In the transformed space Z_i = K^{1/2} X_i, applies anisotropic
    penalty ν_j = min(λ, √(d_j λ)) where d_j are eigenvalues of T_n.

    Parameters
    ----------
    Z : array, shape (n, K)
        Score matrix (cosine basis).
    Y : array, shape (n,)
        Response.
    lam : float
        Base regularization parameter λ_n.
    K : int
        Number of basis functions.

    Returns
    -------
    b_hat : array, shape (K,)
        Estimated coefficients in cosine basis.
    """
    n = Z.shape[0]
    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** 4
    sqrt_mu = np.sqrt(mu_k)

    # Transform to RKHS-scaled space: Xi_tilde = Xi * M
    Z_tilde = Z * sqrt_mu  # (n, K)

    # T_n = Z_tilde^T Z_tilde / n
    T_n = Z_tilde.T @ Z_tilde / n  # (K, K)

    # Eigendecompose T_n
    eigvals, U = np.linalg.eigh(T_n)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    U = U[:, order]

    # Anisotropic penalty: ν_j = √(d_j λ)
    d = np.maximum(eigvals, 0)  # ensure non-negative
    nu = np.sqrt(d * lam)

    # Solve in eigenbasis: γ_j = c_j / (d_j + ν_j)
    c = U.T @ (Z_tilde.T @ Y / n)  # (K,)
    denom = d + nu
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = np.where(denom > 1e-15, c / denom, 0.0)

    # Map back: b_tilde = U gamma, then b = M * b_tilde
    b_tilde = U @ gamma
    b_hat = sqrt_mu * b_tilde

    return b_hat


def oracle_anisotropic_roughness_reg(Z, Y, b_true, K, cov_Z, lam_grid=None):
    """Oracle-tuned anisotropic roughness regularization.

    Parameters
    ----------
    Z : array, shape (n, K)
    Y : array, shape (n,)
    b_true : array, shape (K,)
    K : int
    cov_Z : array, shape (K, K)
    lam_grid : array or None

    Returns
    -------
    b_hat_best : array, shape (K,)
    best_risk : float
    best_lam : float
    """
    if lam_grid is None:
        lam_grid = np.logspace(-12, 2, 200)

    best_risk = np.inf
    b_hat_best = None
    best_lam = None

    for lam in lam_grid:
        b_hat = anisotropic_roughness_reg(Z, Y, lam, K)
        risk = excess_risk(b_hat, b_true, cov_Z)
        if risk < best_risk:
            best_risk = risk
            b_hat_best = b_hat
            best_lam = lam

    return b_hat_best, best_risk, best_lam


def anisotropic_reg_inference(Z, Y, lam, K, x_new, alpha=0.05):
    """Inference for ⟨β, X_new⟩ via anisotropic roughness regularization.

    Variance proxy:
        σ̂_n^ani(x)² = (σ̂_ε² / n) · Σ_j  d_j c_j² / (d_j + ν_j)²

    where d_j, U are from eigendecomposition of T_n = Z̃^T Z̃ / n,
    c = U^T M x_new, ν_j = min(λ, √(d_j λ)).

    Parameters
    ----------
    Z : array, shape (n, K)
        Training score matrix (cosine basis).
    Y : array, shape (n,)
        Training response.
    lam : float
        Regularization parameter λ_n.
    K : int
        Number of basis functions.
    x_new : array, shape (K,)
        Cosine basis scores of the new predictor.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    dict with keys:
        'prediction' : float
        'left' : (float, float) — (-∞, upper)
        'right' : (float, float) — (lower, ∞)
        'two_sided' : (float, float) — (lower, upper)
    """
    from scipy.stats import norm

    n = Z.shape[0]
    ks = np.arange(1, K + 1, dtype=float)
    mu_k = 2.0 / (ks * np.pi) ** 4
    sqrt_mu = np.sqrt(mu_k)

    # Transform to RKHS-scaled space
    Z_tilde = Z * sqrt_mu  # (n, K)

    # T_n eigendecomposition
    T_n = Z_tilde.T @ Z_tilde / n
    eigvals, U = np.linalg.eigh(T_n)
    order = np.argsort(eigvals)[::-1]
    d = np.maximum(eigvals[order], 0)
    U = U[:, order]

    # Anisotropic penalty: ν_j = √(d_j λ)
    nu = np.sqrt(d * lam)

    # Estimation (reuse the same computation as anisotropic_roughness_reg)
    c_est = U.T @ (Z_tilde.T @ Y / n)
    denom = d + nu
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = np.where(denom > 1e-15, c_est / denom, 0.0)
    b_tilde = U @ gamma
    b_hat = sqrt_mu * b_tilde

    # Residual variance estimate
    residuals = Y - Z @ b_hat
    sigma2_hat = np.sum(residuals ** 2) / n

    # Prediction
    y_hat = x_new @ b_hat

    # Variance proxy: (σ̂² / n) · Σ_j d_j c_j² / (d_j + ν_j)²
    c_new = U.T @ (sqrt_mu * x_new)  # project transformed new predictor
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(denom > 1e-15, d * c_new ** 2 / denom ** 2, 0.0)
    var_pred = sigma2_hat / n * np.sum(w)
    se_pred = np.sqrt(var_pred)

    # Normal quantiles
    z_two = norm.ppf(1 - alpha / 2)
    z_one = norm.ppf(1 - alpha)

    return {
        'prediction': y_hat,
        'left': (-np.inf, y_hat + z_one * se_pred),
        'right': (y_hat - z_one * se_pred, np.inf),
        'two_sided': (y_hat - z_two * se_pred, y_hat + z_two * se_pred),
    }


def oracle_fpca(Z, Y, b_true, K, cov_Z, J_max=None):
    """Oracle-tuned FPCA: pick J minimizing excess risk.

    Optimized: eigendecompose once, then incrementally build b_hat
    for J = 1, 2, ..., J_max.

    Parameters
    ----------
    Z : array, shape (n, K)
    Y : array, shape (n,)
    b_true : array, shape (K,)
    K : int
    cov_Z : array, shape (K, K)
        Population covariance of scores in cosine basis.
    J_max : int or None
        Maximum number of components (defaults to K).

    Returns
    -------
    b_hat_best : array, shape (K,)
    best_risk : float
    best_J : int
    """
    if J_max is None:
        J_max = K

    n = Z.shape[0]
    S = Z.T @ Z / n
    eigvals, eigvecs = np.linalg.eigh(S)

    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    ZtY = Z.T @ Y / n

    best_risk = np.inf
    b_hat_best = None
    best_J = 1
    b_hat = np.zeros(K)

    for j in range(J_max):
        d = eigvals[j]
        if d < 1e-12:
            break
        v = eigvecs[:, j]          # (K,)
        alpha = (v @ ZtY) / d      # scalar
        b_hat = b_hat + alpha * v   # incremental update
        risk = excess_risk(b_hat, b_true, cov_Z)
        if risk < best_risk:
            best_risk = risk
            b_hat_best = b_hat.copy()
            best_J = j + 1

    return b_hat_best, best_risk, best_J
