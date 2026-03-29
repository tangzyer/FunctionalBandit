"""Generate functional data (X, Y) for simulation experiments."""

import numpy as np
from .basis import haar_cosine_gram


def true_beta_coeffs(K, power=2):
    """True slope coefficients in the cosine basis.

    b_k = 4 * (-1)^{k-1} / k^power  for k = 1, ..., K.

    Parameters
    ----------
    K : int
        Number of basis functions.
    power : float
        Decay exponent (default 2).

    Returns
    -------
    b : array, shape (K,)
    """
    ks = np.arange(1, K + 1, dtype=float)
    return 4.0 * ((-1.0) ** (ks - 1)) / ks ** power


def generate_data_cosine_basis(n, cov_spec, K, sigma=0.5, rng=None, beta_power=2,
                               beta_vec=None):
    """Generate data when covariance eigenfunctions are the cosine basis.

    X(t) = Σ_k ξ_k φ_k(t), where ξ_k ~ N(0, θ_k).
    Y_i = ∫ X_i(t) β₀(t) dt + ε_i = Σ_k ξ_{ik} b_k + ε_i.

    Parameters
    ----------
    n : int
        Sample size.
    cov_spec : CovarianceSpec
        Covariance specification (must have basis_type='cosine').
    K : int
        Number of basis functions to use.
    sigma : float
        Noise standard deviation.
    rng : np.random.Generator or None

    Returns
    -------
    Z : array, shape (n, K)
        Score matrix: Z[i, k] = ξ_{ik} (random coefficients of X_i).
    Y : array, shape (n,)
        Response values.
    b_true : array, shape (K,)
        True β₀ coefficients in cosine basis.
    cov_Z : array, shape (K, K)
        Population covariance of Z (diagonal: diag(θ_k)).
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = cov_spec.eigenvalues[:K]
    b_true = beta_vec if beta_vec is not None else true_beta_coeffs(K, power=beta_power)

    # Population covariance of scores (diagonal since eigenbasis = cosine)
    cov_Z = np.diag(theta)

    # Generate random scores ξ_{ik} ~ N(0, θ_k)
    Z = rng.normal(size=(n, K)) * np.sqrt(theta)

    # Response: Y_i = Σ_k ξ_{ik} b_k + ε_i
    signal = Z @ b_true
    noise = rng.normal(size=n) * sigma
    Y = signal + noise

    return Z, Y, b_true, cov_Z


def generate_data_haar_basis(n, cov_spec, K, M=None, sigma=0.5, rng=None,
                             beta_power=2, beta_vec=None):
    """Generate data when covariance eigenfunctions are the Haar basis.

    X(t) = Σ_m ξ_m h_m(t), with ξ_m ~ N(0, θ_m).
    The cosine-basis scores are obtained via the cross-Gram matrix:
        Z[i, k] = Σ_m ξ_{im} G[m, k]  where G[m, k] = ⟨h_m, φ_k⟩.

    Parameters
    ----------
    n : int
        Sample size.
    cov_spec : CovarianceSpec
        Covariance specification (must have basis_type='haar').
    K : int
        Number of cosine basis functions for the estimator.
    M : int or None
        Number of Haar basis functions (defaults to K).
    sigma : float
        Noise standard deviation.
    rng : np.random.Generator or None

    Returns
    -------
    Z : array, shape (n, K)
        Projected score matrix in the cosine basis.
    Y : array, shape (n,)
        Response values.
    b_true : array, shape (K,)
        True β₀ coefficients in cosine basis.
    cov_Z : array, shape (K, K)
        Population covariance of Z in cosine basis: G^T diag(θ) G.
    """
    if rng is None:
        rng = np.random.default_rng()
    if M is None:
        M = K

    theta = cov_spec.eigenvalues[:M]
    b_true = beta_vec if beta_vec is not None else true_beta_coeffs(K, power=beta_power)

    # Cross-Gram matrix: G[m, k] = ⟨h_m, φ_k⟩
    G = haar_cosine_gram(M, K)

    # Population covariance of Z in cosine basis: G^T diag(θ) G
    cov_Z = G.T @ np.diag(theta) @ G

    # Generate Haar scores ξ_{im} ~ N(0, θ_m)
    Xi = rng.normal(size=(n, M)) * np.sqrt(theta)

    # Project to cosine basis: Z[i, k] = Σ_m ξ_{im} G[m, k]
    Z = Xi @ G  # (n, K)

    # Response: Y_i = ⟨X_i, β₀⟩ + ε_i = Σ_k Z_{ik} b_k + ε_i
    signal = Z @ b_true
    noise = rng.normal(size=n) * sigma
    Y = signal + noise

    return Z, Y, b_true, cov_Z
