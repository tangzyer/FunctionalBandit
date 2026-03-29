"""Basis function evaluation: cosine and Haar systems."""

import numpy as np


def cosine_basis(K, grid):
    """Evaluate cosine basis functions φ_k(t) = √2 cos(kπt) on a grid.

    Parameters
    ----------
    K : int
        Number of basis functions (k = 1, ..., K).
    grid : array, shape (T,)
        Evaluation points in [0, 1].

    Returns
    -------
    Phi : array, shape (T, K)
        Phi[t, k] = √2 cos((k+1)πt) for k = 0, ..., K-1.
    """
    grid = np.asarray(grid)
    ks = np.arange(1, K + 1)  # k = 1, 2, ..., K
    return np.sqrt(2) * np.cos(np.outer(grid, ks * np.pi))


def haar_basis(M, grid):
    """Evaluate Haar wavelet basis functions on [0, 1].

    Uses the standard Haar system: constant function plus wavelets at
    dyadic scales. Returns M basis functions total.

    Parameters
    ----------
    M : int
        Number of Haar basis functions.
    grid : array, shape (T,)
        Evaluation points in [0, 1].

    Returns
    -------
    H : array, shape (T, M)
        Haar basis evaluated at grid points.
    """
    grid = np.asarray(grid)
    T = len(grid)
    H = np.zeros((T, M))

    # First basis function: constant h_0(t) = 1
    H[:, 0] = 1.0

    idx = 1
    j = 0  # resolution level
    while idx < M:
        n_funcs = 2 ** j  # number of wavelets at level j
        for k in range(n_funcs):
            if idx >= M:
                break
            # Mother wavelet at level j, shift k
            # Support: [k/2^j, (k+1)/2^j]
            left = k / n_funcs
            mid = (k + 0.5) / n_funcs
            right = (k + 1) / n_funcs
            scale = np.sqrt(n_funcs)  # 2^{j/2} normalization
            mask_left = (grid >= left) & (grid < mid)
            mask_right = (grid >= mid) & (grid < right)
            H[mask_left, idx] = scale
            H[mask_right, idx] = -scale
            idx += 1
        j += 1

    return H


def haar_cosine_gram(M, K, n_quad=4096):
    """Compute cross-Gram matrix G[m, k] = ⟨h_m, φ_k⟩.

    Parameters
    ----------
    M : int
        Number of Haar basis functions.
    K : int
        Number of cosine basis functions.
    n_quad : int
        Number of quadrature points for numerical integration.

    Returns
    -------
    G : array, shape (M, K)
        Inner products between Haar and cosine bases.
    """
    grid = np.linspace(0, 1, n_quad, endpoint=False) + 0.5 / n_quad
    dt = 1.0 / n_quad
    H = haar_basis(M, grid)   # (n_quad, M)
    Phi = cosine_basis(K, grid)  # (n_quad, K)
    return H.T @ Phi * dt  # (M, K)
