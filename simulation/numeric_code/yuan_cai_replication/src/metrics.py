"""Metrics for evaluating functional regression estimators."""

import numpy as np


def excess_risk(b_hat, b_true, cov_Z):
    """Compute excess risk = (b̂ − b)ᵀ C_Z (b̂ − b).

    This is E[(⟨β̂ − β₀, X⟩)²], the expected squared prediction error
    for a new X with covariance C_Z in the cosine basis.

    Parameters
    ----------
    b_hat : array, shape (K,)
        Estimated coefficients.
    b_true : array, shape (K,)
        True coefficients.
    cov_Z : array, shape (K, K)
        Covariance matrix of the score vector in the cosine basis.

    Returns
    -------
    risk : float
    """
    diff = b_hat - b_true
    return diff @ cov_Z @ diff
