"""Covariance specifications for Figures 2 and 3."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CovarianceSpec:
    """Specification for the covariance of X(t).

    eigenvalues : array, shape (K,)
        Eigenvalues θ_k of the covariance operator.
    basis_type : str
        'cosine' or 'haar' — the eigenbasis of the covariance.
    label : str
        Human-readable label for plots.
    """
    eigenvalues: np.ndarray
    basis_type: str
    label: str


def figure2_specs(K, r2_values=(1.0, 1.5, 2.0, 2.5, 3.0)):
    """Covariance specs for Figure 2: aligned cosine basis.

    θ_k = k^{-2r₂}, eigenfunctions = cosine basis (aligned with RKHS).
    """
    specs = []
    for r2 in r2_values:
        ks = np.arange(1, K + 1, dtype=float)
        eigenvalues = ks ** (-2 * r2)
        specs.append(CovarianceSpec(
            eigenvalues=eigenvalues,
            basis_type='cosine',
            label=f'$r_2={r2:.1f}$'
        ))
    return specs


def figure3_top_specs(K, k0_values=(5, 10, 15, 20)):
    """Covariance specs for Figure 3 top: shifted location (k₀).

    θ_k = (|k - k₀| + 1)^{-2}, eigenfunctions = cosine basis.
    The eigenvalues peak near k₀, creating ordering misalignment.
    """
    specs = []
    for k0 in k0_values:
        ks = np.arange(1, K + 1, dtype=float)
        eigenvalues = (np.abs(ks - k0) + 1.0) ** (-2)
        specs.append(CovarianceSpec(
            eigenvalues=eigenvalues,
            basis_type='cosine',
            label=f'$k_0={k0}$'
        ))
    return specs


def figure3_bottom_specs(K, r2_values=(1.0, 1.5, 2.0, 2.5)):
    """Covariance specs for Figure 3 bottom: Haar eigenbasis.

    θ_k = k^{-2r₂}, eigenfunctions = Haar basis (misaligned with RKHS).
    """
    specs = []
    for r2 in r2_values:
        ks = np.arange(1, K + 1, dtype=float)
        eigenvalues = ks ** (-2 * r2)
        specs.append(CovarianceSpec(
            eigenvalues=eigenvalues,
            basis_type='haar',
            label=f'$r_2={r2:.1f}$'
        ))
    return specs
