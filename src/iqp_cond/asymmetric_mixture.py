"""Asymmetric two-peak mixture target over Hamming weight (S_n exchangeable)."""

from __future__ import annotations

from math import comb

import numpy as np

from .exact import enumerate_bitstrings


def asymmetric_mixture_profile(n_qubits: int, c1: float, c2: float) -> np.ndarray:
    """Hamming-weight profile: w(c2) Gaussian at mu1 plus (1-w) Gaussian at mu2.

    ``c1`` controls the midpoint (location) and ``c2`` the mixture weight
    (skewness).  At ``c2 = 0.5`` the profile is symmetric-bimodal; away from
    it the two peaks are unequal, so the target is not Z2 (spin-flip)
    symmetric.
    """
    mu0 = n_qubits * (0.3 + 0.4 * c1)
    separation, sigma = 4.0, 1.2
    mu1, mu2 = mu0 - separation / 2.0, mu0 + separation / 2.0
    weight = c2
    k = np.arange(n_qubits + 1, dtype=np.float64)
    return weight * np.exp(-((k - mu1) ** 2) / (2.0 * sigma**2)) + (
        1.0 - weight
    ) * np.exp(-((k - mu2) ** 2) / (2.0 * sigma**2))


def asymmetric_mixture_target(n_qubits: int, c1: float, c2: float) -> np.ndarray:
    """Bitstring probabilities for the asymmetric two-peak mixture target."""
    profile = asymmetric_mixture_profile(n_qubits, c1, c2)
    bits = enumerate_bitstrings(n_qubits)
    weights = bits.sum(axis=1)
    binom = np.asarray([comb(n_qubits, kk) for kk in range(n_qubits + 1)])
    probabilities = profile[weights] / binom[weights]
    probabilities /= probabilities.sum()
    return probabilities
