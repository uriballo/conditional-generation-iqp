"""Correctness checks for the radial-shell (S_n-symmetric) encoding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.encoding import RADIAL_SHELL_ENCODING, radial_shell_amplitudes
from iqp_cond.exact import enumerate_bitstrings, exact_iqp_probabilities_many
from iqp_cond.simulator import estimate_pauli_z

jax.config.update("jax_enable_x64", True)


def _brute_reference(c1: float, c2: float, n: int) -> np.ndarray:
    shell = np.asarray(
        radial_shell_amplitudes(jnp.asarray([c1, c2] + [0.0] * (n - 2)), n)
    )
    weights = enumerate_bitstrings(n).sum(axis=1)
    return shell[weights].astype(np.complex128)


def _fwht(v: np.ndarray) -> np.ndarray:
    t = np.asarray(v, dtype=np.complex128).copy()
    w = 1
    while w < t.shape[0]:
        blocks = t.reshape(-1, 2 * w)
        left = blocks[:, :w].copy()
        right = blocks[:, w:].copy()
        blocks[:, :w] = left + right
        blocks[:, w:] = left - right
        w *= 2
    return t


def _generators(n: int) -> np.ndarray:
    return np.array(
        [
            [int(i in g) for i in range(n)]
            for g in [(0,), (1,), (2,), (3,), (4,), (0, 1), (1, 2), (2, 3), (0, 4)]
        ],
        dtype=np.int8,
    )


def test_radial_shell_exact_matches_brute() -> None:
    rng = np.random.default_rng(0)
    n = 6
    generators = _generators(n)
    params = rng.normal(0, 0.5, size=len(generators))
    c1, c2 = 0.3, 0.7
    angles = np.array([[c1, c2] + [0.0] * (n - 2)])
    got = exact_iqp_probabilities_many(
        params, generators, angles, RADIAL_SHELL_ENCODING
    )[0]
    ref = _brute_reference(c1, c2, n)
    phase = np.zeros(2**n)
    idx = np.arange(2**n, dtype=np.uint32)
    for theta, g in zip(params, generators, strict=True):
        wires = np.flatnonzero(g)
        if wires.size == 1:
            char = 1.0 - 2.0 * ((idx >> int(wires[0])) & 1)
        else:
            char = 1.0 - 2.0 * (((idx >> int(wires[0])) ^ (idx >> int(wires[1]))) & 1)
        phase += theta * char
    diag = ref * np.exp(1j * phase)
    want = np.abs(_fwht(diag) / np.sqrt(2**n)) ** 2
    want = want / want.sum()
    assert np.allclose(got, want, atol=1e-12), np.max(np.abs(got - want))


def test_radial_shell_estimator_matches_exact() -> None:
    rng = np.random.default_rng(1)
    n = 6
    generators = _generators(n)
    params = rng.normal(0, 0.5, size=len(generators))
    ops = enumerate_bitstrings(n).astype(np.float64)
    angles = np.array([[0.4, 0.6] + [0.0] * (n - 2)])
    probs = exact_iqp_probabilities_many(
        params, generators, angles, RADIAL_SHELL_ENCODING
    )[0]
    bits = enumerate_bitstrings(n)
    exact = np.array([(probs * (1 - 2 * ((bits @ op) % 2))).sum() for op in ops])
    estimate = estimate_pauli_z(
        jnp.asarray(params),
        jnp.asarray(generators),
        jnp.asarray(ops),
        jnp.asarray(angles),
        jax.random.PRNGKey(0),
        8000,
        RADIAL_SHELL_ENCODING,
    )
    mc = np.asarray(estimate.values[0])
    assert np.max(np.abs(exact - mc)) < 0.05, np.max(np.abs(exact - mc))
