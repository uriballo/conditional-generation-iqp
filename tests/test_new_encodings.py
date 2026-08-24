"""Correctness checks for the reupload and spin-flip two-branch encodings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.encoding import (
    REUPLOAD_L2_ENCODING,
    REUPLOAD_L3_ENCODING,
    SPIN_FLIP_BRANCH_ENCODING,
    post_hadamard_amplitudes,
    two_branch_shell_amplitudes,
)
from iqp_cond.exact import enumerate_bitstrings, exact_iqp_probabilities_many
from iqp_cond.simulator import estimate_pauli_z

jax.config.update("jax_enable_x64", True)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]])


def _rz(a: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def _hadamard() -> np.ndarray:
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def _brute_reupload(a: float, layers: int) -> np.ndarray:
    u = _ry(a)
    for _ in range(layers - 1):
        u = u @ _rz(np.pi / 2) @ _ry(a)
    return _hadamard() @ u @ np.array([1, 0])


@pytest.mark.parametrize(
    ("encoding", "layers"),
    [(REUPLOAD_L2_ENCODING, 2), (REUPLOAD_L3_ENCODING, 3)],
)
def test_reupload_amplitudes_match_brute(encoding: str, layers: int) -> None:
    for a in np.linspace(-0.5, 0.5, 5):
        z, o = post_hadamard_amplitudes(jnp.asarray(a), encoding)
        brute = _brute_reupload(float(a), layers)
        assert np.allclose(np.asarray(z), brute[0], atol=1e-12)
        assert np.allclose(np.asarray(o), brute[1], atol=1e-12)


def test_two_branch_shell_matches_brute() -> None:
    n = 3
    bits = enumerate_bitstrings(n)
    weights = bits.sum(axis=1)
    for a in np.linspace(-0.4, 0.4, 5):
        shell = np.asarray(two_branch_shell_amplitudes(jnp.asarray(a), n))
        p = np.array([np.cos(a / 2), np.sin(a / 2)])
        psi = np.array([1.0])
        x = np.eye(1)
        for _ in range(n):
            psi = np.kron(psi, p)
            x = np.kron(x, np.array([[0, 1], [1, 0]]))
        state = psi + x @ psi
        h = np.eye(1)
        for _ in range(n):
            h = np.kron(h, _hadamard())
        phi = h @ state
        assert np.allclose(shell[weights], phi, atol=1e-12)


def _exact_expectations(
    params: np.ndarray, generators: np.ndarray, angles: np.ndarray, encoding: str
) -> np.ndarray:
    n = generators.shape[1]
    probs = exact_iqp_probabilities_many(params, generators, angles, encoding)[0]
    bits = enumerate_bitstrings(n)
    ops = bits.astype(np.float64)
    return np.array(
        [(probs * (1 - 2 * ((bits @ op) % 2))).sum() for op in ops]
    )


@pytest.mark.parametrize(
    "encoding",
    [REUPLOAD_L2_ENCODING, REUPLOAD_L3_ENCODING, SPIN_FLIP_BRANCH_ENCODING],
)
def test_estimator_matches_exact(encoding: str) -> None:
    rng = np.random.default_rng(1)
    n = 4
    generators = np.array(
        [
            [int(i in g) for i in range(n)]
            for g in [(0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3), (0, 3)]
        ],
        dtype=np.int8,
    )
    params = rng.normal(0, 0.5, size=len(generators))
    angles = np.array([np.full(n, 0.3)])
    ops = enumerate_bitstrings(n).astype(np.float64)
    exact = _exact_expectations(params, generators, angles, encoding)
    estimate = estimate_pauli_z(
        jnp.asarray(params),
        jnp.asarray(generators),
        jnp.asarray(ops),
        jnp.asarray(angles),
        jax.random.PRNGKey(0),
        4000,
        encoding,
    )
    mc = np.asarray(estimate.values[0])
    assert np.max(np.abs(exact - mc)) < 0.06, (encoding, np.max(np.abs(exact - mc)))
