"""Correctness checks for the two-block (Z2 x Z2) spin-flip encoding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.encoding import (
    TWO_BLOCK_BRANCH_ENCODING,
    two_branch_shell_amplitudes,
)
from iqp_cond.exact import enumerate_bitstrings, exact_iqp_probabilities_many
from iqp_cond.simulator import estimate_pauli_z

jax.config.update("jax_enable_x64", True)


def _brute_reference(a_a: float, a_b: float, m_a: int, m_b: int) -> np.ndarray:
    shell_a = np.asarray(two_branch_shell_amplitudes(jnp.asarray(a_a), m_a))
    shell_b = np.asarray(two_branch_shell_amplitudes(jnp.asarray(a_b), m_b))
    n = m_a + m_b
    ref = np.zeros(2**n, dtype=np.complex128)
    for j in range(2**n):
        k_a = bin(j & ((1 << m_a) - 1)).count("1")
        k_b = bin(j >> m_a).count("1")
        ref[j] = shell_a[k_a] * shell_b[k_b]
    return ref


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


def _generators(m_a: int, m_b: int) -> np.ndarray:
    n = m_a + m_b
    return np.array(
        [
            [int(i in g) for i in range(n)]
            for g in [
                (0,),
                (1,),
                (2,),
                (3,),
                (4,),
                (5,),
                (0, 1),
                (1, 2),
                (3, 4),
                (0, 3),
            ]
        ],
        dtype=np.int8,
    )


def test_two_block_exact_matches_brute() -> None:
    rng = np.random.default_rng(0)
    m_a, m_b = 3, 3
    n = m_a + m_b
    generators = _generators(m_a, m_b)
    params = rng.normal(0, 0.5, size=len(generators))
    a_a, a_b = 0.3, -0.2
    angles = np.array([[a_a] * m_a + [a_b] * m_b])
    got = exact_iqp_probabilities_many(
        params, generators, angles, TWO_BLOCK_BRANCH_ENCODING
    )[0]
    ref = _brute_reference(a_a, a_b, m_a, m_b)
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


def test_two_block_estimator_matches_exact() -> None:
    rng = np.random.default_rng(1)
    m_a, m_b = 3, 3
    n = m_a + m_b
    generators = _generators(m_a, m_b)
    params = rng.normal(0, 0.5, size=len(generators))
    ops = enumerate_bitstrings(n).astype(np.float64)
    angles = np.array([[0.3] * m_a + [-0.2] * m_b])
    probs = exact_iqp_probabilities_many(
        params, generators, angles, TWO_BLOCK_BRANCH_ENCODING
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
        TWO_BLOCK_BRANCH_ENCODING,
    )
    mc = np.asarray(estimate.values[0])
    assert np.max(np.abs(exact - mc)) < 0.05, np.max(np.abs(exact - mc))
