"""Unit tests for 2-qubit block encodings and calibrated reuploading."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.encoding import (
    ANGLE_ENCODING,
    BLOCK_ZZ_PI4_ENCODING,
    BLOCK_ZZ_PI8_ENCODING,
    CALIBRATED_REUPLOAD_ENCODING,
    SYMMETRIC_REUPLOAD_ENCODING,
    block_post_hadamard_amplitudes,
    is_block_encoding,
)
from iqp_cond.exact import (
    exact_iqp_probabilities_many,
    exact_iqp_probabilities_many_jax,
)
from iqp_cond.mmd import conditional_mmd
from iqp_cond.simulator import estimate_pauli_z


def test_is_block_encoding():
    assert is_block_encoding(BLOCK_ZZ_PI8_ENCODING)
    assert is_block_encoding(BLOCK_ZZ_PI4_ENCODING)
    assert not is_block_encoding(ANGLE_ENCODING)
    assert not is_block_encoding(SYMMETRIC_REUPLOAD_ENCODING)
    assert not is_block_encoding(CALIBRATED_REUPLOAD_ENCODING)


@pytest.mark.parametrize("encoding", [BLOCK_ZZ_PI8_ENCODING, BLOCK_ZZ_PI4_ENCODING])
def test_block_amplitudes_normalized(encoding):
    angles = jnp.array(
        [[0.2, 0.4, -0.1, 0.5], [0.0, 0.0, 0.0, 0.0]], dtype=jnp.float64
    )
    block_amps, singleton = block_post_hadamard_amplitudes(angles, encoding)
    assert singleton is None
    assert block_amps.shape == (2, 2, 4)
    probs = jnp.sum(jnp.abs(block_amps) ** 2, axis=-1)
    np.testing.assert_allclose(probs, 1.0, atol=1e-12)


def test_block_amplitudes_odd_qubits():
    angles = jnp.array([[0.2, 0.4, 0.6]], dtype=jnp.float64)
    block_amps, singleton = block_post_hadamard_amplitudes(
        angles, BLOCK_ZZ_PI8_ENCODING
    )
    assert singleton is not None
    s_zero, s_one = singleton
    assert block_amps.shape == (1, 1, 4)
    np.testing.assert_allclose(
        jnp.abs(s_zero) ** 2 + jnp.abs(s_one) ** 2, 1.0, atol=1e-12
    )


def test_exact_probabilities_normalized_all_encodings():
    for n_qubits in [2, 3, 4, 5]:
        ansatz = complete_ansatz(n_qubits)
        params = np.ones(ansatz.n_parameters, dtype=np.float64) * 0.1
        angles = np.array([np.linspace(-0.5, 0.5, n_qubits)], dtype=np.float64)
        for enc in [
            ANGLE_ENCODING,
            SYMMETRIC_REUPLOAD_ENCODING,
            CALIBRATED_REUPLOAD_ENCODING,
            BLOCK_ZZ_PI8_ENCODING,
            BLOCK_ZZ_PI4_ENCODING,
        ]:
            probs_np = exact_iqp_probabilities_many(
                params,
                ansatz.generators,
                angles,
                encoding=enc,
            )
            assert probs_np.shape == (1, 2**n_qubits)
            np.testing.assert_allclose(probs_np.sum(axis=1), 1.0, atol=1e-12)

            probs_jax = exact_iqp_probabilities_many_jax(
                jnp.asarray(params),
                jnp.asarray(ansatz.generators),
                jnp.asarray(angles),
                encoding=enc,
            )
            np.testing.assert_allclose(probs_np, np.asarray(probs_jax), atol=1e-12)


@pytest.mark.parametrize(
    "enc",
    [BLOCK_ZZ_PI8_ENCODING, BLOCK_ZZ_PI4_ENCODING, CALIBRATED_REUPLOAD_ENCODING],
)
def test_monte_carlo_estimator_matches_exact_expectation(enc):
    n_qubits = 4
    ansatz = complete_ansatz(n_qubits)
    params = np.array([0.15] * ansatz.n_parameters, dtype=np.float64)
    angles = np.array([[0.3, -0.2, 0.5, -0.1]], dtype=np.float64)
    ops = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.float64,
    )

    probs = exact_iqp_probabilities_many(
        params,
        ansatz.generators,
        angles,
        encoding=enc,
    )[0]
    bitstrings = np.arange(2**n_qubits, dtype=np.uint32)[:, None]
    shifts = np.arange(n_qubits, dtype=np.uint32)
    bits = (bitstrings >> shifts) & 1
    parities = (bits @ ops.T) % 2
    characters = 1.0 - 2.0 * parities
    exact_exp = np.sum(characters * probs[:, None], axis=0)

    key = jax.random.PRNGKey(42)
    est = estimate_pauli_z(
        jnp.asarray(params),
        jnp.asarray(ansatz.generators),
        jnp.asarray(ops),
        jnp.asarray(angles),
        key,
        n_samples=50_000,
        encoding=enc,
    )
    mc_values = np.asarray(est.values)[0]
    np.testing.assert_allclose(mc_values, exact_exp, atol=0.03)


@pytest.mark.parametrize(
    "enc",
    [BLOCK_ZZ_PI8_ENCODING, BLOCK_ZZ_PI4_ENCODING, CALIBRATED_REUPLOAD_ENCODING],
)
def test_mmd_gradient_computation(enc):
    n_qubits = 4
    ansatz = complete_ansatz(n_qubits)
    params = jnp.array([0.1] * ansatz.n_parameters, dtype=jnp.float64)
    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    angles = jnp.array([[0.2, 0.2, 0.2, 0.2]], dtype=jnp.float64)
    targets = jnp.zeros((1, 10, n_qubits), dtype=jnp.float64)
    key = jax.random.PRNGKey(123)

    def loss_fn(p, current_enc=enc):
        res = conditional_mmd(
            p,
            generators,
            angles,
            targets,
            sigmas=(1.0, 2.0),
            n_ops=16,
            n_iqp_samples=100,
            key=key,
            encoding=current_enc,
        )
        return res.loss

    grad = jax.grad(loss_fn)(params)
    assert grad.shape == (ansatz.n_parameters,)
    assert bool(jnp.all(jnp.isfinite(grad)))
