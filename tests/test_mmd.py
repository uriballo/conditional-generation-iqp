import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond import (
    conditional_mmd,
    exact_iqp_probabilities,
    hamming_rbf,
    mmd_operator_probability,
)
from iqp_cond.exact import enumerate_bitstrings


def test_fourier_operator_probability_reconstructs_hamming_kernel():
    sigma = 1.7
    x = jnp.array([0, 1, 0, 1, 1])
    y = jnp.array([1, 1, 0, 0, 1])
    p = mmd_operator_probability(sigma)
    reconstructed = (1.0 - 2.0 * p) ** jnp.sum(x != y)

    assert jnp.allclose(reconstructed, hamming_rbf(x, y, sigma))


def test_label_swap_detects_conditional_mismatch_with_same_pooled_marginal():
    generators = jnp.eye(2, dtype=jnp.float64)
    params = jnp.zeros(2, dtype=jnp.float64)
    angles = jnp.array([[0.0, 0.0], [jnp.pi, jnp.pi]])
    zeros = jnp.zeros((64, 2), dtype=jnp.float64)
    ones = jnp.ones((64, 2), dtype=jnp.float64)
    aligned = jnp.stack([zeros, ones])
    swapped = jnp.stack([ones, zeros])

    correct = conditional_mmd(
        params,
        generators,
        angles,
        aligned,
        (1.0, 2.0, 4.0),
        2_000,
        200,
        jax.random.PRNGKey(0),
    )
    incorrect = conditional_mmd(
        params,
        generators,
        angles,
        swapped,
        (1.0, 2.0, 4.0),
        2_000,
        200,
        jax.random.PRNGKey(0),
    )

    assert abs(float(correct.loss)) < 1e-12
    assert float(incorrect.loss) > 0.05
    assert np.array_equal(
        np.sort(np.asarray(aligned).reshape(-1, 2), axis=0),
        np.sort(np.asarray(swapped).reshape(-1, 2), axis=0),
    )


def test_conditional_mmd_is_differentiable_and_balanced():
    generators = jnp.array([[1, 0], [0, 1], [1, 1]], dtype=jnp.float64)
    params = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64)
    angles = jnp.array([[0.0, 0.0], [0.4, 0.4]], dtype=jnp.float64)
    targets = jnp.array(
        [
            [[0, 0], [0, 1], [1, 0], [0, 0]],
            [[1, 1], [0, 1], [1, 0], [1, 1]],
        ],
        dtype=jnp.float64,
    )
    key = jax.random.PRNGKey(9)

    def objective(theta):
        return conditional_mmd(
            theta,
            generators,
            angles,
            targets,
            (1.0, 2.0),
            64,
            512,
            key,
        ).loss

    result = conditional_mmd(
        params,
        generators,
        angles,
        targets,
        (1.0, 2.0),
        64,
        512,
        key,
    )
    gradient = jax.grad(objective)(params)

    assert result.loss == jnp.mean(result.per_condition)
    assert result.per_condition.shape == (2,)
    assert result.per_bandwidth_condition.shape == (2, 2)
    assert gradient.shape == params.shape
    assert jnp.all(jnp.isfinite(gradient))


def _brute_force_u_mmd(params, generators, angles, targets, sigmas):
    bitstrings = enumerate_bitstrings(generators.shape[1])
    losses = []
    for condition in range(angles.shape[0]):
        probabilities = exact_iqp_probabilities(
            params,
            generators,
            angles[condition],
        )
        target_indices = (
            targets[condition]
            @ (1 << np.arange(targets.shape[2], dtype=np.int64))
        ).astype(int)
        sigma_losses = []
        for sigma in sigmas:
            distances = np.not_equal(
                bitstrings[:, None, :],
                bitstrings[None, :, :],
            ).sum(axis=-1)
            kernel = np.exp(-distances / (2.0 * sigma**2))
            model_model = probabilities @ kernel @ probabilities
            model_target = np.mean(probabilities @ kernel[:, target_indices])
            target_kernel = kernel[np.ix_(target_indices, target_indices)]
            target_target = (
                target_kernel.sum() - np.trace(target_kernel)
            ) / (len(target_indices) * (len(target_indices) - 1))
            sigma_losses.append(model_model - 2.0 * model_target + target_target)
        losses.append(np.mean(sigma_losses))
    return float(np.mean(losses))


def test_loss_and_gradient_agree_with_brute_force_within_measured_uncertainty():
    generators = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    params = jnp.array([0.17, -0.11, 0.23], dtype=jnp.float64)
    angles = np.array([[0.1, 0.1], [0.6, 0.6]], dtype=float)
    targets = np.array(
        [
            [[0, 0], [0, 1], [1, 0], [0, 0], [1, 1], [0, 0]],
            [[1, 1], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0]],
        ],
        dtype=float,
    )
    sigmas = (1.0, 2.0, 4.0)
    exact_loss = _brute_force_u_mmd(params, generators, angles, targets, sigmas)
    epsilon = 1e-5
    exact_gradient = np.empty(params.size)
    for index in range(params.size):
        direction = np.zeros(params.size)
        direction[index] = epsilon
        exact_gradient[index] = (
            _brute_force_u_mmd(
                np.asarray(params) + direction,
                generators,
                angles,
                targets,
                sigmas,
            )
            - _brute_force_u_mmd(
                np.asarray(params) - direction,
                generators,
                angles,
                targets,
                sigmas,
            )
        ) / (2.0 * epsilon)

    losses = []
    gradients = []
    for repeat in range(8):
        key = jax.random.PRNGKey(200 + repeat)

        def objective(theta, estimate_key=key):
            return conditional_mmd(
                theta,
                jnp.asarray(generators),
                jnp.asarray(angles),
                jnp.asarray(targets),
                sigmas,
                256,
                4_096,
                estimate_key,
            ).loss

        losses.append(float(objective(params)))
        gradients.append(np.asarray(jax.grad(objective)(params)))

    loss_array = np.asarray(losses)
    gradient_array = np.asarray(gradients)
    loss_tolerance = max(
        5.0 * loss_array.std(ddof=1) / np.sqrt(loss_array.size),
        0.03,
    )
    gradient_tolerance = np.maximum(
        5.0 * gradient_array.std(axis=0, ddof=1) / np.sqrt(gradient_array.shape[0]),
        0.04,
    )
    assert loss_array.mean() == pytest.approx(exact_loss, abs=loss_tolerance)
    assert np.all(
        np.abs(gradient_array.mean(axis=0) - exact_gradient)
        <= gradient_tolerance,
    )
