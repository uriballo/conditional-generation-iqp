"""Reusable sampled conditional-MMD loss construction."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .exact_tvd_campaign import expected_bodyness_sigmas
from .exact_tvd_runner import _derived_seed
from .exact_tvd_training import sample_nonidentity_masks
from .mmd import _target_terms
from .simulator import _estimate_pauli_z_unchecked


def sampled_losses(
    g: np.ndarray,
    angles: np.ndarray,
    validation: np.ndarray,
    seed: int,
    encoding: str,
    n_samples: int,
    *,
    n_ops: int = 512,
    sigmas: tuple[float, ...] | None = None,
) -> tuple[Any, Any, dict[str, np.ndarray]]:
    """Build sampled training and fixed-validation conditional MMD losses.

    The operator banks and validation key are deterministic functions of the
    supplied seed.  If the model has ancillas, visible masks are padded with
    zero columns before being passed to the simulator.
    """
    g = np.asarray(g)
    angles = np.asarray(angles)
    validation = np.asarray(validation)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if n_ops < 1:
        raise ValueError("n_ops must be positive")
    if validation.ndim != 3:
        raise ValueError(
            "validation must have shape (conditions, examples, visible qubits)"
        )
    if angles.ndim != 2 or g.ndim != 2 or angles.shape[1] != g.shape[1]:
        raise ValueError("angles must cover every model qubit")
    n_conditions = angles.shape[0]
    if n_conditions < 1 or validation.shape[0] != n_conditions:
        raise ValueError("angles and validation must contain the same conditions")
    if validation.shape[1] < 2:
        raise ValueError(
            "validation must contain at least two data points per condition"
        )
    n_visible = validation.shape[-1]
    if angles.ndim != 2 or g.ndim != 2 or g.shape[1] < n_visible:
        raise ValueError("g, angles, and validation have incompatible dimensions")
    sigmas = expected_bodyness_sigmas(n_visible) if sigmas is None else tuple(sigmas)
    if not sigmas:
        raise ValueError("sigmas must not be empty")

    banks: list[tuple[np.ndarray, np.ndarray, int]] = []
    for band, sigma in enumerate(sigmas):
        for condition in range(n_conditions):
            key = jax.random.PRNGKey(_derived_seed(seed, band, condition) & (2**32 - 1))
            masks, weights = sample_nonidentity_masks(key, n_ops, n_visible, sigma)
            banks.append((np.asarray(masks), np.asarray(weights), condition))

    width = max(len(masks) for masks, _, _ in banks)
    masks = jnp.asarray(
        np.stack(
            [
                np.pad(values, ((0, width - len(values)), (0, 0)))
                for values, _, _ in banks
            ]
        )
    )
    weights = jnp.asarray(
        np.stack([np.pad(values, (0, width - len(values))) for _, values, _ in banks])
    )
    conditions = jnp.asarray([condition for _, _, condition in banks])
    generators = jnp.asarray(g)
    angles_jax = jnp.asarray(angles)
    validation_jax = jnp.asarray(validation)
    n_ancillas = g.shape[1] - n_visible

    def estimate(
        params: jnp.ndarray, targets: jnp.ndarray, key: jax.Array
    ) -> jnp.ndarray:
        targets = jnp.asarray(targets)
        keys = jax.random.split(key, len(banks))

        def one(
            args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array],
        ) -> jnp.ndarray:
            ops, weight, condition, sample_key = args
            model_ops = jnp.pad(ops, ((0, 0), (0, n_ancillas))) if n_ancillas else ops
            model = _estimate_pauli_z_unchecked(
                params,
                generators,
                model_ops,
                angles_jax[condition][None, :],
                sample_key,
                n_samples,
                encoding,
            )
            means, target_squares = _target_terms(targets[condition][None, :, :], ops)
            sums = jnp.sum(model.sample_values, axis=1)
            model_squares = (sums**2 - jnp.sum(model.sample_values**2, axis=1)) / (
                n_samples * (n_samples - 1)
            )
            terms = model_squares - 2 * model.values * means + target_squares
            return jnp.sum(weight * terms[0])

        return jnp.mean(jax.lax.map(one, (masks, weights, conditions, keys)))

    fixed = validation_jax
    validation_key = jax.random.PRNGKey(_derived_seed(seed, "validation") & (2**32 - 1))
    return (
        estimate,
        jax.jit(lambda params: estimate(params, fixed, validation_key)),
        {"masks": np.asarray(masks), "weights": np.asarray(weights)},
    )
