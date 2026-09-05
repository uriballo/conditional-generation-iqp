"""Training with independently resampled conditional angles."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ansatz import Ansatz
from .encoding import ANGLE_ENCODING
from .mmd import conditional_mmd
from .training import (
    TrainingConfig,
    TrainingResult,
    _batch_schedule,
    initialize_parameters,
)


def sample_interval_angles(
    baseline: np.ndarray,
    half_width: float | np.ndarray,
    seed: int | np.integer,
) -> np.ndarray:
    """Sample one angle independently per condition and qubit, deterministically."""
    baseline = np.asarray(baseline, dtype=np.float64)
    if baseline.ndim != 2:
        raise ValueError("baseline must have shape (n_conditions, n_qubits)")
    widths = np.asarray(half_width, dtype=np.float64)
    if widths.shape not in ((), (baseline.shape[1],), baseline.shape):
        raise ValueError(
            "half_width must be scalar, per-qubit, or match baseline shape"
        )
    if not np.all(np.isfinite(widths)) or np.any(widths < 0):
        raise ValueError("half_width must be finite and non-negative")
    rng = np.random.default_rng(int(seed))
    return baseline + rng.uniform(-widths, widths, baseline.shape)


sample_resampled_angles = sample_interval_angles


def train_with_resampled_angles(
    model_name: str,
    ansatz: Ansatz,
    angle_centers: np.ndarray,
    half_width: float | np.ndarray,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    seed: int,
    config: TrainingConfig,
    validation_members: int = 4,
) -> TrainingResult:
    """Train with fresh interval angles at every step and frozen validation angles.

    ``condition_angles`` in the result is the baseline/representative matrix;
    validation loss is the mean over ``validation_members`` frozen samples.
    """
    centers = np.asarray(angle_centers, dtype=np.float64)
    train_targets = np.asarray(training_targets, dtype=np.float64)
    valid_targets = np.asarray(validation_targets, dtype=np.float64)
    if centers.shape != (train_targets.shape[0], ansatz.n_qubits):
        raise ValueError("angle_centers must align with training conditions and qubits")
    if train_targets.ndim != 3 or valid_targets.ndim != 3:
        raise ValueError("targets must be rank three")
    if valid_targets.shape[0] != train_targets.shape[0]:
        raise ValueError("training and validation conditions must align")
    if validation_members < 1:
        raise ValueError("validation_members must be positive")

    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    params = initialize_parameters(ansatz.n_parameters, seed)
    optimizer = optax.adam(
        optax.cosine_decay_schedule(
            config.learning_rate,
            config.steps,
            alpha=config.final_learning_rate / config.learning_rate,
        )
    )
    state = optimizer.init(params)
    schedule = _batch_schedule(
        train_targets.shape[0], train_targets.shape[1], config, seed
    )
    train_angles = jnp.asarray(
        np.stack(
            [
                sample_interval_angles(centers, half_width, seed + 1_000_003 * (i + 1))
                for i in range(config.steps)
            ]
        )
    )
    validation_angles = jnp.asarray(
        np.stack(
            [
                sample_interval_angles(centers, half_width, seed + 2_000_003 * (i + 1))
                for i in range(validation_members)
            ]
        )
    )
    fixed_key = jax.random.PRNGKey(seed + 1_000_000)

    def loss(p, angles, targets, key, n_ops, n_samples):
        return conditional_mmd(
            p,
            generators,
            angles,
            targets,
            config.sigmas,
            n_ops,
            n_samples,
            key,
            ANGLE_ENCODING,
        ).loss

    def validation_loss(p):
        keys = jax.random.split(fixed_key, validation_members)
        return jnp.mean(
            jnp.stack(
                [
                    loss(
                        p,
                        validation_angles[i],
                        valid_targets,
                        keys[i],
                        config.validation_n_ops,
                        config.validation_n_iqp_samples,
                    )
                    for i in range(validation_members)
                ]
            )
        )

    value_grad = jax.jit(jax.value_and_grad(loss), static_argnums=(4, 5))
    evaluate = jax.jit(validation_loss)
    best_params, best_validation, best_step = params, float(evaluate(params)), 0
    train_steps, train_losses, validation_steps, validation_losses = (
        [],
        [],
        [0],
        [best_validation],
    )
    for step in range(config.steps):
        batch = np.stack(
            [train_targets[c, schedule[step, c]] for c in range(train_targets.shape[0])]
        )
        result, gradient = value_grad(
            params,
            train_angles[step],
            batch,
            jax.random.fold_in(jax.random.PRNGKey(seed), step),
            config.n_ops,
            config.n_iqp_samples,
        )
        updates, state = optimizer.update(gradient, state, params)
        params = optax.apply_updates(params, updates)
        train_steps.append(step + 1)
        train_losses.append(float(result))
        completed = step + 1
        if completed % config.validation_interval == 0:
            current = float(evaluate(params))
            validation_steps.append(completed)
            validation_losses.append(current)
            if current < best_validation:
                best_validation, best_params, best_step = current, params, completed
    return TrainingResult(
        model_name,
        ansatz.name,
        seed,
        np.asarray(ansatz.generators),
        centers,
        np.asarray(best_params),
        np.asarray(params),
        best_step,
        best_validation,
        np.asarray(train_steps),
        np.asarray(train_losses),
        np.asarray(validation_steps),
        np.asarray(validation_losses),
        ANGLE_ENCODING,
    )
