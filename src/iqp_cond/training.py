"""Reproducible training for the agreed conditional IQP experiment."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ansatz import Ansatz
from .encoding import ANGLE_ENCODING
from .mmd import conditional_mmd


@dataclass(frozen=True)
class TrainingConfig:
    """Shared optimization and Monte Carlo settings."""

    steps: int = 2_000
    batch_size: int = 256
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-3
    validation_interval: int = 25
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)
    n_ops: int = 64
    n_iqp_samples: int = 8_192
    validation_n_ops: int = 256
    validation_n_iqp_samples: int = 16_384


@dataclass(frozen=True)
class TrainingResult:
    """Best parameters and scalar training history."""

    seed: int
    generators: np.ndarray
    condition_angles: np.ndarray
    best_params: np.ndarray
    final_params: np.ndarray
    best_step: int
    best_validation_loss: float
    train_steps: np.ndarray
    train_losses: np.ndarray
    validation_steps: np.ndarray
    validation_losses: np.ndarray
    encoding: str = ANGLE_ENCODING


def initialize_parameters(n_parameters: int, seed: int) -> jnp.ndarray:
    """Draw generator-count normalized Gaussian parameters."""
    if n_parameters < 1:
        raise ValueError("n_parameters must be positive")
    rng = np.random.default_rng(seed + 10_000)
    scale = 0.1 / np.sqrt(n_parameters)
    return jnp.asarray(rng.normal(0.0, scale, n_parameters), dtype=jnp.float64)


def _batch_schedule(
    n_conditions: int,
    n_examples: int,
    config: TrainingConfig,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        n_examples,
        size=(config.steps, n_conditions, config.batch_size),
    )


def train(
    ansatz: Ansatz,
    condition_angles: np.ndarray,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
    *,
    initial_params: np.ndarray | None = None,
) -> TrainingResult:
    """Train one shared conditional model with fixed validation randomness."""
    condition_angles = np.asarray(condition_angles, dtype=np.float64)
    training_targets = np.asarray(training_targets, dtype=np.float64)
    validation_targets = np.asarray(validation_targets, dtype=np.float64)
    if training_targets.ndim != 3:
        raise ValueError("training_targets must be rank three")
    expected_angles_shape = (training_targets.shape[0], ansatz.n_qubits)
    if condition_angles.shape != expected_angles_shape:
        raise ValueError(f"condition_angles must have shape {expected_angles_shape}")
    if validation_targets.shape[0] != training_targets.shape[0]:
        raise ValueError("training and validation conditions must align")
    if validation_targets.shape[2] > ansatz.n_qubits:
        raise ValueError(
            "validation target qubit count cannot exceed ansatz qubit count",
        )

    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    angles = jnp.asarray(condition_angles, dtype=jnp.float64)
    if initial_params is None:
        params = initialize_parameters(ansatz.n_parameters, seed)
    else:
        candidate = np.asarray(initial_params, dtype=np.float64)
        if candidate.shape != (ansatz.n_parameters,):
            raise ValueError(
                f"initial_params must have shape {(ansatz.n_parameters,)}",
            )
        if not np.isfinite(candidate).all():
            raise ValueError("initial_params must contain only finite values")
        params = jnp.asarray(candidate, dtype=jnp.float64).copy()
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(params)
    batch_indices = _batch_schedule(
        training_targets.shape[0],
        training_targets.shape[1],
        config,
        seed,
    )
    random_key = jax.random.PRNGKey(seed)

    def training_loss(
        current_params: jnp.ndarray,
        target_batch: jnp.ndarray,
        step_key: jax.Array,
    ) -> jnp.ndarray:
        return conditional_mmd(
            current_params,
            generators,
            angles,
            target_batch,
            config.sigmas,
            config.n_ops,
            config.n_iqp_samples,
            step_key,
            encoding,
        ).loss

    def validation_loss(current_params: jnp.ndarray) -> jnp.ndarray:
        return conditional_mmd(
            current_params,
            generators,
            angles,
            jnp.asarray(validation_targets),
            config.sigmas,
            config.validation_n_ops,
            config.validation_n_iqp_samples,
            jax.random.PRNGKey(seed + 1_000_000),
            encoding,
        ).loss

    value_and_gradient = jax.jit(jax.value_and_grad(training_loss))
    evaluate_validation = jax.jit(validation_loss)

    train_steps = []
    train_losses = []
    validation_steps = []
    validation_losses = []
    best_params = params
    best_step = 0
    best_validation = float(evaluate_validation(params))

    validation_steps.append(0)
    validation_losses.append(best_validation)
    for step in range(config.steps):
        indices = batch_indices[step]
        batch = np.stack(
            [
                training_targets[condition, indices[condition]]
                for condition in range(training_targets.shape[0])
            ],
        )
        step_key = jax.random.fold_in(random_key, step)
        loss, gradient = value_and_gradient(params, jnp.asarray(batch), step_key)
        updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        train_steps.append(step + 1)
        train_losses.append(float(loss))

        completed_step = step + 1
        if completed_step % config.validation_interval == 0:
            current_validation = float(evaluate_validation(params))
            validation_steps.append(completed_step)
            validation_losses.append(current_validation)
            if current_validation < best_validation:
                best_validation = current_validation
                best_params = params
                best_step = completed_step

    return TrainingResult(
        seed=seed,
        generators=np.asarray(ansatz.generators),
        condition_angles=condition_angles,
        best_params=np.asarray(best_params),
        final_params=np.asarray(params),
        best_step=best_step,
        best_validation_loss=best_validation,
        train_steps=np.asarray(train_steps),
        train_losses=np.asarray(train_losses),
        validation_steps=np.asarray(validation_steps),
        validation_losses=np.asarray(validation_losses),
        encoding=encoding,
    )

