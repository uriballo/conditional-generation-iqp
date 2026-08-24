"""Reproducible training for the agreed conditional IQP experiment."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ansatz import Ansatz
from .encoding import ANGLE_ENCODING
from .exact import generator_characters
from .mmd import conditional_mmd, exact_conditional_mmd

EXACT_GRADIENT_STRATEGIES = ("mean", "normalized", "pcgrad")


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
    exact_gradient_strategy: str = "mean"


def combine_condition_gradients(
    condition_gradients: jnp.ndarray | np.ndarray,
    strategy: str,
) -> jnp.ndarray:
    """Combine one exact-loss gradient per Condition.

    ``normalized`` gives every nonzero Condition gradient unit norm before
    averaging. ``pcgrad`` additionally removes negative pairwise projections
    in a deterministic cyclic order. The standard ``mean`` path is unchanged.
    """
    if strategy not in EXACT_GRADIENT_STRATEGIES:
        raise ValueError(
            f"unknown exact gradient strategy: {strategy!r}; "
            f"expected one of {EXACT_GRADIENT_STRATEGIES}",
        )
    gradients = jnp.asarray(condition_gradients, dtype=jnp.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 1:
        raise ValueError("condition_gradients must have shape (conditions, parameters)")
    if strategy == "mean":
        return jnp.mean(gradients, axis=0)

    norms = jnp.linalg.norm(gradients, axis=1, keepdims=True)
    normalized = jnp.where(norms > 0.0, gradients / norms, 0.0)
    if strategy == "normalized":
        return jnp.mean(normalized, axis=0)

    projected = normalized
    epsilon = jnp.finfo(normalized.dtype).eps
    n_conditions = normalized.shape[0]
    for condition in range(n_conditions):
        current = projected[condition]
        for offset in range(1, n_conditions):
            other = normalized[(condition + offset) % n_conditions]
            dot = jnp.vdot(current, other).real
            denominator = jnp.vdot(other, other).real
            correction = dot / jnp.maximum(denominator, epsilon) * other
            current = jnp.where(dot < 0.0, current - correction, current)
        projected = projected.at[condition].set(current)
    return jnp.mean(projected, axis=0)


@dataclass(frozen=True)
class TrainingResult:
    """Best parameters and scalar training history."""

    model_name: str
    topology: str
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
    model_name: str,
    ansatz: Ansatz,
    condition_angles: np.ndarray,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
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
    params = initialize_parameters(ansatz.n_parameters, seed)
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
        model_name=model_name,
        topology=ansatz.name,
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


def train_exact(
    model_name: str,
    ansatz: Ansatz,
    condition_angles: np.ndarray,
    target_fouriers: np.ndarray,
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
) -> TrainingResult:
    """Train one shared model with the zero-variance Conditional MMD.

    ``target_fouriers`` are the exact full Fourier coefficients of the
    target laws, shape (n_conditions, 2**n).  The loss is deterministic, so
    training and validation coincide and the best checkpoint minimizes the
    same exact objective.
    """
    condition_angles = np.asarray(condition_angles, dtype=np.float64)
    target_fouriers = np.asarray(target_fouriers, dtype=np.float64)
    if target_fouriers.ndim != 2:
        raise ValueError("target_fouriers must have shape (n_conditions, 2**n)")
    if condition_angles.shape != (target_fouriers.shape[0], ansatz.n_qubits):
        raise ValueError("condition_angles must align with target_fouriers")
    if config.exact_gradient_strategy not in EXACT_GRADIENT_STRATEGIES:
        raise ValueError(
            "exact_gradient_strategy must be one of "
            f"{EXACT_GRADIENT_STRATEGIES}",
        )

    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    angles = jnp.asarray(condition_angles, dtype=jnp.float64)
    params = initialize_parameters(ansatz.n_parameters, seed)
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(params)
    exact_targets = jnp.asarray(target_fouriers, dtype=jnp.float64)
    characters = jnp.asarray(
        generator_characters(ansatz.generators),
        dtype=jnp.float64,
    )

    def exact_loss(current_params: jnp.ndarray) -> jnp.ndarray:
        return exact_conditional_mmd(
            current_params,
            generators,
            angles,
            exact_targets,
            config.sigmas,
            encoding,
            characters,
        ).loss

    value_and_gradient = jax.jit(jax.value_and_grad(exact_loss))

    def per_condition_loss_with_aux(
        current_params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        per_condition = exact_conditional_mmd(
            current_params,
            generators,
            angles,
            exact_targets,
            config.sigmas,
            encoding,
            characters,
        ).per_condition
        return per_condition, per_condition

    condition_jacobian_and_losses = jax.jit(
        jax.jacrev(per_condition_loss_with_aux, has_aux=True),
    )
    evaluate_validation = jax.jit(exact_loss)

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
        if config.exact_gradient_strategy == "mean":
            loss, gradient = value_and_gradient(params)
        else:
            condition_gradients, condition_losses = condition_jacobian_and_losses(
                params,
            )
            loss = jnp.mean(condition_losses)
            gradient = combine_condition_gradients(
                condition_gradients,
                config.exact_gradient_strategy,
            )
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
        model_name=model_name,
        topology=ansatz.name,
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
