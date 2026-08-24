"""Finite-sample MMD training with exchange-symmetry parameter sharing."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ansatz import Ansatz
from .curie_weiss_protocol import CurieWeissFamily
from .mmd import conditional_mmd
from .training import TrainingConfig, _batch_schedule, initialize_parameters


@dataclass(frozen=True, slots=True)
class CurieWeissTrainingResult:
    generators: np.ndarray
    expansion: np.ndarray
    best_coefficients: np.ndarray
    final_coefficients: np.ndarray
    best_params: np.ndarray
    final_params: np.ndarray
    best_step: int
    best_validation_loss: float
    train_steps: np.ndarray
    train_losses: np.ndarray
    validation_steps: np.ndarray
    validation_losses: np.ndarray


def symmetry_expansion(generators: np.ndarray) -> np.ndarray:
    """Expand one one-body and one two-body coefficient to a complete bank."""
    generators = np.asarray(generators, dtype=np.int8)
    weights = generators.sum(axis=1)
    if generators.ndim != 2 or not set(weights) <= {1, 2}:
        raise ValueError("expected a one- and two-local generator bank")
    if np.count_nonzero(weights == 1) != generators.shape[1]:
        raise ValueError("generator bank is missing one-body generators")
    expected_pairs = generators.shape[1] * (generators.shape[1] - 1) // 2
    if np.count_nonzero(weights == 2) != expected_pairs:
        raise ValueError("generator bank is not complete two-local")
    expansion = np.zeros((len(generators), 2), dtype=np.float64)
    expansion[weights == 1, 0] = 1.0
    expansion[weights == 2, 1] = 1.0
    return expansion


def parameter_expansion(ansatz: Ansatz, family: CurieWeissFamily) -> np.ndarray:
    if family is CurieWeissFamily.UNRESTRICTED_CONDITIONED:
        return np.eye(ansatz.n_parameters, dtype=np.float64)
    return symmetry_expansion(np.asarray(ansatz.generators))


def train_curie_weiss(
    ansatz: Ansatz,
    family: CurieWeissFamily,
    condition_angles: np.ndarray,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    seed: int,
    config: TrainingConfig,
) -> CurieWeissTrainingResult:
    angles = np.asarray(condition_angles, dtype=np.float64)
    training_targets = np.asarray(training_targets, dtype=np.float64)
    validation_targets = np.asarray(validation_targets, dtype=np.float64)
    if training_targets.ndim != 3 or validation_targets.ndim != 3:
        raise ValueError("target samples must have condition/sample/qubit axes")
    if angles.shape != (training_targets.shape[0], ansatz.n_qubits):
        raise ValueError("condition angles and training targets do not align")
    if validation_targets.shape[0] != training_targets.shape[0]:
        raise ValueError("training and validation conditions do not align")
    expansion = jnp.asarray(parameter_expansion(ansatz, family))
    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    coefficients = initialize_parameters(expansion.shape[1], seed)
    schedule = optax.cosine_decay_schedule(
        config.learning_rate,
        config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(coefficients)
    batch_indices = _batch_schedule(
        training_targets.shape[0],
        training_targets.shape[1],
        config,
        seed,
    )
    random_key = jax.random.PRNGKey(seed)

    def loss(current: jnp.ndarray, targets: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        return conditional_mmd(
            expansion @ current,
            generators,
            jnp.asarray(angles),
            targets,
            config.sigmas,
            config.n_ops,
            config.n_iqp_samples,
            key,
        ).loss

    def validation(current: jnp.ndarray) -> jnp.ndarray:
        return conditional_mmd(
            expansion @ current,
            generators,
            jnp.asarray(angles),
            jnp.asarray(validation_targets),
            config.sigmas,
            config.validation_n_ops,
            config.validation_n_iqp_samples,
            jax.random.PRNGKey(seed + 1_000_000),
        ).loss

    value_grad = jax.jit(jax.value_and_grad(loss))
    validate = jax.jit(validation)
    train_steps, train_losses = [], []
    validation_steps, validation_losses = [0], []
    best_coefficients = coefficients
    best_step = 0
    best_validation = float(validate(coefficients))
    validation_losses.append(best_validation)
    for step in range(config.steps):
        indices = batch_indices[step]
        batch = np.stack(
            [
                training_targets[index, indices[index]]
                for index in range(training_targets.shape[0])
            ]
        )
        value, gradient = value_grad(
            coefficients,
            jnp.asarray(batch),
            jax.random.fold_in(random_key, step),
        )
        updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            coefficients,
        )
        coefficients = optax.apply_updates(coefficients, updates)
        train_steps.append(step + 1)
        train_losses.append(float(value))
        completed = step + 1
        if completed % config.validation_interval == 0:
            current = float(validate(coefficients))
            validation_steps.append(completed)
            validation_losses.append(current)
            if current < best_validation:
                best_validation = current
                best_coefficients = coefficients
                best_step = completed
    return CurieWeissTrainingResult(
        np.asarray(ansatz.generators),
        np.asarray(expansion),
        np.asarray(best_coefficients),
        np.asarray(coefficients),
        np.asarray(expansion @ best_coefficients),
        np.asarray(expansion @ coefficients),
        best_step,
        best_validation,
        np.asarray(train_steps),
        np.asarray(train_losses),
        np.asarray(validation_steps),
        np.asarray(validation_losses),
    )
