"""Small, reproducible training driver for the exact-TVD campaigns.

The module deliberately keeps the simulator out of the driver: callers provide
the differentiable loss.  This makes the batching, checkpoint, and optimizer
semantics identical for all three data sets and encodings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

OptimizerName = Literal["adam_cosine", "schedule_free_adamw"]


@dataclass(frozen=True)
class CampaignTrainingConfig:
    steps: int = 2_000
    batch_size: int = 100
    n_samples: int = 400
    validation_interval: int = 25
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-3
    optimizer: OptimizerName = "adam_cosine"
    warmup_steps: int = 100
    b1: float = 0.9
    b2: float = 0.999


@dataclass(frozen=True)
class Checkpoint:
    step: int
    params: np.ndarray
    validation_mmd: float
    reason: str


@dataclass(frozen=True)
class CampaignTrainingResult:
    final_params: np.ndarray
    best_params: np.ndarray
    best_step: int
    best_validation_mmd: float
    checkpoints: tuple[Checkpoint, ...]
    train_steps: np.ndarray
    train_losses: np.ndarray
    validation_steps: np.ndarray
    validation_losses: np.ndarray
    batches: np.ndarray


def shuffled_batches(
    n_conditions: int,
    n_examples: int,
    config: CampaignTrainingConfig,
    seed: int,
) -> np.ndarray:
    """Return shuffled epochs, truncating the final epoch to the update budget."""
    if n_conditions < 1 or n_examples < 1 or config.batch_size < 1:
        raise ValueError("condition, example, and batch sizes must be positive")
    if n_examples % config.batch_size:
        raise ValueError("n_examples must be divisible by batch_size")
    batches_per_epoch = n_examples // config.batch_size
    epochs = (config.steps + batches_per_epoch - 1) // batches_per_epoch
    rng = np.random.default_rng(seed)
    result = np.empty((config.steps, n_conditions, config.batch_size), dtype=np.int32)
    row = 0
    for _ in range(epochs):
        take = min(batches_per_epoch, config.steps - row)
        for condition in range(n_conditions):
            permutation = rng.permutation(n_examples)
            result[row : row + take, condition] = permutation.reshape(
                batches_per_epoch, config.batch_size
            )[:take]
        row += take
    return result


def sample_nonidentity_masks(
    key: jax.Array,
    n_ops: int,
    n_qubits: int,
    sigma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample nonidentity masks and return compressed unbiased weights.

    The identity Fourier coefficient has exactly zero population-MMD
    contribution.  Sampling from ``q(mask | mask != 0)`` therefore needs only
    the analytic factor ``1 - q(0)``.  Duplicate masks are represented once
    with their exact empirical multiplicity.
    """
    if n_ops < 1 or n_qubits < 1 or sigma <= 0:
        raise ValueError("n_ops, n_qubits, and sigma must be positive")
    p = float((1.0 - np.exp(-1.0 / (2.0 * sigma**2))) / 2.0)
    q_nonidentity = 1.0 - (1.0 - p) ** n_qubits
    # This function runs once while constructing a run, outside any JIT.  A
    # local NumPy generator makes rejection sampling simple and guarantees the
    # requested number of nonidentity draws even for the low-bodyness kernel.
    seed_material = np.asarray(key, dtype=np.uint32).tobytes()
    seed = int.from_bytes(seed_material[:8], "little")
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = n_ops
    while remaining:
        proposal = rng.binomial(1, p, size=(max(remaining * 2, 64), n_qubits))
        accepted = proposal[np.any(proposal, axis=1)]
        take = min(remaining, accepted.shape[0])
        if take:
            chunks.append(accepted[:take].astype(np.int32, copy=False))
            remaining -= take
    raw = np.concatenate(chunks, axis=0)
    unique, counts = np.unique(raw, axis=0, return_counts=True)
    weights = q_nonidentity * counts.astype(np.float64) / n_ops
    return jnp.asarray(unique), jnp.asarray(weights)


def _optimizer(config: CampaignTrainingConfig) -> optax.GradientTransformation:
    if config.optimizer == "adam_cosine":
        schedule = optax.cosine_decay_schedule(
            config.learning_rate,
            config.steps,
            config.final_learning_rate / config.learning_rate,
        )
        return optax.adam(schedule)
    if config.optimizer == "schedule_free_adamw":
        return optax.contrib.schedule_free_adamw(
            config.learning_rate,
            config.warmup_steps,
            config.b1,
            config.b2,
            weight_decay=0.0,
        )
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def train_campaign(
    params: np.ndarray,
    training_targets: np.ndarray,
    validation_mmd: Callable[[jnp.ndarray], jnp.ndarray],
    training_loss: Callable[[jnp.ndarray, jnp.ndarray, jax.Array], jnp.ndarray],
    seed: int,
    config: CampaignTrainingConfig,
    *,
    include_initial_checkpoint: bool = False,
) -> CampaignTrainingResult:
    """Train with fixed batches while keeping ``training_loss`` differentiable."""
    targets = np.asarray(training_targets)
    if targets.ndim < 3 or targets.shape[1] < config.batch_size:
        raise ValueError("training_targets must have shape (conditions, examples, ...)")
    batches = shuffled_batches(targets.shape[0], targets.shape[1], config, seed)
    current = jnp.asarray(params, dtype=jnp.float64)
    optimizer = _optimizer(config)
    state = optimizer.init(current)
    value_grad = jax.jit(jax.value_and_grad(training_loss))
    checkpoints: list[Checkpoint] = []
    val_steps, val_losses = [0], [float(validation_mmd(current))]
    if include_initial_checkpoint:
        initial_value = val_losses[0]
        checkpoints.append(Checkpoint(0, np.asarray(current), initial_value, "initial"))
    train_steps: list[int] = []
    train_losses: list[float] = []
    best = float("inf")
    best_params = current
    best_step = -1
    for i in range(config.steps):
        batch = jnp.asarray(
            [targets[c, batches[i, c]] for c in range(targets.shape[0])]
        )
        loss, grad = value_grad(
            current, batch, jax.random.fold_in(jax.random.PRNGKey(seed), i)
        )
        updates, state = optimizer.update(grad, state, current)
        current = optax.apply_updates(current, updates)
        step = i + 1
        train_steps.append(step)
        train_losses.append(float(loss))
        if step % config.validation_interval == 0 or step in {
            500,
            1000,
            1500,
            config.steps,
        }:
            evaluated = (
                optax.contrib.schedule_free_eval_params(state, current)
                if config.optimizer == "schedule_free_adamw"
                else current
            )
            value = float(validation_mmd(evaluated))
            val_steps.append(step)
            val_losses.append(value)
            if step in {500, 1000, 1500, config.steps}:
                checkpoints.append(
                    Checkpoint(step, np.asarray(evaluated), value, "scheduled")
                )
            if value < best:
                best, best_params, best_step = value, evaluated, step
    if best_step < 1:
        raise RuntimeError("training produced no validation checkpoint")
    checkpoints.append(
        Checkpoint(best_step, np.asarray(best_params), best, "lowest_validation_mmd")
    )
    return CampaignTrainingResult(
        np.asarray(current),
        np.asarray(best_params),
        best_step,
        best,
        tuple(checkpoints),
        np.asarray(train_steps),
        np.asarray(train_losses),
        np.asarray(val_steps),
        np.asarray(val_losses),
        batches,
    )
