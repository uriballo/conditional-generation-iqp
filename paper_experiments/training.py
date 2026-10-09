"""Sampled conditional training with the manuscript Adam optimizer."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from iqp_cond import ConditionalModel
from iqp_cond.operators import OperatorBank, sample_nonidentity_masks_jax
from iqp_cond.training import TrainingResult


def make_trainer(
    optimizer_settings: dict,
    *,
    shared_validation_samples: bool = False,
) -> Callable:
    """Build a runner-compatible sampled trainer.

    The adapter intentionally supports only the public trainer's step mask
    refresh path; optimizer state remains internal to the returned callable.
    """
    settings = dict(optimizer_settings)
    shared_validation_samples = bool(
        settings.pop("shared_validation_samples", shared_validation_samples)
    )
    name = settings.get("name")
    if name != "adam":
        raise ValueError("unsupported optimizer name")
    learning_rate = float(settings.get("learning_rate", 0.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    b1 = float(settings.get("b1", 0.9))
    if not math.isfinite(b1) or not 0 < b1 < 1:
        raise ValueError("invalid b1")
    ratio = float(settings.get("final_learning_rate_ratio", 0.1))
    if not 0 < ratio <= 1:
        raise ValueError("invalid Adam settings")
    b2 = float(settings.get("b2", 0.999))
    eps = float(settings.get("eps", 1e-8))
    if not math.isfinite(b2) or not 0 < b2 < 1:
        raise ValueError("invalid b2")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid eps")
    rng_seeds = settings.get("rng_seeds")
    if rng_seeds is not None:
        if set(rng_seeds) != {"batches", "bands", "conditions", "masks", "reference"}:
            raise ValueError("rng_seeds must have exactly the required keys")
        if any(
            not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2**32
            for value in rng_seeds.values()
        ):
            raise ValueError("rng_seeds values must be uint32")

    def train(
        model,
        training_targets,
        validation_targets,
        *,
        seed,
        config,
        initial_params=None,
        checkpoint_callback=None,
        selection_metric=None,
    ):
        if config.mask_refresh != "step":
            raise ValueError("sampled optimizer adapter requires step mask refresh")
        if initial_params is None:
            raise ValueError("sampled optimizer adapter requires initial_params")
        training_targets = np.asarray(training_targets, dtype=np.float64)
        validation_targets = np.asarray(validation_targets, dtype=np.float64)
        initial = np.asarray(initial_params, dtype=np.float64)
        if (
            initial.shape != (model.ansatz.n_parameters,)
            or not np.isfinite(initial).all()
        ):
            raise ValueError("invalid initial parameters")
        if training_targets.ndim != 3 or validation_targets.ndim != 3:
            raise ValueError("targets must be rank three")
        if training_targets.shape[0] != model.encoder.n_conditions:
            raise ValueError("training condition count differs")
        for targets in (training_targets, validation_targets):
            if (
                targets.shape[0] != model.encoder.n_conditions
                or targets.shape[2] != model.visible
                or targets.shape[1] < 2
            ):
                raise ValueError("invalid target dimensions")
            if not np.isfinite(targets).all() or not np.isin(targets, [0, 1]).all():
                raise ValueError("targets must be finite binary strings")
        sigmas = tuple(config.resolved_sigmas(model.visible))
        seed_batches = int(rng_seeds["batches"]) if rng_seeds is not None else seed
        seed_bands = int(rng_seeds["bands"]) if rng_seeds is not None else seed + 83_001
        seed_conditions = (
            int(rng_seeds["conditions"]) if rng_seeds is not None else seed + 191_003
        )
        batches = np.random.default_rng(seed_batches).integers(
            0,
            training_targets.shape[1],
            size=(config.steps, training_targets.shape[0], config.batch_size),
        )
        bands = np.random.default_rng(seed_bands).integers(
            0, len(sigmas), size=config.steps
        )
        conditions = np.random.default_rng(seed_conditions).integers(
            0, model.encoder.n_conditions, size=config.steps
        )
        schedule = optax.cosine_decay_schedule(learning_rate, config.steps, alpha=ratio)
        optimizer = optax.adam(schedule, b1=b1, b2=b2, eps=eps)
        params = jnp.asarray(initial)
        state = optimizer.init(params)
        mask_key = jax.random.PRNGKey(
            int(rng_seeds["masks"]) if rng_seeds is not None else seed
        )
        reference_key = jax.random.PRNGKey(
            int(rng_seeds["reference"]) if rng_seeds is not None else seed
        )

        def compile_step(sigma):
            def loss(theta, step_key, target_batch, encoder, bank):
                kwargs = dict(
                    n_samples=config.mc_samples,
                    shared_samples=True,
                    gemm_dtype=config.gemm_dtype,
                    estimator_dtype=config.estimator_dtype,
                    sample_chunk=config.sample_chunk,
                )
                if rng_seeds is None:
                    kwargs.update(sigmas=(sigma,), n_ops=config.n_ops)
                else:
                    kwargs["bank"] = bank
                return (
                    ConditionalModel(model.ansatz, encoder, n_visible=model.n_visible)
                    .mmd(
                        theta,
                        target_batch,
                        step_key,
                        **kwargs,
                    )
                    .loss
                )

            @jax.jit
            def step(theta, opt_state, step_key, target_batch, encoder, mask_step_key):
                bank = None
                if rng_seeds is not None:
                    masks, weights = sample_nonidentity_masks_jax(
                        mask_step_key, config.n_ops, model.visible, sigma
                    )
                    bank = OperatorBank(
                        sigmas=(sigma,),
                        masks=masks[None, None],
                        weights=weights[None, None],
                        n_ops=config.n_ops,
                        n_visible=model.visible,
                        n_masks=config.n_ops,
                    )
                value, grad = jax.value_and_grad(loss)(
                    theta, step_key, target_batch, encoder, bank
                )
                updates, opt_state = optimizer.update(grad, opt_state, theta)
                return value, optax.apply_updates(theta, updates), opt_state

            return step

        steps = tuple(compile_step(sigma) for sigma in sigmas)

        @jax.jit
        def evaluate(theta):
            return model.mmd(
                theta,
                jnp.asarray(validation_targets),
                jax.random.PRNGKey(seed + 1_000_000),
                sigmas=sigmas,
                n_ops=config.validation_n_ops,
                n_samples=config.validation_mc_samples,
                shared_samples=shared_validation_samples,
                gemm_dtype=config.gemm_dtype,
                estimator_dtype=config.estimator_dtype,
            ).loss

        compute_validation = bool(settings.get("compute_validation_mmd", True))
        if not compute_validation and selection_metric is None:
            raise ValueError(
                "selection_metric is required when validation MMD is disabled"
            )
        train_losses = []
        validation_steps, validation_losses, scores = [], [], []
        best_params, best_step, best_score, best_validation = (
            params,
            0,
            float("inf"),
            float("inf") if compute_validation else float("nan"),
        )
        started = time.perf_counter()

        def checkpoint(step, theta):
            nonlocal best_params, best_step, best_score, best_validation
            validation = float(evaluate(theta)) if compute_validation else float("nan")
            score = (
                validation
                if selection_metric is None
                else float(selection_metric(np.asarray(theta)))
            )
            if (
                (compute_validation and not np.isfinite(validation))
                or not np.isfinite(score)
                or not np.isfinite(np.asarray(theta)).all()
            ):
                raise FloatingPointError("nonfinite training checkpoint")
            validation_steps.append(step)
            validation_losses.append(validation)
            scores.append(score)
            if score < best_score:
                best_params, best_step, best_score, best_validation = (
                    np.asarray(theta).copy(),
                    step,
                    score,
                    validation,
                )
            if checkpoint_callback is not None:
                checkpoint_callback(step, np.asarray(theta).copy())
            if step % 500 == 0:
                print(
                    f"sampled {name} step {step}: "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

        checkpoint(0, params)
        for index in range(config.steps):
            condition = int(conditions[index])
            batch = jnp.asarray(
                training_targets[condition : condition + 1, batches[index, condition]]
            )
            step_index = index
            value, params, state = steps[int(bands[index])](
                params,
                state,
                jax.random.fold_in(reference_key, step_index),
                batch,
                model.encoder.select(condition),
                jax.random.fold_in(mask_key, step_index),
            )
            train_losses.append(value)
            if (
                index + 1
            ) % config.validation_interval == 0 or index + 1 == config.steps:
                checkpoint(index + 1, params)
        final = params
        if (
            not np.isfinite(np.asarray(final)).all()
            or not np.isfinite(np.asarray(train_losses)).all()
        ):
            raise FloatingPointError("nonfinite final optimization")
        return TrainingResult(
            seed=seed,
            best_params=np.asarray(best_params),
            final_params=np.asarray(final),
            best_step=best_step,
            best_validation_loss=best_validation,
            best_selection_score=best_score,
            train_steps=np.arange(1, config.steps + 1),
            train_losses=np.asarray(jnp.stack(train_losses)),
            validation_steps=np.asarray(validation_steps),
            validation_losses=np.asarray(validation_losses),
            selection_scores=np.asarray(scores),
        )

    return train
