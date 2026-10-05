"""Sampled conditional training with selectable Optax optimizers."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from iqp_cond import ConditionalModel
from iqp_cond.training import TrainingResult


def make_trainer(
    optimizer_settings: dict,
    diagnostic_sink: dict | None = None,
) -> Callable:
    """Build a runner-compatible sampled trainer.

    The adapter intentionally supports only the public trainer's step mask
    refresh path; optimizer state remains internal to the returned callable.
    """
    settings = dict(optimizer_settings)
    name = settings.get("name")
    if name not in {"adam", "schedule_free"}:
        raise ValueError("optimizer name must be 'adam' or 'schedule_free'")
    learning_rate = float(settings.get("learning_rate", 0.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    b1 = float(settings.get("b1", 0.9))
    if not math.isfinite(b1) or not 0 < b1 < 1:
        raise ValueError("invalid b1")
    if name == "adam":
        ratio = float(settings.get("final_learning_rate_ratio", 0.1))
        if not 0 <= b1 < 1 or not 0 < ratio <= 1:
            raise ValueError("invalid Adam settings")
    else:
        warmup = int(settings.get("warmup_steps", 0))
        if warmup < 0:
            raise ValueError("warmup_steps must be nonnegative")

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
        batches = np.random.default_rng(seed).integers(
            0,
            training_targets.shape[1],
            size=(config.steps, training_targets.shape[0], config.batch_size),
        )
        bands = np.random.default_rng(seed + 83_001).integers(
            0, len(sigmas), size=config.steps
        )
        conditions = np.random.default_rng(seed + 191_003).integers(
            0, model.encoder.n_conditions, size=config.steps
        )
        if name == "adam":
            schedule = optax.cosine_decay_schedule(
                learning_rate, config.steps, alpha=ratio
            )
            optimizer = optax.adam(schedule, b1=b1, b2=0.999, eps=1e-8)
        else:
            optimizer = optax.contrib.schedule_free_adamw(
                learning_rate,
                warmup_steps=warmup or None,
                b1=b1,
                b2=0.999,
                eps=1e-8,
                weight_decay=0.0,
                weight_lr_power=2.0,
                state_dtype=jnp.float64,
            )
        params = jnp.asarray(initial)
        state = optimizer.init(params)
        key = jax.random.PRNGKey(seed)

        def compile_step(sigma):
            def loss(theta, step_key, target_batch, encoder):
                return (
                    ConditionalModel(model.ansatz, encoder, n_visible=model.n_visible)
                    .mmd(
                        theta,
                        target_batch,
                        step_key,
                        sigmas=(sigma,),
                        n_ops=config.n_ops,
                        n_samples=config.mc_samples,
                        shared_samples=True,
                        gemm_dtype=config.gemm_dtype,
                        estimator_dtype=config.estimator_dtype,
                        sample_chunk=config.sample_chunk,
                    )
                    .loss
                )

            @jax.jit
            def step(theta, opt_state, step_key, target_batch, encoder):
                value, grad = jax.value_and_grad(loss)(
                    theta, step_key, target_batch, encoder
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
                gemm_dtype=config.gemm_dtype,
                estimator_dtype=config.estimator_dtype,
            ).loss

        train_losses = []
        validation_steps, validation_losses, scores = [], [], []
        best_params, best_step, best_score, best_validation = (
            params,
            0,
            float("inf"),
            float("inf"),
        )
        started = time.perf_counter()

        def checkpoint(step, theta):
            nonlocal best_params, best_step, best_score, best_validation
            validation = float(evaluate(theta))
            score = (
                validation
                if selection_metric is None
                else float(selection_metric(np.asarray(theta)))
            )
            if (
                not np.isfinite(validation)
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
            value, params, state = steps[int(bands[index])](
                params,
                state,
                jax.random.fold_in(key, index),
                batch,
                model.encoder.select(condition),
            )
            train_losses.append(value)
            if (index + 1) % config.validation_interval == 0:
                eval_params = (
                    params
                    if name == "adam"
                    else optax.contrib.schedule_free_eval_params(state, params)
                )
                checkpoint(index + 1, eval_params)
        final = (
            params
            if name == "adam"
            else optax.contrib.schedule_free_eval_params(state, params)
        )
        if not np.isfinite(np.asarray(final)).all():
            raise FloatingPointError("nonfinite final optimization")
        if diagnostic_sink is not None and name == "schedule_free":
            diagnostic_sink.update(
                final_training_params_y=np.asarray(params),
                final_eval_params_x=np.asarray(final),
                final_auxiliary_params_z=np.asarray(state.z),
                final_weight_sum=float(state.weight_sum),
                stepcount=config.steps,
                optimizer_step_count=int(state.step_count),
            )
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
