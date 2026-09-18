"""Do the step-level review findings survive a full training pipeline?

Runs the shipped trainer with per-step masks (``mask_refresh="step"``) and with
one retained bank (``"cache"``), plus a loop that mirrors the trainer but trains
on masks drawn inside JAX (``iqp_cond.mmd`` with ``sigmas``/``n_ops``), all on
the same data, seed, schedule, and optimizer.  Reports wall clock, loss
trajectories, and a common independent score, then verifies that the returned
``best_params`` reproduce the returned ``best_validation_loss`` against the
trainer's own validation bank.

    .venv/bin/python \
        experiments/library_perf/pipeline.py
"""

from __future__ import annotations

import dataclasses
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from common import problem, save
from iqp_cond.operators import operator_bank

import iqp_cond as iqc


def _batches(targets: np.ndarray, config: iqc.TrainingConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        targets.shape[1],
        size=(config.steps, targets.shape[0], config.batch_size),
    )


def train_with_jax_masks(
    model: iqc.ConditionalModel,
    training_targets: np.ndarray,
    config: iqc.TrainingConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The trainer loop with fresh in-JAX masks as the objective."""
    sigmas = config.resolved_sigmas(model.visible)
    params = iqc.random_parameters(model.ansatz.n_parameters, seed)
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(params)
    indices = _batches(training_targets, config, seed)
    root_key = jax.random.PRNGKey(seed)

    def loss(current_params, target_batch, step_key):
        return model.mmd(
            current_params,
            target_batch,
            step_key,
            sigmas=sigmas,
            n_ops=config.n_ops,
            n_samples=config.mc_samples,
            shared_samples=True,
        ).loss

    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    start = time.perf_counter()
    losses = []
    for step in range(config.steps):
        batch = np.stack(
            [
                training_targets[condition, indices[step, condition]]
                for condition in range(training_targets.shape[0])
            ],
        )
        value, gradient = value_and_gradient(
            params,
            jnp.asarray(batch),
            jax.random.fold_in(root_key, step),
        )
        updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        losses.append(value)
    wall = time.perf_counter() - start
    return np.asarray(params), np.asarray(jnp.stack(losses)), wall


def main() -> None:
    config = iqc.TrainingConfig(steps=500, validation_interval=25)
    sigmas = config.resolved_sigmas(12)
    seeds = (0, 1)
    report: dict = {"config": {"steps": config.steps, "seeds": list(seeds)}}
    for seed in seeds:
        ansatz, encoders, targets = problem(
            n_visible=12,
            n_conditions=5,
            n_targets=2048,
            seed=seed,
        )
        model = iqc.ConditionalModel(ansatz, encoders)
        validation = targets[:, :512]
        common_bank = operator_bank(
            sigmas,
            config.validation_n_ops,
            encoders.n_conditions,
            targets.shape[2],
            seed=seed + 777,
        )
        common_key = jax.random.PRNGKey(seed + 888)

        def independent_score(
            current_params: np.ndarray,
            model=model,
            validation=validation,
            common_bank=common_bank,
            common_key=common_key,
        ) -> float:
            return float(
                model.mmd(
                    current_params,
                    jnp.asarray(validation),
                    common_key,
                    bank=common_bank,
                    n_samples=config.validation_mc_samples,
                ).loss,
            )

        run: dict = {}
        results: dict = {}
        for name, refresh in (("step", iqc.STEP), ("cache", iqc.CACHE)):
            start = time.perf_counter()
            result = iqc.train(
                model,
                targets,
                validation,
                seed=seed,
                config=dataclasses.replace(config, mask_refresh=refresh),
            )
            results[name] = result
            run[name] = {
                "wall_s": time.perf_counter() - start,
                "best_validation_loss": float(result.best_validation_loss),
                "independent_score": independent_score(result.best_params),
                "final_train_loss": float(np.mean(result.train_losses[-20:])),
            }

        jax_params, jax_losses, jax_wall = train_with_jax_masks(
            model,
            targets,
            config,
            seed,
        )
        run["jax"] = {
            "wall_s": jax_wall,
            "independent_score": independent_score(jax_params),
            "final_train_loss": float(np.mean(np.asarray(jax_losses)[-20:])),
        }

        # Pipeline consistency: the reported best loss must be the objective at
        # best_params on the trainer's own validation bank and key.
        expected_bank = operator_bank(
            sigmas,
            config.validation_n_ops,
            encoders.n_conditions,
            targets.shape[2],
            seed + 1,
        )
        expected_key = jax.random.PRNGKey(seed + 1_000_000)
        for name in ("step", "cache"):
            reproduced = float(
                model.mmd(
                    results[name].best_params,
                    jnp.asarray(validation),
                    expected_key,
                    bank=expected_bank,
                    n_samples=config.validation_mc_samples,
                ).loss,
            )
            run[name]["checkpoint_consistent"] = (
                abs(reproduced - run[name]["best_validation_loss"]) <= 1e-12
            )

        report[f"seed_{seed}"] = run
        print(f"\nseed {seed}")
        for name, values in run.items():
            print(f"  {name}: {values}")

    for name in ("step", "cache", "jax"):
        walls = [report[f"seed_{seed}"][name]["wall_s"] for seed in seeds]
        report[f"mean_{name}_wall_s"] = statistics.mean(walls)
    report["jax_vs_step"] = report["mean_jax_wall_s"] / report["mean_step_wall_s"]
    report["cache_vs_step"] = report["mean_cache_wall_s"] / report["mean_step_wall_s"]
    print(
        f"\nwall ratios: jax/step {report['jax_vs_step']:.2f} | "
        f"cache/step {report['cache_vs_step']:.2f}",
    )
    save("pipeline.json", report)


if __name__ == "__main__":
    main()
