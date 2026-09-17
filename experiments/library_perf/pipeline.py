"""Do the step-level review findings survive a full training pipeline?

Runs the shipped trainer with per-step banks (``mask_refresh="step"``) and with
one retained bank (``"cache"``), plus a loop that mirrors the trainer but trains
on the pure-JAX ``sampled_mmd``, all on the same data, seed, schedule, and
optimizer.  Reports wall clock, loss trajectories, and a common independent
score, then verifies that the returned ``best_params`` reproduce the returned
``best_validation_loss`` against the trainer's own validation bank.

    PYTHONPATH=packages/iqp-cond/src .venv/bin/python \
        experiments/library_perf/pipeline.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from common import problem, save
from iqp_cond.loss import banked_mmd, sampled_mmd

import iqp_cond as iqc


def _batches(targets: np.ndarray, config: iqc.TrainingConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        targets.shape[1],
        size=(config.steps, targets.shape[0], config.batch_size),
    )


def train_with_jax_masks(
    ansatz: iqc.Ansatz,
    encoders: iqc.Encoder,
    training_targets: np.ndarray,
    config: iqc.TrainingConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The trainer loop with ``sampled_mmd`` as the objective (masks in JAX)."""
    params = iqc.initialize_parameters(ansatz.n_parameters, seed)
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(params)
    indices = _batches(training_targets, config, seed)
    generators = jnp.asarray(ansatz.generators)
    root_key = jax.random.PRNGKey(seed)

    def loss(current_params, target_batch, step_key):
        return sampled_mmd(
            current_params,
            generators,
            encoders,
            target_batch,
            step_key,
            sigmas=config.sigmas,
            n_ops=config.n_ops,
            n_samples=config.n_iqp_samples,
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
    seeds = (0, 1)
    report: dict = {"config": {"steps": config.steps, "seeds": list(seeds)}}
    for seed in seeds:
        ansatz, encoders, targets = problem(
            n_visible=12,
            n_conditions=5,
            n_targets=2048,
            seed=seed,
        )
        validation = targets[:, :512]
        common_bank = iqc.operator_bank(
            config.sigmas,
            config.validation_n_ops,
            encoders.n_conditions,
            targets.shape[2],
            seed=seed + 777,
        )
        common_key = jax.random.PRNGKey(seed + 888)

        def independent_score(
            current_params: np.ndarray,
            ansatz=ansatz,
            encoders=encoders,
            validation=validation,
            common_bank=common_bank,
            common_key=common_key,
        ) -> float:
            return float(
                banked_mmd(
                    current_params,
                    ansatz.generators,
                    encoders,
                    jnp.asarray(validation),
                    common_bank,
                    config.validation_n_iqp_samples,
                    common_key,
                ).loss,
            )

        run: dict = {}
        start = time.perf_counter()
        step_result = iqc.train(
            ansatz,
            encoders,
            targets,
            validation,
            seed=seed,
            config=config,
        )
        run["step"] = {
            "wall_s": time.perf_counter() - start,
            "best_validation_loss": float(step_result.best_validation_loss),
            "independent_score": independent_score(step_result.best_params),
            "final_train_loss": float(np.mean(step_result.train_losses[-20:])),
        }

        start = time.perf_counter()
        cache_result = iqc.train(
            ansatz,
            encoders,
            targets,
            validation,
            seed=seed,
            config=config,
            mask_refresh=iqc.CACHE,
        )
        run["cache"] = {
            "wall_s": time.perf_counter() - start,
            "best_validation_loss": float(cache_result.best_validation_loss),
            "independent_score": independent_score(cache_result.best_params),
            "final_train_loss": float(np.mean(cache_result.train_losses[-20:])),
        }

        jax_params, jax_losses, jax_wall = train_with_jax_masks(
            ansatz,
            encoders,
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
        expected_bank = iqc.operator_bank(
            config.sigmas,
            config.validation_n_ops,
            encoders.n_conditions,
            targets.shape[2],
            seed + 1,
        )
        expected_key = jax.random.PRNGKey(seed + 1_000_000)
        for name, result in (("step", step_result), ("cache", cache_result)):
            reproduced = float(
                banked_mmd(
                    result.best_params,
                    ansatz.generators,
                    encoders,
                    jnp.asarray(validation),
                    expected_bank,
                    config.validation_n_iqp_samples,
                    expected_key,
                ).loss,
            )
            run[name]["reproduced_best_loss"] = reproduced
            run[name]["checkpoint_consistent"] = (
                abs(reproduced - result.best_validation_loss) <= 1e-12
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
