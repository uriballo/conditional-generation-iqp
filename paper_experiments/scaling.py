"""Fixed-bank scaling selection and independent trajectory evaluation."""

from __future__ import annotations

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np

from .config import RunConfig
from .studies import make_encoder, make_targets

TRAINING_BETAS = (0.6, 0.8, 1.2, 1.4)


def normalized_validation_score(values, scales):
    """Average temperature-normalized MMD without clipping negative estimates."""
    values, scales = np.asarray(values), np.asarray(scales)
    if values.ndim != 1 or values.shape != scales.shape or not values.size:
        raise ValueError("validation values and scales must have matching vectors")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
    ):
        raise ValueError("finite validation values and positive scales required")
    return float(np.mean(values / scales))


def make_validation_selector(model, targets, config, seed):
    """Reproduce the frozen four-bank scaling checkpoint-selection protocol."""
    tc = config.training_config()
    scales = np.asarray(config.validation_scales)
    if scales.shape != (model.encoder.n_conditions,):
        raise ValueError("one validation scale per training condition required")
    normalized_validation_score(np.zeros_like(scales), scales)
    keys = jnp.stack(
        [
            jax.random.PRNGKey(seed + 4_000_000 + rep)
            for rep in range(config.validation_banks)
        ]
    )
    targets = jnp.asarray(targets)

    def evaluate_one(theta, key):
        result = model.mmd(
            theta,
            targets,
            key,
            sigmas=tc.sigmas,
            n_ops=tc.validation_n_ops,
            n_samples=tc.validation_mc_samples,
            shared_samples=True,
            gemm_dtype=tc.gemm_dtype,
            estimator_dtype=tc.estimator_dtype,
            sample_chunk=tc.sample_chunk,
        )
        return result.per_condition

    evaluate = jax.jit(jax.vmap(evaluate_one, in_axes=(None, 0)))

    def select(theta):
        values = np.asarray(evaluate(jnp.asarray(theta), keys))
        if values.shape != (config.validation_banks, model.encoder.n_conditions):
            raise ValueError("validation bank dimensions differ")
        return normalized_validation_score(values.mean(axis=0), scales)

    return select


def _evaluate_mmd_trajectory(
    model: ciqp.ConditionalModel,
    snapshots: dict[int, np.ndarray],
    held_out_betas: np.ndarray,
    seed: int,
    best_step: int,
    *,
    config: RunConfig,
    replicate_banks: int,
    target_draws: int,
    evaluation_interval: int | None = None,
) -> dict:
    """Use evaluation banks independent of training and paired across steps."""
    if replicate_banks < 2 or target_draws < 2:
        raise ValueError(
            "independent evaluation requires at least two replicates/draws"
        )
    available = sorted(snapshots)
    if evaluation_interval is None:
        steps = sorted(
            {
                available[
                    min(
                        range(len(available)),
                        key=lambda i: abs(available[i] - fraction * config.steps),
                    )
                ]
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
            }
            | {best_step}
        )
    else:
        if evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive")
        requested = set(range(0, config.steps + 1, evaluation_interval))
        requested.add(config.steps)
        requested.add(best_step)
        missing = requested.difference(available)
        if missing:
            raise ValueError(
                f"snapshots missing requested evaluation steps: {sorted(missing)}"
            )
        steps = sorted(requested)
    training_config = config.training_config()
    sigmas = training_config.resolved_sigmas(config.n_visible)
    grids = {
        "training": np.asarray(TRAINING_BETAS, dtype=np.float64),
        "held_out": np.asarray(held_out_betas, dtype=np.float64),
    }
    rows = []
    for grid_index, (label, betas) in enumerate(grids.items()):
        evaluation_model = ciqp.ConditionalModel(
            model.ansatz,
            make_encoder(
                betas, config.n_total, config.encoder, scale=config.encoder_scale
            ),
            n_visible=config.n_visible,
        )
        for step in steps:
            estimates = []
            for replicate in range(replicate_banks):
                draw_seed = seed + grid_index * 10_000_000 + replicate
                targets = make_targets(
                    betas,
                    config.n_visible,
                    target_draws,
                    np.random.default_rng(draw_seed),
                )
                estimate = evaluation_model.mmd(
                    snapshots[step],
                    targets,
                    jax.random.PRNGKey(draw_seed),
                    sigmas=sigmas,
                    n_ops=config.validation_n_ops,
                    n_samples=config.validation_mc_samples,
                    shared_samples=True,
                    gemm_dtype=training_config.gemm_dtype,
                    estimator_dtype=training_config.estimator_dtype,
                    sample_chunk=config.sample_chunk,
                )
                estimates.append(np.asarray(estimate.per_condition, dtype=np.float64))
            values = np.stack(estimates)
            rows.append(
                {
                    "grid": label,
                    "step": step,
                    "betas": betas.tolist(),
                    "replicate_per_condition": values.tolist(),
                    "mean_per_condition": values.mean(axis=0).tolist(),
                    "se_per_condition": (
                        values.std(axis=0, ddof=1) / np.sqrt(replicate_banks)
                    ).tolist(),
                    "mean": float(values.mean()),
                    "se": float(
                        values.mean(axis=1).std(ddof=1) / np.sqrt(replicate_banks)
                    ),
                }
            )
    initial = next(
        row for row in rows if row["grid"] == "held_out" and row["step"] == 0
    )
    selected = next(
        row for row in rows if row["grid"] == "held_out" and row["step"] == best_step
    )
    initial_values = np.asarray(initial["replicate_per_condition"]).mean(axis=1)
    selected_values = np.asarray(selected["replicate_per_condition"]).mean(axis=1)
    replicate_decreases = initial_values - selected_values
    decrease_mean = float(replicate_decreases.mean())
    decrease_se = float(replicate_decreases.std(ddof=1) / np.sqrt(replicate_banks))
    return {
        "metric": "training_independent_paired_checkpoint_sampled_squared_mmd",
        "replicate_banks": replicate_banks,
        "target_draws_per_condition_per_bank": target_draws,
        "mask_draws_per_bandwidth": config.validation_n_ops,
        "reference_draws_per_bank": config.validation_mc_samples,
        "selected_step": best_step,
        "evaluation_interval": evaluation_interval,
        "rows": rows,
        "held_out_initial_minus_selected": decrease_mean,
        "held_out_decrease_se": decrease_se,
        "held_out_replicate_decreases": replicate_decreases.tolist(),
        "held_out_decrease_exceeds_95pct_uncertainty": bool(
            decrease_mean > 1.96 * decrease_se
        ),
    }
