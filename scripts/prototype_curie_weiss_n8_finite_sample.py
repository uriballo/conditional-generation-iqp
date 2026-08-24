#!/usr/bin/env python3
"""PROTOTYPE: finite-sample n=8 encoder/objective confirmation.

All checkpoints are selected from role-tagged training and validation samples.
Exact target laws, including the shadow grid, are touched only after every
selected model is frozen. No files are written unless ``--output`` is given.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.curie_weiss_protocol import CurieWeissCampaignManifest
from iqp_cond.curie_weiss_training import symmetry_expansion
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.mmd import conditional_mmd
from iqp_cond.simulator import _estimate_pauli_z_unchecked
from iqp_cond.training import TrainingConfig, _batch_schedule, initialize_parameters

jax.config.update("jax_enable_x64", True)

N_QUBITS = 8
TRAINING_BETAS = np.asarray((0.5, 0.7, 0.9, 1.1, 1.3, 1.5))
ORIGINAL_HELD_OUT = np.asarray((0.6, 0.8, 1.0, 1.2, 1.4))
SHADOW_HELD_OUT = np.asarray(tuple(0.55 + 0.1 * index for index in range(10)))
METHODS = ("fixed_mmd", "learned_mmd", "learned_hybrid", "no_condition")
MOMENT_WEIGHT = 0.03


@dataclass(frozen=True)
class SelectedModel:
    method: str
    seed: int
    restart: int
    validation_loss: float
    phase_coefficients: np.ndarray
    encoder_latents: np.ndarray | None


def _fixed_angles(betas: jnp.ndarray) -> jnp.ndarray:
    alpha = jnp.pi * (0.35 - 0.70 * (betas - 0.5))
    return jnp.repeat(alpha[:, None], N_QUBITS, axis=1)


def _learned_angles(latents: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
    intercept = 0.5 * jnp.tanh(latents[0])
    slope = 1.5 * jax.nn.sigmoid(latents[1])
    alpha = jnp.pi * (intercept - slope * (betas - 1.0))
    return jnp.repeat(alpha[:, None], N_QUBITS, axis=1)


def _encoder_values(latents: np.ndarray) -> tuple[float, float]:
    intercept = 0.5 * np.tanh(latents[0])
    slope = 1.5 / (1.0 + np.exp(-latents[1]))
    return float(intercept), float(slope)


def _moment_operators() -> tuple[jnp.ndarray, int]:
    pair, quadruple = [], []
    for first in range(N_QUBITS):
        for second in range(first + 1, N_QUBITS):
            row = np.zeros(N_QUBITS)
            row[[first, second]] = 1
            pair.append(row)
            for third in range(second + 1, N_QUBITS):
                for fourth in range(third + 1, N_QUBITS):
                    if first < second < third < fourth:
                        row = np.zeros(N_QUBITS)
                        row[[first, second, third, fourth]] = 1
                        quadruple.append(row)
    # The nested construction above repeats no quadruples because first/second
    # are fixed before third/fourth are chosen.
    return jnp.asarray(np.asarray((*pair, *quadruple))), len(pair)


def _sample_moments(samples: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    magnetization = jnp.mean(1.0 - 2.0 * samples, axis=-1)
    return jnp.mean(magnetization**2, axis=1), jnp.mean(magnetization**4, axis=1)


def _model_moments(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    angles: jnp.ndarray,
    key: jax.Array,
    n_samples: int,
    operators: jnp.ndarray,
    n_pairs: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    estimates = _estimate_pauli_z_unchecked(
        params,
        generators,
        operators,
        angles,
        key,
        n_samples,
    ).values
    pair_sum = jnp.sum(estimates[:, :n_pairs], axis=1)
    quadruple_sum = jnp.sum(estimates[:, n_pairs:], axis=1)
    n = float(N_QUBITS)
    second = (n + 2.0 * pair_sum) / n**2
    fourth = (
        3.0 * n**2 - 2.0 * n + (12.0 * n - 16.0) * pair_sum + 24.0 * quadruple_sum
    ) / n**4
    return second, fourth


def _load_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(f"results/curie-weiss-foundations-v1/datasets/n8_seed{seed}.npz")
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["training_betas"], TRAINING_BETAS):
            raise RuntimeError("dataset training grid differs")
        return np.asarray(archive["training"]), np.asarray(archive["validation"])


def _initial_vector(method: str, seed: int, restart: int) -> jnp.ndarray:
    phase_seed = 1_000_000 + seed * 100 + restart
    phases = initialize_parameters(2, phase_seed)
    if not method.startswith("learned"):
        return phases
    slope_latent = np.log((0.7 / 1.5) / (1.0 - 0.7 / 1.5))
    rng = np.random.default_rng(phase_seed + 77)
    encoder = np.asarray((0.0, slope_latent)) + rng.normal(0.0, 0.05, 2)
    return jnp.concatenate((phases, jnp.asarray(encoder)))


def _train_attempt(
    method: str,
    seed: int,
    restart: int,
    training: np.ndarray,
    validation: np.ndarray,
    config: TrainingConfig,
) -> SelectedModel:
    ansatz = complete_ansatz(N_QUBITS)
    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    expansion = jnp.asarray(symmetry_expansion(np.asarray(ansatz.generators)))
    betas = jnp.asarray(TRAINING_BETAS)
    fixed_angles = _fixed_angles(betas)
    neutral_angles = jnp.zeros_like(fixed_angles)
    moment_operators, n_pairs = _moment_operators()
    vector = _initial_vector(method, seed, restart)
    schedule = optax.cosine_decay_schedule(
        config.learning_rate,
        config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(vector)
    batch_indices = _batch_schedule(
        len(TRAINING_BETAS), training.shape[1], config, seed
    )
    base_key = jax.random.PRNGKey(seed)

    def angles_for(current: jnp.ndarray) -> jnp.ndarray:
        if method == "no_condition":
            return neutral_angles
        if method.startswith("learned"):
            return _learned_angles(current[2:], betas)
        return fixed_angles

    def loss(
        current: jnp.ndarray,
        targets: jnp.ndarray,
        key: jax.Array,
        n_ops: int,
        n_samples: int,
    ) -> jnp.ndarray:
        mmd_key, moment_key = jax.random.split(key)
        params = expansion @ current[:2]
        angles = angles_for(current)
        value = conditional_mmd(
            params,
            generators,
            angles,
            targets,
            config.sigmas,
            n_ops,
            n_samples,
            mmd_key,
        ).loss
        if method != "learned_hybrid":
            return value
        target_second, target_fourth = _sample_moments(targets)
        model_second, model_fourth = _model_moments(
            params,
            generators,
            angles,
            moment_key,
            n_samples,
            moment_operators,
            n_pairs,
        )
        penalty = jnp.mean(
            (model_second - target_second) ** 2 + (model_fourth - target_fourth) ** 2
        )
        return value + MOMENT_WEIGHT * penalty

    value_gradient = jax.jit(jax.value_and_grad(loss), static_argnums=(3, 4))

    def validation_loss(current: jnp.ndarray) -> jnp.ndarray:
        validation_key = jax.random.PRNGKey(seed + 1_000_000)
        return loss(
            current,
            jnp.asarray(validation),
            validation_key,
            config.validation_n_ops,
            config.validation_n_iqp_samples,
        )

    validate = jax.jit(validation_loss)
    best_vector = vector
    best_validation = float(validate(vector))
    for step in range(config.steps):
        indices = batch_indices[step]
        batch = np.stack(
            [training[index, indices[index]] for index in range(len(TRAINING_BETAS))]
        )
        _, gradient = value_gradient(
            vector,
            jnp.asarray(batch),
            jax.random.fold_in(base_key, step),
            config.n_ops,
            config.n_iqp_samples,
        )
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, vector)
        vector = optax.apply_updates(vector, updates)
        completed = step + 1
        if completed % config.validation_interval == 0:
            current_validation = float(validate(vector))
            if current_validation < best_validation:
                best_validation = current_validation
                best_vector = vector
    return SelectedModel(
        method,
        seed,
        restart,
        best_validation,
        np.asarray(best_vector[:2]),
        np.asarray(best_vector[2:]) if method.startswith("learned") else None,
    )


def _evaluate(selected: list[SelectedModel]) -> list[dict]:
    """First exact-target access in the program occurs in this function."""
    ansatz = complete_ansatz(N_QUBITS)
    expansion = symmetry_expansion(np.asarray(ansatz.generators))
    bitstrings = enumerate_bitstrings(N_QUBITS)
    evaluation_betas = np.sort(
        np.unique(np.concatenate((TRAINING_BETAS, ORIGINAL_HELD_OUT, SHADOW_HELD_OUT)))
    )
    rows = []
    for model in selected:
        if model.method == "no_condition":
            angles = np.zeros((len(evaluation_betas), N_QUBITS))
        elif model.encoder_latents is not None:
            angles = np.asarray(
                _learned_angles(
                    jnp.asarray(model.encoder_latents), jnp.asarray(evaluation_betas)
                )
            )
        else:
            angles = np.asarray(_fixed_angles(jnp.asarray(evaluation_betas)))
        probabilities = exact_iqp_probabilities_many(
            expansion @ model.phase_coefficients,
            ansatz.generators,
            angles,
        )
        for beta, probability in zip(evaluation_betas, probabilities, strict=True):
            target = exact_curie_weiss_target(N_QUBITS, float(beta))
            magnetization = target.magnetization
            model_m2 = float(probability @ magnetization**2)
            model_m4 = float(probability @ magnetization**4)
            target_m2 = float(target.probabilities @ magnetization**2)
            target_m4 = float(target.probabilities @ magnetization**4)
            rows.append(
                {
                    "method": model.method,
                    "seed": model.seed,
                    "beta": float(beta),
                    "role": (
                        "training"
                        if beta in TRAINING_BETAS
                        else "original_held_out"
                        if beta in ORIGINAL_HELD_OUT
                        else "shadow_held_out"
                    ),
                    "profile_tvd": total_variation_distance(
                        target.profile,
                        shell_probabilities(probability, bitstrings),
                    ),
                    "moment_error": float(
                        np.sqrt(
                            (model_m2 - target_m2) ** 2 + (model_m4 - target_m4) ** 2
                        )
                    ),
                }
            )
    return rows


def _paired_interval(values: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(20_260_814)
    draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(
        axis=1
    )
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def _summarize(rows: list[dict], seeds: tuple[int, ...]) -> dict:
    cells = {}
    for method in METHODS:
        cells[method] = {}
        for role in ("training", "original_held_out", "shadow_held_out"):
            selected = [
                row for row in rows if row["method"] == method and row["role"] == role
            ]
            cells[method][role] = {
                "mean_profile_tvd": float(
                    np.mean([row["profile_tvd"] for row in selected])
                ),
                "mean_moment_error": float(
                    np.mean([row["moment_error"] for row in selected])
                ),
            }
    effects = {}
    for comparison in ("learned_mmd", "learned_hybrid"):
        for role in ("training", "original_held_out", "shadow_held_out"):
            differences = []
            for seed in seeds:
                fixed = np.mean(
                    [
                        row["profile_tvd"]
                        for row in rows
                        if row["method"] == "fixed_mmd"
                        and row["role"] == role
                        and row["seed"] == seed
                    ]
                )
                candidate = np.mean(
                    [
                        row["profile_tvd"]
                        for row in rows
                        if row["method"] == comparison
                        and row["role"] == role
                        and row["seed"] == seed
                    ]
                )
                differences.append(fixed - candidate)
            effects[f"{comparison}_{role}"] = _paired_interval(np.asarray(differences))
    return {"cells": cells, "paired_fixed_minus_candidate": effects}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--budget-scale", type=float, default=1.0)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    manifest = CurieWeissCampaignManifest.from_json(
        Path("configs/curie_weiss_foundations_v1.json")
    )
    settings = manifest.raw["training"]
    if not 0.0 < args.budget_scale <= 1.0:
        raise SystemExit("--budget-scale must lie in (0, 1]")
    training_operators = max(
        2, round(int(settings["training_operators"]) * args.budget_scale)
    )
    training_samples = max(
        16, round(int(settings["training_model_samples"]) * args.budget_scale)
    )
    validation_operators = max(
        2, round(int(settings["validation_operators"]) * args.budget_scale)
    )
    validation_samples = max(
        16, round(int(settings["validation_model_samples"]) * args.budget_scale)
    )
    config = TrainingConfig(
        steps=args.steps,
        batch_size=75,
        learning_rate=float(settings["initial_learning_rate"]),
        final_learning_rate=float(settings["final_learning_rate"]),
        validation_interval=min(25, args.steps),
        sigmas=tuple(settings["bandwidths"]),
        n_ops=4 if args.lightweight else training_operators,
        n_iqp_samples=64 if args.lightweight else training_samples,
        validation_n_ops=8 if args.lightweight else validation_operators,
        validation_n_iqp_samples=128 if args.lightweight else validation_samples,
    )
    print(
        "PROTOTYPE — finite-sample n=8 confirmation; shadow targets remain sealed",
        flush=True,
    )
    selected_models = []
    started = time.perf_counter()
    for seed in seeds:
        training, validation = _load_dataset(seed)
        for method in METHODS:
            attempts = []
            for restart in range(args.restarts):
                attempt_start = time.perf_counter()
                model = _train_attempt(
                    method,
                    seed,
                    restart,
                    training,
                    validation,
                    config,
                )
                attempts.append(model)
                print(
                    json.dumps(
                        {
                            "event": "attempt_complete",
                            "method": method,
                            "seed": seed,
                            "restart": restart,
                            "validation_loss": model.validation_loss,
                            "seconds": time.perf_counter() - attempt_start,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            selected_models.append(min(attempts, key=lambda item: item.validation_loss))
    print("all checkpoints frozen; unsealing exact evaluation grids", flush=True)
    rows = _evaluate(selected_models)
    selected_payload = []
    for model in selected_models:
        encoder = (
            None
            if model.encoder_latents is None
            else dict(
                zip(
                    ("intercept", "slope"),
                    _encoder_values(model.encoder_latents),
                    strict=True,
                )
            )
        )
        selected_payload.append(
            {
                "method": model.method,
                "seed": model.seed,
                "restart": model.restart,
                "validation_loss": model.validation_loss,
                "phase_coefficients": model.phase_coefficients.tolist(),
                "encoder": encoder,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "throwaway_finite_sample_prototype",
        "seeds": list(seeds),
        "restarts": args.restarts,
        "steps": args.steps,
        "lightweight": args.lightweight,
        "budget_scale": args.budget_scale,
        "moment_weight": MOMENT_WEIGHT,
        "shadow_grid": SHADOW_HELD_OUT.tolist(),
        "wall_seconds": time.perf_counter() - started,
        "selected_models": selected_payload,
        "summary": _summarize(rows, seeds),
        "rows": rows,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)


if __name__ == "__main__":
    main()
