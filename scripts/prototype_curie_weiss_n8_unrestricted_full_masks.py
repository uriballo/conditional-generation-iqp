#!/usr/bin/env python3
"""PROTOTYPE: n=8 unrestricted Curie--Weiss training with full-mask MMD.

This throwaway screen removes exchange-symmetry parameter sharing and does not
learn the condition encoder.  Every loss evaluation integrates the Gaussian
Hamming kernel over all 2**8 Fourier character masks, so every one- and
two-body character is present at every update without changing the kernel law.
Optional ancillas use broadcast-all conditioning and are marginalized from the
eight-bit output law. Exact target laws are first accessed after all
checkpoints have been frozen. Nothing is written unless ``--output`` is
supplied.
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

from iqp_cond.ancilla import (
    ancilla_ansatz,
    ancilla_condition_angles,
    exact_ancilla_marginal_probabilities,
    pad_observables_for_ancillas,
)
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.curie_weiss_protocol import CurieWeissCampaignManifest
from iqp_cond.exact import (
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.mmd import _target_terms, mmd_mask_weights, mmd_operator_probability
from iqp_cond.simulator import _estimate_pauli_z_unchecked
from iqp_cond.training import TrainingConfig, _batch_schedule, initialize_parameters

jax.config.update("jax_enable_x64", True)

N_VISIBLE_QUBITS = 8
TRAINING_BETAS = np.asarray((0.5, 0.7, 0.9, 1.1, 1.3, 1.5))
ORIGINAL_HELD_OUT = np.asarray((0.6, 0.8, 1.0, 1.2, 1.4))
SHADOW_HELD_OUT = np.asarray(tuple(0.55 + 0.1 * index for index in range(10)))
METHODS = ("conditioned", "no_condition")


@dataclass(frozen=True)
class SelectedModel:
    method: str
    seed: int
    restart: int
    validation_loss: float
    phases: np.ndarray


def _fixed_angles(betas: jnp.ndarray) -> jnp.ndarray:
    alpha = jnp.pi * (0.35 - 0.70 * (betas - 0.5))
    return jnp.repeat(alpha[:, None], N_VISIBLE_QUBITS, axis=1)


def _load_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(f"results/curie-weiss-foundations-v1/datasets/n8_seed{seed}.npz")
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["training_betas"], TRAINING_BETAS):
            raise RuntimeError("dataset training grid differs")
        return np.asarray(archive["training"]), np.asarray(archive["validation"])


def _full_mask_mmd(
    phases: jnp.ndarray,
    generators: jnp.ndarray,
    angles: jnp.ndarray,
    targets: jnp.ndarray,
    sigmas: tuple[float, ...],
    n_model_samples: int,
    key: jax.Array,
    data_masks: jnp.ndarray,
    model_masks: jnp.ndarray,
    mask_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Finite-sample MMD with exact quadrature over the character-mask law."""
    losses = []
    for sigma, model_key in zip(
        sigmas,
        jax.random.split(key, len(sigmas)),
        strict=True,
    ):
        model = _estimate_pauli_z_unchecked(
            phases,
            generators,
            model_masks,
            angles,
            model_key,
            n_model_samples,
        )
        target_means, target_squares = _target_terms(targets, data_masks)
        model_sums = jnp.sum(model.sample_values, axis=1)
        model_square_sums = jnp.sum(model.sample_values**2, axis=1)
        model_squares = (model_sums**2 - model_square_sums) / (
            n_model_samples * (n_model_samples - 1)
        )
        per_mask = model_squares - 2.0 * model.values * target_means + target_squares
        probability = mmd_operator_probability(float(sigma))
        kernel_weights = probability**mask_weights * (1.0 - probability) ** (
            N_VISIBLE_QUBITS - mask_weights
        )
        losses.append(jnp.mean(jnp.sum(kernel_weights * per_mask, axis=-1)))
    return jnp.mean(jnp.stack(losses))


def _train_attempt(
    method: str,
    seed: int,
    restart: int,
    training: np.ndarray,
    validation: np.ndarray,
    config: TrainingConfig,
    n_ancillas: int,
) -> SelectedModel:
    ansatz = ancilla_ansatz(N_VISIBLE_QUBITS, n_ancillas)
    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    data_masks = jnp.asarray(enumerate_bitstrings(N_VISIBLE_QUBITS), dtype=jnp.float64)
    model_masks = jnp.asarray(
        pad_observables_for_ancillas(data_masks, n_ancillas), dtype=jnp.float64
    )
    mask_weights = mmd_mask_weights(N_VISIBLE_QUBITS)
    fixed_data_angles = _fixed_angles(jnp.asarray(TRAINING_BETAS))
    data_angles = (
        jnp.zeros_like(fixed_data_angles)
        if method == "no_condition"
        else fixed_data_angles
    )
    angles = ancilla_condition_angles(data_angles, n_ancillas, "broadcast_all")
    phase_seed = 1_000_000 + seed * 100 + restart
    phases = initialize_parameters(ansatz.n_parameters, phase_seed)
    schedule = optax.cosine_decay_schedule(
        config.learning_rate,
        config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    optimizer_state = optimizer.init(phases)
    batch_indices = _batch_schedule(
        len(TRAINING_BETAS), training.shape[1], config, seed
    )
    base_key = jax.random.PRNGKey(seed)

    def loss(
        current: jnp.ndarray,
        targets: jnp.ndarray,
        key: jax.Array,
        n_model_samples: int,
    ) -> jnp.ndarray:
        return _full_mask_mmd(
            current,
            generators,
            angles,
            targets,
            config.sigmas,
            n_model_samples,
            key,
            data_masks,
            model_masks,
            mask_weights,
        )

    value_gradient = jax.jit(jax.value_and_grad(loss), static_argnums=(3,))

    def validation_loss(current: jnp.ndarray) -> jnp.ndarray:
        return loss(
            current,
            jnp.asarray(validation),
            jax.random.PRNGKey(seed + 1_000_000),
            config.validation_n_iqp_samples,
        )

    validate = jax.jit(validation_loss)
    best_phases = phases
    best_validation = float(validate(phases))
    for step in range(config.steps):
        indices = batch_indices[step]
        batch = np.stack(
            [training[index, indices[index]] for index in range(len(TRAINING_BETAS))]
        )
        _, gradient = value_gradient(
            phases,
            jnp.asarray(batch),
            jax.random.fold_in(base_key, step),
            config.n_iqp_samples,
        )
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, phases)
        phases = optax.apply_updates(phases, updates)
        if (step + 1) % config.validation_interval == 0:
            current_validation = float(validate(phases))
            if current_validation < best_validation:
                best_validation = current_validation
                best_phases = phases
    return SelectedModel(
        method=method,
        seed=seed,
        restart=restart,
        validation_loss=best_validation,
        phases=np.asarray(best_phases),
    )


def _evaluate(selected: list[SelectedModel], n_ancillas: int) -> list[dict]:
    """First exact-target access in the program occurs in this function."""
    ansatz = ancilla_ansatz(N_VISIBLE_QUBITS, n_ancillas)
    bitstrings = enumerate_bitstrings(N_VISIBLE_QUBITS)
    evaluation_betas = np.sort(
        np.unique(np.concatenate((TRAINING_BETAS, ORIGINAL_HELD_OUT, SHADOW_HELD_OUT)))
    )
    rows = []
    for model in selected:
        data_angles = (
            np.zeros((len(evaluation_betas), N_VISIBLE_QUBITS))
            if model.method == "no_condition"
            else np.asarray(_fixed_angles(jnp.asarray(evaluation_betas)))
        )
        probabilities = exact_ancilla_marginal_probabilities(
            model.phases,
            ansatz.generators,
            data_angles,
            n_data_qubits=N_VISIBLE_QUBITS,
            n_ancillas=n_ancillas,
            scheme="broadcast_all",
        )
        for beta, probability in zip(evaluation_betas, probabilities, strict=True):
            target = exact_curie_weiss_target(N_VISIBLE_QUBITS, float(beta))
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
                    "raw_bitstring_tvd": total_variation_distance(
                        target.probabilities,
                        probability,
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
                "mean_raw_bitstring_tvd": float(
                    np.mean([row["raw_bitstring_tvd"] for row in selected])
                ),
                "mean_moment_error": float(
                    np.mean([row["moment_error"] for row in selected])
                ),
            }
    effects = {}
    for role in ("training", "original_held_out", "shadow_held_out"):
        differences = []
        for seed in seeds:
            conditioned = np.mean(
                [
                    row["profile_tvd"]
                    for row in rows
                    if row["method"] == "conditioned"
                    and row["role"] == role
                    and row["seed"] == seed
                ]
            )
            control = np.mean(
                [
                    row["profile_tvd"]
                    for row in rows
                    if row["method"] == "no_condition"
                    and row["role"] == role
                    and row["seed"] == seed
                ]
            )
            differences.append(control - conditioned)
        effects[role] = _paired_interval(np.asarray(differences))
    return {"cells": cells, "paired_control_minus_conditioned": effects}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--model-sample-scale", type=float, default=1.0)
    parser.add_argument("--ancillas", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.model_sample_scale <= 1.0:
        raise SystemExit("--model-sample-scale must lie in (0, 1]")
    if args.ancillas < 0:
        raise SystemExit("--ancillas must be non-negative")
    manifest = CurieWeissCampaignManifest.from_json(
        Path("configs/curie_weiss_foundations_v1.json")
    )
    settings = manifest.raw["training"]
    training_samples = max(
        16, round(int(settings["training_model_samples"]) * args.model_sample_scale)
    )
    validation_samples = max(
        16, round(int(settings["validation_model_samples"]) * args.model_sample_scale)
    )
    config = TrainingConfig(
        steps=args.steps,
        batch_size=75,
        learning_rate=float(settings["initial_learning_rate"]),
        final_learning_rate=float(settings["final_learning_rate"]),
        validation_interval=min(25, args.steps),
        sigmas=tuple(settings["bandwidths"]),
        n_ops=2**N_VISIBLE_QUBITS,
        n_iqp_samples=training_samples,
        validation_n_ops=2**N_VISIBLE_QUBITS,
        validation_n_iqp_samples=validation_samples,
    )
    seeds = tuple(args.seeds)
    print(
        f"PROTOTYPE — unrestricted n=8+m={args.ancillas}; "
        "all 256 visible masks at every loss evaluation",
        flush=True,
    )
    started = time.perf_counter()
    selected_models = []
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
                    args.ancillas,
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
    rows = _evaluate(selected_models, args.ancillas)
    payload = {
        "schema_version": 1,
        "status": "throwaway_unrestricted_full_mask_prototype",
        "seeds": list(seeds),
        "restarts": args.restarts,
        "steps": args.steps,
        "model_sample_scale": args.model_sample_scale,
        "n_visible_qubits": N_VISIBLE_QUBITS,
        "n_ancillas": args.ancillas,
        "ancilla_conditioning_scheme": "broadcast_all",
        "character_masks_per_bandwidth": 2**N_VISIBLE_QUBITS,
        "mandatory_one_body_masks": N_VISIBLE_QUBITS,
        "mandatory_two_body_masks": (N_VISIBLE_QUBITS * (N_VISIBLE_QUBITS - 1) // 2),
        "trainable_phase_count": ancilla_ansatz(
            N_VISIBLE_QUBITS, args.ancillas
        ).n_parameters,
        "wall_seconds": time.perf_counter() - started,
        "selected_models": [
            {
                "method": model.method,
                "seed": model.seed,
                "restart": model.restart,
                "validation_loss": model.validation_loss,
                "phases": model.phases.tolist(),
            }
            for model in selected_models
        ],
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
