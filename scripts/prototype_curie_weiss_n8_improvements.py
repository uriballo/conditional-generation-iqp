#!/usr/bin/env python3
"""PROTOTYPE: diagnose three Curie--Weiss n=8 improvement ideas.

This uses exact enumeration to separate optimization, encoder, and objective
limitations. It is not finite-sample training evidence and writes no files.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution, minimize

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.artifacts import read_immutable_json
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.curie_weiss_training import symmetry_expansion
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    exact_iqp_probabilities_many_jax,
    fwht_jax,
    generator_characters,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.mmd import mmd_mask_weights, mmd_operator_probability

jax.config.update("jax_enable_x64", True)

N_QUBITS = 8
TRAINING_BETAS = np.asarray((0.5, 0.7, 0.9, 1.1, 1.3, 1.5))
HELD_OUT_BETAS = np.asarray((0.6, 0.8, 1.0, 1.2, 1.4))
EVALUATION_BETAS = np.sort(np.concatenate((TRAINING_BETAS, HELD_OUT_BETAS)))
SIGMAS = (1.0, 2.0, 4.0)
OPTIMIZER_SEED = 20_260_814


def fixed_angles(betas: np.ndarray) -> np.ndarray:
    alpha = np.pi * (0.35 - 0.70 * (betas - 0.5))
    return np.repeat(alpha[:, None], N_QUBITS, axis=1)


def learned_angles(parameters: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
    """Monotone criticality-aware encoder with four bounded coefficients."""
    intercept, slope, amplitude, sharpness = parameters
    coordinate = betas - 1.0
    alpha = jnp.pi * (
        intercept - slope * coordinate - amplitude * jnp.tanh(sharpness * coordinate)
    )
    return jnp.repeat(alpha[:, None], N_QUBITS, axis=1)


def exact_targets(betas: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            exact_curie_weiss_target(N_QUBITS, float(beta)).probabilities
            for beta in betas
        ]
    )


def exact_multiscale_mmd(
    model_probabilities: jnp.ndarray,
    target_probabilities: jnp.ndarray,
) -> jnp.ndarray:
    model_fourier = fwht_jax(model_probabilities)
    target_fourier = fwht_jax(target_probabilities)
    weights = mmd_mask_weights(N_QUBITS)
    losses = []
    for sigma in SIGMAS:
        probability = mmd_operator_probability(sigma)
        mask_law = probability**weights * (1.0 - probability) ** (N_QUBITS - weights)
        losses.append(
            jnp.mean(jnp.sum(mask_law * (model_fourier - target_fourier) ** 2, axis=1))
        )
    return jnp.mean(jnp.stack(losses))


def make_objective(*, learned_encoder: bool, moment_weight: float):
    ansatz = complete_ansatz(N_QUBITS)
    generators = jnp.asarray(ansatz.generators)
    expansion = jnp.asarray(symmetry_expansion(np.asarray(ansatz.generators)))
    characters = jnp.asarray(generator_characters(np.asarray(ansatz.generators)))
    betas = jnp.asarray(TRAINING_BETAS)
    targets = jnp.asarray(exact_targets(TRAINING_BETAS))
    magnetization = jnp.asarray(exact_curie_weiss_target(N_QUBITS, 1.0).magnetization)
    target_m2 = targets @ (magnetization**2)
    target_m4 = targets @ (magnetization**4)
    fixed = jnp.asarray(fixed_angles(TRAINING_BETAS))

    def objective(vector: jnp.ndarray) -> jnp.ndarray:
        coefficients = vector[:2]
        angles = learned_angles(vector[2:], betas) if learned_encoder else fixed
        model = exact_iqp_probabilities_many_jax(
            expansion @ coefficients,
            generators,
            angles,
            characters=characters,
        )
        mmd = exact_multiscale_mmd(model, targets)
        model_m2 = model @ (magnetization**2)
        model_m4 = model @ (magnetization**4)
        moment_loss = jnp.mean(
            (model_m2 - target_m2) ** 2 + (model_m4 - target_m4) ** 2
        )
        return mmd + moment_weight * moment_loss

    value_gradient = jax.jit(jax.value_and_grad(objective))

    def wrapped(vector: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_gradient(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=np.float64)

    return wrapped


def multistart_exact(
    *, learned_encoder: bool, moment_weight: float, starts: int = 16
) -> tuple[np.ndarray, float]:
    objective = make_objective(
        learned_encoder=learned_encoder,
        moment_weight=moment_weight,
    )
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
    if learned_encoder:
        bounds.extend([(-0.5, 0.5), (0.0, 1.5), (0.0, 1.0), (0.25, 8.0)])
    rng = np.random.default_rng(OPTIMIZER_SEED + round(moment_weight * 1000))
    initial = [np.asarray([(lower + upper) / 2 for lower, upper in bounds])]
    if learned_encoder:
        initial[0][2:] = np.asarray((0.0, 0.7, 0.0, 2.0))
    initial.extend(
        np.asarray([rng.uniform(lower, upper) for lower, upper in bounds])
        for _ in range(starts - 1)
    )
    results = [
        minimize(
            objective,
            point,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9},
        )
        for point in initial
    ]
    best = min(results, key=lambda result: float(result.fun))
    return np.asarray(best.x), float(best.fun)


def fixed_encoder_tvd_ceiling() -> tuple[np.ndarray, float]:
    ansatz = complete_ansatz(N_QUBITS)
    expansion = symmetry_expansion(np.asarray(ansatz.generators))
    angles = fixed_angles(TRAINING_BETAS)
    targets = [exact_curie_weiss_target(N_QUBITS, beta) for beta in TRAINING_BETAS]
    bitstrings = enumerate_bitstrings(N_QUBITS)

    def objective(coefficients: np.ndarray) -> float:
        models = exact_iqp_probabilities_many(
            expansion @ coefficients,
            ansatz.generators,
            angles,
        )
        return float(
            np.mean(
                [
                    total_variation_distance(
                        target.profile,
                        shell_probabilities(model, bitstrings),
                    )
                    for target, model in zip(targets, models, strict=True)
                ]
            )
        )

    result = differential_evolution(
        objective,
        bounds=((-np.pi, np.pi), (-np.pi, np.pi)),
        seed=OPTIMIZER_SEED,
        maxiter=80,
        popsize=15,
        polish=True,
        tol=1e-10,
        workers=1,
    )
    return np.asarray(result.x), float(result.fun)


def evaluate(name: str, vector: np.ndarray, *, learned_encoder: bool) -> dict:
    ansatz = complete_ansatz(N_QUBITS)
    expansion = symmetry_expansion(np.asarray(ansatz.generators))
    angles = (
        np.asarray(
            learned_angles(jnp.asarray(vector[2:]), jnp.asarray(EVALUATION_BETAS))
        )
        if learned_encoder
        else fixed_angles(EVALUATION_BETAS)
    )
    models = exact_iqp_probabilities_many(
        expansion @ vector[:2], ansatz.generators, angles
    )
    bitstrings = enumerate_bitstrings(N_QUBITS)
    rows = []
    for beta, model in zip(EVALUATION_BETAS, models, strict=True):
        target = exact_curie_weiss_target(N_QUBITS, float(beta))
        rows.append(
            {
                "beta": float(beta),
                "role": "training" if beta in TRAINING_BETAS else "held_out",
                "profile_tvd": total_variation_distance(
                    target.profile, shell_probabilities(model, bitstrings)
                ),
                "exact_mmd": float(
                    exact_multiscale_mmd(
                        jnp.asarray(model[None, :]),
                        jnp.asarray(target.probabilities[None, :]),
                    )
                ),
                "m2_error": float(
                    model @ (target.magnetization**2)
                    - target.probabilities @ (target.magnetization**2)
                ),
                "m4_error": float(
                    model @ (target.magnetization**4)
                    - target.probabilities @ (target.magnetization**4)
                ),
            }
        )
    training = [row for row in rows if row["role"] == "training"]
    held_out = [row for row in rows if row["role"] == "held_out"]
    result = {
        "method": name,
        "mean_training_profile_tvd": float(
            np.mean([row["profile_tvd"] for row in training])
        ),
        "mean_held_out_profile_tvd": float(
            np.mean([row["profile_tvd"] for row in held_out])
        ),
        "critical_profile_tvd": next(
            row["profile_tvd"] for row in rows if row["beta"] == 1.0
        ),
        "cold_profile_tvd": next(
            row["profile_tvd"] for row in rows if row["beta"] == 1.5
        ),
        "mean_exact_mmd": float(np.mean([row["exact_mmd"] for row in rows])),
        "moment_rmse": float(
            np.sqrt(
                np.mean([row["m2_error"] ** 2 + row["m4_error"] ** 2 for row in rows])
            )
        ),
        "phase_coefficients": vector[:2].tolist(),
        "rows": rows,
    }
    if learned_encoder:
        result["encoder"] = {
            "intercept": float(vector[2]),
            "slope": float(vector[3]),
            "critical_amplitude": float(vector[4]),
            "critical_sharpness": float(vector[5]),
            "angles_over_pi": (angles[:, 0] / np.pi).tolist(),
        }
    return result


def production_baseline() -> dict:
    report = read_immutable_json(
        Path("results/curie-weiss-foundations-v1/report/campaign-report.json")
    )
    rows = [row for row in report["response_rows"] if row["n_qubits"] == N_QUBITS]
    training = [row for row in rows if row["role"] == "training"]
    held_out = [row for row in rows if row["role"] == "held_out"]
    return {
        "method": "production_finite_sample_mmd_mean_of_10_seeds",
        "mean_training_profile_tvd": float(
            np.mean([row["conditioned"]["profile_tvd"] for row in training])
        ),
        "mean_held_out_profile_tvd": float(
            np.mean([row["conditioned"]["profile_tvd"] for row in held_out])
        ),
        "critical_profile_tvd": float(
            np.mean(
                [
                    row["conditioned"]["profile_tvd"]
                    for row in rows
                    if np.isclose(row["beta"], 1.0)
                ]
            )
        ),
        "cold_profile_tvd": float(
            np.mean(
                [
                    row["conditioned"]["profile_tvd"]
                    for row in rows
                    if np.isclose(row["beta"], 1.5)
                ]
            )
        ),
        "mean_exact_mmd": None,
    }


def main() -> None:
    print("PROTOTYPE — exact n=8 diagnostics; not publication training evidence")
    fixed_mmd_vector, fixed_mmd_loss = multistart_exact(
        learned_encoder=False, moment_weight=0.0
    )
    fixed_tvd_vector, fixed_tvd_loss = fixed_encoder_tvd_ceiling()
    learned_vector, learned_loss = multistart_exact(
        learned_encoder=True, moment_weight=0.0, starts=64
    )
    hybrid_vector, hybrid_loss = multistart_exact(
        learned_encoder=True, moment_weight=0.03, starts=64
    )
    methods = [
        production_baseline(),
        evaluate("fixed_encoder_exact_mmd", fixed_mmd_vector, learned_encoder=False),
        evaluate(
            "fixed_encoder_direct_tvd_ceiling", fixed_tvd_vector, learned_encoder=False
        ),
        evaluate("learned_encoder_exact_mmd", learned_vector, learned_encoder=True),
        evaluate(
            "learned_encoder_mmd_plus_0.03_moments",
            hybrid_vector,
            learned_encoder=True,
        ),
    ]
    diagnostics = {
        "schema_version": 1,
        "status": "throwaway_exact_enumeration_prototype",
        "n_qubits": N_QUBITS,
        "training_betas": TRAINING_BETAS.tolist(),
        "held_out_betas": HELD_OUT_BETAS.tolist(),
        "optimization_losses": {
            "fixed_exact_mmd": fixed_mmd_loss,
            "fixed_direct_tvd": fixed_tvd_loss,
            "learned_exact_mmd": learned_loss,
            "learned_hybrid": hybrid_loss,
        },
        "methods": methods,
    }
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
