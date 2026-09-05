#!/usr/bin/env python3
"""Screen stochastic interval encoding on a small Curie--Weiss task.

This is exploratory and intentionally bypasses the production campaign size
guard: the target is evaluated exactly for any ``n >= 2`` so that a fast
four-qubit mechanism test is possible. Fresh per-qubit angles are sampled at
every training step and inference draw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iqp_cond.ancilla import ancilla_only_condition_angles, marginalize_ancillas
from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.stochastic_interval import (
    sample_interval_angles,
    train_with_resampled_angles,
)
from iqp_cond.training import TrainingConfig

TRAINING_BETAS = np.asarray([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
HELDOUT_BETAS = np.asarray([0.6, 0.8, 1.0, 1.2, 1.4])


def beta_to_angles(betas: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Map inverse temperatures to the prototype's broadcast input angle."""
    return np.pi * (0.35 - 0.70 * (np.asarray(betas, dtype=np.float64) - 0.5))


def curie_weiss_probabilities(n_qubits: int, beta: float) -> np.ndarray:
    """Return the exact finite-size zero-field Curie--Weiss law."""
    if n_qubits < 2 or beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("n_qubits must be >= 2 and beta must be positive")
    bits = enumerate_bitstrings(n_qubits)
    magnetization = (2.0 * bits - 1.0).sum(axis=1)
    log_weights = beta * magnetization**2 / (2.0 * n_qubits)
    probabilities = np.exp(log_weights - np.max(log_weights))
    return probabilities / probabilities.sum()


def target_samples(
    betas: np.ndarray, n_qubits: int, seed: int, count: int
) -> np.ndarray:
    bits = enumerate_bitstrings(n_qubits)
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            bits[
                rng.choice(
                    len(bits), count, p=curie_weiss_probabilities(n_qubits, beta)
                )
            ]
            for beta in betas
        ]
    ).astype(np.float64)


def evaluate(
    betas: np.ndarray,
    probabilities: np.ndarray,
    n_qubits: int,
    width: float,
    arm: str,
) -> list[dict[str, object]]:
    bits = enumerate_bitstrings(n_qubits)
    rows = []
    for beta, model in zip(betas, probabilities, strict=True):
        target = curie_weiss_probabilities(n_qubits, float(beta))
        rows.append(
            {
                "beta": float(beta),
                "role": "training" if beta in TRAINING_BETAS else "heldout",
                "arm": arm,
                "half_width_pi": width,
                "profile_tvd": total_variation_distance(
                    shell_probabilities(target, bits), shell_probabilities(model, bits)
                ),
                "bitstring_tvd": total_variation_distance(target, model),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    n_visible = args.n_qubits
    n_total = n_visible + args.ancillas
    ansatz = complete_ansatz(n_total)
    train_targets = target_samples(TRAINING_BETAS, n_visible, 300_000, args.samples)
    validation_targets = target_samples(
        TRAINING_BETAS, n_visible, 400_000, args.validation_samples
    )
    all_betas = np.concatenate((TRAINING_BETAS, HELDOUT_BETAS))
    inference_ancilla_centers = np.repeat(
        beta_to_angles(all_betas)[:, None], args.ancillas, axis=1
    )
    inference_centers = (
        np.asarray(ancilla_only_condition_angles(inference_ancilla_centers, n_visible))
        if args.ancillas
        else np.repeat(beta_to_angles(all_betas)[:, None], n_visible, axis=1)
    )
    centers = inference_centers[: len(TRAINING_BETAS)]
    widths = (
        np.ones(n_total)
        if args.ancillas == 0
        else np.concatenate((np.zeros(n_visible), np.ones(args.ancillas)))
    )
    rows: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    for width in args.half_width_pi:
        result = train_with_resampled_angles(
            f"curie_weiss_interval_{width:g}_seed{args.seed}",
            ansatz,
            centers,
            widths * width * np.pi,
            train_targets,
            validation_targets,
            args.seed,
            TrainingConfig(
                steps=args.steps,
                batch_size=args.batch_size,
                validation_interval=max(1, min(25, args.steps)),
                n_ops=args.n_ops,
                n_iqp_samples=args.n_iqp_samples,
                validation_n_ops=args.n_ops,
                validation_n_iqp_samples=args.validation_iqp_samples,
            ),
            validation_members=args.validation_members,
        )
        ensemble = np.mean(
            np.stack(
                [
                    marginalize_ancillas(
                        exact_iqp_probabilities_many(
                            result.best_params,
                            result.generators,
                            sample_interval_angles(
                                inference_centers,
                                widths * width * np.pi,
                                args.seed + 700_000 + member,
                            ),
                        ),
                        n_visible,
                        args.ancillas,
                    )
                    for member in range(args.ensemble_members)
                ]
            ),
            axis=0,
        )
        rows.extend(
            evaluate(
                all_betas,
                ensemble,
                n_visible,
                width,
                "probability_ensemble",
            )
        )
        fits.append(
            {
                "half_width_pi": float(width),
                "best_step": result.best_step,
                "best_validation_loss": result.best_validation_loss,
            }
        )
    return {
        "status": "prototype",
        "dataset": "Curie-Weiss",
        "n_qubits": n_total,
        "visible_qubits": n_visible,
        "ancillas": args.ancillas,
        "training_betas": TRAINING_BETAS.tolist(),
        "heldout_betas": HELDOUT_BETAS.tolist(),
        "fits": fits,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--ancillas", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--samples", type=int, default=1_500)
    parser.add_argument("--batch-size", type=int, default=75)
    parser.add_argument("--validation-samples", type=int, default=16_384)
    parser.add_argument("--n-ops", type=int, default=256)
    parser.add_argument("--n-iqp-samples", type=int, default=8_192)
    parser.add_argument("--validation-iqp-samples", type=int, default=16_384)
    parser.add_argument("--validation-members", type=int, default=1)
    parser.add_argument("--ensemble-members", type=int, default=8)
    parser.add_argument(
        "--half-width-pi", nargs="+", type=float, default=(0.0, 0.05, 0.10)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n_qubits < 2 or args.ancillas < 0 or args.steps < 1:
        parser.error("n-qubits must be >= 2, ancillas non-negative, and steps positive")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
