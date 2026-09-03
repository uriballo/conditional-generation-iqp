#!/usr/bin/env python3
"""Compare broadcast and interval-based condition encodings on the Gaussian task.

This is an exploratory screen, not part of the immutable Gaussian campaign.
The interval arm keeps the circuit, targets, optimizer, and parameter count
fixed.  It only replaces each broadcast row of angles by deterministic
per-qubit draws from a condition-specific interval.  The ensemble metrics
average exact output probabilities over independently drawn angle matrices;
they therefore describe an ensemble extension rather than one circuit input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.gaussian_data import gaussian_target
from iqp_cond.interval_encoding import (
    build_inference_ensemble,
    build_interval_encoding,
)
from iqp_cond.stochastic_interval import train_with_resampled_angles
from iqp_cond.training import TrainingConfig

TRAINING_CONDITIONS = np.arange(5, dtype=np.float64) / 4.0
HELDOUT_CONDITIONS = np.asarray([1, 3, 5, 7], dtype=np.float64) / 8.0


def _parse_seeds(value: str) -> tuple[int, ...]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return tuple(range(start, end + 1))
    return tuple(int(part) for part in value.split(","))


def _target_samples(
    conditions: np.ndarray,
    n_qubits: int,
    width: float,
    seed: int,
    count: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            gaussian_target(n_qubits, width, float(condition)).sample(rng, count)
            for condition in conditions
        ]
    ).astype(np.float64)


def _metrics(
    conditions: np.ndarray,
    probabilities: np.ndarray,
    n_qubits: int,
    width_factor: float,
    role: str | None,
) -> list[dict[str, object]]:
    bitstrings = enumerate_bitstrings(n_qubits)
    rows = []
    for condition, model in zip(conditions, probabilities, strict=True):
        target = gaussian_target(n_qubits, width_factor, float(condition))
        target_bits = target.bitstring_probabilities(bitstrings)
        rows.append(
            {
                "condition": float(condition),
                "role": role
                or (
                    "heldout"
                    if np.any(np.isclose(HELDOUT_CONDITIONS, condition))
                    else "training"
                ),
                "profile_tvd": total_variation_distance(
                    target.shell_probabilities, shell_probabilities(model, bitstrings)
                ),
                "bitstring_tvd": total_variation_distance(target_bits, model),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    n_qubits = args.n_qubits
    width_factor = args.width_factor
    ansatz = complete_ansatz(n_qubits)
    training_targets = _target_samples(
        TRAINING_CONDITIONS, n_qubits, width_factor, 300_000, 1_000
    )
    validation_targets = _target_samples(
        TRAINING_CONDITIONS, n_qubits, width_factor, 400_000, 500
    )
    all_conditions = np.concatenate((TRAINING_CONDITIONS, HELDOUT_CONDITIONS))
    rows: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []

    for interval_width in args.half_width_pi:
        for seed in _parse_seeds(args.seeds):
            angle_seed = seed + 1_000_000
            training_angles = build_interval_encoding(
                TRAINING_CONDITIONS,
                n_qubits,
                half_width=interval_width * np.pi,
                seed=angle_seed,
            )
            result = train_with_resampled_angles(
                f"interval_width_{interval_width:g}_seed{seed}",
                ansatz,
                training_angles,
                interval_width * np.pi,
                training_targets,
                validation_targets,
                seed,
                TrainingConfig(steps=args.steps),
                validation_members=args.validation_members,
            )
            heldout_angles = build_interval_encoding(
                HELDOUT_CONDITIONS,
                n_qubits,
                half_width=interval_width * np.pi,
                seed=angle_seed + 1,
            )
            fixed_angles = np.concatenate((training_angles, heldout_angles))
            fixed_probabilities = exact_iqp_probabilities_many(
                result.best_params, result.generators, fixed_angles
            )
            fixed_rows = _metrics(
                all_conditions,
                fixed_probabilities,
                n_qubits,
                width_factor,
                None,
            )
            for row in fixed_rows:
                row.update(
                    {
                        "arm": "single_fresh_draw",
                        "half_width_pi": float(interval_width),
                        "seed": seed,
                    }
                )
            rows.extend(fixed_rows)

            ensemble_angles = build_inference_ensemble(
                all_conditions,
                n_qubits,
                args.ensemble_members,
                half_width=interval_width * np.pi,
                seed=angle_seed + 2,
            )
            ensemble_probabilities = np.mean(
                np.stack(
                    [
                        exact_iqp_probabilities_many(
                            result.best_params, result.generators, member_angles
                        )
                        for member_angles in ensemble_angles
                    ]
                ),
                axis=0,
            )
            ensemble_rows = _metrics(
                all_conditions,
                ensemble_probabilities,
                n_qubits,
                width_factor,
                None,
            )
            for row in ensemble_rows:
                row.update(
                    {
                        "arm": "probability_ensemble",
                        "half_width_pi": float(interval_width),
                        "ensemble_members": args.ensemble_members,
                        "seed": seed,
                    }
                )
            rows.extend(ensemble_rows)
            fits.append(
                {
                    "half_width_pi": float(interval_width),
                    "seed": seed,
                    "best_step": result.best_step,
                    "best_validation_loss": result.best_validation_loss,
                }
            )

    return {
        "status": "prototype",
        "target_family": "Gaussian Hamming-profile",
        "n_qubits": n_qubits,
        "width_factor": width_factor,
        "training_conditions": TRAINING_CONDITIONS.tolist(),
        "heldout_conditions": HELDOUT_CONDITIONS.tolist(),
        "half_width_units": "pi radians",
        "fits": fits,
        "rows": rows,
        "interpretation": {
            "single_fresh_draw": "one fresh product-state input per generated sample",
            "probability_ensemble": (
                "mean of exact output probabilities over fresh angle draws"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--width-factor", type=float, default=0.5)
    parser.add_argument(
        "--half-width-pi",
        nargs="+",
        type=float,
        default=(0.0, 0.05, 0.10),
        help="half-widths in multiples of pi",
    )
    parser.add_argument("--ensemble-members", type=int, default=8)
    parser.add_argument("--validation-members", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.n_qubits < 1
        or args.ensemble_members < 1
        or args.validation_members < 1
    ):
        parser.error("steps, n-qubits, and member counts must be positive")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
