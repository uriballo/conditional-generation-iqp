"""Calibrate Monte Carlo budgets against the preregistered accuracy criteria."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond import (
    GaussianHammingTarget,
    complete_ansatz,
    conditional_mmd,
    estimate_pauli_z,
    exact_iqp_probabilities,
)
from iqp_cond.exact import enumerate_bitstrings
from iqp_cond.training import initialize_parameters


def expectation_calibration(repeats: int) -> list[dict[str, float]]:
    """Measure expectation RMSE against exact six-qubit probabilities."""
    rng = np.random.default_rng(41)
    ansatz = complete_ansatz(6)
    params = initialize_parameters(ansatz.n_parameters, 41)
    angles = rng.uniform(0.0, np.pi / 4.0, size=6)
    ops = rng.integers(0, 2, size=(96, 6))
    probabilities = exact_iqp_probabilities(params, ansatz.generators, angles)
    bitstrings = enumerate_bitstrings(6)
    characters = 1.0 - 2.0 * ((bitstrings @ ops.T) % 2)
    exact = probabilities @ characters

    rows = []
    for n_samples in (128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384):
        estimates = []
        for repeat in range(repeats):
            result = estimate_pauli_z(
                params,
                ansatz.generators,
                jnp.asarray(ops),
                jnp.asarray(angles),
                jax.random.PRNGKey(10_000 + repeat),
                n_samples,
            )
            estimates.append(np.asarray(result.values[0]))
        errors = np.asarray(estimates) - exact[None, :]
        rows.append(
            {
                "n_iqp_samples": n_samples,
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "max_abs_bias": float(np.max(np.abs(errors.mean(axis=0)))),
            },
        )
    return rows


def mmd_calibration(repeats: int) -> list[dict[str, float]]:
    """Measure Conditional-MMD variation at representative 19-qubit parameters."""
    rng = np.random.default_rng(73)
    n_qubits = 19
    ansatz = complete_ansatz(n_qubits)
    targets = [
        GaussianHammingTarget.create(n_qubits, n_qubits / 2, sqrt(n_qubits) / 2),
        GaussianHammingTarget.create(n_qubits, n_qubits / 4, sqrt(n_qubits) / 2),
    ]
    target_batch = jnp.asarray(
        np.stack([target.sample(rng, 256) for target in targets]),
        dtype=jnp.float64,
    )
    angles = jnp.asarray(
        np.stack(
            [
                np.zeros(n_qubits),
                np.full(n_qubits, np.pi / 4),
            ],
        ),
    )
    params = initialize_parameters(ansatz.n_parameters, 73)
    rows = []
    for n_ops, n_samples in (
        (64, 512),
        (128, 1_024),
        (256, 2_048),
        (512, 4_096),
        (256, 16_384),
    ):
        values = []
        for repeat in range(repeats):
            result = conditional_mmd(
                params,
                ansatz.generators,
                angles,
                target_batch,
                (1.0, 2.0, 4.0),
                n_ops,
                n_samples,
                jax.random.PRNGKey(20_000 + repeat),
            )
            values.append(float(result.loss))
        values_array = np.asarray(values)
        mean = float(values_array.mean())
        std = float(values_array.std(ddof=1))
        rows.append(
            {
                "n_ops": n_ops,
                "n_iqp_samples": n_samples,
                "mean": mean,
                "std": std,
                "coefficient_of_variation": std / abs(mean),
            },
        )
    return rows


def main() -> None:
    """Run both calibration studies and save a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/calibration.json"),
    )
    args = parser.parse_args()

    report = {
        "criteria": {
            "expectation_rmse_max": 0.01,
            "conditional_mmd_cv_max": 0.05,
        },
        "expectation": expectation_calibration(args.repeats),
        "conditional_mmd": mmd_calibration(args.repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
