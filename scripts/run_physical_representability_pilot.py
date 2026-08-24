#!/usr/bin/env python3
"""Optimistic separate-IQP representability screen for three physical datasets."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import jax
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    exact_iqp_probabilities,
    fwht,
    total_variation_distance,
)
from iqp_cond.physical_representability import (
    PhysicalTarget,
    expectation_errors,
    ferromagnetic_target,
    lattice_gas_target,
    mfc_outgoing_energy_target,
)
from iqp_cond.training import TrainingConfig, train_exact

DEFAULT_OUTPUT = Path("results/physical-representability-pilot-v1/results.json")


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _target_rows(cache: Path) -> list[PhysicalTarget]:
    return [
        *(ferromagnetic_target(beta) for beta in (0.2, 0.4406868, 0.7)),
        *(lattice_gas_target(mu) for mu in (-2.5, -2.0, -1.5)),
        *(mfc_outgoing_energy_target(energy, cache) for energy in (50, 125, 200)),
    ]


def _run_target(
    target: PhysicalTarget,
    seed: int,
    config: TrainingConfig,
) -> dict[str, object]:
    ansatz = complete_ansatz(target.n_qubits)
    angles = np.zeros((1, target.n_qubits), dtype=np.float64)
    target_fourier = fwht(target.probabilities).real[None, :]
    started = time.monotonic()
    result = train_exact(
        f"{target.dataset}_{target.condition_value:g}",
        ansatz,
        angles,
        target_fourier,
        seed,
        config,
    )
    model = exact_iqp_probabilities(
        result.best_params,
        result.generators,
        angles[0],
    )
    return {
        "dataset": target.dataset,
        "condition_name": target.condition_name,
        "condition_value": target.condition_value,
        "n_qubits": target.n_qubits,
        "n_parameters": ansatz.n_parameters,
        "optimizer_seed": seed,
        "target_sha256": target.digest,
        "target_provenance": target.provenance,
        "best_step": result.best_step,
        "exact_mmd_squared": result.best_validation_loss,
        "exact_bitstring_tvd": total_variation_distance(
            target.probabilities,
            model,
        ),
        "expectation_errors": expectation_errors(target, model),
        "runtime_seconds": time.monotonic() - started,
        "parameters_sha256": sha256(
            np.asarray(result.best_params, dtype="<f8").tobytes(),
        ).hexdigest(),
        "model_probabilities": model.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=Path(".scratch/mfc-zenodo"))
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1 or args.seeds < 1:
        parser.error("steps and seeds must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable pilot: {args.output}")

    config = TrainingConfig(
        steps=args.steps,
        validation_interval=25,
        learning_rate=0.01,
        final_learning_rate=0.001,
        sigmas=(1.0, 2.0, 4.0),
    )
    rows = []
    for target in _target_rows(args.cache):
        for seed in range(args.seeds):
            print(
                f"train {target.dataset} {target.condition_value:g} seed={seed}",
                flush=True,
            )
            row = _run_target(target, seed, config)
            rows.append(row)
            print(
                f"  TVD={row['exact_bitstring_tvd']:.4f} "
                f"MMD2={row['exact_mmd_squared']:.5f}",
                flush=True,
            )

    summaries = []
    keys = sorted({(row["dataset"], row["condition_value"]) for row in rows})
    for dataset, condition in keys:
        values = [
            float(row["exact_bitstring_tvd"])
            for row in rows
            if row["dataset"] == dataset and row["condition_value"] == condition
        ]
        summaries.append(
            {
                "dataset": dataset,
                "condition_value": condition,
                "minimum_tvd": min(values),
                "median_tvd": float(np.median(values)),
                "maximum_tvd": max(values),
            },
        )

    payload = {
        "schema_version": 1,
        "status": "preliminary_nonpublication_pilot",
        "purpose": (
            "optimistic separate-model representability screen before any "
            "conditional training"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": _revision(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "protocol": {
            **asdict(config),
            "input_state": "zero_angles_standard_IQP_input",
            "objective": "exact_population_multiscale_Hamming_MMD_squared",
            "evaluation": "exact_enumeration",
            "interpretation": (
                "failure is informative; success only licenses a later "
                "finite-sample conditioning experiment"
            ),
        },
        "rows": rows,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(f"wrote {args.output} sha256={sha256(encoded.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
