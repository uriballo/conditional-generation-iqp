#!/usr/bin/env python3
"""Train independent regular IQP circuits on finite Curie--Weiss laws."""

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
    enumerate_bitstrings,
    exact_iqp_probabilities,
    fwht,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.physical_representability import (
    curie_weiss_target,
    expectation_errors,
)
from iqp_cond.training import TrainingConfig, train_exact

DEFAULT_OUTPUT = Path("results/curie-weiss-representability-v1/results.json")
BETAS = (0.5, 1.0, 1.5)


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1 or args.seeds < 1:
        parser.error("steps and seeds must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable pilot: {args.output}")

    n_qubits = 16
    bitstrings = enumerate_bitstrings(n_qubits)
    ansatz = complete_ansatz(n_qubits)
    angles = np.zeros((1, n_qubits), dtype=np.float64)
    config = TrainingConfig(
        steps=args.steps,
        validation_interval=25,
        learning_rate=0.01,
        final_learning_rate=0.001,
        sigmas=(1.0, 2.0, 4.0),
    )
    rows = []
    targets = {beta: curie_weiss_target(beta, n_qubits=n_qubits) for beta in BETAS}
    for beta, target in targets.items():
        target_profile = shell_probabilities(target.probabilities, bitstrings)
        target_fourier = fwht(target.probabilities).real[None, :]
        for seed in range(args.seeds):
            print(f"train beta={beta:g} seed={seed}", flush=True)
            started = time.monotonic()
            result = train_exact(
                f"curie_weiss_beta_{beta:g}",
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
            model_profile = shell_probabilities(model, bitstrings)
            row = {
                "beta": beta,
                "regime": (
                    "disordered"
                    if beta < 1.0
                    else "critical"
                    if beta == 1.0
                    else "ordered"
                ),
                "optimizer_seed": seed,
                "best_step": result.best_step,
                "exact_mmd_squared": result.best_validation_loss,
                "exact_bitstring_tvd": total_variation_distance(
                    target.probabilities,
                    model,
                ),
                "magnetization_profile_tvd": total_variation_distance(
                    target_profile,
                    model_profile,
                ),
                "expectation_errors": expectation_errors(target, model),
                "runtime_seconds": time.monotonic() - started,
                "parameters_sha256": sha256(
                    np.asarray(result.best_params, dtype="<f8").tobytes(),
                ).hexdigest(),
                "model_probabilities_sha256": sha256(
                    np.asarray(model, dtype="<f8").tobytes(),
                ).hexdigest(),
                "model_magnetization_profile": model_profile.tolist(),
            }
            rows.append(row)
            print(
                f"  raw TVD={row['exact_bitstring_tvd']:.4f} "
                f"profile TVD={row['magnetization_profile_tvd']:.4f} "
                f"MMD2={row['exact_mmd_squared']:.6f}",
                flush=True,
            )

    summaries = []
    for beta in BETAS:
        selected = [row for row in rows if row["beta"] == beta]
        summary: dict[str, object] = {
            "beta": beta,
            "regime": selected[0]["regime"],
        }
        for metric in (
            "exact_bitstring_tvd",
            "magnetization_profile_tvd",
            "exact_mmd_squared",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[metric] = {
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
            }
        summaries.append(summary)

    payload = {
        "schema_version": 1,
        "status": "preliminary_nonpublication_pilot",
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": _revision(),
        "scientific_question": (
            "Can one independently trained regular complete two-local IQP circuit "
            "learn each representative Curie--Weiss law before conditioning?"
        ),
        "model": {
            "n_qubits": n_qubits,
            "n_parameters": ansatz.n_parameters,
            "input": "standard zero-angle IQP product input",
            "target_hamiltonian": "H=-J(sum_i s_i)^2/(2n), J=1, h=0",
            "critical_inverse_temperature": 1.0,
            "betas": list(BETAS),
        },
        "protocol": {
            **asdict(config),
            "objective": "exact_population_multiscale_Hamming_MMD_squared",
            "checkpoint_rule": "minimum exact objective at 25-update intervals",
            "evaluation": "exact enumeration of all 65536 bitstrings",
            "primary_metric": "exact_bitstring_tvd",
            "secondary_metric": "magnetization_profile_tvd",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "targets": [
            {
                "beta": beta,
                "sha256": target.digest,
                "magnetization_profile": shell_probabilities(
                    target.probabilities,
                    bitstrings,
                ).tolist(),
            }
            for beta, target in targets.items()
        ],
        "rows": rows,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(f"wrote {args.output} sha256={sha256(encoded.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
