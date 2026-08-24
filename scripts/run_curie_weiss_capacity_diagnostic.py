#!/usr/bin/env python3
"""Measure a symmetry-aligned IQP capacity ceiling with direct exact TVD search."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.physical_representability import curie_weiss_target

DEFAULT_OUTPUT = Path(
    "results/curie-weiss-representability-v1/capacity-diagnostic.json"
)
BETAS = (0.5, 1.0, 1.5)


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parameters(n_qubits: int, n_parameters: int, phases: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.full(n_qubits, phases[0], dtype=np.float64),
            np.full(n_parameters - n_qubits, phases[1], dtype=np.float64),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--popsize", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable diagnostic: {args.output}")

    n_qubits = 16
    ansatz = complete_ansatz(n_qubits)
    angles = np.zeros(n_qubits, dtype=np.float64)
    bitstrings = enumerate_bitstrings(n_qubits)
    rows = []
    for beta in BETAS:
        target = curie_weiss_target(beta, n_qubits=n_qubits)
        target_profile = shell_probabilities(target.probabilities, bitstrings)

        def objective(
            phases: np.ndarray,
            reference: np.ndarray = target.probabilities,
        ) -> float:
            model = exact_iqp_probabilities(
                _parameters(n_qubits, ansatz.n_parameters, phases),
                ansatz.generators,
                angles,
            )
            return total_variation_distance(reference, model)

        result = differential_evolution(
            objective,
            bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],
            seed=20260813,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=True,
            tol=1e-8,
            workers=1,
        )
        parameters = _parameters(n_qubits, ansatz.n_parameters, result.x)
        model = exact_iqp_probabilities(parameters, ansatz.generators, angles)
        model_profile = shell_probabilities(model, bitstrings)
        rows.append(
            {
                "beta": beta,
                "one_body_phase": float(result.x[0]),
                "two_body_phase": float(result.x[1]),
                "exact_bitstring_tvd": total_variation_distance(
                    target.probabilities,
                    model,
                ),
                "magnetization_profile_tvd": total_variation_distance(
                    target_profile,
                    model_profile,
                ),
                "target_magnetization_profile": target_profile.tolist(),
                "model_magnetization_profile": model_profile.tolist(),
                "optimizer_evaluations": int(result.nfev),
                "optimizer_success": bool(result.success),
                "parameters_sha256": sha256(
                    parameters.astype("<f8").tobytes()
                ).hexdigest(),
            },
        )
        print(f"beta={beta:g} capacity TVD={rows[-1]['exact_bitstring_tvd']:.4f}")

    payload = {
        "schema_version": 1,
        "status": "capacity_diagnostic_not_training_evidence",
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": _revision(),
        "warning": (
            "direct TVD requires exact probabilities and is not the classically "
            "estimable training objective claimed by the paper"
        ),
        "relationship_to_primary_screen": (
            "same ordinary complete two-local IQP circuit, restricted to its "
            "exchange-symmetric parameter subfamily"
        ),
        "search": {
            "algorithm": "scipy differential_evolution with polishing",
            "seed": 20260813,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "bounds": [[-np.pi, np.pi], [-np.pi, np.pi]],
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(f"wrote {args.output} sha256={sha256(encoded.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
