#!/usr/bin/env python3
"""Search the exchange-symmetric subfamily of a regular complete IQP circuit."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import differential_evolution

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities,
    fwht,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.mmd import mmd_mask_weights, mmd_operator_probability
from iqp_cond.physical_representability import (
    curie_weiss_target,
    expectation_errors,
)

DEFAULT_OUTPUT = Path(
    "results/curie-weiss-representability-v1/symmetry-diagnostic.json",
)
BETAS = (0.5, 1.0, 1.5)


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def symmetric_parameters(
    n_qubits: int,
    n_parameters: int,
    one_body_phase: float,
    two_body_phase: float,
) -> np.ndarray:
    """Tie equal one-body and equal two-body phases without changing the circuit."""
    if n_parameters <= n_qubits:
        raise ValueError("complete two-local circuit must contain two-body phases")
    return np.concatenate(
        (
            np.full(n_qubits, one_body_phase, dtype=np.float64),
            np.full(n_parameters - n_qubits, two_body_phase, dtype=np.float64),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--popsize", type=int, default=15)
    args = parser.parse_args()
    if args.maxiter < 1 or args.popsize < 1:
        parser.error("maxiter and popsize must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable diagnostic: {args.output}")

    n_qubits = 16
    ansatz = complete_ansatz(n_qubits)
    angles = np.zeros(n_qubits, dtype=np.float64)
    bitstrings = enumerate_bitstrings(n_qubits)
    mask_weights = mmd_mask_weights(n_qubits)
    kernel_weights = np.mean(
        [
            (probability**mask_weights)
            * ((1.0 - probability) ** (n_qubits - mask_weights))
            for sigma in (1.0, 2.0, 4.0)
            for probability in (mmd_operator_probability(sigma),)
        ],
        axis=0,
    )
    rows = []
    for beta in BETAS:
        target = curie_weiss_target(beta, n_qubits=n_qubits)
        target_fourier = fwht(target.probabilities).real
        target_profile = shell_probabilities(target.probabilities, bitstrings)

        def objective(
            phases: np.ndarray,
            reference_fourier: np.ndarray = target_fourier,
        ) -> float:
            parameters = symmetric_parameters(
                n_qubits,
                ansatz.n_parameters,
                phases[0],
                phases[1],
            )
            model = exact_iqp_probabilities(
                parameters,
                ansatz.generators,
                angles,
            )
            difference = fwht(model).real - reference_fourier
            return float(np.sum(kernel_weights * difference**2))

        result = differential_evolution(
            objective,
            bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],
            seed=20260813,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=True,
            tol=1e-10,
            workers=1,
        )
        parameters = symmetric_parameters(
            n_qubits,
            ansatz.n_parameters,
            result.x[0],
            result.x[1],
        )
        model = exact_iqp_probabilities(parameters, ansatz.generators, angles)
        model_profile = shell_probabilities(model, bitstrings)
        row = {
            "beta": beta,
            "regime": (
                "disordered" if beta < 1.0 else "critical" if beta == 1.0 else "ordered"
            ),
            "one_body_phase": float(result.x[0]),
            "two_body_phase": float(result.x[1]),
            "exact_mmd_squared": float(result.fun),
            "exact_bitstring_tvd": total_variation_distance(
                target.probabilities,
                model,
            ),
            "magnetization_profile_tvd": total_variation_distance(
                target_profile,
                model_profile,
            ),
            "expectation_errors": expectation_errors(target, model),
            "target_magnetization_profile": target_profile.tolist(),
            "model_magnetization_profile": model_profile.tolist(),
            "optimizer_evaluations": int(result.nfev),
            "optimizer_success": bool(result.success),
            "parameters_sha256": sha256(parameters.astype("<f8").tobytes()).hexdigest(),
        }
        rows.append(row)
        print(
            f"beta={beta:g} TVD={row['exact_bitstring_tvd']:.4f} "
            f"MMD2={row['exact_mmd_squared']:.8f}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "status": "preliminary_nonpublication_symmetry_diagnostic",
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": _revision(),
        "relationship_to_primary_screen": (
            "same ordinary 16-qubit complete two-local IQP circuit; restricts "
            "optimization to equal one-body and equal two-body phases"
        ),
        "scientific_role": (
            "separate representability from failure of random-start 136-parameter "
            "optimization to preserve exchange symmetry"
        ),
        "objective": "exact population Hamming-MMD squared, bandwidths (1,2,4)",
        "global_search": {
            "algorithm": "scipy differential_evolution with polishing",
            "seed": 20260813,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "bounds": [[-np.pi, np.pi], [-np.pi, np.pi]],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(f"wrote {args.output} sha256={sha256(encoded.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
