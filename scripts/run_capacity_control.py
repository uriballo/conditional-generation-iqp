"""Run the matched-capacity m=0 control for the ancilla ladder.

Each rung keeps the 19 data qubits and the 190 complete two-local
generators, then adds deterministic 3-local ``Z_i Z_j Z_k`` generators so
the total parameter count matches one rung of the ancilla ladder
(m = 1, 2, 3, 5, 7). The control isolates the traced-out-ancilla mechanism
from raw parameter count under the identical protocol (five training
angles, four held-out angles, ten seeds, 2,000 steps).
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import jax.numpy as jnp
import networkx as nx
import numpy as np

from iqp_cond.ansatz import Ansatz, generators_from_graph
from iqp_cond.encoding import ANGLE_ENCODING
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.interpolation import (
    HELDOUT_ANGLE_FRACTIONS,
    TRAIN_ANGLE_FRACTIONS,
    broadcast_angles,
    interpolation_targets,
    sample_angle_targets,
)
from iqp_cond.training import TrainingConfig, train

MATCHED_PARAMETER_COUNTS = {1: 210, 2: 231, 3: 253, 5: 300, 7: 351}

THREE_LOCAL_SEED = 12345


def matched_capacity_ansatz(n_parameters: int, n_qubits: int = 19) -> Ansatz:
    """Return complete 2-local plus 3-local generators up to a parameter count."""
    two_local = np.asarray(generators_from_graph(nx.complete_graph(n_qubits)))
    n_two_local = two_local.shape[0]
    n_three_local = n_parameters - n_two_local
    if n_three_local < 0:
        raise ValueError(
            f"n_parameters must exceed the 2-local count {n_two_local}",
        )
    triples = list(combinations(range(n_qubits), 3))
    rng = np.random.default_rng(THREE_LOCAL_SEED)
    selected = [triples[i] for i in rng.permutation(len(triples))[:n_three_local]]
    three_local = np.zeros((n_three_local, n_qubits), dtype=np.float64)
    for row, triple in zip(three_local, selected, strict=True):
        row[list(triple)] = 1.0
    generators = np.concatenate([two_local, three_local], axis=0)
    return Ansatz(
        name=f"complete_3local_{n_parameters}",
        graph=nx.complete_graph(n_qubits),
        generators=jnp.asarray(generators, dtype=jnp.float64),
    )


def run_capacity_control(
    n_parameters: int,
    seed: int,
    config: TrainingConfig,
    n_data_qubits: int = 19,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train an m=0 matched-capacity model and evaluate exact 19-qubit metrics."""
    train_angles = np.pi * TRAIN_ANGLE_FRACTIONS
    heldout_angles = np.pi * HELDOUT_ANGLE_FRACTIONS
    ansatz = matched_capacity_ansatz(n_parameters, n_data_qubits)

    training_data = sample_angle_targets(
        train_angles,
        seed + 300_000,
        10_000,
        n_data_qubits,
    )
    validation_data = sample_angle_targets(
        train_angles,
        seed + 400_000,
        2_000,
        n_data_qubits,
    )

    model_name = f"capacity_control_{n_parameters}_complete_3local"
    result = train(
        model_name,
        ansatz,
        broadcast_angles(train_angles, n_data_qubits),
        training_data,
        validation_data,
        seed,
        config,
        ANGLE_ENCODING,
    )

    all_angles = np.concatenate([train_angles, heldout_angles])
    all_roles = ["train"] * len(train_angles) + ["heldout"] * len(heldout_angles)
    all_targets = interpolation_targets(all_angles, n_data_qubits)
    bitstrings = enumerate_bitstrings(n_data_qubits)
    model_probabilities = exact_iqp_probabilities_many(
        result.best_params,
        result.generators,
        broadcast_angles(all_angles, n_data_qubits),
        ANGLE_ENCODING,
    )

    points = []
    for angle, role, target, model_probs in zip(
        all_angles,
        all_roles,
        all_targets,
        model_probabilities,
        strict=True,
    ):
        target_probs = target.bitstring_probabilities(bitstrings)
        model_shells = shell_probabilities(model_probs, bitstrings)
        weights = np.arange(n_data_qubits + 1)
        mean_err = abs(float(target.actual_mean - np.dot(model_shells, weights)))
        std_err = abs(
            float(
                target.actual_std
                - np.sqrt(np.dot(model_shells, (weights - target.actual_mean) ** 2)),
            ),
        )
        points.append(
            {
                "angle": float(angle),
                "angle_fraction_pi": float(angle / np.pi),
                "role": role,
                "target_mean": float(target.mean),
                "model_mean": float(np.dot(model_shells, weights)),
                "shell_tvd": total_variation_distance(
                    target.shell_probabilities,
                    model_shells,
                ),
                "bitstring_tvd": total_variation_distance(target_probs, model_probs),
                "mean_absolute_error": mean_err,
                "std_absolute_error": std_err,
            },
        )

    heldout_points = [p for p in points if p["role"] == "heldout"]
    metrics = {
        "model_name": result.model_name,
        "n_ancillas": 0,
        "total_qubits": n_data_qubits,
        "n_parameters": ansatz.n_parameters,
        "scheme": "none",
        "encoding": ANGLE_ENCODING,
        "seed": seed,
        "heldout_shell_tvd": float(np.mean([p["shell_tvd"] for p in heldout_points])),
        "heldout_bitstring_tvd": float(
            np.mean([p["bitstring_tvd"] for p in heldout_points]),
        ),
        "train_shell_tvd_endpoint_a": points[0]["shell_tvd"],
        "train_shell_tvd_endpoint_b": points[4]["shell_tvd"],
        "train_bitstring_tvd_endpoint_a": points[0]["bitstring_tvd"],
        "train_bitstring_tvd_endpoint_b": points[4]["bitstring_tvd"],
        "points": points,
    }
    result_dict = {
        "model_name": result.model_name,
        "n_ancillas": 0,
        "seed": seed,
        "best_step": result.best_step,
        "best_params": np.asarray(result.best_params),
        "generators": np.asarray(result.generators),
    }
    return result_dict, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rungs",
        default="210,231,253,300,351",
        help="comma-separated matched parameter counts",
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ancilla-control/runs"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rungs = [int(x) for x in args.rungs.split(",")]
    if "-" in args.seeds:
        start, end = (int(p) for p in args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(p) for p in args.seeds.split(",")]

    args.output.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(steps=args.steps)

    for n_parameters in rungs:
        for seed in seeds:
            model_name = f"capacity_control_{n_parameters}_complete_3local"
            destination = args.output / f"{model_name}_seed{seed}.json"
            if args.skip_existing and destination.exists():
                print(f"skip {model_name} seed={seed}")
                continue

            started = time.monotonic()
            print(f"run {model_name} seed={seed}", flush=True)
            res_dict, metrics = run_capacity_control(
                n_parameters=n_parameters,
                seed=seed,
                config=config,
            )
            destination.write_text(json.dumps(metrics, indent=2) + "\n")
            np.savez(
                args.output / f"{model_name}_seed{seed}.npz",
                best_params=res_dict["best_params"],
                generators=res_dict["generators"],
            )
            elapsed = time.monotonic() - started
            print(
                f"done {model_name} seed={seed} sec={elapsed:.1f} "
                f"heldout_shell={metrics['heldout_shell_tvd']:.4f} "
                f"heldout_bit={metrics['heldout_bitstring_tvd']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
