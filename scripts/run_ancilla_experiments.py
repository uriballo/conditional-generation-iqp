"""Run ancilla-assisted conditional IQP Born machine experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iqp_cond.ancilla import (
    ancilla_ansatz,
    ancilla_condition_angles,
    ancilla_only_condition_angles,
    exact_ancilla_marginal_probabilities,
    marginalize_ancillas,
)
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


def run_ancilla_experiment(
    n_ancillas: int,
    seed: int,
    config: TrainingConfig,
    scheme: str = "broadcast_all",
    encoding: str = ANGLE_ENCODING,
    n_data_qubits: int = 19,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train an ancilla-assisted IQP model and evaluate 19-qubit marginal metrics."""
    train_angles = np.pi * TRAIN_ANGLE_FRACTIONS
    heldout_angles = np.pi * HELDOUT_ANGLE_FRACTIONS
    ansatz = ancilla_ansatz(n_data_qubits, n_ancillas)

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

    data_condition_angles = broadcast_angles(train_angles, n_data_qubits)
    if scheme == "ancilla_only":
        full_condition_angles = ancilla_only_condition_angles(
            broadcast_angles(train_angles, n_ancillas),
            n_data_qubits,
        )
    else:
        full_condition_angles = ancilla_condition_angles(
            data_condition_angles,
            n_ancillas,
            scheme,
        )

    model_name = f"ancilla_{n_ancillas}_{scheme}_complete"
    result = train(
        model_name,
        ansatz,
        full_condition_angles,
        training_data,
        validation_data,
        seed,
        config,
        encoding,
    )

    all_angles = np.concatenate([train_angles, heldout_angles])
    all_roles = ["train"] * len(train_angles) + ["heldout"] * len(heldout_angles)
    all_targets = interpolation_targets(all_angles, n_data_qubits)
    bitstrings = enumerate_bitstrings(n_data_qubits)

    all_data_angles = broadcast_angles(all_angles, n_data_qubits)
    if scheme == "ancilla_only":
        all_condition_angles = ancilla_only_condition_angles(
            broadcast_angles(all_angles, n_ancillas),
            n_data_qubits,
        )
        joint_probabilities = exact_iqp_probabilities_many(
            result.best_params,
            result.generators,
            all_condition_angles,
            encoding,
        )
        marginal_probabilities = marginalize_ancillas(
            joint_probabilities,
            n_data_qubits,
            n_ancillas,
        )
    else:
        marginal_probabilities = exact_ancilla_marginal_probabilities(
            result.best_params,
            result.generators,
            all_data_angles,
            n_data_qubits=n_data_qubits,
            n_ancillas=n_ancillas,
            scheme=scheme,
            encoding=encoding,
        )

    points = []
    for angle, role, target, model_probs in zip(
        all_angles,
        all_roles,
        all_targets,
        marginal_probabilities,
        strict=True,
    ):
        target_probs = target.bitstring_probabilities(bitstrings)
        model_shells = shell_probabilities(model_probs, bitstrings)
        weights = np.arange(n_data_qubits + 1)
        mean_err = abs(float(target.actual_mean - np.dot(model_shells, weights)))
        std_err = abs(
            float(
                target.actual_std
                - np.sqrt(
                    np.dot(model_shells, (weights - target.actual_mean) ** 2),
                ),
            ),
        )

        point = {
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
        }
        points.append(point)

    heldout_points = [p for p in points if p["role"] == "heldout"]
    metrics = {
        "model_name": result.model_name,
        "n_ancillas": n_ancillas,
        "total_qubits": n_data_qubits + n_ancillas,
        "n_parameters": ansatz.n_parameters,
        "scheme": scheme,
        "encoding": encoding,
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
        "n_ancillas": n_ancillas,
        "seed": seed,
        "best_step": result.best_step,
        "best_params": np.asarray(result.best_params),
        "generators": np.asarray(result.generators),
    }

    return result_dict, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ancillas",
        default="1,2,3",
        help="comma-separated ancilla counts",
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument(
        "--scheme",
        default="broadcast_all",
        choices=["broadcast_all", "fixed_zero", "ancilla_only"],
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=Path("results/ancilla/runs"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ancilla_counts = [int(x) for x in args.ancillas.split(",")]
    if "-" in args.seeds:
        start, end = (int(p) for p in args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(p) for p in args.seeds.split(",")]

    args.output.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(steps=args.steps)

    for m in ancilla_counts:
        for seed in seeds:
            model_name = f"ancilla_{m}_{args.scheme}_complete"
            destination = args.output / f"{model_name}_seed{seed}.json"
            if args.skip_existing and destination.exists():
                print(f"skip {model_name} seed={seed}")
                continue

            started = time.monotonic()
            print(f"run {model_name} seed={seed}", flush=True)
            res_dict, metrics = run_ancilla_experiment(
                n_ancillas=m,
                seed=seed,
                config=config,
                scheme=args.scheme,
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
