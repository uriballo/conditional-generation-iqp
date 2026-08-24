"""Run the frozen bounded multidimensional conditioning feasibility study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
)
from iqp_cond.mmd import exact_conditional_mmd, exact_target_fouriers
from iqp_cond.multidimensional import (
    HELDOUT_CONDITIONS,
    N_FEASIBILITY_QUBITS,
    TRAIN_CONDITIONS,
    condition_angles,
    distribution_metrics,
    oracle_lookup,
    target_probability_matrix,
    targets_for_conditions,
)
from iqp_cond.training import TrainingConfig, train_exact


def _point(
    condition: np.ndarray,
    role: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    point: dict[str, object] = {
        "condition": np.asarray(condition, dtype=np.float64).tolist(),
        "role": role,
    }
    for key, value in metrics.items():
        if np.isscalar(value):
            point[key] = float(value)
        else:
            point[key] = np.asarray(value).tolist()
    return point


def _train(
    name: str,
    seed: int,
    config: TrainingConfig,
    ansatz,
    angles: np.ndarray,
    target_probabilities: np.ndarray,
):
    return train_exact(
        name,
        ansatz,
        angles,
        exact_target_fouriers(target_probabilities),
        seed,
        config,
    )


def _single_condition_screen(
    seed: int,
    config: TrainingConfig,
    ansatz,
    bitstrings: np.ndarray,
    target_probabilities: np.ndarray,
) -> tuple[list[dict[str, object]], list[object]]:
    points = []
    results = []
    for index, condition in enumerate(TRAIN_CONDITIONS):
        angles = condition_angles(condition[None, :], N_FEASIBILITY_QUBITS)
        result = _train(
            f"multidimensional_single_{index}",
            seed,
            config,
            ansatz,
            angles,
            target_probabilities[index : index + 1],
        )
        probabilities = exact_iqp_probabilities_many(
            result.best_params,
            result.generators,
            angles,
        )[0]
        points.append(
            _point(
                condition,
                "train",
                distribution_metrics(
                    targets_for_conditions(
                        condition[None, :],
                        N_FEASIBILITY_QUBITS,
                    )[0],
                    probabilities,
                    bitstrings,
                ),
            ),
        )
        results.append(result)
    return points, results


def _evaluate_points(
    conditions: np.ndarray,
    probabilities: np.ndarray,
    role: str,
    bitstrings: np.ndarray,
) -> list[dict[str, object]]:
    targets = targets_for_conditions(conditions, N_FEASIBILITY_QUBITS)
    return [
        _point(
            condition,
            role,
            distribution_metrics(target, model, bitstrings),
        )
        for condition, target, model in zip(
            conditions,
            targets,
            probabilities,
            strict=True,
        )
    ]


def _oracle_points(
    dense_training_probabilities: np.ndarray,
    bitstrings: np.ndarray,
) -> list[dict[str, object]]:
    points = []
    targets = targets_for_conditions(HELDOUT_CONDITIONS, N_FEASIBILITY_QUBITS)
    target_probabilities = target_probability_matrix(
        HELDOUT_CONDITIONS,
        N_FEASIBILITY_QUBITS,
    )
    for condition, target, target_probability in zip(
        HELDOUT_CONDITIONS,
        targets,
        target_probabilities,
        strict=True,
    ):
        lookup = oracle_lookup(
            dense_training_probabilities,
            target_probability,
            TRAIN_CONDITIONS,
        )
        best_probability = dense_training_probabilities[lookup["lookup_index"]]
        metrics = distribution_metrics(target, best_probability, bitstrings)
        points.append(
            _point(
                condition,
                "heldout",
                {
                    **metrics,
                    "lookup_index": lookup["lookup_index"],
                    "lookup_condition": lookup["lookup_condition"],
                },
            ),
        )
    return points


def _heldout_mmd(
    result,
    angles: np.ndarray,
    target_probabilities: np.ndarray,
    config: TrainingConfig,
) -> list[float]:
    value = exact_conditional_mmd(
        jnp.asarray(result.best_params),
        jnp.asarray(result.generators),
        jnp.asarray(angles),
        exact_target_fouriers(target_probabilities),
        config.sigmas,
    )
    return np.asarray(value.per_condition, dtype=np.float64).tolist()


def _condition_metric(
    points: list[dict[str, object]],
    condition: np.ndarray,
    key: str,
) -> float:
    condition = np.asarray(condition)
    for point in points:
        if np.array_equal(np.asarray(point["condition"]), condition):
            return float(point[key])
    raise KeyError(f"missing Condition {condition.tolist()}")


def _gate(
    screen: list[dict[str, object]],
    dense_train: list[dict[str, object]],
    dense_heldout: list[dict[str, object]],
    no_condition_train: list[dict[str, object]],
    no_condition_heldout: list[dict[str, object]],
    separate_train: list[dict[str, object]],
    oracle_heldout: list[dict[str, object]],
) -> dict[str, object]:
    screen_tvds = [float(point["bitstring_tvd"]) for point in screen]
    screen_pass = max(screen_tvds) <= 0.30 and np.median(screen_tvds) <= 0.20

    memorization_pass = all(
        dense["bitstring_tvd"] < no_condition["bitstring_tvd"]
        for dense, no_condition in zip(
            dense_train,
            no_condition_train,
            strict=True,
        )
    )
    dense_train_median = float(
        np.median([point["bitstring_tvd"] for point in dense_train]),
    )
    separate_train_median = float(
        np.median([point["bitstring_tvd"] for point in separate_train]),
    )
    sharing_capacity_pass = dense_train_median <= separate_train_median + 0.10

    lookup_differences = np.asarray(
        [
            dense["bitstring_tvd"] - oracle["bitstring_tvd"]
            for dense, oracle in zip(dense_heldout, oracle_heldout, strict=True)
        ],
    )
    interpolation_pass = (
        int(np.count_nonzero(lookup_differences <= 0.02)) >= 4
        and float(lookup_differences.max()) <= 0.05
    )

    mean_differences = np.asarray(
        [
            dense["mean_error"] - no_condition["mean_error"]
            for dense, no_condition in zip(
                dense_heldout,
                no_condition_heldout,
                strict=True,
            )
        ],
    )
    width_differences = np.asarray(
        [
            dense["std_error"] - no_condition["std_error"]
            for dense, no_condition in zip(
                dense_heldout,
                no_condition_heldout,
                strict=True,
            )
        ],
    )
    structural_pass = (
        int(np.count_nonzero(mean_differences < 0.0)) >= 3
        and int(np.count_nonzero(width_differences < 0.0)) >= 3
        and np.any(mean_differences[:3] < 0.0)
        and np.any(mean_differences[3:] < 0.0)
        and np.any(width_differences[:3] < 0.0)
        and np.any(width_differences[3:] < 0.0)
    )

    return {
        "single_condition_screen": bool(screen_pass),
        "memorization": bool(memorization_pass),
        "sharing_capacity": bool(sharing_capacity_pass),
        "interpolation": bool(interpolation_pass),
        "structural_cross": bool(structural_pass),
        "pass": bool(
            screen_pass
            and memorization_pass
            and sharing_capacity_pass
            and interpolation_pass
            and structural_pass
        ),
        "screen_max_bitstring_tvd": float(max(screen_tvds)),
        "screen_median_bitstring_tvd": float(np.median(screen_tvds)),
        "dense_train_median_bitstring_tvd": dense_train_median,
        "separate_train_median_bitstring_tvd": separate_train_median,
        "heldout_dense_minus_oracle_bitstring_tvd": lookup_differences.tolist(),
        "heldout_dense_minus_no_condition_mean_error": mean_differences.tolist(),
        "heldout_dense_minus_no_condition_std_error": width_differences.tolist(),
    }


def run_feasibility(
    seed: int,
    config: TrainingConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run one seed of the frozen exact feasibility setting."""
    ansatz = complete_ansatz(N_FEASIBILITY_QUBITS)
    bitstrings = enumerate_bitstrings(N_FEASIBILITY_QUBITS)
    train_targets = target_probability_matrix(
        TRAIN_CONDITIONS,
        N_FEASIBILITY_QUBITS,
    )
    heldout_targets = target_probability_matrix(
        HELDOUT_CONDITIONS,
        N_FEASIBILITY_QUBITS,
    )
    screen, screen_results = _single_condition_screen(
        seed,
        config,
        ansatz,
        bitstrings,
        train_targets,
    )
    if not (
        max(point["bitstring_tvd"] for point in screen) <= 0.30
        and np.median([point["bitstring_tvd"] for point in screen]) <= 0.20
    ):
        return (
            {
                "seed": seed,
                "status": "single_condition_screen_failed",
                "single_condition_screen": screen,
            },
            {
                f"single_{index}_best_params": result.best_params
                for index, result in enumerate(screen_results)
            },
        )

    all_conditions = np.concatenate([TRAIN_CONDITIONS, HELDOUT_CONDITIONS])
    all_angles = condition_angles(all_conditions, N_FEASIBILITY_QUBITS)
    dense_angles = condition_angles(TRAIN_CONDITIONS, N_FEASIBILITY_QUBITS)
    dense_result = _train(
        "multidimensional_dense",
        seed,
        config,
        ansatz,
        dense_angles,
        train_targets,
    )
    dense_probabilities = exact_iqp_probabilities_many(
        dense_result.best_params,
        dense_result.generators,
        all_angles,
    )

    center_angles = condition_angles(
        np.repeat([[0.5, 0.5]], len(TRAIN_CONDITIONS), axis=0),
        N_FEASIBILITY_QUBITS,
    )
    no_condition_result = _train(
        "multidimensional_no_condition",
        seed,
        config,
        ansatz,
        center_angles,
        train_targets,
    )
    no_condition_probabilities = exact_iqp_probabilities_many(
        no_condition_result.best_params,
        no_condition_result.generators,
        np.repeat(
            condition_angles(np.asarray([[0.5, 0.5]]), N_FEASIBILITY_QUBITS),
            len(all_conditions),
            axis=0,
        ),
    )

    separate_results = []
    separate_probabilities = []
    for index, condition in enumerate(TRAIN_CONDITIONS):
        angles = condition_angles(condition[None, :], N_FEASIBILITY_QUBITS)
        result = _train(
            f"multidimensional_separate_{index}",
            seed,
            config,
            ansatz,
            angles,
            train_targets[index : index + 1],
        )
        separate_results.append(result)
        separate_probabilities.append(
            exact_iqp_probabilities_many(
                result.best_params,
                result.generators,
                angles,
            )[0],
        )
    separate_probabilities = np.stack(separate_probabilities)

    dense_points = _evaluate_points(
        all_conditions[: len(TRAIN_CONDITIONS)],
        dense_probabilities[: len(TRAIN_CONDITIONS)],
        "train",
        bitstrings,
    )
    dense_points_heldout = _evaluate_points(
        HELDOUT_CONDITIONS,
        dense_probabilities[len(TRAIN_CONDITIONS) :],
        "heldout",
        bitstrings,
    )
    no_condition_points = _evaluate_points(
        TRAIN_CONDITIONS,
        no_condition_probabilities[: len(TRAIN_CONDITIONS)],
        "train",
        bitstrings,
    )
    no_condition_points_heldout = _evaluate_points(
        HELDOUT_CONDITIONS,
        no_condition_probabilities[len(TRAIN_CONDITIONS) :],
        "heldout",
        bitstrings,
    )
    separate_points = _evaluate_points(
        TRAIN_CONDITIONS,
        separate_probabilities,
        "train",
        bitstrings,
    )
    oracle_points = _oracle_points(
        dense_probabilities[: len(TRAIN_CONDITIONS)],
        bitstrings,
    )
    gate = _gate(
        screen,
        dense_points,
        dense_points_heldout,
        no_condition_points,
        no_condition_points_heldout,
        separate_points,
        oracle_points,
    )
    metrics = {
        "seed": seed,
        "status": "complete",
        "n_qubits": N_FEASIBILITY_QUBITS,
        "ansatz": ansatz.name,
        "n_parameters": ansatz.n_parameters,
        "train_conditions": TRAIN_CONDITIONS.tolist(),
        "heldout_conditions": HELDOUT_CONDITIONS.tolist(),
        "single_condition_screen": screen,
        "models": {
            "dense": {
                "train": dense_points,
                "heldout": dense_points_heldout,
                "heldout_exact_mmd": _heldout_mmd(
                    dense_result,
                    condition_angles(HELDOUT_CONDITIONS, N_FEASIBILITY_QUBITS),
                    heldout_targets,
                    config,
                ),
            },
            "no_condition": {
                "train": no_condition_points,
                "heldout": no_condition_points_heldout,
            },
            "separate": {"train": separate_points},
            "oracle_lookup": {"heldout": oracle_points},
        },
        "gate": gate,
        "training_config": asdict(config),
    }
    arrays = {
        "dense_best_params": dense_result.best_params,
        "no_condition_best_params": no_condition_result.best_params,
        **{
            f"separate_{index}_best_params": result.best_params
            for index, result in enumerate(separate_results)
        },
        **{
            f"single_{index}_best_params": result.best_params
            for index, result in enumerate(screen_results)
        },
    }
    return metrics, arrays


def _parse_seeds(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/multidimensional/feasibility.json"),
    )
    args = parser.parse_args()

    config = TrainingConfig(
        steps=args.steps,
        validation_interval=args.validation_interval,
    )
    seed_results = []
    array_results = {}
    for seed in _parse_seeds(args.seeds):
        result, arrays = run_feasibility(seed, config)
        seed_results.append(result)
        array_results.update(
            {f"seed{seed}_{name}": value for name, value in arrays.items()},
        )
        if result["status"] != "complete" or not result["gate"]["pass"]:
            break

    complete_results = [
        result for result in seed_results if result["status"] == "complete"
    ]
    all_gates_pass = bool(complete_results) and all(
        result["gate"]["pass"] for result in complete_results
    )
    multi_seed_pass = (
        len(complete_results) == len(_parse_seeds(args.seeds)) and all_gates_pass
    )
    output = {
        "study": "multidimensional_location_width_feasibility",
        "design_document": "docs/multidimensional-conditioning-design.md",
        "seeds_requested": _parse_seeds(args.seeds),
        "seeds_completed": [result["seed"] for result in seed_results],
        "multi_seed_gate": multi_seed_pass,
        "recommendation": (
            "proceed_to_frozen_ten_seed_study"
            if multi_seed_pass
            else "terminate_multidimensional_avenue"
        ),
        "results": seed_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if array_results:
        np.savez_compressed(args.output.with_suffix(".npz"), **array_results)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
