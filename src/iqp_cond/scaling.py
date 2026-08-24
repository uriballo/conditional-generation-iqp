"""Condition-count scaling protocol, evaluation, and aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import pi, sqrt
from pathlib import Path

import numpy as np

from .ansatz import Ansatz, complete_ansatz
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities,
    shell_probabilities,
    total_variation_distance,
)
from .targets import GaussianHammingTarget
from .training import TrainingConfig, TrainingResult, train

SCALING_CONDITION_COUNTS = (2, 3, 5, 9)
SCALING_TRACKS = ("fixed_total", "fixed_per_condition")
TOTAL_TRAINING_SAMPLES = 18_000
TOTAL_VALIDATION_SAMPLES = 3_600
PER_CONDITION_TRAINING_SAMPLES = 10_000
PER_CONDITION_VALIDATION_SAMPLES = 2_000
TOTAL_MINIBATCH_SIZE = 450


@dataclass(frozen=True)
class ScalingBudget:
    """Data and balanced-minibatch budget for one K/track configuration."""

    track: str
    n_conditions: int
    n_training_per_condition: int
    n_validation_per_condition: int
    total_training_samples: int
    total_validation_samples: int
    batch_size_per_condition: int
    total_batch_size: int
    optimizer_updates: int


@dataclass(frozen=True)
class ParameterAccounting:
    """Trainable-parameter counts for shared and separate models."""

    n_conditions: int
    shared_parameters: int
    no_condition_parameters: int
    separate_parameters_per_model: int
    separate_bank_parameters: int
    shared_fraction_of_separate: float


def _validate_n_conditions(n_conditions: int) -> None:
    if n_conditions not in SCALING_CONDITION_COUNTS:
        raise ValueError(
            f"n_conditions must be one of {SCALING_CONDITION_COUNTS}; "
            f"got {n_conditions}",
        )


def condition_angle_fractions(n_conditions: int) -> np.ndarray:
    """Return the nested scalar grid as fractions of pi."""
    _validate_n_conditions(n_conditions)
    return np.arange(n_conditions, dtype=np.float64) / (4.0 * (n_conditions - 1))


def condition_angles(
    n_conditions: int,
    n_qubits: int = 19,
) -> np.ndarray:
    """Broadcast the frozen scalar grid over all qubits."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    fractions = condition_angle_fractions(n_conditions)
    return np.repeat((pi * fractions)[:, None], n_qubits, axis=1)


def nested_condition_grids(
    n_qubits: int = 19,
) -> dict[int, np.ndarray]:
    """Return all frozen grids, useful for audits and protocol tests."""
    return {
        n_conditions: condition_angles(n_conditions, n_qubits)
        for n_conditions in SCALING_CONDITION_COUNTS
    }


def scaling_budget(
    n_conditions: int,
    track: str,
    *,
    optimizer_updates: int = 2_000,
) -> ScalingBudget:
    """Construct the documented data and balanced-minibatch budget."""
    _validate_n_conditions(n_conditions)
    if track not in SCALING_TRACKS:
        raise ValueError(f"track must be one of {SCALING_TRACKS}; got {track!r}")
    if optimizer_updates < 1:
        raise ValueError("optimizer_updates must be positive")
    if track == "fixed_total":
        total_training = TOTAL_TRAINING_SAMPLES
        total_validation = TOTAL_VALIDATION_SAMPLES
        n_training = total_training // n_conditions
        n_validation = total_validation // n_conditions
    else:
        n_training = PER_CONDITION_TRAINING_SAMPLES
        n_validation = PER_CONDITION_VALIDATION_SAMPLES
        total_training = n_conditions * n_training
        total_validation = n_conditions * n_validation
    if total_training % n_conditions or total_validation % n_conditions:
        raise ValueError("frozen total budgets must divide evenly across Conditions")
    if TOTAL_MINIBATCH_SIZE % n_conditions:
        raise ValueError("frozen total minibatch must divide evenly across Conditions")
    batch_size = TOTAL_MINIBATCH_SIZE // n_conditions
    return ScalingBudget(
        track=track,
        n_conditions=n_conditions,
        n_training_per_condition=n_training,
        n_validation_per_condition=n_validation,
        total_training_samples=total_training,
        total_validation_samples=total_validation,
        batch_size_per_condition=batch_size,
        total_batch_size=n_conditions * batch_size,
        optimizer_updates=optimizer_updates,
    )


def targets(
    n_conditions: int,
    n_qubits: int = 19,
) -> tuple[GaussianHammingTarget, ...]:
    """Construct the fixed-width Gaussian target for every grid Condition."""
    angles = condition_angle_fractions(n_conditions) * pi
    width = sqrt(n_qubits) / 2.0
    return tuple(
        GaussianHammingTarget.create(
            n_qubits,
            n_qubits / 2.0 - n_qubits * angle / pi,
            width,
        )
        for angle in angles
    )


def sample_scaling_datasets(
    seed: int,
    budget: ScalingBudget,
    n_qubits: int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample balanced, disjoint Training and Validation arrays.

    The fixed-per-Condition pools are generated first and the fixed-total
    track takes a prefix. This makes the two tracks comparable without ever
    sharing a Training sample with Validation.
    """
    if budget.n_conditions not in SCALING_CONDITION_COUNTS:
        raise ValueError("budget has an unsupported Condition count")
    rng = np.random.default_rng(seed + 100_000)
    target_pair = targets(budget.n_conditions, n_qubits)
    training_pool = np.stack(
        [
            target.sample(rng, PER_CONDITION_TRAINING_SAMPLES)
            for target in target_pair
        ],
    )
    validation_pool = np.stack(
        [
            target.sample(rng, PER_CONDITION_VALIDATION_SAMPLES)
            for target in target_pair
        ],
    )
    return (
        training_pool[:, : budget.n_training_per_condition],
        validation_pool[:, : budget.n_validation_per_condition],
    )


def parameter_accounting(
    n_conditions: int,
    ansatz: Ansatz | None = None,
) -> ParameterAccounting:
    """Return explicit shared-versus-bank trainable parameter counts."""
    _validate_n_conditions(n_conditions)
    if ansatz is None:
        ansatz = complete_ansatz()
    parameters = ansatz.n_parameters
    return ParameterAccounting(
        n_conditions=n_conditions,
        shared_parameters=parameters,
        no_condition_parameters=parameters,
        separate_parameters_per_model=parameters,
        separate_bank_parameters=n_conditions * parameters,
        shared_fraction_of_separate=parameters / (n_conditions * parameters),
    )


def _weight_moments(shells: np.ndarray) -> tuple[float, float]:
    weights = np.arange(shells.size, dtype=np.float64)
    mean = float(weights @ shells)
    variance = float(((weights - mean) ** 2) @ shells)
    return mean, sqrt(max(variance, 0.0))


def _condition_metrics(
    condition: int,
    angle_fraction: float,
    target: GaussianHammingTarget,
    model_probabilities: np.ndarray,
    bitstrings: np.ndarray,
) -> dict[str, object]:
    target_probabilities = target.bitstring_probabilities(bitstrings)
    model_shells = shell_probabilities(model_probabilities, bitstrings)
    model_mean, model_std = _weight_moments(model_shells)
    return {
        "condition": condition,
        "angle_fraction_pi": float(angle_fraction),
        "target_mean": target.actual_mean,
        "model_mean": model_mean,
        "mean_absolute_error": abs(model_mean - target.actual_mean),
        "target_std": target.actual_std,
        "model_std": model_std,
        "std_absolute_error": abs(model_std - target.actual_std),
        "bitstring_tvd": total_variation_distance(
            target_probabilities,
            model_probabilities,
        ),
        "profile_tvd": total_variation_distance(
            target.shell_probabilities,
            model_shells,
        ),
        "target_shell_probabilities": target.shell_probabilities.tolist(),
        "model_shell_probabilities": model_shells.tolist(),
    }


def _params_and_angles(
    results: tuple[TrainingResult, ...],
    method: str,
    n_conditions: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
    if method in {"shared_angle", "no_condition"}:
        result = results[0]
        return (
            np.repeat(result.best_params[None, :], n_conditions, axis=0),
            result.condition_angles,
            [result.best_step] * n_conditions,
            [result.best_validation_loss] * n_conditions,
        )
    if method != "separate_bank" or len(results) != n_conditions:
        raise ValueError("results do not match the scaling method")
    return (
        np.stack([result.best_params for result in results]),
        np.stack([result.condition_angles[0] for result in results]),
        [result.best_step for result in results],
        [result.best_validation_loss for result in results],
    )


def evaluate_scaling_results(
    results: tuple[TrainingResult, ...],
    method: str,
    n_conditions: int,
    config: TrainingConfig,
    budget: ScalingBudget,
) -> dict[str, object]:
    """Evaluate one trained scaling configuration exactly at trained Conditions."""
    if method not in {"shared_angle", "no_condition", "separate_bank"}:
        raise ValueError(f"unknown scaling method: {method!r}")
    ansatz = complete_ansatz()
    params, angles, best_steps, best_losses = _params_and_angles(
        results,
        method,
        n_conditions,
    )
    bitstrings = enumerate_bitstrings(ansatz.n_qubits)
    model_probabilities = np.stack(
        [
            exact_iqp_probabilities(
                params[condition],
                ansatz.generators,
                angles[condition],
            )
            for condition in range(n_conditions)
        ],
    )
    target_pair = targets(n_conditions, ansatz.n_qubits)
    fractions = condition_angle_fractions(n_conditions)
    conditions = [
        _condition_metrics(
            condition,
            float(fraction),
            target,
            model_probs,
            bitstrings,
        )
        for condition, (fraction, target, model_probs) in enumerate(
            zip(fractions, target_pair, model_probabilities, strict=True),
        )
    ]
    accounting = parameter_accounting(n_conditions, ansatz)
    return {
        "protocol": "condition_count_scaling_v1",
        "model_name": f"scaling_{budget.track}_k{n_conditions}_{method}",
        "method": method,
        "track": budget.track,
        "n_conditions": n_conditions,
        "topology": ansatz.name,
        "seed": results[0].seed,
        "n_parameters": (
            accounting.separate_bank_parameters
            if method == "separate_bank"
            else accounting.shared_parameters
        ),
        "shared_parameters": accounting.shared_parameters,
        "separate_bank_parameters": accounting.separate_bank_parameters,
        "best_step": best_steps if method == "separate_bank" else best_steps[0],
        "best_validation_loss": (
            best_losses if method == "separate_bank" else best_losses[0]
        ),
        "budget": asdict(budget),
        "training_config": asdict(config),
        "conditions": conditions,
    }


def run_scaling(
    seed: int,
    n_conditions: int,
    track: str,
    method: str,
    config: TrainingConfig | None = None,
) -> tuple[tuple[TrainingResult, ...], dict[str, object]]:
    """Train and evaluate one resumable K/track/method/seed unit."""
    if config is None:
        budget = scaling_budget(n_conditions, track)
        config = TrainingConfig(
            steps=budget.optimizer_updates,
            batch_size=budget.batch_size_per_condition,
        )
    else:
        budget = scaling_budget(
            n_conditions,
            track,
            optimizer_updates=config.steps,
        )
    if config.batch_size != budget.batch_size_per_condition:
        raise ValueError(
            "config.batch_size must equal the frozen balanced per-Condition "
            f"batch size {budget.batch_size_per_condition}",
        )
    training_data, validation_data = sample_scaling_datasets(seed, budget)
    ansatz = complete_ansatz()
    if method == "shared_angle":
        model_name = f"scaling_{track}_k{n_conditions}_shared_angle"
        results = (
            train(
                model_name,
                ansatz,
                condition_angles(n_conditions),
                training_data,
                validation_data,
                seed,
                config,
            ),
        )
    elif method == "no_condition":
        model_name = f"scaling_{track}_k{n_conditions}_no_condition"
        midpoint_angles = np.full(
            (n_conditions, ansatz.n_qubits),
            pi / 8.0,
            dtype=np.float64,
        )
        results = (
            train(
                model_name,
                ansatz,
                midpoint_angles,
                training_data,
                validation_data,
                seed,
                config,
            ),
        )
    elif method == "separate_bank":
        results = tuple(
            train(
                f"scaling_{track}_k{n_conditions}_separate_condition{condition}",
                ansatz,
                condition_angles(n_conditions)[condition : condition + 1],
                training_data[condition : condition + 1],
                validation_data[condition : condition + 1],
                seed,
                config,
            )
            for condition in range(n_conditions)
        )
    else:
        raise ValueError(f"unknown scaling method: {method!r}")
    metrics = evaluate_scaling_results(
        results,
        method,
        n_conditions,
        config,
        budget,
    )
    return results, metrics


def save_scaling_result(
    results: tuple[TrainingResult, ...],
    metrics: dict[str, object],
    directory: Path,
) -> None:
    """Persist one machine-readable scaling unit."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = str(metrics["model_name"]) + f"_seed{metrics['seed']}"
    first = results[0]
    if len(results) == 1:
        best_params = first.best_params
        final_params = first.final_params
        train_losses = first.train_losses
        validation_losses = first.validation_losses
        best_steps = np.asarray([first.best_step])
        best_validation_losses = np.asarray([first.best_validation_loss])
    else:
        best_params = np.stack([result.best_params for result in results])
        final_params = np.stack([result.final_params for result in results])
        train_losses = np.stack([result.train_losses for result in results])
        validation_losses = np.stack(
            [result.validation_losses for result in results],
        )
        best_steps = np.asarray([result.best_step for result in results])
        best_validation_losses = np.asarray(
            [result.best_validation_loss for result in results],
        )
    np.savez_compressed(
        directory / f"{stem}.npz",
        generators=first.generators,
        condition_angles=np.asarray(
            [result.condition_angles[0] for result in results]
            if len(results) > 1
            else first.condition_angles,
        ),
        best_params=best_params,
        final_params=final_params,
        train_steps=first.train_steps,
        train_losses=train_losses,
        validation_steps=first.validation_steps,
        validation_losses=validation_losses,
        best_steps=best_steps,
        best_validation_losses=best_validation_losses,
        encoding=np.asarray(first.encoding),
    )
    (directory / f"{stem}.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
    )


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
    }


def aggregate_condition_runs(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate exact per-Condition rows without pooling Conditions."""
    if not runs:
        raise ValueError("runs must not be empty")
    seeds = sorted({int(run["seed"]) for run in runs})
    n_conditions = int(runs[0]["n_conditions"])
    metrics = ("profile_tvd", "bitstring_tvd")
    per_condition = []
    for condition in range(n_conditions):
        condition_rows = [
            run["conditions"][condition]
            for run in runs
        ]
        per_condition.append(
            {
                "condition": condition,
                "angle_fraction_pi": condition_rows[0]["angle_fraction_pi"],
                **{
                    metric: _stats([row[metric] for row in condition_rows])
                    for metric in metrics
                },
            },
        )

    per_seed = []
    for run in sorted(runs, key=lambda item: int(item["seed"])):
        row = {"seed": int(run["seed"]), "n_conditions": n_conditions}
        condition_rows = run["conditions"]
        for metric in metrics:
            values = np.asarray([item[metric] for item in condition_rows])
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"median_{metric}"] = float(np.median(values))
            row[f"worst_{metric}"] = float(np.max(values))
        per_seed.append(row)

    aggregate = {}
    for aggregation in ("mean", "median", "worst"):
        aggregate[aggregation] = {
            metric: _stats(
                [row[f"{aggregation}_{metric}"] for row in per_seed],
            )
            for metric in metrics
        }
    return {
        "n_conditions": n_conditions,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "per_condition": per_condition,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def strict_comparison_counts(
    shared_runs: list[dict[str, object]],
    baseline_runs: list[dict[str, object]],
    metric: str,
) -> dict[str, int]:
    """Count per-Condition failures against a matched baseline."""
    baseline_by_seed = {int(run["seed"]): run for run in baseline_runs}
    failures = 0
    total = 0
    for shared in shared_runs:
        baseline = baseline_by_seed[int(shared["seed"])]
        for shared_row, baseline_row in zip(
            shared["conditions"],
            baseline["conditions"],
            strict=True,
        ):
            failures += int(shared_row[metric] >= baseline_row[metric])
            total += 1
    return {"failures": failures, "total": total}
