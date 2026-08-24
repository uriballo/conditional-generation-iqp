"""Exact evaluation for the frozen symmetry-aligned fidelity protocol."""

from __future__ import annotations

import hashlib
import json
from math import pi, sqrt
from pathlib import Path

import numpy as np

from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from .interpolation import (
    HELDOUT_ANGLE_FRACTIONS,
    TRAIN_ANGLE_FRACTIONS,
    broadcast_angles,
    interpolation_targets,
)
from .symmetry import OUTPUT_PROTOCOLS, apply_output_protocol, output_protocol_spec
from .targets import GaussianHammingTarget

FROZEN_THRESHOLDS = {
    "absolute_bitstring_tvd_reduction": 0.15,
    "relative_bitstring_tvd_reduction": 0.40,
    "maximum_shell_tvd_degradation": 0.01,
    "maximum_mean_error_degradation": 0.05,
    "maximum_std_error_degradation": 0.05,
}


def distribution_metrics(
    target: GaussianHammingTarget,
    model_probabilities: np.ndarray,
    bitstrings: np.ndarray,
) -> dict[str, object]:
    """Return exact primary and guardrail metrics for one Condition."""
    model_probabilities = np.asarray(model_probabilities, dtype=np.float64)
    model_shells = shell_probabilities(model_probabilities, bitstrings)
    weights = np.arange(model_shells.size, dtype=np.float64)
    model_mean = float(weights @ model_shells)
    model_variance = float(((weights - model_mean) ** 2) @ model_shells)
    target_probabilities = target.bitstring_probabilities(bitstrings)
    return {
        "bitstring_tvd": total_variation_distance(
            target_probabilities,
            model_probabilities,
        ),
        "shell_tvd": total_variation_distance(
            target.shell_probabilities,
            model_shells,
        ),
        "target_mean": target.actual_mean,
        "model_mean": model_mean,
        "mean_absolute_error": abs(model_mean - target.actual_mean),
        "target_std": target.actual_std,
        "model_std": sqrt(max(model_variance, 0.0)),
        "std_absolute_error": abs(
            sqrt(max(model_variance, 0.0)) - target.actual_std,
        ),
        "model_shell_probabilities": model_shells.tolist(),
    }


def evaluate_endpoint_seed(
    seed: int,
    checkpoint_directory: Path,
) -> list[dict[str, object]]:
    """Evaluate all output protocols at the two frozen endpoint Conditions."""
    bitstrings = enumerate_bitstrings(19)
    target_pair = (
        GaussianHammingTarget.create(19, 19 / 2, sqrt(19) / 2),
        GaussianHammingTarget.create(19, 19 / 4, sqrt(19) / 2),
    )
    paths = {
        "angle": checkpoint_directory / f"angle_complete_seed{seed}.npz",
        "no_condition": checkpoint_directory
        / f"no_condition_complete_seed{seed}.npz",
        "basis_condition": checkpoint_directory
        / f"basis_condition_complete_seed{seed}.npz",
    }
    runs = {name: np.load(path) for name, path in paths.items()}
    probabilities = {
        name: exact_iqp_probabilities_many(
            run["best_params"],
            run["generators"],
            run["condition_angles"],
            _encoding(run),
        )
        for name, run in runs.items()
    }
    controls = []
    for index, target in enumerate(target_pair):
        controls.append(
            {
                name: distribution_metrics(
                    target,
                    probabilities[name][index],
                    bitstrings,
                )
                for name in ("no_condition", "basis_condition")
            },
        )

    records = []
    for protocol in OUTPUT_PROTOCOLS:
        conditions = []
        for index, target in enumerate(target_pair):
            transformed = apply_output_protocol(
                probabilities["angle"][index],
                19,
                protocol,
            )
            conditions.append(
                {
                    "condition": "A" if index == 0 else "B",
                    "angle_fraction_pi": 0.0 if index == 0 else 0.25,
                    "angle": distribution_metrics(target, transformed, bitstrings),
                    "frozen_unaligned_controls": controls[index],
                },
            )
        records.append(
            _base_record("endpoint", seed, protocol, paths, runs["angle"])
            | {"conditions": conditions},
        )
    return records


def evaluate_heldout_seed(
    seed: int,
    interpolation_directory: Path,
    endpoint_directory: Path,
) -> list[dict[str, object]]:
    """Evaluate dense, endpoint-only, and adjacent lookup at Held-out angles."""
    bitstrings = enumerate_bitstrings(19)
    heldout_angles = pi * HELDOUT_ANGLE_FRACTIONS
    train_angles = pi * TRAIN_ANGLE_FRACTIONS
    targets = interpolation_targets(heldout_angles)
    paths = {
        "dense": interpolation_directory
        / f"interpolation_complete_seed{seed}.npz",
        "endpoint": endpoint_directory / f"angle_complete_seed{seed}.npz",
    }
    runs = {name: np.load(path) for name, path in paths.items()}
    dense_probabilities = exact_iqp_probabilities_many(
        runs["dense"]["best_params"],
        runs["dense"]["generators"],
        broadcast_angles(np.concatenate([heldout_angles, train_angles])),
        _encoding(runs["dense"]),
    )
    endpoint_probabilities = exact_iqp_probabilities_many(
        runs["endpoint"]["best_params"],
        runs["endpoint"]["generators"],
        broadcast_angles(heldout_angles),
        _encoding(runs["endpoint"]),
    )
    dense_heldout = dense_probabilities[: len(heldout_angles)]
    dense_training = dense_probabilities[len(heldout_angles) :]

    records = []
    for protocol in OUTPUT_PROTOCOLS:
        points = []
        for index, (fraction, target) in enumerate(
            zip(HELDOUT_ANGLE_FRACTIONS, targets, strict=True),
        ):
            dense_metrics = _transformed_metrics(
                target,
                dense_heldout[index],
                bitstrings,
                protocol,
            )
            endpoint_metrics = _transformed_metrics(
                target,
                endpoint_probabilities[index],
                bitstrings,
                protocol,
            )
            adjacent_metrics = [
                _transformed_metrics(
                    target,
                    dense_training[adjacent_index],
                    bitstrings,
                    protocol,
                )
                for adjacent_index in (index, index + 1)
            ]
            oracle = {}
            for metric in ("shell_tvd", "bitstring_tvd"):
                choice = int(np.argmin([item[metric] for item in adjacent_metrics]))
                oracle[metric] = adjacent_metrics[choice][metric]
                oracle[f"{metric}_angle_fraction_pi"] = float(
                    TRAIN_ANGLE_FRACTIONS[index + choice],
                )
            points.append(
                {
                    "angle_fraction_pi": float(fraction),
                    "dense": dense_metrics,
                    "endpoint": endpoint_metrics,
                    "oracle_adjacent": oracle,
                },
            )
        records.append(
            _base_record("heldout", seed, protocol, paths, runs["dense"])
            | {"points": points},
        )
    return records


def save_study_record(
    record: dict[str, object],
    directory: Path,
    *,
    skip_existing: bool = False,
) -> Path:
    """Persist one independently resumable study/seed/protocol record."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (
        f"{record['study']}_{record['protocol']}_seed{record['seed']}.json"
    )
    if skip_existing and destination.exists():
        return destination
    destination.write_text(json.dumps(record, indent=2) + "\n")
    return destination


def _transformed_metrics(
    target: GaussianHammingTarget,
    probabilities: np.ndarray,
    bitstrings: np.ndarray,
    protocol: str,
) -> dict[str, object]:
    transformed = apply_output_protocol(probabilities, 19, protocol)
    return distribution_metrics(target, transformed, bitstrings)


def _base_record(
    study: str,
    seed: int,
    protocol: str,
    paths: dict[str, Path],
    primary_run: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    spec = output_protocol_spec(protocol)
    parameter_count = int(primary_run["generators"].shape[0])
    return {
        "schema_version": 1,
        "study": study,
        "seed": seed,
        "protocol": protocol,
        "n_parameters": spec.parameter_count(parameter_count),
        "parameter_delta": 0,
        "random_actions_per_sample": spec.random_actions_per_sample,
        "training_updates": 2_000,
        "checkpoint_rule": "lowest frozen Validation Conditional MMD",
        "thresholds": FROZEN_THRESHOLDS,
        "source_checkpoints": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def _encoding(run: np.lib.npyio.NpzFile) -> str:
    return str(np.asarray(run["encoding"]).item()) if "encoding" in run else "angle"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
