"""Immutable planning and deterministic data helpers for the exact-TVD campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .initialization import initialize_parameters_from_data
from .protocol import ProtocolValidationError, content_digest

DATASETS = ("gaussian", "curie_weiss", "mfc")
ANCILLAS = (0, 1, 3, 6, 10)
SEEDS = tuple(range(100, 110))


def expected_bodyness_sigmas(n: int) -> tuple[float, float, float]:
    """Return bandwidths targeting weights 1, midpoint, and 0.49*n."""
    if n < 2:
        raise ValueError("n must be at least two")
    weights = (1.0, (1.0 + 0.49 * n) / 2.0, 0.49 * n)
    return tuple(float((-2.0 * np.log(1.0 - 2.0 * b / n)) ** -0.5) for b in weights)


@dataclass(frozen=True, slots=True)
class RunIdentity:
    configuration_digest: str
    dataset: str
    visible_qubits: int
    ancillas: int
    seed: int
    family: str
    condition_role: str
    condition: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_digest": self.configuration_digest,
            "dataset": self.dataset,
            "visible_qubits": self.visible_qubits,
            "ancillas": self.ancillas,
            "seed": self.seed,
            "family": self.family,
            "condition_role": self.condition_role,
            "condition": self.condition,
        }

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())


def production_identities(config: dict[str, Any]) -> tuple[RunIdentity, ...]:
    digest = manifest_digest(config)
    runs: list[RunIdentity] = []
    for dataset in DATASETS:
        for n in config["datasets"][dataset]["visible_qubits"]:
            spec = config["datasets"][dataset]
            for ancilla in config["ancillas"]:
                for seed in config["seeds"]:
                    runs.append(
                        RunIdentity(
                            digest,
                            dataset,
                            n,
                            ancilla,
                            seed,
                            "conditioned",
                            "shared",
                            None,
                        )
                    )
                    roles = [
                        ("train", i, value)
                        for i, value in enumerate(spec["training_conditions"])
                    ] + [
                        ("held_out", i, value)
                        for i, value in enumerate(spec["held_out_conditions"])
                    ]
                    for role, index, condition in roles:
                        runs.append(
                            RunIdentity(
                                digest,
                                dataset,
                                n,
                                ancilla,
                                seed,
                                "independent",
                                f"{role}_{index}",
                                float(condition),
                            )
                        )
    return tuple(runs)


def data_initialize(
    data: np.ndarray, generators: np.ndarray, ancilla_support: np.ndarray, seed: int
) -> np.ndarray:
    """IQPopt visible initialization plus seeded noise solely on ancilla supports."""
    rows = np.asarray(generators)
    if rows.ndim != 2 or rows.shape[1] < data.shape[1]:
        raise ValueError("generators must cover all visible data qubits")
    params = initialize_parameters_from_data(data, rows[:, : data.shape[1]])
    supports = np.asarray(ancilla_support, dtype=bool)
    if supports.shape != params.shape:
        raise ValueError("ancilla_support must match parameter vector")
    rng = np.random.default_rng(seed)
    params = params.copy()
    params[supports] = rng.normal(0.0, 0.01 / np.sqrt(params.size), supports.sum())
    return params


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if set(value) != {
        "schema_version",
        "campaign_version",
        "datasets",
        "ancillas",
        "seeds",
        "data",
        "training",
        "estimator_calibration",
        "optimizer_pilot",
        "evaluation",
        "execution",
    }:
        raise ProtocolValidationError("manifest keys do not match schema")
    if (
        value["schema_version"] != 1
        or value["campaign_version"] != "exact_tvd_ancilla_v1"
    ):
        raise ProtocolValidationError("campaign identity differs")
    if tuple(value["ancillas"]) != ANCILLAS or tuple(value["seeds"]) != SEEDS:
        raise ProtocolValidationError("ancilla or seed vocabulary differs")
    if set(value["datasets"]) != set(DATASETS):
        raise ProtocolValidationError("dataset vocabulary differs")
    if any(
        value["data"][key] != 1000
        for key in ("train_samples", "validation_samples", "test_samples")
    ):
        raise ProtocolValidationError("all data splits must contain 1000 samples")
    expected_datasets = {
        "gaussian": {
            "visible_qubits": [8, 12, 16],
            "width": 0.5,
            "training_conditions": [0.0, 0.25, 0.5, 0.75, 1.0],
            "held_out_conditions": [
                0.0625,
                0.1875,
                0.3125,
                0.4375,
                0.5625,
                0.6875,
                0.8125,
                0.9375,
            ],
        },
        "curie_weiss": {
            "visible_qubits": [8, 12, 16],
            "training_conditions": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
            "held_out_conditions": [0.6, 0.8, 1.0, 1.2, 1.4],
        },
        "mfc": {
            "visible_qubits": [3],
            "training_conditions": [50, 75, 100, 150, 175, 200],
            "held_out_conditions": [125],
        },
    }
    if value["datasets"] != expected_datasets:
        raise ProtocolValidationError("dataset protocol differs")
    training = value["training"]
    if (
        training["updates"] != 2000
        or training["samples_per_condition_per_update"] != 100
        or training["n_ops_per_condition_per_bandwidth"] != 2000
        or training["candidate_checkpoints"]
        != [500, 1000, 1500, 2000, "lowest_validation_mmd"]
    ):
        raise ProtocolValidationError("training protocol differs")
    calibration = value["estimator_calibration"]
    if calibration["candidates"] != [400, 800, 1600, 3200, 8192] or calibration[
        "representative_visible_qubits"
    ] != {"gaussian": 8, "curie_weiss": 8, "mfc": 3}:
        raise ProtocolValidationError("estimator calibration candidates differ")
    pilot = value["optimizer_pilot"]
    if (
        pilot["development_seeds"] != [0, 1, 2]
        or pilot["matched_cells"] != 18
        or pilot["representative_visible_qubits"]
        != {"gaussian": 8, "curie_weiss": 8, "mfc": 3}
    ):
        raise ProtocolValidationError("optimizer pilot protocol differs")
    return value


def manifest_digest(value: dict[str, Any]) -> str:
    return content_digest(value)
