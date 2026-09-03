"""Frozen capacity pilot for exact two-dimensional grid-Ising targets."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ansatz import complete_ansatz
from .artifacts import (
    read_immutable_json,
    sha256_file,
    write_immutable_bytes,
    write_immutable_json,
)
from .encoding import ANGLE_ENCODING
from .exact import (
    enumerate_bitstrings,
    exact_iqp_visible_probabilities,
    shell_probabilities,
    total_variation_distance,
)
from .exact_tvd_campaign import data_initialize, expected_bodyness_sigmas
from .exact_tvd_runner import _diagnostic_mmd, _weighted_loss_functions
from .exact_tvd_training import CampaignTrainingConfig, train_campaign
from .grid_ising import (
    enumerate_spins,
    exact_probabilities,
    lattice_edges,
    sample_exact,
)
from .protocol import content_digest

CAMPAIGN_VERSION = "grid_ising_capacity_pilot_v1"
CONDITIONS = ((0.20, 0.0), (0.44, 0.0), (0.65, 0.0), (0.44, 0.05))
ANCILLAS = (0, 3)
ROLES = ("train", "validation", "test")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _derived_seed(*coordinates: Any) -> int:
    digest = hashlib.sha256(_canonical(coordinates)).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def manifest_digest(config: Mapping[str, Any]) -> str:
    """Return the frozen manifest digest."""
    return hashlib.sha256(_canonical(config)).hexdigest()


def load_manifest(path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load and strictly validate the frozen pilot manifest."""
    value = (
        dict(path)
        if isinstance(path, Mapping)
        else json.loads(Path(path).read_text())
    )
    expected_keys = {
        "schema_version",
        "campaign_version",
        "visible_qubits",
        "shape",
        "periodic",
        "ancillas",
        "seeds",
        "conditions",
        "data",
        "training",
        "evaluation",
    }
    if set(value) != expected_keys:
        raise ValueError("manifest keys do not match the frozen pilot schema")
    if value["schema_version"] != 1 or value["campaign_version"] != CAMPAIGN_VERSION:
        raise ValueError("campaign identity differs")
    if value["visible_qubits"] != 8 or value["shape"] != [2, 4]:
        raise ValueError("the pilot is frozen to the periodic 2x4 lattice")
    if value["periodic"] is not True:
        raise ValueError("the pilot requires periodic boundary conditions")
    if tuple(value["ancillas"]) != ANCILLAS or tuple(value["seeds"]) != (0,):
        raise ValueError("ancilla or seed grid differs")
    conditions = tuple(tuple(float(x) for x in pair) for pair in value["conditions"])
    if conditions != CONDITIONS:
        raise ValueError("condition grid differs")
    if value["data"] != {
        "root_seed": 271828,
        "train_samples": 1000,
        "validation_samples": 1000,
        "test_samples": 1000,
    }:
        raise ValueError("data protocol differs")
    if value["training"] != {
        "updates": 500,
        "batch_size": 100,
        "n_samples": 8192,
        "n_ops": 2000,
        "optimizer": "adam_cosine",
        "learning_rate": 0.01,
        "final_learning_rate": 0.001,
        "validation_interval": 25,
        "initialization": "data_visible_ancilla_noise",
        "bandwidth_policy": "expected_bodyness_1_mid_0.49n",
    }:
        raise ValueError("training protocol differs")
    if value["evaluation"] != {
        "primary_metric": "exact_analytic_bitstring_tvd",
        "selection_metric": "exact_empirical_validation_tvd",
        "gate_median_tvd": 0.15,
        "gate_max_tvd": 0.25,
        "maximum_ancilla_regression": 0.02,
    }:
        raise ValueError("evaluation protocol differs")
    return value


@dataclass(frozen=True, slots=True)
class PilotRun:
    configuration_digest: str
    beta: float
    field: float
    ancillas: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_digest": self.configuration_digest,
            "beta": self.beta,
            "field": self.field,
            "ancillas": self.ancillas,
            "seed": self.seed,
        }

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())[:16]


def identities(config: Mapping[str, Any]) -> tuple[PilotRun, ...]:
    """Return the eight unique, content-addressed pilot identities."""
    digest = manifest_digest(config)
    return tuple(
        PilotRun(digest, beta, field, ancillas, seed)
        for beta, field in CONDITIONS
        for ancillas in ANCILLAS
        for seed in (0,)
    )


def run_by_id(config: Mapping[str, Any], run_id: str) -> PilotRun:
    matches = [run for run in identities(config) if run.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"unknown pilot run id: {run_id}")
    return matches[0]


def plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{**run.to_dict(), "run_id": run.run_id} for run in identities(config)]


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _write_immutable_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            equal = set(stored.files) == set(arrays) and all(
                np.array_equal(stored[name], np.asarray(value))
                for name, value in arrays.items()
            )
        if not equal:
            raise RuntimeError(f"refusing to replace incompatible artifact: {path}")
        return sha256_file(path)
    return write_immutable_bytes(path, _npz_bytes(arrays))


def data_archive_path(directory: Path) -> Path:
    return directory / "data" / "grid-ising-n8.npz"


def prepare_data_archive(
    config: Mapping[str, Any], directory: Path
) -> tuple[Path, str]:
    """Materialize deterministic, independently seeded splits for every target."""
    shape = tuple(config["shape"])
    digest = manifest_digest(config)
    root = int(config["data"]["root_seed"])
    arrays: dict[str, np.ndarray] = {
        "configuration_digest": np.asarray(digest, dtype="S64"),
        "conditions": np.asarray(CONDITIONS, dtype=np.float64),
    }
    for role in ROLES:
        count = int(config["data"][f"{role}_samples"])
        samples = []
        seeds = []
        for beta, field in CONDITIONS:
            seed = _derived_seed(digest, root, beta, field, role)
            spins = sample_exact(
                shape, beta, field, count, seed=seed, periodic=True
            )
            samples.append(((spins + 1) // 2).astype(np.int8))
            seeds.append(seed)
        arrays[role] = np.asarray(samples, dtype=np.int8)
        arrays[f"{role}_seeds"] = np.asarray(seeds, dtype=np.uint64)
    path = data_archive_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, _write_immutable_npz(path, arrays)


def load_data_archive(
    config: Mapping[str, Any], directory: Path
) -> dict[str, np.ndarray]:
    path = data_archive_path(directory)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if arrays["configuration_digest"].item().decode() != manifest_digest(config):
        raise ValueError("data archive belongs to a different manifest")
    if not np.array_equal(arrays["conditions"], np.asarray(CONDITIONS)):
        raise ValueError("data conditions differ")
    for role in ROLES:
        if arrays[role].shape != (4, 1000, 8):
            raise ValueError(f"{role} split has the wrong shape")
    return arrays


def _condition_index(run: PilotRun) -> int:
    return CONDITIONS.index((run.beta, run.field))


def _ancilla_support(generators: np.ndarray, n_visible: int) -> np.ndarray:
    return np.any(np.asarray(generators)[:, n_visible:] != 0, axis=1)


def checkpoint_path(directory: Path, run: PilotRun) -> Path:
    return directory / "checkpoints" / f"{run.run_id}.npz"


def training_record_path(directory: Path, run: PilotRun) -> Path:
    return directory / "training" / f"{run.run_id}.json"


def evaluation_record_path(directory: Path, run: PilotRun) -> Path:
    return directory / "evaluation" / f"{run.run_id}.json"


def probability_path(directory: Path, run: PilotRun) -> Path:
    return directory / "probabilities" / f"{run.run_id}.npz"


def train_run(
    config: Mapping[str, Any], directory: Path, run: PilotRun
) -> dict[str, Any]:
    """Fit one unconditional complete two-local IQP circuit."""
    record_path = training_record_path(directory, run)
    if record_path.exists():
        return read_immutable_json(record_path)
    start = time.monotonic()
    archive = load_data_archive(config, directory)
    index = _condition_index(run)
    training = archive["train"][index : index + 1]
    validation = archive["validation"][index : index + 1]
    n_visible = int(config["visible_qubits"])
    total_qubits = n_visible + run.ancillas
    ansatz = complete_ansatz(total_qubits)
    generators = np.asarray(ansatz.generators)
    initial = data_initialize(
        training[0],
        generators,
        _ancilla_support(generators, n_visible),
        _derived_seed(run.to_dict(), "initialization"),
    )
    settings = config["training"]
    sigmas = expected_bodyness_sigmas(n_visible)
    angles = np.zeros((1, total_qubits), dtype=np.float64)
    loss, validation_mmd, banks = _weighted_loss_functions(
        generators=generators,
        angles=angles,
        validation_targets=validation,
        sigmas=sigmas,
        n_ops=int(settings["n_ops"]),
        n_samples=int(settings["n_samples"]),
        seed=_derived_seed(run.to_dict(), "operator_bank"),
        encoding=ANGLE_ENCODING,
    )
    training_config = CampaignTrainingConfig(
        steps=int(settings["updates"]),
        batch_size=int(settings["batch_size"]),
        n_samples=int(settings["n_samples"]),
        validation_interval=int(settings["validation_interval"]),
        learning_rate=float(settings["learning_rate"]),
        final_learning_rate=float(settings["final_learning_rate"]),
        optimizer="adam_cosine",
    )
    result = train_campaign(
        initial,
        training,
        validation_mmd,
        loss,
        _derived_seed(run.to_dict(), "training") & (2**31 - 1),
        training_config,
        include_initial_checkpoint=True,
    )
    checkpoint_arrays = {
        "configuration_digest": np.asarray(run.configuration_digest, dtype="S64"),
        "run_id": np.asarray(run.run_id, dtype="S16"),
        "generators": generators,
        "angles": angles,
        "sigmas": np.asarray(sigmas),
        "checkpoint_steps": np.asarray([item.step for item in result.checkpoints]),
        "checkpoint_params": np.asarray([item.params for item in result.checkpoints]),
        "checkpoint_validation_mmd": np.asarray(
            [item.validation_mmd for item in result.checkpoints]
        ),
        "checkpoint_reasons": np.asarray(
            [item.reason for item in result.checkpoints], dtype="S32"
        ),
        "train_steps": result.train_steps,
        "train_losses": result.train_losses,
        "validation_steps": result.validation_steps,
        "validation_losses": result.validation_losses,
    }
    destination = checkpoint_path(directory, run)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = _write_immutable_npz(destination, checkpoint_arrays)
    bank_digest = hashlib.sha256()
    for masks, weights in banks:
        bank_digest.update(masks.tobytes())
        bank_digest.update(weights.tobytes())
    payload = {
        **run.to_dict(),
        "run_id": run.run_id,
        "status": "completed",
        "checkpoint_path": str(destination),
        "checkpoint_sha256": checkpoint_sha,
        "operator_bank_sha256": bank_digest.hexdigest(),
        "parameter_count": int(ansatz.n_parameters),
        "best_validation_mmd": result.best_validation_mmd,
        "best_validation_mmd_step": result.best_step,
        "runtime_seconds": time.monotonic() - start,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(record_path, payload)
    return payload


def empirical_probabilities(samples: np.ndarray) -> np.ndarray:
    bits = np.asarray(samples, dtype=np.int64)
    labels = bits @ (1 << np.arange(bits.shape[1], dtype=np.int64))
    return np.bincount(labels, minlength=2 ** bits.shape[1]).astype(float) / len(bits)


def select_checkpoint(
    checkpoint_params: np.ndarray,
    validation_target: np.ndarray,
    generators: np.ndarray,
    angles: np.ndarray,
    n_visible: int,
    ancillas: int,
    encoding: str = ANGLE_ENCODING,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Select a candidate solely by exact empirical validation TVD."""
    probabilities = np.asarray(
        [
            exact_iqp_visible_probabilities(
                params,
                generators,
                angles,
                n_visible,
                ancillas,
                encoding,
            )[0]
            for params in checkpoint_params
        ]
    )
    tvds = np.asarray(
        [total_variation_distance(validation_target, model) for model in probabilities]
    )
    selected = int(np.argmin(tvds))
    return selected, probabilities, tvds


def _grid_metrics(
    model: np.ndarray,
    target: np.ndarray,
    shape: tuple[int, int],
    sigmas: Sequence[float],
    field: float,
) -> dict[str, float | None]:
    n_visible = shape[0] * shape[1]
    bits = enumerate_bitstrings(n_visible)
    spins = enumerate_spins(shape)
    magnetization = spins.sum(axis=1) / n_visible
    edges = lattice_edges(shape, periodic=True)
    edge_products = np.asarray(
        [spins[:, left] * spins[:, right] for left, right in edges]
    )
    target_edges = edge_products @ target
    model_edges = edge_products @ model
    return {
        "bitstring_tvd": total_variation_distance(target, model),
        "magnetization_profile_tvd": total_variation_distance(
            shell_probabilities(target, bits), shell_probabilities(model, bits)
        ),
        "mean_magnetization_error": float(
            model @ magnetization - target @ magnetization
        ),
        "absolute_magnetization_error": float(
            model @ np.abs(magnetization) - target @ np.abs(magnetization)
        ),
        "mean_edge_correlation_error": float(np.mean(model_edges - target_edges)),
        "edge_correlation_rmse": float(
            np.sqrt(np.mean((model_edges - target_edges) ** 2))
        ),
        "test_mmd": _diagnostic_mmd(model, target, sigmas),
        "spin_flip_invariance_defect": (
            total_variation_distance(model, model[::-1]) if field == 0.0 else None
        ),
    }


def evaluate_run(
    config: Mapping[str, Any], directory: Path, run: PilotRun
) -> dict[str, Any]:
    """Select and evaluate one completed fit without retraining it."""
    record_path = evaluation_record_path(directory, run)
    if record_path.exists():
        return read_immutable_json(record_path)
    start = time.monotonic()
    training_record = read_immutable_json(training_record_path(directory, run))
    checkpoint_file = Path(training_record["checkpoint_path"])
    if sha256_file(checkpoint_file) != training_record["checkpoint_sha256"]:
        raise RuntimeError("checkpoint digest differs from the training record")
    with np.load(checkpoint_file, allow_pickle=False) as stored:
        checkpoint = {name: np.asarray(stored[name]) for name in stored.files}
    archive = load_data_archive(config, directory)
    index = _condition_index(run)
    validation_target = empirical_probabilities(archive["validation"][index])
    selected, candidate_probabilities, validation_tvds = select_checkpoint(
        checkpoint["checkpoint_params"],
        validation_target,
        checkpoint["generators"],
        checkpoint["angles"],
        int(config["visible_qubits"]),
        run.ancillas,
    )
    model = candidate_probabilities[selected]
    target = exact_probabilities(
        tuple(config["shape"]), run.beta, run.field, periodic=True
    )
    initial = candidate_probabilities[0]
    metrics = _grid_metrics(
        model, target, tuple(config["shape"]), checkpoint["sigmas"], run.field
    )
    initial_tvd = total_variation_distance(target, initial)
    metrics["initial_bitstring_tvd"] = initial_tvd
    metrics["learned_tvd_gain"] = initial_tvd - float(metrics["bitstring_tvd"])
    metrics["empirical_test_tvd"] = total_variation_distance(
        empirical_probabilities(archive["test"][index]), model
    )
    probability_arrays = {
        "configuration_digest": np.asarray(run.configuration_digest, dtype="S64"),
        "run_id": np.asarray(run.run_id, dtype="S16"),
        "target": target,
        "selected_model": model,
        "initial_model": initial,
        "candidate_models": candidate_probabilities,
        "candidate_validation_tvds": validation_tvds,
    }
    probabilities_file = probability_path(directory, run)
    probabilities_file.parent.mkdir(parents=True, exist_ok=True)
    probability_sha = _write_immutable_npz(probabilities_file, probability_arrays)
    reasons = [value.decode() for value in checkpoint["checkpoint_reasons"]]
    payload = {
        **run.to_dict(),
        "run_id": run.run_id,
        "status": "completed",
        "selected_candidate_index": selected,
        "selected_step": int(checkpoint["checkpoint_steps"][selected]),
        "selected_reason": reasons[selected],
        "selection_validation_tvd": float(validation_tvds[selected]),
        "metrics": metrics,
        "probability_path": str(probabilities_file),
        "probability_sha256": probability_sha,
        "runtime_seconds": time.monotonic() - start,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(record_path, payload)
    return payload


def status(config: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    runs = identities(config)
    trained = sum(training_record_path(directory, run).exists() for run in runs)
    evaluated = sum(evaluation_record_path(directory, run).exists() for run in runs)
    return {
        "campaign": CAMPAIGN_VERSION,
        "planned": len(runs),
        "trained": trained,
        "evaluated": evaluated,
        "pending_training": len(runs) - trained,
        "pending_evaluation": len(runs) - evaluated,
        "data_ready": data_archive_path(directory).exists(),
    }


def summarize(config: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Apply the preregistered capacity gate after all eight evaluations."""
    records = []
    for run in identities(config):
        path = evaluation_record_path(directory, run)
        if not path.exists():
            raise RuntimeError("all eight evaluations must finish before summarization")
        records.append(read_immutable_json(path))
    by_ancilla: dict[str, Any] = {}
    for ancillas in ANCILLAS:
        selected = [row for row in records if row["ancillas"] == ancillas]
        values = np.asarray([row["metrics"]["bitstring_tvd"] for row in selected])
        by_ancilla[str(ancillas)] = {
            "median_bitstring_tvd": float(np.median(values)),
            "maximum_bitstring_tvd": float(np.max(values)),
            "per_condition": [
                {
                    "beta": row["beta"],
                    "field": row["field"],
                    **row["metrics"],
                }
                for row in selected
            ],
        }
    r0 = by_ancilla["0"]["median_bitstring_tvd"]
    r3 = by_ancilla["3"]["median_bitstring_tvd"]
    gate = config["evaluation"]
    checks = {
        "r3_median_at_most_0.15": r3 <= float(gate["gate_median_tvd"]),
        "r3_maximum_at_most_0.25": by_ancilla["3"]["maximum_bitstring_tvd"]
        <= float(gate["gate_max_tvd"]),
        "r3_not_materially_worse_than_r0": r3 - r0
        <= float(gate["maximum_ancilla_regression"]),
    }
    payload = {
        "campaign": CAMPAIGN_VERSION,
        "configuration_digest": manifest_digest(config),
        "production_evidence": False,
        "by_ancilla": by_ancilla,
        "ancilla_median_improvement": r0 - r3,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }
    destination = directory / "summary.json"
    if destination.exists():
        existing = read_immutable_json(destination)
        if existing != payload:
            raise RuntimeError("existing summary differs")
    else:
        write_immutable_json(destination, payload)
    return payload
