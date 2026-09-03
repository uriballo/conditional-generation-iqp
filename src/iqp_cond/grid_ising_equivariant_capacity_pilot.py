"""Symmetry-equivariant follow-up to the grid-Ising capacity pilot."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .ansatz import complete_ansatz
from .artifacts import (
    read_immutable_json,
    sha256_file,
    write_immutable_bytes,
    write_immutable_json,
)
from .encoding import GRID_ISING_EQUIVARIANT_ENCODING
from .exact import exact_iqp_visible_probabilities, total_variation_distance
from .exact_tvd_campaign import data_initialize, expected_bodyness_sigmas
from .exact_tvd_runner import _weighted_loss_functions
from .exact_tvd_training import CampaignTrainingConfig, train_campaign
from .grid_ising import exact_probabilities
from .grid_ising_capacity_pilot import (
    _grid_metrics,
    empirical_probabilities,
    select_checkpoint,
)
from .protocol import content_digest

CAMPAIGN_VERSION = "grid_ising_equivariant_capacity_pilot_v1"
CONDITIONS = ((0.65, 0.0), (0.44, 0.05))
ANCILLAS = (0, 3)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def manifest_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(config)).hexdigest()


def load_manifest(path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
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
        "encoder",
        "source_data",
        "training",
        "evaluation",
    }
    if set(value) != expected_keys:
        raise ValueError("manifest keys do not match the frozen schema")
    if value["schema_version"] != 1 or value["campaign_version"] != CAMPAIGN_VERSION:
        raise ValueError("campaign identity differs")
    if value["visible_qubits"] != 8 or value["shape"] != [2, 4]:
        raise ValueError("the pilot is frozen to the 2x4 lattice")
    if value["periodic"] is not True:
        raise ValueError("the pilot requires periodic boundaries")
    if tuple(value["ancillas"]) != ANCILLAS or tuple(value["seeds"]) != (0,):
        raise ValueError("ancilla or seed grid differs")
    conditions = tuple(tuple(float(x) for x in pair) for pair in value["conditions"])
    if conditions != CONDITIONS:
        raise ValueError("condition grid differs")
    if value["encoder"] != {
        "name": GRID_ISING_EQUIVARIANT_ENCODING,
        "beta_range": [0.2, 0.65],
        "field_scale": 0.1,
        "branch_angle_map": "pi/2-(pi/3)*(0.1+0.8*v_beta)",
        "branch_weight_map": "delta=pi*w_h/6",
    }:
        raise ValueError("encoder protocol differs")
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
        "gradient_batches": 4,
    }:
        raise ValueError("training protocol differs")
    required_evaluation = {
        "primary_metric",
        "selection_metric",
        "ordered_tvd_limit",
        "field_baseline_tvd",
        "maximum_field_regression",
        "minimum_ordered_ancilla_advantage",
        "minimum_ancilla_parameter_movement",
        "minimum_ordered_learned_gain",
    }
    if set(value["evaluation"]) != required_evaluation:
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
    digest = manifest_digest(config)
    return tuple(
        PilotRun(digest, beta, field, ancillas, 0)
        for beta, field in CONDITIONS
        for ancillas in ANCILLAS
    )


def run_by_id(config: Mapping[str, Any], run_id: str) -> PilotRun:
    matches = [run for run in identities(config) if run.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"unknown run id: {run_id}")
    return matches[0]


def condition_angles(
    config: Mapping[str, Any], beta: float, field: float, total_qubits: int
) -> np.ndarray:
    beta_min, beta_max = config["encoder"]["beta_range"]
    v_beta = (beta - beta_min) / (beta_max - beta_min)
    w_field = field / float(config["encoder"]["field_scale"])
    if not 0.0 <= v_beta <= 1.0 or not -1.0 <= w_field <= 1.0:
        raise ValueError("condition lies outside the frozen encoder range")
    values = np.zeros((1, total_qubits), dtype=np.float64)
    values[0, 0] = v_beta
    values[0, 1] = w_field
    return values


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
    return directory / "data" / "grid-ising-equivariant-n8.npz"


def prepare_data_archive(
    config: Mapping[str, Any], directory: Path, repository: Path
) -> tuple[Path, str]:
    source = repository / config["source_data"]["path"]
    if sha256_file(source) != config["source_data"]["sha256"]:
        raise RuntimeError("frozen source-data digest differs")
    with np.load(source, allow_pickle=False) as stored:
        source_arrays = {name: np.asarray(stored[name]) for name in stored.files}
    source_conditions = [tuple(value) for value in source_arrays["conditions"]]
    indices = [source_conditions.index(condition) for condition in CONDITIONS]
    arrays = {
        "configuration_digest": np.asarray(manifest_digest(config), dtype="S64"),
        "source_data_sha256": np.asarray(config["source_data"]["sha256"], dtype="S64"),
        "conditions": np.asarray(CONDITIONS, dtype=np.float64),
    }
    for role in ("train", "validation", "test"):
        arrays[role] = source_arrays[role][indices]
        arrays[f"{role}_seeds"] = source_arrays[f"{role}_seeds"][indices]
    path = data_archive_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, _write_immutable_npz(path, arrays)


def load_data_archive(
    config: Mapping[str, Any], directory: Path
) -> dict[str, np.ndarray]:
    with np.load(data_archive_path(directory), allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    if arrays["configuration_digest"].item().decode() != manifest_digest(config):
        raise ValueError("data archive belongs to another manifest")
    for role in ("train", "validation", "test"):
        if arrays[role].shape != (2, 1000, 8):
            raise ValueError(f"{role} split has the wrong shape")
    return arrays


def _condition_index(run: PilotRun) -> int:
    return CONDITIONS.index((run.beta, run.field))


def _ancilla_support(generators: np.ndarray, n_visible: int) -> np.ndarray:
    return np.any(np.asarray(generators)[:, n_visible:] != 0, axis=1)


def _gradient_diagnostic(
    initial: np.ndarray,
    training: np.ndarray,
    training_loss: Any,
    visible_support: np.ndarray,
    ancilla_support: np.ndarray,
    count: int,
    seed: int,
) -> dict[str, float | None]:
    value_grad = jax.jit(jax.value_and_grad(training_loss))
    visible_norms = []
    ancilla_norms = []
    losses = []
    for index in range(count):
        batch = jnp.asarray(training[:, index * 100 : (index + 1) * 100])
        loss, gradient = value_grad(
            jnp.asarray(initial),
            batch,
            jax.random.fold_in(jax.random.PRNGKey(seed), index),
        )
        values = np.asarray(gradient)
        losses.append(float(loss))
        visible_norms.append(float(np.linalg.norm(values[visible_support])))
        if np.any(ancilla_support):
            ancilla_norms.append(float(np.linalg.norm(values[ancilla_support])))
    return {
        "mean_loss": float(np.mean(losses)),
        "mean_visible_gradient_l2": float(np.mean(visible_norms)),
        "mean_ancilla_gradient_l2": (
            float(np.mean(ancilla_norms)) if ancilla_norms else None
        ),
        "ancilla_to_visible_gradient_ratio": (
            float(np.mean(ancilla_norms) / np.mean(visible_norms))
            if ancilla_norms and np.mean(visible_norms) > 0
            else None
        ),
    }


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
    ancilla_support = _ancilla_support(generators, n_visible)
    visible_support = ~ancilla_support
    initial = data_initialize(
        training[0],
        generators,
        ancilla_support,
        int(run.run_id, 16) & (2**63 - 1),
    )
    settings = config["training"]
    sigmas = expected_bodyness_sigmas(n_visible)
    angles = condition_angles(config, run.beta, run.field, total_qubits)
    training_loss, validation_mmd, banks = _weighted_loss_functions(
        generators=generators,
        angles=angles,
        validation_targets=validation,
        sigmas=sigmas,
        n_ops=int(settings["n_ops"]),
        n_samples=int(settings["n_samples"]),
        seed=int(run.run_id, 16) & (2**63 - 1),
        encoding=GRID_ISING_EQUIVARIANT_ENCODING,
    )
    gradient = _gradient_diagnostic(
        initial,
        training,
        training_loss,
        visible_support,
        ancilla_support,
        int(settings["gradient_batches"]),
        int(run.run_id[:8], 16),
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
        training_loss,
        int(run.run_id[-8:], 16) & (2**31 - 1),
        training_config,
        include_initial_checkpoint=True,
    )
    arrays = {
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
    checkpoint_sha = _write_immutable_npz(destination, arrays)
    bank_digest = hashlib.sha256()
    for masks, weights in banks:
        bank_digest.update(masks.tobytes())
        bank_digest.update(weights.tobytes())
    payload = {
        **run.to_dict(),
        "run_id": run.run_id,
        "status": "completed",
        "gradient_diagnostic": gradient,
        "checkpoint_path": str(destination),
        "checkpoint_sha256": checkpoint_sha,
        "operator_bank_sha256": bank_digest.hexdigest(),
        "parameter_count": int(ansatz.n_parameters),
        "runtime_seconds": time.monotonic() - start,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(record_path, payload)
    return payload


def evaluate_run(
    config: Mapping[str, Any], directory: Path, run: PilotRun
) -> dict[str, Any]:
    record_path = evaluation_record_path(directory, run)
    if record_path.exists():
        return read_immutable_json(record_path)
    start = time.monotonic()
    training_record = read_immutable_json(training_record_path(directory, run))
    checkpoint_file = Path(training_record["checkpoint_path"])
    if sha256_file(checkpoint_file) != training_record["checkpoint_sha256"]:
        raise RuntimeError("checkpoint digest differs")
    with np.load(checkpoint_file, allow_pickle=False) as stored:
        checkpoint = {name: np.asarray(stored[name]) for name in stored.files}
    archive = load_data_archive(config, directory)
    index = _condition_index(run)
    selected, candidates, validation_tvds = select_checkpoint(
        checkpoint["checkpoint_params"],
        empirical_probabilities(archive["validation"][index]),
        checkpoint["generators"],
        checkpoint["angles"],
        int(config["visible_qubits"]),
        run.ancillas,
        GRID_ISING_EQUIVARIANT_ENCODING,
    )
    model = candidates[selected]
    target = exact_probabilities(tuple(config["shape"]), run.beta, run.field)
    initial = candidates[0]
    zero_phase = exact_iqp_visible_probabilities(
        np.zeros(checkpoint["generators"].shape[0]),
        checkpoint["generators"],
        checkpoint["angles"],
        int(config["visible_qubits"]),
        run.ancillas,
        GRID_ISING_EQUIVARIANT_ENCODING,
    )[0]
    metrics = _grid_metrics(
        model, target, tuple(config["shape"]), checkpoint["sigmas"], run.field
    )
    initial_tvd = total_variation_distance(target, initial)
    metrics.update(
        {
            "initial_bitstring_tvd": initial_tvd,
            "learned_tvd_gain": initial_tvd - float(metrics["bitstring_tvd"]),
            "encoder_only_zero_phase_tvd": total_variation_distance(
                target, zero_phase
            ),
        }
    )
    params = checkpoint["checkpoint_params"]
    ancilla_support = _ancilla_support(
        checkpoint["generators"], int(config["visible_qubits"])
    )
    movement = params[selected] - params[0]
    metrics["visible_parameter_movement_l2"] = float(
        np.linalg.norm(movement[~ancilla_support])
    )
    metrics["ancilla_parameter_movement_l2"] = (
        float(np.linalg.norm(movement[ancilla_support]))
        if np.any(ancilla_support)
        else None
    )
    probabilities_file = probability_path(directory, run)
    probabilities_file.parent.mkdir(parents=True, exist_ok=True)
    probability_sha = _write_immutable_npz(
        probabilities_file,
        {
            "configuration_digest": np.asarray(
                run.configuration_digest, dtype="S64"
            ),
            "run_id": np.asarray(run.run_id, dtype="S16"),
            "target": target,
            "selected_model": model,
            "initial_model": initial,
            "encoder_only_zero_phase": zero_phase,
            "candidate_models": candidates,
            "candidate_validation_tvds": validation_tvds,
        },
    )
    payload = {
        **run.to_dict(),
        "run_id": run.run_id,
        "status": "completed",
        "selected_candidate_index": selected,
        "selected_step": int(checkpoint["checkpoint_steps"][selected]),
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
    records = [
        read_immutable_json(evaluation_record_path(directory, run))
        for run in identities(config)
    ]
    lookup = {
        (row["beta"], row["field"], row["ancillas"]): row for row in records
    }
    ordered_r0 = lookup[(0.65, 0.0, 0)]["metrics"]
    ordered_r3 = lookup[(0.65, 0.0, 3)]["metrics"]
    field_r3 = lookup[(0.44, 0.05, 3)]["metrics"]
    gate = config["evaluation"]
    checks = {
        "ordered_r3_tvd_at_most_0.20": ordered_r3["bitstring_tvd"]
        <= gate["ordered_tvd_limit"],
        "field_r3_regression_at_most_0.02": field_r3["bitstring_tvd"]
        <= gate["field_baseline_tvd"] + gate["maximum_field_regression"],
        "ordered_r3_advantage_at_least_0.02": ordered_r0["bitstring_tvd"]
        - ordered_r3["bitstring_tvd"]
        >= gate["minimum_ordered_ancilla_advantage"],
        "ordered_ancilla_movement_at_least_0.05": ordered_r3[
            "ancilla_parameter_movement_l2"
        ]
        >= gate["minimum_ancilla_parameter_movement"],
        "ordered_learned_gain_at_least_0.02": ordered_r3["learned_tvd_gain"]
        >= gate["minimum_ordered_learned_gain"],
    }
    payload = {
        "campaign": CAMPAIGN_VERSION,
        "configuration_digest": manifest_digest(config),
        "production_evidence": False,
        "runs": records,
        "ordered_ancilla_advantage": ordered_r0["bitstring_tvd"]
        - ordered_r3["bitstring_tvd"],
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
