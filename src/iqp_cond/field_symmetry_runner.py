"""Executable signed-field symmetry confirmatory campaign."""

from __future__ import annotations

import hashlib
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .ansatz import complete_ansatz
from .artifacts import read_immutable_json, sha256_file, write_immutable_json
from .confirmatory_curie import (
    CAMPAIGN_VERSION,
    FIELD_HELD_OUT,
    FIELD_TRAINING,
    SEEDS,
    SYSTEM_SIZES,
    CurieConfirmatoryManifest,
    equivariance_defect,
    signed_field_probabilities,
)
from .encoding import (
    ANGLE_ENCODING,
    FIELD_EQUIVARIANT_ENCODING,
    SPIN_FLIP_BRANCH_ENCODING,
)
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from .gaussian_data import write_immutable_npz
from .ising_training import runtime_environment
from .protocol import ProtocolValidationError, content_digest
from .training import TrainingConfig, train

FIELD_EVALUATION = (-0.30, -0.225, -0.15, -0.075, 0.0, 0.075, 0.15, 0.225, 0.30)
FIELD_TRAINING_TWO_SIDED = (-0.30, -0.15, 0.0, 0.15, 0.30)


class FieldSymmetryArm(StrEnum):
    """Frozen model arms for the conditional-symmetry study."""

    EQUIVARIANT_POSITIVE = "equivariant_positive"
    PRODUCT_POSITIVE = "product_positive"
    PRODUCT_TWO_SIDED = "product_two_sided"
    INVARIANT_EACH_FIELD = "invariant_each_field"

    @property
    def encoding(self) -> str:
        if self is FieldSymmetryArm.EQUIVARIANT_POSITIVE:
            return FIELD_EQUIVARIANT_ENCODING
        if self is FieldSymmetryArm.INVARIANT_EACH_FIELD:
            return SPIN_FLIP_BRANCH_ENCODING
        return ANGLE_ENCODING

    @property
    def training_fields(self) -> tuple[float, ...]:
        if self is FieldSymmetryArm.PRODUCT_TWO_SIDED:
            return FIELD_TRAINING_TWO_SIDED
        return FIELD_TRAINING


@dataclass(frozen=True, slots=True)
class FieldSymmetryRunIdentity:
    """One immutable signed-field fit."""

    campaign_version: str
    n_qubits: int
    arm: FieldSymmetryArm
    seed: int
    configuration_digest: str

    def __post_init__(self) -> None:
        if self.campaign_version != CAMPAIGN_VERSION:
            raise ProtocolValidationError("field run campaign version differs")
        if self.n_qubits not in SYSTEM_SIZES or self.seed not in SEEDS:
            raise ProtocolValidationError("field run coordinates are undeclared")
        if len(self.configuration_digest) != 64:
            raise ProtocolValidationError("configuration_digest must be SHA-256")

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_version": self.campaign_version,
            "study": "signed_field_symmetry",
            "n_qubits": self.n_qubits,
            "arm": self.arm.value,
            "seed": self.seed,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FieldSymmetryRunIdentity:
        expected = {
            "campaign_version",
            "study",
            "n_qubits",
            "arm",
            "seed",
            "configuration_digest",
        }
        if set(value) != expected or value["study"] != "signed_field_symmetry":
            raise ProtocolValidationError("field run identity schema differs")
        return cls(
            campaign_version=str(value["campaign_version"]),
            n_qubits=int(value["n_qubits"]),
            arm=FieldSymmetryArm(str(value["arm"])),
            seed=int(value["seed"]),
            configuration_digest=str(value["configuration_digest"]),
        )


def plan_field_symmetry_runs(
    manifest: CurieConfirmatoryManifest,
) -> tuple[FieldSymmetryRunIdentity, ...]:
    """Return the frozen 3 sizes x 10 seeds x 4 arms design."""
    runs = tuple(
        FieldSymmetryRunIdentity(
            manifest.campaign_version,
            n_qubits,
            arm,
            seed,
            manifest.digest,
        )
        for n_qubits in manifest.system_sizes
        for seed in manifest.seeds
        for arm in FieldSymmetryArm
    )
    if len(runs) != 120 or len({run.run_id for run in runs}) != 120:
        raise RuntimeError("field-symmetry planner must produce 120 unique fits")
    return runs


def field_angle_matrix(fields: tuple[float, ...], n_qubits: int) -> np.ndarray:
    """Apply the frozen broadcast map alpha(h)=-7 pi h / 6."""
    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 1 or n_qubits not in SYSTEM_SIZES:
        raise ValueError("invalid field angle coordinates")
    return np.broadcast_to(
        (-7.0 * np.pi * values / 6.0)[:, None],
        (len(values), n_qubits),
    ).copy()


def sample_field_targets(
    n_qubits: int,
    fields: tuple[float, ...],
    count: int,
    seed: int,
) -> np.ndarray:
    """Draw deterministic finite datasets from the exact declared targets."""
    if count < 1:
        raise ValueError("sample count must be positive")
    bitstrings = enumerate_bitstrings(n_qubits)
    random = np.random.default_rng(seed)
    return np.stack(
        [
            bitstrings[
                random.choice(
                    len(bitstrings),
                    size=count,
                    p=signed_field_probabilities(n_qubits, field),
                )
            ]
            for field in fields
        ]
    )


def field_metrics(
    n_qubits: int,
    field: float,
    model_probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute the preregistered exact metrics at one field."""
    target = signed_field_probabilities(n_qubits, field)
    model = np.asarray(model_probabilities, dtype=np.float64)
    if model.shape != target.shape or not np.isclose(model.sum(), 1.0, atol=1e-10):
        raise ValueError("model probabilities are not a normalized target-shaped law")
    bitstrings = enumerate_bitstrings(n_qubits)
    magnetization = (2.0 * bitstrings - 1.0).sum(axis=1)
    return {
        "bitstring_tvd": total_variation_distance(target, model),
        "profile_tvd": total_variation_distance(
            shell_probabilities(target, bitstrings),
            shell_probabilities(model, bitstrings),
        ),
        "magnetization_error": abs(
            float(magnetization @ model) - float(magnetization @ target)
        ),
    }


def _field_role(arm: FieldSymmetryArm, field: float) -> str:
    if field in arm.training_fields:
        return "training"
    if field in FIELD_HELD_OUT:
        return "held_out"
    return "reflected"


def evaluate_field_model(
    run: FieldSymmetryRunIdentity,
    params: np.ndarray,
    generators: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Exactly evaluate one fit over the common signed-field grid."""
    angles = field_angle_matrix(FIELD_EVALUATION, run.n_qubits)
    probabilities = exact_iqp_probabilities_many(
        params,
        generators,
        angles,
        run.arm.encoding,
    )
    rows: list[dict[str, Any]] = []
    by_field = dict(zip(FIELD_EVALUATION, probabilities, strict=True))
    defects: dict[str, float] = {}
    for field in (0.0, 0.075, 0.15, 0.225, 0.30):
        defect = equivariance_defect(by_field[field], by_field[-field])
        defects[f"{field:.3f}"] = defect
    for field, model in by_field.items():
        rows.append(
            {
                "field": field,
                "role": _field_role(run.arm, field),
                **field_metrics(run.n_qubits, field, model),
                "equivariance_defect": defects[f"{abs(field):.3f}"],
            }
        )
    return (
        {
            "evaluation_rows": rows,
            "equivariance_defects": defects,
            "max_equivariance_defect": max(defects.values()),
            "performance_thresholds_applied": False,
        },
        {
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "configuration_digest": np.asarray(run.configuration_digest, dtype="S64"),
            "fields": np.asarray(FIELD_EVALUATION),
            "roles": np.asarray(
                [_field_role(run.arm, h) for h in FIELD_EVALUATION], dtype="S10"
            ),
            "condition_angles": angles,
            "model_probabilities": probabilities,
        },
    )


def field_training_config(
    manifest: CurieConfirmatoryManifest,
    arm: FieldSymmetryArm,
    updates: int | None = None,
) -> TrainingConfig:
    settings = manifest.raw["training"]
    n_conditions = len(arm.training_fields)
    total = int(settings["total_target_minibatch"])
    if total % n_conditions:
        raise ValueError("target minibatch must divide evenly across fields")
    steps = int(settings["updates"] if updates is None else updates)
    if steps < 1:
        raise ValueError("updates must be positive")
    return TrainingConfig(
        steps=steps,
        batch_size=total // n_conditions,
        learning_rate=float(settings["initial_learning_rate"]),
        final_learning_rate=float(settings["final_learning_rate"]),
        validation_interval=min(int(settings["validation_interval"]), steps),
        sigmas=tuple(float(value) for value in settings["bandwidths"]),
        n_ops=int(settings["visible_character_masks"]),
        n_iqp_samples=int(settings["training_model_samples"]),
        validation_n_ops=int(settings["visible_character_masks"]),
        validation_n_iqp_samples=int(settings["validation_model_samples"]),
    )


class FieldSymmetryStore:
    """Minimal immutable ledger for the signed-field campaign."""

    def __init__(self, root: Path, manifest: CurieConfirmatoryManifest) -> None:
        self.root = root.resolve()
        self.manifest = manifest
        self.results = self.root / "results"
        self.checkpoints = self.root / "checkpoints"
        self.probabilities = self.root / "probabilities"
        self._planned = {run.run_id: run for run in plan_field_symmetry_runs(manifest)}
        self.root.mkdir(parents=True, exist_ok=True)
        snapshot = self.root / "manifest.json"
        if snapshot.exists():
            if read_immutable_json(snapshot) != manifest.to_dict():
                raise RuntimeError("field campaign manifest snapshot differs")
        else:
            write_immutable_json(snapshot, manifest.to_dict())

    def validate(self, run: FieldSymmetryRunIdentity) -> None:
        if self._planned.get(run.run_id) != run:
            raise RuntimeError("field run is outside the frozen plan")

    def result_path(self, run: FieldSymmetryRunIdentity) -> Path:
        return self.results / f"{run.run_id}.json"

    def checkpoint_path(self, run: FieldSymmetryRunIdentity) -> Path:
        return self.checkpoints / f"{run.run_id}.npz"

    def probabilities_path(self, run: FieldSymmetryRunIdentity) -> Path:
        return self.probabilities / f"{run.run_id}.npz"

    def has_complete(self, run: FieldSymmetryRunIdentity) -> bool:
        path = self.result_path(run)
        if not path.exists():
            return False
        payload = read_immutable_json(path)
        if FieldSymmetryRunIdentity.from_dict(payload["run"]) != run:
            raise RuntimeError("conflicting field result identity")
        for key in ("checkpoint", "probabilities"):
            artifact = Path(payload["artifacts"][key]["path"])
            if not artifact.is_absolute():
                artifact = self.root / artifact
            if sha256_file(artifact) != payload["artifacts"][key]["sha256"]:
                raise RuntimeError("field result artifact checksum differs")
        return True

    def load(self, run: FieldSymmetryRunIdentity) -> dict[str, Any]:
        if not self.has_complete(run):
            raise RuntimeError("field result is incomplete")
        return read_immutable_json(self.result_path(run))


def _source_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "configs/curie_weiss_confirmatory_v1.json",
        "src/iqp_cond/ansatz.py",
        "src/iqp_cond/artifacts.py",
        "src/iqp_cond/confirmatory_curie.py",
        "src/iqp_cond/field_symmetry_runner.py",
        "src/iqp_cond/encoding.py",
        "src/iqp_cond/exact.py",
        "src/iqp_cond/gaussian_data.py",
        "src/iqp_cond/ising_training.py",
        "src/iqp_cond/training.py",
        "src/iqp_cond/mmd.py",
    ):
        digest.update(relative.encode())
        digest.update((repository / relative).read_bytes())
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _write_or_verify_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Create an immutable archive, or verify an identical interrupted write."""
    if not path.exists():
        return write_immutable_npz(path, arrays)
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays):
                raise RuntimeError("partial field artifact schema differs")
            for key, expected in arrays.items():
                actual = stored[key]
                if actual.dtype != expected.dtype or actual.shape != expected.shape:
                    raise RuntimeError("partial field artifact array schema differs")
                equal_nan = np.issubdtype(actual.dtype, np.inexact)
                if not np.array_equal(actual, expected, equal_nan=equal_nan):
                    raise RuntimeError("partial field artifact content differs")
    except (OSError, ValueError) as error:
        raise RuntimeError("partial field artifact is unreadable") from error
    return sha256_file(path)


def execute_field_symmetry_run(
    repository: Path,
    manifest: CurieConfirmatoryManifest,
    store: FieldSymmetryStore,
    run: FieldSymmetryRunIdentity,
    *,
    invocation: tuple[str, ...],
    updates: int | None = None,
    training_samples_per_field: int = 10_000,
    validation_samples_per_field: int = 2_000,
) -> dict[str, Any]:
    """Train, exactly evaluate, and atomically record one declared fit."""
    repository = repository.resolve()
    store.validate(run)
    if store.has_complete(run):
        return store.load(run)
    started = datetime.now(UTC)
    start_time = time.perf_counter()
    fields = run.arm.training_fields
    training_seed = run.seed + 100_000
    validation_seed = run.seed + 1_100_000
    training_targets = sample_field_targets(
        run.n_qubits, fields, training_samples_per_field, training_seed
    )
    validation_targets = sample_field_targets(
        run.n_qubits, fields, validation_samples_per_field, validation_seed
    )
    angles = field_angle_matrix(fields, run.n_qubits)
    ansatz = complete_ansatz(run.n_qubits)
    config = field_training_config(manifest, run.arm, updates)
    result = train(
        f"field_{run.run_id[:12]}",
        ansatz,
        angles,
        training_targets,
        validation_targets,
        run.seed,
        config,
        run.arm.encoding,
    )
    metrics, probability_arrays = evaluate_field_model(
        run, result.best_params, np.asarray(result.generators)
    )
    metrics.update(
        {
            "best_validation_conditional_mmd": result.best_validation_loss,
            "best_step": result.best_step,
            "optimizer_updates": config.steps,
            "training_samples_per_field": training_samples_per_field,
            "validation_samples_per_field": validation_samples_per_field,
        }
    )
    checkpoint = store.checkpoint_path(run)
    checkpoint_arrays = {
        "run_id": np.asarray(run.run_id, dtype="S64"),
        "configuration_digest": np.asarray(manifest.digest, dtype="S64"),
        "generators": np.asarray(result.generators, dtype=np.int8),
        "training_fields": np.asarray(fields),
        "condition_angles": result.condition_angles,
        "best_params": result.best_params,
        "final_params": result.final_params,
        "best_step": np.asarray(result.best_step, dtype=np.int64),
        "best_validation_loss": np.asarray(result.best_validation_loss),
        "train_steps": result.train_steps,
        "train_losses": result.train_losses,
        "validation_steps": result.validation_steps,
        "validation_losses": result.validation_losses,
    }
    checkpoint_sha = _write_or_verify_npz(checkpoint, checkpoint_arrays)
    probabilities = store.probabilities_path(run)
    probability_sha = _write_or_verify_npz(probabilities, probability_arrays)
    source_digest = _source_digest(repository)
    completed = datetime.now(UTC)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "completion_status": "completed",
        "run": run.to_dict(),
        "manifest_digest": manifest.digest,
        "code_revision": _git_revision(repository),
        "source_tree_digest": source_digest,
        "environment": {
            **runtime_environment(repository / "uv.lock"),
            "platform": platform.platform(),
        },
        "invocation": list(invocation),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "runtime_seconds": time.perf_counter() - start_time,
        "peak_memory_bytes": _peak_rss_bytes(),
        "fixed_encoder": {
            "encoding": run.arm.encoding,
            "angle_map": "alpha(h)=-7*pi*h/6",
            "training_fields": list(fields),
            "trainable": False,
        },
        "randomness": {
            "optimizer_seed": run.seed,
            "training_data_seed": training_seed,
            "validation_data_seed": validation_seed,
            "validation_estimator_seed": run.seed + 1_000_000,
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint.relative_to(store.root)),
                "sha256": checkpoint_sha,
            },
            "probabilities": {
                "path": str(probabilities.relative_to(store.root)),
                "sha256": probability_sha,
            },
        },
        "metrics": metrics,
    }
    write_immutable_json(store.result_path(run), payload)
    return payload
