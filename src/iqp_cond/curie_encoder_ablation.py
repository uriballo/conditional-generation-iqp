"""Frozen Curie--Weiss encoder/map ablation and local execution runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ansatz import complete_ansatz
from .artifacts import read_immutable_json, sha256_file, write_immutable_json
from .confirmatory_statistics import median_iqr, paired_interval
from .curie_weiss_data import exact_curie_weiss_target
from .encoding import ANGLE_ENCODING, SPIN_FLIP_BRANCH_ENCODING
from .exact import enumerate_bitstrings, exact_iqp_visible_probabilities
from .exact_tvd_campaign import (
    data_initialize,
    expected_bodyness_sigmas,
)
from .exact_tvd_campaign import (
    load_manifest as load_base_manifest,
)
from .exact_tvd_runner import (
    _ancilla_support,
    _curie_angle,
    _diagnostic_mmd,
    _empirical_probabilities,
    _git_revision,
    _metrics,
    _runtime_metadata,
    _weighted_loss_functions,
    _write_immutable_npz,
    load_data_archive,
    load_protocol_selection,
)
from .exact_tvd_training import CampaignTrainingConfig, train_campaign
from .protocol import ProtocolValidationError, content_digest

CAMPAIGN_VERSION = "curie_encoder_ablation_v1"
DEFAULT_MANIFEST = Path("configs/curie_encoder_ablation_v1.json")
ARMS = (
    "product_generic_affine",
    "spin_flip_generic_affine",
    "product_informed_piecewise",
    "spin_flip_informed_piecewise",
)
VISIBLE_QUBITS = (8, 12, 16)
ANCILLAS = (0, 10)
SEEDS = tuple(range(100, 110))
TRAINING_CONDITIONS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_OUT_CONDITIONS = (0.6, 0.8, 1.0, 1.2, 1.4)
NON_LANDMARK_HELD_OUT = (0.6, 0.8, 1.2, 1.4)


@dataclass(frozen=True, slots=True)
class AblationManifest:
    """Strict immutable protocol for the 2x2 encoder/map ablation."""

    raw: Mapping[str, Any]
    source_path: Path

    @classmethod
    def from_json(cls, path: Path = DEFAULT_MANIFEST) -> AblationManifest:
        resolved = Path(path).resolve()
        try:
            raw = json.loads(resolved.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                f"invalid ablation manifest: {resolved}"
            ) from error
        manifest = cls(raw, resolved)
        manifest._validate()
        return manifest

    def _validate(self) -> None:
        expected_keys = {
            "schema_version",
            "campaign_version",
            "base_campaign",
            "base_manifest",
            "base_protocol_selection",
            "base_data_directory",
            "visible_qubits",
            "ancillas",
            "seeds",
            "training_conditions",
            "held_out_conditions",
            "generic_map",
            "arms",
            "training",
            "evaluation",
            "artifacts",
        }
        if set(self.raw) != expected_keys:
            raise ProtocolValidationError("ablation manifest schema differs")
        if (
            self.raw["schema_version"] != 1
            or self.raw["campaign_version"] != CAMPAIGN_VERSION
            or self.raw["base_campaign"] != "exact_tvd_ancilla_v1"
        ):
            raise ProtocolValidationError("ablation manifest identity differs")
        if tuple(self.raw["visible_qubits"]) != VISIBLE_QUBITS:
            raise ProtocolValidationError("visible-qubit coordinates differ")
        if tuple(self.raw["ancillas"]) != ANCILLAS:
            raise ProtocolValidationError("ancilla coordinates differ")
        if tuple(self.raw["seeds"]) != SEEDS:
            raise ProtocolValidationError("seed coordinates differ")
        if tuple(self.raw["training_conditions"]) != TRAINING_CONDITIONS:
            raise ProtocolValidationError("training conditions differ")
        if tuple(self.raw["held_out_conditions"]) != HELD_OUT_CONDITIONS:
            raise ProtocolValidationError("held-out conditions differ")
        if tuple(self.raw["arms"]) != ARMS:
            raise ProtocolValidationError("ablation arms differ")
        if self.raw["generic_map"] != {
            "domain": [0.5, 1.5],
            "offset": 0.125,
            "slope": 0.25,
        }:
            raise ProtocolValidationError("generic map differs")
        if self.raw["training"] != {
            "updates": 2000,
            "batch_size": 100,
            "n_ops_per_condition_per_bandwidth": 2000,
            "candidate_checkpoints": [
                500,
                1000,
                1500,
                2000,
                "lowest_validation_mmd",
            ],
            "checkpoint_selection": "mean_exact_validation_tvd_on_training_conditions",
        }:
            raise ProtocolValidationError("training protocol differs")
        if self.raw["evaluation"] != {
            "primary_metric": "exact_visible_bitstring_tvd",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260901,
            "confidence_level": 0.95,
            "landmark_condition": 1.0,
            "non_landmark_held_out_conditions": [0.6, 0.8, 1.2, 1.4],
        }:
            raise ProtocolValidationError("evaluation protocol differs")
        if self.raw["artifacts"] != {
            "root": "results/curie-encoder-ablation-v1",
            "immutable": True,
        }:
            raise ProtocolValidationError("artifact protocol differs")

    @property
    def digest(self) -> str:
        return content_digest(dict(self.raw))

    @property
    def repository(self) -> Path:
        return self.source_path.parent.parent

    def resolve(self, key: str) -> Path:
        return self.repository / str(self.raw[key])

    @property
    def output_root(self) -> Path:
        return self.repository / str(self.raw["artifacts"]["root"])


@dataclass(frozen=True, slots=True)
class AblationIdentity:
    manifest_digest: str
    arm: str
    visible_qubits: int
    ancillas: int
    seed: int

    def __post_init__(self) -> None:
        if (
            self.arm not in ARMS
            or self.visible_qubits not in VISIBLE_QUBITS
            or self.ancillas not in ANCILLAS
            or self.seed not in SEEDS
            or len(self.manifest_digest) != 64
        ):
            raise ProtocolValidationError("ablation identity is outside the manifest")

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AblationIdentity:
        if set(value) != {
            "manifest_digest",
            "arm",
            "visible_qubits",
            "ancillas",
            "seed",
        }:
            raise ProtocolValidationError("ablation identity schema differs")
        return cls(
            str(value["manifest_digest"]),
            str(value["arm"]),
            int(value["visible_qubits"]),
            int(value["ancillas"]),
            int(value["seed"]),
        )


def generic_angle(beta: float) -> float:
    """Target-independent affine map with no feature at beta=1."""
    if not np.isfinite(beta) or not 0.5 <= beta <= 1.5:
        raise ValueError("beta is outside the generic-map domain")
    return float(np.pi * (0.125 + 0.25 * (float(beta) - 0.5)))


def arm_dispatch(arm: str, beta: float) -> tuple[str, float]:
    """Return the frozen input-state family and angle for one arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown ablation arm: {arm}")
    encoding = (
        SPIN_FLIP_BRANCH_ENCODING if arm.startswith("spin_flip") else ANGLE_ENCODING
    )
    angle = (
        _curie_angle(beta)
        if arm.endswith("informed_piecewise")
        else generic_angle(beta)
    )
    return encoding, angle


def condition_angles(
    arm: str, conditions: Sequence[float], total_qubits: int
) -> np.ndarray:
    """Broadcast the arm-specific scalar map over visible and ancilla qubits."""
    if total_qubits < 1:
        raise ValueError("total_qubits must be positive")
    mapped = np.asarray([arm_dispatch(arm, float(beta))[1] for beta in conditions])
    return np.repeat(mapped[:, None], total_qubits, axis=1)


def plan_runs(manifest: AblationManifest) -> tuple[AblationIdentity, ...]:
    return tuple(
        AblationIdentity(manifest.digest, arm, n, ancillas, seed)
        for arm in ARMS
        for n in VISIBLE_QUBITS
        for ancillas in ANCILLAS
        for seed in SEEDS
    )


def plan_encoder_only(manifest: AblationManifest) -> tuple[dict[str, Any], ...]:
    """Return the 24 deterministic arm/size/ancilla diagnostic cells."""
    return tuple(
        {
            "manifest_digest": manifest.digest,
            "arm": arm,
            "visible_qubits": n,
            "ancillas": ancillas,
            "cell_id": content_digest(
                {
                    "manifest_digest": manifest.digest,
                    "arm": arm,
                    "visible_qubits": n,
                    "ancillas": ancillas,
                    "stage": "encoder_only",
                }
            ),
        }
        for arm in ARMS
        for n in VISIBLE_QUBITS
        for ancillas in ANCILLAS
    )


def encoder_only_visible_probabilities(
    n_visible: int,
    ancillas: int,
    arm: str,
    beta: float,
) -> np.ndarray:
    """Analytically marginalize the zero-phase input over ancillas."""
    if n_visible < 1 or ancillas < 0:
        raise ValueError("invalid visible/ancilla register")
    encoding, angle = arm_dispatch(arm, beta)
    bits = enumerate_bitstrings(n_visible)
    weights = bits.sum(axis=1)
    c = float(np.cos(angle / 2.0))
    s = float(np.sin(angle / 2.0))
    forward = c ** (n_visible - weights) * s**weights
    if encoding == ANGLE_ENCODING:
        probabilities = forward**2
    else:
        reflected = s ** (n_visible - weights) * c**weights
        overlap = float(np.sin(angle))
        numerator = (
            forward**2 + reflected**2 + 2.0 * forward * reflected * overlap**ancillas
        )
        normalization = 2.0 * (1.0 + overlap ** (n_visible + ancillas))
        probabilities = numerator / normalization
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return probabilities


def _training_indices(
    base_config: Mapping[str, Any], archive: Mapping[str, np.ndarray]
) -> np.ndarray:
    declared = np.asarray(base_config["datasets"]["curie_weiss"]["training_conditions"])
    if not np.array_equal(declared, np.asarray(TRAINING_CONDITIONS)):
        raise ProtocolValidationError("base campaign training grid differs")
    return np.asarray(
        [
            int(np.flatnonzero(np.isclose(archive["conditions"], beta))[0])
            for beta in declared
        ]
    )


def _load_base_resources(
    manifest: AblationManifest,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    base = load_base_manifest(manifest.resolve("base_manifest"))
    selection = load_protocol_selection(
        manifest.resolve("base_protocol_selection"), base
    )
    return base, selection, manifest.resolve("base_data_directory")


def _ablation_source_digest(repository: Path) -> str:
    names = (
        "configs/curie_encoder_ablation_v1.json",
        "src/iqp_cond/curie_encoder_ablation.py",
        "src/iqp_cond/encoding.py",
        "src/iqp_cond/exact.py",
        "src/iqp_cond/exact_tvd_runner.py",
        "src/iqp_cond/exact_tvd_training.py",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((repository / name).read_bytes())
    return digest.hexdigest()


def _load_existing_run(
    path: Path, run: AblationIdentity, stage: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_immutable_json(path)
    if (
        payload.get("stage") != stage
        or payload.get("run_id") != run.run_id
        or AblationIdentity.from_dict(payload["run_identity"]) != run
    ):
        raise RuntimeError(f"existing {stage} artifact belongs to another run")
    return payload


def train_run(
    manifest: AblationManifest,
    run: AblationIdentity,
    *,
    output_root: Path | None = None,
    steps_override: int | None = None,
    n_ops_override: int | None = None,
    n_samples_override: int | None = None,
) -> dict[str, Any]:
    """Train one matched ablation identity and persist candidate checkpoints."""
    if run not in set(plan_runs(manifest)):
        raise ProtocolValidationError("run does not belong to this manifest")
    output = manifest.output_root if output_root is None else Path(output_root)
    record_path = output / "training" / f"{run.run_id}.json"
    existing = _load_existing_run(record_path, run, "training")
    if existing is not None:
        return existing
    start = time.monotonic()
    base, selection, data_directory = _load_base_resources(manifest)
    archive = load_data_archive(base, data_directory, "curie_weiss", run.visible_qubits)
    indices = _training_indices(base, archive)
    training = archive["train"][indices]
    validation = archive["validation"][indices]
    total = run.visible_qubits + run.ancillas
    generators = np.asarray(complete_ansatz(total).generators)
    initial = data_initialize(
        training.reshape(-1, run.visible_qubits),
        generators,
        _ancilla_support(generators, run.visible_qubits),
        run.seed,
    )
    angles = condition_angles(run.arm, TRAINING_CONDITIONS, total)
    encoding = arm_dispatch(run.arm, 1.0)[0]
    sigmas = expected_bodyness_sigmas(run.visible_qubits)
    n_ops = int(
        manifest.raw["training"]["n_ops_per_condition_per_bandwidth"]
        if n_ops_override is None
        else n_ops_override
    )
    n_samples = int(
        selection["n_samples"] if n_samples_override is None else n_samples_override
    )
    training_loss, validation_mmd, operator_bank = _weighted_loss_functions(
        generators=generators,
        angles=angles,
        validation_targets=validation,
        sigmas=sigmas,
        n_ops=n_ops,
        n_samples=n_samples,
        seed=run.seed,
        encoding=encoding,
    )
    optimizer_settings = selection.get("optimizer_settings", {})
    steps = int(
        manifest.raw["training"]["updates"]
        if steps_override is None
        else steps_override
    )
    config = CampaignTrainingConfig(
        steps=steps,
        batch_size=int(manifest.raw["training"]["batch_size"]),
        n_samples=n_samples,
        validation_interval=min(int(base["training"]["validation_interval"]), steps),
        optimizer=str(selection["optimizer"]),
        learning_rate=float(optimizer_settings.get("learning_rate", 0.01)),
        final_learning_rate=float(optimizer_settings.get("final_learning_rate", 0.001)),
        warmup_steps=min(int(optimizer_settings.get("warmup_steps", 100)), steps),
        b1=float(optimizer_settings.get("b1", 0.9)),
        b2=float(optimizer_settings.get("b2", 0.999)),
    )
    result = train_campaign(
        initial,
        training,
        validation_mmd,
        training_loss,
        run.seed,
        config,
    )
    checkpoint_path = output / "checkpoints" / f"{run.run_id}.npz"
    mask_digest = hashlib.sha256()
    for masks, weights in operator_bank:
        mask_digest.update(masks.tobytes())
        mask_digest.update(weights.tobytes())
    checkpoint_digest = _write_immutable_npz(
        checkpoint_path,
        {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "initial_params": initial,
            "checkpoint_params": np.stack(
                [value.params for value in result.checkpoints]
            ),
            "checkpoint_steps": np.asarray(
                [value.step for value in result.checkpoints]
            ),
            "checkpoint_validation_mmd": np.asarray(
                [value.validation_mmd for value in result.checkpoints]
            ),
            "checkpoint_reasons": np.asarray(
                [value.reason for value in result.checkpoints], dtype="S32"
            ),
            "train_steps": result.train_steps,
            "train_losses": result.train_losses,
            "validation_steps": result.validation_steps,
            "validation_losses": result.validation_losses,
            "conditions": np.asarray(TRAINING_CONDITIONS),
            "condition_angles": angles,
            "encoding": np.asarray(encoding, dtype=f"S{len(encoding)}"),
        },
    )
    data_path = data_directory / f"curie_weiss-n{run.visible_qubits}.npz"
    payload = {
        "schema_version": 1,
        "campaign": CAMPAIGN_VERSION,
        "manifest_digest": manifest.digest,
        "stage": "training",
        "run_id": run.run_id,
        "run_identity": run.to_dict(),
        "base_campaign_digest": archive["configuration_digest"].item().decode(),
        "protocol_selection_sha256": sha256_file(
            manifest.resolve("base_protocol_selection")
        ),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "split_hashes": {
            role: hashlib.sha256(archive[role].tobytes()).hexdigest()
            for role in ("train", "validation", "test")
        },
        "checkpoint_path": str(checkpoint_path.relative_to(output)),
        "checkpoint_sha256": checkpoint_digest,
        "operator_bank_sha256": mask_digest.hexdigest(),
        "encoding": encoding,
        "condition_map": (
            "informed_piecewise"
            if run.arm.endswith("informed_piecewise")
            else "generic_affine"
        ),
        "training_overrides": {
            "steps": steps_override,
            "n_ops": n_ops_override,
            "n_samples": n_samples_override,
        },
        "source_tree_sha256": _ablation_source_digest(manifest.repository),
        "git_revision": _git_revision(manifest.repository),
        "runtime": _runtime_metadata(start),
    }
    write_immutable_json(record_path, payload)
    return payload


def evaluate_run(
    manifest: AblationManifest,
    run: AblationIdentity,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Select on training-condition validation TVD, then evaluate all conditions."""
    output = manifest.output_root if output_root is None else Path(output_root)
    record_path = output / "evaluations" / f"{run.run_id}.json"
    existing = _load_existing_run(record_path, run, "evaluation")
    if existing is not None:
        return existing
    start = time.monotonic()
    training_payload = _load_existing_run(
        output / "training" / f"{run.run_id}.json", run, "training"
    )
    if training_payload is None:
        raise FileNotFoundError(f"training artifact is absent for {run.run_id}")
    checkpoint_path = output / str(training_payload["checkpoint_path"])
    if sha256_file(checkpoint_path) != training_payload["checkpoint_sha256"]:
        raise RuntimeError("checkpoint digest mismatch")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        candidates = np.asarray(checkpoint["checkpoint_params"])
        steps = np.asarray(checkpoint["checkpoint_steps"])
        reasons = np.asarray(checkpoint["checkpoint_reasons"])
        validation_mmd = np.asarray(checkpoint["checkpoint_validation_mmd"])
    base, _, data_directory = _load_base_resources(manifest)
    archive = load_data_archive(base, data_directory, "curie_weiss", run.visible_qubits)
    indices = _training_indices(base, archive)
    validation_targets = [
        _empirical_probabilities(archive["validation"][index]) for index in indices
    ]
    total = run.visible_qubits + run.ancillas
    generators = np.asarray(complete_ansatz(total).generators)
    encoding = arm_dispatch(run.arm, 1.0)[0]
    validation_angles = condition_angles(run.arm, TRAINING_CONDITIONS, total)
    candidate_tvds: list[float] = []
    for candidate in candidates:
        predictions = exact_iqp_visible_probabilities(
            candidate,
            generators,
            validation_angles,
            run.visible_qubits,
            run.ancillas,
            encoding,
        )
        candidate_tvds.append(
            float(
                np.mean(
                    [
                        0.5 * np.abs(target - model).sum()
                        for target, model in zip(
                            validation_targets, predictions, strict=True
                        )
                    ]
                )
            )
        )
    selected_index = int(np.argmin(candidate_tvds))
    selected_params = candidates[selected_index]
    conditions = np.asarray(sorted((*TRAINING_CONDITIONS, *HELD_OUT_CONDITIONS)))
    predictions = exact_iqp_visible_probabilities(
        selected_params,
        generators,
        condition_angles(run.arm, conditions, total),
        run.visible_qubits,
        run.ancillas,
        encoding,
    )
    sigmas = expected_bodyness_sigmas(run.visible_qubits)
    rows: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []
    for beta, prediction in zip(conditions, predictions, strict=True):
        target = exact_curie_weiss_target(run.visible_qubits, float(beta)).probabilities
        targets.append(target)
        split = "training" if float(beta) in TRAINING_CONDITIONS else "held_out"
        data_index = int(np.flatnonzero(np.isclose(archive["conditions"], beta))[0])
        row = {
            "condition": float(beta),
            "split": split,
            "critical_landmark": bool(np.isclose(beta, 1.0)),
            "non_landmark_held_out": bool(
                split == "held_out"
                and any(np.isclose(beta, x) for x in NON_LANDMARK_HELD_OUT)
            ),
            **_metrics(
                "curie_weiss",
                run.visible_qubits,
                float(beta),
                target,
                prediction,
                sigmas,
            ),
        }
        if split == "training":
            empirical_train = _empirical_probabilities(archive["train"][data_index])
            row["training_mmd"] = _diagnostic_mmd(prediction, empirical_train, sigmas)
        rows.append(row)
    probability_path = output / "probabilities" / f"{run.run_id}.npz"
    probability_digest = _write_immutable_npz(
        probability_path,
        {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "conditions": conditions,
            "probabilities": predictions,
            "target_probabilities": np.asarray(targets),
            "selected_params": selected_params,
            "selected_checkpoint_index": np.asarray(selected_index),
        },
    )
    payload = {
        "schema_version": 1,
        "campaign": CAMPAIGN_VERSION,
        "manifest_digest": manifest.digest,
        "stage": "evaluation",
        "run_id": run.run_id,
        "run_identity": run.to_dict(),
        "selection": {
            "conditions": list(TRAINING_CONDITIONS),
            "held_out_conditions_used": [],
            "candidate_steps": steps.tolist(),
            "candidate_reasons": [value.decode() for value in reasons],
            "candidate_validation_mmd": validation_mmd.tolist(),
            "candidate_mean_exact_validation_tvd": candidate_tvds,
            "selected_index": selected_index,
            "selected_step": int(steps[selected_index]),
        },
        "metrics": rows,
        "probabilities_path": str(probability_path.relative_to(output)),
        "probabilities_sha256": probability_digest,
        "source_tree_sha256": _ablation_source_digest(manifest.repository),
        "runtime": _runtime_metadata(start),
    }
    write_immutable_json(record_path, payload)
    return payload


def run_encoder_only(
    manifest: AblationManifest,
    *,
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Materialize every deterministic zero-phase diagnostic cell."""
    output = manifest.output_root if output_root is None else Path(output_root)
    records: list[dict[str, Any]] = []
    conditions = np.asarray(sorted((*TRAINING_CONDITIONS, *HELD_OUT_CONDITIONS)))
    for cell in plan_encoder_only(manifest):
        path = output / "encoder-only" / f"{cell['cell_id']}.json"
        if path.exists():
            records.append(read_immutable_json(path))
            continue
        predictions = np.stack(
            [
                encoder_only_visible_probabilities(
                    int(cell["visible_qubits"]),
                    int(cell["ancillas"]),
                    str(cell["arm"]),
                    float(beta),
                )
                for beta in conditions
            ]
        )
        targets = np.stack(
            [
                exact_curie_weiss_target(
                    int(cell["visible_qubits"]), float(beta)
                ).probabilities
                for beta in conditions
            ]
        )
        sigmas = expected_bodyness_sigmas(int(cell["visible_qubits"]))
        rows = [
            {
                "condition": float(beta),
                "split": (
                    "training" if float(beta) in TRAINING_CONDITIONS else "held_out"
                ),
                "critical_landmark": bool(np.isclose(beta, 1.0)),
                "non_landmark_held_out": bool(
                    float(beta) in HELD_OUT_CONDITIONS
                    and any(np.isclose(beta, x) for x in NON_LANDMARK_HELD_OUT)
                ),
                **_metrics(
                    "curie_weiss",
                    int(cell["visible_qubits"]),
                    float(beta),
                    target,
                    prediction,
                    sigmas,
                ),
            }
            for beta, target, prediction in zip(
                conditions, targets, predictions, strict=True
            )
        ]
        probability_path = (
            output / "encoder-only-probabilities" / f"{cell['cell_id']}.npz"
        )
        probability_digest = _write_immutable_npz(
            probability_path,
            {
                "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
                "cell_id": np.asarray(cell["cell_id"], dtype="S64"),
                "conditions": conditions,
                "probabilities": predictions,
                "target_probabilities": targets,
            },
        )
        payload = {
            "schema_version": 1,
            "campaign": CAMPAIGN_VERSION,
            "manifest_digest": manifest.digest,
            "stage": "encoder_only",
            "cell": cell,
            "metrics": rows,
            "probabilities_path": str(probability_path.relative_to(output)),
            "probabilities_sha256": probability_digest,
        }
        write_immutable_json(path, payload)
        records.append(payload)
    return records


def run_smoke(
    manifest: AblationManifest,
    output_root: Path,
    *,
    steps: int = 10,
) -> list[dict[str, Any]]:
    """Train and evaluate all four arms at n=8, r in {0,10}, seed=100."""
    if Path(output_root).resolve() == manifest.output_root.resolve():
        raise ValueError("smoke output must be isolated from production artifacts")
    records = []
    for arm in ARMS:
        for ancillas in ANCILLAS:
            run = AblationIdentity(manifest.digest, arm, 8, ancillas, 100)
            train_run(
                manifest,
                run,
                output_root=output_root,
                steps_override=steps,
                n_ops_override=4,
                n_samples_override=16,
            )
            records.append(evaluate_run(manifest, run, output_root=output_root))
    return records


def status(
    manifest: AblationManifest, output_root: Path | None = None
) -> dict[str, Any]:
    output = manifest.output_root if output_root is None else Path(output_root)
    planned = plan_runs(manifest)
    trained = (
        {path.stem for path in (output / "training").glob("*.json")}
        if (output / "training").is_dir()
        else set()
    )
    evaluated = (
        {path.stem for path in (output / "evaluations").glob("*.json")}
        if (output / "evaluations").is_dir()
        else set()
    )
    failures = (
        list((output / "failures").glob("*.json"))
        if (output / "failures").is_dir()
        else []
    )
    run_ids = {run.run_id for run in planned}
    return {
        "campaign": CAMPAIGN_VERSION,
        "manifest_digest": manifest.digest,
        "planned": len(planned),
        "trained": len(trained & run_ids),
        "evaluated": len(evaluated & run_ids),
        "failed": len(failures),
        "encoder_only_planned": len(plan_encoder_only(manifest)),
        "encoder_only_completed": (
            len(list((output / "encoder-only").glob("*.json")))
            if (output / "encoder-only").is_dir()
            else 0
        ),
    }


def _scope_predicates() -> dict[str, Callable[[Mapping[str, Any]], bool]]:
    return {
        "training": lambda row: row.get("split") == "training",
        "landmark_beta_1": lambda row: row.get("critical_landmark") is True,
        "non_landmark_held_out": lambda row: row.get("non_landmark_held_out") is True,
        "all_held_out": lambda row: row.get("split") == "held_out",
    }


def summarize(
    manifest: AblationManifest,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Finalize complete seed-level cells and frozen factorial contrasts."""
    output = manifest.output_root if output_root is None else Path(output_root)
    runs = plan_runs(manifest)
    records: dict[str, dict[str, Any]] = {}
    for run in runs:
        path = output / "evaluations" / f"{run.run_id}.json"
        if not path.exists():
            raise RuntimeError(
                f"cannot summarize incomplete campaign; missing {run.run_id}"
            )
        records[run.run_id] = read_immutable_json(path)
    predicates = _scope_predicates()

    def seed_values(
        arm: str,
        n: int,
        ancillas: int,
        scope: str,
        metric: str,
    ) -> dict[int, float]:
        result: dict[int, float] = {}
        for run in runs:
            if (run.arm, run.visible_qubits, run.ancillas) != (arm, n, ancillas):
                continue
            rows = [
                row for row in records[run.run_id]["metrics"] if predicates[scope](row)
            ]
            if not rows:
                raise RuntimeError(f"empty summary scope: {scope}")
            result[run.seed] = float(np.mean([float(row[metric]) for row in rows]))
        if set(result) != set(SEEDS):
            raise RuntimeError("summary cell has an incomplete seed set")
        return result

    def gap_values(arm: str, n: int, ancillas: int, metric: str) -> dict[int, float]:
        training = seed_values(arm, n, ancillas, "training", metric)
        held_out = seed_values(arm, n, ancillas, "all_held_out", metric)
        return {seed: held_out[seed] - training[seed] for seed in SEEDS}

    metrics = ("bitstring_tvd", "profile_tvd")
    cells: list[dict[str, Any]] = []
    for arm in ARMS:
        for n in VISIBLE_QUBITS:
            for ancillas in ANCILLAS:
                summaries: dict[str, Any] = {}
                for scope in predicates:
                    summaries[scope] = {
                        metric: median_iqr(
                            list(seed_values(arm, n, ancillas, scope, metric).values())
                        )
                        for metric in metrics
                    }
                summaries["held_out_minus_training"] = {
                    metric: median_iqr(
                        list(gap_values(arm, n, ancillas, metric).values())
                    )
                    for metric in metrics
                }
                cells.append(
                    {
                        "arm": arm,
                        "visible_qubits": n,
                        "ancillas": ancillas,
                        "summaries": summaries,
                    }
                )
    stats = manifest.raw["evaluation"]

    def interval(values: dict[int, float]) -> dict[str, object]:
        return paired_interval(
            values,
            replicates=int(stats["bootstrap_replicates"]),
            seed=int(stats["bootstrap_seed"]),
            confidence_level=float(stats["confidence_level"]),
        )

    def values_for(
        arm: str, n: int, ancillas: int, scope: str, metric: str
    ) -> dict[int, float]:
        return (
            gap_values(arm, n, ancillas, metric)
            if scope == "held_out_minus_training"
            else seed_values(arm, n, ancillas, scope, metric)
        )

    contrasts: list[dict[str, Any]] = []
    scopes = (*predicates.keys(), "held_out_minus_training")
    for n in VISIBLE_QUBITS:
        for scope in scopes:
            for metric in metrics:
                for arm in ARMS:
                    r0 = values_for(arm, n, 0, scope, metric)
                    r10 = values_for(arm, n, 10, scope, metric)
                    contrasts.append(
                        {
                            "kind": "ancilla_effect",
                            "arm": arm,
                            "visible_qubits": n,
                            "scope": scope,
                            "metric": metric,
                            "contrast": "r0_minus_r10",
                            "positive_favors": "r10",
                            "interval": interval(
                                {seed: r0[seed] - r10[seed] for seed in SEEDS}
                            ),
                        }
                    )
                for ancillas in ANCILLAS:
                    for map_name in ("generic_affine", "informed_piecewise"):
                        product = values_for(
                            f"product_{map_name}", n, ancillas, scope, metric
                        )
                        spin = values_for(
                            f"spin_flip_{map_name}", n, ancillas, scope, metric
                        )
                        contrasts.append(
                            {
                                "kind": "symmetry_effect",
                                "map": map_name,
                                "visible_qubits": n,
                                "ancillas": ancillas,
                                "scope": scope,
                                "metric": metric,
                                "contrast": "product_minus_spin_flip",
                                "positive_favors": "spin_flip",
                                "interval": interval(
                                    {seed: product[seed] - spin[seed] for seed in SEEDS}
                                ),
                            }
                        )
                    for encoder in ("product", "spin_flip"):
                        generic = values_for(
                            f"{encoder}_generic_affine",
                            n,
                            ancillas,
                            scope,
                            metric,
                        )
                        informed = values_for(
                            f"{encoder}_informed_piecewise",
                            n,
                            ancillas,
                            scope,
                            metric,
                        )
                        contrasts.append(
                            {
                                "kind": "map_effect",
                                "encoder": encoder,
                                "visible_qubits": n,
                                "ancillas": ancillas,
                                "scope": scope,
                                "metric": metric,
                                "contrast": "generic_minus_informed",
                                "positive_favors": "informed_piecewise",
                                "interval": interval(
                                    {
                                        seed: generic[seed] - informed[seed]
                                        for seed in SEEDS
                                    }
                                ),
                            }
                        )
                    pg = values_for(
                        "product_generic_affine", n, ancillas, scope, metric
                    )
                    sg = values_for(
                        "spin_flip_generic_affine", n, ancillas, scope, metric
                    )
                    pi = values_for(
                        "product_informed_piecewise", n, ancillas, scope, metric
                    )
                    si = values_for(
                        "spin_flip_informed_piecewise", n, ancillas, scope, metric
                    )
                    contrasts.append(
                        {
                            "kind": "symmetry_by_map_interaction",
                            "visible_qubits": n,
                            "ancillas": ancillas,
                            "scope": scope,
                            "metric": metric,
                            "contrast": (
                                "(product_generic-spin_generic)-"
                                "(product_informed-spin_informed)"
                            ),
                            "interval": interval(
                                {
                                    seed: (pg[seed] - sg[seed]) - (pi[seed] - si[seed])
                                    for seed in SEEDS
                                }
                            ),
                        }
                    )
    return {
        "schema_version": 1,
        "campaign": CAMPAIGN_VERSION,
        "manifest_digest": manifest.digest,
        "replication_unit": "optimizer_seed",
        "cells": cells,
        "contrasts": contrasts,
        "interpretation_note": (
            "beta=1 is landmark-aware realization; generalization claims must use "
            "non_landmark_held_out separately"
        ),
    }


def _select_runs(
    manifest: AblationManifest,
    run_id: str | None,
    index: int | None,
    all_runs: bool,
) -> tuple[AblationIdentity, ...]:
    planned = plan_runs(manifest)
    if sum((run_id is not None, index is not None, all_runs)) != 1:
        raise ValueError("select exactly one of --run-id, --index, or --all")
    if all_runs:
        return planned
    if index is not None:
        if not 0 <= index < len(planned):
            raise ValueError("run index is outside the plan")
        return (planned[index],)
    selected = tuple(run for run in planned if run.run_id == run_id)
    if len(selected) != 1:
        raise ValueError("run ID is outside the plan")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "encoder-only",
            "smoke",
            "train",
            "evaluate",
            "status",
            "summary",
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--index", type=int)
    parser.add_argument("--all", action="store_true", dest="all_runs")
    parser.add_argument("--smoke-steps", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    manifest = AblationManifest.from_json(args.manifest)
    output = manifest.output_root if args.output_root is None else args.output_root
    if args.command == "plan":
        payload: Any = {
            "campaign": CAMPAIGN_VERSION,
            "manifest_digest": manifest.digest,
            "count": len(plan_runs(manifest)),
            "runs": [
                run.to_dict() | {"run_id": run.run_id} for run in plan_runs(manifest)
            ],
        }
    elif args.command == "encoder-only":
        payload = run_encoder_only(manifest, output_root=output)
    elif args.command == "smoke":
        payload = run_smoke(manifest, output, steps=args.smoke_steps)
    elif args.command == "status":
        payload = status(manifest, output)
    elif args.command == "summary":
        payload = summarize(manifest, output_root=output)
    else:
        selected = _select_runs(manifest, args.run_id, args.index, args.all_runs)
        function = train_run if args.command == "train" else evaluate_run
        payload = [function(manifest, run, output_root=output) for run in selected]
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if args.json_output is None:
        print(text, end="")
    else:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
