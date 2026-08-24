"""Evidence-derived scientific decisions for the frozen Ising campaign.

The orchestration layer deliberately knows how to make a gate monotonic, but
not how to decide whether a scientific gate passed.  This module is the sealed
acceptance seam: it loads complete declared outputs, verifies their provenance
and cardinality, derives the preregistered decision, writes canonical evidence,
and only then records the terminal gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    CompletionStatus,
    DuplicateArtifactError,
    ResultBundle,
    read_immutable_json,
    sha256_file,
    write_immutable_json,
)
from .campaign import (
    CampaignStore,
    Gate,
    GateDecision,
    GateEvidenceBinding,
    GateRecord,
    TuningDecision,
    planned_runs,
)
from .campaign_statistics import (
    CONDITION_MODULATED_PHASE,
    FIXED_INPUT,
    NO_CONDITION,
    ORACLE_ADJACENT,
    AncillaEvaluationRecord,
    AncillaSuccessResult,
    BenefitRetentionGateResult,
    InterpolationGateResult,
    PairedEvaluationRecord,
    ancilla_descriptive_success_rule,
    benefit_retention_gate,
    interpolation_gate,
)
from .ising import CalibrationThresholds
from .protocol import (
    SHARED_PRIMARY_FAMILIES,
    CampaignManifest,
    ModelFamily,
    RunIdentity,
    Stage,
    canonical_json_bytes,
    content_digest,
)


class AcceptanceEvidenceError(ValueError):
    """Raised when archived evidence is incomplete, corrupt, or out of protocol."""


@dataclass(frozen=True, slots=True)
class GateFinalization:
    """A terminal gate record and its byte-for-byte archived evidence binding."""

    record: GateRecord
    binding: GateEvidenceBinding
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DerivedFinalization:
    """A sealed descriptive result that is not itself a run-release gate."""

    path: Path
    payload_digest: str
    file_sha256: str
    passed: bool
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PrimaryFinalization:
    """The two independently binding decisions derived from one primary matrix."""

    benefit_retention: GateFinalization
    interpolation: GateFinalization
    paired_records: tuple[PairedEvaluationRecord, ...]


ExactTVDRows = tuple[tuple[str, float, float], ...]
PrimaryBundleValue = tuple[ResultBundle, ExactTVDRows]


def _is_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _number(value: object, context: str) -> float:
    if not _is_number(value):
        raise AcceptanceEvidenceError(f"{context} must be finite and numeric")
    return float(value)


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AcceptanceEvidenceError(f"{context} must be an integer")
    return value


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AcceptanceEvidenceError(f"{context} must be a string-keyed object")
    return value


def _sequence(value: object, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AcceptanceEvidenceError(f"{context} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise AcceptanceEvidenceError(
            f"{context} schema mismatch; missing={missing}, extra={extra}",
        )


def _same_float(left: object, right: float, context: str) -> None:
    value = _number(left, context)
    if not math.isclose(value, right, rel_tol=0.0, abs_tol=1e-14):
        raise AcceptanceEvidenceError(f"{context} does not match the manifest")


def _relative_file(repository: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AcceptanceEvidenceError(f"{context} must be a non-empty path")
    relative = Path(value)
    if ".." in relative.parts:
        raise AcceptanceEvidenceError(f"{context} must not contain '..'")
    resolved = (
        relative.resolve()
        if relative.is_absolute()
        else (repository / relative).resolve()
    )
    try:
        resolved.relative_to(repository.resolve())
    except ValueError as error:
        raise AcceptanceEvidenceError(
            f"{context} resolves outside the repository",
        ) from error
    if not resolved.is_file():
        raise AcceptanceEvidenceError(f"{context} does not exist: {resolved}")
    return resolved


def _json_file(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceEvidenceError(f"could not read {context}: {path}") from error
    return _mapping(value, context)


def _stage_bundles(
    store: CampaignStore,
    stage: Stage,
) -> tuple[tuple[RunIdentity, ResultBundle], ...]:
    """Load exactly the declared terminal bundle set for one stage."""
    expected = planned_runs(
        store.manifest,
        stage,
        tuning_decision=store.tuning_decision(),
    )
    expected_by_id = {run.run_id: run for run in expected}
    if len(expected_by_id) != len(expected):
        raise AcceptanceEvidenceError(f"{stage.value} plan contains duplicate runs")

    actual: dict[str, ResultBundle] = {}
    for path in sorted(store.artifacts.runs_directory.glob("*.json")):
        bundle = store.result(path.stem)
        if bundle is None or bundle.run.stage is not stage:
            continue
        actual[bundle.run.run_id] = bundle
    missing = sorted(set(expected_by_id) - set(actual))
    extra = sorted(set(actual) - set(expected_by_id))
    if missing or extra:
        raise AcceptanceEvidenceError(
            f"{stage.value} result cardinality is incomplete; "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing[:3]}, extra={extra[:3]}",
        )

    result = []
    for run in expected:
        bundle = actual[run.run_id]
        if bundle.run != run or bundle.manifest_digest != store.manifest.digest:
            raise AcceptanceEvidenceError(
                f"{stage.value} bundle does not match its declared identity",
            )
        reservation = store.reservation(run.run_id)
        if reservation is None or reservation.run != run:
            raise AcceptanceEvidenceError(
                f"{stage.value} bundle lacks its exact ledger reservation",
            )
        result.append((run, bundle))
    return tuple(result)


def _canonical_evidence_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(payload) + b"\n"


def _seal_gate(
    store: CampaignStore,
    gate: Gate,
    decision: GateDecision,
    evidence: Mapping[str, Any],
    *,
    reason: str,
    decided_at: str | None,
    candidate_ids: Mapping[ModelFamily, str] | None = None,
) -> GateFinalization:
    """Seal canonical evidence, record its digest, then archive exact bytes."""
    encoded = _canonical_evidence_bytes(evidence)
    digest = hashlib.sha256(encoded).hexdigest()
    tuning_decision = None
    if candidate_ids is not None:
        tuning_decision = TuningDecision.create(
            store.manifest,
            candidate_ids,
            evidence_digest=digest,
        )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=store.root,
        prefix=f".{gate.value}-evidence-",
        suffix=".json",
        delete=False,
    ) as stream:
        source = Path(stream.name)
        stream.write(encoded)
        stream.flush()
    try:
        record = store._release_finalized_gate(
            gate,
            decision,
            evidence_digest=digest,
            reason=reason,
            decided_at=decided_at,
            tuning_decision=tuning_decision,
        )
        binding = store.archive_gate_evidence(gate, source)
    finally:
        source.unlink(missing_ok=True)
    return GateFinalization(record=record, binding=binding, evidence=dict(evidence))


def record_failed_gate_with_evidence(
    store: CampaignStore,
    gate: Gate,
    evidence_path: Path,
    *,
    reason: str,
    decided_at: str | None = None,
) -> GateFinalization:
    """Record an externally explained stop while preserving its exact evidence."""
    evidence_bytes = evidence_path.read_bytes()
    digest = hashlib.sha256(evidence_bytes).hexdigest()
    record = store.release_gate(
        gate,
        GateDecision.FAILED,
        evidence_digest=digest,
        reason=reason,
        decided_at=decided_at,
    )
    binding = store.archive_gate_evidence(gate, evidence_path)
    return GateFinalization(
        record=record,
        binding=binding,
        evidence={"source_path": str(evidence_path), "source_sha256": digest},
    )


def _write_derived(
    store: CampaignStore,
    name: str,
    evidence: Mapping[str, Any],
    *,
    passed: bool,
) -> DerivedFinalization:
    path = store.root / "derived" / f"{name}.json"
    if path.exists():
        existing = read_immutable_json(path)
        if existing != dict(evidence):
            raise DuplicateArtifactError(
                f"derived result already exists with different evidence: {path}",
            )
        digest = content_digest(existing)
    else:
        digest = write_immutable_json(path, evidence)
    return DerivedFinalization(
        path=path,
        payload_digest=digest,
        file_sha256=sha256_file(path),
        passed=passed,
        evidence=dict(evidence),
    )


def _target_archive_record(
    record: object,
    repository: Path,
    manifest: CampaignManifest,
) -> dict[str, Any]:
    value = _mapping(record, "target archive record")
    _exact_keys(
        value,
        {"coupling_seed", "archive", "archive_sha256", "target_digests"},
        "target archive record",
    )
    seed = _integer(value["coupling_seed"], "target coupling_seed")
    if seed not in manifest.coupling_seeds:
        raise AcceptanceEvidenceError("target archive uses an undeclared instance")
    archive = _relative_file(repository, value["archive"], "target archive")
    actual_sha = sha256_file(archive)
    if value["archive_sha256"] != actual_sha:
        raise AcceptanceEvidenceError("target archive SHA-256 does not match")
    digests = tuple(_sequence(value["target_digests"], "target digests"))
    if len(digests) != 9 or any(
        not isinstance(item, str)
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in digests
    ):
        raise AcceptanceEvidenceError("target archive requires nine SHA-256 digests")
    try:
        with np.load(archive, allow_pickle=False) as arrays:
            required = {
                "coupling_seed",
                "betas",
                "couplings",
                "bitstrings",
                "energies",
                "probabilities",
            }
            if set(arrays.files) != required:
                raise AcceptanceEvidenceError("target NPZ schema is not exact")
            if int(arrays["coupling_seed"]) != seed:
                raise AcceptanceEvidenceError("target NPZ seed does not match")
            betas = np.asarray(arrays["betas"], dtype=np.float64)
            expected_betas = np.asarray(
                manifest.beta_grid.training + manifest.beta_grid.held_out,
            )
            if betas.shape != (9,) or not np.allclose(
                betas,
                expected_betas,
                atol=1e-14,
                rtol=0.0,
            ):
                raise AcceptanceEvidenceError("target NPZ beta grid does not match")
            if arrays["probabilities"].shape != (9, 2**16):
                raise AcceptanceEvidenceError(
                    "target NPZ must contain nine exactly enumerated 16-bit laws",
                )
            if arrays["bitstrings"].shape != (2**16, 16):
                raise AcceptanceEvidenceError("target NPZ enumeration shape is wrong")
    except (OSError, ValueError) as error:
        if isinstance(error, AcceptanceEvidenceError):
            raise
        raise AcceptanceEvidenceError(
            f"could not verify target NPZ {archive}",
        ) from error
    return {
        "coupling_seed": seed,
        "archive": str(archive.relative_to(repository.resolve())),
        "archive_sha256": actual_sha,
        "target_digests": list(digests),
    }


def finalize_target_calibration(
    store: CampaignStore,
    repository: Path,
    *,
    decided_at: str | None = None,
) -> GateFinalization:
    """Verify the exact Target-only grid and seal the calibration gate."""
    manifest = store.manifest
    source = _relative_file(
        repository,
        manifest.calibration.artifact_path,
        "manifest calibration artifact",
    )
    source_sha = sha256_file(source)
    if manifest.calibration.artifact_sha256 != source_sha:
        raise AcceptanceEvidenceError(
            "calibration artifact differs from the manifest-bound SHA-256",
        )
    payload = _json_file(source, "calibration artifact")
    _exact_keys(
        payload,
        {
            "schema_version",
            "campaign_version",
            "source_provenance",
            "selection_rule",
            "thresholds",
            "coupling_seeds",
            "selected_interval",
            "training_betas",
            "heldout_betas",
            "candidate_records",
            "target_archives",
        },
        "calibration artifact",
    )
    if payload["schema_version"] != 1 or payload["campaign_version"] != (
        manifest.campaign_version
    ):
        raise AcceptanceEvidenceError("calibration campaign identity does not match")
    if tuple(payload["coupling_seeds"]) != manifest.coupling_seeds:
        raise AcceptanceEvidenceError("calibration coupling instances do not match")
    training = tuple(float(item) for item in payload["training_betas"])
    heldout = tuple(float(item) for item in payload["heldout_betas"])
    if (
        training != manifest.beta_grid.training
        or heldout != manifest.beta_grid.held_out
    ):
        raise AcceptanceEvidenceError("calibration beta split does not match manifest")
    selected = tuple(float(item) for item in payload["selected_interval"])
    if selected != (training[0], training[-1]):
        raise AcceptanceEvidenceError("calibration selected interval is inconsistent")

    threshold_values = _mapping(payload["thresholds"], "calibration thresholds")
    expected_thresholds = CalibrationThresholds().__dict__
    if threshold_values != expected_thresholds:
        raise AcceptanceEvidenceError("calibration thresholds are not frozen values")
    candidate_rows = _sequence(payload["candidate_records"], "candidate records")
    eligible: list[tuple[float, float, float, tuple[float, float]]] = []
    verified_candidates = []
    candidate_schema = {
        "interval",
        "training_betas",
        "heldout_betas",
        "min_adjacent_tvd",
        "median_adjacent_tvd",
        "max_adjacent_tvd",
        "min_normalized_entropy",
        "max_probability",
        "passed",
    }
    for index, raw_row in enumerate(candidate_rows):
        row = _mapping(raw_row, f"calibration candidate {index}")
        _exact_keys(row, candidate_schema, f"calibration candidate {index}")
        interval = tuple(float(item) for item in row["interval"])
        if len(interval) != 2 or not interval[0] < interval[1]:
            raise AcceptanceEvidenceError("candidate interval is invalid")
        metrics = {
            name: _number(row[name], f"candidate {index} {name}")
            for name in (
                "min_adjacent_tvd",
                "median_adjacent_tvd",
                "max_adjacent_tvd",
                "min_normalized_entropy",
                "max_probability",
            )
        }
        passed = (
            metrics["min_adjacent_tvd"] >= expected_thresholds["min_adjacent_tvd"]
            and metrics["max_adjacent_tvd"] <= expected_thresholds["max_adjacent_tvd"]
            and metrics["min_normalized_entropy"]
            >= expected_thresholds["min_normalized_entropy"]
            and metrics["max_probability"]
            <= expected_thresholds["max_endpoint_probability"]
        )
        if row["passed"] is not passed:
            raise AcceptanceEvidenceError("candidate pass flag disagrees with metrics")
        if passed:
            eligible.append(
                (
                    interval[1] - interval[0],
                    metrics["min_adjacent_tvd"],
                    -interval[0],
                    interval,
                ),
            )
        verified_candidates.append(dict(row))
    if not eligible or max(eligible, key=lambda item: item[:3])[3] != selected:
        raise AcceptanceEvidenceError(
            "selected interval does not follow the frozen target-only rule",
        )

    archives = tuple(
        _target_archive_record(record, repository, manifest)
        for record in _sequence(payload["target_archives"], "target archives")
    )
    if len(archives) != len(manifest.coupling_seeds) or {
        row["coupling_seed"] for row in archives
    } != set(manifest.coupling_seeds):
        raise AcceptanceEvidenceError("calibration does not archive all eight targets")

    evidence = {
        "schema_version": 1,
        "finalizer": "target_calibration_v1",
        "manifest_digest": manifest.digest,
        "source_path": manifest.calibration.artifact_path,
        "source_sha256": source_sha,
        "model_evidence_fields": [],
        "selection_rule": payload["selection_rule"],
        "selected_interval": list(selected),
        "training_betas": list(training),
        "heldout_betas": list(heldout),
        "candidate_records": verified_candidates,
        "target_archives": list(archives),
        "decision": GateDecision.PASSED.value,
    }
    return _seal_gate(
        store,
        Gate.TARGET_CALIBRATION,
        GateDecision.PASSED,
        evidence,
        reason="exact Target-only calibration matches the frozen five/four beta grid",
        decided_at=decided_at,
    )


_AUDIT_KEYS = {
    "coupling_seed",
    "beta",
    "n_chains",
    "samples_per_chain",
    "min_energy_ess",
    "max_chain_energy_z",
    "energy_mean_z",
    "magnetization_error",
    "correlation_rmse",
    "character_rmse",
    "empirical_bitstring_tvd",
    "passed",
    "failed_checks",
}


def _verify_sampler_audit(
    raw: object,
    *,
    seed: int,
    beta: float,
    samples_per_chain: int,
    manifest: CampaignManifest,
    context: str,
) -> tuple[bool, dict[str, Any]]:
    value = _mapping(raw, context)
    _exact_keys(value, _AUDIT_KEYS, context)
    if _integer(value["coupling_seed"], f"{context} coupling_seed") != seed:
        raise AcceptanceEvidenceError(f"{context} instance does not match its cell")
    _same_float(value["beta"], beta, f"{context} beta")
    if _integer(value["n_chains"], f"{context} n_chains") != (
        manifest.sampler_protocol.n_chains
    ):
        raise AcceptanceEvidenceError(f"{context} chain count is not frozen")
    if _integer(value["samples_per_chain"], f"{context} samples_per_chain") != (
        samples_per_chain
    ):
        raise AcceptanceEvidenceError(f"{context} per-chain count is wrong")

    sampler = manifest.sampler_protocol
    checks = {
        "effective_sample_size": _number(
            value["min_energy_ess"],
            f"{context} min_energy_ess",
        )
        >= sampler.min_ess_per_chain,
        "chain_agreement": _number(
            value["max_chain_energy_z"],
            f"{context} max_chain_energy_z",
        )
        <= sampler.max_chain_energy_z,
        "energy_moment": _number(
            value["energy_mean_z"],
            f"{context} energy_mean_z",
        )
        <= sampler.max_energy_mean_z,
        "magnetization": _number(
            value["magnetization_error"],
            f"{context} magnetization_error",
        )
        <= sampler.max_magnetization_error,
        "correlations": _number(
            value["correlation_rmse"],
            f"{context} correlation_rmse",
        )
        <= sampler.max_correlation_rmse,
        "distributional_characters": _number(
            value["character_rmse"],
            f"{context} character_rmse",
        )
        <= sampler.max_character_rmse,
    }
    empirical_tvd = _number(
        value["empirical_bitstring_tvd"],
        f"{context} empirical_bitstring_tvd",
    )
    if not 0.0 <= empirical_tvd <= 1.0:
        raise AcceptanceEvidenceError(f"{context} empirical TVD is out of range")
    failed = tuple(name for name, passed in checks.items() if not passed)
    if tuple(value["failed_checks"]) != failed or value["passed"] is not (not failed):
        raise AcceptanceEvidenceError(
            f"{context} pass fields disagree with frozen thresholds",
        )
    return not failed, dict(value)


def _verify_dataset_archive(
    value: Mapping[str, Any],
    repository: Path,
    *,
    seed: int,
    beta: float,
    manifest: CampaignManifest,
) -> dict[str, str]:
    archive = _relative_file(repository, value["archive"], "sampler dataset archive")
    archive_sha = sha256_file(archive)
    if value["archive_sha256"] != archive_sha:
        raise AcceptanceEvidenceError("sampler dataset archive SHA-256 differs")
    try:
        with np.load(archive, allow_pickle=False) as arrays:
            expected_keys = {
                "training",
                "validation",
                "training_seed",
                "validation_seed",
                "beta",
            }
            if set(arrays.files) != expected_keys:
                raise AcceptanceEvidenceError("sampler dataset NPZ schema is not exact")
            training = np.asarray(arrays["training"])
            validation = np.asarray(arrays["validation"])
            if training.shape != (
                manifest.sampler_protocol.training_samples_per_condition,
                16,
            ) or validation.shape != (
                manifest.sampler_protocol.validation_samples_per_condition,
                16,
            ):
                raise AcceptanceEvidenceError("sampler dataset counts or shape differ")
            if not np.all((training == 0) | (training == 1)) or not np.all(
                (validation == 0) | (validation == 1),
            ):
                raise AcceptanceEvidenceError("sampler datasets must be binary")
            if (
                int(arrays["training_seed"]) != value["training_seed"]
                or int(
                    arrays["validation_seed"],
                )
                != value["validation_seed"]
            ):
                raise AcceptanceEvidenceError("sampler stream seeds differ from NPZ")
            _same_float(arrays["beta"].item(), beta, "sampler NPZ beta")
            training_sha = hashlib.sha256(training.tobytes()).hexdigest()
            validation_sha = hashlib.sha256(validation.tobytes()).hexdigest()
    except (OSError, ValueError) as error:
        if isinstance(error, AcceptanceEvidenceError):
            raise
        raise AcceptanceEvidenceError(
            f"could not verify sampler dataset archive for {seed}/{beta}",
        ) from error
    if (
        value["training_sha256"] != training_sha
        or value["validation_sha256"] != validation_sha
    ):
        raise AcceptanceEvidenceError("sampler split SHA-256 differs from NPZ arrays")
    return {
        "archive": str(archive.relative_to(repository.resolve())),
        "archive_sha256": archive_sha,
        "training_sha256": training_sha,
        "validation_sha256": validation_sha,
    }


def finalize_sampling_quality(
    store: CampaignStore,
    summary_path: Path,
    repository: Path,
    *,
    decided_at: str | None = None,
) -> GateFinalization:
    """Verify every independent-chain cell and apply all frozen sampler checks."""
    manifest = store.manifest
    try:
        summary_relative = summary_path.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise AcceptanceEvidenceError(
            "sampler summary must be inside the declared repository root",
        ) from error
    payload = _json_file(summary_path, "sampler audit summary")
    _exact_keys(
        payload,
        {
            "schema_version",
            "campaign_version",
            "manifest_digest",
            "sampler_protocol",
            "configuration_digest",
            "expected_cells",
            "records",
            "passed",
        },
        "sampler audit summary",
    )
    if payload["schema_version"] != 1 or payload["campaign_version"] != (
        manifest.campaign_version
    ):
        raise AcceptanceEvidenceError("sampler summary campaign identity differs")
    if payload["manifest_digest"] != manifest.digest:
        raise AcceptanceEvidenceError("sampler summary manifest digest differs")
    if payload["sampler_protocol"] != manifest.sampler_protocol.to_dict():
        raise AcceptanceEvidenceError("sampler summary protocol differs")
    sampler = manifest.sampler_protocol
    schedule = {
        "n_chains": sampler.n_chains,
        "burn_in": sampler.burn_in_sweeps,
        "thinning": sampler.thinning_sweeps,
        "symmetry_mix": True,
    }
    if payload["configuration_digest"] != content_digest(schedule):
        raise AcceptanceEvidenceError("sampler summary schedule digest differs")
    expected_cells = len(manifest.coupling_seeds) * len(manifest.beta_grid.training)
    if payload["expected_cells"] != expected_cells:
        raise AcceptanceEvidenceError("sampler expected-cell count is not frozen")
    records = _sequence(payload["records"], "sampler records")
    if len(records) != expected_cells:
        raise AcceptanceEvidenceError(
            f"sampling gate requires all {expected_cells} instance/beta cells",
        )

    expected_identities = {
        (seed, beta)
        for seed in manifest.coupling_seeds
        for beta in manifest.beta_grid.training
    }
    actual_identities: set[tuple[int, float]] = set()
    verified_records = []
    cell_passes = []
    record_schema = {
        "coupling_seed",
        "beta",
        "training_seed",
        "validation_seed",
        "n_chains",
        "burn_in_sweeps",
        "thinning_sweeps",
        "single_spin_proposals_per_sweep",
        "global_spin_flip_on_retained_samples",
        "training_count",
        "validation_count",
        "training_sha256",
        "validation_sha256",
        "archive",
        "archive_sha256",
        "training_audit",
        "validation_audit",
        "passed",
    }
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"sampler record {index}")
        _exact_keys(record, record_schema, f"sampler record {index}")
        seed = _integer(record["coupling_seed"], "sampler coupling_seed")
        beta = _number(record["beta"], "sampler beta")
        matching = [
            identity
            for identity in expected_identities
            if identity[0] == seed
            and math.isclose(identity[1], beta, rel_tol=0.0, abs_tol=1e-14)
        ]
        if len(matching) != 1 or matching[0] in actual_identities:
            raise AcceptanceEvidenceError(
                "sampler cells must be the unique declared instance/beta product",
            )
        identity = matching[0]
        actual_identities.add(identity)
        if (
            record["n_chains"] != sampler.n_chains
            or record["burn_in_sweeps"] != sampler.burn_in_sweeps
            or record["thinning_sweeps"] != sampler.thinning_sweeps
            or record["single_spin_proposals_per_sweep"] != 16
            or record["global_spin_flip_on_retained_samples"] is not True
            or record["training_count"] != sampler.training_samples_per_condition
            or record["validation_count"] != sampler.validation_samples_per_condition
        ):
            raise AcceptanceEvidenceError("sampler record does not use frozen schedule")
        training_seed = _integer(record["training_seed"], "training seed")
        validation_seed = _integer(record["validation_seed"], "validation seed")
        if training_seed == validation_seed:
            raise AcceptanceEvidenceError(
                "Training and Validation sampler streams must be independent",
            )
        archive = _verify_dataset_archive(
            record,
            repository,
            seed=seed,
            beta=identity[1],
            manifest=manifest,
        )
        train_passed, training_audit = _verify_sampler_audit(
            record["training_audit"],
            seed=seed,
            beta=identity[1],
            samples_per_chain=(
                sampler.training_samples_per_condition // sampler.n_chains
            ),
            manifest=manifest,
            context=f"sampler record {index} Training audit",
        )
        validation_passed, validation_audit = _verify_sampler_audit(
            record["validation_audit"],
            seed=seed,
            beta=identity[1],
            samples_per_chain=(
                sampler.validation_samples_per_condition // sampler.n_chains
            ),
            manifest=manifest,
            context=f"sampler record {index} Validation audit",
        )
        passed = train_passed and validation_passed
        if record["passed"] is not passed:
            raise AcceptanceEvidenceError("sampler cell pass flag is inconsistent")
        cell_passes.append(passed)
        verified_records.append(
            {
                "coupling_seed": seed,
                "beta": identity[1],
                "training_seed": training_seed,
                "validation_seed": validation_seed,
                **archive,
                "training_audit": training_audit,
                "validation_audit": validation_audit,
                "passed": passed,
            },
        )
    if actual_identities != expected_identities:
        raise AcceptanceEvidenceError("sampler summary is missing declared cells")
    all_passed = all(cell_passes)
    if payload["passed"] is not all_passed:
        raise AcceptanceEvidenceError("sampler summary pass flag is inconsistent")

    decision = GateDecision.PASSED if all_passed else GateDecision.FAILED
    evidence = {
        "schema_version": 1,
        "finalizer": "sampling_quality_v1",
        "manifest_digest": manifest.digest,
        "source_path": str(summary_relative),
        "source_sha256": sha256_file(summary_path),
        "schedule": schedule,
        "thresholds": manifest.sampler_protocol.to_dict(),
        "expected_cells": expected_cells,
        "records": sorted(
            verified_records,
            key=lambda row: (row["coupling_seed"], row["beta"]),
        ),
        "decision": decision.value,
    }
    reason = (
        "all 40 Training/Validation chain collections passed every frozen check"
        if all_passed
        else "at least one complete sampler cell failed a frozen quality check"
    )
    return _seal_gate(
        store,
        Gate.SAMPLING_QUALITY,
        decision,
        evidence,
        reason=reason,
        decided_at=decided_at,
    )


def _bundle_rows(
    bundles: Sequence[tuple[RunIdentity, ResultBundle]],
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run.run_id,
            "bundle_digest": bundle.digest,
            "completion_status": bundle.completion_status.value,
            "error": bundle.error,
        }
        for run, bundle in bundles
    ]


def _assert_sampled_training_protocol(
    bundle: ResultBundle,
    manifest: CampaignManifest,
    *,
    context: str,
) -> Mapping[str, Any]:
    metrics = bundle.raw_metrics
    expected = manifest.estimator_budget
    if (
        metrics.get("objective") != "sampled_conditional_mmd"
        or tuple(metrics.get("bandwidths", ())) != expected.bandwidths
        or metrics.get("operators_per_bandwidth") != expected.operators_per_bandwidth
        or metrics.get("reference_samples_per_operator_condition")
        != expected.reference_samples_per_operator_condition
        or metrics.get("max_updates") != manifest.training_protocol.max_updates
        or metrics.get("batch_size_per_condition")
        != manifest.training_protocol.batch_size
    ):
        raise AcceptanceEvidenceError(
            f"{context} does not use the frozen estimator/training budget",
        )
    if metrics.get("model_family") != bundle.run.model_family.value:
        raise AcceptanceEvidenceError(f"{context} model-family provenance differs")
    if metrics.get("tuning_candidate_id") != bundle.run.tuning_candidate_id:
        raise AcceptanceEvidenceError(f"{context} tuning-candidate provenance differs")
    if metrics.get("phase_feature") != bundle.run.phase_feature_assignment:
        raise AcceptanceEvidenceError(f"{context} phase-feature provenance differs")
    return metrics


def _has_heldout_evidence(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            "heldout" in key.lower().replace("_", "") or _has_heldout_evidence(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_heldout_evidence(item) for item in value)
    return isinstance(value, str) and "heldout" in value.lower().replace("_", "")


def _assert_shared_training_hash_schema(
    bundle: ResultBundle,
    manifest: CampaignManifest,
    *,
    context: str,
) -> None:
    names = {name for name, _ in bundle.dataset_hashes}
    required = {
        f"{role}_beta_{index}"
        for role in ("training", "validation")
        for index in range(len(manifest.beta_grid.training))
    }
    allowed = required | {
        f"{role}_source_beta_{index}"
        for role in ("training", "validation")
        for index in range(len(manifest.beta_grid.training))
    }
    if not required.issubset(names) or not names.issubset(allowed):
        raise AcceptanceEvidenceError(
            f"{context} dataset-hash schema is incomplete or contains forbidden roles",
        )


def finalize_runtime_feasibility(
    store: CampaignStore,
    *,
    decided_at: str | None = None,
) -> GateFinalization:
    """Accept the fixed estimator budget only after its sole declared run ends."""
    bundles = _stage_bundles(store, Stage.RUNTIME_FEASIBILITY)
    if len(bundles) != 1:
        raise AcceptanceEvidenceError("runtime feasibility requires exactly one run")
    _, bundle = bundles[0]
    passed = bundle.completion_status is CompletionStatus.COMPLETED
    telemetry: dict[str, Any] | None = None
    if passed:
        metrics = _assert_sampled_training_protocol(
            bundle,
            store.manifest,
            context="runtime feasibility result",
        )
        if metrics.get("exact_tvd") not in (None, []):
            raise AcceptanceEvidenceError(
                "runtime feasibility cannot access exact or Held-out Targets",
            )
        _assert_shared_training_hash_schema(
            bundle,
            store.manifest,
            context="runtime feasibility",
        )
        if _has_heldout_evidence(metrics) or any(
            "heldout" in name.lower().replace("_", "") or "exact_target" in name.lower()
            for name, _ in bundle.dataset_hashes
        ):
            raise AcceptanceEvidenceError(
                "runtime feasibility contains forbidden Held-out/exact evidence",
            )
        telemetry = {
            "runtime_seconds": float(bundle.runtime_seconds),
            "peak_memory_bytes": bundle.peak_memory_bytes,
            "operators_per_bandwidth": metrics["operators_per_bandwidth"],
            "reference_samples_per_operator_condition": metrics[
                "reference_samples_per_operator_condition"
            ],
            "bandwidths": metrics["bandwidths"],
            "max_updates": metrics["max_updates"],
        }
    decision = GateDecision.PASSED if passed else GateDecision.FAILED
    evidence = {
        "schema_version": 1,
        "finalizer": "runtime_feasibility_v1",
        "manifest_digest": store.manifest.digest,
        "declared_run_count": 1,
        "run": _bundle_rows(bundles)[0],
        "telemetry": telemetry,
        "decision": decision.value,
    }
    return _seal_gate(
        store,
        Gate.RUNTIME_FEASIBILITY,
        decision,
        evidence,
        reason=(
            "the sole run completed with the frozen 2,000-by-2,000 estimator budget"
            if passed
            else "the sole frozen-budget feasibility run failed terminally"
        ),
        decided_at=decided_at,
    )


def _exact_tvd_rows(bundle: ResultBundle) -> tuple[tuple[str, float, float], ...]:
    rows = _sequence(bundle.raw_metrics.get("exact_tvd"), "exact_tvd rows")
    values: list[tuple[str, float, float]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"exact_tvd row {index}")
        _exact_keys(row, {"role", "beta", "tvd"}, f"exact_tvd row {index}")
        role = row["role"]
        if role not in {"training", "held_out"}:
            raise AcceptanceEvidenceError("exact_tvd role is not declared")
        beta = _number(row["beta"], "exact_tvd beta")
        tvd = _number(row["tvd"], "exact_tvd value")
        if not 0.0 <= tvd <= 1.0:
            raise AcceptanceEvidenceError("exact_tvd must lie in [0, 1]")
        values.append((str(role), beta, tvd))
    identities = [(role, beta) for role, beta, _ in values]
    if len(identities) != len(set(identities)):
        raise AcceptanceEvidenceError("exact_tvd rows contain duplicate cells")
    return tuple(values)


def _tvd_at(
    rows: Sequence[tuple[str, float, float]],
    role: str,
    beta: float,
) -> float:
    matches = [
        tvd
        for row_role, row_beta, tvd in rows
        if row_role == role and math.isclose(row_beta, beta, rel_tol=0.0, abs_tol=1e-14)
    ]
    if len(matches) != 1:
        raise AcceptanceEvidenceError(
            f"exact_tvd requires one {role} row at beta={beta:.17g}",
        )
    return matches[0]


def finalize_representability(
    store: CampaignStore,
    *,
    decided_at: str | None = None,
) -> GateFinalization:
    """Apply the exact six-screen median and every-cell TVD thresholds."""
    bundles = _stage_bundles(store, Stage.REPRESENTABILITY_SCREEN)
    if len(bundles) != 6:
        raise AcceptanceEvidenceError(
            "representability finalization requires all six declared screens",
        )
    failed_runs = [
        run.run_id
        for run, bundle in bundles
        if bundle.completion_status is not CompletionStatus.COMPLETED
    ]
    cells = []
    median_tvd = None
    max_tvd = None
    if not failed_runs:
        for run, bundle in bundles:
            metrics = _assert_sampled_training_protocol(
                bundle,
                store.manifest,
                context=f"representability run {run.run_id}",
            )
            if run.model_family is not ModelFamily.SEPARATE:
                raise AcceptanceEvidenceError(
                    "representability screen must use Separate models",
                )
            try:
                beta_index = int(run.condition_scope.removeprefix("train_beta_"))
                beta = store.manifest.beta_grid.training[beta_index]
            except (ValueError, IndexError) as error:
                raise AcceptanceEvidenceError(
                    "representability Condition scope is invalid",
                ) from error
            rows = _exact_tvd_rows(bundle)
            if len(rows) != 1 or rows[0][0] != "training":
                raise AcceptanceEvidenceError(
                    "representability requires one exact Training TVD",
                )
            tvd = _tvd_at(rows, "training", beta)
            if "exact_targets" not in dict(bundle.dataset_hashes):
                raise AcceptanceEvidenceError(
                    "representability bundle lacks exact-Target provenance",
                )
            cells.append(
                {
                    "coupling_seed": run.coupling_seed,
                    "beta_index": beta_index,
                    "beta": beta,
                    "tvd": tvd,
                    "run_id": run.run_id,
                    "bundle_digest": bundle.digest,
                    "objective": metrics["objective"],
                },
            )
        values = np.asarray([cell["tvd"] for cell in cells], dtype=np.float64)
        median_tvd = float(np.median(values))
        max_tvd = float(np.max(values))
        thresholds = store.manifest.statistics
        passed = (
            median_tvd <= thresholds.representability_median_tvd_max
            and max_tvd <= thresholds.representability_cell_tvd_max
        )
    else:
        passed = False
    decision = GateDecision.PASSED if passed else GateDecision.FAILED
    evidence = {
        "schema_version": 1,
        "finalizer": "representability_v1",
        "manifest_digest": store.manifest.digest,
        "declared_run_count": 6,
        "terminal_bundle_count": len(bundles),
        "failed_run_ids": failed_runs,
        "thresholds": {
            "median_tvd_max": (
                store.manifest.statistics.representability_median_tvd_max
            ),
            "every_cell_tvd_max": (
                store.manifest.statistics.representability_cell_tvd_max
            ),
        },
        "cells": sorted(
            cells,
            key=lambda row: (row["coupling_seed"], row["beta_index"]),
        ),
        "median_tvd": median_tvd,
        "max_tvd": max_tvd,
        "sole_diagnostic_eligible": decision is GateDecision.FAILED,
        "decision": decision.value,
    }
    reason = (
        "all six Separate screens satisfy median TVD <= 0.25 and cell TVD <= 0.35"
        if passed
        else "the complete six-screen representability evidence failed the frozen rule"
    )
    return _seal_gate(
        store,
        Gate.REPRESENTABILITY,
        decision,
        evidence,
        reason=reason,
        decided_at=decided_at,
    )


def finalize_representability_diagnostic(
    store: CampaignStore,
) -> DerivedFinalization:
    """Seal the sole exact-objective diagnostic after representability failure."""
    gate = store.gate_record(Gate.REPRESENTABILITY)
    if gate is None or gate.decision is not GateDecision.FAILED:
        raise AcceptanceEvidenceError(
            "the diagnostic exists only after a failed representability gate",
        )
    bundles = _stage_bundles(store, Stage.REPRESENTABILITY_DIAGNOSTIC)
    if len(bundles) != 1:
        raise AcceptanceEvidenceError("exactly one diagnostic run is allowed")
    run, bundle = bundles[0]
    if bundle.completion_status is not CompletionStatus.COMPLETED:
        raise AcceptanceEvidenceError("the sole diagnostic did not complete")
    metrics = bundle.raw_metrics
    if metrics.get("objective") != "exact_conditional_mmd_diagnostic":
        raise AcceptanceEvidenceError("diagnostic did not use the exact objective")
    rows = _exact_tvd_rows(bundle)
    if len(rows) != 1 or rows[0][0] != "training":
        raise AcceptanceEvidenceError("diagnostic requires one exact Training TVD")
    evidence = {
        "schema_version": 1,
        "finalizer": "representability_diagnostic_v1",
        "manifest_digest": store.manifest.digest,
        "representability_gate_digest": gate.digest,
        "run_id": run.run_id,
        "bundle_digest": bundle.digest,
        "objective": metrics["objective"],
        "exact_tvd": [
            {"role": role, "beta": beta, "tvd": tvd} for role, beta, tvd in rows
        ],
        "changes_gate_decision": False,
        "additional_diagnostics_allowed": 0,
    }
    return _write_derived(
        store,
        "representability_diagnostic",
        evidence,
        passed=False,
    )


def finalize_tuning(
    store: CampaignStore,
    *,
    decided_at: str | None = None,
) -> GateFinalization:
    """Select exactly one Training/Validation-only winner for each shared family."""
    bundles = _stage_bundles(store, Stage.TUNING)
    if len(bundles) != store.manifest.run_allocation.tuning or len(bundles) != 36:
        raise AcceptanceEvidenceError("tuning finalization requires all 36 runs")
    failed_runs = [
        run.run_id
        for run, bundle in bundles
        if bundle.completion_status is not CompletionStatus.COMPLETED
    ]
    run_rows = _bundle_rows(bundles)
    if failed_runs:
        evidence = {
            "schema_version": 1,
            "finalizer": "tuning_v1",
            "manifest_digest": store.manifest.digest,
            "declared_run_count": 36,
            "runs": run_rows,
            "failed_run_ids": failed_runs,
            "held_out_access_detected": False,
            "candidate_scores": [],
            "selections": [],
            "decision": GateDecision.FAILED.value,
        }
        return _seal_gate(
            store,
            Gate.TUNING_PROTOCOL,
            GateDecision.FAILED,
            evidence,
            reason="at least one of the 36 bounded tuning runs failed terminally",
            decided_at=decided_at,
        )

    scores: dict[str, list[float]] = defaultdict(list)
    for run, bundle in bundles:
        metrics = _assert_sampled_training_protocol(
            bundle,
            store.manifest,
            context=f"tuning run {run.run_id}",
        )
        if _has_heldout_evidence(metrics) or any(
            "heldout" in name.lower().replace("_", "") or "exact_target" in name.lower()
            for name, _ in bundle.dataset_hashes
        ):
            raise AcceptanceEvidenceError(
                "tuning evidence contains forbidden Held-out/Target information",
            )
        if metrics.get("exact_tvd") not in (None, []):
            raise AcceptanceEvidenceError("tuning cannot contain exact TVD evidence")
        _assert_shared_training_hash_schema(
            bundle,
            store.manifest,
            context=f"tuning run {run.run_id}",
        )
        if run.tuning_candidate_id is None:
            raise AcceptanceEvidenceError("tuning run has no candidate identity")
        candidate = store.manifest.training_protocol.candidate(
            run.tuning_candidate_id,
        )
        if candidate.model_family is not run.model_family:
            raise AcceptanceEvidenceError("tuning run uses another family's candidate")
        score = _number(
            metrics.get("best_validation_conditional_mmd"),
            "tuning Validation Conditional MMD",
        )
        if score < 0.0:
            raise AcceptanceEvidenceError("tuning Validation MMD must be non-negative")
        checkpoint = bundle.checkpoint
        if (
            checkpoint is None
            or checkpoint.validation_metric != "conditional_mmd"
            or not math.isclose(
                checkpoint.validation_value,
                score,
                rel_tol=0.0,
                abs_tol=1e-14,
            )
        ):
            raise AcceptanceEvidenceError(
                "tuning checkpoint is not bound to its Validation score",
            )
        scores[candidate.candidate_id].append(score)

    candidate_rows = []
    winners: dict[ModelFamily, str] = {}
    for family in SHARED_PRIMARY_FAMILIES:
        family_scores = []
        candidates = store.manifest.training_protocol.candidates_for(family)
        if len(candidates) != 3:
            raise AcceptanceEvidenceError(
                "each shared family must have exactly three candidates",
            )
        for candidate in candidates:
            values = scores[candidate.candidate_id]
            if len(values) != 4:
                raise AcceptanceEvidenceError(
                    "each tuning candidate requires two instances by two seeds",
                )
            median = float(np.median(np.asarray(values, dtype=np.float64)))
            row = {
                "model_family": family.value,
                "candidate_id": candidate.candidate_id,
                "validation_scores": sorted(values),
                "median_validation_conditional_mmd": median,
            }
            candidate_rows.append(row)
            family_scores.append((median, candidate.candidate_id))
        winners[family] = min(family_scores)[1]
    if len(winners) != 3:
        raise AcceptanceEvidenceError("tuning did not yield three family winners")

    evidence = {
        "schema_version": 1,
        "finalizer": "tuning_v1",
        "manifest_digest": store.manifest.digest,
        "declared_run_count": 36,
        "runs": run_rows,
        "failed_run_ids": [],
        "held_out_access_detected": False,
        "selection_rule": store.manifest.training_protocol.tuning_selection_rule,
        "candidate_scores": sorted(
            candidate_rows,
            key=lambda row: (row["model_family"], row["candidate_id"]),
        ),
        "selections": [
            {"model_family": family.value, "candidate_id": winners[family]}
            for family in SHARED_PRIMARY_FAMILIES
        ],
        "decision": GateDecision.PASSED.value,
    }
    return _seal_gate(
        store,
        Gate.TUNING_PROTOCOL,
        GateDecision.PASSED,
        evidence,
        reason="all 36 Training/Validation-only runs yielded three frozen winners",
        decided_at=decided_at,
        candidate_ids=winners,
    )


def _validate_full_exact_grid(
    bundle: ResultBundle,
    manifest: CampaignManifest,
) -> tuple[tuple[str, float, float], ...]:
    rows = _exact_tvd_rows(bundle)
    if len(rows) != 9:
        raise AcceptanceEvidenceError(
            "primary/ancilla bundles require five Training and four Held-out TVDs",
        )
    for beta in manifest.beta_grid.training:
        _tvd_at(rows, "training", beta)
    for beta in manifest.beta_grid.held_out:
        _tvd_at(rows, "held_out", beta)
    if "exact_targets" not in dict(bundle.dataset_hashes):
        raise AcceptanceEvidenceError(
            "exact evaluation lacks Target SHA-256 provenance",
        )
    return rows


def _retention_evidence(result: BenefitRetentionGateResult) -> dict[str, Any]:
    return {
        "median_retention": result.median_retention,
        "bootstrap": {
            "point_estimate": result.bootstrap.point_estimate,
            "lower": result.bootstrap.lower,
            "upper": result.bootstrap.upper,
            "confidence_level": result.bootstrap.confidence_level,
            "seed": result.bootstrap.seed,
            "replicates": int(result.bootstrap.replicates.size),
        },
        "denominators_valid": result.denominators_valid,
        "invalid_denominator_cells": [
            list(cell) for cell in result.invalid_denominator_cells
        ],
        "median_at_least_half": result.median_at_least_half,
        "lower_bound_positive": result.lower_bound_positive,
        "passed": result.passed,
    }


def _interpolation_evidence(result: InterpolationGateResult) -> dict[str, Any]:
    return {
        "paired_improvement": {
            "point_estimate": result.paired_improvement.point_estimate,
            "lower": result.paired_improvement.lower,
            "upper": result.paired_improvement.upper,
            "confidence_level": result.paired_improvement.confidence_level,
            "seed": result.paired_improvement.seed,
            "replicates": int(result.paired_improvement.replicates.size),
        },
        "interval_strictly_positive": result.interval_strictly_positive,
        "cell_wins": result.cell_wins,
        "cell_total": result.cell_total,
        "cell_win_fraction": result.cell_win_fraction,
        "cell_win_rule_passed": result.cell_win_rule_passed,
        "regressed_instances": list(result.regressed_instances),
        "no_instance_regression": result.no_instance_regression,
        "passed": result.passed,
    }


def _paired_record_payload(record: PairedEvaluationRecord) -> dict[str, Any]:
    return {
        "coupling_instance": record.coupling_instance,
        "optimizer_seed": record.optimizer_seed,
        "condition": record.condition,
        "split": record.split,
        "method_tvds": dict(record.method_tvds),
    }


def _primary_bundle_maps(
    bundles: Sequence[tuple[RunIdentity, ResultBundle]],
    manifest: CampaignManifest,
) -> tuple[
    dict[tuple[int, int, ModelFamily], PrimaryBundleValue],
    dict[tuple[int, int, int], PrimaryBundleValue],
]:
    shared = {}
    separate = {}
    target_hashes: dict[int, set[str]] = defaultdict(set)
    for run, bundle in bundles:
        _assert_sampled_training_protocol(
            bundle,
            manifest,
            context=f"primary run {run.run_id}",
        )
        rows = _validate_full_exact_grid(bundle, manifest)
        target_hashes[run.coupling_seed].add(
            dict(bundle.dataset_hashes)["exact_targets"]
        )
        if run.model_family in SHARED_PRIMARY_FAMILIES:
            key = (run.coupling_seed, run.optimizer_seed, run.model_family)
            if key in shared:
                raise AcceptanceEvidenceError("primary shared cell is duplicated")
            shared[key] = (bundle, rows)
        elif run.model_family is ModelFamily.SEPARATE:
            try:
                beta_index = int(run.condition_scope.removeprefix("train_beta_"))
            except ValueError as error:
                raise AcceptanceEvidenceError(
                    "primary Separate scope is invalid",
                ) from error
            if beta_index not in range(len(manifest.beta_grid.training)):
                raise AcceptanceEvidenceError("primary Separate beta index is invalid")
            key = (run.coupling_seed, run.optimizer_seed, beta_index)
            if key in separate:
                raise AcceptanceEvidenceError("primary Separate cell is duplicated")
            separate[key] = (bundle, rows)
        else:
            raise AcceptanceEvidenceError("primary matrix contains an unknown family")
    if any(len(values) != 1 for values in target_hashes.values()):
        raise AcceptanceEvidenceError(
            "primary pairings do not share one exact-Target archive per instance",
        )
    expected_shared = (
        len(manifest.coupling_seeds)
        * len(manifest.optimizer_seeds)
        * len(SHARED_PRIMARY_FAMILIES)
    )
    expected_separate = (
        len(manifest.coupling_seeds)
        * len(manifest.optimizer_seeds)
        * len(manifest.beta_grid.training)
    )
    if len(shared) != expected_shared or len(separate) != expected_separate:
        raise AcceptanceEvidenceError(
            "primary matrix is not the exact shared/Separate Cartesian product",
        )
    return shared, separate


def _primary_paired_records(
    bundles: Sequence[tuple[RunIdentity, ResultBundle]],
    manifest: CampaignManifest,
) -> tuple[PairedEvaluationRecord, ...]:
    shared, separate = _primary_bundle_maps(bundles, manifest)
    records = []
    for coupling_seed in manifest.coupling_seeds:
        for optimizer_seed in manifest.optimizer_seeds:
            fixed_rows = shared[
                (coupling_seed, optimizer_seed, ModelFamily.FIXED_INPUT)
            ][1]
            phase_rows = shared[
                (
                    coupling_seed,
                    optimizer_seed,
                    ModelFamily.CONDITION_MODULATED_PHASE,
                )
            ][1]
            no_condition_rows = shared[
                (coupling_seed, optimizer_seed, ModelFamily.NO_CONDITION)
            ][1]
            for heldout_index, beta in enumerate(manifest.beta_grid.held_out):
                lower_rows = separate[(coupling_seed, optimizer_seed, heldout_index)][1]
                upper_rows = separate[
                    (coupling_seed, optimizer_seed, heldout_index + 1)
                ][1]
                lower_tvd = _tvd_at(lower_rows, "held_out", beta)
                upper_tvd = _tvd_at(upper_rows, "held_out", beta)
                records.append(
                    PairedEvaluationRecord(
                        coupling_instance=str(coupling_seed),
                        optimizer_seed=optimizer_seed,
                        condition=beta,
                        method_tvds={
                            FIXED_INPUT: _tvd_at(fixed_rows, "held_out", beta),
                            CONDITION_MODULATED_PHASE: _tvd_at(
                                phase_rows,
                                "held_out",
                                beta,
                            ),
                            NO_CONDITION: _tvd_at(
                                no_condition_rows,
                                "held_out",
                                beta,
                            ),
                            ORACLE_ADJACENT: min(lower_tvd, upper_tvd),
                            "oracle_lower_separate": lower_tvd,
                            "oracle_upper_separate": upper_tvd,
                        },
                    ),
                )
    expected = (
        len(manifest.coupling_seeds)
        * len(manifest.optimizer_seeds)
        * len(manifest.beta_grid.held_out)
    )
    if (
        len(records) != expected
        or len({record.identity for record in records}) != expected
    ):
        raise AcceptanceEvidenceError(
            "primary evidence is not the exact 8x5x4 paired Held-out product",
        )
    return tuple(records)


def finalize_primary(
    store: CampaignStore,
    *,
    decided_at: str | None = None,
) -> PrimaryFinalization:
    """Derive both binding empirical gates from all 320 primary bundles."""
    bundles = _stage_bundles(store, Stage.PRIMARY)
    if len(bundles) != store.manifest.run_allocation.primary or len(bundles) != 320:
        raise AcceptanceEvidenceError("primary finalization requires exactly 320 runs")
    failed_runs = [
        run.run_id
        for run, bundle in bundles
        if bundle.completion_status is not CompletionStatus.COMPLETED
    ]
    if failed_runs:
        common = {
            "schema_version": 1,
            "finalizer": "primary_v1",
            "manifest_digest": store.manifest.digest,
            "declared_run_count": 320,
            "unique_run_count": len({run.run_id for run, _ in bundles}),
            "runs": _bundle_rows(bundles),
            "failed_run_ids": failed_runs,
            "paired_cell_count": 0,
            "paired_records": [],
            "benefit_retention": None,
            "interpolation": None,
            "decision_basis": "terminal run failure prevents scientific acceptance",
        }
        retention_evidence = {**common, "gate": Gate.BENEFIT_RETENTION.value}
        interpolation_evidence = {**common, "gate": Gate.INTERPOLATION.value}
        retention = _seal_gate(
            store,
            Gate.BENEFIT_RETENTION,
            GateDecision.FAILED,
            retention_evidence,
            reason="the 320-run primary matrix contains failed terminal runs",
            decided_at=decided_at,
        )
        interpolation = _seal_gate(
            store,
            Gate.INTERPOLATION,
            GateDecision.FAILED,
            interpolation_evidence,
            reason="the 320-run primary matrix contains failed terminal runs",
            decided_at=decided_at,
        )
        return PrimaryFinalization(retention, interpolation, ())

    records = _primary_paired_records(bundles, store.manifest)
    retention_result = benefit_retention_gate(records)
    interpolation_result = interpolation_gate(records)
    common = {
        "schema_version": 1,
        "finalizer": "primary_v1",
        "manifest_digest": store.manifest.digest,
        "declared_run_count": 320,
        "unique_run_count": len({run.run_id for run, _ in bundles}),
        "run_bundle_digests": [
            {"run_id": run.run_id, "bundle_digest": bundle.digest}
            for run, bundle in bundles
        ],
        "paired_cell_shape": [8, 5, 4],
        "paired_cell_count": len(records),
        "paired_records": [_paired_record_payload(record) for record in records],
        "oracle_rule": (
            "minimum exact Held-out TVD of the two adjacent Separate checkpoints"
        ),
        "bootstrap": {
            "replicates": store.manifest.statistics.bootstrap_replicates,
            "seed": store.manifest.statistics.bootstrap_seed,
            "confidence_level": store.manifest.statistics.confidence_level,
            "pairing": (
                "instance first; optimizer seed within instance; methods and "
                "Conditions retained"
            ),
        },
        "benefit_retention": _retention_evidence(retention_result),
        "interpolation": _interpolation_evidence(interpolation_result),
    }
    retention_decision = (
        GateDecision.PASSED if retention_result.passed else GateDecision.FAILED
    )
    interpolation_decision = (
        GateDecision.PASSED if interpolation_result.passed else GateDecision.FAILED
    )
    retention = _seal_gate(
        store,
        Gate.BENEFIT_RETENTION,
        retention_decision,
        {**common, "gate": Gate.BENEFIT_RETENTION.value},
        reason=(
            "complete paired primary evidence passed the frozen retention rule"
            if retention_result.passed
            else "complete paired primary evidence failed the frozen retention rule"
        ),
        decided_at=decided_at,
    )
    interpolation = _seal_gate(
        store,
        Gate.INTERPOLATION,
        interpolation_decision,
        {**common, "gate": Gate.INTERPOLATION.value},
        reason=(
            "complete paired primary evidence passed all interpolation rules"
            if interpolation_result.passed
            else "complete paired primary evidence failed an interpolation rule"
        ),
        decided_at=decided_at,
    )
    return PrimaryFinalization(retention, interpolation, records)


def release_ancilla_extension(
    store: CampaignStore,
    *,
    decided_at: str | None = None,
) -> GateFinalization:
    """Release the 24-run extension only after both primary gates pass."""
    primary_records = {}
    for gate in (Gate.BENEFIT_RETENTION, Gate.INTERPOLATION):
        record = store.gate_record(gate)
        if record is None or record.decision is not GateDecision.PASSED:
            raise AcceptanceEvidenceError(
                "ancilla release requires passed benefit-retention and "
                "interpolation gates",
            )
        binding = store.gate_evidence_binding(gate)
        if binding is None:
            raise AcceptanceEvidenceError(
                f"ancilla release requires verified {gate.value} evidence",
            )
        primary_records[gate.value] = {
            "gate_record_digest": record.digest,
            "evidence_path": binding.evidence_path,
            "evidence_sha256": binding.evidence_sha256,
        }
    evidence = {
        "schema_version": 1,
        "finalizer": "ancilla_release_v1",
        "manifest_digest": store.manifest.digest,
        "primary_gates": primary_records,
        "declared_run_count": store.manifest.run_allocation.ancilla,
        "coupling_seeds": list(store.manifest.ancilla.coupling_seeds),
        "optimizer_seeds": list(store.manifest.ancilla.optimizer_seeds),
        "rungs": list(store.manifest.ancilla.rungs),
        "families": [
            ModelFamily.FIXED_INPUT.value,
            ModelFamily.MATCHED_CAPACITY_CONTROL.value,
        ],
        "decision": GateDecision.PASSED.value,
    }
    return _seal_gate(
        store,
        Gate.ANCILLA_EXTENSION,
        GateDecision.PASSED,
        evidence,
        reason="both visible-only primary gates passed with verified evidence",
        decided_at=decided_at,
    )


def _ancilla_result_evidence(result: AncillaSuccessResult) -> dict[str, Any]:
    return {
        "rungs": [
            {
                "rung": rung.rung,
                "median_improvement_vs_m0": rung.median_improvement_vs_m0,
                "median_improvement_vs_matched_capacity": (
                    rung.median_improvement_vs_matched_capacity
                ),
                "median_rules_passed": rung.median_rules_passed,
                "wins_vs_m0": rung.wins_vs_m0,
                "wins_vs_matched_capacity": rung.wins_vs_matched_capacity,
                "cell_total": rung.cell_total,
                "win_fraction_vs_m0": rung.win_fraction_vs_m0,
                "win_fraction_vs_matched_capacity": (
                    rung.win_fraction_vs_matched_capacity
                ),
                "cell_win_rules_passed": rung.cell_win_rules_passed,
                "regressed_instances_vs_m0": list(
                    rung.regressed_instances_vs_m0,
                ),
                "no_instance_regression": rung.no_instance_regression,
                "passed": rung.passed,
            }
            for rung in result.rungs
        ],
        "successful_rungs": list(result.successful_rungs),
        "passed": result.passed,
        "population_claim_allowed": result.population_claim_allowed,
    }


def finalize_ancilla_result(store: CampaignStore) -> DerivedFinalization:
    """Verify all 24 released runs and seal the descriptive both-rung result."""
    release = store.gate_record(Gate.ANCILLA_EXTENSION)
    if release is None or release.decision is not GateDecision.PASSED:
        raise AcceptanceEvidenceError("ancilla results require a passed release gate")
    if store.gate_evidence_binding(Gate.ANCILLA_EXTENSION) is None:
        raise AcceptanceEvidenceError("ancilla release evidence is not archived")
    bundles = _stage_bundles(store, Stage.ANCILLA)
    if len(bundles) != store.manifest.run_allocation.ancilla or len(bundles) != 24:
        raise AcceptanceEvidenceError("ancilla result requires exactly 24 runs")
    failed = [
        run.run_id
        for run, bundle in bundles
        if bundle.completion_status is not CompletionStatus.COMPLETED
    ]
    if failed:
        raise AcceptanceEvidenceError(
            "all 24 ancilla and Matched-capacity runs must complete",
        )

    primary_bundles = _stage_bundles(store, Stage.PRIMARY)
    shared, _ = _primary_bundle_maps(primary_bundles, store.manifest)
    ancilla_map: dict[
        tuple[int, int, int, ModelFamily],
        tuple[ResultBundle, tuple[tuple[str, float, float], ...]],
    ] = {}
    for run, bundle in bundles:
        _assert_sampled_training_protocol(
            bundle,
            store.manifest,
            context=f"ancilla run {run.run_id}",
        )
        rows = _validate_full_exact_grid(bundle, store.manifest)
        if run.ancilla_rung not in store.manifest.ancilla.rungs:
            raise AcceptanceEvidenceError("ancilla run has an undeclared rung")
        key = (
            run.coupling_seed,
            run.optimizer_seed,
            run.ancilla_rung,
            run.model_family,
        )
        if key in ancilla_map:
            raise AcceptanceEvidenceError("ancilla run cell is duplicated")
        ancilla_map[key] = (bundle, rows)

        m0_bundle = shared[
            (run.coupling_seed, run.optimizer_seed, ModelFamily.FIXED_INPUT)
        ][0]
        reused = bundle.raw_metrics.get("reused_m0_checkpoint")
        if m0_bundle.checkpoint is None or reused != m0_bundle.checkpoint.to_dict():
            raise AcceptanceEvidenceError(
                "ancilla bundle does not reuse its exact paired m=0 checkpoint",
            )
        if (
            dict(bundle.dataset_hashes)["exact_targets"]
            != dict(
                m0_bundle.dataset_hashes,
            )["exact_targets"]
        ):
            raise AcceptanceEvidenceError(
                "ancilla and m=0 pair use different exact Target evidence",
            )

    expected_keys = {
        (seed, optimizer_seed, rung, family)
        for seed in store.manifest.ancilla.coupling_seeds
        for optimizer_seed in store.manifest.ancilla.optimizer_seeds
        for rung in store.manifest.ancilla.rungs
        for family in (
            ModelFamily.FIXED_INPUT,
            ModelFamily.MATCHED_CAPACITY_CONTROL,
        )
    }
    if set(ancilla_map) != expected_keys:
        raise AcceptanceEvidenceError(
            "ancilla result is not the exact 2x3x2x2 declared product",
        )

    records = []
    for seed in store.manifest.ancilla.coupling_seeds:
        for optimizer_seed in store.manifest.ancilla.optimizer_seeds:
            m0_rows = shared[(seed, optimizer_seed, ModelFamily.FIXED_INPUT)][1]
            for rung in store.manifest.ancilla.rungs:
                ancilla_rows = ancilla_map[
                    (seed, optimizer_seed, rung, ModelFamily.FIXED_INPUT)
                ][1]
                matched_rows = ancilla_map[
                    (
                        seed,
                        optimizer_seed,
                        rung,
                        ModelFamily.MATCHED_CAPACITY_CONTROL,
                    )
                ][1]
                for beta in store.manifest.beta_grid.held_out:
                    records.append(
                        AncillaEvaluationRecord(
                            coupling_instance=str(seed),
                            optimizer_seed=optimizer_seed,
                            condition=beta,
                            rung=rung,
                            ancilla_tvd=_tvd_at(ancilla_rows, "held_out", beta),
                            m0_tvd=_tvd_at(m0_rows, "held_out", beta),
                            matched_capacity_tvd=_tvd_at(
                                matched_rows,
                                "held_out",
                                beta,
                            ),
                        ),
                    )
    expected_records = 2 * 3 * 2 * 4
    if len(records) != expected_records:
        raise AcceptanceEvidenceError("ancilla paired record cardinality is wrong")
    result = ancilla_descriptive_success_rule(records)
    evidence = {
        "schema_version": 1,
        "finalizer": "ancilla_result_v1",
        "manifest_digest": store.manifest.digest,
        "release_gate_digest": release.digest,
        "declared_run_count": 24,
        "unique_run_count": len({run.run_id for run, _ in bundles}),
        "run_bundle_digests": [
            {"run_id": run.run_id, "bundle_digest": bundle.digest}
            for run, bundle in bundles
        ],
        "paired_cell_shape": [2, 3, 2, 4],
        "paired_record_count": len(records),
        "paired_records": [
            {
                "coupling_instance": record.coupling_instance,
                "optimizer_seed": record.optimizer_seed,
                "condition": record.condition,
                "rung": record.rung,
                "ancilla_tvd": record.ancilla_tvd,
                "m0_tvd": record.m0_tvd,
                "matched_capacity_tvd": record.matched_capacity_tvd,
            }
            for record in records
        ],
        "both_rungs_reported": tuple(item.rung for item in result.rungs) == (3, 5),
        "descriptive_only": True,
        "result": _ancilla_result_evidence(result),
    }
    return _write_derived(
        store,
        "ancilla_result",
        evidence,
        passed=result.passed,
    )
