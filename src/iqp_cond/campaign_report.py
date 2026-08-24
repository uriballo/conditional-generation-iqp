"""Deterministic campaign-level acceptance and reproducibility reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    ArtifactError,
    CompletionStatus,
    ResultBundle,
    read_immutable_json,
)
from .campaign import (
    CampaignStore,
    Gate,
    GateDecision,
    RunReservation,
    gate_ancestors,
)
from .campaign_statistics import (
    PairedEvaluationRecord,
    benefit_retention_gate,
    interpolation_gate,
)
from .exact import enumerate_bitstrings
from .ising import ising_energies, periodic_lattice_couplings
from .protocol import (
    BudgetPool,
    CampaignManifest,
    Stage,
    canonical_json_bytes,
    content_digest,
)

REPORT_SCHEMA_VERSION = 1

_STAGE_DECLARED_COUNTS = {
    Stage.RUNTIME_FEASIBILITY: 1,
    Stage.REPRESENTABILITY_SCREEN: 6,
    Stage.REPRESENTABILITY_DIAGNOSTIC: 1,
    Stage.TUNING: 36,
    Stage.PRIMARY: 320,
    Stage.ANCILLA: 24,
}

_STAGE_RELEASE_GATE = {
    Stage.RUNTIME_FEASIBILITY: Gate.SAMPLING_QUALITY,
    Stage.REPRESENTABILITY_SCREEN: Gate.RUNTIME_FEASIBILITY,
    Stage.TUNING: Gate.REPRESENTABILITY,
    Stage.PRIMARY: Gate.TUNING_PROTOCOL,
    Stage.ANCILLA: Gate.ANCILLA_EXTENSION,
}

_STAGE_POOL = {
    Stage.RUNTIME_FEASIBILITY: BudgetPool.FEASIBILITY_REPRESENTABILITY,
    Stage.REPRESENTABILITY_SCREEN: BudgetPool.FEASIBILITY_REPRESENTABILITY,
    Stage.REPRESENTABILITY_DIAGNOSTIC: (BudgetPool.FEASIBILITY_REPRESENTABILITY),
    Stage.TUNING: BudgetPool.TUNING,
    Stage.PRIMARY: BudgetPool.PRIMARY,
    Stage.ANCILLA: BudgetPool.ANCILLA,
}

_FIGURE_SOURCES = {
    "fig1_pipeline_targets": (
        "scripts/plot_gaussian_campaign.py",
        "results/gaussian-foundations-v1/report/complete-report.json",
    ),
    "gaussian_evidence": (
        "scripts/plot_foundations_evidence.py",
        "results/arxiv-foundations-v1/gaussian/paired-evidence.json",
    ),
    "ising_representability": (
        "scripts/plot_foundations_evidence.py",
        "results/arxiv-foundations-v1/evidence/representability.json",
    ),
    "response_support": (
        "scripts/generate_quantum_draft_figures.py",
        "docs/theorem-novelty-dossier.md",
    ),
    "ising_calibration": (
        "scripts/generate_quantum_draft_figures.py",
        "results/quantum-campaign-v1/calibration.json",
    ),
    "sampler_pilots": (
        "scripts/generate_quantum_draft_figures.py",
        "results/quantum-campaign-v1/sampler-attempts/thinning-10/summary.json",
        "results/quantum-campaign-v1/sampler-attempts/thinning-16/summary.json",
    ),
    "weight_fit": (
        "scripts/summarize_results.py",
        "results/summary/metrics.md",
    ),
    "fig5_ablations": (
        "scripts/plot_gaussian_campaign.py",
        "results/gaussian-foundations-v1/report/complete-report.json",
    ),
    "curie_weiss_thermodynamics": (
        "scripts/plot_curie_weiss_campaign.py",
        "results/curie-weiss-foundations-v1/report/campaign-report.json",
    ),
    "encoding_benchmark": (
        "scripts/summarize_encoding_benchmark.py",
        "results/encoding_benchmark/summary/metrics.json",
    ),
    "matched_capacity_v2": (
        "scripts/plot_curie_weiss_capacity_control.py",
        "results/curie-weiss-capacity-control-v2/summary.json",
    ),
    "two_block_screen": (
        "scripts/plot_symmetry_screens.py",
        "results/two-block-screen-v1/rows",
    ),
    "radial_shell": (
        "scripts/plot_symmetry_screens.py",
        "results/radial-shell-screen-v1/rows",
    ),
}

_SCALING_SOURCE_PATHS = {
    "configs/quantum_campaign_v1.json",
    "scripts/run_estimator_scaling.py",
    "src/iqp_cond/estimator_scaling.py",
    "src/iqp_cond/exact.py",
    "src/iqp_cond/response.py",
    "src/iqp_cond/simulator.py",
}

_BLOCKED_SPECTRUM_KEYS = {
    "schema_version",
    "evidence_type",
    "status",
    "campaign_version",
    "manifest_digest",
    "configuration_digest",
    "full_training_runs_used",
    "budget_pool",
    "budget_accounting",
    "blockers",
    "compatible_checkpoints",
    "interpretation",
}

_BLOCKED_SPECTRUM_CODES = {
    "theorem_novelty_failed",
    "fixed_input_checkpoint_missing",
    "condition_modulated_phase_checkpoint_missing",
}


@dataclass(frozen=True)
class CampaignReport:
    """Canonical report payload with a stable content digest."""

    payload_json: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("campaign report payload is invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("campaign report payload must be a JSON object")
        canonical = canonical_json_bytes(payload).decode("utf-8")
        if self.payload_json != canonical:
            raise ValueError("campaign report payload must use canonical JSON")

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> CampaignReport:
        """Canonicalize one fully assembled report payload."""
        return cls(canonical_json_bytes(payload).decode("utf-8"))

    @property
    def payload(self) -> dict[str, Any]:
        """Return a fresh mutable view of the report payload."""
        return json.loads(self.payload_json)

    @property
    def digest(self) -> str:
        """Return the stable digest of every report row."""
        return content_digest(self.payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a self-verifying serialized report document."""
        return {**self.payload, "report_digest": self.digest}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def _repository_path(repository: Path, value: object) -> Path:
    """Resolve one portable repository-relative artifact path."""
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be repository-relative")
    resolved = (repository / relative).resolve()
    try:
        resolved.relative_to(repository.resolve())
    except ValueError as error:
        raise ValueError("artifact path resolves outside the repository") from error
    return resolved


def _load_reservations(store: CampaignStore) -> tuple[RunReservation, ...]:
    reservations = []
    for path in sorted(store.reservations_directory.glob("*.json")):
        reservation = store.reservation(path.stem)
        if reservation is None:
            raise ValueError(f"reservation vanished while reporting: {path}")
        reservations.append(reservation)
    return tuple(reservations)


def _load_bundles(store: CampaignStore) -> tuple[ResultBundle, ...]:
    bundles = []
    for path in sorted(store.artifacts.runs_directory.glob("*.json")):
        bundle = store.result(path.stem)
        if bundle is None:
            raise ValueError(f"result vanished while reporting: {path}")
        bundles.append(bundle)
    return tuple(bundles)


def _first_failed_gate(
    store: CampaignStore,
    gate: Gate,
    *,
    include_gate: bool = False,
) -> Gate | None:
    candidates = (*gate_ancestors(gate), gate) if include_gate else gate_ancestors(gate)
    for candidate in candidates:
        record = store.gate_record(candidate)
        if record is not None and record.decision is GateDecision.FAILED:
            return candidate
    return None


def _gate_evidence_rows(store: CampaignStore) -> list[dict[str, object]]:
    rows = []
    for gate in Gate:
        record = store.gate_record(gate)
        if record is None:
            continue
        try:
            binding = store.gate_evidence_binding(gate)
        except (ArtifactError, OSError, ValueError) as error:
            rows.append(
                {
                    "gate": gate.value,
                    "status": "fail",
                    "evidence_path": None,
                    "expected_sha256": record.evidence_digest,
                    "actual_sha256": None,
                    "details": str(error),
                },
            )
            continue
        if binding is None:
            rows.append(
                {
                    "gate": gate.value,
                    "status": "fail",
                    "evidence_path": None,
                    "expected_sha256": record.evidence_digest,
                    "actual_sha256": None,
                    "details": "terminal gate has no immutable evidence binding",
                },
            )
            continue
        rows.append(
            {
                "gate": gate.value,
                "status": "pass",
                "evidence_path": binding.evidence_path,
                "expected_sha256": record.evidence_digest,
                "actual_sha256": binding.evidence_sha256,
                "details": "archived bytes, terminal decision, and manifest agree",
            },
        )
    return rows


def _gate_rows(
    store: CampaignStore,
    evidence_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    evidence_by_gate = {str(row["gate"]): row for row in evidence_rows}
    rows = []
    for gate in Gate:
        record = store.gate_record(gate)
        if record is not None:
            evidence = evidence_by_gate[gate.value]
            rows.append(
                {
                    "gate": gate.value,
                    "status": record.decision.value,
                    "decision": record.decision.value,
                    "evidence_digest": record.evidence_digest,
                    "evidence_path": evidence["evidence_path"],
                    "evidence_status": evidence["status"],
                    "decided_at": record.decided_at,
                    "reason": record.reason,
                },
            )
            continue
        failed = _first_failed_gate(store, gate)
        if failed is not None:
            rows.append(
                {
                    "gate": gate.value,
                    "status": "blocked",
                    "decision": None,
                    "evidence_digest": None,
                    "evidence_path": None,
                    "evidence_status": "not_applicable",
                    "decided_at": None,
                    "reason": f"terminally blocked by failed {failed.value}",
                },
            )
        else:
            rows.append(
                {
                    "gate": gate.value,
                    "status": "pending",
                    "decision": None,
                    "evidence_digest": None,
                    "evidence_path": None,
                    "evidence_status": "not_applicable",
                    "decided_at": None,
                    "reason": "no terminal decision recorded",
                },
            )
    return rows


def _stage_status(store: CampaignStore, stage: Stage) -> tuple[str, str]:
    if stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
        failed = _first_failed_gate(store, Gate.REPRESENTABILITY)
        if failed is not None:
            return "blocked", f"terminally blocked by failed {failed.value}"
        representability = store.gate_record(Gate.REPRESENTABILITY)
        if representability is None:
            return "pending", "conditional on representability failure"
        if representability.decision is GateDecision.FAILED:
            return "eligible", "sole exact-objective diagnostic released"
        return "not_released", "representability passed; diagnostic not permitted"

    release_gate = _STAGE_RELEASE_GATE[stage]
    failed = _first_failed_gate(store, release_gate, include_gate=True)
    if failed is not None:
        return "blocked", f"terminally blocked by failed {failed.value}"
    record = store.gate_record(release_gate)
    if record is None:
        return "pending", f"awaiting passed {release_gate.value}"
    return "released", f"released by passed {release_gate.value}"


def _run_rows(
    store: CampaignStore,
    reservations: Sequence[RunReservation],
    bundles: Sequence[ResultBundle],
) -> list[dict[str, object]]:
    rows = []
    for stage in Stage:
        stage_reservations = [item for item in reservations if item.run.stage is stage]
        stage_bundles = [item for item in bundles if item.run.stage is stage]
        completed = sum(
            bundle.completion_status is CompletionStatus.COMPLETED
            for bundle in stage_bundles
        )
        failed_results = sum(
            bundle.completion_status is CompletionStatus.FAILED
            for bundle in stage_bundles
        )
        status, reason = _stage_status(store, stage)
        rows.append(
            {
                "stage": stage.value,
                "budget_pool": _STAGE_POOL[stage].value,
                "declared": _STAGE_DECLARED_COUNTS[stage],
                "reserved": len(stage_reservations),
                "completed": completed,
                "failed": failed_results,
                "status": status,
                "reason": reason,
            },
        )
    return rows


def _budget_rows(
    manifest: CampaignManifest,
    reservations: Sequence[RunReservation],
    bundles: Sequence[ResultBundle],
    theorem_failed: bool,
) -> list[dict[str, object]]:
    bundle_by_id = {bundle.run.run_id: bundle for bundle in bundles}
    rows = []
    for pool in BudgetPool:
        pool_reservations = [item for item in reservations if item.budget_pool is pool]
        pool_bundles = [
            bundle_by_id[item.run.run_id]
            for item in pool_reservations
            if item.run.run_id in bundle_by_id
        ]
        ceiling = manifest.run_allocation.for_pool(pool)
        rows.append(
            {
                "pool": pool.value,
                "ceiling": ceiling,
                "reserved": len(pool_reservations),
                "completed": sum(
                    bundle.completion_status is CompletionStatus.COMPLETED
                    for bundle in pool_bundles
                ),
                "failed": sum(
                    bundle.completion_status is CompletionStatus.FAILED
                    for bundle in pool_bundles
                ),
                "remaining": ceiling - len(pool_reservations),
                "transferable": False,
                "status": "blocked_unspent" if theorem_failed else "active",
            },
        )
    rows.append(
        {
            "pool": "total",
            "ceiling": sum(row["ceiling"] for row in rows),
            "reserved": sum(row["reserved"] for row in rows),
            "completed": sum(row["completed"] for row in rows),
            "failed": sum(row["failed"] for row in rows),
            "remaining": sum(row["remaining"] for row in rows),
            "transferable": False,
            "status": "blocked_unspent" if theorem_failed else "active",
        },
    )
    return rows


def _walk_keys(value: Any) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key).lower())
            keys.extend(_walk_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            keys.extend(_walk_keys(item))
    return tuple(keys)


def _leakage_rows(
    manifest: CampaignManifest,
    repository: Path,
    bundles: Sequence[ResultBundle],
) -> list[dict[str, object]]:
    calibration_path = repository / manifest.calibration.artifact_path
    calibration_keys: tuple[str, ...] = ()
    if calibration_path.exists():
        try:
            calibration_payload = json.loads(calibration_path.read_text())
        except (OSError, json.JSONDecodeError):
            calibration_payload = {}
        calibration_keys = _walk_keys(calibration_payload)
    forbidden_calibration_keys = {
        "model_metrics",
        "checkpoint_metrics",
        "validation_loss",
        "training_loss",
    }
    calibration_leaks = sorted(
        forbidden_calibration_keys.intersection(calibration_keys)
    )

    split_collisions = []
    training_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    preprimary_heldout_inputs = []
    checkpoint_mismatches = []
    split_key_violations = []
    for bundle in bundles:
        hashes = dict(bundle.dataset_hashes)
        training_by_index = {
            int(match.group(1)): digest
            for name, digest in hashes.items()
            if (match := re.fullmatch(r"training_beta_(\d+)", name)) is not None
        }
        validation_by_index = {
            int(match.group(1)): digest
            for name, digest in hashes.items()
            if (match := re.fullmatch(r"validation_beta_(\d+)", name)) is not None
        }
        if bundle.run.condition_scope == "all_training_betas":
            expected_indices = set(range(len(manifest.beta_grid.training)))
        elif bundle.run.condition_scope.startswith("train_beta_"):
            expected_indices = {0}
        elif bundle.run.stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
            expected_indices = {0, 1, 2}
        else:
            expected_indices = set(training_by_index)
        if (
            set(training_by_index) != expected_indices
            or set(validation_by_index) != expected_indices
            or "training" in hashes
            or "validation" in hashes
        ):
            split_key_violations.append(bundle.run.run_id)
        for index in sorted(set(training_by_index).intersection(validation_by_index)):
            training_digest = training_by_index[index]
            validation_digest = validation_by_index[index]
            training_hashes.add(training_digest)
            validation_hashes.add(validation_digest)
            if training_digest == validation_digest:
                split_collisions.append(f"{bundle.run.run_id}:beta_{index}")
        if bundle.run.stage in {
            Stage.RUNTIME_FEASIBILITY,
            Stage.REPRESENTABILITY_SCREEN,
            Stage.TUNING,
        } and any("held_out" in name.lower() for name in hashes):
            preprimary_heldout_inputs.append(bundle.run.run_id)
        if (
            bundle.checkpoint is not None
            and "validation" not in bundle.checkpoint.selection_rule.lower()
        ):
            checkpoint_mismatches.append(bundle.run.run_id)
    split_collisions.extend(
        f"cross-run digest {digest}"
        for digest in sorted(training_hashes.intersection(validation_hashes))
    )

    return [
        {
            "check": "target_only_calibration_inputs",
            "status": "pass" if not calibration_leaks else "fail",
            "violations": calibration_leaks,
            "details": "model/checkpoint performance keys are forbidden",
        },
        {
            "check": "dataset_split_key_schema",
            "status": "pass" if not split_key_violations else "fail",
            "violations": sorted(split_key_violations),
            "details": (
                "every finite-data input must use matched "
                "training_beta_i/validation_beta_i content-digest keys"
            ),
        },
        {
            "check": "training_validation_hash_disjointness",
            "status": "pass" if not split_collisions else "fail",
            "violations": sorted(split_collisions),
            "details": "Training and Validation hashes must differ within every run",
        },
        {
            "check": "held_out_preprimary_access",
            "status": "pass" if not preprimary_heldout_inputs else "fail",
            "violations": sorted(preprimary_heldout_inputs),
            "details": "Held-out hashes are forbidden before frozen primary evaluation",
        },
        {
            "check": "validation_only_checkpoint_selection",
            "status": "pass" if not checkpoint_mismatches else "fail",
            "violations": sorted(checkpoint_mismatches),
            "details": manifest.training_protocol.checkpoint_rule,
        },
    ]


def _split_markdown_row(line: str) -> list[str]:
    text = line.strip()
    if not text.startswith("|") or not text.endswith("|"):
        return []
    fields: list[str] = []
    current: list[str] = []
    in_code = False
    for character in text[1:-1]:
        if character == "`":
            in_code = not in_code
            current.append(character)
        elif character == "|" and not in_code:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    fields.append("".join(current).strip())
    return fields


def claim_support_records(path: Path) -> tuple[dict[str, str], ...]:
    """Parse every Abstract/Introduction claim row from the binding registry."""
    records = []
    for line in path.read_text().splitlines():
        fields = _split_markdown_row(line)
        if len(fields) != 5:
            continue
        identifier = fields[0].strip("`")
        if re.fullmatch(r"(?:A|I)-\d{2}", identifier) is None:
            continue
        records.append(
            {
                "claim_id": identifier,
                "section": "abstract"
                if identifier.startswith("A-")
                else "introduction",
                "claim": fields[1],
                "evidence": fields[2],
                "status": fields[3].replace("`", ""),
                "allowed_wording": fields[4],
            },
        )
    identifiers = [record["claim_id"] for record in records]
    if not records or len(identifiers) != len(set(identifiers)):
        raise ValueError("claim registry has no claim rows or duplicate IDs")
    return tuple(records)


def _figure_inventory(
    repository: Path, manuscript_sources: Sequence[Path]
) -> list[dict[str, object]]:
    references = []
    pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    for source in manuscript_sources:
        for match in pattern.finditer(source.read_text()):
            if "#" in match.group(1):
                # Ignore parameter placeholders inside reusable figure macros.
                # Concrete invocations are audited by the active manuscript's
                # own source/build checks.
                continue
            referenced = (source.parent / match.group(1)).resolve()
            stem = referenced.stem
            sources = _FIGURE_SOURCES.get(stem, ())
            source_rows = [
                {
                    "path": item,
                    "exists": (repository / item).exists(),
                }
                for item in sources
            ]
            references.append(
                {
                    "figure": stem,
                    "manuscript_source": _relative(source, repository),
                    "output": _relative(referenced, repository),
                    "output_exists": referenced.exists(),
                    "source_files": source_rows,
                    "status": (
                        "pass"
                        if referenced.exists()
                        and sources
                        and all(item["exists"] for item in source_rows)
                        else "fail"
                    ),
                },
            )
    return sorted(references, key=lambda row: (str(row["figure"]), str(row["output"])))


def _paired_records(
    bundles: Sequence[ResultBundle],
) -> tuple[PairedEvaluationRecord, ...]:
    by_identity: dict[tuple[str, int, str, float], PairedEvaluationRecord] = {}
    for bundle in bundles:
        if bundle.run.stage is not Stage.PRIMARY:
            continue
        values = bundle.raw_metrics.get("paired_evaluations", [])
        if not isinstance(values, list):
            raise ValueError("paired_evaluations must be a JSON list")
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError("paired evaluation rows must be objects")
            methods = value.get("method_tvds")
            if not isinstance(methods, Mapping):
                raise ValueError("paired evaluation method_tvds must be an object")
            record = PairedEvaluationRecord(
                coupling_instance=str(value["coupling_instance"]),
                optimizer_seed=int(value["optimizer_seed"]),
                condition=float(value["condition"]),
                split=str(value.get("split", "held_out")),
                method_tvds={
                    str(key): float(metric) for key, metric in methods.items()
                },
            )
            previous = by_identity.get(record.identity)
            if previous is not None and dict(previous.method_tvds) != dict(
                record.method_tvds,
            ):
                raise ValueError(f"conflicting paired metrics for {record.identity}")
            by_identity[record.identity] = record
    return tuple(by_identity[key] for key in sorted(by_identity))


def _statistics_summary(
    records: Sequence[PairedEvaluationRecord],
    theorem_failed: bool,
) -> dict[str, object]:
    if not records:
        return {
            "status": "blocked" if theorem_failed else "unavailable",
            "paired_cells": 0,
            "benefit_retention": None,
            "interpolation": None,
            "reason": (
                "theorem novelty failed before primary evaluation"
                if theorem_failed
                else "no paired primary metrics are archived"
            ),
        }
    retention = benefit_retention_gate(records)
    interpolation = interpolation_gate(records)
    return {
        "status": "computed",
        "paired_cells": len(records),
        "benefit_retention": {
            "median": retention.median_retention,
            "lower": retention.bootstrap.lower,
            "upper": retention.bootstrap.upper,
            "denominators_valid": retention.denominators_valid,
            "median_at_least_half": retention.median_at_least_half,
            "lower_bound_positive": retention.lower_bound_positive,
            "passed": retention.passed,
        },
        "interpolation": {
            "mean_improvement": interpolation.paired_improvement.point_estimate,
            "lower": interpolation.paired_improvement.lower,
            "upper": interpolation.paired_improvement.upper,
            "cell_wins": interpolation.cell_wins,
            "cell_total": interpolation.cell_total,
            "cell_win_fraction": interpolation.cell_win_fraction,
            "regressed_instances": list(interpolation.regressed_instances),
            "passed": interpolation.passed,
        },
        "reason": "computed from pairing-preserving archived Held-out cells",
    }


def _target_digest(
    coupling_seed: int,
    beta: float,
    couplings: np.ndarray,
    probabilities: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(coupling_seed, dtype=np.int64).tobytes())
    digest.update(np.asarray(beta, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(couplings, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(probabilities, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _same_float_sequence(value: object, expected: Sequence[float]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    try:
        return bool(np.allclose(value, expected, atol=1e-14, rtol=0.0))
    except (TypeError, ValueError):
        return False


def _validate_target_archive(
    path: Path,
    coupling_seed: int,
    betas: tuple[float, ...],
    expected_target_digests: object,
) -> list[str]:
    violations: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "coupling_seed",
                "betas",
                "couplings",
                "bitstrings",
                "energies",
                "probabilities",
            }
            if set(archive.files) != required:
                return ["unexpected NPZ member keys"]
            stored_seed = int(archive["coupling_seed"])
            stored_betas = np.asarray(archive["betas"], dtype=np.float64)
            couplings = np.asarray(archive["couplings"], dtype=np.float64)
            bitstrings = np.asarray(archive["bitstrings"])
            energies = np.asarray(archive["energies"], dtype=np.float64)
            probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    except (OSError, ValueError, TypeError) as error:
        return [f"could not load target archive: {error}"]

    if stored_seed != coupling_seed:
        violations.append("embedded coupling seed differs")
    if stored_betas.shape != (len(betas),) or not np.allclose(
        stored_betas,
        betas,
        atol=1e-14,
        rtol=0.0,
    ):
        violations.append("embedded beta grid differs from manifest")
    expected_bitstrings = enumerate_bitstrings(16)
    if bitstrings.shape != expected_bitstrings.shape or not np.array_equal(
        bitstrings,
        expected_bitstrings,
    ):
        violations.append("bitstrings are not the exact 65,536-state ordering")
    expected_couplings = periodic_lattice_couplings(coupling_seed)
    if couplings.shape != (16, 16) or not np.array_equal(
        couplings,
        expected_couplings,
    ):
        violations.append("couplings differ from the frozen generator")
    energies_valid = energies.shape == (2**16,) and np.allclose(
        energies,
        ising_energies(expected_bitstrings, expected_couplings),
        atol=1e-12,
        rtol=0.0,
    )
    if not energies_valid:
        violations.append("energies are inconsistent with states and couplings")
    if probabilities.shape != (len(betas), 2**16):
        violations.append("probability table has the wrong shape")
    elif (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    ):
        violations.append("probability rows are not normalized distributions")
    elif energies_valid:
        for index, beta in enumerate(betas):
            log_weights = -beta * energies
            log_weights -= log_weights.max()
            expected = np.exp(log_weights)
            expected /= expected.sum()
            if not np.allclose(
                probabilities[index],
                expected,
                atol=1e-14,
                rtol=1e-13,
            ):
                violations.append(f"probability row {index} is not the exact Gibbs law")

    if not isinstance(expected_target_digests, list) or len(
        expected_target_digests,
    ) != len(betas):
        violations.append("target digest list has the wrong length")
    elif probabilities.shape == (len(betas), 2**16):
        actual_digests = [
            _target_digest(coupling_seed, beta, couplings, probabilities[index])
            for index, beta in enumerate(betas)
        ]
        if actual_digests != expected_target_digests:
            violations.append("target content digests differ from calibration record")
    return violations


def _calibration_and_target_validation(
    repository: Path,
    manifest: CampaignManifest,
) -> tuple[str | None, list[str], list[dict[str, object]], list[str]]:
    calibration_path = repository / manifest.calibration.artifact_path
    if not calibration_path.exists():
        return None, ["calibration artifact is missing"], [], ["no target index"]
    calibration_digest = _sha256_file(calibration_path)
    try:
        payload = json.loads(calibration_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return (
            calibration_digest,
            [f"invalid calibration JSON: {error}"],
            [],
            [
                "no target index",
            ],
        )
    if not isinstance(payload, Mapping):
        return (
            calibration_digest,
            ["calibration must be a JSON object"],
            [],
            [
                "no target index",
            ],
        )

    calibration_violations: list[str] = []
    expected_keys = {
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
    }
    if set(payload) != expected_keys:
        calibration_violations.append("calibration keys do not match schema")
    if payload.get("schema_version") != 1:
        calibration_violations.append("calibration schema_version is not 1")
    if payload.get("campaign_version") != manifest.campaign_version:
        calibration_violations.append("calibration campaign_version differs")
    if payload.get("coupling_seeds") != list(manifest.coupling_seeds):
        calibration_violations.append("calibration coupling seeds differ")
    if not _same_float_sequence(
        payload.get("training_betas"),
        manifest.beta_grid.training,
    ):
        calibration_violations.append("calibration Training grid differs")
    if not _same_float_sequence(
        payload.get("heldout_betas"),
        manifest.beta_grid.held_out,
    ):
        calibration_violations.append("calibration Held-out grid differs")
    expected_interval = [
        manifest.beta_grid.training[0],
        manifest.beta_grid.training[-1],
    ]
    if not _same_float_sequence(
        payload.get("selected_interval", []),
        expected_interval,
    ):
        calibration_violations.append("selected interval differs from manifest grid")
    candidate_records = payload.get("candidate_records")
    if not isinstance(candidate_records, list) or not any(
        isinstance(record, Mapping)
        and record.get("passed") is True
        and _same_float_sequence(record.get("interval"), expected_interval)
        for record in candidate_records or []
    ):
        calibration_violations.append("selected eligible candidate record is absent")
    source = payload.get("source_provenance")
    canonical_sha = (
        source.get("canonical_coupling_sha256") if isinstance(source, Mapping) else None
    )
    actual_canonical_sha = hashlib.sha256(
        periodic_lattice_couplings(666).tobytes(),
    ).hexdigest()
    if canonical_sha != actual_canonical_sha:
        calibration_violations.append("canonical coupling provenance digest differs")

    target_rows: list[dict[str, object]] = []
    target_violations: list[str] = []
    records = payload.get("target_archives")
    if not isinstance(records, list):
        return (
            calibration_digest,
            calibration_violations,
            [],
            [
                "target_archives must be a list",
            ],
        )
    by_seed = {
        record.get("coupling_seed"): record
        for record in records
        if isinstance(record, Mapping)
    }
    if set(by_seed) != set(manifest.coupling_seeds) or len(records) != len(by_seed):
        target_violations.append("target archive index does not match coupling seeds")
    betas = manifest.beta_grid.training + manifest.beta_grid.held_out
    for seed in manifest.coupling_seeds:
        record = by_seed.get(seed)
        if not isinstance(record, Mapping):
            target_violations.append(f"missing target archive record for seed {seed}")
            continue
        try:
            path = _repository_path(repository, record.get("archive"))
        except ValueError as error:
            target_violations.append(f"seed {seed}: {error}")
            continue
        expected_path = (
            calibration_path.parent / "targets" / f"coupling_{seed}.npz"
        ).resolve()
        row_violations = []
        if path != expected_path:
            row_violations.append("archive path differs from frozen campaign layout")
        actual_sha = _sha256_file(path) if path.is_file() else None
        expected_sha = record.get("archive_sha256")
        if actual_sha is None:
            row_violations.append("archive is missing")
        elif actual_sha != expected_sha:
            row_violations.append("archive SHA differs from calibration index")
        if actual_sha is not None:
            row_violations.extend(
                _validate_target_archive(
                    path,
                    seed,
                    betas,
                    record.get("target_digests"),
                ),
            )
        target_rows.append(
            {
                "coupling_seed": seed,
                "path": _relative(path, repository),
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "status": "pass" if not row_violations else "fail",
                "violations": row_violations,
            },
        )
        target_violations.extend(
            f"seed {seed}: {violation}" for violation in row_violations
        )
    return (
        calibration_digest,
        calibration_violations,
        target_rows,
        target_violations,
    )


def _scaling_record_digest(record: Mapping[str, Any]) -> str:
    deterministic = {
        name: value
        for name, value in record.items()
        if name
        not in {
            "runtime_seconds",
            "peak_rss_bytes",
            "peak_rss_increment_bytes",
            "numerical_results_digest",
        }
    }
    reference = deterministic.get("reference")
    if isinstance(reference, Mapping):
        deterministic["reference"] = {
            name: value
            for name, value in reference.items()
            if name != "runtime_seconds"
        }
    return content_digest(deterministic)


def _scaling_provenance_violations(
    repository: Path,
    scaling_path: Path,
    payload: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    provenance = payload.get("provenance")
    required = {
        "code_revision",
        "working_tree_dirty",
        "invocation",
        "source_digests",
        "environment",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != required:
        return ["provenance does not use the exact required schema"]
    revision = provenance["code_revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        violations.append("code_revision is not a full Git commit SHA")
    if not isinstance(provenance["working_tree_dirty"], bool):
        violations.append("working_tree_dirty is not Boolean")

    invocation = provenance["invocation"]
    expected_script = (repository / "scripts/run_estimator_scaling.py").resolve()
    if (
        not isinstance(invocation, list)
        or len(invocation) < 2
        or any(not isinstance(argument, str) or not argument for argument in invocation)
    ):
        violations.append("invocation is not a complete non-empty argv list")
    else:
        invocation_script = Path(invocation[1])
        same_location = invocation_script.resolve() == expected_script
        same_relocatable_path = invocation_script.parts[-2:] == (
            "scripts",
            "run_estimator_scaling.py",
        )
        if not (same_location or same_relocatable_path):
            violations.append("invocation does not name the archived scaling CLI")
        if "--output" not in invocation:
            violations.append("invocation does not declare its output artifact")
        else:
            output_index = invocation.index("--output") + 1
            if output_index >= len(invocation):
                violations.append("invocation has no value for --output")
            else:
                declared_output = Path(invocation[output_index])
                if not declared_output.is_absolute():
                    declared_output = repository / declared_output
                if declared_output.resolve() != scaling_path.resolve():
                    violations.append("invocation --output differs from this artifact")

    source_digests = provenance["source_digests"]
    if not isinstance(source_digests, Mapping) or set(source_digests) != (
        _SCALING_SOURCE_PATHS
    ):
        violations.append("source_digests does not name every required source")
    else:
        snapshot_root = scaling_path.parent / "execution-source-files"
        for source_name, declared_digest in sorted(source_digests.items()):
            try:
                source_path = _repository_path(snapshot_root, source_name)
            except ValueError as error:
                violations.append(f"{source_name}: {error}")
                continue
            if not source_path.is_file():
                violations.append(f"{source_name}: source is missing")
            elif _sha256_file(source_path) != declared_digest:
                violations.append(f"{source_name}: declared digest differs")

    environment = provenance["environment"]
    required_environment = {
        "backend",
        "devices",
        "implementation",
        "jax",
        "jaxlib",
        "machine",
        "numpy",
        "python",
        "release",
        "system",
    }
    if not isinstance(environment, Mapping) or set(environment) != required_environment:
        violations.append("environment does not use the complete runtime schema")
    elif not isinstance(environment["devices"], list) or not environment["devices"]:
        violations.append("environment has no device inventory")

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        violations.append("scaling records are absent")
    else:
        record_digests = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                violations.append(f"scaling record {index} is not an object")
                continue
            expected = record.get("numerical_results_digest")
            actual = _scaling_record_digest(record)
            if expected != actual:
                violations.append(f"scaling record {index} numerical digest differs")
            record_digests.append(expected)
        if payload.get("numerical_results_digest") != content_digest(record_digests):
            violations.append("top-level numerical_results_digest differs")
    return violations


def _blocked_spectrum_violations(
    repository: Path,
    manifest: CampaignManifest,
    store: CampaignStore,
    payload: Mapping[str, Any],
) -> list[str]:
    """Verify the canonical zero-run spectrum preflight for the stopped v1."""
    violations: list[str] = []
    if set(payload) != _BLOCKED_SPECTRUM_KEYS:
        violations.append("blocked spectrum payload schema differs")
    if payload.get("schema_version") != 1:
        violations.append("blocked spectrum schema_version differs")
    if payload.get("evidence_type") != "response_spectrum_production_preflight":
        violations.append("spectrum artifact is not a production preflight")
    if payload.get("status") != "blocked":
        violations.append("spectrum preflight is not blocked")
    if payload.get("campaign_version") != manifest.campaign_version:
        violations.append("spectrum campaign version differs")
    if payload.get("manifest_digest") != manifest.digest:
        violations.append("spectrum manifest digest differs")
    if (
        payload.get("full_training_runs_used") != 0
        or payload.get("budget_pool") != "none"
        or payload.get("budget_accounting")
        != "excluded_from_all_full-training-run_pools"
    ):
        violations.append("spectrum preflight consumed or claimed training budget")
    if payload.get("compatible_checkpoints") != {}:
        violations.append("blocked v1 spectrum preflight claims compatible checkpoints")

    blockers = payload.get("blockers")
    blocker_codes = (
        {
            row.get("code")
            for row in blockers
            if isinstance(row, Mapping) and isinstance(row.get("code"), str)
        }
        if isinstance(blockers, list)
        else set()
    )
    if (
        not isinstance(blockers, list)
        or len(blockers) != len(_BLOCKED_SPECTRUM_CODES)
        or blocker_codes != _BLOCKED_SPECTRUM_CODES
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"code", "detail"}
            or not isinstance(row.get("detail"), str)
            or not row["detail"]
            for row in blockers
        )
    ):
        violations.append("spectrum blocker set differs from the stopped campaign")

    theorem = store.gate_record(Gate.THEOREM_NOVELTY)
    if theorem is None or theorem.decision is not GateDecision.FAILED:
        violations.append("blocked spectrum artifact lacks the failed theorem gate")
    else:
        configuration = {
            "manifest_digest": manifest.digest,
            "fixed_bundle_path": None,
            "phase_bundle_path": None,
            "theorem_verdict": theorem.decision.value,
            "theorem_evidence_digest": theorem.evidence_digest,
            "campaign_root": _relative(store.root, repository),
            "grid_size": 193,
            "compatible_checkpoint_sha256": {},
        }
        if payload.get("configuration_digest") != content_digest(configuration):
            violations.append("spectrum configuration digest is stale or incompatible")
    return violations


def _provenance_rows(
    repository: Path,
    manifest: CampaignManifest,
    store: CampaignStore,
    reservations: Sequence[RunReservation],
    bundles: Sequence[ResultBundle],
    gate_evidence: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    calibration = repository / manifest.calibration.artifact_path
    (
        calibration_digest,
        calibration_violations,
        target_archives,
        target_violations,
    ) = _calibration_and_target_validation(repository, manifest)
    campaign_artifacts = calibration.parent
    scaling_path = campaign_artifacts / "evidence" / "estimator-scaling.json"
    scaling_error = None
    try:
        scaling_payload = (
            read_immutable_json(scaling_path) if scaling_path.exists() else None
        )
    except ArtifactError as error:
        scaling_payload = None
        scaling_error = str(error)
    scaling_records = (
        scaling_payload.get("records", []) if scaling_payload is not None else []
    )
    scaling_violations = (
        _scaling_provenance_violations(
            repository,
            scaling_path,
            scaling_payload,
        )
        if scaling_payload is not None
        else [scaling_error or "scaling artifact is missing"]
    )
    scaling_valid = bool(
        scaling_payload is not None
        and scaling_payload.get("manifest_digest") == manifest.digest
        and scaling_payload.get("full_training_runs_used") == 0
        and scaling_payload.get("budget_accounting")
        == "excluded_from_all_full-training-run_pools"
        and isinstance(scaling_records, list)
        and scaling_records
        and not scaling_violations
        and all(
            isinstance(record, Mapping)
            and record.get("full_training_runs_used") == 0
            and record.get("budget_pool") == "none"
            for record in scaling_records
        )
    )
    spectrum_path = campaign_artifacts / "evidence" / "response-spectrum.json"
    spectrum_applicable = (
        store.root.resolve() == (campaign_artifacts / "campaign").resolve()
    )
    spectrum_error = None
    spectrum_payload = None
    if spectrum_applicable:
        try:
            spectrum_payload = (
                read_immutable_json(spectrum_path) if spectrum_path.exists() else None
            )
        except ArtifactError as error:
            spectrum_error = str(error)
    spectrum_violations = (
        _blocked_spectrum_violations(
            repository,
            manifest,
            store,
            spectrum_payload,
        )
        if spectrum_applicable and spectrum_payload is not None
        else (
            [spectrum_error or "response-spectrum artifact is missing"]
            if spectrum_applicable
            else []
        )
    )
    spectrum_valid = not spectrum_applicable or (
        spectrum_payload is not None and not spectrum_violations
    )
    reservation_ids = {reservation.run.run_id for reservation in reservations}
    orphaned_results = sorted(
        bundle.run.run_id
        for bundle in bundles
        if bundle.run.run_id not in reservation_ids
    )
    gate_count = sum(store.gate_record(gate) is not None for gate in Gate)
    return [
        {
            "check": "manifest_content_digest",
            "status": "pass",
            "expected": manifest.digest,
            "actual": manifest.digest,
        },
        {
            "check": "immutable_campaign_snapshot",
            "status": "pass",
            "path": _relative(store.manifest_path, repository),
            "details": "snapshot parsed and matched by CampaignStore",
        },
        {
            "check": "calibration_artifact_digest",
            "status": (
                "pass"
                if calibration_digest is not None
                and manifest.calibration.artifact_sha256 is not None
                and calibration_digest == manifest.calibration.artifact_sha256
                and not calibration_violations
                else "fail"
            ),
            "expected": manifest.calibration.artifact_sha256,
            "actual": calibration_digest,
            "path": _relative(calibration, repository),
            "violations": calibration_violations,
        },
        {
            "check": "exact_target_archives",
            "status": "pass" if not target_violations else "fail",
            "expected": len(manifest.coupling_seeds),
            "actual": sum(row["status"] == "pass" for row in target_archives),
            "archives": target_archives,
            "violations": target_violations,
        },
        {
            "check": "zero-run_estimator_scaling_evidence",
            "status": "pass" if scaling_valid else "fail",
            "expected": manifest.digest,
            "actual": (
                scaling_payload.get("manifest_digest")
                if scaling_payload is not None
                else None
            ),
            "records": len(scaling_records),
            "artifact_digest": (
                content_digest(scaling_payload) if scaling_payload is not None else None
            ),
            "numerical_results_digest": (
                scaling_payload.get("numerical_results_digest")
                if scaling_payload is not None
                else None
            ),
            "path": _relative(scaling_path, repository),
            "artifact_sha256": (
                _sha256_file(scaling_path) if scaling_path.exists() else None
            ),
            "envelope_payload_sha256": (
                content_digest(scaling_payload) if scaling_payload is not None else None
            ),
            "provenance": (
                scaling_payload.get("provenance")
                if scaling_payload is not None
                else None
            ),
            "violations": scaling_violations,
        },
        {
            "check": "blocked_response_spectrum_evidence",
            "status": "pass" if spectrum_valid else "fail",
            "expected": "blocked_zero_run_preflight",
            "actual": (
                spectrum_payload.get("status")
                if spectrum_payload is not None
                else "not_applicable_to_fixture"
                if not spectrum_applicable
                else None
            ),
            "configuration_digest": (
                spectrum_payload.get("configuration_digest")
                if spectrum_payload is not None
                else None
            ),
            "path": _relative(spectrum_path, repository),
            "artifact_sha256": (
                _sha256_file(spectrum_path) if spectrum_path.exists() else None
            ),
            "violations": spectrum_violations,
            "details": (
                "canonical production evidence"
                if spectrum_applicable
                else "external reduced-fixture campaign; canonical spectrum omitted"
            ),
        },
        {
            "check": "immutable_gate_records",
            "status": (
                "pass"
                if all(row["status"] == "pass" for row in gate_evidence)
                else "fail"
            ),
            "actual": gate_count,
            "details": (
                "all present gate envelopes, archived evidence paths, SHA-256 "
                "digests, terminal decisions, and manifest bindings must agree"
            ),
            "gate_evidence": list(gate_evidence),
        },
        {
            "check": "reservation_result_compatibility",
            "status": "pass" if not orphaned_results else "fail",
            "reservations": len(reservations),
            "results": len(bundles),
            "orphaned_results": orphaned_results,
        },
    ]


def _campaign_status(
    store: CampaignStore,
    provenance: Sequence[Mapping[str, object]],
    leakage: Sequence[Mapping[str, object]],
) -> tuple[str, str | None]:
    if any(row.get("status") != "pass" for row in (*provenance, *leakage)):
        return "rejected_provenance_or_leakage", None
    first_failed = next(
        (
            gate
            for gate in Gate
            if (record := store.gate_record(gate)) is not None
            and record.decision is GateDecision.FAILED
        ),
        None,
    )
    if first_failed is not None:
        return f"stopped_{first_failed.value}_failed", first_failed.value
    records = [store.gate_record(gate) for gate in Gate]
    if all(
        record is not None and record.decision is GateDecision.PASSED
        for record in records
    ):
        return "accepted", None
    return "active_or_incomplete", None


def build_campaign_report(
    repository: Path,
    manifest_path: Path,
    campaign_root: Path,
    *,
    claim_registry_path: Path | None = None,
    manuscript_sources: Sequence[Path] | None = None,
) -> CampaignReport:
    """Load and verify one campaign store, then assemble every report surface."""
    repository = repository.resolve()
    manifest = CampaignManifest.from_json(manifest_path)
    store = CampaignStore(campaign_root, manifest)
    reservations = _load_reservations(store)
    bundles = _load_bundles(store)
    theorem = store.gate_record(Gate.THEOREM_NOVELTY)
    theorem_failed = theorem is not None and theorem.decision is GateDecision.FAILED

    registry = claim_registry_path or repository / "docs" / "claim-evidence-registry.md"
    sources = tuple(
        manuscript_sources
        or (repository / "paper" / "main.tex", repository / "paper" / "supplement.tex")
    )
    claims = claim_support_records(registry)
    gate_evidence = _gate_evidence_rows(store)
    gates = _gate_rows(store, gate_evidence)
    runs = _run_rows(store, reservations, bundles)
    budgets = _budget_rows(
        manifest,
        reservations,
        bundles,
        theorem_failed,
    )
    statistics = _statistics_summary(_paired_records(bundles), theorem_failed)
    figures = _figure_inventory(repository, sources)
    provenance = _provenance_rows(
        repository,
        manifest,
        store,
        reservations,
        bundles,
        gate_evidence,
    )
    leakage = _leakage_rows(manifest, repository, bundles)
    campaign_status, first_failed_gate = _campaign_status(
        store,
        provenance,
        leakage,
    )

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "campaign_status": campaign_status,
        "first_failed_gate": first_failed_gate,
        "provenance_checks": provenance,
        "budget_accounting": budgets,
        "leakage_checks": leakage,
        "gate_decisions": gates,
        "run_table": runs,
        "statistics": statistics,
        "figure_source_inventory": figures,
        "claim_registry": {
            "path": _relative(registry, repository),
            "sha256": _sha256_file(registry),
            "records": list(claims),
        },
        "manuscript_tables": {
            "gate_status": gates,
            "budget_ledger": budgets,
            "run_status": runs,
            "statistical_decisions": statistics,
        },
    }
    return CampaignReport.create(payload)


def _markdown_cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(
    columns: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_markdown_cell(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def render_campaign_report(report: CampaignReport) -> str:
    """Render a deterministic manuscript-facing Markdown report."""
    payload = report.payload
    lines = [
        "# Frozen Quantum campaign report",
        "",
        f"- Campaign: `{payload['campaign_version']}`",
        f"- Manifest digest: `{payload['manifest_digest']}`",
        f"- Report digest: `{report.digest}`",
        f"- Status: `{payload['campaign_status']}`",
        "",
        "## Provenance checks",
        "",
        _markdown_table(
            ("check", "status", "expected", "actual", "details"),
            payload["provenance_checks"],
        ),
        "",
        "## Gate decisions",
        "",
        _markdown_table(
            (
                "gate",
                "status",
                "decision",
                "evidence_status",
                "evidence_path",
                "evidence_digest",
                "reason",
            ),
            payload["gate_decisions"],
        ),
        "",
        "## Non-transferable budget ledger",
        "",
        _markdown_table(
            (
                "pool",
                "ceiling",
                "reserved",
                "completed",
                "failed",
                "remaining",
                "status",
            ),
            payload["budget_accounting"],
        ),
        "",
        "## Declared run matrix",
        "",
        _markdown_table(
            (
                "stage",
                "declared",
                "reserved",
                "completed",
                "failed",
                "status",
                "reason",
            ),
            payload["run_table"],
        ),
        "",
        "## Leakage checks",
        "",
        _markdown_table(
            ("check", "status", "violations", "details"),
            payload["leakage_checks"],
        ),
        "",
        "## Statistical decisions",
        "",
        "```json",
        json.dumps(payload["statistics"], indent=2, sort_keys=True),
        "```",
        "",
        "## Figure-source inventory",
        "",
        _markdown_table(
            ("figure", "output", "output_exists", "status"),
            payload["figure_source_inventory"],
        ),
        "",
        "## Claim-support registry",
        "",
        _markdown_table(
            ("claim_id", "section", "status", "evidence"),
            payload["claim_registry"]["records"],
        ),
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_campaign_report(
    report: CampaignReport,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Atomically regenerate deterministic JSON and Markdown report artifacts."""
    json_path = output_directory / "campaign_report.json"
    markdown_path = output_directory / "campaign_report.md"
    _atomic_write(json_path, canonical_json_bytes(report.to_dict()) + b"\n")
    _atomic_write(markdown_path, render_campaign_report(report).encode("utf-8"))
    return json_path, markdown_path
