"""End-to-end campaign acceptance and deterministic-report tests."""

from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from iqp_cond.artifacts import (
    CheckpointProvenance,
    CompletionStatus,
    ParameterAccounting,
    ResultBundle,
    read_immutable_json,
)
from iqp_cond.campaign import (
    CampaignStore,
    Gate,
    GateDecision,
    TuningDecision,
    plan_ancilla,
    plan_primary,
    plan_representability_screen,
    plan_runtime_feasibility,
    plan_tuning,
)
from iqp_cond.campaign_report import (
    _blocked_spectrum_violations,
    _campaign_status,
    _scaling_provenance_violations,
    _validate_target_archive,
    build_campaign_report,
    write_campaign_report,
)
from iqp_cond.campaign_statistics import (
    CONDITION_MODULATED_PHASE,
    FIXED_INPUT,
    NO_CONDITION,
    ORACLE_ADJACENT,
    PairedEvaluationRecord,
    benefit_retention_gate,
    interpolation_gate,
)
from iqp_cond.ising_training import (
    DatasetRole,
    ExactTargetSet,
    RoleTaggedDataset,
    TrainingValidationData,
)
from iqp_cond.protocol import (
    BudgetPool,
    CampaignManifest,
    ModelFamily,
    Stage,
    canonical_json_bytes,
    content_digest,
)

REPOSITORY = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY / "configs" / "quantum_campaign_v1.json"
PRODUCTION_CAMPAIGN = REPOSITORY / "results" / "quantum-campaign-v1" / "campaign"
REPORT_SCRIPT = REPOSITORY / "scripts" / "build_campaign_report.py"
PREPARATION_SCRIPT = REPOSITORY / "scripts" / "prepare_ising_campaign.py"
TIMESTAMP = "2026-08-10T10:00:00+00:00"


def _evidence(name: str) -> str:
    return content_digest({"fixture_evidence": name})


def _gate_evidence_bytes(gate: Gate) -> bytes:
    return f"tiny exact fixture evidence for {gate.value}\n".encode()


def _release(
    store: CampaignStore,
    gate: Gate,
    decision: GateDecision = GateDecision.PASSED,
    *,
    tuning_decision: TuningDecision | None = None,
) -> None:
    evidence_bytes = _gate_evidence_bytes(gate)
    evidence = hashlib.sha256(evidence_bytes).hexdigest()
    if tuning_decision is not None:
        assert tuning_decision.evidence_digest == evidence
    release = (
        store.release_gate
        if gate is Gate.THEOREM_NOVELTY or decision is GateDecision.FAILED
        else store._release_finalized_gate
    )
    release(
        gate,
        decision,
        evidence_digest=evidence,
        reason=f"tiny exact fixture decision for {gate.value}",
        decided_at=TIMESTAMP,
        tuning_decision=tuning_decision,
    )
    source = store.root / "fixture-evidence-sources" / f"{gate.value}.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(evidence_bytes)
    store.archive_gate_evidence(gate, source)


def _accounting(run) -> ParameterAccounting:
    if run.stage is not Stage.ANCILLA:
        return ParameterAccounting(16, 136, 136, 0)
    total_qubits = 16 + run.ancilla_rung
    parameters = total_qubits * (total_qubits + 1) // 2
    if run.model_family is ModelFamily.FIXED_INPUT:
        return ParameterAccounting(
            total_qubits,
            parameters,
            parameters,
            run.ancilla_rung,
        )
    return ParameterAccounting(16, parameters, parameters, 0)


def _bundle(
    run,
    data: TrainingValidationData,
    targets: tuple[ExactTargetSet, ExactTargetSet],
    *,
    raw_metrics: dict[str, object] | None = None,
) -> ResultBundle:
    checkpoint_digest = _evidence(f"checkpoint-{run.run_id}")
    checkpoint = CheckpointProvenance(
        source_run_id=run.run_id,
        manifest_digest=run.configuration_digest,
        selection_rule="lowest Validation Conditional MMD",
        selected_step=2,
        validation_metric="conditional_mmd",
        validation_value=0.125,
        artifact_path=f"checkpoints/{run.run_id}.npz",
        artifact_sha256=checkpoint_digest,
    )
    if run.condition_scope == "all_training_betas":
        dataset_hashes = data.dataset_hashes()
    else:
        dataset_hashes = {
            "training_beta_0": data.training[0].digest,
            "validation_beta_0": data.validation[0].digest,
        }
    if run.stage is Stage.REPRESENTABILITY_SCREEN:
        dataset_hashes["exact_targets"] = targets[0].source_sha256
    if run.stage in {Stage.PRIMARY, Stage.ANCILLA}:
        dataset_hashes["exact_targets"] = targets[1].source_sha256
    return ResultBundle.create(
        run=run,
        manifest_digest=run.configuration_digest,
        code_revision="tiny-fixture-revision",
        environment={"python": "test", "backend": "exact-two-qubit"},
        invocation=("tiny-campaign", "--run-id", run.run_id),
        started_at=TIMESTAMP,
        completed_at="2026-08-10T10:00:01+00:00",
        runtime_seconds=1.0,
        peak_memory_bytes=4_096,
        parameter_accounting=_accounting(run),
        dataset_hashes=dataset_hashes,
        estimator_randomness={"operators": 11, "reference": 13, "validation": 17},
        checkpoint=checkpoint,
        raw_metrics=raw_metrics or {"exact_tvd": 0.2},
        completion_status=CompletionStatus.COMPLETED,
    )


def _tiny_exact_data() -> tuple[
    TrainingValidationData,
    tuple[ExactTargetSet, ExactTargetSet],
]:
    manifest = CampaignManifest.from_json(MANIFEST_PATH)
    training_samples = np.asarray([[0, 0], [0, 1]], dtype=np.int8)
    validation_samples = np.asarray([[1, 0], [1, 1]], dtype=np.int8)
    training = tuple(
        RoleTaggedDataset(
            DatasetRole.TRAINING,
            666,
            beta,
            101 + 2 * index,
            training_samples,
        )
        for index, beta in enumerate(manifest.beta_grid.training)
    )
    validation = tuple(
        RoleTaggedDataset(
            DatasetRole.VALIDATION,
            666,
            beta,
            102 + 2 * index,
            validation_samples,
        )
        for index, beta in enumerate(manifest.beta_grid.training)
    )
    data = TrainingValidationData(training, validation)
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    training_targets = ExactTargetSet(
        DatasetRole.TRAINING,
        666,
        manifest.beta_grid.training,
        np.tile(probabilities, (5, 1)),
        _evidence("tiny-training-target"),
    )
    heldout_targets = ExactTargetSet(
        DatasetRole.HELD_OUT,
        666,
        manifest.beta_grid.held_out,
        np.tile(probabilities[::-1], (4, 1)),
        _evidence("tiny-heldout-target"),
    )
    return data, (training_targets, heldout_targets)


def _paired_rows() -> tuple[PairedEvaluationRecord, ...]:
    return tuple(
        PairedEvaluationRecord(
            coupling_instance=f"tiny-{instance}",
            optimizer_seed=seed,
            condition=0.4,
            method_tvds={
                FIXED_INPUT: 0.3,
                CONDITION_MODULATED_PHASE: 0.2,
                NO_CONDITION: 0.6,
                ORACLE_ADJACENT: 0.5,
            },
        )
        for instance in range(2)
        for seed in range(2)
    )


def _paired_payload(
    records: tuple[PairedEvaluationRecord, ...],
) -> list[dict[str, object]]:
    return [
        {
            "coupling_instance": record.coupling_instance,
            "optimizer_seed": record.optimizer_seed,
            "condition": record.condition,
            "split": record.split,
            "method_tvds": dict(record.method_tvds),
        }
        for record in records
    ]


@pytest.fixture
def accepted_tiny_campaign(tmp_path):
    manifest = CampaignManifest.from_json(MANIFEST_PATH)
    data, targets = _tiny_exact_data()
    store = CampaignStore(tmp_path / "campaign", manifest)

    assert set(map(tuple, data.training[0].samples)).isdisjoint(
        set(map(tuple, data.validation[0].samples)),
    )
    assert targets[0].probabilities.shape == (5, 4)
    assert targets[1].probabilities.shape == (4, 4)

    for gate in (
        Gate.THEOREM_NOVELTY,
        Gate.TARGET_CALIBRATION,
        Gate.SAMPLING_QUALITY,
    ):
        _release(store, gate)

    runtime = plan_runtime_feasibility(manifest)[0]
    store.execute_one(
        runtime,
        lambda run: _bundle(run, data, targets),
        reserved_at=TIMESTAMP,
    )
    _release(store, Gate.RUNTIME_FEASIBILITY)

    representability = plan_representability_screen(manifest)[0]
    store.execute_one(
        representability,
        lambda run: _bundle(run, data, targets, raw_metrics={"exact_tvd": 0.2}),
        reserved_at=TIMESTAMP,
    )
    _release(store, Gate.REPRESENTABILITY)

    tuning = plan_tuning(manifest)[0]
    store.execute_one(
        tuning,
        lambda run: _bundle(run, data, targets),
        reserved_at=TIMESTAMP,
    )
    tuning_evidence = hashlib.sha256(
        _gate_evidence_bytes(Gate.TUNING_PROTOCOL),
    ).hexdigest()
    tuning_decision = TuningDecision.create(
        manifest,
        {
            family: manifest.training_protocol.candidates_for(family)[0].candidate_id
            for family in (
                ModelFamily.FIXED_INPUT,
                ModelFamily.CONDITION_MODULATED_PHASE,
                ModelFamily.NO_CONDITION,
            )
        },
        evidence_digest=tuning_evidence,
    )
    _release(store, Gate.TUNING_PROTOCOL, tuning_decision=tuning_decision)

    paired = _paired_rows()
    primary = plan_primary(manifest, tuning_decision)[0]
    store.execute_one(
        primary,
        lambda run: _bundle(
            run,
            data,
            targets,
            raw_metrics={
                "exact_tvd": 0.3,
                "paired_evaluations": _paired_payload(paired),
            },
        ),
        reserved_at=TIMESTAMP,
    )
    retention = benefit_retention_gate(paired)
    interpolation = interpolation_gate(paired)
    assert retention.passed
    assert interpolation.passed
    _release(store, Gate.BENEFIT_RETENTION)
    _release(store, Gate.INTERPOLATION)
    _release(store, Gate.ANCILLA_EXTENSION)

    ancilla = plan_ancilla(manifest, tuning_decision)[0]
    store.execute_one(
        ancilla,
        lambda run: _bundle(run, data, targets),
        reserved_at=TIMESTAMP,
    )
    return manifest, store, data, targets, paired


def test_tiny_exact_acceptance_roundtrips_bundles_statistics_and_gates(
    accepted_tiny_campaign,
):
    manifest, store, data, targets, paired = accepted_tiny_campaign

    assert data.training[0].role is DatasetRole.TRAINING
    assert data.validation[0].role is DatasetRole.VALIDATION
    assert targets[1].role is DatasetRole.HELD_OUT
    assert len(list(store.artifacts.runs_directory.glob("*.json"))) == 5
    for path in store.artifacts.runs_directory.glob("*.json"):
        bundle = store.result(path.stem)
        assert bundle is not None
        assert ResultBundle.from_dict(bundle.to_dict()) == bundle
    assert benefit_retention_gate(paired).passed
    assert interpolation_gate(paired).passed
    assert store.gate_record(Gate.ANCILLA_EXTENSION).decision is GateDecision.PASSED
    assert store.used_budget(BudgetPool.FEASIBILITY_REPRESENTABILITY) == 2
    assert store.used_budget(BudgetPool.TUNING) == 1
    assert store.used_budget(BudgetPool.PRIMARY) == 1
    assert store.used_budget(BudgetPool.ANCILLA) == 1
    assert manifest.digest == store.manifest.digest


def test_tiny_campaign_report_is_complete_and_deterministic_in_clean_outputs(
    accepted_tiny_campaign,
    tmp_path,
):
    _, store, *_ = accepted_tiny_campaign
    first = build_campaign_report(REPOSITORY, MANIFEST_PATH, store.root)
    second = build_campaign_report(REPOSITORY, MANIFEST_PATH, store.root)
    first_paths = write_campaign_report(first, tmp_path / "first")
    second_paths = write_campaign_report(second, tmp_path / "second")

    assert first.digest == second.digest
    assert first.to_dict() == second.to_dict()
    assert first_paths[0].read_bytes() == second_paths[0].read_bytes()
    assert first_paths[1].read_bytes() == second_paths[1].read_bytes()
    payload = first.payload
    assert payload["statistics"]["status"] == "computed"
    assert payload["statistics"]["benefit_retention"]["passed"]
    assert payload["statistics"]["interpolation"]["passed"]
    assert payload["campaign_status"] == "accepted"
    assert all(row["status"] == "pass" for row in payload["leakage_checks"])
    assert all(row["status"] == "pass" for row in payload["provenance_checks"])
    assert any(
        row["stage"] == "primary" and row["completed"] == 1
        for row in payload["run_table"]
    )
    document = json.loads(first_paths[0].read_text())
    assert document["report_digest"] == first.digest


def test_production_report_discloses_terminal_stop_and_zero_downstream_runs():
    report = build_campaign_report(
        REPOSITORY,
        MANIFEST_PATH,
        PRODUCTION_CAMPAIGN,
    )
    payload = report.payload

    assert payload["campaign_status"] == "stopped_theorem_novelty_failed"
    assert payload["first_failed_gate"] == "theorem_novelty"
    assert payload["gate_decisions"][0]["status"] == "failed"
    assert all(row["status"] == "blocked" for row in payload["gate_decisions"][1:])
    assert all(
        row["status"] == "blocked"
        and row["reserved"] == row["completed"] == row["failed"] == 0
        for row in payload["run_table"]
    )
    assert all(
        row["reserved"] == row["completed"] == row["failed"] == 0
        for row in payload["budget_accounting"]
    )
    assert payload["statistics"] == {
        "status": "blocked",
        "paired_cells": 0,
        "benefit_retention": None,
        "interpolation": None,
        "reason": "theorem novelty failed before primary evaluation",
    }
    assert len(payload["claim_registry"]["records"]) == 37
    assert all(row["status"] == "pass" for row in payload["provenance_checks"])
    assert all(row["status"] == "pass" for row in payload["leakage_checks"])
    assert all(row["status"] == "pass" for row in payload["figure_source_inventory"])


def test_missing_gate_evidence_binding_prevents_an_accepted_report(
    accepted_tiny_campaign,
):
    _, store, *_ = accepted_tiny_campaign
    binding = (
        store.gate_evidence_bindings_directory / f"{Gate.BENEFIT_RETENTION.value}.json"
    )
    binding.unlink()

    payload = build_campaign_report(REPOSITORY, MANIFEST_PATH, store.root).payload

    assert payload["campaign_status"] == "rejected_provenance_or_leakage"
    gate_row = next(
        row
        for row in payload["gate_decisions"]
        if row["gate"] == Gate.BENEFIT_RETENTION.value
    )
    assert gate_row["decision"] == "passed"
    assert gate_row["evidence_status"] == "fail"
    provenance = next(
        row
        for row in payload["provenance_checks"]
        if row["check"] == "immutable_gate_records"
    )
    assert provenance["status"] == "fail"


def test_generic_dataset_hash_keys_fail_leakage_and_campaign_acceptance(
    accepted_tiny_campaign,
):
    _, store, *_ = accepted_tiny_campaign
    result_path = next(store.artifacts.runs_directory.glob("*.json"))
    envelope = json.loads(result_path.read_text())
    hashes = envelope["payload"]["dataset_hashes"]
    training = next(
        digest for name, digest in hashes.items() if name.startswith("training_beta_")
    )
    validation = next(
        digest for name, digest in hashes.items() if name.startswith("validation_beta_")
    )
    envelope["payload"]["dataset_hashes"] = {
        "training": training,
        "validation": validation,
    }
    envelope["payload_sha256"] = content_digest(envelope["payload"])
    result_path.write_bytes(canonical_json_bytes(envelope) + b"\n")

    payload = build_campaign_report(REPOSITORY, MANIFEST_PATH, store.root).payload

    assert payload["campaign_status"] == "rejected_provenance_or_leakage"
    split_schema = next(
        row
        for row in payload["leakage_checks"]
        if row["check"] == "dataset_split_key_schema"
    )
    assert split_schema["status"] == "fail"


def test_campaign_preparation_is_immutable_and_version_aware(tmp_path):
    manifest = CampaignManifest.from_json(MANIFEST_PATH)
    preparation = runpy.run_path(str(REPOSITORY / "scripts/prepare_ising_campaign.py"))
    atomic_json = preparation["_atomic_json"]
    atomic_npz = preparation["_atomic_npz"]
    assert_output_version = preparation["_assert_output_version"]
    repository_relative = preparation["_repository_relative"]
    json_path = tmp_path / "record.json"
    npz_path = tmp_path / "record.npz"
    atomic_json(json_path, {"campaign_version": manifest.campaign_version})
    original_json = json_path.read_bytes()
    atomic_json(json_path, {"campaign_version": manifest.campaign_version})
    assert json_path.read_bytes() == original_json
    with pytest.raises(RuntimeError, match="refusing to rewrite"):
        atomic_json(json_path, {"campaign_version": "another_campaign"})

    values = np.asarray([1.0, 2.0], dtype=np.float64)
    first_digest = atomic_npz(npz_path, values=values)
    assert atomic_npz(npz_path, values=values.copy()) == first_digest
    with pytest.raises(RuntimeError, match="refusing to rewrite"):
        atomic_npz(npz_path, values=values + 1.0)

    output = tmp_path / "versioned"
    output.mkdir()
    (output / "calibration.json").write_text(
        json.dumps({"campaign_version": "another_campaign"}),
    )
    with pytest.raises(RuntimeError, match="another campaign version"):
        assert_output_version(manifest, output)
    assert repository_relative(REPOSITORY / "results/example.npz") == (
        "results/example.npz"
    )
    with pytest.raises(ValueError, match="inside the repository"):
        repository_relative(tmp_path / "outside.npz")


def test_preparation_rejects_sampler_arguments_outside_frozen_manifest():
    manifest = CampaignManifest.from_json(MANIFEST_PATH)
    preparation = runpy.run_path(str(REPOSITORY / "scripts/prepare_ising_campaign.py"))
    validate_sampler_configuration = preparation["_validate_sampler_configuration"]
    protocol = manifest.sampler_protocol

    validate_sampler_configuration(
        manifest,
        n_chains=protocol.n_chains,
        burn_in=protocol.burn_in_sweeps,
        thinning=protocol.thinning_sweeps,
    )
    with pytest.raises(ValueError, match="frozen manifest protocol"):
        validate_sampler_configuration(
            manifest,
            n_chains=protocol.n_chains,
            burn_in=protocol.burn_in_sweeps,
            thinning=protocol.thinning_sweeps // 2,
        )


def test_preparation_cli_refuses_before_target_or_dataset_work_after_novelty_fail():
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARATION_SCRIPT),
            "--manifest",
            str(MANIFEST_PATH),
            "--campaign-root",
            str(PRODUCTION_CAMPAIGN),
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "passed theorem_novelty gate (current: failed)" in completed.stderr


def test_exact_target_content_validation_detects_rehashed_wrong_probabilities(
    tmp_path,
):
    manifest = CampaignManifest.from_json(MANIFEST_PATH)
    calibration = json.loads(
        (REPOSITORY / manifest.calibration.artifact_path).read_text(),
    )
    record = next(
        item for item in calibration["target_archives"] if item["coupling_seed"] == 666
    )
    source = REPOSITORY / record["archive"]
    copied = tmp_path / source.name
    shutil.copyfile(source, copied)
    betas = manifest.beta_grid.training + manifest.beta_grid.held_out

    assert not _validate_target_archive(
        copied,
        666,
        betas,
        record["target_digests"],
    )

    with np.load(copied, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    probabilities = arrays["probabilities"].copy()
    probabilities[0, 0] += 1e-4
    probabilities[0] /= probabilities[0].sum()
    arrays["probabilities"] = probabilities
    with copied.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    violations = _validate_target_archive(
        copied,
        666,
        betas,
        record["target_digests"],
    )
    assert any("exact Gibbs law" in violation for violation in violations)
    assert any("content digests" in violation for violation in violations)


def test_scaling_provenance_verifies_envelope_sources_and_numerical_digests():
    path = REPOSITORY / "results/quantum-campaign-v1/evidence/estimator-scaling.json"
    payload = read_immutable_json(path)

    assert not _scaling_provenance_violations(REPOSITORY, path, payload)

    altered = json.loads(json.dumps(payload))
    altered["provenance"]["source_digests"]["src/iqp_cond/estimator_scaling.py"] = (
        "0" * 64
    )
    altered["records"][0]["active_generator_count"] += 1
    violations = _scaling_provenance_violations(REPOSITORY, path, altered)
    assert any("declared digest differs" in violation for violation in violations)
    assert any("numerical digest differs" in violation for violation in violations)


def test_blocked_spectrum_preflight_is_bound_to_the_actual_failed_gate():
    manifest = CampaignManifest.from_json(MANIFEST_PATH)
    path = REPOSITORY / "results/quantum-campaign-v1/evidence/response-spectrum.json"
    payload = read_immutable_json(path)
    store = CampaignStore(PRODUCTION_CAMPAIGN, manifest)

    assert not _blocked_spectrum_violations(REPOSITORY, manifest, store, payload)

    altered = json.loads(json.dumps(payload))
    altered["configuration_digest"] = "0" * 64
    altered["blockers"] = altered["blockers"][:-1]
    violations = _blocked_spectrum_violations(
        REPOSITORY,
        manifest,
        store,
        altered,
    )
    assert any("configuration digest" in violation for violation in violations)
    assert any("blocker set" in violation for violation in violations)


def test_provenance_failure_takes_priority_over_an_historical_failed_gate():
    manifest = CampaignManifest.from_json(MANIFEST_PATH)
    store = CampaignStore(PRODUCTION_CAMPAIGN, manifest)

    status, first_failed = _campaign_status(
        store,
        ({"status": "fail"},),
        ({"status": "pass"},),
    )

    assert status == "rejected_provenance_or_leakage"
    assert first_failed is None


def test_report_cli_regenerates_byte_identical_production_artifacts(tmp_path):
    outputs = []
    digests = []
    for name in ("first", "second"):
        destination = tmp_path / name
        completed = subprocess.run(
            [
                sys.executable,
                str(REPORT_SCRIPT),
                "--repository",
                str(REPOSITORY),
                "--manifest",
                str(MANIFEST_PATH),
                "--campaign-root",
                str(PRODUCTION_CAMPAIGN),
                "--output",
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        digests.append(
            next(
                line.split("=", 1)[1]
                for line in completed.stdout.splitlines()
                if line.startswith("campaign_report_digest=")
            ),
        )
        outputs.append(
            (
                (destination / "campaign_report.json").read_bytes(),
                (destination / "campaign_report.md").read_bytes(),
            ),
        )

    assert digests[0] == digests[1]
    assert outputs[0] == outputs[1]
