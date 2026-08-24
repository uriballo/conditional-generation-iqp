"""Immutable ledger and kill-tree tests for the foundations campaign."""

from pathlib import Path

import pytest

from iqp_cond.foundations_campaign import (
    FoundationsCampaignStore,
    FoundationsGate,
    GateBlockedError,
    GateDecision,
    UndeclaredRunError,
)
from iqp_cond.foundations_protocol import (
    FoundationsManifest,
    FoundationsTuningDecision,
    plan_foundations_primary,
    plan_foundations_representability,
    plan_foundations_runtime,
)
from iqp_cond.protocol import RunIdentity, Stage

MANIFEST_PATH = Path(__file__).parents[1] / "configs" / "arxiv_foundations_v1.json"


@pytest.fixture
def manifest() -> FoundationsManifest:
    return FoundationsManifest.from_json(MANIFEST_PATH)


def _pass(store: FoundationsCampaignStore, gate: FoundationsGate) -> None:
    store.finalize_gate(gate, GateDecision.PASSED, evidence_digest="a" * 64)


def test_store_snapshots_new_manifest_and_blocks_runs_until_predecessors_pass(
    tmp_path,
    manifest,
):
    store = FoundationsCampaignStore(tmp_path, manifest)
    runtime = plan_foundations_runtime(manifest)[0]

    assert store.manifest_snapshot.exists()
    with pytest.raises(GateBlockedError, match="data_ready"):
        store.reserve(runtime)
    _pass(store, FoundationsGate.DATA_READY)
    with pytest.raises(GateBlockedError, match="estimator_calibration"):
        store.reserve(runtime)
    _pass(store, FoundationsGate.ESTIMATOR_CALIBRATION)
    store.reserve(runtime)
    with pytest.raises(ValueError, match="already reserved"):
        store.reserve(runtime)


def test_store_rejects_undeclared_and_partial_reuse(tmp_path, manifest):
    store = FoundationsCampaignStore(tmp_path, manifest)
    _pass(store, FoundationsGate.DATA_READY)
    _pass(store, FoundationsGate.ESTIMATOR_CALIBRATION)
    runtime = plan_foundations_runtime(manifest)[0]
    foreign = RunIdentity.from_dict(
        {**runtime.to_dict(), "condition_scope": "undeclared"},
    )
    with pytest.raises(UndeclaredRunError):
        store.reserve(foreign)

    store.reserve(runtime)
    with pytest.raises(ValueError, match="partial result reuse"):
        store.require_complete(runtime)


def test_failed_gate_is_terminal_and_blocks_every_descendant(tmp_path, manifest):
    store = FoundationsCampaignStore(tmp_path, manifest)
    _pass(store, FoundationsGate.DATA_READY)
    _pass(store, FoundationsGate.ESTIMATOR_CALIBRATION)
    store.finalize_gate(
        FoundationsGate.RUNTIME_FEASIBILITY,
        GateDecision.FAILED,
        evidence_digest="b" * 64,
    )
    run = plan_foundations_representability(manifest)[0]
    with pytest.raises(GateBlockedError, match=r"runtime_feasibility.*failed"):
        store.reserve(run)
    with pytest.raises(ValueError, match="already terminal"):
        _pass(store, FoundationsGate.RUNTIME_FEASIBILITY)


def test_primary_plan_requires_evidence_bound_tuning_decision(tmp_path, manifest):
    store = FoundationsCampaignStore(tmp_path, manifest)
    for gate in (
        FoundationsGate.DATA_READY,
        FoundationsGate.ESTIMATOR_CALIBRATION,
        FoundationsGate.RUNTIME_FEASIBILITY,
        FoundationsGate.REPRESENTABILITY,
    ):
        _pass(store, gate)
    decision = FoundationsTuningDecision.reference(manifest)
    store.finalize_tuning(decision)
    planned = store.planned_runs(Stage.PRIMARY)

    assert planned == plan_foundations_primary(manifest, decision)
    assert len(planned) == 36
    store.reserve(planned[0])
