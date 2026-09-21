from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.scalar_gaussian_translation import (
    analysis,
    campaign,
    production,
)

PROJECT = Path(__file__).resolve().parents[1] / "scalar_gaussian_translation"
MANIFEST = PROJECT / "manifests" / "production_proposal.json"
REVIEW = PROJECT / "review"


def _clean_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    state = dict(production.runner.repo_state())
    state["paper_dirty"] = False
    monkeypatch.setattr(production.runner, "repo_state", lambda: state)
    return state


def _create_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict]:
    _clean_state(monkeypatch)
    output = tmp_path / "authorization.json"
    payload = production.create_authorization(
        MANIFEST,
        PROJECT / "manifests" / "audit.json",
        REVIEW / "audit_attestation.json",
        REVIEW / "audit_rows.csv",
        REVIEW / "audit_gate_check.json",
        REVIEW / "source_equivalence.json",
        REVIEW / "audit_provenance_root.json",
        REVIEW / "frozen_gates.json",
        output,
        "Explicit test authorization.",
    )
    return output, payload


def test_canary_is_predeclared_and_covers_required_operational_cases() -> None:
    payload, jobs = campaign._load_manifest(MANIFEST)
    production._assert_production_manifest(payload, jobs)
    canary = production.select_canary_jobs(jobs)

    assert len(canary) == 4
    assert {job.encoder for job in canary} == {
        "broadcast_affine",
        "qubit_affine",
        "anti_block",
        "condition_blind",
    }
    assert any(job.budget_mode == "fixed_total" for job in canary)
    assert any(
        job.anchor_count == 5 and job.budget_mode == "matched_exposure"
        for job in canary
    )
    assert len({job.job_id for job in canary}) == 4


def test_authorization_binds_clean_checkout_and_all_reviewed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, created = _create_authorization(tmp_path, monkeypatch)
    manifest, jobs, validated = production.validate_authorization(MANIFEST, output)

    assert manifest["job_count"] == len(jobs) == 3120
    assert created["content_sha256"] == validated["content_sha256"]
    assert created["authorized_repository_state"]["paper_dirty"] is False
    assert len(created["canary_jobs"]) == 4
    assert created["matrix"]["optimizer_seeds"] == list(range(501, 511))
    assert created["artifacts"]["audit_attestation"]["content_sha256"]
    assert created["artifacts"]["source_equivalence"]["content_sha256"]
    assert created["artifacts"]["provenance_root"]["content_sha256"]


def test_authorization_rejects_dirty_state_and_coordinated_decision_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, payload = _create_authorization(tmp_path, monkeypatch)
    dirty = dict(payload["authorized_repository_state"])
    dirty["paper_dirty"] = True
    monkeypatch.setattr(production.runner, "repo_state", lambda: dirty)
    with pytest.raises(RuntimeError, match="clean repository"):
        production.validate_authorization(MANIFEST, output)

    monkeypatch.setattr(
        production.runner,
        "repo_state",
        lambda: payload["authorized_repository_state"],
    )
    tampered = json.loads(output.read_text())
    tampered["decision"] = "denied"
    tampered["content_sha256"] = analysis._content_digest(tampered)
    output.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="decision is invalid"):
        production.validate_authorization(MANIFEST, output)


def test_legacy_campaign_runner_still_rejects_production() -> None:
    payload, _ = campaign._load_manifest(MANIFEST)
    with pytest.raises(RuntimeError, match="cannot be run"):
        campaign._assert_runnable(payload, MANIFEST)
