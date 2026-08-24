from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import iqp_cond.spectrum_audit as spectrum_module
from iqp_cond.artifacts import read_immutable_json
from iqp_cond.campaign import CampaignStore, Gate, GateDecision
from iqp_cond.protocol import CampaignManifest, ModelFamily
from iqp_cond.spectrum_audit import (
    FrozenCheckpoint,
    SpectrumAuditError,
    archive_spectrum_report,
    describe_unresolved_mass_relation,
    periodic_spectral_mass,
    production_spectrum_preflight,
    project_onto_frequency_support,
    recover_periodic_spectrum,
    synthetic_spectrum_fixture,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_dense_multivariate_grid_recovers_known_frequency_support() -> None:
    size = 24
    axis = 2.0 * np.pi * np.arange(size) / size
    first, second = np.meshgrid(axis, axis, indexing="ij")
    values = 0.5 + 0.2 * np.cos(first) + 0.3 * np.sin(second)

    spectrum = recover_periodic_spectrum(values)
    recovered = {tuple(mode["frequency"]) for mode in spectrum["modes"]}

    assert recovered == {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
    assert spectrum["significant_mode_count"] == 5
    assert math.isclose(spectrum["parseval_power"], np.mean(values**2))


def test_injected_out_of_support_mass_is_detected() -> None:
    size = 32
    axis = 2.0 * np.pi * np.arange(size) / size
    values = np.cos(axis) + 0.3 * np.cos(3.0 * axis)
    allowed = np.array([[-1], [0], [1]])

    mass = periodic_spectral_mass(values, allowed)

    assert math.isclose(mass["outside_power"], 0.3**2 / 2.0, rel_tol=1e-12)
    assert {tuple(mode["frequency"]) for mode in mass["out_of_support_modes"]} == {
        (-3,),
        (3,),
    }


def test_finite_dictionary_projection_accepts_noninteger_frequencies() -> None:
    points = np.linspace(-1.0, 1.0, 101)
    values = np.cos(1.5 * points) + 0.2 * np.sin(0.5 * points)
    support = np.array([[-1.5], [1.5], [-0.5], [0.5]])

    projection = project_onto_frequency_support(points, values, support)

    assert projection["design_rank"] == 4
    assert projection["reconstruction_rmse"] < 1e-12
    assert projection["unresolved_power_fraction"] < 1e-24


def test_unresolved_mass_relation_is_explicitly_descriptive() -> None:
    relation = describe_unresolved_mass_relation(
        np.array([0.01, 0.04, 0.09]),
        np.array([0.1, 0.2, 0.3]),
    )

    assert relation["status"] == "descriptive"
    assert relation["pearson_correlation"] > 0.9
    assert "not causal" in relation["interpretation"]
    assert "trainability" in relation["interpretation"]


def test_synthetic_fixture_compares_fixed_and_phase_supports() -> None:
    report = synthetic_spectrum_fixture(grid_size=24, injected_amplitude=0.25)
    comparison = report["comparison"]
    fixed = comparison["fixed_input"]
    phase = comparison["condition_modulated_phase"]

    assert report["status"] == "synthetic_fixture_completed"
    assert report["full_training_runs_used"] == 0
    assert report["budget_pool"] == "none"
    assert report["fixture_truth"]["absolute_power_error"] < 1e-15
    assert report["fixture_truth"]["interpolation_point_count"] > 0
    assert fixed["target_mass_outside_allowed_support"]["outside_power"] > 0.0
    assert phase["target_mass_outside_allowed_support"]["outside_power"] < 1e-24
    assert fixed["interpolation_rmse"] > 0.0
    assert phase["interpolation_rmse"] == 0.0
    assert report["unresolved_mass_error_relation"]["pearson_correlation"] > 0.9


def test_production_preflight_blocks_without_released_checkpoints() -> None:
    report = production_spectrum_preflight(
        manifest_path=REPOSITORY_ROOT / "configs/quantum_campaign_v1.json",
        repository_root=REPOSITORY_ROOT,
        fixed_bundle_path=None,
        phase_bundle_path=None,
    )
    blocker_codes = {blocker["code"] for blocker in report["blockers"]}

    assert report["status"] == "blocked"
    assert report["full_training_runs_used"] == 0
    assert blocker_codes == {
        "theorem_novelty_failed",
        "fixed_input_checkpoint_missing",
        "condition_modulated_phase_checkpoint_missing",
    }
    assert report["compatible_checkpoints"] == {}
    assert "comparison" not in report
    assert "no synthetic response" in report["interpretation"]


def test_production_preflight_reads_missing_gate_instead_of_hardcoding_failure(
    tmp_path,
) -> None:
    report = production_spectrum_preflight(
        manifest_path=REPOSITORY_ROOT / "configs/quantum_campaign_v1.json",
        repository_root=REPOSITORY_ROOT,
        campaign_root=tmp_path / "campaign-without-gates",
        fixed_bundle_path=None,
        phase_bundle_path=None,
    )

    assert report["status"] == "blocked"
    assert {blocker["code"] for blocker in report["blockers"]} == {
        "theorem_novelty_missing",
        "fixed_input_checkpoint_missing",
        "condition_modulated_phase_checkpoint_missing",
    }


def test_passed_actual_gate_executes_checkpoint_audit_instead_of_ready_preflight(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = CampaignManifest.from_json(
        REPOSITORY_ROOT / "configs/quantum_campaign_v1.json",
    )
    campaign_root = tmp_path / "campaign"
    store = CampaignStore(campaign_root, manifest)
    store.release_gate(
        Gate.THEOREM_NOVELTY,
        GateDecision.PASSED,
        evidence_digest="0" * 64,
        reason="fixture novelty passed",
    )
    generators = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int8,
    )

    def fake_load(_path, *, expected_family, **_kwargs):
        return FrozenCheckpoint(
            family=expected_family,
            coupling_seed=666,
            optimizer_seed=0,
            manifest_digest=manifest.digest,
            source_run_id=("1" if expected_family is ModelFamily.FIXED_INPUT else "2")
            * 64,
            checkpoint_sha256=(
                "3" if expected_family is ModelFamily.FIXED_INPUT else "4"
            )
            * 64,
            parameters=np.zeros(3),
            generators=generators,
            phase_feature=(
                None
                if expected_family is ModelFamily.FIXED_INPUT
                else "common_affine_multiplier"
            ),
        )

    calls = []

    def fake_evaluate(_manifest, fixed, phase, *, grid_size):
        calls.append((fixed.family, phase.family, grid_size))
        return {
            "schema_version": 1,
            "evidence_type": "response_spectrum_production",
            "status": "completed",
            "full_training_runs_used": 0,
        }

    monkeypatch.setattr(spectrum_module, "load_frozen_checkpoint", fake_load)
    monkeypatch.setattr(spectrum_module, "evaluate_production_spectrum", fake_evaluate)
    report = production_spectrum_preflight(
        manifest_path=REPOSITORY_ROOT / "configs/quantum_campaign_v1.json",
        repository_root=tmp_path,
        campaign_root=campaign_root,
        fixed_bundle_path=tmp_path / "fixed.json",
        phase_bundle_path=tmp_path / "phase.json",
        grid_size=193,
    )

    assert report["status"] == "completed"
    assert calls == [
        (
            ModelFamily.FIXED_INPUT,
            ModelFamily.CONDITION_MODULATED_PHASE,
            193,
        ),
    ]
    assert report["theorem_gate"]["decision"] == "passed"
    assert set(report["compatible_checkpoints"]) == {
        "fixed_input",
        "condition_modulated_phase",
    }


def test_spectrum_report_is_content_verified(tmp_path) -> None:
    report = synthetic_spectrum_fixture(grid_size=16, injected_amplitude=0.2)
    destination = tmp_path / "spectrum.json"

    digest = archive_spectrum_report(destination, report)

    assert len(digest) == 64
    assert read_immutable_json(destination) == report


def test_periodic_grid_rejects_noninteger_allowed_modes() -> None:
    with pytest.raises(SpectrumAuditError, match="must be integers"):
        periodic_spectral_mass(
            np.ones(8),
            np.array([[0.5]]),
        )


def test_periodic_grid_rejects_aliased_allowed_support() -> None:
    with pytest.raises(SpectrumAuditError, match="alias-free"):
        periodic_spectral_mass(
            np.ones(8),
            np.array([[4]]),
        )


def test_production_cli_archives_and_returns_clear_blocked_status(
    tmp_path,
) -> None:
    output = tmp_path / "blocked-spectrum.json"
    environment = dict(os.environ)
    python_path = str(REPOSITORY_ROOT / "src")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/audit_response_spectrum.py"),
            "--manifest",
            str(REPOSITORY_ROOT / "configs/quantum_campaign_v1.json"),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    archived = read_immutable_json(output)
    assert completed.returncode == 2
    assert summary["status"] == "blocked"
    assert archived["status"] == "blocked"
    assert archived["full_training_runs_used"] == 0
    assert "comparison" not in archived
