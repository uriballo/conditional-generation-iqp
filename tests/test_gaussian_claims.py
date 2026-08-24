"""Boundary tests for Gaussian claim finalization and source packaging."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from iqp_cond.artifacts import write_immutable_json

REPOSITORY = Path(__file__).parents[1]
FINALIZER = runpy.run_path(
    REPOSITORY / "scripts" / "finalize_gaussian_claims.py",
    run_name="gaussian_claim_finalizer_test",
)
ARCHIVER = runpy.run_path(
    REPOSITORY / "scripts" / "build_gaussian_arxiv_archive.py",
    run_name="gaussian_archiver_test",
)


def _interval(lower: float, upper: float) -> dict[str, float]:
    return {"estimate": (lower + upper) / 2.0, "lower": lower, "upper": upper}


def _cell(
    lower: float,
    upper: float,
    *,
    n_qubits: int = 8,
    width_factor: float = 0.35,
) -> dict[str, object]:
    interval = _interval(lower, upper)
    return {
        "n_qubits": n_qubits,
        "width_factor": width_factor,
        "paired_intervals": {
            "training_no_condition_minus_conditioned_profile_tvd": interval,
            "training_conditioned_minus_single_peak_profile_tvd": interval,
            "heldout_lookup_minus_conditioned_profile_tvd": interval,
            "heldout_mixture_minus_conditioned_profile_tvd": interval,
            "training_no_condition_minus_conditioned_bitstring_tvd": interval,
            "heldout_lookup_minus_conditioned_bitstring_tvd": interval,
        },
        "cell_wins": {
            name: {"wins": 1, "total": 1}
            for name in (
                "training_vs_no_condition_profile_tvd",
                "heldout_vs_lookup_profile_tvd",
                "heldout_vs_mixture_profile_tvd",
                "training_vs_no_condition_bitstring_tvd",
                "heldout_vs_lookup_bitstring_tvd",
            )
        },
    }


def test_claim_status_boundaries_are_conservative():
    assert FINALIZER["_status"](_interval(0.001, 0.2)) == "supported"
    assert FINALIZER["_status"](_interval(-0.2, -0.001)) == "unsupported"
    assert FINALIZER["_status"](_interval(0.0, 0.2)) == "mixed"
    assert FINALIZER["_status"](_interval(-0.2, 0.0)) == "mixed"


def test_registry_preserves_all_cells_without_a_performance_gate():
    report = {
        "campaign_version": "gaussian_foundations_v1",
        "manifest_digest": "a" * 64,
        "completed_atomic_models": 1_020,
        "factorial_cells": [
            *[_cell(0.01, 0.02) for _ in range(5)],
            *[_cell(-0.01, 0.01) for _ in range(4)],
            *[_cell(-0.02, -0.01) for _ in range(3)],
        ],
    }
    registry = FINALIZER["_registry"](report, "b" * 64)
    assert registry["performance_gate"] is False
    assert len(registry["claims"]) == 5
    for claim in registry["claims"]:
        assert claim["counts"] == {
            "supported": 5,
            "mixed": 4,
            "unsupported": 3,
        }
        assert len(claim["cells"]) == 12


def test_arxiv_source_filter_rejects_excluded_material(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "main.tex").write_text("Gaussian-only source")
    ARCHIVER["_check_forbidden"](clean)
    (clean / "supplement.tex").write_text("Excluded Ising campaign material")
    with pytest.raises(RuntimeError, match="forbidden phrase"):
        ARCHIVER["_check_forbidden"](clean)


def test_arxiv_command_capture_tolerates_non_utf8_tool_output(tmp_path):
    output = ARCHIVER["_run"](
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff')"],
        tmp_path,
        os.environ.copy(),
    )
    assert "\N{REPLACEMENT CHARACTER}" in output


def test_claim_finalizer_reads_content_verified_report_envelope(tmp_path):
    report = {
        "campaign_version": "gaussian_foundations_v1",
        "manifest_digest": "a" * 64,
        "completed_atomic_models": 1_020,
        "factorial_cells": [
            *[_cell(0.01, 0.02) for _ in range(11)],
            _cell(0.01, 0.02, n_qubits=19, width_factor=0.5),
        ],
    }
    report_path = tmp_path / "complete-report.json"
    write_immutable_json(report_path, report)
    registry = tmp_path / "claim-registry.json"
    markdown = tmp_path / "claim-registry.md"
    tex = tmp_path / "claims.tex"
    assert (
        FINALIZER["main"](
            [
                "--report",
                str(report_path),
                "--registry-json",
                str(registry),
                "--registry-md",
                str(markdown),
                "--tex",
                str(tex),
            ]
        )
        == 0
    )
    assert json.loads(registry.read_text())["completed_atomic_models"] == 1_020
    assert "HeadlineProfileBankGap" in tex.read_text()


def test_arxiv_source_population_rejects_registry_not_bound_to_report(tmp_path):
    repository = tmp_path / "repository"
    report_directory = repository / "campaign" / "report"
    report_directory.mkdir(parents=True)
    execution_source = {"schema_version": 1, "manifest_digest": "a" * 64}
    report = {
        "completed_atomic_models": 1_020,
        "manifest_digest": "a" * 64,
        "execution_source": execution_source,
    }
    write_immutable_json(report_directory / "complete-report.json", report)
    write_immutable_json(
        repository / "campaign" / "evidence" / "execution-source.json",
        execution_source,
    )
    (report_directory / "claim-registry.json").write_text(
        json.dumps(
            {
                "completed_atomic_models": 1_020,
                "manifest_digest": "a" * 64,
                "report_sha256": "b" * 64,
            }
        )
    )
    args = SimpleNamespace(
        campaign=Path("campaign"),
        paper=Path("paper"),
    )
    with pytest.raises(RuntimeError, match="does not bind"):
        ARCHIVER["_populate"](repository, tmp_path / "stage", args)
