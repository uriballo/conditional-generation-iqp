from __future__ import annotations

import math

import pytest

from iqp_cond.artifacts import (
    DuplicateArtifactError,
    read_immutable_json,
)
from iqp_cond.estimator_scaling import (
    ScalingCase,
    ScalingTrack,
    archive_scaling_report,
    build_axis_scaling_cases,
    generator_bank,
    run_scaling_case,
    run_scaling_plan,
)


def test_generator_banks_realize_linear_and_quadratic_tracks() -> None:
    n_qubits = 6
    bounded = generator_bank(ScalingTrack.BOUNDED_DEGREE, n_qubits)
    complete = generator_bank(ScalingTrack.COMPLETE_TWO_LOCAL, n_qubits)

    assert bounded.shape == (2 * n_qubits, n_qubits)
    assert complete.shape == (
        n_qubits + n_qubits * (n_qubits - 1) // 2,
        n_qubits,
    )
    assert complete.shape[0] > bounded.shape[0]
    assert set(bounded.sum(axis=1)) == {1, 2}
    assert set(complete.sum(axis=1)) == {1, 2}


def test_axis_plan_is_deterministic_and_covers_all_resource_axes() -> None:
    kwargs = {
        "qubit_sizes": (2, 4),
        "condition_counts": (1, 3),
        "condition_dimensions": (1, 2),
        "n_samples": 32,
        "n_repeats": 2,
        "seed": 19,
        "exact_max_qubits": 2,
    }
    first = build_axis_scaling_cases(**kwargs)
    second = build_axis_scaling_cases(**kwargs)

    assert [case.digest for case in first] == [case.digest for case in second]
    assert {case.track for case in first} == set(ScalingTrack)
    assert {case.axis for case in first} == {
        "qubit_count",
        "generator_count",
        "condition_count",
        "condition_dimension",
    }
    assert {case.n_conditions for case in first} >= {1, 3}
    assert {case.condition_dimension for case in first} >= {1, 2}
    assert any(case.exact_reference for case in first)
    assert any(not case.exact_reference for case in first)


def test_small_case_reports_exact_estimator_and_gradient_errors() -> None:
    case = ScalingCase(
        track=ScalingTrack.COMPLETE_TWO_LOCAL,
        axis="fixture",
        n_qubits=2,
        n_generators=3,
        n_conditions=2,
        condition_dimension=2,
        n_samples=128,
        n_repeats=3,
        seed=31,
        exact_reference=True,
    )
    record = run_scaling_case(case)
    repeated = run_scaling_case(case)

    assert record["reference"]["kind"] == "exact_statevector"
    assert record["active_generator_count"] > 0
    assert record["full_training_runs_used"] == 0
    assert record["estimator_repeat_seeds"] == [10031, 10032, 10033]
    for summary_name in (
        "expectation_error",
        "gradient_error",
        "active_gradient_error",
    ):
        summary = record[summary_name]
        assert set(summary) == {
            "mean_signed_bias",
            "max_absolute_bias",
            "rmse",
            "mean_empirical_variance",
        }
        assert all(math.isfinite(value) for value in summary.values())
    assert record["runtime_seconds"]["case_total"] > 0.0
    assert record["peak_rss_bytes"] > 0
    assert record["synthetic_input_bytes"] > 0
    assert record["numerical_results_digest"] == repeated["numerical_results_digest"]


def test_large_style_case_reports_consistency_not_exact_reference() -> None:
    case = ScalingCase(
        track=ScalingTrack.BOUNDED_DEGREE,
        axis="fixture",
        n_qubits=3,
        n_generators=6,
        n_conditions=3,
        condition_dimension=2,
        n_samples=64,
        n_repeats=2,
        seed=43,
        exact_reference=False,
    )
    record = run_scaling_case(case)

    assert record["reference"]["kind"] == "repeated_estimator_consistency"
    assert "expectation_error" not in record
    assert "gradient_error" not in record
    assert "active_gradient_error" not in record
    assert record["expectation_consistency"]["consecutive_repeat_rmse"] >= 0.0
    assert record["gradient_consistency"]["consecutive_repeat_rmse"] >= 0.0
    assert record["active_gradient_consistency"]["consecutive_repeat_rmse"] >= 0.0


def test_plan_archives_immutably_without_trainability_proxy(tmp_path) -> None:
    case = ScalingCase(
        track=ScalingTrack.BOUNDED_DEGREE,
        axis="fixture",
        n_qubits=2,
        n_generators=2,
        n_conditions=1,
        condition_dimension=1,
        n_samples=32,
        n_repeats=2,
        seed=7,
        exact_reference=False,
    )
    report = run_scaling_plan(
        (case,),
        campaign_version="test_campaign",
        manifest_digest="0" * 64,
    )
    destination = tmp_path / "scaling.json"
    digest = archive_scaling_report(destination, report)

    assert len(digest) == 64
    assert read_immutable_json(destination) == report
    assert report["full_training_runs_used"] == 0
    assert len(report["numerical_results_digest"]) == 64
    assert "host-dependent" in report["determinism_scope"]
    assert set(report["provenance"]) == {
        "code_revision",
        "working_tree_dirty",
        "invocation",
        "source_digests",
        "environment",
    }
    assert report["provenance"]["environment"]["backend"]
    assert report["provenance"]["environment"]["devices"]
    serialized_keys = {key for record in report["records"] for key in record}
    assert "gradient_magnitude" not in serialized_keys
    assert "gradient_norm" not in serialized_keys
    with pytest.raises(DuplicateArtifactError):
        archive_scaling_report(destination, report)
