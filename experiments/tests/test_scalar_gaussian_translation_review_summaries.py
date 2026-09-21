"""Focused tests for deterministic scalar Gaussian review summaries."""

import csv
import json
from pathlib import Path

import pytest

from experiments.scalar_gaussian_translation import campaign, protocol
from experiments.scalar_gaussian_translation import review_summaries as summaries


def _row(**updates):
    row = {
        "encoder": "broadcast_affine",
        "protocol": "interpolation",
        "anchor_count": "3",
        "placement": "even",
        "layout_seed": "",
        "budget_mode": "fixed_total",
        "optimizer_seed": "401",
        "hyperparameter_id": "hp-a",
        "model_kind": "conditional",
        "category": "held_out_interpolation",
        "condition": "0.25",
        "bitstring_tvd": "0.1",
        "complement_covariance_error": "",
    }
    row.update(updates)
    return row


def test_equal_configuration_estimand_averages_conditions_and_excludes_mirrors():
    rows = [
        _row(bitstring_tvd="0.1", condition="0.25"),
        _row(bitstring_tvd="0.3", condition="0.75"),
        _row(
            bitstring_tvd="0.8",
            optimizer_seed="402",
            hyperparameter_id="hp-b",
        ),
        _row(protocol="mirror_far", bitstring_tvd="0.99"),
    ]
    result = summaries.equal_configuration_sensitivity(rows)["encoders"][
        "broadcast_affine"
    ]["held_out_interpolation"]
    assert result["n_rows"] == 3
    assert result["n_configurations"] == 2
    assert result["raw_row_mean"] == pytest.approx(0.4)
    assert result["raw_row_median"] == pytest.approx(0.3)
    assert result["equal_configuration_mean"] == pytest.approx(0.5)
    assert result["equal_configuration_median"] == pytest.approx(0.5)
    assert result["equal_configuration_max"] == pytest.approx(0.8)


def test_complement_summaries_skip_blank_values_and_stratify():
    rows = [
        _row(complement_covariance_error="0.02", condition="0.25"),
        _row(
            encoder="anti_block",
            protocol="near",
            category="near_extrapolation",
            complement_covariance_error="0.04",
            condition="0.3",
        ),
        _row(complement_covariance_error="", condition="0.4"),
    ]
    result = summaries.complement_covariance_summaries(rows)
    broadcast = result["overall"]["broadcast_affine"]
    assert broadcast["n"] == 1
    assert broadcast["mean"] == pytest.approx(0.02)
    assert broadcast["condition_pairs"] == [{"c": 0.25, "complement": 0.75}]
    anti = result["by_protocol_category"]["anti_block"]
    assert anti[0]["protocol"] == "near"
    assert anti[0]["category"] == "near_extrapolation"
    assert anti[0]["n"] == 1


def test_condition_blind_uses_equal_configuration_means():
    rows = [
        _row(encoder="condition_blind", bitstring_tvd="0.2", condition="0.1"),
        _row(encoder="condition_blind", bitstring_tvd="0.4", condition="0.2"),
        _row(
            encoder="condition_blind",
            bitstring_tvd="0.8",
            optimizer_seed="402",
            condition="0.3",
        ),
    ]
    result = summaries.condition_blind_summaries(rows)
    assert result == [
        {
            "protocol": "interpolation",
            "anchor_count": 3,
            "category": "held_out_interpolation",
            "n_configurations": 2,
            "equal_configuration_mean": pytest.approx(0.55),
            "equal_configuration_median": pytest.approx(0.55),
            "equal_configuration_max": pytest.approx(0.8),
        }
    ]


def test_direct_reference_global_and_condition_resolved():
    result = summaries.direct_reference_summaries(
        [
            {"condition": "0.25", "bitstring_tvd": "0.2"},
            {"condition": "0.25", "bitstring_tvd": "0.4"},
            {"condition": "0.75", "bitstring_tvd": ""},
        ]
    )
    assert result["global"]["n"] == 2
    assert result["global"]["mean"] == pytest.approx(0.3)
    assert result["global"]["median"] == pytest.approx(0.3)
    assert result["global"]["max"] == pytest.approx(0.4)
    assert result["by_condition"][0]["condition"] == 0.25
    assert result["by_condition"][0]["mean"] == pytest.approx(0.3)


def test_classical_baseline_contains_expected_strata_and_audit_grid():
    result = summaries.classical_interpolation_summaries()
    strata = {(row["protocol"], row["anchor_count"]) for row in result["aggregates"]}
    assert strata == {
        (protocol_name, anchor_count)
        for protocol_name in campaign.ORIGINAL_PROTOCOLS
        for anchor_count in protocol.ANCHOR_COUNTS
    }
    assert result["grid_size"] == len(protocol.evaluation_grid("audit"))
    assert result["placement"] == "even"
    assert {row["placement"] for row in result["by_condition"]} == {"even"}
    assert all(row["n"] > 0 for row in result["aggregates"])
    categories_by_protocol = {
        protocol_name: {
            row["category"]
            for row in result["aggregates"]
            if row["protocol"] == protocol_name
        }
        for protocol_name in campaign.ORIGINAL_PROTOCOLS
    }
    assert categories_by_protocol == {
        "interpolation": {"held_out_interpolation"},
        "near": {"held_out_interpolation", "near_extrapolation"},
        "far": {"held_out_interpolation", "far_extrapolation"},
    }


def test_cli_writes_deterministic_json_and_treats_blank_as_missing(tmp_path: Path):
    audit_path = tmp_path / "audit.csv"
    direct_path = tmp_path / "direct.csv"
    output = tmp_path / "summaries.json"
    fields = list(_row().keys())
    with audit_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_row(complement_covariance_error="0.01"))
        writer.writerow(_row(complement_covariance_error=""))
    with direct_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition", "bitstring_tvd"])
        writer.writeheader()
        writer.writerow({"condition": "0.5", "bitstring_tvd": "0.2"})
    summaries.main(
        [
            "--audit-rows",
            str(audit_path),
            "--direct-rows",
            str(direct_path),
            "--output",
            str(output),
        ]
    )
    first = output.read_text()
    parsed = json.loads(first)
    assert parsed["complement_covariance"]["overall"]["broadcast_affine"]["n"] == 1
    assert parsed["direct_reference"]["global"]["n"] == 1
    summaries.main(
        [
            "--audit-rows",
            str(audit_path),
            "--direct-rows",
            str(direct_path),
            "--output",
            str(output),
        ]
    )
    assert output.read_text() == first
