from __future__ import annotations

import json

import numpy as np
import pytest

from iqp_cond.confirmatory_estimator_audit import (
    ENCODER_FAMILIES,
    SAMPLE_BUDGETS,
    TOPOLOGIES,
    audit_cell,
    build_case,
    convergence_summary,
    load_manifest,
    write_immutable_json,
)


def test_protocol_coordinates_are_frozen() -> None:
    assert SAMPLE_BUDGETS == (256, 1024, 4096, 16384)
    assert ENCODER_FAMILIES == ("product", "purified", "field_equivariant")
    assert TOPOLOGIES == ("sparse", "complete")
    assert load_manifest()["sample_budgets"] == list(SAMPLE_BUDGETS)


@pytest.mark.parametrize("encoder", ENCODER_FAMILIES)
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_each_encoder_and_topology_matches_exact_small_system_values(
    encoder: str, topology: str
) -> None:
    record = audit_cell(build_case(encoder, topology), 256, 0)
    assert record["n_ancillas"] == (2 if encoder == "purified" else 0)
    assert np.isfinite(record["moment_rmse"])
    assert np.isfinite(record["mmd_absolute_error"])
    assert np.isfinite(record["gradient_rmse"])
    assert record["runtime_seconds"] > 0.0
    assert record["moment_rmse"] < 0.3
    assert record["gradient_rmse"] < 0.3


def test_convergence_summary_recovers_inverse_square_root_fixture() -> None:
    records = []
    for budget in SAMPLE_BUDGETS:
        for repetition in range(2):
            value = (1.0 + 0.05 * repetition) / np.sqrt(budget)
            records.append(
                {
                    "encoder": "product",
                    "topology": "sparse",
                    "sample_budget": budget,
                    "moment_rmse": value,
                    "mmd_absolute_error": 2.0 * value,
                    "gradient_rmse": 3.0 * value,
                    "runtime_seconds": budget / 1000,
                }
            )
    summary = convergence_summary(records)[0]
    for metric in summary["metrics"].values():
        assert metric["log_log_slope"] == pytest.approx(-0.5)


def test_immutable_writer_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "audit.json"
    write_immutable_json(path, {"z": 1, "a": 2})
    assert json.loads(path.read_text()) == {"a": 2, "z": 1}
    with pytest.raises(FileExistsError):
        write_immutable_json(path, {"a": 3})


def test_sampling_metrics_are_deterministic() -> None:
    case = build_case("product", "sparse")
    first = audit_cell(case, 256, 3)
    second = audit_cell(case, 256, 3)
    for metric in ("moment_rmse", "mmd_estimate", "gradient_rmse"):
        assert first[metric] == second[metric]
