from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from iqp_cond.curie_single_temperature import (
    SingleTemperatureManifest,
    SingleTemperatureRunIdentity,
    plan_single_temperature_runs,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from plot_curie_weiss_confirmatory import index_direct_runs, index_runs, quantiles

DIRECT_MANIFEST = SingleTemperatureManifest.from_json(
    Path(__file__).parents[1] / "configs" / "curie_weiss_single_temperature_v1.json"
)


def minimal_manifest() -> dict[str, object]:
    return {
        "campaign_version": "campaign_v1",
        "system_sizes": [8],
        "models": ["full_joint_r5", "no_condition_m5"],
        "seeds": [100],
        "training": {"updates": 2000},
    }


def payload(arm: str) -> dict[str, object]:
    return {
        "run_identity": {
            "campaign_version": "campaign_v1",
            "n_qubits": 8,
            "arm": arm,
            "seed": 100,
        },
        "lightweight": False,
        "optimizer_updates": 2000,
        "manifest_digest": "a" * 64,
    }


def direct_payload(run: SingleTemperatureRunIdentity) -> dict[str, object]:
    return {
        "run_identity": run.to_dict(),
        "lightweight": False,
        "optimizer_updates": 2000,
        "manifest_digest": DIRECT_MANIFEST.digest,
    }


def test_quantiles_are_computed_pointwise() -> None:
    median, lower, upper = quantiles([[1, 2], [3, 4], [5, 8]])
    assert np.allclose(median, [3, 4])
    assert np.allclose(lower, [2, 3])
    assert np.allclose(upper, [4, 6])


def test_index_runs_accepts_exact_manifest_coverage() -> None:
    runs = index_runs(
        [payload("full_joint_r5"), payload("no_condition_m5")], minimal_manifest()
    )
    assert set(runs) == {(8, "full_joint_r5", 100), (8, "no_condition_m5", 100)}


def test_index_runs_rejects_duplicate_coordinate() -> None:
    first = payload("full_joint_r5")
    with pytest.raises(ValueError, match="conflicting duplicate run coordinate"):
        index_runs([first, copy.deepcopy(first)], minimal_manifest())


def test_index_runs_rejects_incomplete_manifest_coverage() -> None:
    with pytest.raises(ValueError, match="manifest coverage mismatch"):
        index_runs([payload("full_joint_r5")], minimal_manifest())


def test_index_direct_runs_accepts_complete_frozen_grid() -> None:
    planned = plan_single_temperature_runs(DIRECT_MANIFEST)

    indexed = index_direct_runs(
        (direct_payload(run) for run in planned), DIRECT_MANIFEST
    )

    assert len(indexed) == 330
    assert (8, 0.5, 100) in indexed
    assert (16, 1.5, 109) in indexed


def test_index_direct_runs_rejects_duplicate_coordinate() -> None:
    planned = plan_single_temperature_runs(DIRECT_MANIFEST)
    records = [direct_payload(run) for run in planned]
    records.append(copy.deepcopy(records[0]))

    with pytest.raises(ValueError, match="conflicting duplicate direct-reference"):
        index_direct_runs(records, DIRECT_MANIFEST)


def test_index_direct_runs_rejects_missing_coordinate() -> None:
    planned = plan_single_temperature_runs(DIRECT_MANIFEST)

    with pytest.raises(ValueError, match="direct-reference coverage mismatch"):
        index_direct_runs(
            (direct_payload(run) for run in planned[:-1]), DIRECT_MANIFEST
        )
