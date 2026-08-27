from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from iqp_cond.curie_single_temperature import (
    BETAS,
    CAMPAIGN_VERSION,
    SEEDS,
    SYSTEM_SIZES,
    SingleTemperatureManifest,
    SingleTemperatureRunIdentity,
    plan_single_temperature_runs,
)
from iqp_cond.protocol import ProtocolValidationError

REPOSITORY = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY / "configs" / "curie_weiss_single_temperature_v1.json"


def manifest() -> SingleTemperatureManifest:
    return SingleTemperatureManifest.from_json(MANIFEST_PATH)


def test_manifest_plans_exact_unique_grid_and_parameter_counts() -> None:
    planned = plan_single_temperature_runs(manifest())

    assert len(planned) == 330
    assert len({run.run_id for run in planned}) == 330
    assert {(run.n_qubits, run.beta, run.seed) for run in planned} == {
        (n_qubits, beta, seed)
        for n_qubits in SYSTEM_SIZES
        for beta in BETAS
        for seed in SEEDS
    }
    assert {run.parameter_count for run in planned} == {91, 153, 231}


def test_manifest_and_identity_reject_protocol_drift(tmp_path: Path) -> None:
    raw = json.loads(MANIFEST_PATH.read_text())
    drifted = copy.deepcopy(raw)
    drifted["training"]["batch_size"] = 450
    path = tmp_path / "configs" / MANIFEST_PATH.name
    path.parent.mkdir()
    path.write_text(json.dumps(drifted))

    with pytest.raises(ProtocolValidationError, match="training differs"):
        SingleTemperatureManifest.from_json(path)

    run = plan_single_temperature_runs(manifest())[0]
    assert SingleTemperatureRunIdentity.from_dict(run.to_dict()) == run
    invalid = run.to_dict()
    invalid["beta"] = 1.6
    with pytest.raises(ProtocolValidationError, match="temperature"):
        SingleTemperatureRunIdentity.from_dict(invalid)


def test_cli_shards_are_disjoint_and_cover_all_runs() -> None:
    shards: list[set[str]] = []
    for index in range(10):
        completed = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "scripts" / "run_curie_single_temperature.py"),
                "--manifest",
                str(MANIFEST_PATH),
                "--shard-count",
                "10",
                "--shard-index",
                str(index),
                "--dry-run",
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["count"] == 33
        shards.append({row["run_id"] for row in payload["runs"]})

    assert len(set.union(*shards)) == 330
    assert sum(len(shard) for shard in shards) == 330
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(shards)
        for right in shards[index + 1 :]
    )


def test_run_identity_has_stable_campaign_version() -> None:
    run = plan_single_temperature_runs(manifest())[0]
    assert run.campaign_version == CAMPAIGN_VERSION
    assert run.role == "training"
