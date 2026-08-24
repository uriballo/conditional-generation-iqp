import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from iqp_cond.gaussian_protocol import (
    DataRegime,
    GaussianCampaignManifest,
    GaussianFamily,
    plan_condition_count_additions,
    plan_factorial_runs,
    plan_gaussian_runs,
)

REPOSITORY = Path(__file__).parents[1]
MANIFEST = GaussianCampaignManifest.from_json(
    REPOSITORY / "configs" / "gaussian_foundations_v1.json"
)


def test_manifest_and_exact_run_cardinalities():
    assert len(plan_factorial_runs(MANIFEST)) == 840
    assert len(plan_condition_count_additions(MANIFEST)) == 180
    runs = plan_gaussian_runs(MANIFEST)
    assert len(runs) == len({run.run_id for run in runs}) == 1_020
    assert MANIFEST.training_grid(2) == (0.0, 1.0)
    assert MANIFEST.training_grid(3) == (0.0, 0.5, 1.0)
    assert MANIFEST.training_grid(5) == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert MANIFEST.training_grid(9) == tuple(index / 8 for index in range(9))


def test_factorial_and_nested_reuse_accounting():
    runs = plan_gaussian_runs(MANIFEST)
    factorial = [run for run in runs if run.study.value == "factorial"]
    additions = [run for run in runs if run.study.value == "condition_count"]
    assert len(factorial) == 4 * 3 * 10 * 7
    assert len(additions) == 10 * (4 + 2 * (3 + 4))
    primary_bank = [
        run
        for run in runs
        if run.n_qubits == 19
        and run.width_factor == 0.5
        and run.family is GaussianFamily.SINGLE_PEAK
    ]
    assert len(primary_bank) == 9 * 10
    assert {run.bank_peak_index for run in primary_bank} == set(range(9))


def test_methods_share_master_data_seed():
    runs = plan_factorial_runs(MANIFEST)
    cell = [
        run
        for run in runs
        if run.n_qubits == 8 and run.width_factor == 0.35 and run.optimizer_seed == 0
    ]
    assert len({run.data_seed for run in cell}) == 1


def test_fixed_total_counts_are_frozen():
    assert MANIFEST.samples_per_peak(2, DataRegime.FIXED_TOTAL) == (9_000, 1_800)
    assert MANIFEST.samples_per_peak(3, DataRegime.FIXED_TOTAL) == (6_000, 1_200)
    assert MANIFEST.samples_per_peak(5, DataRegime.FIXED_TOTAL) == (3_600, 720)
    assert MANIFEST.samples_per_peak(9, DataRegime.FIXED_TOTAL) == (2_000, 400)
    assert MANIFEST.samples_per_peak(9, DataRegime.FIXED_PER_PEAK) == (
        10_000,
        2_000,
    )


def test_identity_rejects_bank_index_on_shared_family():
    run = next(
        run
        for run in plan_factorial_runs(MANIFEST)
        if run.family is GaussianFamily.CONDITIONED
    )
    with pytest.raises(ValueError, match="only single-peak"):
        replace(run, bank_peak_index=0)


def test_sharded_dry_run_is_disjoint_complete_and_resumable(tmp_path):
    shards = []
    campaign = tmp_path / "campaign"
    for shard_index in range(5):
        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/run_gaussian_campaign.py",
                "run",
                "--campaign",
                str(campaign),
                "--dry-run",
                "--shard-count",
                "5",
                "--shard-index",
                str(shard_index),
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["count"] == 204
        shards.append(set(payload["run_ids"]))
    assert len(set.union(*shards)) == 1_020
    assert sum(len(shard) for shard in shards) == len(set.union(*shards))
