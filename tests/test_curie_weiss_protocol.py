from dataclasses import replace
from pathlib import Path

import pytest

from iqp_cond.curie_weiss_protocol import (
    CurieWeissCampaignManifest,
    CurieWeissFamily,
    conditioned_angle,
    plan_curie_weiss_runs,
)

ROOT = Path(__file__).parents[1]
MANIFEST = CurieWeissCampaignManifest.from_json(
    ROOT / "configs" / "curie_weiss_foundations_v1.json"
)


def test_frozen_grid_and_exact_270_fit_cardinality():
    runs = plan_curie_weiss_runs(MANIFEST)
    assert len(runs) == len({run.run_id for run in runs}) == 270
    assert MANIFEST.evaluation_betas == pytest.approx(
        tuple(0.5 + 0.1 * i for i in range(11))
    )
    counts = {
        family: sum(r.family is family for r in runs) for family in CurieWeissFamily
    }
    assert counts == {
        CurieWeissFamily.CONDITIONED: 30,
        CurieWeissFamily.NO_CONDITION: 30,
        CurieWeissFamily.TEMPERATURE_REFERENCE: 180,
        CurieWeissFamily.UNRESTRICTED_CONDITIONED: 30,
    }


def test_methods_are_seed_matched_and_references_are_declared_only_at_train_betas():
    runs = [
        r
        for r in plan_curie_weiss_runs(MANIFEST)
        if r.n_qubits == 8 and r.optimizer_seed == 2
    ]
    assert len({r.data_seed for r in runs}) == 1
    references = [
        r.reference_beta
        for r in runs
        if r.family is CurieWeissFamily.TEMPERATURE_REFERENCE
    ]
    assert tuple(references) == MANIFEST.training_betas
    with pytest.raises(ValueError, match="training beta"):
        replace(
            references_run := next(r for r in runs if r.reference_beta is not None),
            reference_beta=1.0,
        )
    assert references_run.reference_beta == 0.5


def test_disclosed_condition_map_is_broadcast_and_not_outcome_dependent():
    assert conditioned_angle(0.5, 8) == pytest.approx((0.35 * 3.141592653589793,) * 8)
    assert conditioned_angle(1.0, 12) == pytest.approx((0.0,) * 12)
    assert conditioned_angle(1.5, 16) == pytest.approx(
        (-0.35 * 3.141592653589793,) * 16
    )


def test_five_dry_run_shards_are_disjoint_and_complete(tmp_path):
    import json
    import subprocess

    shards = []
    for index in range(5):
        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/run_curie_weiss_campaign.py",
                "run",
                "--campaign",
                str(tmp_path / "campaign"),
                "--dry-run",
                "--shard-count",
                "5",
                "--shard-index",
                str(index),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["count"] == 54
        shards.append(set(payload["run_ids"]))
    assert len(set.union(*shards)) == 270
    assert sum(map(len, shards)) == 270
