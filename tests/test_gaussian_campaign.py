from dataclasses import replace
from pathlib import Path

import pytest

from iqp_cond.gaussian_campaign import (
    GaussianCampaignError,
    GaussianCampaignStore,
    PartialGaussianResultError,
    UndeclaredGaussianRunError,
)
from iqp_cond.gaussian_protocol import (
    GaussianCampaignManifest,
    plan_gaussian_runs,
)

REPOSITORY = Path(__file__).parents[1]
MANIFEST = GaussianCampaignManifest.from_json(
    REPOSITORY / "configs" / "gaussian_foundations_v1.json"
)


def test_store_rejects_undeclared_and_duplicate_reservations_are_idempotent(tmp_path):
    store = GaussianCampaignStore(tmp_path / "store", MANIFEST)
    run = plan_gaussian_runs(MANIFEST)[0]
    first = store.reserve(run, require_preflight=False)
    assert store.reserve(run, require_preflight=False) == first
    with pytest.raises(UndeclaredGaussianRunError):
        store.reserve(replace(run, optimizer_seed=99), require_preflight=False)


def test_preflight_is_a_resource_gate_not_a_performance_gate(tmp_path):
    store = GaussianCampaignStore(tmp_path / "store", MANIFEST)
    run = plan_gaussian_runs(MANIFEST)[0]
    with pytest.raises(GaussianCampaignError, match="preflight"):
        store.reserve(run)
    evidence = {
        "manifest_digest": MANIFEST.digest,
        "passed": True,
        "performance_gate": False,
    }
    store.record_preflight(evidence)
    assert store.reserve(run).exists()


def test_store_rejects_partial_result_reuse(tmp_path):
    store = GaussianCampaignStore(tmp_path / "store", MANIFEST)
    run = plan_gaussian_runs(MANIFEST)[0]
    store.reserve(run, require_preflight=False)
    store.checkpoint_path(run).parent.mkdir(parents=True)
    store.checkpoint_path(run).write_bytes(b"partial")
    with pytest.raises(PartialGaussianResultError):
        store.load_result(run)
    with pytest.raises(GaussianCampaignError, match="incomplete"):
        store.load_result(plan_gaussian_runs(MANIFEST)[1])
