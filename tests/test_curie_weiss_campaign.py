from dataclasses import replace
from pathlib import Path

import pytest

from iqp_cond.curie_weiss_campaign import (
    CurieWeissCampaignError,
    CurieWeissCampaignStore,
    PartialCurieWeissResultError,
    UndeclaredCurieWeissRunError,
)
from iqp_cond.curie_weiss_protocol import (
    CurieWeissCampaignManifest,
    plan_curie_weiss_runs,
)

ROOT = Path(__file__).parents[1]
MANIFEST = CurieWeissCampaignManifest.from_json(
    ROOT / "configs/curie_weiss_foundations_v1.json"
)


def test_store_rejects_undeclared_duplicate_and_partial_reuse(tmp_path):
    store = CurieWeissCampaignStore(tmp_path / "store", MANIFEST)
    run = plan_curie_weiss_runs(MANIFEST)[0]
    first = store.reserve(run, require_preflight=False)
    assert first.exists()
    with pytest.raises(PartialCurieWeissResultError, match="partial restart"):
        store.reserve(run, require_preflight=False)
    with pytest.raises(UndeclaredCurieWeissRunError):
        store.reserve(
            replace(run, data_seed=run.data_seed + 1), require_preflight=False
        )
    store.checkpoint_path(run).parent.mkdir(parents=True, exist_ok=True)
    store.checkpoint_path(run).write_bytes(b"partial")
    with pytest.raises(PartialCurieWeissResultError):
        store.load_result(run)


def test_preflight_is_resource_only_and_manifest_bound(tmp_path):
    store = CurieWeissCampaignStore(tmp_path / "store", MANIFEST)
    run = plan_curie_weiss_runs(MANIFEST)[0]
    with pytest.raises(CurieWeissCampaignError, match="preflight"):
        store.reserve(run)
    store.record_preflight(
        {"manifest_digest": MANIFEST.digest, "passed": True, "performance_gate": False}
    )
    assert store.reserve(run).exists()
