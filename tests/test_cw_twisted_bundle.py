import hashlib
import json

import pytest

from paper_experiments import cw_twisted_bundle as bundle


def _root(tmp_path, family):
    root = tmp_path / family
    (root / "final").mkdir(parents=True)
    (root / "campaign.json").write_text("{}")
    (root / "final" / "campaign.json").write_text("{}")
    data = dict(
        family=family,
        trial_id=7,
        config={},
        beta=[0.0],
        curves={},
        marker_beta=[0.4],
        marker_curves={},
        repeat_scores=[{}] * 10,
        provenance=dict(protocol=dict(protocol_sha256="p", campaign_id="c")),
    )
    path = root / "final" / "figure_data.json"
    path.write_text(json.dumps(data))
    status = dict(
        status="final_complete",
        figure_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (root / "final" / "review-ready.json").write_text(json.dumps(status))
    return root


def test_bundle_holds_only_twisted_families(tmp_path):
    families = ("shared_twisted_parity", "independent_twisted_parity")
    roots = [_root(tmp_path, f) for f in families]
    payload = bundle.build(roots, tmp_path / "out")
    assert payload["kind"] == "cw_twisted_final"
    assert set(payload["families"]) == set(families)
    with pytest.raises(ValueError):
        bundle.build([_root(tmp_path, "shared_parity")], tmp_path / "other")


def test_optional_root_is_skipped_until_its_final_stage_exists(tmp_path):
    shared = _root(tmp_path, "shared_twisted_parity")
    pending = tmp_path / "independent_twisted_parity"
    payload = bundle.build([shared], tmp_path / "out", optional=[pending])
    assert set(payload["families"]) == {"shared_twisted_parity"}
    assert payload["omitted_roots"] == [str(pending)]
    _root(tmp_path, "independent_twisted_parity")
    payload = bundle.build([shared], tmp_path / "out", optional=[pending])
    assert len(payload["families"]) == 2 and "omitted_roots" not in payload
    with pytest.raises(ValueError):
        bundle.build([], tmp_path / "none", optional=[tmp_path / "x"])
