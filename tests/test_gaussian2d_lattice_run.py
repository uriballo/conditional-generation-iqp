"""Selection isolation and strict resume checks without running fits."""

import json

import numpy as np
import pytest

from paper_experiments import gaussian2d_lattice_run as r


def test_conditions_separate_training_selection_and_reporting():
    for design in r.DESIGNS:
        sets = r.load_conditions(design)
        assert sets["fitting"].shape == (9, 2)
        for first, second in [
            ("fitting", "development"),
            ("fitting", "reporting"),
            ("development", "reporting"),
        ]:
            assert not any(np.allclose(a, b) for a in sets[first] for b in sets[second])
    assert len(r.load_conditions("grid")["reporting"]) == 32
    assert len(r.load_conditions("axes")["reporting"]) == 16


def test_independent_parameters_and_geometry_are_distinct():
    theta = r.initial_parameters(7, r.PROTOCOL["candidates"]["mpi3_lr0.01"])
    assert theta.shape == (40,)
    assert np.std(theta[:16]) > 0 and np.std(theta[16:]) > 0
    assert np.mean(theta[:16]) > 0.8
    assert r.DESIGNS == ("grid", "axes")


def test_campaign_freezes_both_winners_before_final_reporting(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "identity", lambda: {"fixed": "identity"})
    calls = []

    def fake(output, design, stage, candidate, seed, **kwargs):
        if stage == "final" and design in r.DESIGNS:
            winners = json.loads((tmp_path / "frozen_winners.json").read_text())[
                "winners"
            ]
            assert set(winners) == set(r.DESIGNS)
            assert len([c for c in calls if c[1] != "final"]) == 16
        calls.append((design, stage, candidate, seed))
        return {
            "status": "complete",
            "checkpoint_sha256": "fake",
            "best_score": float(list(r.PROTOCOL["candidates"]).index(candidate))
            if candidate != "fixed"
            else 0.01,
        }

    state = r.run_campaign(tmp_path, fit_runner=fake)
    assert state["status"] == "complete"
    assert len(calls) == 31
    assert len([c for c in calls if c[1] == "screen"]) == 8
    assert len([c for c in calls if c[1] == "confirm"]) == 8
    assert len([c for c in calls if c[0] == "translation"]) == 5
    assert not set(r.PROTOCOL["seeds"]["final"]) & set(r.PROTOCOL["seeds"]["confirm"])


def test_source_change_and_checkpoint_corruption_reject_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "identity", lambda: {"new": True})
    (tmp_path / "campaign_receipt.json").write_text(
        json.dumps({"identity": {"old": True}, "smoke": False})
    )
    with pytest.raises(RuntimeError, match="changed"):
        r.run_campaign(tmp_path)
    path = tmp_path / "fit"
    path.mkdir()
    expected = {"config": "same"}
    (path / "fit.npz").write_bytes(b"changed")
    (path / "fit.json").write_text(
        json.dumps(
            {"status": "complete", "request": expected, "checkpoint_sha256": "wrong"}
        )
    )
    with pytest.raises(RuntimeError, match="integrity"):
        r._compatible_fit(path, expected)
