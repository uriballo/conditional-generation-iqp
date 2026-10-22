import json

import numpy as np
import pytest

from paper_experiments import cw_followup_stages as stages
from paper_experiments.cw_followup_config import FAMILIES, MARKERS, stage_betas
from paper_experiments.cw_followup_fit import digest, write_json, write_npz


def test_stage_transition_preserves_every_numerical_source_and_environment():
    old = dict(paper_sources={"fit.py": "fixed"}, backend="gpu")
    new = dict(
        paper_sources={
            "fit.py": "fixed",
            "paper_experiments/cw_followup_stages.py": "new",
        },
        backend="gpu",
    )
    assert stages._transition(old, new) == {
        "paper_experiments/cw_followup_stages.py": dict(before=None, after="new")
    }
    with pytest.raises(ValueError, match="environment"):
        stages._transition(old, dict(new, backend="cpu"))
    with pytest.raises(ValueError, match="fit source"):
        stages._transition(old, dict(new, paper_sources={"fit.py": "changed"}))


def test_stage_freeze_rejects_changed_selection(tmp_path):
    path = tmp_path / "manifest.json"
    stages._freeze(path, {"trial": 44})
    stages._freeze(path, {"trial": 44})
    with pytest.raises(ValueError, match="manifest differs"):
        stages._freeze(path, {"trial": 103})
    assert json.loads(path.read_text()) == {"trial": 44}


def test_stages_confirm_then_freeze_and_retain_all_final_repeats(tmp_path, monkeypatch):
    root = tmp_path / "hpo"
    root.mkdir()
    write_json(root / "tpe-review-ready.json", {})
    write_json(root / "tpe-continuation.json", {})
    identity = dict(paper_sources={"fit.py": "fixed"}, backend="test")
    proposals = {
        family: [dict(trial_id=i, config={"id": i}) for i in range(3)]
        for family in FAMILIES
    }
    monkeypatch.setattr(stages.parent, "source_identity", lambda: identity)
    monkeypatch.setattr(
        stages, "_screening", lambda *_: (dict(identity=identity), proposals, {})
    )
    calls = []

    def fake_worker(args):
        job, destination = args
        calls.append(
            (job["stage"], job["family"], job["trial_id"], job["repeat"], job["beta"])
        )
        spec = stages.protocol(True)
        beta = np.asarray(stage_betas(spec, job["stage"], job["beta"]))
        value = [0.1, 0.106, 0.2][job["trial_id"]]
        physics = [0.1, 0.01, 0.001][job["trial_id"]]
        shells = np.full((len(beta), 17), 1 / 17)
        curves = dict(
            beta=beta.tolist(),
            tvd=[value] * len(beta),
            magnetization_error=[physics] * len(beta),
            heat_error=[0] * len(beta),
            shells=shells.tolist(),
            target_shells=shells.tolist(),
        )
        directory = destination / "fits" / digest(job)
        result = dict(
            status="ok",
            job=job,
            metrics_curves=curves,
            artifact_sha256={},
            directory=str(directory),
        )
        write_json(directory / "metrics.json", result)
        if job["stage"] == "final" and job["family"].startswith("shared_"):
            marker_curves = dict(curves)
            marker_curves.update(
                beta=np.asarray(MARKERS),
                tvd=np.full(3, value),
                magnetization_error=np.full(3, physics),
                heat_error=np.zeros(3),
                shells=np.full((3, 17), 1 / 17),
                target_shells=np.full((3, 17), 1 / 17),
            )
            write_npz(directory / "markers.npz", **marker_curves)
        return result

    monkeypatch.setattr(stages, "_worker", fake_worker)
    result = stages.run_all(root, phase="reporting", smoke=True)
    assert result["winners"] == dict.fromkeys(FAMILIES, 1)
    assert len(calls) == 32  # 24 confirmation circuit jobs, 8 final circuit jobs.
    final = json.loads((tmp_path / "final/figure_data.json").read_text())
    assert final["smoke"] is True
    assert final["repeats"] == 1
    assert final["starts"] == 1
    assert {r["trial_id"] for r in final["families"].values()} == {1}
    for family in FAMILIES:
        row = final["families"][family]
        assert row["marker_beta"] == list(MARKERS)
        assert len(row["marker_curves"]["tvd"]) == 1
        assert len(row["marker_curves"]["tvd"][0]) == 3
        expected = (
            len(stage_betas(stages.protocol(True), "final"))
            if family.startswith("shared_")
            else 3
        )
        assert len(row["curves"]["tvd"][0]) == expected
    # Changing selection on resume must fail before fitting final again.
    winners = tmp_path / "confirmation/winners.json"
    tampered = json.loads(winners.read_text())
    tampered["shared_parity"]["trial_id"] = 2
    write_json(winners, tampered)
    with pytest.raises(ValueError, match="manifest differs"):
        stages.run_all(root, phase="final", smoke=True)
    assert len(calls) == 32


def test_final_requires_completed_confirmation(tmp_path, monkeypatch):
    root = tmp_path / "hpo"
    root.mkdir()
    write_json(root / "tpe-review-ready.json", {})
    write_json(root / "tpe-continuation.json", {})
    identity = dict(paper_sources={}, backend="test")
    monkeypatch.setattr(stages.parent, "source_identity", lambda: identity)
    monkeypatch.setattr(
        stages, "_screening", lambda *_: (dict(identity=identity), {}, {})
    )
    with pytest.raises(ValueError, match="completed confirmation"):
        stages.run_all(root, phase="final", smoke=True)
