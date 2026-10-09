import json

import numpy as np
import optuna
import pytest

from paper_experiments import cw_followup_hpo as hpo
from paper_experiments.cw_followup_metrics import robust_score


def test_independent_objective_aggregates_within_repeat_first():
    curves = np.array([[0.1, 0.5, 0.2], [0.5, 0.1, 0.4]])
    records = []
    for repeat, curve in enumerate(curves):
        for beta, tvd in zip(hpo.MARKERS, curve, strict=True):
            records.append(
                dict(
                    status="ok",
                    job=dict(repeat=repeat, beta=beta),
                    metrics_curves=dict(
                        beta=[beta], tvd=[tvd], magnetization_error=[0], heat_error=[0]
                    ),
                )
            )
    result = hpo.score_records(records, "independent_parity", 2)
    expected = np.mean([robust_score(row, hpo.MARKERS) for row in curves])
    assert result["J"] == expected
    assert result["J"] != robust_score(curves.mean(0), hpo.MARKERS)
    assert result["J"] != curves.max(1).mean()
    with pytest.raises(ValueError, match="missing"):
        hpo.score_records(records[:-1], "independent_parity", 2)
    records[0]["status"] = "numerical_failure"
    result = hpo.score_records(records, "independent_parity", 2)
    assert result == dict(
        J=1.0, P=None, valid=False, search_penalty_not_measured_tvd=True
    )


def test_random_freeze_resume_and_four_study_barrier(tmp_path, monkeypatch):
    monkeypatch.setattr(hpo, "source_identity", lambda: {"test": "same"})
    calls = []

    def fake_worker(args):
        context, proposal, root = args
        key = (context["family"], proposal["trial_id"])
        if len(calls) == 3:
            calls.append("interruption")
            raise RuntimeError("simulated process interruption")
        calls.append(key)
        jobs = list(hpo.jobs(context, proposal))
        assert {job["stage"] for job in jobs} == {"screen"}
        result = dict(
            **proposal, J=0.2 + proposal["trial_id"] * 0.01, P=0.1, valid=True, fits=[]
        )
        hpo.write_json(root / "proposals" / f"{proposal['trial_id']:03d}.json", result)
        return context["family"], result

    monkeypatch.setattr(hpo, "_trial_worker", fake_worker)
    with pytest.raises(RuntimeError, match="interruption"):
        hpo.run_all(tmp_path, smoke=True)
    frozen = (tmp_path / "random-proposals.json").read_bytes()
    assert not (tmp_path / "random-review-ready.json").exists()
    result = hpo.run_all(tmp_path, smoke=True)
    assert (tmp_path / "random-proposals.json").read_bytes() == frozen
    assert result["counts"] == dict.fromkeys(hpo.FAMILIES, 3)
    assert result["status"] == "paused_for_random_review"
    assert result["tpe_started"] is False
    assert result["confirmation_started"] is False
    assert result["final_started"] is False
    assert len([c for c in calls if c != "interruption"]) == 12
    for family in hpo.FAMILIES:
        study = hpo._open_study(tmp_path / "studies" / family, family)
        assert len(study.trials) == 3
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        assert all(t.user_attrs["phase"] == "random" for t in study.trials)
    assert hpo.run_all(tmp_path, smoke=True) == result
    monkeypatch.setattr(hpo, "source_identity", lambda: {"test": "changed"})
    with pytest.raises(ValueError, match="identity differs"):
        hpo.run_all(tmp_path, smoke=True)
    assert json.loads((tmp_path / "progress.json").read_text())["target_per_study"] == 3


def test_seed_manifest_has_namespace_and_no_collisions():
    manifest = hpo.seed_manifest()
    records = manifest["streams"]
    assert len({r["seed"] for r in records}) == len(records)
    assert all(r["keys"][1] == "cw-parity-hpo-followup-20261009" for r in records)
    assert any(r["keys"][2:4] == ["sampler", "shared_parity"] for r in records)
