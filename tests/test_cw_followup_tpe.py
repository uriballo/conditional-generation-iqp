import hashlib
import json

import optuna
import pytest

from paper_experiments import cw_followup_hpo as parent
from paper_experiments import cw_followup_tpe as tpe


def _identity():
    return dict(
        paper_sources={"fit.py": "unchanged", "run_experiments.py": "old"},
        library_sources={"lib.py": "unchanged"},
        backend="test",
    )


def _worker(arguments):
    context, proposal, root = arguments
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for job in parent.jobs(context, proposal):
        directory = root / "fits" / parent.digest(job)
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "parameters.npz"
        artifact.write_bytes(b"fixture parameters")
        betas = [job["beta"]] if job["beta"] is not None else [0.05, 0.89, 1.975]
        tvd = 0.1 + 0.01 * proposal["trial_id"]
        row = dict(
            job=job,
            directory=str(directory),
            status="ok",
            metrics_curves=dict(
                beta=betas,
                tvd=[tvd] * len(betas),
                magnetization_error=[0.02] * len(betas),
                heat_error=[0.03] * len(betas),
            ),
            artifact_sha256={
                "parameters.npz": hashlib.sha256(artifact.read_bytes()).hexdigest()
            },
        )
        parent.write_json(directory / "metrics.json", row)
        records.append(row)
    score = parent.score_records(records, context["family"], 1)
    row = dict(**proposal, **score, fits=[r["directory"] for r in records])
    parent.write_json(root / "proposals" / f"{proposal['trial_id']:03d}.json", row)
    return context["family"], row


def _random(root, monkeypatch):
    monkeypatch.setattr(parent, "source_identity", _identity)
    monkeypatch.setattr(parent, "_trial_worker", _worker)
    parent.run_all(root, smoke=True)


def test_continuation_preserves_random_history_and_stops_at_budget(
    tmp_path, monkeypatch
):
    _random(tmp_path, monkeypatch)
    baseline = (tmp_path / "campaign.json").read_bytes()
    frozen = (tmp_path / "random-proposals.json").read_bytes()
    calls = []

    def adaptive_worker(arguments):
        family = arguments[0]["family"]
        number = arguments[1]["trial_id"]
        study = tpe._load_study(tmp_path, family)
        assert number >= 3
        assert all(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials[:number]
        )
        assert len(study.trials) == number + 1
        calls.append((family, number))
        return _worker(arguments)

    monkeypatch.setattr(tpe, "_evaluate_trial", adaptive_worker)
    result = tpe.run_all(tmp_path, smoke=True)
    assert result["status"] == "paused_for_tpe_review"
    assert result["counts"] == dict.fromkeys(parent.FAMILIES, 6)
    assert len(calls) == 12
    assert (tmp_path / "campaign.json").read_bytes() == baseline
    assert (tmp_path / "random-proposals.json").read_bytes() == frozen
    for family in parent.FAMILIES:
        study = tpe._load_study(tmp_path, family)
        assert [t.user_attrs["phase"] for t in study.trials] == ["random"] * 3 + [
            "tpe"
        ] * 3
        backup = optuna.load_study(
            study_name=family,
            storage="sqlite:///"
            + str(tmp_path / "random-stage-backup" / family / "trials.sqlite3"),
        )
        assert len(backup.trials) == 3
    assert not (tmp_path / "complete.json").exists()
    assert result["confirmation_started"] is False and result["final_started"] is False
    assert tpe.run_all(tmp_path, smoke=True) == result
    assert len(calls) == 12
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["phase"] == "tpe" and progress["target_per_study"] == 6


def test_interrupted_adaptive_sequence_resumes_without_redraw(tmp_path, monkeypatch):
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "resume"
    _random(uninterrupted, monkeypatch)
    _random(interrupted, monkeypatch)
    monkeypatch.setattr(tpe, "_evaluate_trial", _worker)
    tpe.run_all(uninterrupted, smoke=True)
    calls = []

    def interrupt(arguments):
        calls.append((arguments[0]["family"], arguments[1]["trial_id"]))
        if len(calls) == 3:
            raise RuntimeError("fixture interruption")
        return _worker(arguments)

    monkeypatch.setattr(tpe, "_evaluate_trial", interrupt)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        tpe.run_all(interrupted, smoke=True)
    monkeypatch.setattr(tpe, "_evaluate_trial", _worker)
    tpe.run_all(interrupted, smoke=True)
    for family in parent.FAMILIES:
        a = tpe._load_study(uninterrupted, family)
        b = tpe._load_study(interrupted, family)
        assert [t.params for t in a.trials] == [t.params for t in b.trials]
        assert [t.value for t in a.trials] == [t.value for t in b.trials]


def test_rejects_changed_fit_code_and_missing_review(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="review checkpoint"):
        tpe.run_all(tmp_path, smoke=True)
    _random(tmp_path, monkeypatch)
    changed = _identity()
    changed["paper_sources"]["fit.py"] = "different"
    monkeypatch.setattr(parent, "source_identity", lambda: changed)
    with pytest.raises(ValueError, match="fit/source identity"):
        tpe.run_all(tmp_path, smoke=True)
    changed = _identity()
    changed["backend"] = "different"
    monkeypatch.setattr(parent, "source_identity", lambda: changed)
    with pytest.raises(ValueError, match="numerical environment"):
        tpe.run_all(tmp_path, smoke=True)
    allowed = _identity()
    allowed["paper_sources"]["run_experiments.py"] = "new dispatcher"
    allowed["paper_sources"]["paper_experiments/cw_followup_tpe.py"] = "new coordinator"
    assert (
        set(tpe._source_transition(_identity(), allowed)) == tpe._ALLOWED_ORCHESTRATION
    )


def test_corrupt_random_artifact_halts_before_any_adaptive_ask(tmp_path, monkeypatch):
    _random(tmp_path, monkeypatch)
    artifact = next((tmp_path / "studies").rglob("parameters.npz"))
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact"):
        tpe.run_all(tmp_path, smoke=True)
    assert all(len(tpe._load_study(tmp_path, f).trials) == 3 for f in parent.FAMILIES)


def test_orphaned_adaptive_proposal_is_not_rerolled(tmp_path, monkeypatch):
    _random(tmp_path, monkeypatch)
    monkeypatch.setattr(tpe, "_evaluate_trial", _worker)
    original_save = tpe._save_state
    failed = False

    def crash(path, sampler, proposal, identity):
        nonlocal failed
        if proposal is not None and not failed:
            failed = True
            raise RuntimeError("crash before proposal checkpoint")
        return original_save(path, sampler, proposal, identity)

    monkeypatch.setattr(tpe, "_save_state", crash)
    with pytest.raises(RuntimeError, match="before proposal"):
        tpe.run_all(tmp_path, smoke=True)
    monkeypatch.setattr(tpe, "_save_state", original_save)
    with pytest.raises(ValueError, match="orphan adaptive proposal"):
        tpe.run_all(tmp_path, smoke=True)
    assert len(tpe._load_study(tmp_path, parent.FAMILIES[0]).trials) == 4


def test_resume_after_tell_before_sampler_checkpoint(tmp_path, monkeypatch):
    _random(tmp_path, monkeypatch)
    monkeypatch.setattr(tpe, "_evaluate_trial", _worker)
    original_save = tpe._save_state
    failed = False

    def crash(path, sampler, proposal, identity):
        nonlocal failed
        family = path.stem
        if proposal is None and not failed:
            study = tpe._load_study(tmp_path, family)
            if (
                len(study.trials) == 4
                and study.trials[-1].state == optuna.trial.TrialState.COMPLETE
            ):
                failed = True
                raise RuntimeError("crash after tell")
        return original_save(path, sampler, proposal, identity)

    monkeypatch.setattr(tpe, "_save_state", crash)
    with pytest.raises(RuntimeError, match="after tell"):
        tpe.run_all(tmp_path, smoke=True)
    first = tpe._load_study(tmp_path, parent.FAMILIES[0]).trials[3].params
    monkeypatch.setattr(tpe, "_save_state", original_save)
    result = tpe.run_all(tmp_path, smoke=True)
    assert result["counts"] == dict.fromkeys(parent.FAMILIES, 6)
    assert tpe._load_study(tmp_path, parent.FAMILIES[0]).trials[3].params == first


def test_sampler_checkpoint_corruption_halts(tmp_path, monkeypatch):
    _random(tmp_path, monkeypatch)
    monkeypatch.setattr(tpe, "_evaluate_trial", _worker)
    tpe.run_all(tmp_path, smoke=True)
    path = tmp_path / "tpe-samplers" / (parent.FAMILIES[0] + ".json")
    state = json.loads(path.read_text())
    state["sampler_sha256"] = "0" * 64
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="sampler checkpoint corrupt"):
        tpe.run_all(tmp_path, smoke=True)
