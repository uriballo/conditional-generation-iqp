import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_exact_tvd_explicit_queue.py"
SPEC = importlib.util.spec_from_file_location("run_exact_tvd_explicit_queue", SCRIPT)
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def _plan(*jobs, deadline=0):
    return {
        "host": "test-host",
        "repository": "/tmp/test-repository",
        "deadline": deadline,
        "blocks": [
            {
                "name": "test-block",
                "training_workers": 1,
                "evaluation_workers": 1,
                "jobs": list(jobs),
            }
        ],
    }


def _job(index, run_id=None):
    return {"index": index, "run_id": run_id or f"run-{index}"}


def test_validate_accepts_distinct_in_scope_indices_and_run_ids():
    jobs = [_job(0), _job(3899)]

    assert QUEUE.validate(_plan(*jobs)) == jobs


def test_temperature_guard_defaults_on_and_requires_explicit_opt_out():
    assert QUEUE.temperature_limit_reached({}, [90.0])
    assert not QUEUE.temperature_limit_reached({}, [89.9])
    assert not QUEUE.temperature_limit_reached({}, [])
    assert not QUEUE.temperature_limit_reached(
        {"enforce_temperature_limit": False}, [91.0]
    )
    plan = _plan(_job(1))
    plan["enforce_temperature_limit"] = "false"
    with pytest.raises(ValueError, match="must be boolean"):
        QUEUE.validate(plan)


@pytest.mark.parametrize(
    "jobs, message",
    [
        ([_job(1), _job(1, "run-2")], "duplicate identities"),
        ([_job(1), _job(2, "run-1")], "duplicate run IDs"),
        ([_job(3900)], "out-of-scope identity"),
    ],
)
def test_validate_rejects_duplicate_or_out_of_scope_jobs(jobs, message):
    with pytest.raises(ValueError, match=message):
        QUEUE.validate(_plan(*jobs))


def test_worker_cpus_splits_configured_cpu_set_by_slot():
    cpu_set = list(range(8)) + list(range(16, 24))
    plan = _plan(_job(1))
    plan["cpu_set"] = cpu_set

    groups = [QUEUE.worker_cpus(plan, 4, slot) for slot in range(4)]

    assert groups == [[0, 4, 16, 20], [1, 5, 17, 21], [2, 6, 18, 22], [3, 7, 19, 23]]
    assert {cpu for group in groups for cpu in group} == set(cpu_set)
    assert len({cpu for group in groups for cpu in group}) == sum(map(len, groups))


@pytest.mark.parametrize("cpu_set", [[], [0, 0], [0, 32]])
def test_validate_rejects_invalid_cpu_sets(cpu_set):
    plan = _plan(_job(1))
    plan["cpu_set"] = cpu_set

    with pytest.raises(ValueError, match="invalid CPU allocation"):
        QUEUE.validate(plan)


def test_validate_rejects_worker_count_exceeding_cpu_set():
    plan = _plan(_job(1))
    plan["cpu_set"] = list(range(4))
    plan["blocks"][0]["training_workers"] = 5

    with pytest.raises(ValueError, match="workers exceed assigned CPUs"):
        QUEUE.validate(plan)


def test_old_cpu_groups_partition_all_32_cpus_exactly():
    groups = QUEUE.old_cpu_groups()

    assert len(groups) == 10
    assert [cpu for group in groups for cpu in group] == list(range(32))


def test_slot_complete_requires_every_slot_artifact(tmp_path):
    canonical = {
        index: {"run_id": f"run-{index}"}
        for index in range(3060, 3180, 4)
    }
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    slot_indices = list(range(3060, 3180, 4))[2::10]

    assert not QUEUE.slot_complete(2, canonical, tmp_path)
    for index in slot_indices:
        (evaluations / f"run-{index}.json").write_text("{}")

    assert QUEUE.slot_complete(2, canonical, tmp_path)


@pytest.mark.parametrize(
    "state, start_ticks, expected",
    [("S", 123, True), ("S", 124, False), ("Z", 123, False)],
)
def test_process_matches_checks_state_and_start_ticks(
    monkeypatch, state, start_ticks, expected
):
    record = {"pid": 42, "start_ticks": 123}

    def fake_read_text(self):
        assert str(self) == "/proc/42/stat"
        return (
            f"42 (worker process) {state} "
            + " ".join(["0"] * 18)
            + f" {start_ticks}"
        )

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert QUEUE.process_matches(record) is expected


def test_process_matches_returns_false_for_missing_process(monkeypatch):
    def missing_read_text(self):
        raise FileNotFoundError

    monkeypatch.setattr(Path, "read_text", missing_read_text)

    assert not QUEUE.process_matches({"pid": 42, "start_ticks": 123})


def _mock_checks(calls):
    def check_output(command, **kwargs):
        calls.append(command)
        if command[:2] == [
            sys.executable,
            str(Path("/tmp/test-repository") / "scripts/run_exact_tvd_campaign.py"),
        ]:
            return json.dumps([{}, {"run_id": "run-1"}])
        if command[:3] == ["ps", "-eo", "pid,args"]:
            return ""
        raise AssertionError(f"unexpected subprocess check: {command}")

    return check_output


def test_expired_deadline_saves_paused_status_without_launching_workers(
    tmp_path, monkeypatch
):
    plan_path = tmp_path / "queue.json"
    plan_path.write_text(json.dumps(_plan(_job(1))))
    calls = []
    monkeypatch.setattr(subprocess, "check_output", _mock_checks(calls))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(plan_path)])

    QUEUE.main()

    status = json.loads((tmp_path / "queue.status.json").read_text())
    assert status["status"] == "paused"
    assert not any(command and command[0] == "taskset" for command in calls)


def test_no_jobs_reports_complete_with_mocked_subprocess_checks(tmp_path, monkeypatch):
    plan_path = tmp_path / "queue.json"
    plan_path.write_text(json.dumps(_plan(deadline=10**12)))
    calls = []
    monkeypatch.setattr(subprocess, "check_output", _mock_checks(calls))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(plan_path)])

    QUEUE.main()

    status = json.loads((tmp_path / "queue.status.json").read_text())
    assert status["status"] == "complete"
    assert any(command[:3] == ["ps", "-eo", "pid,args"] for command in calls)


def _backfill_fixture(tmp_path, *selected, slots=(0,), jobs=None):
    old_indices = list(range(3060, 3180, 4))
    canonical = {index: {"run_id": f"canonical-{index}"} for index in old_indices}
    canonical.update({index: {"run_id": f"canonical-{index}"} for index in selected})
    out = tmp_path / "results"
    evaluations = out / "evaluations"
    evaluations.mkdir(parents=True)
    for index in old_indices[slots[0] :: 10]:
        (evaluations / f"canonical-{index}.json").write_text("{}")
    queue_jobs = jobs or [
        {"index": index, "run_id": f"run-{index}", "family": "independent"}
        for index in selected
    ]
    plan = _plan(*queue_jobs)
    plan["backfill"] = {
        "finished_slots": list(slots),
        "cpu_groups": [list(range(0, 4)), list(range(4, 8))]
        if tuple(slots) == (0, 1)
        else [list(range(0, 4))],
        "indices": list(selected),
    }
    return plan, canonical, out


def test_validate_backfill_accepts_finished_slot_and_queued_independent_indices(
    tmp_path,
):
    plan, canonical, out = _backfill_fixture(tmp_path, 3180)

    QUEUE.validate_backfill(plan, canonical, out)


@pytest.mark.parametrize(
    "mutate, message",
    [
        ("unfinished", "old slot has unfinished work"),
        ("cpu", "CPU groups do not match old worker slots"),
        ("duplicate", "backfill must be a unique subset of this host queue"),
        ("conditioned", "backfill is restricted to independent CW n12 r10 fits"),
        ("out-of-host", "backfill must be a unique subset of this host queue"),
    ],
)
def test_validate_backfill_rejects_invalid_claims(tmp_path, mutate, message):
    jobs = None
    selected = (3180,)
    if mutate == "duplicate":
        selected = (3180, 3180)
        jobs = [{"index": 3180, "run_id": "run-3180", "family": "independent"}]
    elif mutate == "conditioned":
        jobs = [{"index": 3180, "run_id": "run-3180", "family": "conditioned"}]
    elif mutate == "out-of-host":
        selected = (3181,)
        jobs = [{"index": 3180, "run_id": "run-3180", "family": "independent"}]
    plan, canonical, out = _backfill_fixture(tmp_path, *selected, jobs=jobs)
    if mutate == "unfinished":
        (out / "evaluations" / "canonical-3140.json").unlink()
    elif mutate == "cpu":
        plan["backfill"]["cpu_groups"] = [list(range(1, 5))]

    with pytest.raises(ValueError, match=message):
        QUEUE.validate_backfill(plan, canonical, out)
