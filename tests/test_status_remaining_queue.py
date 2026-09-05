import base64
import importlib.util
import json
import types
import zlib
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "dashboard", Path(__file__).parents[1] / "scripts/status_remaining_queue.py"
)
DASH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DASH)


def test_union_deduplicates_and_excludes_foreign_ids():
    runs = [dict(run_id="a", dataset="gaussian", visible_qubits=8)]
    hosts = {
        "58": dict(training_ids=["a", "mfc"], evaluation_ids=["a"]),
        "67": dict(training_ids=["a"], evaluation_ids=["a"]),
    }
    rows = DASH.campaign_progress(runs, hosts, ["a"], ["a"])
    assert sum(r["total"] for r in rows) == 1
    assert sum(r["trained"] for r in rows) == 1
    assert sum(r["evaluated"] for r in rows) == 1


def test_offline_counts_are_explicit_lower_bound():
    hosts = {"58": {"error": "timeout"}}
    output = DASH.render(hosts, [], [], now=0)
    assert "OFFLINE" in output and "lower bound" in output


def test_stale_state_detects_dead_processes_not_old_timestamp():
    snapshot = dict(
        state=dict(status="running", updated=0, active={"0": {"pid": 42}}),
        scheduler_pids=[12],
        workers=[{"pid": 42}],
    )
    assert DASH.warnings(snapshot) == []
    snapshot["workers"] = []
    snapshot["scheduler_pids"] = []
    assert len(DASH.warnings(snapshot)) == 2


def test_worker_details_and_units():
    worker = dict(
        index=3180,
        phase="train",
        elapsed=3660,
        cpus=list(range(8)),
        rss_gib=7.25,
        cpu_avg=640,
    )
    run = dict(
        family="conditioned",
        condition=None,
        dataset="curie_weiss",
        visible_qubits=12,
        ancillas=10,
        seed=100,
    )
    output = DASH.format_worker(worker, run, "queue")
    for fragment in ("3180", "shared", "n12 r10", "s100", "1h01m", "8 thr", "640%"):
        assert fragment in output


def test_probe_compiles_without_running():
    program = DASH.PROBE.replace("__ROOT__", repr("/tmp/example")).replace(
        "__HOST__", repr("58")
    )
    compile(program, "<remote probe>", "exec")


def test_remote_snapshot_decodes_compressed_response(monkeypatch):
    payload = {"state": {"status": "running"}, "workers": []}
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()

    def fake_run(command, **kwargs):
        assert command[-2:] == ["python3", "-"]
        assert "controls/remaining-20260903" in kwargs["input"]
        return types.SimpleNamespace(stdout=encoded)

    monkeypatch.setattr(DASH.subprocess, "run", fake_run)
    assert DASH.read("58") == ("58", payload)
