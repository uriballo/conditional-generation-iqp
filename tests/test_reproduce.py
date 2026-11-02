import json

import reproduce


def test_check_flags_changed_and_missing_results(tmp_path, monkeypatch):
    data, run = tmp_path / "data", tmp_path / "run"
    stored = dict(curve=[0.1, 0.2], trial_id=3, seconds=10.0, sha256="a")
    for name in ("a.json", "b.json"):
        (data / "bundle").mkdir(parents=True, exist_ok=True)
        (data / "bundle" / name).write_text(json.dumps(stored))
    monkeypatch.setattr(reproduce, "CHECKS", [("out", "bundle")])
    (run / "out").mkdir(parents=True)
    # Timings and strings are provenance, not results.
    same = dict(stored, seconds=99.0, sha256="b")
    (run / "out" / "a.json").write_text(json.dumps(same))
    (run / "out" / "b.json").write_text(json.dumps(same))
    assert reproduce.check(run, data) == 0

    (run / "out" / "a.json").write_text(json.dumps(dict(same, curve=[0.1, 0.3])))
    (run / "out" / "b.json").unlink()
    assert reproduce.check(run, data) == 2
