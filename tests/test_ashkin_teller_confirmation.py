"""Guard tests never evaluate an actual held-out Ashkin--Teller target."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from iqp_cond.artifacts import write_immutable_json

spec = importlib.util.spec_from_file_location(
    "confirmation",
    Path(__file__).resolve().parents[1] / "scripts/ashkin_teller_confirmation.py",
)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    original = run._laws
    calls = []

    def perimeter_only(conditions):
        np.testing.assert_array_equal(conditions, run.TRAIN)
        calls.append(conditions.copy())
        return original(conditions)

    monkeypatch.setattr(run, "_laws", perimeter_only)
    run.freeze(tmp_path)
    assert len(calls) == 1
    return tmp_path


def test_freeze_contains_no_test_target_laws(frozen):
    assert len(run.TRAIN) == 16 and len(run.TEST) == 9
    assert not set(map(tuple, run.TRAIN)) & set(map(tuple, run.TEST))
    assert set(map(tuple, run.TEST)) == set(map(tuple, run.TEST[:, ::-1]))
    for axis in (0, 1):
        assert set(run.TEST[:, axis]) <= set(run.TRAIN[:, axis])
    with np.load(frozen / "data.npz") as d:
        assert "test_targets" not in d.files
        assert d["samples"].shape == (10, 2, 16, 1000, 8)
        assert d["initial_params"].shape == (10, 30)
        assert not np.array_equal(d["samples"][0, 0], d["samples"][0, 1])
    assert run.freeze(frozen) == run.verify_manifest(frozen)


def test_no_target_access_without_complete_prediction_seal(frozen, monkeypatch):
    def forbidden(_):
        raise AssertionError("Held-out laws accessed before complete seal")

    monkeypatch.setattr(run, "_laws", forbidden)
    with pytest.raises(RuntimeError, match="must be sealed"):
        run.evaluate(frozen)
    write_immutable_json(frozen / "predictions-seal.json", {"incomplete": True})
    with pytest.raises(RuntimeError, match="Missing prediction bundle"):
        run.evaluate(frozen)
    assert not (frozen / "evaluation-started.json").exists()


def test_fitting_is_closed_after_evaluation_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "verify_manifest", lambda _: None)
    (tmp_path / "evaluation-started.json").write_text("{}")
    with pytest.raises(RuntimeError, match="further fitting is closed"):
        run.train(tmp_path, "cat", 2701)
    with pytest.raises(RuntimeError, match="further fitting is closed"):
        run.classical(tmp_path, 2701)
