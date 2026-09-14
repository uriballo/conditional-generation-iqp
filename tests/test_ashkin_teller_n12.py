"""Guard tests for the n=12 interpolation protocol."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from iqp_cond.artifacts import write_immutable_json

spec = importlib.util.spec_from_file_location(
    "ashkin_teller_n12",
    Path(__file__).resolve().parents[1] / "scripts/ashkin_teller_n12.py",
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


def test_freeze_has_n12_shapes_and_no_test_targets(frozen):
    assert run.BITS.shape == (4096, 12)
    assert len(run.TRAIN) == 16 and len(run.TEST) == 9
    assert not set(map(tuple, run.TRAIN)) & set(map(tuple, run.TEST))
    with np.load(frozen / "data.npz") as data:
        assert "test_targets" not in data.files
        assert data["samples"].shape == (10, 2, 16, 1000, 12)
        assert data["generators"].shape == (78, 12)
        assert data["generators_full"].shape == (78, 12)
        assert data["initial_params"].shape == (10, 78)
        assert data["mapping"].shape == (78,)


def test_no_target_access_before_complete_seal(frozen, monkeypatch):
    def forbidden(_):
        raise AssertionError("held-out target accessed")

    monkeypatch.setattr(run, "_laws", forbidden)
    with pytest.raises(RuntimeError, match="must be sealed"):
        run.evaluate(frozen)
    write_immutable_json(frozen / "predictions-seal.json", {"incomplete": True})
    with pytest.raises(RuntimeError, match="Missing prediction bundle"):
        run.evaluate(frozen)
    assert not (frozen / "evaluation-started.json").exists()


def test_fitting_closes_after_evaluation_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "verify_manifest", lambda _: None)
    (tmp_path / "evaluation-started.json").write_text("{}")
    with pytest.raises(RuntimeError, match="further fitting is closed"):
        run.train(tmp_path, "cat", 3711)
    with pytest.raises(RuntimeError, match="further fitting is closed"):
        run.classical(tmp_path, 3711)
