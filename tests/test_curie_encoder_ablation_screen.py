import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from iqp_cond.protocol import ProtocolValidationError

_SPEC = importlib.util.spec_from_file_location(
    "ablation_screen",
    Path(__file__).parents[1] / "scripts/run_curie_encoder_ablation_screen.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Screen, main = _MODULE.Screen, _MODULE.main


def test_screen_selects_exact_coordinates_and_reuses_completed_ids():
    screen = Screen()
    rows = screen.plan()
    assert len(rows) == 24
    assert {(row["visible_qubits"], row["ancillas"], row["seed"]) for row in rows} == {
        (8, ancillas, seed) for ancillas in (0, 10) for seed in (100, 101, 102)
    }
    assert len({row["run_id"] for row in rows}) == 24
    assert all(row["manifest_digest"] == screen.parent_digest for row in rows)
    existing = {
        row["run_id"]
        for row in rows
        if row["arm"] == "product_generic_affine" and row["seed"] == 100
    }
    assert len(existing) == 2


def test_screen_rejects_out_of_screen_selection():
    with pytest.raises(ValueError, match="outside the screen"):
        main(["train", "--index", "24"])
    with pytest.raises(ValueError, match="outside the screen"):
        main(["evaluate", "--run-id", "not-a-parent-run"])


def test_train_requires_exactly_one_selector():
    with pytest.raises(ValueError, match="exactly one"):
        main(["train"])
    with pytest.raises(ValueError, match="exactly one"):
        main(["train", "--index", "0", "--run-id", "x"])


def test_screen_pins_parent_manifest_digest(monkeypatch):
    monkeypatch.setattr(
        _MODULE.AblationManifest,
        "from_json",
        lambda _: SimpleNamespace(digest="0" * 64),
    )
    with pytest.raises(ProtocolValidationError, match="parent manifest digest differs"):
        Screen()
