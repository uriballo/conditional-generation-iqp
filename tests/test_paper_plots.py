from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_scaling():
    path = Path(__file__).parents[1] / "scripts/plot_paper_ancilla_scaling.py"
    spec = importlib.util.spec_from_file_location("paper_scaling", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows(seeds=None):
    if seeds is None:
        seeds = range(100, 110)
    return [
        {
            "dataset": "gaussian",
            "visible_qubits": 8,
            "ancillas": 0,
            "group": "held_out",
            "seed": seed,
            "conditioned_tvd": seed / 1000,
        }
        for seed in seeds
    ]


def test_scaling_values_requires_exact_unique_production_seeds():
    scaling = _load_scaling()
    values = scaling._values(_rows(), "gaussian", 8, 0, "held_out", "conditioned_tvd")
    assert values.shape == (10,)
    with pytest.raises(ValueError, match="unique seeds"):
        scaling._values(
            _rows(range(99, 109)), "gaussian", 8, 0, "held_out", "conditioned_tvd"
        )
    with pytest.raises(ValueError, match="unique seeds"):
        scaling._values(
            _rows() + _rows([108]),
            "gaussian",
            8,
            0,
            "held_out",
            "conditioned_tvd",
        )


def test_scaling_values_rejects_nonfinite_values():
    scaling = _load_scaling()
    rows = _rows()
    rows[3]["conditioned_tvd"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        scaling._values(rows, "gaussian", 8, 0, "held_out", "conditioned_tvd")


def test_condition_values_requires_ten_unique_seeds():
    scaling = _load_scaling()
    rows = [
        {
            "dataset": "gaussian",
            "visible_qubits": 8,
            "ancillas": 0,
            "condition": 0.25,
            "split": "training",
            "seed": seed,
            "conditioned_tvd": 0.1,
        }
        for seed in range(100, 110)
    ]
    assert (
        scaling._condition_values(rows, "gaussian", 8, 0, 0.25, "training").size == 10
    )
    with pytest.raises(ValueError, match="unique seeds"):
        scaling._condition_values(rows[:-1], "gaussian", 8, 0, 0.25, "training")


def test_n12_independent_profile_uses_r10_bank():
    path = Path(__file__).parents[1] / "scripts/plot_curie_weiss_n12_partial.py"
    spec = importlib.util.spec_from_file_location("cw12_profiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for ancillas, p in ((3, np.array([0.8, 0.2])), (10, np.array([0.3, 0.7]))):
        records.append(
            SimpleNamespace(
                identity=dict(
                    dataset="curie_weiss",
                    visible_qubits=12,
                    ancillas=ancillas,
                    family="independent",
                    seed=100,
                ),
                conditions=np.array([0.7]),
                at=lambda beta, p=p: (p, None, None),
            )
        )
    np.testing.assert_array_equal(module.independent(records, 0.7, 100), [0.3, 0.7])


def test_condition_response_renders_math_labels(tmp_path):
    import matplotlib as mpl

    scaling = _load_scaling()
    conditions = np.linspace(0.5, 1.5, 11)
    rows = [
        dict(
            dataset="curie_weiss",
            visible_qubits=n,
            ancillas=r,
            condition=float(beta),
            seed=seed,
            conditioned_tvd=0.2 / (r + 1),
            split="training" if i % 2 == 0 else "held_out",
        )
        for n in (8, 12)
        for r in scaling.ANCILLAS
        for i, beta in enumerate(conditions)
        for seed in range(100, 110)
    ]
    with mpl.rc_context({"text.usetex": False, "mathtext.fontset": "cm"}):
        scaling._condition_response({"condition_rows": rows}, tmp_path, "curie_weiss")
    assert (tmp_path / "curie_weiss_condition_response.pdf").stat().st_size > 0
    assert (tmp_path / "curie_weiss_condition_response.png").stat().st_size > 0
