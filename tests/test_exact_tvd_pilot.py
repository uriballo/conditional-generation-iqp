from pathlib import Path

import pytest

from iqp_cond.exact_tvd_campaign import load_manifest, manifest_digest
from iqp_cond.exact_tvd_pilot import (
    calibration_passes,
    select_n_samples,
    select_optimizer,
    write_selection,
)
from iqp_cond.exact_tvd_runner import load_protocol_selection


def manifest():
    return {
        "estimator_calibration": {
            "candidates": [400, 800],
            "ancillas": [0, 10],
            "development_seeds": [0, 1, 2],
            "max_pauli_rmse": 0.01,
            "max_mmd_cv": 0.05,
            "min_gradient_cosine": 0.95,
            "max_relative_gradient_error": 0.1,
        },
        "optimizer_pilot": {
            "matched_cells": 18,
            "minimum_median_improvement": 0.02,
            "minimum_matched_cell_wins": 12,
            "schedule_free_adamw": [{"learning_rate": 0.01, "b1": 0.9}],
        },
    }


def test_all_calibration_gates():
    assert calibration_passes(
        {
            "pauli_expectation_rmse": 0.01,
            "mmd_cv": 0.05,
            "gradient_cosine": 0.95,
            "relative_gradient_error": 0.1,
        },
        manifest()["estimator_calibration"],
    )
    assert not calibration_passes(
        {
            "pauli_expectation_rmse": 0.011,
            "mmd_cv": 0.01,
            "gradient_cosine": 1,
            "relative_gradient_error": 0,
        },
        manifest()["estimator_calibration"],
    )


def test_smallest_passing_candidate():
    rows = []
    for d in ("gaussian", "curie_weiss", "mfc"):
        for a in (0, 10):
            for s in (0, 1, 2):
                for n in (400, 800):
                    rows.append(
                        {
                            "dataset": d,
                            "ancillas": a,
                            "seed": s,
                            "n_samples": n,
                            "pauli_expectation_rmse": 0.02 if n == 400 else 0.001,
                            "mmd_cv": 0.01,
                            "gradient_cosine": 0.99,
                            "relative_gradient_error": 0.01,
                        }
                    )
    assert select_n_samples(rows, manifest()) == 800


def test_calibration_requires_complete_grid():
    with pytest.raises(ValueError):
        select_n_samples([], manifest())


def test_optimizer_requires_18_cells_and_adopts():
    rows = []
    keys = [
        (d, n, a, s)
        for d, n in (("gaussian", 8), ("curie_weiss", 8), ("mfc", 3))
        for a in (0, 10)
        for s in (0, 1, 2)
    ]
    for d, n, a, s in keys:
        base = {
            "dataset": d,
            "visible_qubits": n,
            "ancillas": a,
            "seed": s,
            "optimizer": "adam_cosine",
            "validation_tvd": 1.0,
        }
        rows.append(base)
        rows.append(
            {
                **base,
                "optimizer": "schedule_free_adamw",
                "learning_rate": 0.01,
                "b1": 0.9,
                "validation_tvd": 0.9,
            }
        )
    assert select_optimizer(rows, manifest())["optimizer"] == "schedule_free_adamw"


def test_protocol_selection_is_immutable_and_production_readable(tmp_path: Path):
    full_manifest = load_manifest(Path("configs/exact_tvd_ancilla_v1.json"))
    path = tmp_path / "selection.json"
    optimizer = {
        "optimizer": "adam_cosine",
        "learning_rate": 0.01,
        "b1": 0.9,
        "median_improvement": 0.0,
        "wins": 0,
        "candidates": [],
    }
    preflight = {
        "passed": True,
        "total_qubits": 26,
        "peak_rss_bytes": 8 * 2**30,
        "configuration_digest": manifest_digest(full_manifest),
    }
    write_selection(
        path,
        full_manifest,
        {"selected_n_samples": 8192},
        optimizer,
        [preflight],
    )
    loaded = load_protocol_selection(path, full_manifest)
    assert loaded["n_samples"] == 8192
    assert loaded["optimizer"] == "adam_cosine"
    with pytest.raises(FileExistsError):
        write_selection(
            path,
            full_manifest,
            {"selected_n_samples": 8192},
            optimizer,
            [preflight],
        )
