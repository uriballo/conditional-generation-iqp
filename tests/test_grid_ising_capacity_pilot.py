from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import enumerate_bitstrings
from iqp_cond.exact_tvd_campaign import data_initialize, expected_bodyness_sigmas
from iqp_cond.grid_ising import exact_probabilities
from iqp_cond.grid_ising_capacity_pilot import (
    ANCILLAS,
    CONDITIONS,
    _ancilla_support,
    _grid_metrics,
    identities,
    load_data_archive,
    load_manifest,
    manifest_digest,
    prepare_data_archive,
)

MANIFEST = Path("configs/grid_ising_capacity_pilot_v1.json")


def test_manifest_defines_eight_unique_content_addressed_runs() -> None:
    config = load_manifest(MANIFEST)
    runs = identities(config)
    assert len(runs) == 8
    assert len({run.run_id for run in runs}) == 8
    assert {(run.beta, run.field) for run in runs} == set(CONDITIONS)
    assert {run.ancillas for run in runs} == set(ANCILLAS)


def test_manifest_rejects_protocol_drift(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text())
    value["training"]["updates"] = 501
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="training protocol"):
        load_manifest(path)


def test_data_archive_is_deterministic_and_role_separated(tmp_path: Path) -> None:
    config = load_manifest(MANIFEST)
    first_path, first_digest = prepare_data_archive(config, tmp_path)
    second_path, second_digest = prepare_data_archive(config, tmp_path)
    assert first_path == second_path
    assert first_digest == second_digest
    archive = load_data_archive(config, tmp_path)
    assert archive["train"].shape == (4, 1000, 8)
    seeds = (
        set(archive["train_seeds"])
        | set(archive["validation_seeds"])
        | set(archive["test_seeds"])
    )
    assert len(seeds) == 12
    assert not np.array_equal(archive["train"], archive["validation"])
    assert archive["configuration_digest"].item().decode() == manifest_digest(config)


def test_data_initialization_only_uses_own_visible_split() -> None:
    generators = np.asarray(complete_ansatz(11).generators)
    support = _ancilla_support(generators, 8)
    samples = enumerate_bitstrings(8)[:100]
    params = data_initialize(samples, generators, support, seed=7)
    assert params.shape == (66,)
    assert np.any(params[support] != 0)
    assert np.max(np.abs(params[support])) < 0.02


def test_grid_metrics_are_exact_for_the_target() -> None:
    target = exact_probabilities((2, 4), 0.44, 0.0)
    metrics = _grid_metrics(
        target,
        target,
        (2, 4),
        expected_bodyness_sigmas(8),
        0.0,
    )
    assert metrics["bitstring_tvd"] == pytest.approx(0.0)
    assert metrics["magnetization_profile_tvd"] == pytest.approx(0.0)
    assert metrics["edge_correlation_rmse"] == pytest.approx(0.0)
    assert metrics["spin_flip_invariance_defect"] == pytest.approx(0.0)
