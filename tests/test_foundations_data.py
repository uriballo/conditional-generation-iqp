"""Exact finite-data contracts for the arXiv foundations study."""

from pathlib import Path

import numpy as np
import pytest

from iqp_cond.foundations_data import (
    derive_exact_split_seed,
    draw_exact_dataset,
    load_foundations_training_validation,
    prepare_foundations_datasets,
)
from iqp_cond.foundations_protocol import FoundationsManifest
from iqp_cond.ising_training import DatasetRole

MANIFEST_PATH = Path(__file__).parents[1] / "configs" / "arxiv_foundations_v1.json"
TARGETS = Path(__file__).parents[1] / "results" / "quantum-campaign-v1" / "targets"


@pytest.fixture(scope="module")
def manifest() -> FoundationsManifest:
    return FoundationsManifest.from_json(MANIFEST_PATH)


def test_exact_split_seeds_are_stable_and_role_condition_instance_distinct(manifest):
    seeds = {
        derive_exact_split_seed(manifest, coupling_seed, beta_index, role)
        for coupling_seed in (666, 101)
        for beta_index in (0, 1)
        for role in (DatasetRole.TRAINING, DatasetRole.VALIDATION)
    }
    assert len(seeds) == 8
    assert derive_exact_split_seed(manifest, 666, 0, DatasetRole.TRAINING) == (
        derive_exact_split_seed(manifest, 666, 0, DatasetRole.TRAINING)
    )


def test_exact_draws_use_only_the_declared_finite_sample_interface(manifest):
    bitstrings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    probabilities = np.array([0.0, 1.0, 0.0, 0.0])
    first = draw_exact_dataset(
        role=DatasetRole.TRAINING,
        coupling_seed=666,
        beta=manifest.beta_grid.training[0],
        data_seed=11,
        bitstrings=bitstrings,
        probabilities=probabilities,
        n_samples=5,
    )
    second = draw_exact_dataset(
        role=DatasetRole.TRAINING,
        coupling_seed=666,
        beta=manifest.beta_grid.training[0],
        data_seed=11,
        bitstrings=bitstrings,
        probabilities=probabilities,
        n_samples=5,
    )

    np.testing.assert_array_equal(first.samples, np.tile([0, 1], (5, 1)))
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.digest == second.digest
    with pytest.raises(ValueError, match="Held-out draws are forbidden"):
        draw_exact_dataset(
            role=DatasetRole.HELD_OUT,
            coupling_seed=666,
            beta=manifest.beta_grid.held_out[0],
            data_seed=12,
            bitstrings=bitstrings,
            probabilities=probabilities,
            n_samples=5,
        )


def test_prepared_foundations_archives_roundtrip_all_training_conditions(
    tmp_path,
    manifest,
):
    prepared = prepare_foundations_datasets(
        manifest,
        TARGETS,
        tmp_path / "datasets",
        coupling_seeds=(666,),
    )
    repeated = prepare_foundations_datasets(
        manifest,
        TARGETS,
        tmp_path / "datasets",
        coupling_seeds=(666,),
    )
    data = load_foundations_training_validation(
        tmp_path / "datasets",
        manifest,
        666,
        manifest.beta_grid.training,
    )

    assert prepared == repeated
    assert len(prepared) == 1
    assert data.stacked_training().shape == (5, 5_000, 16)
    assert data.stacked_validation().shape == (5, 2_000, 16)
    assert data.betas == manifest.beta_grid.training
    assert all(
        training.data_seed != validation.data_seed
        for training, validation in zip(
            data.training,
            data.validation,
            strict=True,
        )
    )
    assert not any("held" in key for key in data.dataset_hashes())
