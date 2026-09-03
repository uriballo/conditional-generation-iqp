import numpy as np
import pytest

from iqp_cond.exact_tvd_campaign import (
    data_initialize,
    expected_bodyness_sigmas,
    load_manifest,
    production_identities,
    split_indices,
)


def test_manifest_and_exact_run_count():
    manifest = load_manifest(
        __import__("pathlib").Path("configs/exact_tvd_ancilla_v1.json")
    )
    runs = production_identities(manifest)
    assert len(runs) == 4300
    assert len(set(runs)) == 4300


def test_splits_are_disjoint_and_reproducible():
    first = split_indices(5000, seed=7)
    second = split_indices(5000, seed=7)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert len(set(np.concatenate(tuple(first.values())))) == 3000


def test_bodyness_sigmas_and_validation():
    sigmas = expected_bodyness_sigmas(8)
    assert len(sigmas) == 3 and all(np.isfinite(sigmas))
    with pytest.raises(ValueError):
        expected_bodyness_sigmas(1)


def test_ancilla_noise_only_touches_marked_supports():
    data = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
    generators = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]])
    mask = np.array([False, False, False, True])
    result = data_initialize(data, generators, mask, 3)
    baseline = data_initialize(data, generators, mask, 3)
    assert np.array_equal(result, baseline)
    assert result[3] != 0.0
