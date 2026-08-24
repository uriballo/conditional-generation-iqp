"""Tests for sparse-grid held-out interpolation with ancillas."""

from math import pi

import numpy as np
import pytest

from iqp_cond.exact import enumerate_bitstrings
from iqp_cond.interpolation import interpolation_targets
from iqp_cond.sparse_interpolation import (
    SPARSE2_HELDOUT_FRACTIONS,
    SPARSE2_TRAIN_FRACTIONS,
    SPARSE3_HELDOUT_FRACTIONS,
    SPARSE3_TRAIN_FRACTIONS,
    grid_fractions,
    oracle_adjacent_lookup,
)


def test_grid_fractions_reject_unknown_grid():
    with pytest.raises(ValueError, match="unknown sparse grid"):
        grid_fractions(4)


def test_three_peak_heldout_are_midpoints():
    for fraction in SPARSE3_HELDOUT_FRACTIONS:
        neighbors = SPARSE3_TRAIN_FRACTIONS[
            np.argsort(np.abs(SPARSE3_TRAIN_FRACTIONS - fraction), kind="stable")[:2]
        ]
        assert fraction == pytest.approx(float(np.mean(neighbors)))


def test_two_peak_heldout_contains_true_midpoint():
    assert 0.125 in SPARSE2_HELDOUT_FRACTIONS
    assert float(np.mean([0.0, 0.25])) == pytest.approx(0.125)
    assert set(SPARSE2_TRAIN_FRACTIONS).isdisjoint(SPARSE2_HELDOUT_FRACTIONS)


def test_grids_share_two_heldout_angles():
    shared = set(SPARSE2_HELDOUT_FRACTIONS) & set(SPARSE3_HELDOUT_FRACTIONS)
    assert shared == {0.0625, 0.1875}


def test_all_heldout_angles_lie_between_training_extremes():
    for train_fractions, heldout_fractions in (
        (SPARSE2_TRAIN_FRACTIONS, SPARSE2_HELDOUT_FRACTIONS),
        (SPARSE3_TRAIN_FRACTIONS, SPARSE3_HELDOUT_FRACTIONS),
    ):
        assert all(
            train_fractions.min() < fraction < train_fractions.max()
            for fraction in heldout_fractions
        )
        assert all(
            np.isclose(fraction * 32, np.round(fraction * 32))
            for fraction in heldout_fractions
        )


def test_oracle_lookup_uses_lower_of_two_adjacent_outputs():
    n_qubits = 5
    train_fractions = np.asarray([0.0, 0.5])
    heldout_fractions = np.asarray([0.25])
    targets = interpolation_targets(pi * heldout_fractions, n_qubits)
    bitstrings = enumerate_bitstrings(n_qubits)
    train_probs = np.zeros((2, 2**n_qubits))
    train_probs[0, 0] = 1.0
    train_probs[1, -1] = 1.0

    _shell_tvds, bitstring_tvds = oracle_adjacent_lookup(
        train_probs,
        train_fractions,
        heldout_fractions,
        targets,
        bitstrings,
    )
    target_probs = targets[0].bitstring_probabilities(bitstrings)
    first_tvd = np.abs(target_probs - train_probs[0]).sum() / 2
    second_tvd = np.abs(target_probs - train_probs[1]).sum() / 2
    assert bitstring_tvds[0] == pytest.approx(min(first_tvd, second_tvd))


def test_oracle_lookup_shape_mismatch_raises():
    targets = interpolation_targets(pi * np.asarray([0.25]), 5)
    bitstrings = enumerate_bitstrings(5)
    with pytest.raises(ValueError, match="one probability row per training angle"):
        oracle_adjacent_lookup(
            np.zeros((3, 32)),
            np.asarray([0.0, 0.5]),
            np.asarray([0.25]),
            targets,
            bitstrings,
        )
