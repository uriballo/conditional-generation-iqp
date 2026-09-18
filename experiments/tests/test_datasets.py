"""Target-family calibration and the frozen condition grids."""

from __future__ import annotations

import numpy as np

from experiments.common import datasets


def test_one_parameter_target_reproduces_its_moments() -> None:
    for n_visible in (8, 12):
        for condition in (0.0, 0.5, 1.0):
            target = datasets.gaussian_target(n_visible, 0.5, condition)
            mean = datasets.target_mean(n_visible, condition)
            assert abs(target.actual_mean - mean) < 1e-9
            assert 0.0 < target.actual_std < n_visible / 2
            assert abs(target.shell_probabilities.sum() - 1.0) < 1e-12


def test_target_bitstring_probabilities_match_the_shell_law() -> None:
    target = datasets.gaussian_target(4, 0.5, 0.25)
    states = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    probabilities = target.bitstring_probabilities(states)
    for row, state in enumerate(states):
        weight = int(state.sum())
        expected = target.shell_probabilities[weight] / len(
            [None for index in range(2**4) if bin(index).count("1") == weight],
        )
        assert probabilities[row] == expected


def test_two_parameter_grid_structure() -> None:
    grid = datasets.grid_conditions()
    assert grid.shape == (25, 2)
    training = datasets.training_conditions()
    held_out = datasets.held_out_conditions()
    assert training.shape == (16, 2)
    assert held_out.shape == (9, 2)
    assert all(datasets.is_perimeter(condition) for condition in training)
    assert not any(datasets.is_perimeter(condition) for condition in held_out)


def test_two_parameter_target_reproduces_mean_and_deviation() -> None:
    for condition in ((0.375, 0.30), (0.625, 0.45), (0.5, 0.375)):
        for n_visible in (8, 12):
            target = datasets.prescribed_target(n_visible, condition)
            u, v = condition
            assert abs(target.actual_mean - u * n_visible) < 1e-8
            assert abs(target.actual_std - v * np.sqrt(n_visible)) < 1e-8
