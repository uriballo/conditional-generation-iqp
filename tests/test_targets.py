from math import sqrt

import numpy as np
import pytest

from iqp_cond import GaussianHammingTarget
from iqp_cond.exact import enumerate_bitstrings


@pytest.mark.parametrize("mean", [19 / 2, 19 / 4])
def test_calibrated_target_has_exact_requested_mean(mean):
    target = GaussianHammingTarget.create(19, mean, sqrt(19) / 2)

    assert target.actual_mean == pytest.approx(mean, abs=1e-11)
    assert target.shell_probabilities.sum() == pytest.approx(1.0)
    assert np.all(target.shell_probabilities > 0)


def test_bitstring_probabilities_are_uniform_in_shell_and_normalized():
    target = GaussianHammingTarget.create(7, 7 / 4, sqrt(7) / 2)
    bitstrings = enumerate_bitstrings(7)
    probabilities = target.bitstring_probabilities(bitstrings)

    assert probabilities.sum() == pytest.approx(1.0)
    for weight in range(8):
        shell = probabilities[bitstrings.sum(axis=1) == weight]
        assert np.allclose(shell, shell[0])
        assert shell.sum() == pytest.approx(target.shell_probabilities[weight])


def test_sampling_recovers_shell_moments():
    target = GaussianHammingTarget.create(8, 2.0, sqrt(8) / 2)
    samples = target.sample(np.random.default_rng(3), 30_000)
    weights = samples.sum(axis=1)

    assert weights.mean() == pytest.approx(target.actual_mean, abs=0.03)
    assert weights.std() == pytest.approx(target.actual_std, abs=0.03)
