import pytest

from iqp_cond.gaussian_statistics import paired_bootstrap_interval


def test_paired_bootstrap_is_deterministic_and_boundary_exact():
    values = {seed: float(seed + 1) for seed in range(10)}
    first = paired_bootstrap_interval(values)
    second = paired_bootstrap_interval(values)
    assert first == second
    assert first.estimate == 5.5
    assert first.lower < first.estimate < first.upper
    with pytest.raises(ValueError):
        paired_bootstrap_interval({0: 1.0})
