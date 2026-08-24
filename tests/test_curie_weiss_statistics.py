import numpy as np
import pytest

from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.curie_weiss_statistics import (
    adjacent_profile_oracle,
    evidence_status,
    midpoint_mixture,
    paired_interval,
    thermodynamic_metrics,
)


def test_exact_metrics_and_thermodynamic_observables():
    target = exact_curie_weiss_target(8, 1.0)
    metrics = thermodynamic_metrics(target, target.probabilities)
    assert metrics.profile_tvd == pytest.approx(0.0)
    assert metrics.bitstring_tvd == pytest.approx(0.0)
    assert metrics.susceptibility > 0 and np.isfinite(metrics.binder_cumulant)
    assert metrics.heat_capacity_per_spin >= 0


def test_midpoint_mixture_is_target_blind_and_oracle_uses_profile_tvd():
    training = np.asarray([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    refs = np.stack(
        [exact_curie_weiss_target(8, beta).probabilities for beta in training]
    )
    assert np.allclose(midpoint_mixture(training, refs, 1.0), 0.5 * (refs[2] + refs[3]))
    _, selected = adjacent_profile_oracle(
        training, refs, exact_curie_weiss_target(8, 1.0)
    )
    assert selected in {2, 3}


def test_paired_bootstrap_is_frozen_and_status_boundaries_are_conservative():
    interval = paired_interval({seed: 0.1 for seed in range(10)})
    assert interval.bootstrap_seed == 20_260_812
    assert evidence_status(interval) == "supported"
    zero = paired_interval({seed: 0.0 for seed in range(10)})
    assert evidence_status(zero) == "mixed"
