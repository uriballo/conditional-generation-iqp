import numpy as np
import pytest

from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.curie_weiss_statistics import thermodynamic_metrics


def test_exact_metrics_and_thermodynamic_observables():
    target = exact_curie_weiss_target(8, 1.0)
    metrics = thermodynamic_metrics(target, target.probabilities)
    assert metrics.profile_tvd == pytest.approx(0.0)
    assert metrics.bitstring_tvd == pytest.approx(0.0)
    assert metrics.susceptibility > 0 and np.isfinite(metrics.binder_cumulant)
    assert metrics.heat_capacity_per_spin >= 0
