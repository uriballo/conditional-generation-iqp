import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "ashkin_teller_structure_report",
    Path(__file__).parents[1] / "scripts/ashkin_teller_structure_report.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

diagnostic_metrics = _MODULE.diagnostic_metrics
product_marginals = _MODULE.product_marginals
shell_uniformize = _MODULE.shell_uniformize
summarize = _MODULE.summarize


def test_product_marginals_preserves_little_endian_layer_marginals():
    q = (np.arange(16, dtype=float)[None, :] + 1) / 136
    r = product_marginals(q)
    table = q.reshape(1, 4, 4)
    expected = table.sum(2)[:, :, None] * table.sum(1)[:, None, :]
    expected /= expected.sum((1, 2), keepdims=True)
    np.testing.assert_allclose(r, expected.reshape(1, 16))
    np.testing.assert_allclose(r.sum(1), 1)


def test_shell_uniformization_preserves_joint_shell_profile():
    q = np.arange(256, dtype=float)[None, :] + 1
    e = shell_uniformize(q)
    assert np.all(e >= 0)
    np.testing.assert_allclose(e.sum(1), q.sum(1))
    # Every state in a shell has the same output probability.
    n = 8
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    ka = bits[:, :4].sum(1)
    kb = bits[:, 4:].sum(1)
    for x, y in zip(ka, kb, strict=True):
        mask = (ka == x) & (kb == y)
        np.testing.assert_allclose(e[0, mask], e[0, mask][0])
    # A=0011 (index 12) and A=0101 (index 10) have equal weight but
    # different ring energies; the diagnostic shell deliberately identifies them.
    np.testing.assert_allclose(e[0, 12], e[0, 10])
    assert q[0, 12] != q[0, 10]


def test_diagnostics_can_have_positive_and_negative_signed_effects():
    rng = np.random.default_rng(1)
    positive = []
    for _ in range(100):
        q = rng.dirichlet(np.ones(16))[None, :]
        p = rng.dirichlet(np.ones(16))[None, :]
        metrics = diagnostic_metrics(q, p)
        positive.append(metrics["delta_spatial"][0])
        if metrics["delta_spatial"][0] > 0:
            assert np.isfinite(metrics["delta_dependence"][0])
            break
    assert any(value > 0 for value in positive)

    uniform = np.ones((1, 16)) / 16
    concentrated = uniform.copy()
    concentrated[0, 12] += 0.05
    concentrated[0, 10] -= 0.05
    assert diagnostic_metrics(concentrated, uniform)["delta_spatial"][0] < 0


def test_summary_reports_condition_and_seed_signs():
    values = np.array([[1, -1], [-1, -1]], dtype=float)
    result = summarize(values, np.array([[0.2, 0.2], [0.3, 0.3]]), seeds=(1, 2))
    assert result["negative_count"] == 3
    assert result["negative_condition_means"] == 1
