import numpy as np

from iqp_cond.ashkin_teller_baselines import (
    composition_predict,
    expand_orbits,
    fit_baselines,
    loglinear_objective,
    orbit_counts,
)


def _conditions():
    levels = (0.1, 0.22, 0.38, 0.54, 0.7)
    return np.asarray(
        [(a, 0.1) for a in levels]
        + [(a, 0.7) for a in levels]
        + [(0.1, b) for b in levels[1:-1]]
        + [(0.7, b) for b in levels[1:-1]]
    )


def test_orbits_count_and_expansion_are_normalized_and_symmetric():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 2, (37, 8))
    counts = orbit_counts(x)
    assert counts.shape == (64,)
    assert counts.sum() == 37
    q = np.arange(64, dtype=float) + 1
    q /= q.sum()
    full = expand_orbits(q)
    assert np.isclose(full.sum(), 1.0)
    for state in range(256):
        assert full[state] == full[state ^ 15] == full[state ^ 240] == full[state ^ 255]


def test_loglinear_gradient_matches_finite_difference_with_l2():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(5, 3))
    counts = rng.integers(1, 20, size=(5, 64))
    z = rng.normal(size=189)
    _, grad = loglinear_objective(z, features, counts, 0.07)
    eps = 1e-6
    numeric = np.empty_like(z)
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        numeric[i] = (
            loglinear_objective(zp, features, counts, 0.07)[0]
            - loglinear_objective(zm, features, counts, 0.07)[0]
        ) / (2 * eps)
    np.testing.assert_allclose(grad, numeric, rtol=2e-5, atol=2e-6)


def test_composition_recovers_affine_orbit_logits():
    rng = np.random.default_rng(4)
    conditions = _conditions()
    base, ax, by = rng.normal(size=(3, 64))
    logits = (
        base + np.sin(conditions[:, :1] * 11) * ax + np.cos(conditions[:, 1:] * 7) * by
    )
    laws = np.exp(logits - logits.max(1, keepdims=True))
    laws /= laws.sum(1, keepdims=True)
    query = np.asarray([[0.22, 0.38], [0.54, 0.38]])
    expected_logits = (
        base + np.sin(query[:, :1] * 11) * ax + np.cos(query[:, 1:] * 7) * by
    )
    expected = np.exp(expected_logits - expected_logits.max(1, keepdims=True))
    expected /= expected.sum(1, keepdims=True)
    np.testing.assert_allclose(
        composition_predict(conditions, laws, query), expected, atol=1e-12
    )


def test_small_fit_has_shapes_and_convergence_metadata():
    rng = np.random.default_rng(5)
    conditions = _conditions()
    training = rng.integers(0, 2, (16, 1000, 8), dtype=np.int8)
    validation = rng.integers(0, 2, (16, 1000, 8), dtype=np.int8)
    predictions, metadata = fit_baselines(
        conditions, training, validation, np.asarray([[0.22, 0.38]])
    )
    assert predictions["log_composition"].shape == (1, 256)
    assert predictions["log_linear"].shape == (1, 256)
    np.testing.assert_allclose(predictions["log_composition"].sum(1), 1)
    np.testing.assert_allclose(predictions["log_linear"].sum(1), 1)
    assert all(item["success"] for item in metadata["convergence"])
    assert len(metadata["counts"]) == 16
