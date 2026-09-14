import numpy as np

from iqp_cond.ashkin_teller_baselines import expand_orbits as old_expand
from iqp_cond.ashkin_teller_baselines import orbit_counts as old_counts
from iqp_cond.ashkin_teller_size_baselines import (
    composition_predict,
    expand_orbits,
    fit_baselines,
    loglinear_objective,
    orbit_counts,
)


def test_validation_size_does_not_change_fitting_or_its_relative_weights():
    conditions = np.asarray([[0.1, 0.1], [0.1, 0.7], [0.7, 0.1], [0.7, 0.7]])
    rng = np.random.default_rng(4720)
    training = rng.integers(0, 2, (4, 40, 4), dtype=np.int8)
    validation = rng.integers(0, 2, (4, 40, 4), dtype=np.int8)
    q1, m1 = fit_baselines(conditions, training, validation, conditions)
    q2, m2 = fit_baselines(
        conditions, training, np.tile(validation, (1, 3, 1)), conditions
    )
    assert m1["selected_hyperparams"] == m2["selected_hyperparams"]
    np.testing.assert_array_equal(m1["counts"], m2["counts"])
    for model in q1:
        np.testing.assert_allclose(q1[model], q2[model], atol=1e-12)
    for model in m1["all_validation_NLL"]:
        np.testing.assert_allclose(
            m1["all_validation_NLL"][model], m2["all_validation_NLL"][model], atol=1e-12
        )


def test_n8_matches_fixed_orbits_and_expansion():
    rng = np.random.default_rng(18)
    samples = rng.integers(0, 2, (71, 8), dtype=np.int8)
    np.testing.assert_array_equal(orbit_counts(samples), old_counts(samples))
    q = rng.random(64)
    q /= q.sum()
    np.testing.assert_allclose(
        expand_orbits(q),
        old_expand(q),
    )


def test_n12_orbits_normalize_and_respect_independent_flips():
    rng = np.random.default_rng(19)
    samples = rng.integers(0, 2, (500, 12), dtype=np.int8)
    counts = orbit_counts(samples)
    assert counts.shape == (1024,)
    assert counts.sum() == len(samples)
    q = np.arange(1, 1025, dtype=float)
    q /= q.sum()
    full = expand_orbits(q)
    assert full.shape == (4096,)
    assert np.isclose(full.sum(), 1)
    for x in (0, 1, 17, 2048):
        assert full[x] == full[x ^ 63] == full[x ^ (63 << 6)] == full[x ^ 4095]


def test_gradient_matches_finite_difference_for_general_k():
    rng = np.random.default_rng(20)
    features = rng.normal(size=(4, 3))
    counts = rng.integers(1, 10, size=(4, 8))
    z = rng.normal(size=21)
    _, grad = loglinear_objective(z, features, counts, 0.03)
    eps = 1e-6
    numeric = np.empty_like(z)
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        numeric[i] = (
            loglinear_objective(zp, features, counts, 0.03)[0]
            - loglinear_objective(zm, features, counts, 0.03)[0]
        ) / (2 * eps)
    np.testing.assert_allclose(grad, numeric, rtol=3e-5, atol=3e-6)


def test_composition_recovers_additive_logits():
    levels = (0.1, 0.22, 0.38, 0.54, 0.7)
    conditions = np.asarray(
        [(a, 0.1) for a in levels]
        + [(a, 0.7) for a in levels]
        + [(0.1, b) for b in levels[1:-1]]
        + [(0.7, b) for b in levels[1:-1]]
    )
    rng = np.random.default_rng(21)
    base, ax, by = rng.normal(size=(3, 1024))
    logits = base + conditions[:, :1] * ax + conditions[:, 1:] * by
    laws = np.exp(logits - logits.max(1, keepdims=True))
    laws /= laws.sum(1, keepdims=True)
    query = np.asarray([[0.22, 0.38], [0.54, 0.38]])
    expected_logits = base + query[:, :1] * ax + query[:, 1:] * by
    expected = np.exp(expected_logits - expected_logits.max(1, keepdims=True))
    expected /= expected.sum(1, keepdims=True)
    np.testing.assert_allclose(
        composition_predict(conditions, laws, query), expected, atol=1e-12
    )
