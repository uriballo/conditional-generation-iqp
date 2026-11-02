"""Checks for the two-coordinate diagonal-conditioning baseline."""

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from paper_experiments import gaussian2d_diagonal as d
from paper_experiments import gaussian2d_hpo as hpo
from paper_experiments import gaussian2d_lattice as lattice
from paper_experiments.ansatz import square_lattice_ansatz
from paper_experiments.training import make_trainer as reference_trainer

SETTINGS = dict(name="adam", b1=0.9, b2=0.999, eps=1e-8, final_learning_rate_ratio=0.1)
SEED_KEYS = ("batches", "bands", "conditions", "masks", "reference")


def complete_generators(n=16):
    """All Z_i, then Z_iZ_j for i < j in row-major order, built by hand."""
    rows = [np.eye(n, dtype=int)[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n, dtype=int)
            row[[i, j]] = 1
            rows.append(row)
    return np.asarray(rows)


def statevector_probabilities(theta, generators=None):
    """Direct 2^16 statevector: H on each axis, explicit Z-string phases."""
    n = d.N_QUBITS
    if generators is None:
        generators = np.asarray(square_lattice_ansatz(n).generators, dtype=int)
    index = np.arange(1 << n)
    phase = np.zeros(1 << n)
    for angle, row in zip(theta, generators, strict=True):
        parity = np.zeros(1 << n, dtype=int)
        for qubit in np.flatnonzero(row):
            parity ^= (index >> qubit) & 1
        phase += angle * (1 - 2 * parity)
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1
    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def layer(vector):
        # Little-endian: qubit q is tensor axis n-1-q; H acts on every axis.
        tensor = vector.reshape((2,) * n)
        for axis in range(n):
            tensor = np.moveaxis(np.tensordot(gate, tensor, axes=(1, axis)), 0, axis)
        return tensor.reshape(-1)

    return abs(layer(np.exp(1j * phase) * layer(state))) ** 2


def test_unconditioned_weights_reproduce_standard_iqp_on_zero_input():
    rng = np.random.default_rng(3)
    w0 = rng.normal(0, 0.7, 40)
    weights = np.concatenate([w0, np.zeros(80)])
    pairs = np.array([[0.0, 0.0], [0.37, -1.0], [-1.0, 0.31]])
    q = d.probabilities(weights, pairs)
    direct = statevector_probabilities(w0)
    np.testing.assert_allclose(q, np.broadcast_to(direct, q.shape), atol=1e-13)
    oracle = ciqp.ConditionalModel(square_lattice_ansatz(16), d.zero_encoder(1))
    np.testing.assert_allclose(q[0], oracle.probabilities(w0)[0], atol=1e-12)


def test_conditioned_probabilities_match_statevector_at_mixed_pair():
    weights = np.random.default_rng(4).normal(0, 0.5, 120)
    u, v = 0.37, -0.31
    theta = weights[:40] + u * weights[40:80] + v * weights[80:]
    np.testing.assert_allclose(
        d.probabilities(weights, [[u, v]])[0],
        statevector_probabilities(theta),
        atol=1e-13,
    )


def test_characters_match_parent_lattice_bank():
    np.testing.assert_array_equal(d.characters(), lattice.characters())


def test_theta_map_and_gradient_follow_chain_rule():
    rng = np.random.default_rng(5)
    weights = rng.normal(size=120)
    coefficients = rng.normal(size=40)
    u, v = -0.73, 0.69
    theta = np.asarray(d.theta_of(weights, u, v))
    w = weights.reshape(3, 40)
    np.testing.assert_allclose(theta, w[0] + u * w[1] + v * w[2], atol=1e-15)
    np.testing.assert_allclose(d.thetas(weights, [[u, v]])[0], theta, atol=1e-15)

    def objective(values):
        return jnp.sum(coefficients * jnp.sin(d.theta_of(values, u, v)))

    gradient = np.asarray(jax.grad(objective)(jnp.asarray(weights)))
    outer = coefficients * np.cos(theta)
    np.testing.assert_allclose(
        gradient, np.concatenate([outer, u * outer, v * outer]), atol=1e-13
    )


def test_quarter_turn_one_body_phase_is_uniform():
    initial = hpo._model("diagonal")[0]
    config = dict(init_scale=0.0, one_body_mean=np.pi / 4)
    weights = initial(config, np.random.default_rng(1))
    assert np.allclose(weights[:16], np.pi / 4) and not weights[16:].any()
    pairs = np.array([[0.0, 0.0], [1.0, -1.0], [-0.37, 0.31]])
    np.testing.assert_allclose(d.probabilities(weights, pairs), 2.0**-16, atol=1e-15)


def test_metrics_format_matches_parent_metrics():
    weights = np.random.default_rng(6).normal(0, 0.3, 120)
    pairs = hpo.conditions()["development"]
    ours = d.metrics(weights, pairs)
    assert set(lattice.metrics(np.zeros(40), pairs[:1])) <= set(ours)
    profiles = d.metrics(weights, pairs, profiles=True)["profiles"]
    np.testing.assert_allclose(np.sum(profiles, 1), 1, atol=1e-12)


def test_layouts_and_protocol_parameter_counts():
    assert d.LAYOUTS == {
        120: ("lattice", 3, 40),
        200: ("lattice", 5, 40),
        408: ("complete", 3, 136),
        680: ("complete", 5, 136),
    }
    for campaign in hpo.CAMPAIGNS:
        if campaign == "input":
            continue
        spec = hpo.protocol(campaign)
        expected = d.n_terms(spec["basis"]) * d.n_phases(spec["ansatz"])
        assert spec["model"]["n_parameters"] == expected
        initial = hpo._model(campaign)[0]
        weights = initial(
            dict(init_scale=0.1, one_body_mean=0.5), np.random.default_rng(2)
        )
        assert weights.shape == (expected,)
        assert not weights[d.n_phases(spec["ansatz"]) :].any()


def _tiny_config(steps=3):
    return ciqp.TrainingConfig(
        steps=steps,
        batch_size=4,
        n_ops=8,
        mc_samples=16,
        sample_chunk=8,
        gemm_dtype=jnp.float32,
        validation_n_ops=1,
        validation_mc_samples=2,
        sigmas=(1.0, 2.0),
        learning_rate=0.05,
        final_learning_rate=0.005,
        validation_interval=1,
        mask_refresh="step",
    )


@pytest.mark.parametrize("basis", ["affine", "quadratic"])
def test_trainer_matches_reference_trainer_when_conditions_are_central(basis):
    """At u = v = 0 only w_0 moves, so it must follow the reference trajectory."""
    ansatz = square_lattice_ansatz(16)
    targets = np.random.default_rng(8).integers(0, 2, (2, 16, 16)).astype(float)
    w0 = np.random.default_rng(9).normal(0, 0.3, 40)
    settings = dict(
        SETTINGS,
        learning_rate=0.05,
        rng_seeds={k: 100 + i for i, k in enumerate(SEED_KEYS)},
    )
    ours = d.make_trainer(settings)(
        ansatz,
        d.zero_encoder(2),
        np.zeros((2, 2)),
        targets,
        seed=0,
        config=_tiny_config(),
        initial_params=np.concatenate([w0, np.zeros(40 * (d.n_terms(basis) - 1))]),
        selection_metric=lambda w: float(np.sum(w**2)),
    )
    theirs = reference_trainer(dict(settings, compute_validation_mmd=False))(
        ciqp.ConditionalModel(ansatz, d.zero_encoder(2)),
        targets,
        targets,
        seed=0,
        config=_tiny_config(),
        initial_params=w0,
        selection_metric=lambda w: float(np.sum(w**2)),
    )
    np.testing.assert_allclose(ours.final_params[:40], theirs.final_params, atol=1e-12)
    assert not ours.final_params[40:].any()
    np.testing.assert_allclose(ours.train_losses, theirs.train_losses, atol=1e-12)


def test_quadratic_map_with_zero_curvature_reproduces_affine_exactly():
    rng = np.random.default_rng(11)
    affine = rng.normal(0, 0.5, 120)
    quadratic = np.concatenate([affine, np.zeros(80)])
    pairs = np.array([[0.37, -0.31], [-1.0, 1.0], [0.0, 0.69], [-0.73, 0.0]])
    for u, v in pairs:
        np.testing.assert_array_equal(
            d.theta_of(quadratic, u, v), d.theta_of(affine, u, v)
        )
    np.testing.assert_array_equal(d.thetas(quadratic, pairs), d.thetas(affine, pairs))
    np.testing.assert_array_equal(
        d.probabilities(quadratic, pairs), d.probabilities(affine, pairs)
    )
    first, second = d.metrics(quadratic, pairs), d.metrics(affine, pairs)
    assert first == second


def test_quadratic_theta_map_and_gradient_follow_chain_rule():
    rng = np.random.default_rng(12)
    weights = rng.normal(size=200)
    coefficients = rng.normal(size=40)
    u, v = 0.37, -0.69
    w = weights.reshape(5, 40)
    theta = np.asarray(d.theta_of(weights, u, v))
    np.testing.assert_allclose(
        theta, w[0] + u * w[1] + v * w[2] + u**2 * w[3] + v**2 * w[4], atol=1e-14
    )
    np.testing.assert_allclose(d.thetas(weights, [[u, v]])[0], theta, atol=1e-14)

    def objective(values):
        return jnp.sum(coefficients * jnp.sin(d.theta_of(values, u, v)))

    gradient = np.asarray(jax.grad(objective)(jnp.asarray(weights)))
    outer = coefficients * np.cos(theta)
    np.testing.assert_allclose(
        gradient,
        np.concatenate([outer, u * outer, v * outer, u**2 * outer, v**2 * outer]),
        atol=1e-13,
    )
    with pytest.raises(ValueError):
        d.theta_of(np.zeros(160), u, v)


def test_complete_ansatz_matches_independent_statevector():
    rng = np.random.default_rng(21)
    w0 = rng.normal(0, 0.4, 136)
    np.testing.assert_array_equal(
        np.asarray(d.ansatz_of("complete").generators), complete_generators()
    )
    pairs = np.array([[0.0, 0.0], [0.37, -1.0], [-1.0, 0.31]])
    direct = statevector_probabilities(w0, complete_generators())
    for rows in (3, 5):
        weights = np.concatenate([w0, np.zeros((rows - 1) * 136)])
        q = d.probabilities(weights, pairs)
        np.testing.assert_allclose(q, np.broadcast_to(direct, q.shape), atol=1e-13)
    weights = rng.normal(0, 0.3, 680)
    u, v = -0.37, 0.69
    w = weights.reshape(5, 136)
    theta = w[0] + u * w[1] + v * w[2] + u**2 * w[3] + v**2 * w[4]
    np.testing.assert_allclose(
        d.probabilities(weights, [[u, v]])[0],
        statevector_probabilities(theta, complete_generators()),
        atol=1e-13,
    )
