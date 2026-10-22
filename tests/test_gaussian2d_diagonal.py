"""Checks for the two-coordinate diagonal-conditioning baseline."""

import json
from pathlib import Path

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from paper_experiments import gaussian2d_diagonal as d
from paper_experiments import gaussian2d_diagonal_report as report
from paper_experiments import gaussian2d_diagonal_run as r
from paper_experiments import gaussian2d_lattice as lattice
from paper_experiments.ansatz import square_lattice_ansatz
from paper_experiments.training import make_trainer as reference_trainer

ARCHIVE = Path(__file__).resolve().parents[1] / (
    "results/gaussian-centered-lattice-20261010/axes"
)


@pytest.fixture
def quadratic(monkeypatch):
    """Point the runner at the quadratic protocol for one test only."""
    for name in ("PROTOCOL", "PROTOCOL_PATH", "PARENT_PATH"):
        monkeypatch.setattr(r, name, getattr(r, name))
    return r.use_protocol("quadratic")


@pytest.fixture(params=["affine", "quadratic"])
def complete(monkeypatch, request):
    """Point the runner at an all-to-all protocol for one test only."""
    for name in ("PROTOCOL", "PROTOCOL_PATH", "PARENT_PATH"):
        monkeypatch.setattr(r, name, getattr(r, name))
    return r.use_protocol(request.param, "complete")


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
    candidate = dict(r.PROTOCOL["candidates"]["mpi4_lr0.01"], init_scale=0.0)
    weights = d.initial_weights(1, candidate)
    assert np.allclose(weights[:16], np.pi / 4) and not weights[16:].any()
    pairs = np.array([[0.0, 0.0], [1.0, -1.0], [-0.37, 0.31]])
    np.testing.assert_allclose(d.probabilities(weights, pairs), 2.0**-16, atol=1e-15)
    origin = d.metrics(weights, [[0.0, 0.0]])
    assert origin["full_tvd"][0] < 1e-12


def test_metrics_format_matches_parent_metrics():
    weights = np.random.default_rng(6).normal(0, 0.3, 120)
    pairs = r.load_conditions()["development"]
    ours = d.metrics(weights, pairs)
    assert set(lattice.metrics(np.zeros(40), pairs[:1])) <= set(ours)
    profiles = d.metrics(weights, pairs, profiles=True)["profiles"]
    np.testing.assert_allclose(np.sum(profiles, 1), 1, atol=1e-12)
    assert ours["normalization_max_error"] < 1e-12


def test_protocol_inherits_parent_axes_settings():
    parent = r.check_parent()
    assert r.PROTOCOL["parent_protocol"]["sha256"] == r.sha(r.PARENT_PATH)
    ours, theirs = r.PROTOCOL["candidates"], parent["candidates"]
    assert sorted((c["init_scale"], c["learning_rate"]) for c in ours.values()) == (
        sorted((c["init_scale"], c["learning_rate"]) for c in theirs.values())
    )
    assert {c["local_mean"] for c in ours.values()} == {0, np.pi / 4}
    assert r.PROTOCOL["model"]["n_parameters"] == d.N_WEIGHTS == 120


@pytest.mark.parametrize("seed", [2026103901, 2026103911, 2026103921, 2026103925])
def test_pools_seeds_and_initialization_pair_with_published_axes_fits(seed):
    parent = json.loads(r.PARENT_PATH.read_text())
    fitting = np.asarray(parent["protocols"]["axes"]["fitting"])
    n_train = parent["budgets"]["n_train"]
    expected = lattice.sample_targets(
        fitting, n_train, np.random.default_rng(seed + 101)
    )
    pool = r.target_pool(seed, n_train)
    np.testing.assert_array_equal(pool, expected)
    assert r.rng_seeds(seed) == {
        k: seed + v for k, v in parent["rng_offsets"].items() if k != "pool"
    }
    from paper_experiments.gaussian2d_lattice_run import initial_parameters

    candidate = parent["candidates"]["m0_lr0.01"]
    np.testing.assert_array_equal(
        d.initial_weights(seed, candidate)[:40], initial_parameters(seed, candidate)
    )
    for path in ARCHIVE.glob(f"*/*/{seed}/fit.json"):
        record = json.loads(path.read_text())
        assert record["pool_sha256"] == r.hashlib.sha256(pool.tobytes()).hexdigest()


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
        r.PROTOCOL["optimizer"],
        learning_rate=0.05,
        rng_seeds={k: 100 + i for i, k in enumerate(r.rng_seeds(0))},
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


def test_campaign_freezes_winner_before_final(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "identity", lambda: {"fixed": "identity"})
    calls = []
    order = list(r.PROTOCOL["candidates"])

    def fake(output, stage, candidate, seed, **kwargs):
        if stage == "final":
            frozen = json.loads((tmp_path / "frozen_winners.json").read_text())
            assert frozen["winners"]["axes"] == candidate
            assert len([c for c in calls if c[0] != "final"]) == 8
        calls.append((stage, candidate, seed))
        return {
            "status": "complete",
            "checkpoint_sha256": "fake",
            "best_score": float(order.index(candidate)),
            "seconds": 1.0,
        }

    state = r.run_campaign(tmp_path, fit_runner=fake)
    assert state["status"] == "complete" and len(calls) == 13
    assert state["shortlist"] == order[:2] and state["winner"] == order[0]
    monkeypatch.setattr(r, "identity", lambda: {"changed": True})
    with pytest.raises(RuntimeError, match="changed"):
        r.run_campaign(tmp_path, fit_runner=fake)


def test_smoke_campaign_and_report_complete_end_to_end(tmp_path):
    _smoke(tmp_path, 120)


def test_quadratic_smoke_campaign_and_report_complete_end_to_end(tmp_path, quadratic):
    summary = _smoke(tmp_path, 200)
    assert summary["basis"] == "quadratic"
    receipt = json.loads((tmp_path / "campaign_receipt.json").read_text())
    assert receipt["protocol"] == "gaussian2d_diagonal_quadratic_protocol.json"
    fit = json.loads(
        (
            tmp_path / "axes/final" / receipt["winner"] / "2026103921/fit.json"
        ).read_text()
    )
    assert fit["request"]["model"] == "diagonal_quadratic_zero_input"


def test_complete_smoke_campaign_and_report_complete_end_to_end(tmp_path, complete):
    n_parameters = {"affine": 408, "quadratic": 680}[r.basis_name()]
    summary = _smoke(tmp_path, n_parameters)
    assert summary["ansatz"] == "complete"
    assert summary["diagonal"]["ansatz"] == "complete"
    receipt = json.loads((tmp_path / "campaign_receipt.json").read_text())
    assert receipt["ansatz"] == "complete"
    fit = json.loads(
        (
            tmp_path / "axes/final" / receipt["winner"] / "2026103921/fit.json"
        ).read_text()
    )
    assert fit["request"]["model"] == f"diagonal_complete_{r.basis_name()}_zero_input"
    data = np.load(tmp_path / "axes/final" / receipt["winner"] / "2026103921/fit.npz")
    np.testing.assert_array_equal(data["generators"], complete_generators())


def _smoke(tmp_path, n_parameters):
    r.freeze(tmp_path)
    state = r.run_campaign(tmp_path, smoke=True)
    assert state["status"] == "complete" and len(state["records"]) == 13
    summary = report.extract(
        tmp_path, tmp_path / "report", grid_points=3, axis_points=3
    )
    assert summary["smoke"] and len(summary["fits"]) == 5
    assert summary["diagonal"]["n_parameters"] == n_parameters
    assert set(report.AXIS_KEYS) <= set(summary["diagonal"])
    assert set(report.AXIS_KEYS) <= set(summary["ciqp"])
    arrays = np.load(tmp_path / "report/DIAGNOSTICS.npz")
    assert arrays["grid_tvd"].shape == (5, 3, 3)
    assert arrays["model_mean"].shape == (5, 3, 3)
    assert abs(summary["ciqp"]["mixed_full_tvd"]["mean"] - 0.0468) < 5e-5
    return summary


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
    candidate = r.PROTOCOL["candidates"]["mpi4_lr0.01"]
    np.testing.assert_array_equal(
        d.initial_weights(7, candidate, "quadratic"),
        np.concatenate([d.initial_weights(7, candidate), np.zeros(80)]),
    )


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


def test_quadratic_protocol_differs_from_affine_only_in_the_map(quadratic):
    parent = r.check_parent()
    affine = json.loads(
        (r.ROOT / "paper_experiments" / r.PROTOCOL_NAMES["affine"]).read_text()
    )
    assert quadratic["basis"] == "quadratic" and r.basis_name() == "quadratic"
    assert quadratic["model"]["n_parameters"] == 200
    assert quadratic["parent_protocol"]["sha256"] == r.sha(r.PARENT_PATH)
    assert quadratic["sibling_protocol"]["sha256"] == r.sha(
        r.ROOT / quadratic["sibling_protocol"]["path"]
    )
    for key in (
        "protocols",
        "candidates",
        "budgets",
        "seeds",
        "optimizer",
        "rng_offsets",
        "selection",
        "target_formula",
        "parent_protocol",
    ):
        assert quadratic[key] == affine[key], key
    assert quadratic["protocols"]["axes"] == parent["protocols"]["axes"]
    assert "u*v" in quadratic["model"]["rationale"]


def test_quadratic_campaign_stages_match_affine(tmp_path, monkeypatch, quadratic):
    monkeypatch.setattr(r, "identity", lambda: {"fixed": "identity"})
    calls = []

    def fake(output, stage, candidate, seed, **kwargs):
        calls.append((stage, candidate, seed))
        return {
            "status": "complete",
            "checkpoint_sha256": "fake",
            "best_score": 0.1,
            "seconds": 1.0,
        }

    state = r.run_campaign(tmp_path, fit_runner=fake)
    assert state["status"] == "complete" and len(calls) == 13
    assert state["basis"] == "quadratic"


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


def test_complete_initialization_and_lattice_defaults():
    candidate = r.PROTOCOL["candidates"]["mpi4_lr0.01"]
    lattice_w = d.initial_weights(5, candidate)
    complete_w = d.initial_weights(5, candidate, "quadratic", "complete")
    assert lattice_w.shape == (120,) and complete_w.shape == (680,)
    draws = np.random.default_rng(5).normal(0, 0.1, 136)
    np.testing.assert_array_equal(complete_w[16:136], draws[16:])
    np.testing.assert_array_equal(complete_w[:16], draws[:16] + np.pi / 4)
    assert not complete_w[136:].any()
    assert d.LAYOUTS == {
        120: ("lattice", 3, 40),
        200: ("lattice", 5, 40),
        408: ("complete", 3, 136),
        680: ("complete", 5, 136),
    }
    assert r.ansatz_name() == "lattice" and r.model_name() == (
        "diagonal_affine_zero_input"
    )


def _differences(first, second, prefix=""):
    keys = set(first) | set(second)
    out = set()
    for key in keys:
        a, b = first.get(key), second.get(key)
        if isinstance(a, dict) and isinstance(b, dict) and key == "model":
            out |= _differences(a, b, f"{key}.")
        elif a != b:
            out.add(prefix + key)
    return out


def test_complete_protocols_differ_from_lattice_parents_only_as_declared(complete):
    r.check_parent()
    parent_info = complete["diagonal_parent_protocol"]
    parent_path = r.ROOT / parent_info["path"]
    assert parent_info["sha256"] == r.sha(parent_path)
    parent = json.loads(parent_path.read_text())
    assert parent.get("basis", "affine") == r.basis_name()
    declared = {
        "name",
        "ansatz",
        "diagonal_parent_protocol",
        "model.ansatz",
        "model.weights_layout",
        "model.n_parameters",
        "model.initialization",
        "model.connectivity_rationale",
        "limitations",
    }
    assert _differences(complete, parent) == declared
    assert complete["model"]["n_parameters"] == 136 * d.n_terms(r.basis_name())
    assert "0.1" in complete["model"]["initialization"]
    assert "120 pairs" in complete["model"]["initialization"]
    assert (
        complete["candidates"]
        == json.loads(
            (r.ROOT / "paper_experiments" / r.PROTOCOL_NAMES["affine"]).read_text()
        )["candidates"]
    )
