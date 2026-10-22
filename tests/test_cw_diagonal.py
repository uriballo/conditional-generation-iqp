import json

import jax
import jax.numpy as jnp
import numpy as np
import optuna
import pytest

from paper_experiments import cw_diagonal_fit
from paper_experiments import cw_encoder_hpo as hpo
from paper_experiments import cw_followup_hpo as audited
from paper_experiments import diagonal_conditioning as dc
from paper_experiments.cw_followup_config import candidate as audited_candidate
from paper_experiments.cw_followup_config import structured_seed
from paper_experiments.cw_followup_fit import _exact_kernel, digest
from paper_experiments.cw_followup_fit import run_fit as audited_run_fit

PARITY = np.array([bin(i).count("1") % 2 for i in range(1 << 16)])


def test_one_term_reproduces_constant_theta_kernel():
    theta = np.random.default_rng(1).normal(0, 0.3, 136)
    reference = dc.ghz_encoder(1).post_hadamard_amplitudes()
    expected = np.asarray(_exact_kernel(16)(jnp.asarray(theta), reference))
    for beta in (0.0, 0.55, 2.0):
        q = dc.exact_probabilities([beta], theta, 1)
        np.testing.assert_allclose(q, expected, atol=1e-14, rtol=0)


def test_chebyshev_values_and_chain_rule():
    x = np.array([-1.0, 0.0, 1.0])
    values = np.asarray(dc.chebyshev(x + 1, 4))
    expected = np.stack([np.ones(3), x, 2 * x**2 - 1, 4 * x**3 - 3 * x], axis=-1)
    np.testing.assert_allclose(values, expected, atol=0)
    rng = np.random.default_rng(2)
    weights, c = rng.normal(size=4 * 136), rng.normal(size=136)
    beta = 1.37

    def f(w):
        return jnp.sum(c * jnp.sin(dc.theta_of(w, beta, 4)))

    theta = np.asarray(dc.theta_of(weights, beta, 4))
    t = np.polynomial.chebyshev.chebvander(beta - 1, 3)[0]
    hand = (t[:, None] * (c * np.cos(theta))[None, :]).reshape(-1)
    np.testing.assert_allclose(np.asarray(jax.grad(f)(weights)), hand, atol=1e-12)


def test_spin_flip_symmetry_for_random_weights():
    rng = np.random.default_rng(3)
    for terms in (2, 4):
        weights = rng.normal(0, 1, terms * 136)
        report = dc.evaluate((0.0, 0.4, 1.0, 1.7, 2.0), weights, terms)
        assert np.max(report["flip_defect"]) <= 1e-10


def test_ghz_reference_is_uniform_on_even_parity_strings():
    encoder = dc.ghz_encoder(4)
    magnitudes = np.abs(np.asarray(encoder.post_hadamard_amplitudes())) ** 2
    np.testing.assert_allclose(magnitudes[:, PARITY == 0], 2.0**-15, rtol=1e-12)
    assert np.all(magnitudes[:, PARITY == 1] == 0)
    samples = encoder.sample(jax.random.PRNGKey(0), 64)
    assert np.all(np.asarray(samples).sum(-1) % 2 == 0)
    masks = jnp.asarray(np.random.default_rng(4).integers(0, 2, (32, 16)))
    ratios = np.abs(np.asarray(encoder.amplitude_ratios(samples, masks)))
    odd = np.asarray(masks).sum(1) % 2 == 1
    np.testing.assert_allclose(ratios[..., ~odd], 1, atol=1e-12)
    assert np.all(ratios[..., odd] == 0)


def _smoke_job(family, config, identity):
    return dict(
        family=family,
        candidate=config,
        trial_id=0,
        stage="screen",
        repeat=0,
        beta=None,
        smoke=True,
        identity=identity,
    )


def test_smoke_fit_pairs_with_shared_parity(tmp_path):
    base = dict(
        learning_rate=0.01,
        init_scale=0.1,
        b1=0.9,
        b2=0.99,
        schedule="cosine",
        expected_weights=(2, 4, 6),
    )
    diagonal = cw_diagonal_fit.run_fit(
        _smoke_job("diagonal_ghz_k2", dict(base, chebyshev_terms=2), "stub"),
        tmp_path / "diagonal",
    )
    parity = audited_run_fit(
        _smoke_job("shared_parity", dict(base, beta_scale=0.75), "stub"),
        tmp_path / "parity",
    )
    assert diagonal["status"] == parity["status"] == "ok"
    assert [p["sha256"] for p in diagonal["target_pools"]] == [
        p["sha256"] for p in parity["target_pools"]
    ]
    assert diagonal["streams"] == parity["streams"]
    with (
        np.load(next((tmp_path / "diagonal").glob("fits/*/restart-0.npz"))) as d,
        np.load(next((tmp_path / "parity").glob("fits/*/restart-0.npz"))) as p,
    ):
        assert d["initial_params"].shape == (272,)
        np.testing.assert_array_equal(d["initial_params"][:136], p["initial_params"])
        assert np.all(d["initial_params"][136:] == 0)


# Digests of the 60 real random proposals and their screen, confirmation and
# final jobs (stub identity), recorded from the source that ran the campaigns.
GOLDEN = {
    "twisted": "5f47115d34ae7ccd18cfa72af8c73014a416444b9c41ef63625e2f4ad2ad47e9",
    "twisted_independent": (
        "d2ef4ecbcd7dae88f7a8c916b792fa0a4f2d3f457dfb8f613781e7711c8675b6"
    ),
}


def _proposal_digest(name):
    hpo.configure(name)
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(
            seed=structured_seed("sampler", hpo.FAMILY, "random")
        )
    )
    rows = []
    for _ in range(60):
        trial = study.ask()
        rows.append(
            dict(
                trial_id=trial.number,
                config=json.loads(json.dumps(hpo.candidate(trial))),
                optuna_params=trial.params,
            )
        )
    context = dict(family=hpo.FAMILY, smoke=False, identity={"stub": 1})
    jobs = [
        job
        for proposal in rows
        for stage in ("screen", "confirmation", "final")
        for job in audited.jobs(context, proposal, stage)
    ]
    return digest([digest(job) for job in jobs])


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_existing_campaign_jobs_are_unchanged(name):
    assert _proposal_digest(name) == GOLDEN[name]
    assert hpo.FITS.get(name, audited._fit_with_retry) is audited._fit_with_retry


@pytest.mark.parametrize("terms", [4])
def test_diagonal_search_space_has_no_encoder_hyperparameter(terms):
    hpo.configure(f"diagonal_k{terms}")
    assert hpo.FAMILY == f"diagonal_ghz_k{terms}"
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=5))
    for _ in range(10):
        trial = study.ask()
        config = hpo.candidate(trial)
        reference = audited_candidate(
            optuna.trial.FixedTrial(trial.params), "independent_unconditional"
        )
        assert config == dict(reference, chebyshev_terms=terms)


def test_sharded_smoke_campaign_runs_every_phase(tmp_path):
    root = tmp_path / "campaign"
    run = dict(smoke=True, campaign="diagonal_k4")
    for phase in ("random", "confirmation", "final"):
        if phase == "confirmation":
            hpo.run(root, "tpe", **run)
        hpo.run(root, phase, prepare=True, **run)
        for index in range(2):
            status = hpo.run(root, phase, shard=(index, 2), **run)
            assert status["status"] == "shard_complete"
        status = hpo.run(root, phase, **run)
    assert status["status"] == "final_complete"
    data = json.loads((root / "final" / "figure_data.json").read_text())
    assert data["kind"] == "cw_diagonal_k4_final"
    assert data["family"] == "diagonal_ghz_k4"
    assert data["config"]["chebyshev_terms"] == 4
    assert "beta_scale" not in data["config"]
    assert len(data["beta"]) == len(data["curves"]["tvd"][0])
    assert data["marker_beta"] == [0.4, 1.0, 2.0]
    fit = next((root / "final" / "fits").glob("*/metrics.json"))
    assert json.loads(fit.read_text())["model"]["n_parameters"] == 544
    phases = [t.user_attrs["phase"] for t in hpo._study(root, None).trials]
    assert phases == ["random"] * 3 + ["tpe"] * 3
