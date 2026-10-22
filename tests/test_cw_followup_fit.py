from types import SimpleNamespace

import iqp_cond as ciqp
import numpy as np
import pytest

from paper_experiments import cw_followup_fit as fit
from paper_experiments.cw_followup_config import C_STAR, protocol, selection_betas


@pytest.mark.parametrize(
    "family", ["shared_parity", "shared_product", "independent_unconditional"]
)
@pytest.mark.parametrize("scale", [0.25, 1.5])
def test_exact_probabilities_match_public_simulator(family, scale):
    ansatz = ciqp.Ansatz.complete(4)
    params = np.random.default_rng(4).normal(0, 0.2, ansatz.n_parameters)
    betas = (0.0, 0.2, 1.0, 2.0)
    expected = ciqp.ConditionalModel(
        ansatz, fit.encoder(family, betas, scale, 4)
    ).probabilities(params)
    actual = fit.exact_probabilities(family, betas, params, scale, 4)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)


def test_heat_uses_full_energy_moments():
    betas = (0.2, 1.0, 2.0)
    params = np.linspace(-0.2, 0.3, 10)
    report = fit.evaluate("shared_parity", betas, params, 1.2, 4)
    q = fit.exact_probabilities("shared_parity", betas, params, 1.2, 4)
    bits = (np.arange(16)[:, None] >> np.arange(4)) & 1
    m = (2 * bits.sum(1) - 4) / 4
    energy = -4 * m**2 / 2
    expected = np.array(betas) ** 2 / 4 * (q @ energy**2 - (q @ energy) ** 2)
    np.testing.assert_allclose(report["heat"], expected, atol=1e-12)
    np.testing.assert_allclose(report["flip_defect"], 0, atol=1e-12)


def test_global_checkpoint_pool_schedule_and_recovery(tmp_path, monkeypatch):
    original_limits = fit.followup_limits

    def budget(spec):
        value = original_limits(spec)
        value["starts"]["confirmation"] = 3
        return value

    monkeypatch.setattr(fit, "followup_limits", budget)
    calls, grids, settings_seen = [], [], []
    values = [(0.100, 0.8), (0.105, 0.3), (0.113, 0.01)]

    def evaluate(_family, betas, params, _scale):
        grids.append(tuple(betas))
        start = int(params[0])
        j, p = values[start]
        return dict(
            beta=np.array(betas),
            tvd=np.full(len(betas), j),
            magnetization_error=np.zeros(len(betas)),
            heat_error=np.full(len(betas), 2 * p * C_STAR),
        )

    monkeypatch.setattr(fit, "evaluate", evaluate)

    def make_trainer(settings):
        settings_seen.append(settings)

        def train(_model, _targets, _validation, **kwargs):
            start = len(calls)
            calls.append(start)
            theta = np.full(136, start, float)
            for step in (0, 1, 2):
                kwargs["selection_metric"](theta)
                kwargs["checkpoint_callback"](step, theta)
            return SimpleNamespace(
                final_params=theta,
                train_steps=np.array([1, 2]),
                train_losses=np.array([0.1, 0.09]),
            )

        return train

    monkeypatch.setattr(fit, "make_trainer", make_trainer)
    job = dict(
        family="shared_parity",
        stage="confirmation",
        repeat=0,
        beta=None,
        smoke=True,
        identity={"test": True},
        trial_id=7,
        candidate=dict(
            learning_rate=0.01,
            init_scale=0.1,
            b1=0.9,
            b2=0.99,
            expected_weights=[2, 6, 7.5],
            beta_scale=0.75,
            schedule="constant",
        ),
    )
    result = fit.run_fit(job, tmp_path)
    assert result["selected_start"] == 1
    assert result["selected_step"] == 0
    assert len(result["checkpoints"]) == 9
    assert all(s["final_learning_rate_ratio"] == 1 for s in settings_seen)
    assert all(s["compute_validation_mmd"] is False for s in settings_seen)
    assert all(g == selection_betas(protocol()) for g in grids[:-1])
    assert len(grids[-1]) == 37 and np.all(np.diff(grids[-1]) > 0)
    assert len(result["target_pools"]) == 4
    with np.load(result["directory"] + "/parameters.npz") as archive:
        assert archive["checkpoint_params"].shape == (9, 136)
        np.testing.assert_array_equal(archive["best_params"], np.ones(136))
    assert fit.run_fit(job, tmp_path) == result
    assert len(calls) == 3
    # A corrupt archive must never be silently reused.
    from pathlib import Path

    (Path(result["directory"]) / "parameters.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact"):
        fit.run_fit(job, tmp_path)
