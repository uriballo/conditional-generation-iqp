"""Verify the experiment runner's memory-bounded loss matches the trainer."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.exact_tvd_campaign import expected_bodyness_sigmas
from iqp_cond.exact_tvd_runner import _weighted_loss_functions

spec = importlib.util.spec_from_file_location(
    "at_development",
    Path(__file__).resolve().parents[1] / "scripts/ashkin_teller_development.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "encoder", [runner.TWO_BLOCK_BRANCH_ENCODING, runner.BOND_ENCODING]
)
def test_padded_sequential_loss_matches_existing_estimator(encoder):
    g = np.eye(8, dtype=np.int8)
    angles = np.repeat(runner.CONDITIONS, 4, axis=1)
    if encoder == runner.TWO_BLOCK_BRANCH_ENCODING:
        angles = 0.225 * np.pi - 0.20 * np.pi * (angles - 0.1) / 0.6
    data = np.random.default_rng(1).integers(0, 2, (11, 20, 8))
    new, newval, _ = runner.sampled_losses(g, angles, data, 1701, encoder, 16)
    old, oldval, _ = _weighted_loss_functions(
        generators=g,
        angles=angles,
        validation_targets=data,
        sigmas=expected_bodyness_sigmas(8),
        n_ops=512,
        n_samples=16,
        seed=1701,
        encoding=encoder,
    )
    params = jnp.linspace(0.01, 0.2, 8)
    key = jax.random.PRNGKey(72)
    newvg = jax.jit(jax.value_and_grad(new))(params, jnp.asarray(data), key)
    oldvg = jax.value_and_grad(old)(params, jnp.asarray(data), key)
    for a, b in zip(newvg, oldvg, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(newval(params), oldval(params), atol=1e-12)


def test_frozen_data_are_paired_and_phase_dedup_is_correct(tmp_path):
    manifest = runner.bundle(tmp_path)
    assert runner.bundle(tmp_path) == manifest
    with np.load(tmp_path / "data.npz") as d:
        assert d["initial_params"].shape == (3, 30)
        assert d["samples"].shape == (3, 2, 11, 1000, 8)
        assert not np.array_equal(d["samples"][0, 0], d["samples"][0, 1])
        np.testing.assert_allclose(d["targets"].sum(1), 1, atol=1e-14)
        assert np.array_equal(d["conditions"], runner.CONDITIONS)
