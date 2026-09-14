"""Regression tests for the reusable sampled conditional-MMD helper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.bond_reference import BOND_ENCODING
from iqp_cond.conditional_training_loss import sampled_losses
from iqp_cond.encoding import ANGLE_ENCODING, TWO_BLOCK_BRANCH_ENCODING
from iqp_cond.exact_tvd_runner import _weighted_loss_functions

jax.config.update("jax_enable_x64", True)


def _targets(couplings: np.ndarray, n: int, count: int = 3) -> np.ndarray:
    rng = np.random.default_rng(12)
    return rng.integers(0, 2, size=(len(couplings), count, n), dtype=np.int8)


@pytest.mark.parametrize("encoding", [TWO_BLOCK_BRANCH_ENCODING, BOND_ENCODING])
def test_reusable_loss_matches_runner_value_and_gradient(encoding: str) -> None:
    n = 6
    conditions = np.array([[0.3] * 3 + [0.5] * 3, [0.5] * 3 + [0.3] * 3])
    generators = np.asarray(
        [
            [int(i in edge) for i in range(n)]
            for edge in [(0,), (1,), (2,), (0, 1), (3, 4)]
        ]
    )
    validation = _targets(conditions, n)
    params = jnp.asarray(np.linspace(-0.2, 0.25, len(generators)))
    old, old_validation, _ = _weighted_loss_functions(
        generators=generators,
        angles=conditions,
        validation_targets=validation,
        sigmas=(0.2,),
        n_ops=16,
        n_samples=16,
        seed=19,
        encoding=encoding,
    )
    new, new_validation, _ = sampled_losses(
        generators, conditions, validation, 19, encoding, 16, n_ops=16, sigmas=(0.2,)
    )
    key = jax.random.PRNGKey(7)
    assert np.allclose(old(params, validation, key), new(params, validation, key))
    assert np.allclose(old_validation(params), new_validation(params))
    assert np.allclose(
        jax.grad(lambda p: old(p, validation, key))(params),
        jax.grad(lambda p: new(p, validation, key))(params),
    )


def test_reusable_loss_pads_visible_masks_for_ancillas() -> None:
    generators = np.asarray([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    angles = np.full((2, 6), 0.3)
    validation = _targets(angles, 4)
    loss, validation_loss, bank = sampled_losses(
        generators,
        angles,
        validation,
        4,
        ANGLE_ENCODING,
        8,
        n_ops=8,
        sigmas=(0.2,),
    )
    assert bank["masks"].shape[0] == 2
    assert bank["masks"].shape[2] == 4
    assert np.isfinite(validation_loss(jnp.zeros(2)))
    assert np.isfinite(loss(jnp.zeros(2), validation, jax.random.PRNGKey(2)))
