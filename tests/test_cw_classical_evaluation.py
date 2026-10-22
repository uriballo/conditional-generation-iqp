import jax
import jax.numpy as jnp
import numpy as np
import pytest

from paper_experiments import cw_classical_evaluation as evaluation
from paper_experiments import cw_classical_rbm as rbm
from paper_experiments import cw_classical_transformer as transformer
from paper_experiments import cw_classical_vae as vae

SETTINGS = {"initial_nodes": 32, "max_nodes": 512, "tvd_tolerance": 1e-5}
CASES = [("rbm", rbm), ("transformer", transformer), ("vae", vae)]


def states():
    return ((jnp.arange(16)[:, None] >> jnp.arange(4)) & 1).astype(jnp.int32)


@pytest.mark.parametrize("family,module", CASES)
@pytest.mark.parametrize("capacity", ["small", "large"])
@pytest.mark.parametrize("beta", [0.15, 1.3])
def test_optimized_law_matches_reference(family, module, capacity, beta):
    params = module.init(jax.random.key(100), n=4, capacity=capacity)
    q, metadata = evaluation.optimized_probabilities(family, params, beta, 4, SETTINGS)
    if family == "vae":
        ref, ref_metadata = module.checked_probabilities(
            params,
            beta,
            initial_nodes=SETTINGS["initial_nodes"],
            max_nodes=SETTINGS["max_nodes"],
            tolerance=SETTINGS["tvd_tolerance"],
        )
        assert metadata["nodes"] == ref_metadata["nodes"]
        assert metadata["errors"] == pytest.approx(ref_metadata["errors"], abs=1e-14)
        assert metadata["tolerance"] == ref_metadata["tolerance"]
    else:
        ref = jnp.exp(module.log_prob(params, states(), beta))
        assert metadata == {"kind": "exact_finite_state_law"}
    np.testing.assert_allclose(q, ref, rtol=2e-10, atol=2e-10)


def test_parameter_dependence_and_spin_flip():
    params = rbm.init(jax.random.key(11), n=4)
    q0, _ = evaluation.optimized_probabilities("rbm", params, 0.2, 4, SETTINGS)
    changed = dict(params, a0=params["a0"].at[0].add(0.7))
    q1, _ = evaluation.optimized_probabilities("rbm", changed, 0.2, 4, SETTINGS)
    assert not np.allclose(q0, q1)
    np.testing.assert_allclose(q1, q1[::-1], atol=1e-14)


def test_high_scale_vae_agrees_and_converges():
    params = vae.init(jax.random.key(12), n=4, capacity="large", init_scale=4.0)
    q, metadata = evaluation.optimized_probabilities("vae", params, 1.3, 4, SETTINGS)
    ref, _ = vae.checked_probabilities(
        params,
        1.3,
        initial_nodes=SETTINGS["initial_nodes"],
        max_nodes=SETTINGS["max_nodes"],
        tolerance=SETTINGS["tvd_tolerance"],
    )
    np.testing.assert_allclose(q, ref, rtol=2e-10, atol=2e-10)
    assert metadata["nodes"] <= 512
    assert np.isclose(np.asarray(q).sum(), 1.0)


def test_vae_nonconvergence_propagates():
    params = vae.init(jax.random.key(13), n=4, capacity="large", init_scale=3.0)
    settings = {"initial_nodes": 32, "max_nodes": 128, "tvd_tolerance": 1e-20}
    with pytest.raises(FloatingPointError, match="did not converge"):
        evaluation.optimized_probabilities("vae", params, 1.3, 4, settings)


def test_vae_nonfinite_propagates():
    params = vae.init(jax.random.key(14), n=4)
    bad_w = params["dec2"]["w"].at[0, 0].set(jnp.nan)
    bad = dict(params, dec2=dict(params["dec2"], w=bad_w))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        evaluation.optimized_probabilities("vae", bad, 0.8, 4, SETTINGS)
