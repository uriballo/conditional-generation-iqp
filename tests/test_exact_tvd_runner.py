from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact_tvd_campaign import load_manifest
from iqp_cond.exact_tvd_runner import (
    _git_revision,
    _weighted_loss_functions,
    condition_angles,
    load_data_archive,
    prepare_data_archive,
)


def test_git_revision_is_optional_for_clean_source_bundle(tmp_path: Path):
    assert _git_revision(tmp_path) is None


def test_synthetic_archives_are_fixed_role_separated_and_disjoint(tmp_path: Path):
    config = load_manifest(Path("configs/exact_tvd_ancilla_v1.json"))
    path, first_digest = prepare_data_archive(
        config, tmp_path, dataset="gaussian", n_visible=8
    )
    _, second_digest = prepare_data_archive(
        config, tmp_path, dataset="gaussian", n_visible=8
    )
    assert path.exists() and first_digest == second_digest
    arrays = load_data_archive(config, tmp_path, "gaussian", 8)
    assert arrays["train"].shape == (13, 1000, 8)
    assert not np.array_equal(arrays["train"], arrays["validation"])
    assert not np.array_equal(arrays["validation"], arrays["test"])


def test_condition_maps_cover_all_total_qubits():
    assert condition_angles("gaussian", [0.25], 18).shape == (1, 18)
    curie = condition_angles("curie_weiss", [0.8, 1.0, 1.2], 18)
    assert curie.shape == (3, 18)
    assert np.allclose(curie[1], 0.0)
    assert condition_angles("mfc", [125], 13).shape == (1, 13)


def test_weighted_mmd_loss_and_gradient_are_finite():
    generators = np.asarray(complete_ansatz(3).generators)
    targets = np.asarray(
        [[[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]]],
        dtype=np.float64,
    )
    loss, validation, _ = _weighted_loss_functions(
        generators=generators,
        angles=np.zeros((1, 3)),
        validation_targets=targets,
        sigmas=(0.8, 1.5, 3.0),
        n_ops=32,
        n_samples=16,
        seed=1,
        encoding="angle",
    )
    params = jnp.zeros(generators.shape[0])
    value, gradient = jax.value_and_grad(loss)(
        params, jnp.asarray(targets), jax.random.PRNGKey(4)
    )
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.isfinite(float(validation(params)))
