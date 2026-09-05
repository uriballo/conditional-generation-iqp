from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact_tvd_campaign import load_manifest, production_identities
from iqp_cond.exact_tvd_runner import (
    _git_revision,
    _weighted_loss_functions,
    condition_angles,
    evaluate_run,
    load_data_archive,
    prepare_data_archive,
    train_run,
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
    with pytest.raises(ValueError, match="execution is unavailable"):
        condition_angles("mfc", [125], 13)


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


def test_retired_mfc_preparation_fails_before_writing(tmp_path):
    config = load_manifest(Path("configs/exact_tvd_ancilla_v1.json"))
    with pytest.raises(ValueError, match="execution is unavailable"):
        prepare_data_archive(config, tmp_path, dataset="mfc", n_visible=3)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("execute", [train_run, evaluate_run])
def test_retired_mfc_execution_fails_before_accessing_artifacts(tmp_path, execute):
    config = load_manifest(Path("configs/exact_tvd_ancilla_v1.json"))
    run = next(run for run in production_identities(config) if run.dataset == "mfc")
    extra = {"selection": {}} if execute is train_run else {}
    with pytest.raises(ValueError, match="execution is unavailable"):
        execute(
            repository=tmp_path,
            config=config,
            run=run,
            data_directory=tmp_path,
            output_directory=tmp_path,
            **extra,
        )
    assert not list(tmp_path.iterdir())
