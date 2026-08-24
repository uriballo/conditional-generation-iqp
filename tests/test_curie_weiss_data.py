from pathlib import Path

import numpy as np
import pytest

from iqp_cond.curie_weiss_data import (
    CurieWeissDatasetRole,
    derive_split_seed,
    exact_curie_weiss_target,
    load_dataset_for_run,
    prepare_dataset_archive,
)
from iqp_cond.curie_weiss_protocol import (
    CurieWeissCampaignManifest,
    CurieWeissFamily,
    plan_curie_weiss_runs,
)

ROOT = Path(__file__).parents[1]
MANIFEST = CurieWeissCampaignManifest.from_json(
    ROOT / "configs/curie_weiss_foundations_v1.json"
)


def test_exact_target_normalization_spin_flip_and_permutation_symmetry():
    target = exact_curie_weiss_target(8, 1.0)
    assert target.probabilities.sum() == pytest.approx(1.0)
    assert np.allclose(target.probabilities, target.probabilities[::-1])
    for weight in range(9):
        shell = target.probabilities[
            np.isclose(target.magnetization, 2 * weight / 8 - 1)
        ]
        assert np.allclose(shell, shell[0])


def test_role_seeds_are_deterministic_and_separated():
    kwargs = dict(n_qubits=8, optimizer_seed=0, beta_index=0)
    a = derive_split_seed(MANIFEST, role=CurieWeissDatasetRole.TRAINING, **kwargs)
    b = derive_split_seed(MANIFEST, role=CurieWeissDatasetRole.TRAINING, **kwargs)
    c = derive_split_seed(MANIFEST, role=CurieWeissDatasetRole.VALIDATION, **kwargs)
    assert a == b and a != c


def test_archive_has_no_held_out_role_and_no_condition_is_constant(tmp_path):
    path, _ = prepare_dataset_archive(MANIFEST, tmp_path, n_qubits=8, optimizer_seed=0)
    with np.load(path, allow_pickle=False) as archive:
        assert not any("held" in name for name in archive.files)
    run = next(
        r
        for r in plan_curie_weiss_runs(MANIFEST)
        if r.n_qubits == 8
        and r.optimizer_seed == 0
        and r.family is CurieWeissFamily.NO_CONDITION
    )
    batch = load_dataset_for_run(MANIFEST, tmp_path, run)
    assert np.allclose(batch.condition_angles, batch.condition_angles[0])
