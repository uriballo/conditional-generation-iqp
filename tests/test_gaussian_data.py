from pathlib import Path

import numpy as np

from iqp_cond.gaussian_data import (
    GaussianDatasetRole,
    condition_angle,
    derive_split_seed,
    gaussian_target,
    grid_indices,
    load_dataset_for_run,
    master_grid,
    prepare_dataset_archive,
)
from iqp_cond.gaussian_protocol import (
    DataRegime,
    GaussianCampaignManifest,
    GaussianFamily,
    plan_gaussian_runs,
)

REPOSITORY = Path(__file__).parents[1]
MANIFEST = GaussianCampaignManifest.from_json(
    REPOSITORY / "configs" / "gaussian_foundations_v1.json"
)


def test_target_path_and_angle_map_are_exact():
    for n_qubits in MANIFEST.system_sizes:
        for condition in (0.0, 0.5, 1.0):
            target = gaussian_target(n_qubits, 0.5, condition)
            assert np.isclose(target.actual_mean, n_qubits * (0.15 + 0.70 * condition))
    assert np.isclose(condition_angle(0.0), 0.35 * np.pi)
    assert np.isclose(condition_angle(0.5), 0.0)
    assert np.isclose(condition_angle(1.0), -0.35 * np.pi)


def test_role_seeds_are_deterministic_and_disjoint():
    arguments = dict(n_qubits=8, width_factor=0.35, optimizer_seed=0, peak_index=0)
    training = derive_split_seed(
        MANIFEST, **arguments, role=GaussianDatasetRole.TRAINING
    )
    validation = derive_split_seed(
        MANIFEST, **arguments, role=GaussianDatasetRole.VALIDATION
    )
    assert training == derive_split_seed(
        MANIFEST, **arguments, role=GaussianDatasetRole.TRAINING
    )
    assert training != validation


def test_archive_is_deterministic_and_fixed_total_is_a_prefix(tmp_path):
    path, digest = prepare_dataset_archive(
        MANIFEST, tmp_path, n_qubits=8, width_factor=0.35, optimizer_seed=0
    )
    assert prepare_dataset_archive(
        MANIFEST, tmp_path, n_qubits=8, width_factor=0.35, optimizer_seed=0
    ) == (path, digest)
    runs = plan_gaussian_runs(MANIFEST)
    fixed_per = next(
        run
        for run in runs
        if run.n_qubits == 8
        and run.width_factor == 0.35
        and run.optimizer_seed == 0
        and run.family is GaussianFamily.CONDITIONED
    )
    batch = load_dataset_for_run(MANIFEST, tmp_path, fixed_per)
    assert batch.training.shape == (5, 10_000, 8)
    assert batch.validation.shape == (5, 2_000, 8)
    assert np.array_equal(grid_indices(5), (0, 2, 4, 6, 8))
    fixed_total = next(
        run
        for run in runs
        if run.n_qubits == 19
        and run.width_factor == 0.5
        and run.optimizer_seed == 0
        and run.peak_count == 5
        and run.data_regime is DataRegime.FIXED_TOTAL
        and run.family is GaussianFamily.CONDITIONED
    )
    fixed_per_peak = next(
        run
        for run in runs
        if run.n_qubits == 19
        and run.width_factor == 0.5
        and run.optimizer_seed == 0
        and run.peak_count == 5
        and run.data_regime is DataRegime.FIXED_PER_PEAK
        and run.family is GaussianFamily.CONDITIONED
    )
    prepare_dataset_archive(
        MANIFEST, tmp_path, n_qubits=19, width_factor=0.5, optimizer_seed=0
    )
    total_data = load_dataset_for_run(MANIFEST, tmp_path, fixed_total)
    per_peak_data = load_dataset_for_run(MANIFEST, tmp_path, fixed_per_peak)
    assert np.array_equal(total_data.training, per_peak_data.training[:, :3_600])
    assert np.array_equal(total_data.validation, per_peak_data.validation[:, :720])
    assert set(MANIFEST.held_out_grid).isdisjoint(master_grid())


def test_no_condition_and_single_peak_contracts(tmp_path):
    prepare_dataset_archive(
        MANIFEST, tmp_path, n_qubits=19, width_factor=0.5, optimizer_seed=0
    )
    runs = plan_gaussian_runs(MANIFEST)
    no_condition = next(
        run
        for run in runs
        if run.n_qubits == 19
        and run.width_factor == 0.5
        and run.optimizer_seed == 0
        and run.peak_count == 9
        and run.data_regime is DataRegime.FIXED_TOTAL
        and run.family is GaussianFamily.NO_CONDITION
    )
    no_condition_data = load_dataset_for_run(MANIFEST, tmp_path, no_condition)
    assert no_condition_data.training.shape == (9, 2_000, 19)
    assert np.array_equal(no_condition_data.condition_angles, np.zeros((9, 19)))
    bank = next(
        run
        for run in runs
        if run.n_qubits == 19
        and run.width_factor == 0.5
        and run.optimizer_seed == 0
        and run.family is GaussianFamily.SINGLE_PEAK
        and run.bank_peak_index == 3
    )
    bank_data = load_dataset_for_run(MANIFEST, tmp_path, bank)
    assert bank_data.training.shape == (1, 10_000, 19)
    assert bank_data.training_conditions.tolist() == [3 / 8]
