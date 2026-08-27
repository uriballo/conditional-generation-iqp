"""Contracts for the Curie--Weiss confirmatory campaign."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from iqp_cond.confirmatory_curie import (
    ANCILLA_RUNGS,
    CRITICAL_BETAS,
    FIELD_HELD_OUT,
    FIELD_REFLECTED_TRAINING,
    FIELD_TRAINING,
    HELD_OUT_BETAS,
    SEEDS,
    SYSTEM_SIZES,
    TRAINING_BETAS,
    CurieConfirmatoryArm,
    CurieConfirmatoryManifest,
    curie_weiss_probabilities,
    disconnected_ansatz,
    equivariance_defect,
    plan_curie_confirmatory_runs,
    signed_field_probabilities,
    visible_matched_ansatz,
)

ROOT = Path(__file__).parents[1]
MANIFEST = CurieConfirmatoryManifest.from_json(
    ROOT / "configs" / "curie_weiss_confirmatory_v1.json"
)


def test_manifest_is_frozen_digestible_and_matches_confirmatory_coordinates() -> None:
    assert MANIFEST.system_sizes == SYSTEM_SIZES
    assert MANIFEST.seeds == SEEDS
    assert MANIFEST.ancilla_rungs == ANCILLA_RUNGS
    assert MANIFEST.training_betas == TRAINING_BETAS
    assert MANIFEST.held_out_betas == HELD_OUT_BETAS
    assert tuple(MANIFEST.raw["temperatures"]["critical"]) == CRITICAL_BETAS
    assert tuple(MANIFEST.raw["field_study"]["training_fields"]) == FIELD_TRAINING
    assert (
        tuple(MANIFEST.raw["field_study"]["reflected_training_fields"])
        == FIELD_REFLECTED_TRAINING
    )
    assert tuple(MANIFEST.raw["field_study"]["held_out_fields"]) == FIELD_HELD_OUT
    assert len(MANIFEST.digest) == 64
    with pytest.raises(TypeError):
        MANIFEST.raw["campaign_version"] = "changed"


def test_run_planner_has_390_unique_seed_matched_fits() -> None:
    runs = plan_curie_confirmatory_runs(MANIFEST)
    assert len(runs) == len({run.run_id for run in runs}) == 390
    assert {run.arm for run in runs} == set(CurieConfirmatoryArm)
    for n_qubits in SYSTEM_SIZES:
        for seed in SEEDS:
            cell = [
                run for run in runs if run.n_qubits == n_qubits and run.seed == seed
            ]
            assert len(cell) == len(CurieConfirmatoryArm) == 13
            assert {run.configuration_digest for run in cell} == {MANIFEST.digest}
            assert all(type(run).from_dict(run.to_dict()) == run for run in cell)

    with pytest.raises(ValueError, match="ancilla count"):
        replace(runs[0], ancillas=5)


@pytest.mark.parametrize("n_qubits", SYSTEM_SIZES)
@pytest.mark.parametrize("n_ancillas", ANCILLA_RUNGS)
def test_disconnected_bank_has_no_cross_register_generators(
    n_qubits: int,
    n_ancillas: int,
) -> None:
    ansatz = disconnected_ansatz(n_qubits, n_ancillas)
    generators = np.asarray(ansatz.generators)
    expected = (
        n_qubits
        + n_ancillas
        + n_qubits * (n_qubits - 1) // 2
        + n_ancillas * (n_ancillas - 1) // 2
    )
    assert ansatz.n_parameters == expected
    for row in generators:
        assert not (np.any(row[:n_qubits]) and np.any(row[n_qubits:]))


@pytest.mark.parametrize("n_qubits", SYSTEM_SIZES)
@pytest.mark.parametrize("n_ancillas", ANCILLA_RUNGS)
def test_visible_control_matches_full_joint_count_with_three_local_terms(
    n_qubits: int,
    n_ancillas: int,
) -> None:
    first = visible_matched_ansatz(n_qubits, n_ancillas)
    second = visible_matched_ansatz(n_qubits, n_ancillas)
    target_count = (n_qubits + n_ancillas) * (n_qubits + n_ancillas + 1) // 2
    assert first.n_parameters == target_count
    assert np.array_equal(first.generators, second.generators)
    localities = np.asarray(first.generators).sum(axis=1)
    assert set(localities) == {1.0, 2.0, 3.0}
    assert len({tuple(row) for row in np.asarray(first.generators)}) == target_count


@pytest.mark.parametrize("n_qubits", SYSTEM_SIZES)
def test_exact_targets_normalize_and_obey_signed_field_equivariance(
    n_qubits: int,
) -> None:
    for beta in (*TRAINING_BETAS, *HELD_OUT_BETAS):
        probabilities = curie_weiss_probabilities(n_qubits, beta)
        assert probabilities.shape == (2**n_qubits,)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-14)
        assert np.all(probabilities >= 0.0)
        assert np.array_equal(probabilities, probabilities[::-1])

    for field in (0.075, 0.15, 0.225, 0.30):
        positive = signed_field_probabilities(n_qubits, field)
        negative = signed_field_probabilities(n_qubits, -field)
        assert equivariance_defect(positive, negative) == pytest.approx(0.0, abs=1e-14)


def test_equivariance_defect_rejects_invalid_probability_vectors() -> None:
    with pytest.raises(ValueError, match="normalized"):
        equivariance_defect(np.asarray([0.2, 0.2]), np.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="power of two"):
        equivariance_defect(np.ones(3) / 3.0, np.ones(3) / 3.0)
