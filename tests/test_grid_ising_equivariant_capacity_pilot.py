from __future__ import annotations

from math import comb
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.encoding import (
    GRID_ISING_EQUIVARIANT_ENCODING,
    grid_ising_equivariant_shell_amplitudes,
)
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
)
from iqp_cond.grid_ising_equivariant_capacity_pilot import (
    condition_angles,
    identities,
    load_manifest,
    prepare_data_archive,
)
from iqp_cond.simulator import estimate_pauli_z

MANIFEST = Path("configs/grid_ising_equivariant_capacity_pilot_v1.json")


def test_manifest_has_four_unique_runs_and_reuses_frozen_data(tmp_path: Path) -> None:
    config = load_manifest(MANIFEST)
    runs = identities(config)
    assert len(runs) == 4
    assert len({run.run_id for run in runs}) == 4
    path, _ = prepare_data_archive(config, tmp_path, Path.cwd())
    with np.load(path, allow_pickle=False) as stored:
        assert stored["train"].shape == (2, 1000, 8)
        assert stored["source_data_sha256"].item().decode() == config[
            "source_data"
        ]["sha256"]


def test_encoder_is_normalized_and_exactly_field_equivariant() -> None:
    config = load_manifest(MANIFEST)
    n_qubits = 8
    positive_angles = condition_angles(config, 0.44, 0.05, n_qubits)[0]
    negative_angles = condition_angles(config, 0.44, -0.05, n_qubits)[0]
    positive = np.asarray(
        grid_ising_equivariant_shell_amplitudes(
            jnp.asarray(positive_angles), n_qubits
        )
    )
    negative = np.asarray(
        grid_ising_equivariant_shell_amplitudes(
            jnp.asarray(negative_angles), n_qubits
        )
    )
    multiplicities = np.asarray([comb(n_qubits, k) for k in range(n_qubits + 1)])
    assert np.isclose(np.sum(multiplicities * np.abs(positive) ** 2), 1.0)
    np.testing.assert_allclose(
        negative,
        ((-1.0) ** np.arange(n_qubits + 1)) * positive,
        atol=1e-12,
    )
    ansatz = complete_ansatz(n_qubits)
    params = np.random.default_rng(4).normal(0.0, 0.2, ansatz.n_parameters)
    probabilities = exact_iqp_probabilities_many(
        params,
        ansatz.generators,
        np.stack([positive_angles, negative_angles]),
        GRID_ISING_EQUIVARIANT_ENCODING,
    )
    np.testing.assert_allclose(probabilities[0], probabilities[1][::-1], atol=1e-12)


def test_sampled_estimator_matches_exact_pauli_moments() -> None:
    config = load_manifest(MANIFEST)
    n_qubits = 5
    ansatz = complete_ansatz(n_qubits)
    params = np.random.default_rng(8).normal(0.0, 0.15, ansatz.n_parameters)
    angles = condition_angles(config, 0.44, 0.05, n_qubits)
    ops = enumerate_bitstrings(n_qubits)[1:]
    probabilities = exact_iqp_probabilities_many(
        params,
        ansatz.generators,
        angles,
        GRID_ISING_EQUIVARIANT_ENCODING,
    )[0]
    bits = enumerate_bitstrings(n_qubits)
    exact = np.asarray(
        [np.sum(probabilities * (1 - 2 * ((bits @ op) % 2))) for op in ops]
    )
    estimate = estimate_pauli_z(
        jnp.asarray(params),
        jnp.asarray(ansatz.generators),
        jnp.asarray(ops),
        jnp.asarray(angles),
        jax.random.PRNGKey(11),
        32_768,
        GRID_ISING_EQUIVARIANT_ENCODING,
    )
    np.testing.assert_allclose(np.asarray(estimate.values[0]), exact, atol=0.035)
