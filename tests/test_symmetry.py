import numpy as np
import pytest

from iqp_cond.exact import (
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.symmetry import (
    PERMUTATION_TWIRL,
    UNALIGNED,
    XOR_PAIR_CONTROL,
    apply_output_protocol,
    apply_output_protocol_to_samples,
    output_protocol_spec,
)
from iqp_cond.symmetry_fidelity import (
    FROZEN_THRESHOLDS,
    distribution_metrics,
    save_study_record,
)
from iqp_cond.targets import GaussianHammingTarget


def test_permutation_twirl_uniformizes_each_shell_and_preserves_shell_mass():
    probabilities = np.asarray([0.04, 0.06, 0.12, 0.18, 0.09, 0.11, 0.17, 0.23])
    expected = np.asarray(
        [0.04, 0.09, 0.09, 0.15333333333333332,
         0.09, 0.15333333333333332, 0.15333333333333332, 0.23],
    )

    transformed = apply_output_protocol(
        probabilities,
        n_qubits=3,
        protocol=PERMUTATION_TWIRL,
    )

    np.testing.assert_allclose(transformed, expected)
    bitstrings = enumerate_bitstrings(3)
    np.testing.assert_allclose(
        shell_probabilities(transformed, bitstrings),
        shell_probabilities(probabilities, bitstrings),
    )


def test_permutation_twirl_makes_bitstring_tvd_equal_profile_tvd():
    bitstrings = enumerate_bitstrings(4)
    model = np.asarray(
        [
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
            0.09, 0.10, 0.11, 0.12, 0.04, 0.05, 0.06, 0.07,
        ],
    )
    model /= model.sum()
    target = GaussianHammingTarget.create(4, mean=1.5, width=1.0)
    target_probabilities = target.bitstring_probabilities(bitstrings)

    transformed = apply_output_protocol(model, 4, PERMUTATION_TWIRL)

    assert total_variation_distance(
        target_probabilities,
        transformed,
    ) == pytest.approx(
        total_variation_distance(
            target.shell_probabilities,
            shell_probabilities(model, bitstrings),
        ),
    )


def test_xor_pair_control_averages_with_the_fixed_mask_reindexing():
    probabilities = np.arange(1, 9, dtype=np.float64)
    probabilities /= probabilities.sum()

    transformed = apply_output_protocol(
        probabilities,
        n_qubits=3,
        protocol=XOR_PAIR_CONTROL,
    )

    mask = 0b101
    expected = 0.5 * (
        probabilities + probabilities[np.arange(probabilities.size) ^ mask]
    )
    np.testing.assert_allclose(transformed, expected)


def test_protocols_add_no_trainable_parameters_and_do_not_mutate_input():
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
    original = probabilities.copy()

    for protocol in (UNALIGNED, PERMUTATION_TWIRL, XOR_PAIR_CONTROL):
        spec = output_protocol_spec(protocol)
        assert spec.parameter_count(190) == 190
        transformed = apply_output_protocol(probabilities, 2, protocol)
        assert transformed.sum() == pytest.approx(1.0)
        assert np.all(transformed >= 0.0)

    np.testing.assert_array_equal(probabilities, original)


def test_sampled_permutation_twirl_converges_to_exact_distribution():
    rng = np.random.default_rng(481)
    probabilities = np.asarray([0.02, 0.03, 0.06, 0.09, 0.12, 0.18, 0.20, 0.30])
    bitstrings = enumerate_bitstrings(3)
    samples = bitstrings[rng.choice(8, size=80_000, p=probabilities)]

    transformed_samples = apply_output_protocol_to_samples(
        samples,
        PERMUTATION_TWIRL,
        rng,
    )

    indices = transformed_samples @ (1 << np.arange(3))
    empirical = np.bincount(indices, minlength=8) / indices.size
    exact = apply_output_protocol(probabilities, 3, PERMUTATION_TWIRL)
    np.testing.assert_allclose(empirical, exact, atol=0.006)


@pytest.mark.parametrize("protocol", [UNALIGNED, PERMUTATION_TWIRL, XOR_PAIR_CONTROL])
def test_output_protocol_rejects_wrong_distribution_size(protocol):
    with pytest.raises(ValueError, match=r"2\*\*n_qubits"):
        apply_output_protocol(np.ones(7) / 7, 3, protocol)


def test_distribution_metrics_expose_exact_guardrails():
    bitstrings = enumerate_bitstrings(3)
    target = GaussianHammingTarget.create(3, mean=1.25, width=0.8)
    target_probabilities = target.bitstring_probabilities(bitstrings)

    metrics = distribution_metrics(
        target,
        target_probabilities,
        bitstrings,
    )

    assert metrics["bitstring_tvd"] == pytest.approx(0.0)
    assert metrics["shell_tvd"] == pytest.approx(0.0)
    assert metrics["mean_absolute_error"] == pytest.approx(0.0)
    assert metrics["std_absolute_error"] == pytest.approx(0.0)


def test_study_record_is_independently_resumable(tmp_path):
    record = {
        "study": "endpoint",
        "seed": 3,
        "protocol": PERMUTATION_TWIRL,
        "thresholds": FROZEN_THRESHOLDS,
    }

    destination = save_study_record(record, tmp_path)

    assert destination.name == "endpoint_permutation_twirl_seed3.json"
    assert destination.exists()
    assert save_study_record(record, tmp_path, skip_existing=True) == destination
