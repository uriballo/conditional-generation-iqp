import numpy as np
import pytest

from iqp_cond.initialization import initialize_parameters_from_data


def test_one_and_two_body_iqopt_values_and_default_scale():
    data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]], dtype=float)
    generators = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float)
    values = initialize_parameters_from_data(data, generators, scale=2.0)
    means = data.mean(axis=0)
    signed = 2.0 * data - 1.0
    covariance = np.cov(signed, rowvar=False, bias=True)
    expected_pair = covariance[0, 1] * 2.0 * np.pi / abs(covariance[0, 1])
    np.testing.assert_allclose(values[:2], np.arcsin(np.sqrt(means)))
    np.testing.assert_allclose(values[2], expected_pair)

    default = initialize_parameters_from_data(data, generators)
    off_diag = covariance.copy()
    np.fill_diagonal(off_diag, 0.0)
    max_cov = np.max(np.abs(off_diag))
    default_scale = np.sqrt(2 * data.shape[1] / np.sum(off_diag**2)) * max_cov / np.pi
    np.testing.assert_allclose(
        default[2], covariance[0, 1] * default_scale * np.pi / max_cov,
    )
    np.testing.assert_allclose(np.sum(default[2:] ** 2), data.shape[1])


def test_other_generator_weights_are_zero():
    data = np.asarray([[0, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=float)
    generators = np.asarray([[1, 1, 1], [0, 0, 0]], dtype=float)
    np.testing.assert_allclose(initialize_parameters_from_data(data, generators), 0.0)


@pytest.mark.parametrize(
    "data,generators",
    [
        ([[0, 2], [1, 0]], [[1, 0]]),
        ([[0, np.nan], [1, 0]], [[1, 0]]),
        ([[0, 1]], [[1, 0, 1]]),
        ([[0, 1]], [[1, 0.5]]),
    ],
)
def test_initializer_rejects_malformed_inputs(data, generators):
    with pytest.raises(ValueError):
        initialize_parameters_from_data(data, generators)


def test_initializer_rejects_degenerate_covariance_and_bad_scale():
    data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    with pytest.raises(ValueError, match="nonzero off-diagonal"):
        initialize_parameters_from_data(data, [[1, 1]])
    with pytest.raises(ValueError, match="nonnegative"):
        initialize_parameters_from_data(data, [[1, 0]], scale=-1)
    with pytest.raises(ValueError, match="finite"):
        initialize_parameters_from_data(data, [[1, 0]], scale=np.inf)
