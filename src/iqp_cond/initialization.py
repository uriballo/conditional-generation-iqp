"""Data-based parameter initializers compatible with IQPopt conventions."""

from __future__ import annotations

import numpy as np


def initialize_parameters_from_data(
    data: np.ndarray,
    generators: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Initialize one- and two-body IQP parameters from binary observations.

    One-body rows use ``arcsin(sqrt(mean))``.  For two-body rows, the
    covariance of the ``{-1,+1}`` representation is scaled by ``pi`` and by
    ``scale / max_abs_cov``.  With ``scale=None`` the IQPopt-main heuristic
    ``sqrt(2*n_features/sum(cov**2))*max_abs_cov/pi`` is used.  Rows of other
    weights are deterministically initialized to zero.
    """
    values = np.asarray(data)
    rows = np.asarray(generators)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a non-empty rank-two array")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("data must contain only finite binary values 0 and 1")
    if rows.ndim != 2 or rows.shape[1] != values.shape[1]:
        raise ValueError("generators must have shape (n_generators, n_features)")
    if not np.isfinite(rows).all() or not np.isin(rows, (0, 1)).all():
        raise ValueError("generators must contain only finite binary values 0 and 1")
    if scale is not None and (not np.isfinite(scale) or scale < 0):
        raise ValueError("scale must be finite and nonnegative")

    n_features = values.shape[1]
    signed = 2.0 * values - 1.0
    covariance = np.cov(signed, rowvar=False, bias=True)
    covariance = np.atleast_2d(covariance)
    off_diagonal = covariance.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    result = np.zeros(rows.shape[0], dtype=np.float64)
    means = np.mean(values, axis=0)
    has_two_body = bool(np.any(np.sum(rows, axis=1) == 2))
    if has_two_body:
        max_abs_cov = float(np.max(np.abs(off_diagonal)))
        if max_abs_cov == 0.0:
            raise ValueError(
                "covariance initialization requires nonzero off-diagonal covariance",
            )
        cov_sq_sum = float(np.sum(off_diagonal**2))
        if scale is None:
            scale = np.sqrt(2.0 * n_features / cov_sq_sum) * max_abs_cov / np.pi
    for index, row in enumerate(rows):
        weight = int(np.sum(row))
        if weight == 1:
            result[index] = np.arcsin(np.sqrt(means[np.flatnonzero(row)[0]]))
        elif weight == 2:
            left, right = np.flatnonzero(row)
            assert scale is not None
            result[index] = covariance[left, right] * scale * np.pi / max_abs_cov
    return result


# Descriptive alias for callers that use IQPopt's terminology.
initialize_from_data = initialize_parameters_from_data
