"""Conditional IQP generation with classically tractable product-state training."""

import jax

jax.config.update("jax_enable_x64", True)

from .ansatz import (
    Ansatz,
    complete_ansatz,
    heavy_hex_ansatz,
    lattice_ansatz,
)
from .encoding import (
    ANGLE_ENCODING,
    ENCODINGS,
    SYMMETRIC_REUPLOAD_ENCODING,
    product_state_amplitudes,
)
from .exact import (
    exact_iqp_probabilities,
    exact_iqp_probabilities_many,
    total_variation_distance,
)
from .mmd import (
    ConditionalMMDResult,
    conditional_mmd,
    hamming_rbf,
    mmd_operator_probability,
    sample_mmd_operators,
)
from .simulator import (
    ExpectationEstimate,
    estimate_pauli_z,
    post_hadamard_amplitudes,
)
from .targets import GaussianHammingTarget
from .training import TrainingConfig, TrainingResult, train

__all__ = [
    "ANGLE_ENCODING",
    "ENCODINGS",
    "SYMMETRIC_REUPLOAD_ENCODING",
    "Ansatz",
    "ConditionalMMDResult",
    "ExpectationEstimate",
    "GaussianHammingTarget",
    "TrainingConfig",
    "TrainingResult",
    "complete_ansatz",
    "conditional_mmd",
    "estimate_pauli_z",
    "exact_iqp_probabilities",
    "exact_iqp_probabilities_many",
    "hamming_rbf",
    "heavy_hex_ansatz",
    "lattice_ansatz",
    "mmd_operator_probability",
    "post_hadamard_amplitudes",
    "product_state_amplitudes",
    "sample_mmd_operators",
    "total_variation_distance",
    "train",
]
