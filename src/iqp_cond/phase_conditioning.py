"""Parameter-matched Condition-modulated IQP phase baselines.

The primary Ising comparison has exactly 136 trainable coefficients, matching
the visible-only Complete two-local ansatz.  A fixed, target-independent
feature assignment converts the normalized scalar Condition ``u in [0, 1]``
into one effective phase per generator.  The feature assignment is selected
before Held-out evaluation; it contains no learned or Target-derived values.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .exact import exact_iqp_probabilities
from .simulator import ExpectationEstimate, estimate_pauli_z

N_PHASE_COEFFICIENTS = 136

COMMON_AFFINE_MULTIPLIER = "common_affine_multiplier"
SIGNED_AFFINE_MULTIPLIERS = "signed_affine_multipliers"
BALANCED_CONSTANT_SLOPE_BANK = "balanced_constant_slope_bank"

# Concise aliases retained inside the numerical API.
COMMON_AFFINE = COMMON_AFFINE_MULTIPLIER
DETERMINISTIC_SIGNED_AFFINE = SIGNED_AFFINE_MULTIPLIERS
BALANCED_CONSTANT_SLOPE = BALANCED_CONSTANT_SLOPE_BANK
PHASE_FEATURE_MAPS = (
    COMMON_AFFINE_MULTIPLIER,
    SIGNED_AFFINE_MULTIPLIERS,
    BALANCED_CONSTANT_SLOPE_BANK,
)

FIXED_INPUT_FAMILY = "fixed_input"
PHASE_CONDITIONED_FAMILY = "condition_modulated_phase"
NO_CONDITION_FAMILY = "no_condition"
DEPLOYMENT_FAMILIES = (
    FIXED_INPUT_FAMILY,
    PHASE_CONDITIONED_FAMILY,
    NO_CONDITION_FAMILY,
)

PhaseFeatureMap = Literal[
    "common_affine_multiplier",
    "signed_affine_multipliers",
    "balanced_constant_slope_bank",
]
DeploymentFamily = Literal[
    "fixed_input",
    "condition_modulated_phase",
    "no_condition",
]


@dataclass(frozen=True, slots=True)
class ParameterSignature:
    """Stable identity and accounting for one parameter-matched model."""

    coefficient_count: int
    generator_count: int
    qubit_count: int
    coefficients_sha256: str
    generators_sha256: str


@dataclass(frozen=True, slots=True)
class DeploymentSignature:
    """Condition-wise circuit identity used to audit deployment semantics."""

    family: str
    n_conditions: int
    parameters: ParameterSignature
    generator_sha256_by_condition: tuple[str, ...]
    phase_sha256_by_condition: tuple[str, ...]
    input_sha256_by_condition: tuple[str, ...]
    generators_invariant: bool
    phases_invariant: bool
    inputs_invariant: bool


def normalize_conditions(
    conditions: np.ndarray | list[float] | tuple[float, ...],
    lower: float,
    upper: float,
) -> np.ndarray:
    """Map a frozen scalar Condition interval affinely to ``[0, 1]``."""
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("conditions must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("conditions must be finite")
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("upper must be finite and strictly greater than lower")
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(lower), abs(upper))
    if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
        raise ValueError("conditions must lie inside the frozen interval")
    normalized = (values - lower) / (upper - lower)
    return np.clip(normalized, 0.0, 1.0)


def _normalized_condition_array(
    normalized_conditions: np.ndarray | jnp.ndarray,
) -> np.ndarray:
    values = np.asarray(normalized_conditions, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError(
            "normalized_conditions must be a non-empty one-dimensional array",
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized_conditions must be finite")
    tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("normalized_conditions must lie in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def phase_feature_matrix(
    normalized_conditions: np.ndarray | jnp.ndarray,
    feature_map: PhaseFeatureMap,
) -> np.ndarray:
    """Return the frozen ``(Conditions, 136)`` feature assignment.

    ``common_affine_multiplier`` broadcasts ``0.5 + u``, retaining a nonzero
    intercept across the full interval.  The signed map alternates
    ``1 + (2u-1)`` and ``1 - (2u-1)`` in generator order.  The balanced bank
    alternates a constant feature and ``u``, yielding exactly 68 generators of
    each kind.
    """
    values = _normalized_condition_array(normalized_conditions)
    if feature_map not in PHASE_FEATURE_MAPS:
        raise ValueError(
            f"feature_map must be one of {PHASE_FEATURE_MAPS}; got {feature_map!r}",
        )
    u = values[:, None]
    generator_indices = np.arange(N_PHASE_COEFFICIENTS)
    if feature_map == COMMON_AFFINE:
        common = 0.5 + u
        return np.broadcast_to(
            common,
            (values.size, N_PHASE_COEFFICIENTS),
        ).copy()
    if feature_map == DETERMINISTIC_SIGNED_AFFINE:
        positive_slope = generator_indices % 2 == 0
        centered = 2.0 * u - 1.0
        return np.where(
            positive_slope[None, :],
            1.0 + centered,
            1.0 - centered,
        )
    constant_bank = generator_indices % 2 == 0
    return np.where(constant_bank[None, :], 1.0, u)


def _coefficient_array(coefficients: np.ndarray | jnp.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (N_PHASE_COEFFICIENTS,):
        raise ValueError(
            f"coefficients must have shape ({N_PHASE_COEFFICIENTS},)",
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be finite")
    return values


def exact_effective_phases(
    coefficients: np.ndarray | jnp.ndarray,
    normalized_conditions: np.ndarray | jnp.ndarray,
    feature_map: PhaseFeatureMap,
) -> np.ndarray:
    """Return NumPy effective phases for exact evaluation and artifacts."""
    coefficient_values = _coefficient_array(coefficients)
    return phase_feature_matrix(normalized_conditions, feature_map) * coefficient_values


def sampled_effective_phases(
    coefficients: np.ndarray | jnp.ndarray,
    normalized_conditions: np.ndarray | jnp.ndarray,
    feature_map: PhaseFeatureMap,
) -> jnp.ndarray:
    """Return differentiable JAX effective phases for sampled estimation."""
    features = phase_feature_matrix(normalized_conditions, feature_map)
    coefficient_values = jnp.asarray(coefficients, dtype=jnp.float64)
    if coefficient_values.shape != (N_PHASE_COEFFICIENTS,):
        raise ValueError(
            f"coefficients must have shape ({N_PHASE_COEFFICIENTS},)",
        )
    return jnp.asarray(features, dtype=jnp.float64) * coefficient_values[None, :]


def _array_digest(values: np.ndarray | jnp.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _generator_array(generators: np.ndarray | jnp.ndarray) -> np.ndarray:
    values = np.asarray(generators)
    if values.ndim != 2 or values.shape[0] != N_PHASE_COEFFICIENTS:
        raise ValueError(
            "generators must have shape (136, n_qubits) for the "
            "parameter-matched phase baseline",
        )
    if values.shape[1] < 1 or not np.all((values == 0) | (values == 1)):
        raise ValueError("generators must be a non-empty binary matrix")
    return values


def parameter_signature(
    coefficients: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
) -> ParameterSignature:
    """Return parameter accounting and content digests for one model."""
    coefficient_values = _coefficient_array(coefficients)
    generator_values = _generator_array(generators)
    return ParameterSignature(
        coefficient_count=coefficient_values.size,
        generator_count=generator_values.shape[0],
        qubit_count=generator_values.shape[1],
        coefficients_sha256=_array_digest(coefficient_values),
        generators_sha256=_array_digest(generator_values),
    )


def _condition_generator_batch(
    generators: np.ndarray | jnp.ndarray,
    n_conditions: int,
) -> np.ndarray:
    values = np.asarray(generators)
    if values.ndim == 2:
        _generator_array(values)
        return np.broadcast_to(values, (n_conditions, *values.shape))
    if (
        values.ndim != 3
        or values.shape[0] != n_conditions
        or values.shape[1] != N_PHASE_COEFFICIENTS
        or values.shape[2] < 1
    ):
        raise ValueError(
            "generators must have shape (136, n_qubits) or "
            "(n_conditions, 136, n_qubits)",
        )
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("generators must contain only zeros and ones")
    return values


def deployment_signature(
    family: DeploymentFamily,
    coefficients: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    effective_phases: np.ndarray | jnp.ndarray,
    input_angles: np.ndarray | jnp.ndarray,
) -> DeploymentSignature:
    """Return exact Condition-wise digests without applying a contract."""
    if family not in DEPLOYMENT_FAMILIES:
        raise ValueError(f"family must be one of {DEPLOYMENT_FAMILIES}")
    coefficient_values = _coefficient_array(coefficients)
    phase_values = np.asarray(effective_phases, dtype=np.float64)
    input_values = np.asarray(input_angles, dtype=np.float64)
    if phase_values.ndim != 2 or phase_values.shape[1] != N_PHASE_COEFFICIENTS:
        raise ValueError("effective_phases must have shape (n_conditions, 136)")
    if phase_values.shape[0] < 2 or not np.all(np.isfinite(phase_values)):
        raise ValueError("at least two finite effective-phase rows are required")
    if (
        input_values.ndim != 2
        or input_values.shape[0] != phase_values.shape[0]
        or input_values.shape[1] < 1
        or not np.all(np.isfinite(input_values))
    ):
        raise ValueError(
            "input_angles must have shape (n_conditions, n_qubits) and be finite",
        )
    generator_batch = _condition_generator_batch(
        generators,
        phase_values.shape[0],
    )
    if generator_batch.shape[2] != input_values.shape[1]:
        raise ValueError("generators and input_angles must use the same qubit count")

    generator_digests = tuple(_array_digest(row) for row in generator_batch)
    phase_digests = tuple(_array_digest(row) for row in phase_values)
    input_digests = tuple(_array_digest(row) for row in input_values)
    return DeploymentSignature(
        family=family,
        n_conditions=phase_values.shape[0],
        parameters=parameter_signature(coefficient_values, generator_batch[0]),
        generator_sha256_by_condition=generator_digests,
        phase_sha256_by_condition=phase_digests,
        input_sha256_by_condition=input_digests,
        generators_invariant=len(set(generator_digests)) == 1,
        phases_invariant=len(set(phase_digests)) == 1,
        inputs_invariant=len(set(input_digests)) == 1,
    )


def assert_deployment_contract(
    family: DeploymentFamily,
    coefficients: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    effective_phases: np.ndarray | jnp.ndarray,
    input_angles: np.ndarray | jnp.ndarray,
) -> DeploymentSignature:
    """Validate the frozen deployment distinction for a model family."""
    signature = deployment_signature(
        family,
        coefficients,
        generators,
        effective_phases,
        input_angles,
    )
    if not signature.generators_invariant:
        raise ValueError("compiled generators must be byte-identical across Conditions")

    phase_values = np.asarray(effective_phases, dtype=np.float64)
    coefficient_values = _coefficient_array(coefficients)
    if family in {FIXED_INPUT_FAMILY, NO_CONDITION_FAMILY}:
        expected = np.broadcast_to(coefficient_values, phase_values.shape)
        if not np.array_equal(phase_values, expected):
            raise ValueError(
                f"{family} must deploy the learned phase values unchanged",
            )

    if family == FIXED_INPUT_FAMILY:
        if not signature.phases_invariant or signature.inputs_invariant:
            raise ValueError(
                "fixed_input requires invariant phases and Condition-varying inputs",
            )
    elif family == PHASE_CONDITIONED_FAMILY:
        if signature.phases_invariant or not signature.inputs_invariant:
            raise ValueError(
                "condition_modulated_phase requires varying phases and a fixed input",
            )
    elif not signature.phases_invariant or not signature.inputs_invariant:
        raise ValueError("no_condition requires invariant phases and invariant inputs")
    return signature


def assert_fixed_input_contract(
    coefficients: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    input_angles: np.ndarray | jnp.ndarray,
) -> DeploymentSignature:
    """Validate fixed-circuit input conditioning."""
    inputs = np.asarray(input_angles, dtype=np.float64)
    phases = np.broadcast_to(_coefficient_array(coefficients), (inputs.shape[0], 136))
    return assert_deployment_contract(
        FIXED_INPUT_FAMILY,
        coefficients,
        generators,
        phases,
        inputs,
    )


def assert_no_condition_contract(
    coefficients: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    input_angles: np.ndarray | jnp.ndarray,
) -> DeploymentSignature:
    """Validate the No-condition family."""
    inputs = np.asarray(input_angles, dtype=np.float64)
    phases = np.broadcast_to(_coefficient_array(coefficients), (inputs.shape[0], 136))
    return assert_deployment_contract(
        NO_CONDITION_FAMILY,
        coefficients,
        generators,
        phases,
        inputs,
    )


def assert_phase_conditioned_contract(
    coefficients: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    normalized_conditions: np.ndarray | jnp.ndarray,
    fixed_input_angles: np.ndarray | jnp.ndarray,
    feature_map: PhaseFeatureMap,
) -> DeploymentSignature:
    """Validate a parameter-matched reprogrammable phase deployment."""
    phases = exact_effective_phases(
        coefficients,
        normalized_conditions,
        feature_map,
    )
    inputs = np.asarray(fixed_input_angles, dtype=np.float64)
    if inputs.ndim == 1:
        inputs = np.broadcast_to(inputs, (phases.shape[0], inputs.size))
    return assert_deployment_contract(
        PHASE_CONDITIONED_FAMILY,
        coefficients,
        generators,
        phases,
        inputs,
    )


def exact_phase_conditioned_probabilities(
    coefficients: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    normalized_conditions: np.ndarray | jnp.ndarray,
    fixed_input_angles: np.ndarray | jnp.ndarray,
    feature_map: PhaseFeatureMap,
) -> np.ndarray:
    """Evaluate exact output laws with a fixed input and reprogrammed phases."""
    generator_values = _generator_array(generators)
    phases = exact_effective_phases(
        coefficients,
        normalized_conditions,
        feature_map,
    )
    fixed_input = np.asarray(fixed_input_angles, dtype=np.float64)
    if fixed_input.shape != (generator_values.shape[1],):
        raise ValueError("fixed_input_angles must have shape (n_qubits,)")
    return np.stack(
        [
            exact_iqp_probabilities(phase, generator_values, fixed_input)
            for phase in phases
        ],
    )


def sampled_phase_conditioned_expectations(
    coefficients: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    normalized_conditions: np.ndarray | jnp.ndarray,
    fixed_input_angles: np.ndarray | jnp.ndarray,
    feature_map: PhaseFeatureMap,
    ops: np.ndarray | jnp.ndarray,
    key: jax.Array,
    n_samples: int,
) -> ExpectationEstimate:
    """Estimate observables for each Condition's effective phase vector."""
    generator_values = _generator_array(generators)
    phases = sampled_effective_phases(
        coefficients,
        normalized_conditions,
        feature_map,
    )
    fixed_input = jnp.asarray(fixed_input_angles, dtype=jnp.float64)
    if fixed_input.shape != (generator_values.shape[1],):
        raise ValueError("fixed_input_angles must have shape (n_qubits,)")
    condition_keys = jax.random.split(key, phases.shape[0])
    estimates = [
        estimate_pauli_z(
            phase,
            jnp.asarray(generator_values),
            ops,
            fixed_input,
            condition_key,
            n_samples,
        )
        for phase, condition_key in zip(phases, condition_keys, strict=True)
    ]
    return ExpectationEstimate(
        values=jnp.concatenate([estimate.values for estimate in estimates], axis=0),
        standard_errors=jnp.concatenate(
            [estimate.standard_errors for estimate in estimates],
            axis=0,
        ),
        sample_values=jnp.concatenate(
            [estimate.sample_values for estimate in estimates],
            axis=0,
        ),
    )
