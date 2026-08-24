"""Condition-independent output protocols for symmetry fidelity experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNALIGNED = "unaligned"
PERMUTATION_TWIRL = "permutation_twirl"
XOR_PAIR_CONTROL = "xor_pair_control"
OUTPUT_PROTOCOLS = (UNALIGNED, PERMUTATION_TWIRL, XOR_PAIR_CONTROL)


@dataclass(frozen=True)
class OutputProtocolSpec:
    """Auditable capacity and randomization metadata for an output protocol."""

    name: str
    random_actions_per_sample: int

    def parameter_count(self, model_parameter_count: int) -> int:
        """Return the unchanged model capacity under post-measurement processing."""
        if model_parameter_count < 0:
            raise ValueError("model_parameter_count must be nonnegative")
        return model_parameter_count


def output_protocol_spec(protocol: str) -> OutputProtocolSpec:
    """Return the frozen capacity metadata for an output protocol."""
    _validate_protocol(protocol)
    return OutputProtocolSpec(
        name=protocol,
        random_actions_per_sample=0 if protocol == UNALIGNED else 1,
    )


def apply_output_protocol(
    probabilities: np.ndarray,
    n_qubits: int,
    protocol: str,
) -> np.ndarray:
    """Apply one frozen protocol to an exact little-endian distribution."""
    _validate_protocol(protocol)
    probabilities = _validate_distribution(probabilities, n_qubits)
    if protocol == UNALIGNED:
        return probabilities.copy()
    if protocol == XOR_PAIR_CONTROL:
        indices = np.arange(probabilities.size, dtype=np.uint32)
        mask = sum(1 << wire for wire in range(0, n_qubits, 2))
        return 0.5 * (probabilities + probabilities[indices ^ mask])

    indices = np.arange(probabilities.size, dtype=np.uint32)
    weights = np.bitwise_count(indices)
    shell_mass = np.bincount(
        weights,
        weights=probabilities,
        minlength=n_qubits + 1,
    )
    shell_size = np.bincount(weights, minlength=n_qubits + 1)
    return shell_mass[weights] / shell_size[weights]


def apply_output_protocol_to_samples(
    samples: np.ndarray,
    protocol: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one independently sampled output action to each bitstring."""
    _validate_protocol(protocol)
    samples = np.asarray(samples, dtype=np.int8)
    if samples.ndim != 2 or samples.shape[1] < 1:
        raise ValueError("samples must have shape (n_samples, n_qubits)")
    if np.any((samples != 0) & (samples != 1)):
        raise ValueError("samples must be binary")
    transformed = samples.copy()
    if protocol == UNALIGNED:
        return transformed
    if protocol == XOR_PAIR_CONTROL:
        apply_mask = rng.integers(0, 2, size=samples.shape[0], dtype=np.int8)
        xor_mask = np.zeros(samples.shape[1], dtype=np.int8)
        xor_mask[::2] = 1
        return transformed ^ (apply_mask[:, None] * xor_mask[None, :])

    permutations = np.argsort(
        rng.random(samples.shape),
        axis=1,
    )
    return np.take_along_axis(transformed, permutations, axis=1)


def _validate_protocol(protocol: str) -> None:
    if protocol not in OUTPUT_PROTOCOLS:
        raise ValueError(f"unknown output protocol: {protocol}")


def _validate_distribution(
    probabilities: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if n_qubits < 1 or probabilities.shape != (2**n_qubits,):
        raise ValueError("probabilities must have shape (2**n_qubits,)")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
        raise ValueError("probabilities must sum to one")
    return probabilities
