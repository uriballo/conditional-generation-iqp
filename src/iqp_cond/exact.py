"""Exact small-system and visible-marginal evaluation utilities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .encoding import (
    ANGLE_ENCODING,
    block_post_hadamard_amplitudes,
    is_block_encoding,
    is_radial_encoding,
    is_radial_shell_encoding,
    is_two_block_encoding,
    post_hadamard_amplitudes,
    radial_encoding_amplitudes,
    radial_shell_amplitudes,
    two_block_shell_amplitudes,
)


def enumerate_bitstrings(n_qubits: int) -> np.ndarray:
    """Enumerate bitstrings in little-endian integer order."""
    if n_qubits < 1 or n_qubits > 30:
        raise ValueError("n_qubits must be between 1 and 30")
    indices = np.arange(2**n_qubits, dtype=np.uint32)[:, None]
    shifts = np.arange(n_qubits, dtype=np.uint32)
    return ((indices >> shifts) & 1).astype(np.int8)


def fwht(values: np.ndarray) -> np.ndarray:
    """In-place-free Walsh-Hadamard transform in little-endian order."""
    return _fwht(np.asarray(values, dtype=np.complex128))


def _fwht(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.complex128).copy()
    size = transformed.shape[0]
    width = 1
    while width < size:
        blocks = transformed.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return transformed


def exact_iqp_probabilities(
    params: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    angles: np.ndarray | jnp.ndarray,
    encoding: str = ANGLE_ENCODING,
) -> np.ndarray:
    """Compute exact probabilities for ``RY -> H -> D_Z -> H``."""
    probabilities = exact_iqp_probabilities_many(
        params,
        generators,
        np.asarray(angles)[None, :],
        encoding,
    )
    return probabilities[0]


def exact_iqp_probabilities_many(
    params: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    angles: np.ndarray | jnp.ndarray,
    encoding: str = ANGLE_ENCODING,
) -> np.ndarray:
    """Compute exact probabilities for several input-angle vectors."""
    params = np.asarray(params, dtype=np.float64)
    generators = np.asarray(generators, dtype=np.int8)
    angles = np.asarray(angles, dtype=np.float64)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    n_qubits = generators.shape[1]
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
    if angles.ndim != 2 or angles.shape[1] != n_qubits:
        raise ValueError("angles must have shape (n_conditions, n_qubits)")

    phase = _phase_values(params, generators)

    phase_factors = np.exp(1j * phase)
    probabilities = [
        _probabilities_from_phase(phase_factors, row, n_qubits, encoding)
        for row in angles
    ]
    return np.stack(probabilities)


def _reference_state(
    condition_angles: np.ndarray,
    n_qubits: int,
    encoding: str,
) -> np.ndarray:
    n_pairs = n_qubits // 2
    if is_radial_shell_encoding(encoding):
        shell_amp = radial_shell_amplitudes(jnp.asarray(condition_angles), n_qubits)
        weights = enumerate_bitstrings(n_qubits).sum(axis=1)
        return np.asarray(shell_amp, dtype=np.complex128)[weights]
    if is_two_block_encoding(encoding):
        m_a = n_qubits // 2
        m_b = n_qubits - m_a
        shell_a, shell_b = two_block_shell_amplitudes(
            jnp.asarray(condition_angles), m_a, m_b
        )
        bits = enumerate_bitstrings(n_qubits)
        k_a = bits[:, :m_a].sum(axis=1)
        k_b = bits[:, m_a:].sum(axis=1)
        return (
            np.asarray(shell_a, dtype=np.complex128)[k_a]
            * np.asarray(shell_b, dtype=np.complex128)[k_b]
        )
    if is_radial_encoding(encoding):
        shell_amp = radial_encoding_amplitudes(
            jnp.asarray(condition_angles), n_qubits, encoding
        )
        weights = enumerate_bitstrings(n_qubits).sum(axis=1)
        return np.asarray(shell_amp, dtype=np.complex128)[weights]
    if is_block_encoding(encoding):
        block_amps, singleton = block_post_hadamard_amplitudes(
            jnp.asarray(condition_angles), encoding
        )
        block_amps = np.asarray(block_amps[0], dtype=np.complex128)
        if singleton is not None:
            s_zero, s_one = singleton
            reference = np.array([s_zero[0], s_one[0]], dtype=np.complex128)
            start_k = n_pairs - 1
        else:
            reference = block_amps[n_pairs - 1]
            start_k = n_pairs - 2
        for k in range(start_k, -1, -1):
            reference = np.outer(reference, block_amps[k]).ravel()
        return reference
    amp_zero, amp_one = post_hadamard_amplitudes(
        jnp.asarray(condition_angles), encoding
    )
    amp_zero_rev = amp_zero[::-1]
    amp_one_rev = amp_one[::-1]
    reference = np.array([amp_zero_rev[0], amp_one_rev[0]], dtype=np.complex128)
    for i in range(1, n_qubits):
        reference = np.outer(reference, [amp_zero_rev[i], amp_one_rev[i]]).ravel()
    return reference


def _probabilities_from_phase(
    phase_factors: np.ndarray,
    condition_angles: np.ndarray,
    n_qubits: int,
    encoding: str,
) -> np.ndarray:
    reference = _reference_state(condition_angles, n_qubits, encoding)
    final_state = _fwht(reference * phase_factors) / np.sqrt(2**n_qubits)
    probabilities = np.abs(final_state) ** 2
    return probabilities / probabilities.sum()


def _phase_values(params: np.ndarray, generators: np.ndarray) -> np.ndarray:
    """Return diagonal phases, using a mask-indexed FWHT path for 2-local IQPs."""
    n_qubits = generators.shape[1]
    size = 2**n_qubits
    supports = np.count_nonzero(generators, axis=1)
    if np.all(supports <= 2):
        coefficients = np.zeros(size, dtype=np.float64)
        masks = (
            generators.astype(np.uint32) << np.arange(n_qubits, dtype=np.uint32)
        ).sum(axis=1)
        np.add.at(coefficients, masks.astype(np.intp), params)
        return np.real(fwht(coefficients))

    indices = np.arange(size, dtype=np.uint32)
    phase = np.zeros(size, dtype=np.float64)
    for theta, generator in zip(params, generators, strict=True):
        mask = sum(1 << int(w) for w in np.flatnonzero(generator))
        phase += theta * (1.0 - 2.0 * (np.bitwise_count(indices & mask) & 1))
    return phase


def exact_iqp_visible_probabilities(
    params: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    angles: np.ndarray | jnp.ndarray,
    n_visible: int,
    n_ancillas: int = 0,
    encoding: str = ANGLE_ENCODING,
) -> np.ndarray:
    """Evaluate one condition at a time and return its visible marginal.

    Ancillas are assumed to be appended after the visible qubits.  Only one
    joint probability vector is live at a time, avoiding a stacked
    ``(n_conditions, 2**n_total)`` allocation.
    """
    params = np.asarray(params, dtype=np.float64)
    generators = np.asarray(generators)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    if n_visible < 1 or n_ancillas < 0:
        raise ValueError("n_visible must be positive and n_ancillas non-negative")
    if generators.shape[1] != n_visible + n_ancillas:
        raise ValueError("generator width must equal n_visible + n_ancillas")
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
    angles = np.asarray(angles)
    if angles.ndim == 1:
        angles = angles[None, :]
    if angles.ndim != 2 or angles.shape[1] != generators.shape[1]:
        raise ValueError("angles must have shape (n_conditions, n_qubits)")
    visible_size = 2**n_visible
    phase_factors = np.exp(1j * _phase_values(params, generators))
    outputs = []
    for condition_angles in angles:
        probabilities = _probabilities_from_phase(
            phase_factors,
            condition_angles,
            generators.shape[1],
            encoding,
        )
        outputs.append(
            probabilities
            if n_ancillas == 0
            else probabilities.reshape(2**n_ancillas, visible_size).sum(axis=0)
        )
    return np.stack(outputs)


def exact_iqp_visible_probabilities_product_amplitudes(
    params: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    input_amplitudes: np.ndarray | jnp.ndarray,
    n_visible: int,
    n_ancillas: int = 0,
) -> np.ndarray:
    """Evaluate visible marginals for explicit local product-state inputs.

    The final axis of ``input_amplitudes`` contains the computational-basis
    amplitudes before the first Hadamard. Conditions are evaluated one at a
    time and ancillas are marginalized immediately.
    """
    params = np.asarray(params, dtype=np.float64)
    generators = np.asarray(generators)
    amplitudes = np.asarray(input_amplitudes, dtype=np.complex128)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    if n_visible < 1 or n_ancillas < 0:
        raise ValueError("n_visible must be positive and n_ancillas non-negative")
    n_qubits = n_visible + n_ancillas
    if generators.shape[1] != n_qubits:
        raise ValueError("generator width must equal n_visible + n_ancillas")
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
    if amplitudes.ndim == 2:
        amplitudes = amplitudes[None, ...]
    if amplitudes.ndim != 3 or amplitudes.shape[1:] != (n_qubits, 2):
        raise ValueError(
            "input_amplitudes must have shape (conditions, qubits, 2)"
        )
    norms = np.sum(np.abs(amplitudes) ** 2, axis=-1)
    if not np.all(np.isfinite(norms)) or not np.allclose(
        norms, 1.0, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("each local input state must be finite and normalized")

    scale = np.sqrt(2.0)
    post_zero = (amplitudes[..., 0] + amplitudes[..., 1]) / scale
    post_one = (amplitudes[..., 0] - amplitudes[..., 1]) / scale
    phase_factors = np.exp(1j * _phase_values(params, generators))
    visible_size = 2**n_visible
    outputs = []
    for zero, one in zip(post_zero, post_one, strict=True):
        zero = zero[::-1]
        one = one[::-1]
        reference = np.asarray([zero[0], one[0]], dtype=np.complex128)
        for index in range(1, n_qubits):
            reference = np.outer(reference, [zero[index], one[index]]).ravel()
        final_state = _fwht(reference * phase_factors) / np.sqrt(2**n_qubits)
        probabilities = np.abs(final_state) ** 2
        probabilities /= probabilities.sum()
        outputs.append(
            probabilities
            if n_ancillas == 0
            else probabilities.reshape(2**n_ancillas, visible_size).sum(axis=0)
        )
    return np.stack(outputs)


def total_variation_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """Return half the L1 distance between probability vectors."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("probability vectors must be one-dimensional and aligned")
    return float(0.5 * np.abs(reference - candidate).sum())


def shell_probabilities(
    probabilities: np.ndarray,
    bitstrings: np.ndarray,
) -> np.ndarray:
    """Aggregate bitstring probabilities by Hamming weight."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    bitstrings = np.asarray(bitstrings)
    if probabilities.shape != (bitstrings.shape[0],):
        raise ValueError("probabilities and bitstrings must align")
    return np.bincount(
        bitstrings.sum(axis=1),
        weights=probabilities,
        minlength=bitstrings.shape[1] + 1,
    )


def generator_characters(generators: np.ndarray) -> np.ndarray:
    """Return the ``(n_generators, 2**n)`` character matrix in numpy.

    Character ``(m, z)`` is ``(-1)**(g_m . z)`` in little-endian order.
    Callers that JIT must precompute this matrix once and pass it in as a
    closure constant rather than a traced argument.
    """
    generators = np.asarray(generators, dtype=np.int8)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    bitstrings = enumerate_bitstrings(generators.shape[1])
    parity = (bitstrings.astype(np.int8) @ generators.T) % 2
    return np.where(parity == 0, 1.0, -1.0)


def exact_iqp_probabilities_many_jax(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    angles: jnp.ndarray,
    encoding: str = ANGLE_ENCODING,
    characters: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """JAX-differentiable exact probabilities for several input-angle vectors."""
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.int8)
    angles = jnp.asarray(angles, dtype=jnp.float64)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    n_qubits = generators.shape[1]
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
    if angles.ndim != 2 or angles.shape[1] != n_qubits:
        raise ValueError("angles must have shape (n_conditions, n_qubits)")
    if characters is None:
        characters = jnp.asarray(generator_characters(generators))
    characters = jnp.asarray(characters, dtype=jnp.float64)
    phases = characters @ params
    phase_factors = jnp.exp(1j * phases)

    probabilities = []
    n_pairs = n_qubits // 2
    for condition_angles in angles:
        if is_radial_shell_encoding(encoding):
            shell_amp = radial_shell_amplitudes(condition_angles, n_qubits)
            weights = jnp.asarray(
                enumerate_bitstrings(n_qubits).sum(axis=1),
                dtype=jnp.int32,
            )
            reference = shell_amp[weights].astype(jnp.complex128)
        elif is_two_block_encoding(encoding):
            m_a = n_qubits // 2
            m_b = n_qubits - m_a
            shell_a, shell_b = two_block_shell_amplitudes(condition_angles, m_a, m_b)
            bits = enumerate_bitstrings(n_qubits)
            k_a = jnp.asarray(bits[:, :m_a].sum(axis=1), dtype=jnp.int32)
            k_b = jnp.asarray(bits[:, m_a:].sum(axis=1), dtype=jnp.int32)
            reference = shell_a[k_a].astype(jnp.complex128) * shell_b[k_b].astype(
                jnp.complex128
            )
        elif is_radial_encoding(encoding):
            shell_amp = radial_encoding_amplitudes(
                condition_angles,
                n_qubits,
                encoding,
            )
            weights = jnp.asarray(
                enumerate_bitstrings(n_qubits).sum(axis=1),
                dtype=jnp.int32,
            )
            reference = shell_amp[weights].astype(jnp.complex128)
        elif is_block_encoding(encoding):
            block_amps, singleton = block_post_hadamard_amplitudes(
                condition_angles,
                encoding,
            )
            block_amps = block_amps[0]  # (n_pairs, 4)
            if singleton is not None:
                s_zero, s_one = singleton
                reference = jnp.asarray([s_zero[0], s_one[0]], dtype=jnp.complex128)
                start_k = n_pairs - 1
            else:
                reference = block_amps[n_pairs - 1]
                start_k = n_pairs - 2
            for k in range(start_k, -1, -1):
                reference = jnp.outer(reference, block_amps[k]).ravel()
        else:
            amp_zero, amp_one = post_hadamard_amplitudes(condition_angles, encoding)
            amp_zero = jnp.asarray(amp_zero, dtype=jnp.complex128)
            amp_one = jnp.asarray(amp_one, dtype=jnp.complex128)
            amp_zero_rev = amp_zero[::-1]
            amp_one_rev = amp_one[::-1]
            reference = jnp.asarray(
                [amp_zero_rev[0], amp_one_rev[0]],
                dtype=jnp.complex128,
            )
            for i in range(1, n_qubits):
                reference = jnp.outer(
                    reference,
                    jnp.asarray(
                        [amp_zero_rev[i], amp_one_rev[i]],
                        dtype=jnp.complex128,
                    ),
                ).ravel()

        diagonal_state = reference * phase_factors
        final_state = _fwht_jax(diagonal_state) / jnp.sqrt(2**n_qubits)
        condition_probabilities = jnp.abs(final_state) ** 2
        probabilities.append(
            condition_probabilities / condition_probabilities.sum(),
        )
    return jnp.stack(probabilities)


def fwht_jax(values: jnp.ndarray) -> jnp.ndarray:
    """JAX-differentiable Walsh-Hadamard transform on the last axis."""
    values = jnp.asarray(values)
    leading = values.shape[:-1]
    size = values.shape[-1]
    width = 1
    while width < size:
        values = values.reshape((*leading, -1, 2 * width))
        left = values[..., :width]
        right = values[..., width:]
        values = jnp.concatenate([left + right, left - right], axis=-1)
        width *= 2
    return values.reshape((*leading, size))


def _fwht_jax(values: jnp.ndarray) -> jnp.ndarray:
    return fwht_jax(values)
