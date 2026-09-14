"""Classically tractable product and block encodings for conditional IQP circuits."""

from __future__ import annotations

import jax.numpy as jnp

ANGLE_ENCODING = "angle"
SYMMETRIC_REUPLOAD_ENCODING = "symmetric_reupload"
CALIBRATED_REUPLOAD_ENCODING = "calibrated_reupload"
BLOCK_ZZ_PI8_ENCODING = "block_zz_pi8"
BLOCK_ZZ_PI4_ENCODING = "block_zz_pi4"
REUPLOAD_L2_ENCODING = "reupload_l2"
REUPLOAD_L3_ENCODING = "reupload_l3"
SPIN_FLIP_BRANCH_ENCODING = "spin_flip_branch"
FIELD_EQUIVARIANT_ENCODING = "field_equivariant_branch"
TWO_BLOCK_BRANCH_ENCODING = "two_block_branch"
RADIAL_SHELL_ENCODING = "radial_shell"
BETA_BINOMIAL_EQUIVARIANT_ENCODING = "beta_binomial_equivariant"
BETA_BINOMIAL_SHELL_PHASE_ENCODING = "beta_binomial_shell_phase"
GRID_ISING_EQUIVARIANT_ENCODING = "grid_ising_equivariant"

# Frozen quartic shell law for the radial-shell encoding (see
# radial_shell_amplitudes).  log q_c(k) is a centered quartic in the
# normalized weight x = (k - n/2)/n: location (lam1) and skew (lam3) are
# driven by the two condition coordinates; lam2, lam4 fix the double-well.
RADIAL_LAM1_SLOPE = 16.0
RADIAL_LAM2 = 72.0
RADIAL_LAM3_SLOPE = 120.0
RADIAL_LAM4 = 1296.0

BLOCK_ENCODINGS = (BLOCK_ZZ_PI8_ENCODING, BLOCK_ZZ_PI4_ENCODING)
RADIAL_ENCODINGS = (
    SPIN_FLIP_BRANCH_ENCODING,
    FIELD_EQUIVARIANT_ENCODING,
    BETA_BINOMIAL_EQUIVARIANT_ENCODING,
    BETA_BINOMIAL_SHELL_PHASE_ENCODING,
    GRID_ISING_EQUIVARIANT_ENCODING,
)
SINGLE_QUBIT_ENCODINGS = (
    ANGLE_ENCODING,
    SYMMETRIC_REUPLOAD_ENCODING,
    CALIBRATED_REUPLOAD_ENCODING,
)
ENCODINGS = SINGLE_QUBIT_ENCODINGS + BLOCK_ENCODINGS


def is_block_encoding(encoding: str) -> bool:
    """Return True if encoding uses 2-qubit entangled blocks."""
    return encoding in BLOCK_ENCODINGS


def is_radial_encoding(encoding: str) -> bool:
    """Return True if the encoding is a radial (Hamming-shell) reference state."""
    return encoding in RADIAL_ENCODINGS


def is_two_block_encoding(encoding: str) -> bool:
    """Return True if the encoding is a two-block spin-flip reference state."""
    return encoding == TWO_BLOCK_BRANCH_ENCODING


def is_radial_shell_encoding(encoding: str) -> bool:
    """Return True if the encoding is a general radial (S_n-symmetric) state."""
    return encoding == RADIAL_SHELL_ENCODING


def _ry(
    zero: jnp.ndarray,
    one: jnp.ndarray,
    angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    cosine = jnp.cos(angles / 2.0)
    sine = jnp.sin(angles / 2.0)
    return cosine * zero - sine * one, sine * zero + cosine * one


def product_state_amplitudes(
    angles: jnp.ndarray,
    encoding: str = ANGLE_ENCODING,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return local amplitudes immediately before the first Hadamards.

    ``symmetric_reupload`` uses
    ``RY(a/2) RZ(pi/2 sin(4a)) RY(a/2)|0>``.
    ``calibrated_reupload`` uses
    ``RY(a/2) RZ(pi/4 sin(2a)) RY(a/2)|0>``.
    """
    angles = jnp.asarray(angles, dtype=jnp.float64)
    zero = jnp.ones_like(angles, dtype=jnp.complex128)
    one = jnp.zeros_like(angles, dtype=jnp.complex128)
    if encoding == ANGLE_ENCODING:
        return _ry(zero, one, angles)
    if encoding == SYMMETRIC_REUPLOAD_ENCODING:
        zero, one = _ry(zero, one, angles / 2.0)
        phase = (jnp.pi / 2.0) * jnp.sin(4.0 * angles)
        zero = zero * jnp.exp(-0.5j * phase)
        one = one * jnp.exp(0.5j * phase)
        return _ry(zero, one, angles / 2.0)
    if encoding == CALIBRATED_REUPLOAD_ENCODING:
        zero, one = _ry(zero, one, angles / 2.0)
        phase = (jnp.pi / 4.0) * jnp.sin(2.0 * angles)
        zero = zero * jnp.exp(-0.5j * phase)
        one = one * jnp.exp(0.5j * phase)
        return _ry(zero, one, angles / 2.0)
    raise ValueError(
        f"unknown single-qubit encoding: {encoding!r}; "
        f"expected one of {SINGLE_QUBIT_ENCODINGS}"
    )


def _reupload_post_hadamard(
    angles: jnp.ndarray,
    layers: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Noncommuting L-layer data reupload: ``RY(a) [RZ(pi/2) RY(a)]^{L-1}`` then H.

    Same-axis repeats collapse to one rotation, so the fixed ``RZ(pi/2)`` gates
    between data rotations make the layers genuinely noncommuting.  The map is
    linear in the broadcast angle, matching the monotone-map guidance in
    ``docs/encoding-recommendations-by-dataset.md``.
    """
    a = jnp.asarray(angles)
    cosine = jnp.cos(a / 2.0)
    sine = jnp.sin(a / 2.0)
    zero = jnp.ones_like(a, dtype=jnp.complex128)
    one = jnp.zeros_like(a, dtype=jnp.complex128)
    rz_zero = jnp.exp(-0.25j * jnp.pi)
    rz_one = jnp.exp(0.25j * jnp.pi)
    zero, one = cosine * zero - sine * one, sine * zero + cosine * one
    for _ in range(layers - 1):
        zero, one = rz_zero * zero, rz_one * one
        zero, one = cosine * zero - sine * one, sine * zero + cosine * one
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return (zero + one) / scale, (zero - one) / scale


def two_branch_shell_amplitudes(
    angles: jnp.ndarray,
    n_qubits: int,
) -> jnp.ndarray:
    """Post-Hadamard Hamming-shell amplitudes of the spin-flip two-branch state.

    ``E|0^n> = |p>^n + X^n |p>^n`` with ``|p> = RY(a)|0>``.  After the first
    Hadamards the reference is radial with

        phi_k = A^{n-k} B^k (1 + (-1)^k),   A=(c+s)/sqrt2, B=(c-s)/sqrt2,

    so odd-weight amplitudes vanish and the state is spin-flip symmetric.  The
    broadcast angle is read from the first qubit (``angles[..., 0]``).
    """
    a = jnp.asarray(angles)
    angle = a if a.ndim == 0 else a[..., 0]
    c = jnp.cos(angle / 2.0)
    s = jnp.sin(angle / 2.0)
    big_a = (c + s) / jnp.sqrt(2.0)
    big_b = (c - s) / jnp.sqrt(2.0)
    k = jnp.arange(n_qubits + 1)
    amp = (big_a[..., None] ** (n_qubits - k)) * (big_b[..., None] ** k)
    parity = jnp.where(k % 2 == 0, 1.0, -1.0)
    return amp * (1.0 + parity)


def field_equivariant_shell_amplitudes(
    angles: jnp.ndarray,
    n_qubits: int,
) -> jnp.ndarray:
    """Post-Hadamard shell amplitudes of the signed-field encoder.

    The input is the normalized state

    ``|eta(h)> + X^n |eta(-h)>``,

    where ``|eta(h)> = RY(a(h))|0>^n`` and ``a(-h)=-a(h)``.  The two
    branches are orthogonal, so the normalization is ``sqrt(2)``.  After the
    first Hadamards, every computational-basis amplitude depends only on its
    Hamming weight and can therefore use the radial sampler and ratio query.
    """
    a = jnp.asarray(angles)
    angle = a if a.ndim == 0 else a[..., 0]
    c = jnp.cos(angle / 2.0)
    s = jnp.sin(angle / 2.0)
    big_a = (c + s) / jnp.sqrt(2.0)
    big_b = (c - s) / jnp.sqrt(2.0)
    k = jnp.arange(n_qubits + 1)
    forward = (big_a[..., None] ** (n_qubits - k)) * (big_b[..., None] ** k)
    reflected = (big_b[..., None] ** (n_qubits - k)) * (big_a[..., None] ** k)
    parity = jnp.where(k % 2 == 0, 1.0, -1.0)
    return (forward + parity * reflected) / jnp.sqrt(2.0)


def grid_ising_equivariant_shell_amplitudes(
    angles: jnp.ndarray, n_qubits: int
) -> jnp.ndarray:
    """Return a two-coordinate spin-flip-equivariant grid-Ising reference.

    The first coordinate controls the separation of two globally flipped
    product branches. The second swaps their weights under sign reversal.
    Neither coordinate specifies the lattice graph or a critical point.
    """
    a = jnp.asarray(angles, dtype=jnp.float64)
    if a.ndim == 0 or a.shape[-1] < 2:
        raise ValueError("grid-Ising encoding requires two condition coordinates")
    beta = a[..., 0]
    field = a[..., 1]
    angle = jnp.pi / 2 - (jnp.pi / 3) * (0.1 + 0.8 * beta)
    delta = jnp.pi * field / 6
    plus = jnp.cos(jnp.pi / 4 - delta)
    minus = jnp.sin(jnp.pi / 4 - delta)
    c, s = jnp.cos(angle / 2), jnp.sin(angle / 2)
    big_a = (c + s) / jnp.sqrt(2.0)
    big_b = (c - s) / jnp.sqrt(2.0)
    k = jnp.arange(n_qubits + 1)
    base = big_a[..., None] ** (n_qubits - k) * big_b[..., None] ** k
    norm = jnp.sqrt(
        1.0 + 2.0 * plus * minus * jnp.sin(angle) ** n_qubits
    )
    parity = jnp.where(k % 2 == 0, 1.0, -1.0)
    return base * (plus + parity * minus) / norm


def radial_encoding_amplitudes(
    angles: jnp.ndarray,
    n_qubits: int,
    encoding: str,
) -> jnp.ndarray:
    """Return shell amplitudes for a named radial encoder."""
    if encoding == SPIN_FLIP_BRANCH_ENCODING:
        return two_branch_shell_amplitudes(angles, n_qubits)
    if encoding == FIELD_EQUIVARIANT_ENCODING:
        return field_equivariant_shell_amplitudes(angles, n_qubits)
    if encoding == BETA_BINOMIAL_EQUIVARIANT_ENCODING:
        return beta_binomial_equivariant_shell_amplitudes(angles, n_qubits)
    if encoding == BETA_BINOMIAL_SHELL_PHASE_ENCODING:
        return beta_binomial_shell_phase_amplitudes(angles, n_qubits)
    if encoding == GRID_ISING_EQUIVARIANT_ENCODING:
        return grid_ising_equivariant_shell_amplitudes(angles, n_qubits)
    raise ValueError(f"unknown radial encoding: {encoding!r}")


def two_block_shell_amplitudes(
    angles: jnp.ndarray,
    m_a: int,
    m_b: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Post-Hadamard shell amplitudes of the two-block spin-flip state.

    ``|phi> = (|p_A>^m_a + X^m_a |p_A>^m_a) (x) (|p_B>^m_b + X^m_b |p_B>^m_b)``
    with ``|p_A> = RY(a_A)|0>`` and ``|p_B> = RY(a_B)|0>``.  The reference
    factorizes into per-block even-parity radial amplitudes; ``a_A`` is read
    from qubit 0 and ``a_B`` from qubit ``m_a``.  This is the ``Z2 x Z2``
    (independent block bitflip) generalization of the single two-branch state.
    """
    a = jnp.asarray(angles)
    shell_a = two_branch_shell_amplitudes(a[..., :m_a], m_a)
    shell_b = two_branch_shell_amplitudes(a[..., m_a:], m_b)
    return shell_a, shell_b


def radial_shell_amplitudes(
    angles: jnp.ndarray,
    n_qubits: int,
) -> jnp.ndarray:
    """General radial (S_n-symmetric) reference with a frozen quartic shell law.

    The two condition coordinates are read from ``angles[..., 0]`` (location)
    and ``angles[..., 1]`` (skewness).  The post-Hadamard amplitude is
    ``phi_k = sqrt(q_c(k) / C(n, k))`` with the centered quartic weight law

        log q_c(k) = lam1*x + lam2*x^2 + lam3*x^3 - lam4*x^4,
        x = (k - n/2) / n,

    where ``lam1`` and ``lam3`` are frozen affine functions of the two
    conditions.  This family is asymmetric in Hamming weight (no Z2 flip
    symmetry) and remains classically tractable via the shared radial
    sampling and ratio machinery.
    """
    from math import comb

    a = jnp.asarray(angles)
    c1 = a[..., 0]
    c2 = a[..., 1]
    lam1 = RADIAL_LAM1_SLOPE * (c1 - 0.5)
    lam3 = RADIAL_LAM3_SLOPE * (0.5 - c2)
    k = jnp.arange(n_qubits + 1, dtype=jnp.float64)
    x = (k - n_qubits / 2.0) / n_qubits
    logq = (
        lam1[..., None] * x
        + RADIAL_LAM2 * x**2
        + lam3[..., None] * x**3
        - RADIAL_LAM4 * x**4
    )
    binom = jnp.asarray(
        [comb(n_qubits, kk) for kk in range(n_qubits + 1)],
        dtype=jnp.float64,
    )
    return jnp.exp(logq / 2.0) / jnp.sqrt(binom)


def beta_binomial_equivariant_shell_amplitudes(
    angles: jnp.ndarray,
    n_qubits: int,
) -> jnp.ndarray:
    """Post-Hadamard shell amplitudes for the beta-binomial pilot encoder.

    ``angles[..., :2]`` contains normalized ``(u, v)`` coordinates.  The
    pre-Hadamard state is the normalized coherent sum of two broadcast RY
    product states, with branch angles ``pi/2 +/- b(v) + pi/6*(2*u-1)``.
    The exact overlap normalization makes the family satisfy
    ``X^n Phi(u,v) = Phi(1-u,v)`` up to floating-point roundoff.
    """
    a = jnp.asarray(angles, dtype=jnp.float64)
    u, v = a[..., 0], a[..., 1]
    b = jnp.pi / 4.0 + (jnp.pi / 24.0 - jnp.pi / 4.0) * v
    shift = (jnp.pi / 6.0) * (2.0 * u - 1.0)
    theta_minus = jnp.pi / 2.0 - b + shift
    theta_plus = jnp.pi / 2.0 + b + shift
    k = jnp.arange(n_qubits + 1)

    def branch(theta: jnp.ndarray) -> jnp.ndarray:
        c, s = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
        big_a = (c + s) / jnp.sqrt(2.0)
        big_b = (c - s) / jnp.sqrt(2.0)
        return big_a[..., None] ** (n_qubits - k) * big_b[..., None] ** k

    # The branch overlap is cos(theta_minus-theta_plus)/2 raised to n,
    # i.e. cos(b)^n.  This is real and positive for the prescribed range.
    normalization = jnp.sqrt(2.0 + 2.0 * jnp.cos(b) ** n_qubits)
    return (branch(theta_minus) + branch(theta_plus)) / normalization[..., None]


def beta_binomial_shell_phase_amplitudes(
    angles: jnp.ndarray, n_qubits: int
) -> jnp.ndarray:
    """Symmetric shell state with a fixed branch gap and phase modulation."""
    a = jnp.asarray(angles, dtype=jnp.float64)
    u, v = a[..., 0], a[..., 1]
    shift = (jnp.pi / 6.0) * (2.0 * u - 1.0)
    theta_minus = jnp.pi / 2.0 - jnp.pi / 6.0 + shift
    theta_plus = jnp.pi / 2.0 + jnp.pi / 6.0 + shift
    k = jnp.arange(n_qubits + 1)

    def branch(theta):
        c, s = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
        big_a, big_b = (c + s) / jnp.sqrt(2.0), (c - s) / jnp.sqrt(2.0)
        return big_a[..., None] ** (n_qubits - k) * big_b[..., None] ** k

    normalization = jnp.sqrt(2.0 + 2.0 * jnp.cos(jnp.pi / 6.0) ** n_qubits)
    base = (branch(theta_minus) + branch(theta_plus)) / normalization[..., None]
    gamma = jnp.pi / 12.0 + (jnp.pi / 4.0) * v
    q = ((n_qubits - 2.0 * k) / n_qubits) ** 2
    return base * jnp.exp(1j * gamma[..., None] * q)


def post_hadamard_amplitudes(
    angles: jnp.ndarray,
    encoding: str = ANGLE_ENCODING,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return local amplitudes after the first Hadamards."""
    if encoding == REUPLOAD_L2_ENCODING:
        return _reupload_post_hadamard(angles, 2)
    if encoding == REUPLOAD_L3_ENCODING:
        return _reupload_post_hadamard(angles, 3)
    zero, one = product_state_amplitudes(angles, encoding)
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return (zero + one) / scale, (zero - one) / scale


def block_post_hadamard_amplitudes(
    angles: jnp.ndarray,
    encoding: str = BLOCK_ZZ_PI8_ENCODING,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray] | None]:
    """Return 2-qubit block post-Hadamard amplitudes for {00, 01, 10, 11}.

    For an n-qubit system with angles shape (n_conditions, n_qubits):
    - Forms n_pairs = n_qubits // 2 blocks.
    - If n_qubits is odd, the last qubit is a singleton.
    Returns:
        block_amplitudes: shape (n_conditions, n_pairs, 4) in complex128,
            corresponding to basis states |00>, |10>, |01>, |11> (index = z0 + 2*z1).
        singleton: optional tuple (amp_zero, amp_one) of shape (n_conditions,) or None.
    """
    angles = jnp.asarray(angles, dtype=jnp.float64)
    if angles.ndim == 1:
        angles = angles[None, :]
    n_conditions, n_qubits = angles.shape
    n_pairs = n_qubits // 2

    if encoding == BLOCK_ZZ_PI8_ENCODING:
        kappa = jnp.pi / 8.0
    elif encoding == BLOCK_ZZ_PI4_ENCODING:
        kappa = jnp.pi / 4.0
    else:
        raise ValueError(
            f"unknown block encoding: {encoding!r}; expected one of {BLOCK_ENCODINGS}"
        )

    if n_pairs > 0:
        a0 = angles[:, 0 : 2 * n_pairs : 2]  # (n_conditions, n_pairs)
        a1 = angles[:, 1 : 2 * n_pairs : 2]  # (n_conditions, n_pairs)

        c0 = jnp.cos(a0 / 2.0)
        s0 = jnp.sin(a0 / 2.0)
        c1 = jnp.cos(a1 / 2.0)
        s1 = jnp.sin(a1 / 2.0)

        phase_plus = jnp.exp(-0.5j * kappa)
        phase_minus = jnp.exp(+0.5j * kappa)

        psi_00 = phase_plus * (c0 * c1)
        psi_10 = phase_minus * (s0 * c1)
        psi_01 = phase_minus * (c0 * s1)
        psi_11 = phase_plus * (s0 * s1)

        phi_00 = 0.5 * (psi_00 + psi_10 + psi_01 + psi_11)
        phi_10 = 0.5 * (psi_00 - psi_10 + psi_01 - psi_11)
        phi_01 = 0.5 * (psi_00 + psi_10 - psi_01 - psi_11)
        phi_11 = 0.5 * (psi_00 - psi_10 - psi_01 + psi_11)

        block_amplitudes = jnp.stack([phi_00, phi_10, phi_01, phi_11], axis=-1)
    else:
        block_amplitudes = jnp.zeros((n_conditions, 0, 4), dtype=jnp.complex128)

    singleton = None
    if n_qubits % 2 == 1:
        last_angles = angles[:, -1]
        singleton = post_hadamard_amplitudes(last_angles, ANGLE_ENCODING)

    return block_amplitudes, singleton
