"""Twisted parity encoder P(t) [R_x(-pi/4)^(x)(n-1) T(mu)|+>^(n-1) (x) |+>].

P(t) = exp[-i (t/2) sum_{i<n} Z_i Z_n] is the parity-encoder star layer and
T(mu) = exp[-i (mu/(n-1)) sum_{i<j<n} Z_i Z_j] is a one-axis twist of the n-1
leaf qubits. The twist followed by the fixed collective turn R_x(-pi/4) on
every leaf is the standard one-axis-twisting spin-squeezing protocol
(Kitagawa--Ueda). The angles follow the fixed schedule sin t = tanh(a beta^2)
and mu = kappa beta cos t, so the encoder has no trainable parameters. At
mu = 0, R_x(-pi/4)|+> is |+> up to a global phase and the state is the parity
encoder with star angle t. As gates: H on every qubit, IsingZZ(2 mu/(n-1)) on
every leaf pair, RX(-pi/4) on every leaf and IsingZZ(t) from each leaf to the
root.

Every factor commutes with X^n and with permutations of the leaves, so the
amplitudes depend only on the root bit and the leaf Hamming weight. Sampling,
amplitude ratios and exact reference amplitudes reuse the 2 x n table of the
squeezed parity encoder.
"""

from __future__ import annotations

from math import comb

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from .squeezed_parity_encoder import SqueezedParityEncoder, _shells

TURN = -np.pi / 4


def angles(beta, scale, kappa):
    """Star angle t and leaf twist strength mu of one condition."""
    t = np.arcsin(np.tanh(scale * beta**2))
    return t, kappa * beta * np.cos(t)


def _leaf_operators(leaves):
    """sum_{i<j} Z_i Z_j and J_x in the normalized Dicke basis of the leaves."""
    j = leaves / 2
    jz = np.diag(j - np.arange(leaves + 1)).astype(complex)
    m = j - np.arange(leaves)
    lower = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), -1).astype(complex)
    jx = (lower + lower.T) / 2
    return 2 * jz @ jz - leaves / 2 * np.eye(leaves + 1), jx


def _input_table(ts, mus, n):
    """Input amplitude of one bitstring with root bit r and leaf weight w."""
    leaves = n - 1
    w = np.arange(leaves + 1)
    norm = np.sqrt([comb(leaves, k) for k in w])
    pairs, jx = _leaf_operators(leaves)
    turn = expm(-1j * TURN * jx)
    star = np.stack([leaves - 2 * w, -(leaves - 2 * w)])  # root Z = +1, -1
    rows = []
    for t, mu in zip(ts, mus, strict=True):
        twist = np.exp(-1j * mu / leaves * np.diag(pairs).real)
        dicke = turn @ (twist * norm / 2 ** (leaves / 2))
        rows.append(np.exp(-1j * t / 2 * star) * (dicke / norm)[None] / np.sqrt(2))
    return np.asarray(rows)


@jax.tree_util.register_pytree_node_class
class TwistedParityEncoder(SqueezedParityEncoder):
    """Same fields and table interface as SqueezedParityEncoder; only the leaf
    state differs. ``scale`` is the schedule constant a."""

    @classmethod
    def from_betas(cls, betas, n_qubits, scale=0.475, kappa=0.4):
        values = np.asarray(betas, dtype=np.float64).reshape(-1)
        if isinstance(n_qubits, bool) or int(n_qubits) != n_qubits or n_qubits < 2:
            raise ValueError("n_qubits must be an integer >= 2")
        if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("betas must be nonempty, finite, and nonnegative")
        scale, kappa = float(scale), float(kappa)
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(kappa):
            raise ValueError("scale must be positive and finite; kappa finite")
        n = int(n_qubits)
        ts, mus = angles(values, scale, kappa)
        shells, counts = _shells(_input_table(ts, mus, n), n)
        return cls(
            jnp.asarray(values),
            jnp.asarray(shells),
            jnp.asarray(counts),
            n,
            scale,
            kappa,
        )

    def input_amplitudes(self):
        """Input amplitudes indexed by (condition, root bit, leaf weight)."""
        ts, mus = angles(np.asarray(self.betas), self.scale, self.kappa)
        return _input_table(ts, mus, self.width)

    def preparation(self, condition_index=0, wires=None):
        """Gate-level input preparation; wire 0 most significant, root last."""
        import pennylane as qml

        if not 0 <= condition_index < self.n_conditions:
            raise IndexError("condition outside encoder")
        wires = list(range(self.width)) if wires is None else list(wires)
        ts, mus = angles(np.asarray(self.betas), self.scale, self.kappa)
        t, mu = float(ts[condition_index]), float(mus[condition_index])
        leaves, root = wires[:-1], wires[-1]

        def queue():
            for w in wires:
                qml.Hadamard(wires=w)
            for i, first in enumerate(leaves):
                for second in leaves[i + 1 :]:
                    qml.IsingZZ(2 * mu / len(leaves), wires=[first, second])
            for w in leaves:
                qml.RX(TURN, wires=w)
            for w in leaves:
                qml.IsingZZ(t, wires=[w, root])

        return queue
