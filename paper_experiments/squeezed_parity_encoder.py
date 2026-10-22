"""Squeezed parity encoder P(t) [S(r)|+>^(n-1) (x) |+>].

P(t) = exp[-i (t/2) sum_{i<n} Z_i Z_n] is the parity-encoder star layer and
S(r) = exp[i r (J_y J_z + J_z J_y) / (n-1)] is a two-axis-twist squeezing pulse
on the n-1 leaf qubits, with J the collective leaf spin. The angles follow the
fixed schedule sin t = tanh(a beta^2) and r = -kappa beta cos t, so the encoder
has no trainable parameters. With r = 0 and t = a beta the state is the
parity encoder at scale a. The squeeze makes the input probabilities
non-uniform, and the star layer acts after it, so the condition dependence is
not a diagonal phase on an IQP state.

Every factor commutes with X^n and with permutations of the leaves. Input and
reference amplitudes therefore depend only on the root bit and the leaf
Hamming weight, which gives exact sampling and amplitude ratios from a
2 x n table.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from .gaussian2d_lattice import _krawtchouk


def _leaf_twist(leaves):
    """(J_y J_z + J_z J_y)/(n-1) in the normalized Dicke basis of the leaves."""
    j = leaves / 2
    jz = np.diag(j - np.arange(leaves + 1)).astype(complex)
    m = j - np.arange(leaves)
    lower = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), -1).astype(complex)
    jy = (lower.T - lower) / 2j
    return (jy @ jz + jz @ jy) / leaves


def angles(beta, scale, kappa):
    """Star angle t and leaf squeeze r of one condition."""
    t = np.arcsin(np.tanh(scale * beta**2))
    return t, -kappa * beta * np.cos(t)


def _input_table(ts, rs, n):
    """Input amplitude of one bitstring with root bit r and leaf weight w."""
    leaves = n - 1
    w = np.arange(leaves + 1)
    norm = np.sqrt([comb(leaves, k) for k in w])
    twist = _leaf_twist(leaves)
    star = np.stack([leaves - 2 * w, -(leaves - 2 * w)])  # root Z = +1, -1
    rows = []
    for t, r in zip(ts, rs, strict=True):
        dicke = expm(1j * r * twist) @ (norm / 2 ** (leaves / 2))
        rows.append(np.exp(-1j * t / 2 * star) * (dicke / norm)[None] / np.sqrt(2))
    return np.asarray(rows)


def _shells(inputs, n):
    """Reference amplitudes after the Hadamard layer, by (root, leaf weight)."""
    leaves = inputs @ _krawtchouk(n - 1).T / 2 ** ((n - 1) / 2)
    shells = np.stack(
        [leaves[:, 0] + leaves[:, 1], leaves[:, 0] - leaves[:, 1]], axis=1
    ) / np.sqrt(2)
    counts = np.array([comb(n - 1, w) for w in range(n)], dtype=np.float64)
    if np.max(abs((abs(shells) ** 2 * counts).sum((1, 2)) - 1)) > 1e-10:
        raise FloatingPointError("reference amplitudes are not normalized")
    # X^n invariance gives exact zeros at odd total parity.
    parity = (np.arange(2)[:, None] + np.arange(n)[None]) % 2
    return np.where(parity[None] == 1, 0, shells), counts


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SqueezedParityEncoder:
    betas: jax.Array
    shells: jax.Array
    counts: jax.Array
    width: int
    scale: float
    kappa: float

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
        ts, rs = angles(values, scale, kappa)
        shells, counts = _shells(_input_table(ts, rs, n), n)
        return cls(
            jnp.asarray(values),
            jnp.asarray(shells),
            jnp.asarray(counts),
            n,
            scale,
            kappa,
        )

    def tree_flatten(self):
        children = (self.betas, self.shells, self.counts)
        return children, (self.width, self.scale, self.kappa)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children, *metadata)

    @property
    def n_conditions(self):
        return int(self.betas.shape[0])

    @property
    def n_qubits(self):
        return self.width

    def select(self, indices):
        idx = jnp.atleast_1d(indices)
        return type(self)(
            self.betas[idx],
            self.shells[idx],
            self.counts,
            self.width,
            self.scale,
            self.kappa,
        )

    def sample(self, key, n_samples):
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        key_cell, key_bits = jax.random.split(key)
        law = (abs(self.shells) ** 2 * self.counts[None, None]).reshape(
            self.n_conditions, -1
        )
        logits = jnp.where(law > 0, jnp.log(jnp.where(law > 0, law, 1.0)), -jnp.inf)
        cell = jax.random.categorical(
            key_cell, logits[:, None, :], shape=(self.n_conditions, n_samples)
        )
        root, weight = cell // self.width, cell % self.width
        scores = jax.random.uniform(
            key_bits, (self.n_conditions, n_samples, self.width - 1)
        )
        ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
        leaves = (ranks < weight[..., None]).astype(jnp.float64)
        return jnp.concatenate([leaves, root[..., None].astype(jnp.float64)], -1)

    def amplitude_ratios(self, samples, masks):
        z = jnp.asarray(samples)
        a = jnp.asarray(masks)
        root = z[..., -1].astype(jnp.int32)
        weight = z[..., :-1].sum(-1).astype(jnp.int32)
        flipped_weight = (
            weight[..., None]
            + a[:, :-1].sum(-1)[None, None]
            - 2 * jnp.einsum("csn,mn->csm", z[..., :-1], a[:, :-1])
        ).astype(jnp.int32)
        flipped_root = jnp.bitwise_xor(
            root[..., None], a[:, -1].astype(jnp.int32)[None, None]
        )
        rows = jnp.arange(self.n_conditions)
        denominator = self.shells[rows[:, None], root, weight]
        numerator = self.shells[rows[:, None, None], flipped_root, flipped_weight]
        safe = jnp.where(denominator != 0, denominator, 1)[..., None]
        return jnp.where(denominator[..., None] != 0, numerator / safe, 0)

    def post_hadamard_amplitudes(self):
        if self.width > 20:
            raise ValueError("exact amplitudes require at most 20 qubits")
        index = np.arange(1 << self.width)
        root = (index >> (self.width - 1)) & 1
        weight = np.array([int(x).bit_count() for x in index]) - root
        return self.shells[:, root, weight]

    def input_amplitudes(self):
        """Input amplitudes indexed by (condition, root bit, leaf weight)."""
        ts, rs = angles(np.asarray(self.betas), self.scale, self.kappa)
        return _input_table(ts, rs, self.width)

    def preparation(self, condition_index=0, wires=None):
        """Exact preparation of the input state for at most 20 qubits.

        The physical circuit is H on every wire, the collective leaf squeeze
        S(r), and IsingZZ(t) between each leaf and the root. The squeeze is
        not a native gate, so the dense state is loaded directly.
        """
        import pennylane as qml

        if not 0 <= condition_index < self.n_conditions:
            raise IndexError("condition outside encoder")
        if self.width > 20:
            raise ValueError("dense preparation requires at most 20 qubits")
        wires = list(range(self.width)) if wires is None else list(wires)
        index = np.arange(1 << self.width)
        # PennyLane order: wire 0 is the most significant bit, the root is last.
        bits = (index[:, None] >> (self.width - 1 - np.arange(self.width))) & 1
        table = np.asarray(self.input_amplitudes()[condition_index])
        state = table[bits[:, -1], bits[:, :-1].sum(1)]

        def queue():
            qml.StatePrep(state, wires=wires)

        return queue
