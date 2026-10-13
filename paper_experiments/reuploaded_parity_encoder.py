"""Re-uploaded parity encoder P(beta) RX(a beta)^n P(beta)|+>^n.

P(beta) = exp[-i (lambda beta / 4) sum_{i<n} Z_i Z_n] is half the parity-encoder
phase layer, so turn = 0 recovers ParityEncoder at the same scale. Every layer
commutes with X^n and with permutations of the n-1 leaf qubits. Input and
reference amplitudes therefore depend only on the root bit and the leaf
Hamming weight, which gives exact sampling and amplitude ratios from a
2 x n table. The turn makes the input probabilities non-uniform, and the final
P layer prevents the gauge lemma from absorbing the turn into one-body phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from .gaussian2d_lattice import _krawtchouk


def _leaf_jx(leaves):
    """Collective Jx of the leaf register in the normalized Dicke basis."""
    j = leaves / 2
    m = j - np.arange(leaves + 1)
    coupling = np.sqrt(j * (j + 1) - m[:-1] * (m[:-1] - 1)) / 2
    return np.diag(coupling, -1) + np.diag(coupling, 1)


def _input_table(betas, n, scale, turn):
    """Input amplitude of one bitstring with root bit r and leaf weight w."""
    leaves = n - 1
    w = np.arange(leaves + 1)
    counts = np.array([comb(leaves, k) for k in w], dtype=np.float64)
    star = np.stack([leaves - 2 * w, -(leaves - 2 * w)])  # root Z = +1, -1
    values, vectors = np.linalg.eigh(_leaf_jx(leaves))
    rows = []
    for beta in betas:
        layer = np.exp(-1j * scale * beta / 4 * star)
        # Dicke-basis amplitudes of |+>^n, indexed by (root, leaf weight).
        state = np.sqrt(counts)[None] / 2 ** (n / 2) * layer
        angle = turn * beta
        leaf_rx = vectors @ np.diag(np.exp(-1j * angle * values)) @ vectors.T
        state = state @ leaf_rx.T
        root_rx = np.array(
            [
                [np.cos(angle / 2), -1j * np.sin(angle / 2)],
                [-1j * np.sin(angle / 2), np.cos(angle / 2)],
            ]
        )
        state = root_rx @ state
        state = layer * state
        rows.append(state / np.sqrt(counts)[None])
    return np.asarray(rows)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ReuploadedParityEncoder:
    betas: jax.Array
    shells: jax.Array
    counts: jax.Array
    width: int
    scale: float
    turn: float

    @classmethod
    def from_betas(cls, betas, n_qubits, scale=0.5, turn=0.0):
        values = np.asarray(betas, dtype=np.float64).reshape(-1)
        if isinstance(n_qubits, bool) or int(n_qubits) != n_qubits or n_qubits < 2:
            raise ValueError("n_qubits must be an integer >= 2")
        if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("betas must be nonempty, finite, and nonnegative")
        scale, turn = float(scale), float(turn)
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(turn):
            raise ValueError("scale must be positive and finite; turn finite")
        n = int(n_qubits)
        inputs = _input_table(values, n, scale, turn)
        # Hadamard on the leaves (Krawtchouk transform) and on the root.
        leaves = inputs @ _krawtchouk(n - 1).T / 2 ** ((n - 1) / 2)
        shells = np.stack(
            [leaves[:, 0] + leaves[:, 1], leaves[:, 0] - leaves[:, 1]], axis=1
        ) / np.sqrt(2)
        counts = np.array([comb(n - 1, w) for w in range(n)], dtype=np.float64)
        if np.max(abs((abs(shells) ** 2 * counts).sum((1, 2)) - 1)) > 1e-10:
            raise FloatingPointError("reference amplitudes are not normalized")
        # X^n invariance gives exact zeros at odd total parity.
        parity = (np.arange(2)[:, None] + np.arange(n)[None]) % 2
        shells = np.where(parity[None] == 1, 0, shells)
        return cls(
            jnp.asarray(values),
            jnp.asarray(shells),
            jnp.asarray(counts),
            n,
            scale,
            turn,
        )

    def tree_flatten(self):
        children = (self.betas, self.shells, self.counts)
        return children, (self.width, self.scale, self.turn)

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
            self.turn,
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
        return _input_table(np.asarray(self.betas), self.width, self.scale, self.turn)

    def preparation(self, condition_index=0, wires=None):
        import pennylane as qml

        if not 0 <= condition_index < self.n_conditions:
            raise IndexError("condition outside encoder")
        wires = list(range(self.width)) if wires is None else list(wires)
        beta = float(self.betas[condition_index])
        # exp(-i c Z_i Z_n) is IsingZZ(2c) with c = scale beta / 4.
        pair = self.scale * beta / 2

        def star_layer():
            for wire in wires[:-1]:
                qml.IsingZZ(pair, wires=[wire, wires[-1]])

        def queue():
            for wire in wires:
                qml.Hadamard(wires=wire)
            star_layer()
            for wire in wires:
                qml.RX(self.turn * beta, wires=wire)
            star_layer()

        return queue
