"""Parity constrained RX--CNOT input encoder."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParityEncoder:
    betas: jax.Array
    width: int
    scale: float = 0.5

    @classmethod
    def from_betas(cls, betas, n_qubits, scale=0.5):
        values = np.asarray(betas, dtype=np.float64).reshape(-1)
        if isinstance(n_qubits, bool) or int(n_qubits) != n_qubits or n_qubits < 2:
            raise ValueError("n_qubits must be an integer >= 2")
        scale = float(scale)
        if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("betas must be nonempty, finite, and nonnegative")
        if not np.isfinite(scale) or scale <= 0 or np.any(scale * values > np.pi):
            raise ValueError("positive finite scale and angles <= pi required")
        return cls(jnp.asarray(values), int(n_qubits), scale)

    def tree_flatten(self):
        return (self.betas,), (self.width, self.scale)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        width, scale = metadata
        return cls(children[0], width, scale)

    @property
    def n_conditions(self):
        return int(self.betas.shape[0])

    @property
    def n_qubits(self):
        return self.width

    def select(self, indices):
        return type(self)(self.betas[jnp.atleast_1d(indices)], self.width, self.scale)

    def sample(self, key, n_samples):
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        t = self.scale * self.betas[:, None, None]
        leaves = jax.random.bernoulli(
            key, jnp.sin(t / 2) ** 2, (self.n_conditions, n_samples, self.width - 1)
        ).astype(jnp.float64)
        root = jnp.mod(jnp.sum(leaves, axis=-1), 2)[..., None]
        return jnp.concatenate([leaves, root], axis=-1)

    def amplitude_ratios(self, samples, masks):
        z = jnp.asarray(samples)
        a = jnp.asarray(masks)
        odd = jnp.mod(jnp.sum(a, axis=1), 2) == 1
        delta = jnp.sum(a[..., :-1], axis=1)[None, None, :] - 2 * jnp.einsum(
            "csn,mn->csm", z[..., :-1], a[..., :-1]
        )
        tan = jnp.tan((self.scale * self.betas) / 2)[:, None, None]
        base = -1j * tan
        # Avoid 0**negative and preserve the exact t=0 reference support.
        ratios = jnp.where(
            tan == 0,
            jnp.where(delta == 0, 1.0 + 0j, 0.0 + 0j),
            jnp.power(base, delta),
        )
        return jnp.where(odd[None, None, :], 0.0 + 0j, ratios)

    def _reference(self):
        n = self.width
        bits = ((jnp.arange(1 << n)[:, None] >> jnp.arange(n)) & 1).astype(jnp.float64)
        leaves = bits[:, :-1]
        parity = jnp.mod(jnp.sum(leaves, axis=1), 2)
        t = self.scale * self.betas[:, None]
        amp = jnp.cos(t / 2) ** (n - 1 - leaves.sum(1))
        amp *= (-1j * jnp.sin(t / 2)) ** leaves.sum(1)
        amp = jnp.where(bits[:, -1][None, :] == parity[None, :], amp, 0j)
        return amp

    def post_hadamard_amplitudes(self):
        if self.width > 20:
            raise ValueError("exact amplitudes require at most 20 qubits")
        return self._reference()

    def preparation(self, condition_index=0, wires=None):
        import pennylane as qml

        if not 0 <= condition_index < self.n_conditions:
            raise IndexError("condition outside encoder")
        wires = list(range(self.width)) if wires is None else list(wires)
        t = float(self.scale * self.betas[condition_index])

        def queue():
            for wire in wires[:-1]:
                qml.RX(t, wires=wire)
                qml.CNOT(wires=[wire, wires[-1]])
            for wire in wires:
                qml.Hadamard(wires=wire)

        return queue
