"""Memory-bounded access to the unchanged broadcast RY--RZ input state.

For identical local reference amplitudes a0,a1, flipping a mask changes the
number of ones by d=|a|-2 z.a. The exact product ratio is (a1/a0)**d. Counting
flips avoids the generic samples-by-masks-by-qubits complex product tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BroadcastPhaseEncoder:
    betas: jax.Array
    width: int
    amp_zero: jax.Array
    amp_one: jax.Array
    beta_scale: float = 0.75

    @classmethod
    def from_betas(cls, betas, width, beta_scale=0.75):
        values = np.asarray(betas, dtype=np.float64).reshape(-1)
        if width < 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("nonempty finite temperatures and positive width required")
        scale = float(beta_scale)
        if not np.isfinite(scale) or scale <= 0 or np.any(abs(scale * values) > np.pi):
            raise ValueError(
                "positive finite scale and encoder angles within [-pi, pi] required"
            )
        angles = np.stack([np.full_like(values, np.pi / 2), scale * values], axis=1)[
            ..., None
        ]
        # The public one-qubit encoder pins the same floating-point amplitudes.
        local = ciqp.AngleEncoder(angles, axes=("Y", "Z")).post_hadamard_amplitudes()
        return cls(jnp.asarray(values), width, local[:, 0], local[:, 1], scale)

    def tree_flatten(self):
        return (self.betas, self.amp_zero, self.amp_one), (self.width, self.beta_scale)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        width, scale = metadata
        return cls(children[0], width, children[1], children[2], scale)

    @property
    def n_conditions(self):
        return int(self.betas.shape[0])

    @property
    def n_qubits(self):
        return self.width

    @property
    def angles(self):
        return jnp.broadcast_to(
            jnp.stack(
                [jnp.full_like(self.betas, jnp.pi / 2), self.beta_scale * self.betas],
                axis=1,
            )[..., None],
            (self.n_conditions, 2, self.width),
        )

    def select(self, condition):
        if not 0 <= condition < self.n_conditions:
            raise IndexError("condition outside encoder")
        sl = slice(condition, condition + 1)
        return type(self)(
            self.betas[sl],
            self.width,
            self.amp_zero[sl],
            self.amp_one[sl],
            self.beta_scale,
        )

    def sample(self, key, n_samples):
        return jax.random.bernoulli(
            key,
            p=jnp.abs(self.amp_one[:, None, None]) ** 2,
            shape=(self.n_conditions, n_samples, self.width),
        ).astype(jnp.float64)

    def amplitude_ratios(self, samples, masks):
        overlap = jnp.einsum("csn,mn->csm", samples, masks)
        delta = jnp.sum(masks, axis=1)[None, None, :] - 2 * overlap
        base = self.amp_one / self.amp_zero
        return jnp.exp(delta * jnp.log(base[:, None, None]))

    def preparation(self, condition):
        return ciqp.AngleEncoder(
            self.angles, axes=("Y" * self.width, "Z" * self.width)
        ).preparation(condition)

    def post_hadamard_amplitudes(self):
        if self.width > 20:
            raise ValueError("exact amplitudes require at most 20 qubits")
        return ciqp.AngleEncoder(
            self.angles, axes=("Y" * self.width, "Z" * self.width)
        ).post_hadamard_amplitudes()
