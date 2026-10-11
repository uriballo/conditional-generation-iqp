"""Two-coordinate phase encoding on a uniform input with an untied lattice.

The central input is exactly |+>^n. Every IQP unitary preserves this state,
so its uniform bitstring law is an architectural baseline, not a learned fit.
Only the reference amplitudes are permutation symmetric. Untied output phases
require full bitstring probabilities for accuracy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from .ansatz import rectangular_lattice_ansatz


@lru_cache
def _krawtchouk(n: int) -> np.ndarray:
    return np.asarray(
        [
            [
                sum(
                    (-1) ** ell * comb(j, ell) * comb(n - j, k - ell)
                    for ell in range(max(0, k - n + j), min(j, k) + 1)
                )
                for k in range(n + 1)
            ]
            for j in range(n + 1)
        ],
        dtype=np.float64,
    )


def _conditions(values) -> np.ndarray:
    pairs = np.asarray(values, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) < 1:
        raise ValueError("conditions must be a nonempty (C,2) array")
    if not np.isfinite(pairs).all() or np.any(abs(pairs) > 1):
        raise ValueError("condition coordinates must lie in [-1,1]")
    return pairs


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CollectivePhaseEncoder:
    """Polynomial reference access for exp[-iu Jz/2-iv Jz²/40]|+>^n."""

    conditions: jax.Array
    shells: jax.Array
    counts: jax.Array
    n: int

    @classmethod
    def from_conditions(cls, conditions, n=16):
        values = _conditions(conditions)
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        jz = n / 2 - np.arange(n + 1)
        u, v = values.T
        amplitudes = np.exp(
            -1j * (u[:, None] * jz / 2 + v[:, None] * jz**2 / 40)
        ) / 2 ** (n / 2)
        shells = amplitudes @ _krawtchouk(n).T / 2 ** (n / 2)
        # Pin the exact central support rather than sampling cancellation noise.
        central = np.all(values == 0, axis=1)
        shells[central] = 0
        shells[central, 0] = 1
        counts = np.asarray([comb(n, j) for j in range(n + 1)])
        if np.max(abs((abs(shells) ** 2 * counts).sum(axis=1) - 1)) > 1e-11:
            raise FloatingPointError("reference amplitudes are not normalized")
        return cls(jnp.asarray(values), jnp.asarray(shells), jnp.asarray(counts), n)

    def tree_flatten(self):
        return (self.conditions, self.shells, self.counts), self.n

    @classmethod
    def tree_unflatten(cls, n, children):
        return cls(*children, n)

    @property
    def n_conditions(self):
        return int(self.shells.shape[0])

    @property
    def n_qubits(self):
        return self.n

    def select(self, condition):
        if not 0 <= condition < self.n_conditions:
            raise IndexError("condition outside encoder")
        sl = slice(condition, condition + 1)
        return type(self)(self.conditions[sl], self.shells[sl], self.counts, self.n)

    def sample(self, key, n_samples):
        key_weight, key_bits = jax.random.split(key)
        weights = jax.random.categorical(
            key_weight,
            jnp.log(abs(self.shells[:, None]) ** 2 * self.counts[None, None]),
            shape=(self.n_conditions, n_samples),
        )
        scores = jax.random.uniform(key_bits, (self.n_conditions, n_samples, self.n))
        ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
        return (ranks < weights[..., None]).astype(jnp.float64)

    def amplitude_ratios(self, samples, masks):
        weight = samples.sum(-1).astype(jnp.int32)
        flipped = (
            weight[..., None]
            + masks.sum(-1)[None, None]
            - 2 * jnp.einsum("csn,mn->csm", samples, masks)
        ).astype(jnp.int32)
        denominator = self.shells[jnp.arange(self.n_conditions)[:, None], weight]
        numerator = self.shells[jnp.arange(self.n_conditions)[:, None, None], flipped]
        safe = jnp.where(denominator != 0, denominator, 1)[..., None]
        return jnp.where(denominator[..., None] != 0, numerator / safe, 0)

    def post_hadamard_amplitudes(self):
        return self.shells[:, basis(self.n)[1]]

    def preparation(self, condition):
        import pennylane as qml

        u, v = np.asarray(self.conditions)[condition]

        def queue():
            for wire in range(self.n):
                qml.Hadamard(wires=wire)
                qml.RZ(u / 2, wires=wire)
            for left in range(self.n):
                for right in range(left + 1, self.n):
                    qml.IsingZZ(v / 40, wires=[left, right])
            qml.GlobalPhase(v * self.n / 160)

        return queue


@lru_cache
def basis(n=16):
    if n < 1 or n > 20:
        raise ValueError("dense evaluation requires 1 <= n <= 20")
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.float64)
    return bits, bits.sum(1).astype(int)


def hadamard(states):
    out = np.asarray(states, dtype=np.complex128).copy()
    width = out.shape[-1]
    if width < 2 or width & (width - 1):
        raise ValueError("state width must be a power of two")
    for bit in range(width.bit_length() - 1):
        blocks = out.reshape(len(out), -1, 2, 1 << bit)
        left, right = blocks[:, :, 0], blocks[:, :, 1]
        out = np.stack((left + right, left - right), axis=2).reshape(out.shape)
        out /= np.sqrt(2)
    return out


@lru_cache
def characters(n=16):
    generators = np.asarray(rectangular_lattice_ansatz(n).generators)
    return 1 - 2 * ((basis(n)[0] @ generators.T) % 2)


def probabilities(theta, conditions, n=16):
    encoder = CollectivePhaseEncoder.from_conditions(conditions, n=n)
    reference = np.asarray(encoder.post_hadamard_amplitudes())
    phase = np.exp(1j * (characters(n) @ np.asarray(theta)))
    return abs(hadamard(reference * phase)) ** 2


def target_profiles(conditions, n=16):
    """Gaussian-shaped exponential tilts with an exact binomial center.

    rho(k|u,v) is proportional to Binomial(n,1/2)(k) times
    exp[-(k-n/2-3u)^2/(2 sigma(v)^2)+(k-n/2)^2/(2 sigma(0)^2)].
    The nominal mean/width coordinates are not exact finite-support moments.
    """
    values = _conditions(conditions)
    u, v = values.T
    center_sd = np.sqrt(n) / 2
    sd = center_sd * (1 + 0.2 * v)
    weights = np.arange(n + 1)
    displacement = weights - n / 2
    counts = np.asarray([comb(n, k) for k in weights], dtype=np.float64)
    log_profile = (
        np.log(counts)[None]
        - (displacement - 3 * u[:, None]) ** 2 / (2 * sd[:, None] ** 2)
        + displacement**2 / (2 * center_sd**2)
    )
    profile = np.exp(log_profile - log_profile.max(1, keepdims=True))
    profile /= profile.sum(1, keepdims=True)
    profile[np.all(values == 0, axis=1)] = counts / 2**n
    return profile


def target_probabilities(conditions, n=16):
    weights = basis(n)[1]
    counts = np.asarray([comb(n, k) for k in range(n + 1)])
    return target_profiles(conditions, n)[:, weights] / counts[weights]


def sample_targets(conditions, n_samples, rng, n=16):
    # Sample a weight, then a uniform subset, without enumerating bitstrings.
    profiles = target_profiles(conditions, n)
    weights = np.stack([rng.choice(n + 1, n_samples, p=p) for p in profiles])
    scores = rng.random((len(profiles), n_samples, n))
    ranks = np.argsort(np.argsort(scores, axis=-1), axis=-1)
    return (ranks < weights[..., None]).astype(np.float64)


def metrics(theta, conditions, n=16):
    q = probabilities(theta, conditions, n)
    p = target_probabilities(conditions, n)
    weights = basis(n)[1]
    profiles = np.stack(
        [np.bincount(weights, weights=row, minlength=n + 1) for row in q]
    )
    target = target_profiles(conditions, n)
    counts = np.asarray([comb(n, k) for k in range(n + 1)])
    mean = q @ weights
    sd = np.sqrt(np.maximum(q @ weights**2 - mean**2, 0))
    pmean = p @ weights
    psd = np.sqrt(np.maximum(p @ weights**2 - pmean**2, 0))
    return {
        "full_tvd": (abs(q - p).sum(1) / 2).tolist(),
        "profile_tvd": (abs(profiles - target).sum(1) / 2).tolist(),
        "within_shell_residual": (
            abs(q - profiles[:, weights] / counts[weights]).sum(1) / 2
        ).tolist(),
        "generated_mean": mean.tolist(),
        "generated_sd": sd.tolist(),
        "target_mean": pmean.tolist(),
        "target_sd": psd.tolist(),
        "normalization_max_error": float(abs(q.sum(1) - 1).max()),
    }
