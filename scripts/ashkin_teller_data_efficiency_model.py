"""Ancilla interaction model for the n=12 data-efficiency study.

This module is deliberately separate from the frozen training/runtime sources.
Ancillas are prepared in ``|0>`` before the first Hadamard, hence they are
uniform bits in the post-Hadamard reference distribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.bond_reference import BOND_ENCODING, sample_two_rings, two_ring_ratios
from iqp_cond.encoding import TWO_BLOCK_BRANCH_ENCODING, two_block_shell_amplitudes
from iqp_cond.exact import _phase_values, _reference_state, fwht
from iqp_cond.exact_tvd_campaign import expected_bodyness_sigmas
from iqp_cond.exact_tvd_runner import _derived_seed
from iqp_cond.exact_tvd_training import sample_nonidentity_masks
from iqp_cond.mmd import _target_terms
from iqp_cond.simulator import (
    _amplitude_ratios_two_block,
    _phase_differences,
    _sample_reference_two_block,
)

TRAIN = np.array(
    [
        (a, b)
        for a in (0.10, 0.22, 0.38, 0.54, 0.70)
        for b in (0.10, 0.22, 0.38, 0.54, 0.70)
        if a in (0.10, 0.70) or b in (0.10, 0.70)
    ]
)


def generators(n_ancillas: int) -> np.ndarray:
    """Return the 12-visible complete bank plus visible--ancilla couplings."""
    if n_ancillas not in (0, 2, 4):
        raise ValueError("n_ancillas must be 0, 2, or 4")
    base = np.asarray(complete_ansatz(12).generators, dtype=np.int8)
    if not n_ancillas:
        return base
    rows = []
    for anc in range(n_ancillas):
        for visible in range(12):
            row = np.zeros(12 + n_ancillas, dtype=np.int8)
            row[visible] = row[12 + anc] = 1
            rows.append(row)
    return np.vstack([np.pad(base, ((0, 0), (0, n_ancillas))), rows])


def _joint_samples(key, condition, encoder, n_ancillas, n_samples):
    if encoder == "cat":
        a = jnp.asarray(condition)[None, :].repeat(6, axis=1)
        a = 0.225 * jnp.pi - 0.20 * jnp.pi * (a - 0.1) / 0.6
        sa, sb = two_block_shell_amplitudes(a, 6, 6)
        visible = _sample_reference_two_block(key, sa, sb, n_samples)[0]
    elif encoder == "bond":
        visible = sample_two_rings(key, jnp.asarray(condition), 12, n_samples)
    else:
        raise ValueError("encoder must be 'cat' or 'bond'")
    if not n_ancillas:
        return visible
    bits = jax.random.bernoulli(
        jax.random.fold_in(key, 918273 + n_ancillas),
        0.5,
        (n_samples, n_ancillas),
    ).astype(jnp.float64)
    return jnp.concatenate([visible, bits], axis=1)


def sample_values(params, g, ops, condition, encoder, key, n_samples=2048):
    """Return per-draw observable estimates, shape ``(n_samples, n_ops)``."""
    g = jnp.asarray(g, dtype=jnp.float64)
    ops = jnp.asarray(ops, dtype=jnp.float64)
    samples = _joint_samples(key, condition, encoder, int(g.shape[1] - 12), n_samples)
    visible_ops = ops[:, :12]
    if encoder == "cat":
        a = jnp.asarray(condition)[None, :].repeat(6, axis=1)
        a = 0.225 * jnp.pi - 0.20 * jnp.pi * (a - 0.1) / 0.6
        sa, sb = two_block_shell_amplitudes(a, 6, 6)
        ratios = _amplitude_ratios_two_block(
            samples[None, :, :12], visible_ops, sa, sb
        )[0]
    else:
        ratios = two_ring_ratios(samples[:, :12], visible_ops, jnp.asarray(condition))
    phase = _phase_differences(jnp.asarray(params), g, ops, samples[None, ...])[0]
    return jnp.real(ratios * jnp.exp(1j * phase))


def exact_predictions(params, g, conditions, encoder):
    """Exact visible marginals after tracing appended ancillas."""
    out = []
    for condition in np.asarray(conditions):
        phase = _phase_values(np.asarray(params), np.asarray(g))
        ref_angles = np.repeat(condition, 6)
        if encoder == "cat":
            ref_angles = 0.225 * np.pi - 0.20 * np.pi * (ref_angles - 0.1) / 0.6
        ref = _reference_state(
            ref_angles,
            12,
            TWO_BLOCK_BRANCH_ENCODING if encoder == "cat" else BOND_ENCODING,
        )
        ref = np.tile(ref, 2 ** (g.shape[1] - 12)) / np.sqrt(2 ** (g.shape[1] - 12))
        amp = fwht(ref * np.exp(1j * phase)) / np.sqrt(2 ** g.shape[1])
        p = np.abs(amp) ** 2
        p /= p.sum()
        out.append(p.reshape(2 ** (g.shape[1] - 12), 4096).sum(axis=0))
    return np.asarray(out)


def sampled_losses(g, validation, seed, encoder, n_samples=2048, n_ops=512):
    """Build a compact train/validation MMD pair for the ancilla study."""
    g = np.asarray(g, dtype=np.int8)
    validation = np.asarray(validation)
    banks = []
    for band, sigma in enumerate(expected_bodyness_sigmas(12)):
        for condition in range(len(TRAIN)):
            key = jax.random.PRNGKey(_derived_seed(seed, band, condition) & (2**32 - 1))
            masks, weights = sample_nonidentity_masks(key, n_ops, 12, sigma)
            banks.append((np.asarray(masks), np.asarray(weights), condition))
    width = max(len(x[0]) for x in banks)
    masks = jnp.asarray(
        np.stack([np.pad(x[0], ((0, width - len(x[0])), (0, 0))) for x in banks])
    )
    weights = jnp.asarray(
        np.stack([np.pad(x[1], (0, width - len(x[1]))) for x in banks])
    )
    conditions = jnp.asarray([x[2] for x in banks])
    train_jax = jnp.asarray(TRAIN)

    def estimate(params, targets, key):
        targets = jnp.asarray(targets)
        keys = jax.random.split(key, len(banks))

        def one(args):
            ops, weight, c, sample_key = args
            ops = jnp.pad(ops, ((0, 0), (0, g.shape[1] - 12)))
            sv = sample_values(
                params, g, ops, train_jax[c], encoder, sample_key, n_samples
            )
            means, target_sq = _target_terms(targets[c][None, ...], ops[:, :12])
            sums = jnp.sum(sv, axis=0)
            model_sq = (sums**2 - jnp.sum(sv**2, axis=0)) / (
                n_samples * (n_samples - 1)
            )
            return jnp.sum(
                weight * (model_sq - 2 * jnp.mean(sv, axis=0) * means[0] + target_sq[0])
            )

        return jnp.mean(jax.lax.map(one, (masks, weights, conditions, keys)))

    key = jax.random.PRNGKey(_derived_seed(seed, "validation") & (2**32 - 1))
    return (
        estimate,
        jax.jit(lambda p: estimate(p, jnp.asarray(validation), key)),
        {"masks": np.asarray(masks), "weights": np.asarray(weights)},
    )
