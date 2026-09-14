import itertools

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.bond_reference import (
    ring_amplitudes,
    sample_ring,
    two_ring_amplitudes,
)


def _states(m):
    return jnp.asarray(list(itertools.product((0, 1), repeat=m)), dtype=jnp.int32)


def test_ring_amplitudes_are_normalized_and_spin_flip_symmetric():
    bits = _states(4)
    amp = np.asarray(ring_amplitudes(bits, 0.3))
    assert np.isclose(np.sum(np.abs(amp) ** 2), 1.0, atol=1e-12)
    assert np.max(np.abs(amp[bits.sum(axis=1) % 2 == 1])) < 1e-12


def test_sampler_matches_small_ring_distribution():
    bits = _states(4)
    target = np.asarray(np.abs(ring_amplitudes(bits, 0.7)) ** 2)
    draws = np.asarray(sample_ring(jax.random.key(17), 0.7, 4, 50000))
    indices = np.asarray(draws @ (1 << np.arange(4)), dtype=np.int32)
    empirical = np.bincount(indices, minlength=16) / len(indices)
    assert 0.5 * np.abs(empirical - target).sum() < 0.02


def test_two_ring_factorization_and_jit_scaling():
    bits = _states(8)[:32]
    couplings = jnp.asarray((0.1, 0.7))
    out = two_ring_amplitudes(bits, couplings)
    expected = ring_amplitudes(bits[:, :4], couplings[0]) * ring_amplitudes(
        bits[:, 4:], couplings[1]
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)
    large = jax.jit(sample_ring, static_argnums=(2, 3))(jax.random.key(1), 0.3, 32, 8)
    assert large.shape == (8, 32)
    assert np.all(np.isfinite(large))


def test_brute_amplitudes_and_all_prefix_conditionals():
    from iqp_cond.bond_reference import _step_matrix, _suffix, _transfer
    from iqp_cond.exact import enumerate_bitstrings, fwht

    for m in (4, 6):
        bits = enumerate_bitstrings(m)
        for g in (0.1, 0.7):
            s = 1 - 2 * bits
            energy = (s * np.roll(s, -1, axis=1)).sum(1)
            physical = np.exp((g / 2 + 1j * np.pi / 8) * energy)
            physical /= np.linalg.norm(physical)
            want = fwht(physical) / np.sqrt(2**m)
            got = np.asarray(ring_amplitudes(bits, g))
            np.testing.assert_allclose(got, want, atol=1e-13)
            law = np.abs(want) ** 2
            t = _transfer(jnp.asarray(g), jnp.pi / 8)
            e = []
            for z in (0, 1):
                a = np.asarray(_step_matrix(t, jnp.asarray(z)))
                e.append(np.kron(a, a.conj()))
            suffix = np.asarray(_suffix(t, m))
            for k in range(m):
                for prefix_bits in itertools.product((0, 1), repeat=k):
                    mask = np.all(bits[:, :k] == np.array(prefix_bits), axis=1)
                    mass = law[mask].sum()
                    if mass < 1e-14:
                        continue
                    prefix = np.eye(4, dtype=complex)
                    for z in prefix_bits:
                        prefix = prefix @ e[z]
                    weights = np.array(
                        [np.trace(prefix @ ez @ suffix[k + 1]).real for ez in e]
                    )
                    exact = law[mask & (bits[:, k] == 1)].sum() / mass
                    np.testing.assert_allclose(
                        weights[1] / weights.sum(), exact, atol=1e-11
                    )


def test_invalid_coupling_visible_and_large_support():
    for g in (-0.1, np.nan, 0.8):
        draws = np.asarray(sample_ring(jax.random.key(9), g, 4, 4))
        assert np.isnan(draws).all()
        assert np.isnan(np.asarray(ring_amplitudes(_states(4), g))).all()
    draws = np.asarray(
        jax.jit(sample_ring, static_argnums=(2, 3))(jax.random.key(8), 0.7, 32, 128)
    )
    assert np.isfinite(draws).all()
    assert np.all(draws.sum(1) % 2 == 0)
    amplitude = np.asarray(ring_amplitudes(draws, 0.7))
    assert np.isfinite(amplitude).all()
    assert np.all(np.abs(amplitude) > 0)


def test_ratio_normalizers_cancel_without_changing_queries():
    from iqp_cond.bond_reference import sample_two_rings, two_ring_ratios

    for m in (4, 6, 32):
        c = jnp.asarray([0.1, 0.7])
        samples = sample_two_rings(jax.random.key(m), c, 2 * m, 32)
        ops = np.random.default_rng(m).integers(0, 2, (8, 2 * m))
        fast = np.asarray(two_ring_ratios(samples, jnp.asarray(ops), c))
        z = np.asarray(samples, dtype=np.int32)
        slow = (
            np.asarray(two_ring_amplitudes(z[:, None, :] ^ ops[None, :, :], c))
            / np.asarray(two_ring_amplitudes(z, c))[:, None]
        )
        np.testing.assert_allclose(fast, slow, atol=1e-10, rtol=1e-8)
