"""Check the conditional rotations in the preparation resource derivation."""

import numpy as np
import pytest

from iqp_cond.exact import enumerate_bitstrings


@pytest.mark.parametrize("m", [3, 4, 6])
@pytest.mark.parametrize("g", [0.1, 0.7])
def test_coherent_ring_preparation(m, g):
    bits = enumerate_bitstrings(m)
    spins = 1 - 2 * bits
    v = np.exp(g * np.array([[1, -1], [-1, 1]]))
    probabilities = np.full(2**m, 0.5)
    for i in range(1, m):
        suffix = np.linalg.matrix_power(v, m - i)
        total = np.linalg.matrix_power(v, m - i + 1)
        a, u, b = bits[:, 0], bits[:, i - 1], bits[:, i]
        probabilities *= v[u, b] * suffix[b, a] / total[u, a]
    energy = (spins * np.roll(spins, -1, axis=1)).sum(1)
    target = np.exp(g * energy)
    target /= target.sum()
    np.testing.assert_allclose(probabilities, target, atol=1e-14)
    prepared = np.sqrt(probabilities) * np.exp(1j * np.pi / 8 * energy)
    direct = np.exp((g / 2 + 1j * np.pi / 8) * energy)
    direct /= np.linalg.norm(direct)
    np.testing.assert_allclose(prepared, direct, atol=1e-14)
