"""Experiment-level generator banks."""

from __future__ import annotations

import iqp_cond
import numpy as np


def open_rectangle_shape(n: int) -> tuple[int, int]:
    """Return the closest-factor open rectangle for ``n`` qubits."""
    if n < 1:
        raise ValueError("n must be positive")
    factors = [(d, n // d) for d in range(1, int(np.sqrt(n)) + 1) if n % d == 0]
    if not factors:
        raise ValueError("n must have an integer rectangle factorization")
    height, width = min(factors, key=lambda pair: (pair[1] - pair[0], pair[0]))
    return height, width


def rectangular_lattice_ansatz(n: int, *, name: str | None = None) -> iqp_cond.Ansatz:
    """Return an open nearest-neighbor lattice with row-major qubit order."""
    height, width = open_rectangle_shape(n)
    edges = [
        (row * width + col, row * width + col + 1)
        for row in range(height)
        for col in range(width - 1)
    ]
    edges += [
        (row * width + col, (row + 1) * width + col)
        for row in range(height - 1)
        for col in range(width)
    ]
    edges = sorted(edges)
    rows = np.zeros((n + len(edges), n), dtype=np.float64)
    rows[np.arange(n), np.arange(n)] = 1.0
    for row_index, (left, right) in enumerate(edges, start=n):
        rows[row_index, [left, right]] = 1.0
    bank_name = name or f"lattice_{height}x{width}"
    return iqp_cond.Ansatz(np.asarray(rows), name=bank_name)


def square_lattice_ansatz(n_visible: int = 16) -> iqp_cond.Ansatz:
    """Return the Gaussian study's open 4x4 lattice, with sorted pair rows."""
    if n_visible != 16:
        raise ValueError("square lattice ansatz requires exactly 16 visible qubits")
    edges = [(i, i + 1) for i in range(16) if i % 4 != 3]
    edges += [(i, i + 4) for i in range(12)]
    rows = [np.eye(16, dtype=np.float64)[i] for i in range(16)]
    for left, right in sorted(edges):
        row = np.zeros(16, dtype=np.float64)
        row[[left, right]] = 1.0
        rows.append(row)
    return iqp_cond.Ansatz(np.asarray(rows), name="square_lattice")
