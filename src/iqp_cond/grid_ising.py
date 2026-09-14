"""Exact ferromagnetic Ising targets on small rectangular lattices."""

from __future__ import annotations

from collections import deque
from math import comb
from typing import Any

import numpy as np

Shape = tuple[int, int]
Edge = tuple[int, int]


def _validate_shape(shape: Shape) -> None:
    if len(shape) != 2 or any(not isinstance(value, int) for value in shape):
        raise ValueError("shape must contain two integer dimensions")
    if any(value < 2 for value in shape):
        raise ValueError("both lattice dimensions must be at least two")


def lattice_edges(shape: Shape, *, periodic: bool = True) -> tuple[Edge, ...]:
    """Return unique undirected nearest-neighbor edges in row-major order.

    Periodic lattices with a dimension of length two contain one undirected
    edge between the two sites, not two parallel interactions.
    """
    _validate_shape(shape)
    rows, columns = shape
    result: set[Edge] = set()
    for row in range(rows):
        for column in range(columns):
            source = row * columns + column
            for delta_row, delta_column in ((1, 0), (0, 1)):
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if periodic:
                    neighbor_row %= rows
                    neighbor_column %= columns
                elif neighbor_row >= rows or neighbor_column >= columns:
                    continue
                target = neighbor_row * columns + neighbor_column
                if source != target:
                    result.add(tuple(sorted((source, target))))
    return tuple(sorted(result))


def enumerate_spins(shape: Shape) -> np.ndarray:
    """Enumerate all spin strings in little-endian integer order."""
    _validate_shape(shape)
    n_spins = shape[0] * shape[1]
    labels = np.arange(1 << n_spins, dtype=np.uint64)[:, None]
    bits = (labels >> np.arange(n_spins, dtype=np.uint64)) & 1
    return (2 * bits.astype(np.int8) - 1).astype(np.int8)


def sufficient_statistics(
    spins: np.ndarray,
    shape: Shape,
    *,
    periodic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-neighbor alignment and magnetization for each sample."""
    values = np.asarray(spins, dtype=np.int8)
    if values.shape[-1] != shape[0] * shape[1]:
        raise ValueError("the final spin dimension does not match shape")
    edge_list = lattice_edges(shape, periodic=periodic)
    left = np.asarray([edge[0] for edge in edge_list])
    right = np.asarray([edge[1] for edge in edge_list])
    interaction = np.sum(values[..., left] * values[..., right], axis=-1)
    magnetization = np.sum(values, axis=-1)
    return interaction, magnetization


def exact_probabilities(
    shape: Shape,
    beta: float,
    field: float,
    *,
    periodic: bool = True,
) -> np.ndarray:
    r"""Return ``p(s) propto exp(beta sum_(ij) s_i s_j + field sum_i s_i)``.

    ``beta`` and ``field`` are dimensionless. The coupling is fixed to one.
    """
    _validate_shape(shape)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not np.isfinite(field):
        raise ValueError("field must be finite")
    interaction, magnetization = sufficient_statistics(
        enumerate_spins(shape), shape, periodic=periodic
    )
    logits = beta * interaction + field * magnetization
    logits = logits - np.max(logits)
    probability = np.exp(logits)
    return probability / np.sum(probability)


def sample_exact(
    shape: Shape,
    beta: float,
    field: float,
    size: int,
    *,
    seed: int,
    periodic: bool = True,
) -> np.ndarray:
    """Draw exact independent samples by categorical enumeration."""
    if size < 0:
        raise ValueError("size must be nonnegative")
    probability = exact_probabilities(shape, beta, field, periodic=periodic)
    rng = np.random.default_rng(seed)
    indices = rng.choice(probability.size, size=size, p=probability)
    return enumerate_spins(shape)[indices]


def total_variation_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the total variation distance between two probability vectors."""
    return float(0.5 * np.sum(np.abs(np.asarray(left) - np.asarray(right))))


def fast_walsh_hadamard(values: np.ndarray) -> np.ndarray:
    """Return the unnormalized Walsh-Hadamard transform."""
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1 or result.size == 0 or result.size & (result.size - 1):
        raise ValueError("values must be a nonempty power-of-two vector")
    width = 1
    while width < result.size:
        for start in range(0, result.size, 2 * width):
            left = result[start : start + width].copy()
            right = result[start + width : start + 2 * width].copy()
            result[start : start + width] = left + right
            result[start + width : start + 2 * width] = left - right
        width *= 2
    return result


def character_moments(probability: np.ndarray) -> np.ndarray:
    """Return every binary-character expectation in integer mask order."""
    return fast_walsh_hadamard(probability)


def _pair_graph_distances(shape: Shape, *, periodic: bool) -> dict[Edge, int]:
    edge_list = lattice_edges(shape, periodic=periodic)
    n_spins = shape[0] * shape[1]
    adjacency = [set() for _ in range(n_spins)]
    for left, right in edge_list:
        adjacency[left].add(right)
        adjacency[right].add(left)
    distances: dict[Edge, int] = {}
    for source in range(n_spins):
        found = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in found:
                    found[neighbor] = found[node] + 1
                    queue.append(neighbor)
        for target in range(source + 1, n_spins):
            distances[(source, target)] = found[target]
    return distances


def _mask_for_pair(pair: Edge) -> int:
    return (1 << pair[0]) | (1 << pair[1])


def spectrum_diagnostic(
    shape: Shape,
    beta: float,
    field: float,
    *,
    n_ops: int = 2_000,
    periodic: bool = True,
) -> dict[str, Any]:
    """Quantify exact Hamming-RBF signal without attributing locality to it."""
    from .exact_tvd_campaign import expected_bodyness_sigmas

    if n_ops < 1:
        raise ValueError("n_ops must be positive")
    target = exact_probabilities(shape, beta, field, periodic=periodic)
    uniform = np.full_like(target, 1.0 / target.size)
    delta = character_moments(target - uniform)
    n_spins = shape[0] * shape[1]
    mask_weights = np.asarray([mask.bit_count() for mask in range(1 << n_spins)])
    pair_distances = _pair_graph_distances(shape, periodic=periodic)
    edge_set = set(lattice_edges(shape, periodic=periodic))

    bands: list[dict[str, Any]] = []
    for sigma in expected_bodyness_sigmas(n_spins):
        inclusion_probability = (1.0 - np.exp(-1.0 / (2.0 * sigma**2))) / 2.0
        per_mask_weight = inclusion_probability**mask_weights * (
            1.0 - inclusion_probability
        ) ** (n_spins - mask_weights)
        squared_difference = delta**2
        signal_by_weight = {
            str(weight): float(
                np.sum(
                    per_mask_weight[mask_weights == weight]
                    * squared_difference[mask_weights == weight]
                )
            )
            for weight in range(n_spins + 1)
        }
        signal_by_distance: dict[str, float] = {}
        for distance in sorted(set(pair_distances.values())):
            masks = np.asarray(
                [
                    _mask_for_pair(pair)
                    for pair, pair_distance in pair_distances.items()
                    if pair_distance == distance
                ],
                dtype=int,
            )
            signal_by_distance[str(distance)] = float(
                np.sum(per_mask_weight[masks] * squared_difference[masks])
            )
        edge_masks = np.asarray(
            [_mask_for_pair(pair) for pair in sorted(edge_set)], dtype=int
        )
        nonedge_masks = np.asarray(
            [_mask_for_pair(pair) for pair in pair_distances if pair not in edge_set],
            dtype=int,
        )
        edge_signal = float(
            np.sum(per_mask_weight[edge_masks] * squared_difference[edge_masks])
        )
        nonedge_signal = float(
            np.sum(per_mask_weight[nonedge_masks] * squared_difference[nonedge_masks])
        )
        nonidentity_probability = 1.0 - (1.0 - inclusion_probability) ** n_spins
        weight_two_probability = (
            comb(n_spins, 2)
            * inclusion_probability**2
            * (1.0 - inclusion_probability) ** (n_spins - 2)
            / nonidentity_probability
        )
        expected_edges = (
            n_ops * weight_two_probability * len(edge_set) / comb(n_spins, 2)
        )
        total_signal = float(np.sum(per_mask_weight * squared_difference))
        bands.append(
            {
                "sigma": float(sigma),
                "inclusion_probability": float(inclusion_probability),
                "total_mmd_squared_vs_uniform": total_signal,
                "signal_by_weight": signal_by_weight,
                "weight2_edge_signal": edge_signal,
                "weight2_nonedge_signal": nonedge_signal,
                "weight2_edge_fraction": edge_signal / (edge_signal + nonedge_signal)
                if edge_signal + nonedge_signal > 0
                else 0.0,
                "weight2_signal_by_graph_distance": signal_by_distance,
                "expected_edge_masks_per_operator_bank": float(expected_edges),
            }
        )
    return {
        "shape": list(shape),
        "periodic": periodic,
        "beta": float(beta),
        "field": float(field),
        "n_ops": n_ops,
        "edge_count": len(edge_set),
        "bands": bands,
    }
