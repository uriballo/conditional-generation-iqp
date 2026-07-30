"""IQP generator families for the topology comparison."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import networkx as nx
import numpy as np

HEAVY_HEX_DISTANCE_3_EDGES = (
    (0, 9),
    (0, 13),
    (1, 13),
    (1, 14),
    (2, 14),
    (3, 9),
    (3, 15),
    (4, 15),
    (4, 16),
    (5, 12),
    (5, 16),
    (6, 17),
    (7, 17),
    (7, 18),
    (8, 12),
    (8, 18),
    (10, 14),
    (10, 16),
    (11, 15),
    (11, 17),
)


@dataclass(frozen=True)
class Ansatz:
    """One- and two-body IQP generators induced by an interaction graph."""

    name: str
    graph: nx.Graph
    generators: jnp.ndarray

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return self.graph.number_of_nodes()

    @property
    def n_parameters(self) -> int:
        """Number of independently parameterized generators."""
        return int(self.generators.shape[0])


def generators_from_graph(graph: nx.Graph) -> jnp.ndarray:
    """Return all one-body rows followed by one two-body row per graph edge."""
    nodes = sorted(graph.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError("graph nodes must be contiguous integers starting at zero")
    n_qubits = len(nodes)
    rows = np.eye(n_qubits, dtype=np.float64).tolist()
    for left, right in sorted(tuple(sorted(edge)) for edge in graph.edges()):
        row = np.zeros(n_qubits, dtype=np.float64)
        row[left] = 1.0
        row[right] = 1.0
        rows.append(row.tolist())
    return jnp.asarray(rows, dtype=jnp.float64)


def _ansatz(name: str, graph: nx.Graph) -> Ansatz:
    if not nx.is_connected(graph):
        raise ValueError(f"{name} graph must be connected")
    return Ansatz(name=name, graph=graph, generators=generators_from_graph(graph))


def heavy_hex_ansatz() -> Ansatz:
    """Return the canonical distance-3 heavy-hex ansatz on 19 qubits."""
    graph = nx.Graph()
    graph.add_nodes_from(range(19))
    graph.add_edges_from(HEAVY_HEX_DISTANCE_3_EDGES)
    return _ansatz("heavy_hex", graph)


def lattice_ansatz() -> Ansatz:
    """Return the agreed 4x5 nearest-neighbor lattice with one corner removed."""
    full = nx.grid_2d_graph(4, 5)
    full.remove_node((3, 4))
    ordered = sorted(full.nodes())
    graph = nx.relabel_nodes(full, {node: idx for idx, node in enumerate(ordered)})
    return _ansatz("lattice", graph)


def complete_ansatz(n_qubits: int = 19) -> Ansatz:
    """Return a complete two-local ansatz."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    return _ansatz("complete", nx.complete_graph(n_qubits))
