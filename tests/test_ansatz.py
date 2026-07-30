import networkx as nx
import numpy as np

from iqp_cond import complete_ansatz, heavy_hex_ansatz, lattice_ansatz


def test_agreed_ansatz_sizes_and_connectivity():
    heavy_hex = heavy_hex_ansatz()
    lattice = lattice_ansatz()
    complete = complete_ansatz()

    assert (heavy_hex.n_qubits, heavy_hex.graph.number_of_edges()) == (19, 20)
    assert (lattice.n_qubits, lattice.graph.number_of_edges()) == (19, 29)
    assert (complete.n_qubits, complete.graph.number_of_edges()) == (19, 171)
    assert heavy_hex.n_parameters == 39
    assert lattice.n_parameters == 48
    assert complete.n_parameters == 190
    assert all(nx.is_connected(item.graph) for item in (heavy_hex, lattice, complete))


def test_every_ansatz_includes_all_one_body_rows_first():
    for ansatz in (heavy_hex_ansatz(), lattice_ansatz(), complete_ansatz()):
        expected = np.eye(ansatz.n_qubits)
        assert np.array_equal(
            np.asarray(ansatz.generators[: ansatz.n_qubits]),
            expected,
        )
        assert np.all(np.asarray(ansatz.generators[ansatz.n_qubits :]).sum(axis=1) == 2)
