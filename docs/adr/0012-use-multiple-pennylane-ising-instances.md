# Use multiple PennyLane-Ising coupling instances

The non-shell experiment will use the official PennyLane periodic
random-coupling (4\times4) Ising generator across Conditions. Its canonical
seed-666 coupling matrix is the comparison anchor, and at least three
additional frozen coupling seeds use the same topology and qubit count so that
robustness is tested without confounding model capacity. Exact probabilities
and Bitstring TVD over all (2^{16}) states remain available.
