# Lattice scaling

Eighteen fits, three per size, at 64, 128, 256, 512, 1024 and 2048 qubits under
`scaling_protocol.json` (identical to `paper_experiments/scaling_hpo_protocol.json`).

- `figure_data.json`: held-out MMD curves per size and seed.
- `physics/fits/`, `physics/dense/`: classically estimated ⟨m²⟩ of the selected fits across 41 temperatures in [0, 2].
- `locality/`: lattice versus all-to-all two-body models at 64 qubits, and the sixteen-qubit comparison.

MMD kernels depend on the size, so MMD values are not comparable across sizes.
