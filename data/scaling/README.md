# Lattice scaling

Eighteen fits, three per size, at 64, 128, 256, 512, 1024 and 2048 qubits.
`scaling_protocol.json` is identical to `paper_experiments/scaling_hpo_protocol.json`
(SHA-256 `066e26d1…`). The protocol file was extended between stages, and each
receipt records the hash of the version it ran under:

| Stage | Recorded hash | Version |
|---|---|---|
| Search at 64 and 256 qubits | `95491a43…` | Earlier text, not recoverable. All 120 search draws and the evaluation budgets match the shipped search space, seeds and evaluation section. |
| Confirmation and final fits, 64–2048 qubits (`physics/fits/`) | `02bbfbeb…` | Shipped values without the `final_evaluation` section, and `final.note` ending in "(author decision 2026-10-15)". |
| All-to-all comparison (`locality/`) and the hardware fits | `066e26d1…` | Shipped file. |

The shipped file adds `final_evaluation`, which fixes the held-out and dense
evaluation budgets, and shortens `final.note`; no value read by confirmation or
final training differs from `02bbfbeb…`. The all-to-all and hardware runs
originally recorded `67378254…`, the shipped file with the longer `final.note`,
and their hash field was updated when this bundle was packaged.

- `figure_data.json`: held-out MMD curves per size and seed.
- `physics/fits/`, `physics/dense/`: classically estimated ⟨m²⟩ of the selected fits across 41 temperatures in [0, 2].
- `locality/`: lattice versus all-to-all two-body models at 64 qubits, and the sixteen-qubit comparison.

MMD kernels depend on the size, so MMD values are not comparable across sizes.
