# Lattice scaling

Eighteen fits, three per size, at 64, 128, 256, 512, 1024 and 2048 qubits.
`scaling_protocol.json` is identical to `paper_experiments/scaling_hpo_protocol.json`
(SHA-256 `066e26d1…`). The protocol file was extended between stages, and each
receipt records the hash of the version it ran under:

| Stage | Recorded hash | Version |
|---|---|---|
| Search at 64 and 256 qubits | `95491a43…` | Earlier text, not recoverable. All 120 search draws and the evaluation budgets match the shipped search space, seeds and evaluation section. |
| Confirmation and final fits, 64–2048 qubits (`physics/fits/`) | `02bbfbeb…` | Shipped values without the `final_evaluation` section, and `final.note` ending in "(author decision 2026-10-15)". |
| All-to-all comparisons (`locality/all-to-all-64*/`) and the hardware fits | `066e26d1…` | Shipped file. |

The shipped file adds `final_evaluation`, which fixes the held-out and dense
evaluation budgets, and shortens `final.note`; no value read by confirmation or
final training differs from `02bbfbeb…`. The first all-to-all run
(`locality/all-to-all-64/`) and the hardware runs originally recorded
`67378254…`, the shipped file with the longer `final.note`, and their hash
field was updated when this bundle was packaged; `locality/all-to-all-64-budget512/`
ran under the shipped file.

- `figure_data.json`: held-out MMD curves per size and seed.
- `physics/fits/`, `physics/dense/`: classically estimated ⟨m²⟩ of the selected fits across 41 temperatures in [0, 2].
- `locality/`: lattice versus all-to-all two-body models at 64 qubits, and the sixteen-qubit comparison.
  The paper's all-to-all comparison is `locality/all-to-all-64-budget512/`, trained at the
  estimator budget of the lattice fits. `locality/all-to-all-64/`, whose search also varied
  the budget, is the earlier comparison and is kept for reference.
- `score_study/`: sixteen-qubit check of the selection score against exact TVD.

MMD kernels depend on the size, so MMD values are not comparable across sizes.
