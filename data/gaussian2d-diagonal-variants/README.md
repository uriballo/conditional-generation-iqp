# Gaussian-shaped family, diagonal-conditioning variants

Three variants of `../gaussian2d-diagonal/` under the same protocol.

| Directory | Diagonal layer | Condition map | Parameters | Test-pair TVD |
|---|---|---|---|---|
| `lattice-quadratic/` | 4x4 lattice (40 phases) | w0 + u wu + v wv + u^2 wuu + v^2 wvv | 200 | 0.231 ± 0.012 |
| `complete-affine/` | all 136 one- and two-body phases | w0 + u wu + v wv | 408 | 0.262 ± 0.076 |
| `complete-quadratic/` | all 136 one- and two-body phases | quadratic as above | 680 | 0.385 ± 0.023 |

Each directory holds `SUMMARY.json`, `PARAMETERS.npz` and `DIAGNOSTICS.npz` in
the format of `../gaussian2d-diagonal/`. `five_way.json` compares the variants
with the affine baseline and the conditional IQP model.

Protocols: `gaussian2d_diagonal_quadratic_protocol.json`,
`gaussian2d_diagonal_complete_protocol.json` and
`gaussian2d_diagonal_complete_quadratic_protocol.json`. Run with
`gaussian2d_diagonal_run --stage campaign --basis {affine,quadratic} --ansatz {lattice,complete}`.
