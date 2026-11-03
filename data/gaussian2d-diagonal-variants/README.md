# Gaussian-shaped family, diagonal-conditioning variants

Three variants of `../gaussian2d-diagonal/` under the same protocol.

| Directory | Diagonal layer | Condition map | Parameters | Mixed-pair TVD |
|---|---|---|---|---|
| `lattice-quadratic/` | 4x4 lattice (40 phases) | w0 + u wu + v wv + u^2 wuu + v^2 wvv | 200 | 0.078 ± 0.002 |
| `complete-affine/` | all 136 one- and two-body phases | w0 + u wu + v wv | 408 | 0.103 ± 0.004 |
| `complete-quadratic/` | all 136 one- and two-body phases | quadratic as above | 680 | 0.109 ± 0.005 |

Each directory holds `SUMMARY.json`, `PARAMETERS.npz`, `DIAGNOSTICS.npz` and
`protocol.json` in the format of `../gaussian2d-diagonal/`. `five_way.json`
compares the variants with the affine baseline and the conditional IQP model.

Campaigns `diagonal_quadratic`, `diagonal_complete` and
`diagonal_complete_quadratic` of `paper_experiments.gaussian2d_hpo`; bundles from
`gaussian2d_diagonal_report --campaign ... --output ...`, and `five_way.json` from
`gaussian2d_diagonal_report --five-way --output data/gaussian2d-diagonal-variants/five_way.json`.
