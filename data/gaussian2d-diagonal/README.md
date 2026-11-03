# Hamming-weight family, diagonal conditioning

Fixed input |0>^16 with the 4x4 lattice diagonal layer, whose 40 phases depend
on the condition through theta(u, v) = w0 + u w_u + v w_v (120 parameters).
Conditions, budgets, search space, stages, target pools and fit streams are
those of the conditional IQP campaign (`../gaussian_lattice/`), paired by
stage, repeat and start.

- `protocol.json`: protocol.
- `PARAMETERS.npz`: `axes`, the ten selected (3 * 40,) weight vectors.
- `DIAGNOSTICS.npz`: 41x41 full and profile TVD grids (`grid_tvd[fit, v, u]`) and axis responses.
- `SUMMARY.json`: search record, per-fit errors and the matching conditional IQP rows.

```sh
for phase in random tpe confirmation final; do
  uv run python -m paper_experiments.gaussian2d_hpo --campaign diagonal --root results/gaussian/diagonal --phase $phase --workers 6
done
uv run python -m paper_experiments.gaussian2d_diagonal_report --campaign results/gaussian/diagonal --output data/gaussian2d-diagonal
```
