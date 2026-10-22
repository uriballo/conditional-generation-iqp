# Gaussian-shaped family, diagonal conditioning

Fixed input |0>^16 with the 4x4 lattice diagonal layer, whose 40 phases depend
on the condition through theta(u, v) = w0 + u w_u + v w_v (120 parameters).
Conditions, budgets, optimizer, seeds, target pools and selection follow the
conditional IQP axis fits. The one-body initialization mean is {0, pi/4}, since
pi/4 gives the uniform distribution for the |0> input.

- `protocol.json`: protocol.
- `PARAMETERS.npz`: the five selected (3, 40) weight matrices.
- `DIAGNOSTICS.npz`: 41x41 full and profile TVD grids (`grid_tvd[fit, v, u]`) and axis responses.
- `SUMMARY.json`: selection scores, per-fit errors and the matching conditional IQP rows.

Selected configuration: one-body mean pi/4, learning rate 0.01, from 4
screening, 2x2 confirmation and 5 final fits.

```sh
uv run python -m paper_experiments.gaussian2d_diagonal_run --output results/gaussian-diagonal --stage campaign
uv run python -m paper_experiments.gaussian2d_diagonal_report --campaign results/gaussian-diagonal --output results/gaussian-diagonal/replay
```
