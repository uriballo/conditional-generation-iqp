# Gaussian-shaped family, conditional IQP

Binomial-centred Gaussian-shaped targets with two-coordinate centre/width
conditioning, encoded by exp[-i(u/2)Jz] exp[-i(v/40)Jz^2] |+>^16 and a shared
4x4 lattice layer with 40 independent phases. Training uses the nine axis
conditions; selection uses eight development conditions on the axes; the
sixteen mixed pairs are reported only.

- `protocol.json`: targets, encoder, condition sets, budgets, search space, stages and random streams.
- `PARAMETERS.npz`: `axes`, the ten selected (40,) parameter vectors of the final fits.
- `DIAGNOSTICS.npz`: 41x41 full and profile TVD grids (`grid_tvd[fit, v, u]`), target moments and per-fit axis responses.
- `SUMMARY.json`: search record (screening, confirmation, winner), per-condition errors per fit and aggregation.

Search: 60 random and 60 TPE proposals with five repeats each, three
configurations confirmed with five repeats of three starts, and the winner
refitted with ten repeats of three starts. Full TVD on the development
conditions selects checkpoints, starts and configurations. At the central
condition the target is uniform, as is every model.

```sh
uv run python run_experiments.py --mode figures --experiment gaussian-lattice
for phase in random tpe confirmation final; do
  uv run python -m paper_experiments.gaussian2d_hpo --campaign input --root results/gaussian/input --phase $phase --workers 6
done
uv run python -m paper_experiments.gaussian2d_lattice_report --campaign results/gaussian/input --output data/gaussian_lattice
```
