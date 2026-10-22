# Gaussian-shaped family, conditional IQP

Binomial-centred Gaussian-shaped targets with translation and two-coordinate
centre/width conditioning, on a 4x4 lattice with 40 independent phases.

- `protocol.json`: targets, encoders, condition sets, budgets, candidates and random streams.
- `PARAMETERS.npz`: the five selected parameter vectors for grid fitting, axis fitting and translation.
- `DIAGNOSTICS.npz`: full and profile TVD sweeps, target moments and per-fit responses.
- `SUMMARY.json`: per-condition errors, fit identities, source hashes, software versions and aggregation.

The campaign comprises eight screening, eight confirmation, ten final
two-coordinate and five translation fits. Full TVD on development conditions
selects checkpoints and configurations. At the central condition the target is
uniform, as is every model, so that condition is excluded from the averages.

```sh
uv run python run_experiments.py --mode figures --experiment gaussian-lattice
uv run python -m paper_experiments.gaussian2d_lattice_run --output results/gaussian --stage campaign
uv run python -m paper_experiments.gaussian2d_lattice_report --campaign results/gaussian --output results/gaussian/replay
```
