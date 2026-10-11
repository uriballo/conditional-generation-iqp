# Binomial-centered Gaussian-shaped family

This is the current evidence for single-coordinate profile translation and two-coordinate center/width conditioning. It replaces the earlier discrete-Gaussian and one-angle demonstrations in the manuscript.

- `protocol.json` specifies targets, encoders, condition sets, independent lattice parameters, budgets, candidates and random streams.
- `PARAMETERS.npz` contains all five selected 40-parameter vectors for each of grid fitting, axis fitting and translation.
- `DIAGNOSTICS.npz` contains unsmoothed full/profile TVD sweeps, actual target moments and per-fit response arrays.
- `SUMMARY.json` records per-condition selected errors, terminal errors, fit identities, frozen training-source hashes, software versions and aggregation. SHA-256 values verify the replay inputs.

At the central condition the target and every parameterized model produce uniform bitstrings, with an exact Binomial(16, 1/2) Hamming-weight profile. This match is architectural and is excluded from reporting averages. Encoder-only probabilities are uniform at every condition. The targets are Gaussian-shaped binomial tilts, not literal discrete Gaussians; nominal tilt coordinates are not their actual moments. The untied learned lattice can break permutation symmetry, so full bitstring TVD is the primary accuracy and development-selection metric. Translation profile panels use four held-out reporting conditions; legend labels show their actual target means, while model selection is unchanged.

The campaign comprises eight screening fits, eight confirmation fits, ten final two-coordinate fits and five translation fits. Both two-coordinate configurations were frozen before final reporting. Final fits use independent repeat seeds. Conditions and phase features were explored in preceding design work. Exact development targets select checkpoints and configurations; this is an exploratory capability demonstration, not a finite-data-only or prospective encoder-selection study. It does not establish a need for forty independent parameters.

Replay figures without fitting:

```sh
uv run python run_experiments.py --mode figures --experiment gaussian-lattice
```

Run a new campaign and extract a separate bundle:

```sh
uv run python -m paper_experiments.gaussian2d_lattice_run --output results/gaussian-new --stage smoke
uv run python -m paper_experiments.gaussian2d_lattice_run --output results/gaussian-campaign --stage campaign
uv run python -m paper_experiments.gaussian2d_lattice_report --campaign results/gaussian-campaign --output results/gaussian-campaign/replay
```

Smoke and scientific campaigns require different output roots. Campaigns freeze the exact sources before fitting and reject incompatible resumption. New outputs never replace these reviewed inputs automatically.

The original complete campaign, including fitting pools, trajectories and every screening/confirmation checkpoint, is preserved locally at `results/gaussian-centered-lattice-20261010/`. Publication must archive this evidence and the exact validated library version identified in `SUMMARY.json`, including any recorded local changes. The selected parameters and compact diagnostics here permit current figure and accuracy replay, but do not replace that complete training archive. Earlier Gaussian bundles are historical and explicitly replayable.
