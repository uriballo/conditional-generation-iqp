# Sixteen-qubit selection-score study

Checks, at sixteen qubits on the 4×4 lattice where exact probabilities are
available, how well the scaling selection score E_n ranks checkpoints by exact
full-distribution TVD. Produced by `paper_experiments/scaling_score_study.py`.

## Settings

- Model: parity encoder, open 4×4 lattice (40 parameters), Curie–Weiss target.
- Configurations: 120 random draws from the scaling search space at the
  reference size (alpha = gamma = 0), 60 with the linear schedule (S0) and 60
  with the saturating schedule (S1). Hyperparameter seed 20261015, target seed
  20261015, stream seed 20261016.
- Training: 2000 Adam steps at β ∈ {0.6, 0.8, 1.2, 1.4}, 10000 training and
  2000 validation draws per temperature, batch 450, checkpoints every 100 steps
  including step 0 (21 per configuration, 2520 in total).
- E_n: (2 max + trapezoidal mean) / 3 over the dev grid
  {0.3, 0.5, …, 1.7} of the MMD², averaged over two estimation banks
  (1024 operators, 2048 reference samples) and both bandwidths; negative
  estimates retained. 2000 dev target draws per temperature.
- Training-temperature MMD²: validation MMD² averaged over the four training
  temperatures, same banks and budget.
- Ground truth: exact TVD from statevector probabilities at 40 test
  temperatures β = 0.025, 0.075, …, 1.975, aggregated by the same
  (2 max + trapezoidal mean) / 3 rule. `mean_tvd` is the plain mean over the
  40 temperatures.

## Result

Spearman correlation over all 2520 checkpoints of the 120 configurations
(`spearman.json`):

| Score | vs aggregated TVD | vs mean TVD |
| --- | --- | --- |
| E_n | 0.967 | 0.954 |
| Training-temperature MMD² | 0.827 | 0.868 |

Restricted to the 60 saturating configurations: 0.986 and 0.903. Restricted to
the 60 linear configurations: 0.956 and 0.727. Using only the checkpoints at
multiples of 200 steps (the search interval): 0.967 and 0.839.

## Files

- `checkpoints.json`: per configuration, the hyperparameters, checkpoint steps,
  dev-grid MMD² (8 values), training-temperature MMD² (4 values) and exact TVD
  (40 values) at every checkpoint.
- `spearman.json`: the correlations above.

## Provenance and rerun

`checkpoints.json` was collected from the 2026-10-15 development run of these
settings on the Strix GPU. Rerunning S0-000 on CPU with the shipped script
reproduces all its stored values to within 1e-9.

```sh
uv run python -m paper_experiments.scaling_score_study analyze   # from checkpoints.json
uv run python -m paper_experiments.scaling_score_study run results/score-study
uv run python -m paper_experiments.scaling_score_study collect results/score-study
```

Training takes about 7 minutes per configuration on an Apple-silicon CPU (about
14 hours for all 120) and about 45 seconds on the Strix GPU. `collect`
overwrites `checkpoints.json`.
