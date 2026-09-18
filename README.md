# conditional-generation-iqp

Experiments for conditional generation with instantaneous quantum
polynomial-time circuits, built on the standalone [`iqp-cond`](https://github.com/uriballo/iqp-cond)
library.

## Layout

- `iqp-cond/`: the library as a git submodule tracking its `main` branch.  It is
  installed editable by `uv sync`; initialize it with
  `git submodule update --init` and pull the latest library commit with
  `git submodule update --remote iqp-cond` (then commit the updated pointer).
  The repository is private, so access needs credentials.
- `experiments/`: studies and shared infrastructure.
  - `common/`: datasets, encoder maps, run config and harness, metrics,
    tracking, sweeps, and plotting style.
  - `gaussian/`: one-parameter Gaussian Hamming-profile study.
  - `gaussian2d/`: two-parameter Gaussian study (joint mean and width).
  - `tests/`: framework tests, including the public-API boundary check.
- `paper/`: placeholder for the manuscript source; its content is ignored until
  it is dropped in.
- `results/`: local run outputs, ignored by git.  Each run writes
  `results/<study>/<run_id>/` with `config.json`, `run.json` (paper and library
  revisions), `metrics.json`, and `checkpoint.npz`; `results/index.jsonl`
  indexes completed runs.

## Workflow

```bash
uv sync                     # library (editable) plus dev tools
uv run pytest               # framework and study tests
uv run ruff check .         # lint
uv run python -m experiments.gaussian.study --smoke
uv run python -m experiments.gaussian2d.study --smoke --encoder two_block
```

A full run drops `--smoke`; budgets and seed lists come from each study's
`config.json`.  `--seeds 1,2,3` overrides them.

## Adding a study

1. Add the target family under `experiments/common/datasets.py` (or a study-local
   module) and the encoder map under `experiments/common/encoders.py`.
2. Create `experiments/<study>/` with a `Study` descriptor (conditions, target
   generator, encoder factory), a `config.json` of frozen parameters, and a
   `study.py` entry point that calls `experiments.common.runner`.
3. Cover the new pieces in `experiments/tests/`.

Experiments use only the public `iqp_cond` API; anything the library cannot
express belongs in the library repository, not here.

## Planned integrations

- **MLflow** tracking via `experiments/common/tracking.py:log_mlflow`, mirroring
  each run directory into a local MLflow experiment once the dependency is
  added.
- **Optuna** sweeps via `experiments/common/sweeps.py:optuna_objective`, with
  the run harness as the objective.
