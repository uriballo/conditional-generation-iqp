# conditional-generation-iqp

Experiments for conditional generation with instantaneous quantum
polynomial-time circuits, built on the standalone [`iqp-cond`](https://github.com/uriballo/iqp-cond)
library.

## Two repositories

The work is deliberately split across two private repositories:

- **`iqp-cond`** ([github.com/uriballo/iqp-cond](https://github.com/uriballo/iqp-cond))
  is the library: encoder interface, conditional Pauli estimator, MMD loss,
  trainer, ancilla conventions, and the PennyLane/Lightning sampling bridge.  It
  has its own tests, README, tags (`v0.1.0` through `v0.3.0`), and release
  cadence.
- **`conditional-generation-iqp`** (this repository) is the experiment
  workspace: shared infrastructure in `experiments/common`, the Gaussian and
  two-parameter Gaussian studies, and the run harness.  It ships no importable
  package and consumes the library editable from the submodule at `iqp-cond/`.

The link is a git submodule tracking the library's `main` plus an editable uv
source, so `uv sync` installs exactly the recorded checkout and adopting a new
library commit is an explicit change here.

| task | repository |
|---|---|
| target dataset, encoder map, study, metric, sweep | this one |
| estimator, MMD loss, trainer, encoder protocol, sampling bridge | `iqp-cond` |

Experiments import only the public `iqp_cond` API
(`experiments/tests/test_api_boundary.py` enforces this), so the library can
evolve and release on its own schedule.

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
