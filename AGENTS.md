# Agent notes

## Layout

- `iqp-cond/`: library submodule, installed editable by `uv sync`.  It tracks
  the library's `main`; update with `git submodule update --remote iqp-cond` and
  commit the pointer.  Library changes belong in that repository, not here.
- `experiments/`: studies (`gaussian/`, `gaussian2d/`) and shared infrastructure
  (`common/`), with tests in `experiments/tests/`.
- `paper/`: placeholder for the manuscript source; its content is gitignored.
- `results/`: local run outputs, gitignored.  Do not commit them.

## Commands

- `uv sync` installs the environment (library editable plus dev tools).
- `uv run pytest` runs the experiment tests.
- `uv run ruff check .` and `uv run ruff format .` for lint and formatting.
- Smoke runs: `uv run python -m experiments.gaussian.study --smoke` and
  `uv run python -m experiments.gaussian2d.study --smoke --encoder two_block`.

## Conventions

- Experiments import only the public `iqp_cond` API; the boundary test in
  `experiments/tests/test_api_boundary.py` enforces it.
- Every run goes through `experiments.common.runner` so it produces a run
  directory with config, revisions, metrics, and checkpoint.
- New studies follow the `Study` descriptor pattern: conditions, target
  generator, encoder factory, `config.json`, and a `study.py` entry point.
- MLflow and Optuna integrations are planned but not installed; keep their
  adapter hooks in `experiments/common/tracking.py` and
  `experiments/common/sweeps.py`.
