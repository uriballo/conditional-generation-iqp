# Paper reproduction repository

This repository contains the paper, a pinned `iqp-cond` library submodule, and
one reproduction entry point backed by `paper_experiments/`.

- Do not edit or publish the library from this repository. Use its public API.
- Retain only experiments and diagnostics described by the current manuscript.
- Current Curie--Weiss HPO is `cw_followup_*`: parity/product inputs and independent
  parity/unconditional baselines. Final comparisons are pending. Older product
  evidence is retained only for the transitional appendix.
- Default reported fits cover Gaussian translation and original product-input
  lattice scaling. Phase-conditioned Gaussian and switching16 have compact replay
  workflows; their frozen training sources remain in local research archives.
- Resume pre-cleanup campaigns only from their frozen source workspaces; never
  weaken source identity checks or replace a running campaign's source.
- Preserve manuscript settings, seed schedules, selection and aggregation when
  refactoring. Record run configurations, revisions and metrics for every fit.
- New runs write to ignored `results/`; never replace reviewed paper artifacts
  automatically. HPO is an explicit, expensive mode.
- Keep historical local research in `.local_research/`, outside the tracked tree.

Commands:

```sh
uv sync --locked --extra hpo
uv run python run_experiments.py --dry-run
uv run python run_experiments.py --smoke
uv run python run_experiments.py --mode figures
uv run pytest
uv run ruff check paper_experiments tests run_experiments.py
```

Use `local_flash` for bounded implementation tasks after the primary agent has
specified behavior and validation. The primary agent owns scientific decisions
and reviews the final diff. Preserve unrelated local files and library edits.
