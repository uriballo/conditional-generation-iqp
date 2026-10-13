# Paper reproduction repository

This repository contains the paper, a pinned `iqp-cond` library submodule, and
one reproduction entry point backed by `paper_experiments/`.

- Do not edit or publish the library from this repository. Use its public API.
- Retain only experiments and diagnostics described by the current manuscript.
- Current Curie--Weiss HPO is `cw_followup_*`: parity/product inputs and independent
  parity/unconditional baselines. Audited final comparisons are bundled in
  `paper/data/curie-weiss-followup/`. The main Curie--Weiss models use the squeezed
  parity encoder (`squeezed_parity_encoder.py`), run by `cw_encoder_hpo.py --campaign
  squeezed|squeezed_independent` under the same protocol, and bundled by
  `cw_squeezed_bundle.py` in `paper/data/curie-weiss-squeezed/`. The re-uploaded parity
  campaign (`--campaign reupload`) is development history, not paper evidence. Older product evidence remains explicitly
  replayable and is excluded from current manuscript comparisons.
- Current reported Gaussian evidence is the self-contained binomial-centred
  Gaussian-shaped family, covering translation and two-coordinate conditioning
  with 40 independent 4x4 lattice parameters. Gaussian campaign selection uses
  exact development full TVD. Parity-input lattice scaling remains the default
  scaling evidence. Legacy Gaussian bundles are historical replay only.
  Signed-graph switching is an encoder/symmetry construction; empirical response
  replay is historical and explicitly selected.
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
uv run python run_experiments.py --experiment scaling --dry-run
uv run python run_experiments.py --experiment scaling --smoke
uv run python run_experiments.py --mode figures
uv run pytest
uv run ruff check paper_experiments tests run_experiments.py
```

Use `local_flash` for bounded implementation tasks after the primary agent has
specified behavior and validation. The primary agent owns scientific decisions
and reviews the final diff. Preserve unrelated local files and library edits.
