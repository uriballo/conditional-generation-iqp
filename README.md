# Conditional generation with IQP Born machines

This repository contains the [paper](paper/main.pdf), its manuscript experiments,
and a pinned [iqp-cond library](https://github.com/uriballo/iqp-cond) submodule.
Historical experiments are kept outside the tracked tree in `.local_research/`.

## Install

Use Python 3.13 and [uv](https://docs.astral.sh/uv/):

```sh
git clone --recurse-submodules https://github.com/uriballo/conditional-generation-iqp.git
cd conditional-generation-iqp
uv sync --locked --extra hpo
```

For an existing clone, run `git submodule update --init`. The library is pinned
to `f90875da72b3379c1ea38b844a0a908a20eacf24`. Use its public API and preserve
the pin when reproducing the paper.

## Current experiment scope

| Experiment | Retained workflow |
| --- | --- |
| Gaussian-shaped translation and center/width control | Self-contained sampled campaign; five final fits per design with independent lattice parameters; compact replay in `paper/data/gaussian_lattice/` |
| Curie–Weiss | Audited parity/product comparison with frozen HPO, confirmation and final data in `paper/data/curie-weiss-followup/` |
| Lattice scaling, 64–2048 qubits | Parity-encoder transfer with dedicated scaling HPO and compact figure replay |
| Sixteen-qubit signed-graph switching | Encoder and symmetry construction; historical response replay remains explicit |

The current Gaussian campaign has its own explicit entry point because it includes configuration selection and 31 fits. The reported runner supports the eighteen scaling fits. Curie--Weiss uses its separate HPO/confirmation/final workflow. No default command silently starts a historical Gaussian experiment.

```sh
uv run python run_experiments.py --experiment scaling --dry-run
uv run python run_experiments.py --experiment scaling --smoke
uv run python run_experiments.py --experiment scaling
```

Set `--workers N` for concurrent scaling fits and `--output PATH` for a separate output root. Normal scaling fits write to `results/paper/reported/`; smoke fits use a separate namespace and cannot establish paper results. Full runs are computationally expensive.
Every fit records configuration, source/library identity, checkpoints and
metrics. Matching completed jobs resume; incompatible configurations or sources
are rejected.

## Curie–Weiss HPO

The sole maintained Curie–Weiss search is frozen in
`paper_experiments/cw_followup_protocol.json`. It compares shared parity,
shared product, independent parity and independent unconditional models. Each
study receives 60 random proposals and an explicit 60-proposal TPE continuation.
The search uses Adam with constant or cosine learning rates, exact full-TVD
scores and the specified physics tolerance for checkpoint selection.

```sh
uv run python run_experiments.py --mode hpo --experiment curie-weiss --dry-run
uv run python run_experiments.py --mode hpo --experiment curie-weiss --smoke --workers 2 --output results/cw-cleanup-smoke
uv run python run_experiments.py --mode hpo --experiment curie-weiss --workers 2 --output results/cw-new-campaign
# After reviewing a completed random stage, explicitly continue that output root:
uv run python run_experiments.py --mode hpo --experiment curie-weiss --hpo-phase tpe --workers 2 --output results/cw-new-campaign
```

`--experiment curie-weiss-followup` remains an alias. `--mode hpo` without an
experiment also selects this campaign. Both phases stop for review; neither
starts confirmation or final fitting automatically. Completed audited
confirmation and final results are stored in `paper/data/curie-weiss-followup/`.

After both HPO stages have stopped for review, explicitly authorized
confirmation and final reporting use `--hpo-phase confirmation` and
`--hpo-phase final`. Confirmation uses five fresh repeats and three starts;
final reporting uses ten fresh repeats and three starts. Reporting runs both
stages explicitly and writes sibling `confirmation/` and `final/` outputs,
including final `figure_data.json`. All HPO screening remains paused for review
before these stages. Existing frozen Strix sources are preserved.

```sh
uv run python run_experiments.py --mode hpo --experiment curie-weiss --hpo-phase reporting --workers 2 --output results/cw-new-campaign
```

This runs the two authorized stages sequentially while preserving the frozen
HPO campaign, and writes `results/cw-new-campaign/confirmation/` and
`results/cw-new-campaign/final/` alongside `results/cw-new-campaign/hpo/`.

Existing campaigns must run and resume from their frozen source workspaces.
The cleanup changes source hashes, so this tree cannot resume the earlier
Strix campaign. Its workspace, launch snapshots and results are preserved.
Do not weaken the identity checks or copy the cleaned tree over that workspace.

## Figures and numerical data

Replay current numerical figures to ignored outputs without fitting:

```sh
uv run python run_experiments.py --mode figures
uv run python run_experiments.py --mode figures --experiment gaussian-lattice
uv run python run_experiments.py --mode figures --experiment gaussian2d-phase
uv run python run_experiments.py --mode figures --data-root results/paper/reported
uv run python run_experiments.py --mode figures --experiment curie-weiss --data-root PATH
```

Historical signed-graph response figures remain available explicitly (they are
not part of the current manuscript figure set):

```sh
uv run python run_experiments.py --mode figures --experiment switching16
```

For the current Curie--Weiss replay, the default paper data root contains
`curie-weiss-followup/figure_data.json` from completed final data. To use a
different root, `PATH` must contain that same relative file. Screening and
smoke artifacts are insufficient, and replay never replaces reviewed paper
artifacts automatically.

Historical one-angle Gaussian controls and finite-validation evidence have
separate replay scripts documented in
[their data README](paper/data/gaussian2d_phase/README.md). The signed-graph
encoder illustrations have an additional source in
[the switching data directory](paper/data/switching16/README.md).

The historical product-input Curie–Weiss evidence remains available explicitly:

```sh
uv run python run_experiments.py --mode figures --experiment curie-weiss-historical
```

That replay is excluded from the default current-figure set and is not current
manuscript evidence. Generated figures and tables never replace the reviewed
paper automatically.

The current Gaussian evidence is a self-contained sampled campaign for the
binomial-centred Gaussian-shaped family. Replay it with
`python -m paper_experiments.gaussian2d_lattice_report --campaign ROOT
--output ROOT/replay`; legacy Gaussian bundles remain available for historical
replay.

The campaign itself is explicit and expensive:
`python -m paper_experiments.gaussian2d_lattice_run --output results/gaussian2d-lattice --stage campaign`.

## Checks

```sh
uv run pytest
uv run ruff check paper_experiments tests run_experiments.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The TeX build requires REVTeX, Quantikz and TikZ. Preserve reviewed numerical
artifacts when refactoring. Gaussian translation summarizes seed-wise mean
errors by their median; Curie–Weiss summaries use means and sample standard
deviations of per-repeat metrics. Scaling retains negative unbiased MMD estimates,
size-dependent kernels and the original 512-mask/1024-reference budgets.

## Before publication

- [ ] Check that the manuscript's Data Availability statement identifies all code, data, trained parameters and environments needed to reproduce the reported results, with accurate links and an explicit description of any separately archived material.
- [ ] Archive the complete current Gaussian campaign, including all fitting pools, screening/confirmation trajectories, final checkpoints and exact validated library sources/version. The compact `gaussian_lattice/` bundle includes selected parameters and diagnostics; full training evidence currently remains local. Distinguish current evidence from historical Gaussian replay.
- [ ] Publish the signed-graph illustration source and reference-state access instructions used by the manuscript construction; retain historical learning arrays and response replay as explicitly labelled archival material.
- [ ] Create a stable release or archive of the reviewed code and supporting artifacts, and verify that all cited repository and archive links are publicly accessible.
- [ ] Verify installation and documented smoke and figure-replay commands from a fresh checkout. Document the resource requirements and commands for full reproduction without launching expensive fits solely for this check.
- [ ] Reconcile the final manuscript, numerical tables, figures and supporting artifacts, resolve remaining author notes, and compile and visually inspect the submission PDF.
