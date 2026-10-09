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
| Gaussian translation | Frozen five-seed training and compact figure replay |
| Phase-conditioned Gaussian mean and width | Compact figure, control and finite-validation replay; frozen training sources in local research archives |
| Curie–Weiss | Parity and product inputs; four-family random/TPE HPO; confirmation and final configurations pending |
| Lattice scaling, 64–2048 qubits | Original product-input training settings and compact figure replay |
| Sixteen-qubit signed-graph switching | Compact figure replay; frozen training sources in local research archives |

The default training inventory covers the two experiments with maintained fit
runners: five Gaussian translation fits and eighteen scaling fits. It does not
start HPO or substitute historical Curie–Weiss fits for the pending comparison.

```sh
uv run python run_experiments.py --dry-run
uv run python run_experiments.py --smoke
uv run python run_experiments.py
```

Select `--experiment gaussian` or `--experiment scaling`, set `--workers N`
for concurrent fits, and use `--output PATH` to choose an output root. Normal
fits write to `results/paper/reported/`; smoke fits use a separate namespace
and cannot establish paper results. Full runs are computationally expensive.
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
starts confirmation or final fitting automatically. The manuscript's current
Curie–Weiss numerical comparison remains pending.

Existing campaigns must run and resume from their frozen source workspaces.
The cleanup changes source hashes, so this tree cannot resume the earlier
Strix campaign. Its workspace, launch snapshots and results are preserved.
Do not weaken the identity checks or copy the cleaned tree over that workspace.

## Figures and numerical data

Replay current numerical figures to ignored outputs without fitting:

```sh
uv run python run_experiments.py --mode figures
uv run python run_experiments.py --mode figures --experiment gaussian2d-phase
uv run python run_experiments.py --mode figures --experiment switching16
uv run python run_experiments.py --mode figures --data-root results/paper/reported
```

The two-dimensional Gaussian controls and finite-validation evidence have
separate replay scripts documented in
[their data README](paper/data/gaussian2d_phase/README.md). The signed-graph
encoder illustrations have an additional source in
[the switching data directory](paper/data/switching16/README.md).

The appendix still cites earlier product-input Curie–Weiss evidence. Its compact
data and renderer remain available explicitly:

```sh
uv run python run_experiments.py --mode figures --experiment curie-weiss-historical
```

That replay is excluded from the default current-figure set. It does not produce
the pending parity comparison. Generated figures and tables never replace the
reviewed paper automatically.

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
