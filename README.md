# Conditional generation with IQP Born machines

This repository contains the [paper](paper/main.pdf), its numerical experiments,
and a pinned link to the [iqp-cond library](https://github.com/uriballo/iqp-cond).

## Install

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). Repository access is required
while either repository is private.

```sh
git clone --recurse-submodules https://github.com/uriballo/conditional-generation-iqp.git
cd conditional-generation-iqp
uv sync --locked --extra hpo
```

For an existing clone, run `git submodule update --init` before installation.
The submodule is pinned to `f90875da72b3379c1ea38b844a0a908a20eacf24`.
Do not update it to a moving branch when reproducing this paper. Checkpoint
recording and optimizer orchestration live here and require no library patches.

## Reproduce

One entry point runs the paper experiments with their reported settings:

```sh
uv run python run_experiments.py --dry-run
uv run python run_experiments.py --smoke
uv run python run_experiments.py
```

The default campaign includes Gaussian translation, the shared Curie--Weiss
model, both independent baseline families, original lattice scaling from 64 to
2048 qubits, and the three-qubit symmetry example. It uses the selected frozen
optimizer settings; it does not repeat hyperparameter searches.

Select a subset with `--experiment gaussian`, `curie-weiss`, `scaling`, or
`symmetry`. Set `--workers N` for concurrent fits and `--output PATH` to choose
an output directory. The default is one local worker and `results/paper/`.
Full runs are computationally expensive. Smoke mode reduces budgets and uses a
separate output namespace; its results are not paper results.

Repeat the full staged hyperparameter searches explicitly:

```sh
uv run python run_experiments.py --mode hpo --dry-run
uv run python run_experiments.py --mode hpo --smoke
uv run python run_experiments.py --mode hpo
```

HPO includes 48 unpruned trials each for shared-model cosine Adam and
Schedule-Free Adam, and one aggregate 48-trial Adam study per independent
baseline family. Screening, confirmation and final evaluation use distinct
seed groups. Proposal batches and sampler-update order are fixed. The expanded
shared-model development grid supplies selection diagnostics, not extra
training conditions. Independent baselines fit the reported temperatures.

Runs record their configuration, code and library revision, checkpoints and
metrics. Matching completed jobs resume; incompatible run configurations are
rejected. Regenerated results and plots stay in the selected output directory
and do not overwrite the reviewed paper.

## Figures and numerical data

Compact audited data in `paper/data/` retain per-seed profiles, profile/full TVD,
scaling trajectories, selected settings and provenance. Regenerate numerical
figures and tables without training:

```sh
uv run python run_experiments.py --mode figures
# Render completed new runs instead of committed data:
uv run python run_experiments.py --mode figures --data-root results/paper/reported
```

Figures show raw circuit distributions. Curie--Weiss tables summarize per-seed
errors by mean and sample standard deviation; they are not errors of the mean
plotted profile. Scaling uses median and interquartile range across seed-level
bank averages, retains negative unbiased estimates and uses size-dependent
bandwidths. Its original training budgets remain 512 masks and 1024 reference
draws, with expected mask weights two and four.

## Paper and checks

The source is `paper/main.tex`, with bibliography `paper/references.bib` and
editable illustration sources in `paper/figure_sources/`. Compile with a TeX
installation providing REVTeX, Quantikz and TikZ:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Run the focused reproduction checks from the repository root:

```sh
uv run pytest
uv run ruff check paper_experiments tests run_experiments.py
```

The supporting modules contain only paper targets, encoders, training,
evaluation, HPO and plotting. Exploratory studies and historical research
artifacts are excluded; local results and archives are ignored by Git.
