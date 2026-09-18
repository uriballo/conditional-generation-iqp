# Conditional IQP generation

This repository studies conditional generative models built from a fixed IQP
circuit and a condition-dependent, classically tractable input state. Training
uses sampled Pauli-character estimates of a multiscale Hamming-RBF MMD. Exact
visible-distribution TVD is the main evaluation metric.

The current paper uses three benchmarks:

- a translated discrete Gaussian over Hamming weight, with a broadcast
  product-state encoder;
- the finite-size Curie--Weiss model, with either a product-state encoder or a
  spin-flip-symmetric two-branch encoder;
- a two-parameter discrete Gaussian whose prescribed weight mean and width move
  independently, with a permutation-symmetric shell reference whose mean and
  width are set by two affine maps.

MFC, MNIST, beta-binomial, grid-Ising and Ashkin--Teller experiments are no
longer part of the paper.

## Repository map

- `Overleaf/`: canonical manuscript package. Upload this directory directly to
  Overleaf and compile `main.tex` with pdfLaTeX.
- `iqp-cond` (external, pinned in `pyproject.toml`): the standalone library
  (encoder interface, conditional Pauli estimator, conditional MMD, trainer,
  ancilla conventions), developed at `github.com/uriballo/iqp-cond`;
  `experiments/library_perf/` benchmarks it against the campaign code.
- `configs/exact_tvd_ancilla_v1.json`: frozen Exact-TVD campaign manifest.
- `src/iqp_cond/exact_tvd_*`: campaign planning, training, evaluation, and
  result processing.
- `results/exact-tvd-ancilla-v1/`: active Gaussian and Curie--Weiss artifacts.
- `configs/curie_encoder_ablation_v1.json` and
  `results/curie-encoder-ablation-v1/`: the eight-qubit encoder ablation.
- `configs/gaussian2d_v1.json` and `src/iqp_cond/gaussian2d*.py`: the
  two-parameter Gaussian benchmark, its campaign layer and its ablation arms.
- `results/gaussian2d-v1/`: two-parameter benchmark artifacts.
- `scripts/{run,report,plot}_gaussian2d*.py`: campaign, summary, and figure
  entry points for that benchmark.
- `docs/exact-tvd-ancilla-v1.md`: campaign protocol and execution stages.
- `CONTEXT.md`: current terminology and interpretation rules.

Historical result stores and abandoned experimental lines are intentionally
absent from the working tree. Git history remains the source for tracked legacy
files.

## Setup

```bash
uv sync --group dev
uv run pytest
uv run ruff check src scripts tests
```

Python 3.13 is required. The main numerical stack is JAX, Optax, NumPy, and
SciPy.

## Exact-TVD campaign

Inspect the frozen plan and the current atomic-wave status with:

```bash
uv run python scripts/run_exact_tvd_campaign.py plan
uv run python scripts/status_atomic_campaign.py
```

The original manifest declares 4,300 fits, including 400 MFC fits. The paper
reports the complete 8- and 12-visible-qubit subset: 2,600 Gaussian and
Curie--Weiss fits and exact evaluations. The 16-qubit extension is deferred
for computational reasons; the frozen manifest and historical artifacts are
unchanged. Both Strix queues completed on 4 September 2026.

The production settings are 2,000 optimizer updates, 8,192 estimator samples,
2,000 sampled character operators per bandwidth, three fixed bandwidths, and
ten matched seeds. Exact TVD uses the full visible distribution after immediate
ancilla marginalization.

## Manuscript figures

The manuscript reads publication figures from `Overleaf/figures/`. Plotting
scripts regenerate those files from retained campaign artifacts. Do not restore
the old `paper/` tree. It belonged to a superseded manuscript.

See [the complete-result audit and reproduction commands](docs/paper_8_12/README.md)
for the verified data scope, paired numerical summaries, and figure workflow.

## Cleanup boundary

The in-scope Exact-TVD campaign has finished. Retained shared modules and
figure scripts reproduce the Gaussian and Curie-Weiss results. Retired
confirmatory, phase-OOD, latent-input, and MFC execution code has been removed;
Git history preserves those implementations.

The frozen plan still contains all 4,300 original identities. Data preparation
and default training/evaluation shards now execute only Gaussian and
Curie-Weiss, preserving their original indices. Explicit MFC execution is
unsupported. Historical development plans and selectors retain MFC cells so
archived calibration and optimizer-selection evidence can still be checked;
re-running those MFC cells requires the historical source bundle.
