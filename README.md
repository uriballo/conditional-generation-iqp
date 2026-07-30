# Conditional IQP generation

This repository studies classically trainable conditional generation with IQP
circuits. A condition enters only through a real product-state encoding,

\[
|\psi(\boldsymbol{\alpha})\rangle
=\bigotimes_i R_Y(\alpha_i)|0\rangle,
\]

while every condition shares the same IQP parameters. The implementation
extends the probabilistic expectation-estimation strategy of den Nest to this
product-state family and trains with unbiased, class-balanced Conditional MMD.

The main 19-qubit experiment learns two discrete Gaussian distributions over
Hamming-weight shells. It compares heavy-hex, square-lattice, and complete
two-local ansätze, then contrasts angle conditioning with no-condition and
basis-condition baselines.

## Project map

- `src/iqp_cond/simulator.py`: product-state Pauli-\(Z\) estimator.
- `src/iqp_cond/mmd.py`: unbiased multiscale Conditional MMD.
- `src/iqp_cond/targets.py`: calibrated Gaussian Hamming-weight targets.
- `src/iqp_cond/ansatz.py`: heavy-hex, lattice, and complete generators.
- `src/iqp_cond/training.py`: matched-seed JAX/Optax training.
- `src/iqp_cond/exact.py`: exact 19-qubit evaluation.
- `src/iqp_cond/interpolation.py`: held-out angle interpolation protocol.
- `docs/experiment-design.md`: preregistered experimental design.
- `docs/interpolation-design.md`: frozen interpolation path and decision rules.
- `results/summary/success_criterion.md`: generated condition-wise decision audit.
- `paper/main.tex`: five-page manuscript source.
- `output/pdf/conditional_iqp_generation.pdf`: rendered manuscript.
- `CONTEXT.md`: canonical domain vocabulary.

## Setup and verification

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

Run the estimator calibration:

```bash
uv run python scripts/calibrate_estimators.py
```

Run or resume the full experiment matrix:

```bash
uv run python scripts/run_experiments.py \
  --models all \
  --seeds 0-9 \
  --skip-existing
```

Summarize completed results and generate paper figures:

```bash
uv run python scripts/summarize_results.py
```

Run the held-out conditional-interpolation study:

```bash
uv run python scripts/run_interpolation.py \
  --seeds 0-9 \
  --skip-existing
uv run python scripts/summarize_interpolation.py
```

Its five training angles, four interleaved held-out angles, target path, and
frozen decision rules are specified in `docs/interpolation-design.md`.

Build the manuscript from `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=../tmp/pdfs main.tex
```

## Result at a glance

The preregistered strict criterion is not met. The complete angle-conditioned
model beats both controls on three of the four required condition/metric
comparisons, but the computational-basis control is substantially better on
condition A shell TVD. The angle-conditioned model is much better after
balancing over both conditions; that aggregate is reported as a diagnostic,
not substituted for the condition-wise rule. Exact bitstring TVD also exposes
within-shell nonuniformity that shell histograms alone would hide.

## Claim boundary

The repository demonstrates efficient classical estimation and
backpropagation for the observables used in MMD training. It does not claim
that these particular 19-qubit circuits demonstrate classically hard sampling
or quantum advantage; see `docs/adr/0001-limit-the-claim-to-classical-training.md`.
