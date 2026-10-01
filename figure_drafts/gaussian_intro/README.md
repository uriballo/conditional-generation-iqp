# Gaussian introductory figure

This editable source generates the installed manuscript figure
`paper/figures/gaussian_intro.pdf`. It uses the default 40-parameter open 4x4
square lattice: 16 one-body phases and 24 horizontal/vertical nearest-neighbor
phases. The campaign has 10 seeds, 300 optimization updates, and 30 jobs for
trained, encoder-only, and frozen-initial-phase arms.

The three profile panels use the lattice checkpoint from seed 17100 at
conditions 0.1291667, 0.5041667, and 0.8791667. These three conditions belong to the 80-condition held-out evaluation grid
and were excluded from training and checkpoint selection. The figure follows the
Curie--Weiss print style: serif Computer Modern text, black axes and labels,
inward major and minor ticks, blue trained curves, neutral dashed targets, and
dotted grey encoder-only curves. The retained `errors.csv` contains the
ten-seed median and interquartile-range data as supporting data for the
profiles-only figure, although it is not plotted here.

The audited summary reports median seed-mean Hamming-profile TVD of 0.06697
for the trained lattice, 0.29216 for encoder-only, and 0.30241 for frozen
initial phases. Median seed-mean full-bitstring TVD is 0.08764, 0.29216, and
0.31013, respectively. There are 109 local profile regressions against the
encoder-only baseline among 800 seed-condition comparisons, so the aggregate
improvement is not uniform. Hamming-profile and full-bitstring TVD are distinct;
the experiment does not establish that two-body gates are necessary.

Evidence is recorded in `results/lattice_20260930/summary.json`,
`results/lattice_20260930/gaussian_manifest.json`, and
`data/provenance.json`. The source snapshot is isolated because the checkout
was changing concurrently during the audit. The historical
`experiments/gaussian_isotropic/RESULTS.md` remains unchanged.

Build with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error gaussian_intro.tex
```

To regenerate the CSV data from audited run files:

```sh
uv run python figure_drafts/gaussian_intro/export_data.py
```
