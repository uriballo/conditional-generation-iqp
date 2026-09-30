# Gaussian introductory figure

This editable source generates the installed manuscript figure
`paper/figures/gaussian_intro.pdf`. It uses the default 40-parameter open 4x4
square lattice: 16 one-body phases and 24 horizontal/vertical nearest-neighbor
phases. The campaign has 10 seeds, 300 optimization updates, and 30 jobs for
trained, encoder-only, and frozen-initial-phase arms.

The three profile panels use the lattice checkpoint from seed 17100 at
conditions 0.1291667, 0.5041667, and 0.8791667. The error panel shows the
pointwise median and interquartile range across all ten seeds. All 80 plotted
conditions were held out from training and checkpoint selection. The editable
TVD axis uses a 0 to 0.15 range.

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

The committed CSVs are the audited inputs for this PDF. Regenerating them
from checkpoints requires the local campaign artifacts and experiment exporter,
which are outside this figures-only commit.
