# Two-dimensional Gaussian conditioning data

This directory contains dense-grid TVD arrays and a compact numerical summary
used by the two-dimensional Gaussian figure. The 41 by 41 grid includes the
fitting and development conditions. The common withheld table domain contains
1,652 pairs excluded from both designs. The plotted values average the five
frozen fits at each condition; the reported uncertainty is the sample standard
deviation of the five per-fit condition averages.

The dense-grid evaluation follows model selection and therefore serves as a
post-selection diagnostic. The original fitting records, campaign manifests,
and implementation-level receipts remain in the ignored `results/` and
`.local_research/` directories. `provenance.json` records their paths and
hashes without publishing library source files or absolute local paths.

Rebuild the figure from the repository root with:

```sh
.venv/bin/python paper/figure_sources/gaussian2d_heatmaps.py \
  --results paper/data/gaussian2d \
  --output results/gaussian2d-figures
```

The correlated input state is part of the model resource. These results make
no quantum-advantage claim and do not establish perfect calibration of the
width response.
