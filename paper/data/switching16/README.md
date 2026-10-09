# Conditional switching response assets

These assets contain the frozen empirical response used for the manuscript's
conditional symmetry section. `response.npz` is an exact copy of
`results/switching16-20261009/ANALYSIS_RANGE2P4.npz`; it contains five frozen
fits evaluated on 121 points over `0 <= beta <= 2.4`. The response summary
records the original analysis source hash and the five-fit aggregation. The
separate `reporting.npz` and `reporting-summary.json` preserve the original
12-point report and are not used to select or refit the response curves.

The figures in `paper/figures/` can be regenerated with:

```sh
MPLCONFIGDIR=/tmp/iqp-circuit-mpl .venv/bin/python paper/data/switching16/render.py
MPLCONFIGDIR=/tmp/iqp-circuit-mpl .venv/bin/python paper/data/switching16/draw_encoders.py
```

The renderer performs no training or model evaluation. It only aggregates the
stored fit arrays and writes provenance receipts in this directory. The
triangle motif diagram is generated from its explicit PennyLane tape.
