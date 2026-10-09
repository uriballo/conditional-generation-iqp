# Conditional switching response assets

These assets contain the frozen empirical response used for the manuscript's
conditional symmetry section. `response.npz` is an exact copy of
`results/switching16-20261009/ANALYSIS_RANGE2P4.npz`; it contains five frozen
fits evaluated on 121 points over `0 <= beta <= 2.4`. The response summary
records the original analysis source hash and the five-fit aggregation. The
separate `reporting.npz` and `reporting-summary.json` preserve the original
12-point report and are not used to select or refit the response curves.

Replay response figures to ignored results with:

```sh
MPLCONFIGDIR=/tmp/iqp-circuit-mpl .venv/bin/python paper/data/switching16/render.py
```

The renderer performs no training or model evaluation. It aggregates the stored
fit arrays and writes figures and provenance receipts to
`results/switching16-publication-replay/`. The repository command
`uv run python run_experiments.py --mode figures --experiment switching16`
also replays both response figures to its selected output directory.

The encoder illustration source has a separate explicit publication rebuild:

```sh
MPLCONFIGDIR=/tmp/iqp-circuit-mpl .venv/bin/python paper/data/switching16/draw_encoders.py
```

This illustration command writes the reviewed `paper/figures/` locations.
`draw_encoders.py` generates the seven-vertex graph schematic with Matplotlib
and the triangle motif diagram from its explicit PennyLane tape.
