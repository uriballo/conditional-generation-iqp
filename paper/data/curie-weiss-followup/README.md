# Curie–Weiss follow-up data

This bundle contains the audited final numerical inputs for the four-family
Curie–Weiss comparison: shared parity, shared product, independent parity and
independent unconditional. Full-precision selected configurations are stored
in `figure_data.json`; `REPORTING_AUDIT.json` and the review-ready records
document the source identities and completion checks.

The protocol is frozen in
`paper_experiments/cw_followup_protocol.json`. Shared models train at anchors
`[0.1, 0.7, 1.3, 1.95]`. Checkpoint selection uses A17, and shared
confirmation ranks candidates on A17 union B20. The F101 grid is reserved for
final reporting: it does not select checkpoints or configurations. Marker
values beta=0.4, 1 and 2 are excluded from shared fitting and selection;
independent models fit and select directly at those markers.

The HPO stage used 60 random proposals followed by 60 TPE proposals per
family. Confirmation retained five fresh repeats with three starts. Final
reporting retained ten fresh repeats with three starts. Every final repeat is
included in the reported mean and sample standard deviation (ddof=1). Training
uses sampled squared MMD. Selection uses the robust full-TVD score J on A17
for shared checkpoints, A17 union B20 for shared confirmation, and the direct
markers for independent models. The physics score P is the secondary selection
score within the J-minimum-plus-0.01 guardrail. Final J and P on F101 evaluate
the frozen shared fits and select nothing; independent final scores use only
the three markers. Scores from these different domains are not directly
comparable.

The probability-mass plots use magnetization
`m=(2k-16)/16`, where `k` is the bitstring Hamming weight. At beta=0 the shared
inputs produce the exact uniform distribution by construction; this endpoint is
architectural and does not establish learned generalization. The comparison
does not establish a pure causal symmetry effect or a thermodynamic response
law.

## Files and provenance

- `figure_data.json`: final ten-repeat curves, shells, targets and metrics used
  by the figure/table replay.
- `REPORTING_AUDIT.json`: production audit and identity checks.
- `review-ready.json`: final completion status, winners and figure-data hash.
- `provenance/`: confirmation and final campaign manifests, result summaries,
  confirmation winners and review-ready records.

Raw fit, checkpoint and parameter archives remain in the frozen Strix workspace
`/home/uribagi/cw-parity-hpo-followup-20261009/results/campaign/` and are not
bundled in the repository. The portable data SHA256 is
`fa7b0e446903d2d1892049e567a500b6b4532cfbe741afcf7ab623c5ceeb3974`.

## Replay

From the repository root, replay the current figures without training:

```sh
uv run python run_experiments.py --mode figures
uv run python run_experiments.py --mode figures --experiment curie-weiss \
  --data-root paper/data
```

Outputs are written under ignored `results/`. The historical product-input
bundle remains available through `--experiment curie-weiss-historical` but is
not part of the current manuscript evidence.
