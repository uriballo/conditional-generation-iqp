# Compact manuscript data

These bundles retain the numerical inputs, per-fit metrics and provenance needed
to replay manuscript figures and tables without training. Reviewed publication
inputs remain unchanged during code cleanup. Full checkpoints, target pools and
frozen training sources remain in ignored `results/` and `.local_research/`.

| Bundle | Manuscript use | Aggregation and limitations |
| --- | --- | --- |
| `gaussian_lattice/` | Current translation and two-coordinate Gaussian figures, diagnostics and parameters | Five final fits per selected configuration; 40 independent 4x4 lattice parameters and an exact binomial centre at the origin. Full TVD is the selection metric. |
| `gaussian/` | Historical translation profiles, response and TVD table | Five fits; median of seed-wise mean errors over eighty held-out conditions. Development profiles used for selection. Full TVD also detects within-shell errors. |
| `gaussian2d_phase/` | Historical one-angle heatmaps, responses, controls and finite-validation table | Five selected fits per design in initial plots; twenty paired finite-validation fits in the supplement. Prior development examined reporting conditions. Exact-model replay does not establish a need for Gaussian preparation or iterative learning. |
| `scaling/` | Parity-encoder lattice runs, 64–2048 qubits | Median/IQR across seed-level bank averages. Retain negative unbiased MMD estimates. Size-dependent kernels prevent direct accuracy comparisons across sizes. |
| `switching16/` | Signed-graph encoder illustrations and reference-state access material | The diagrams and construction are current manuscript material. Frozen empirical fits, responses and fluctuation diagnostics are retained for historical replay only and are excluded from default figure generation. |
| `curie-weiss-followup/` | Audited Curie–Weiss parity/product comparison and current figures/tables | Ten fresh final repeats with three starts per selected configuration; five fresh confirmation repeats are retained in provenance. |

## Current Curie–Weiss comparison

The audited four-family parity/product follow-up is frozen in
`paper_experiments/cw_followup_protocol.json`. The compact current bundle is
`curie-weiss-followup/`; its full-precision configurations, final curves and
provenance are retained there. Raw fit, checkpoint and parameter archives are
preserved in the frozen Strix workspace and are not bundled here.

Final reporting uses ten fresh repeats and three starts, retaining every
repeat and reporting means with sample standard deviation (ddof=1). Shared
encoders train at anchors [0.1, 0.7, 1.3, 1.95]. Checkpoint selection uses A17;
shared confirmation ranks on A17 union B20. F101 is used only for final
reporting and does not select a checkpoint or configuration. The marker values
beta=0.4, 1, and 2 are excluded from shared fitting and selection. Independent
models fit and select directly at their marker. The primary metric is full
total variation distance (TVD), with the physics score P as a secondary
selection score within the J+0.01 guardrail. For probability-mass
figures, magnetization is m=(2k-16)/16, where k is the bitstring Hamming
weight. At beta=0 the shared inputs produce the exact uniform distribution by
construction; this endpoint is architectural and is not evidence of learned
generalization. The comparison does not establish a pure causal symmetry effect
or a thermodynamic response law.

The earlier `curie-weiss/figure_data.json`, `hpo-summary.json` and response sweep
retain the historical comparison and checkpoint provenance. Render them explicitly
with `--experiment curie-weiss-historical`; default figure replay excludes them.
The response sweep includes fitting and development temperatures and reuses
confirmation seeds. Its energy-fluctuation statistic does not verify a thermal
response derivative for learned distributions.

## Replay

From the repository root:

```sh
uv run python run_experiments.py --mode figures
uv run python run_experiments.py --mode figures --experiment curie-weiss-historical
```

Outputs stay under ignored results. `gaussian2d_phase/README.md` documents the
separate control and finite-validation replays. `switching16/README.md` documents
its current encoder illustrations and the historical response replay.

Superseded amplitude-conditioned Gaussian, projected-product Curie–Weiss and
three-qubit symmetry bundles have been archived outside the tracked tree.
