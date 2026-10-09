# Compact manuscript data

These bundles retain the numerical inputs, per-fit metrics and provenance needed
to replay manuscript figures and tables without training. Reviewed publication
inputs remain unchanged during code cleanup. Full checkpoints, target pools and
frozen training sources remain in ignored `results/` and `.local_research/`.

| Bundle | Manuscript use | Aggregation and limitations |
| --- | --- | --- |
| `gaussian/` | Translation profiles, response and TVD table | Five fits; median of seed-wise mean errors over eighty held-out conditions. Development profiles used for selection. Full TVD also detects within-shell errors. |
| `gaussian2d_phase/` | Phase-conditioned mean–width heatmaps, responses, controls and finite-validation table | Five selected fits per design in initial plots; twenty paired finite-validation fits in the supplement. Prior development examined reporting conditions. Exact-model replay does not establish a need for Gaussian preparation or iterative learning. |
| `scaling/` | Original product-input lattice runs, 64–2048 qubits | Median/IQR across seed-level bank averages. Retain negative unbiased MMD estimates. Size-dependent kernels prevent direct accuracy comparisons across sizes. |
| `switching16/` | Signed-graph switching response, encoder diagrams, report and fluctuation diagnostics | Five frozen fits. Dense response evaluated after selection; energy variance calculated within each fit. Exact symmetry does not establish a performance advantage over the classical control. |
| `curie-weiss/` | Earlier product-input evidence still cited in the transitional appendix | Five winning confirmation runs reused for reporting. Historical fitted temperatures and selection differ from the pending parity follow-up. |

## Current Curie–Weiss comparison

The manuscript describes the four-family parity/product follow-up frozen in
`paper_experiments/cw_followup_protocol.json`. HPO and later fresh final fits
must supply its numerical comparison. No old compact bundle is a substitute
for those pending results.

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
its response replay and encoder illustration sources.

Superseded amplitude-conditioned Gaussian, projected-product Curie–Weiss and
three-qubit symmetry bundles have been archived outside the tracked tree.
