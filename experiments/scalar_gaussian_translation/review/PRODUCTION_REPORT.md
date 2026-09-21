# Scalar Gaussian translation production report

## Decision

The authorized n=8 production campaign completed successfully. Both general
affine encoders pass all 28 frozen qualification checks on ten predeclared
optimizer seeds. The broadcast and qubit-dependent maps have nearly identical
aggregate accuracy, so the production evidence does not support a meaningful
performance distinction between them.

The anti-block encoder has much lower error, but it is a symmetry-informed
control rather than a general encoder. Its result cannot support a claim that
the learned shared IQP unitary discovered the complement symmetry.

## Execution and integrity

The campaign ran the frozen 3,120-job matrix at paper revision
`2d9df5608d7f11730440484eab75f11c48c613b2` and library revision
`40fb93788fcbc7f01083a4e1f7bc03a4c1658490`.

- Strix host 0 ran shard 0 with 1,560 jobs in 20 minutes 45 seconds.
- Strix host 1 ran shard 1 with 1,560 jobs in 20 minutes 29 seconds.
- Every job completed two optimization restarts and wrote the five required run
  artifacts plus its job record.
- The union has 3,120 unique job IDs and 3,120 unique run IDs. It has no missing,
  unexpected, or overlapping jobs.
- All JSON numbers and NumPy arrays are finite. Every run records the authorized
  paper and library revisions, n=8, and zero ancillas.
- Checkpoint regeneration produced 73,380 long-form rows. Their canonical CSV
  SHA-256 is
  `ba08d91dfa6484dac497810125b302c13a77f5a39e2e3630dca3914f1f2c92a4`.

The content-addressed production attestation has content digest
`fe5dbc75d0a81170f8a0d5eb02950a8f2ca6e89b01e2d228bbf9d2457d08e811`.
It binds the authorization, manifest, frozen gates, all job records and run
artifacts, regenerated rows, summaries, and production figures.

## Frozen gate results

The values below are the frozen evaluator's median seed means. The last column
is the worst individual far-extrapolation condition.

| Encoder | Training | Held interpolation | Near extrapolation | Far extrapolation | Worst far point |
|---|---:|---:|---:|---:|---:|
| Broadcast affine | 0.267 | 0.281 | 0.271 | 0.343 | 0.801 |
| Qubit affine | 0.262 | 0.283 | 0.267 | 0.338 | 0.810 |
| Anti-block control | 0.080 | 0.110 | 0.116 | 0.128 | 0.358 |

Both affine encoders pass all 28 checks. The anti-block control passes all 12
descriptive checks, but the protocol does not assign it a general-encoder
qualification decision.

The general-encoder production medians reproduce the audit results closely.
Most changes are below 0.016 TVD. The main exception is the qubit-affine worst
far point, which rose from 0.687 in audit to 0.810 in production and remains
just below the frozen 0.82 cap.

## Interpretation

The shared IQP unitary makes a large learned contribution for both general
encoders. Median improvement over the corresponding encoder-only distribution
is 0.499 for broadcast affine and 0.501 for qubit affine. Both models preserve
the direction of target-mean movement for every evaluated configuration. Their
median movement ratios are 0.626 and 0.653.

Equal-configuration means differ by at most 0.0046 TVD between the two general
encoders across training, interpolation, near extrapolation, and far
extrapolation. The qubit-dependent affine map therefore provides no material
advantage over broadcast affine in this benchmark.

Matched exposure changes the held-out mean by at most 0.0098 TVD across the
three encoders and protocols. The interpolation and extrapolation differences
cannot be explained by update exposure alone.

The condition-blind model is substantially worse. Its equal-configuration
median TVD is 0.447 for held interpolation, 0.617 for near extrapolation, and
0.701 for far extrapolation. The corresponding broadcast values are 0.276,
0.272, and 0.331. The qubit-affine values are 0.279, 0.268, and 0.336.

## Anti-block fixed point

At `c=0.5`, every anti-block angle is `pi/2`, so the input is exactly
`|+>^8`. The model's first Hadamard layer maps this state to `|0>^8`; the
diagonal IQP layer can then add only a global phase, and the final Hadamards
return the uniform bitstring distribution. This behavior is fixed by the
encoder and circuit structure, not learned parameters.

All 380 production training rows at this fixed point have TVD
`0.013920594816259` up to floating-point roundoff. That value equals the TVD
between the uniform bitstring distribution and the truncated target at
`c=0.5`. The anti-block's central accuracy must therefore be described as a
designed symmetry fixed point. Its useful evidence lies away from the center,
where it also preserves complement covariance to numerical precision.

## Visible failure modes and limits

- Both general encoders have their worst far point at `c=0.9765625` for random
  layout 1101 with anchors `[0, 0.4440795155, 0.5]`. Broadcast reaches 0.801
  TVD and qubit affine reaches 0.810 TVD. The latter passes the 0.82 cap by a
  narrow margin.
- The mirrored far diagnostic remains direction-sensitive for qubit affine.
  In the controlled k=3, even, fixed-total slice, its reflected-side mean is
  0.051 TVD higher than the original far side. Broadcast differs by 0.007 and
  anti-block by 0.008.
- Exact TVD and checkpoint selection are simulation diagnostics available at
  n=8. They do not establish scalable exact evaluation.
- The production campaign does not establish quantum advantage, superiority
  over classical methods, hardware performance, or behavior at n=12.
- Independent unconditional IQP fits remain audit references. Production did
  not rerun them because the authorized replication matrix targeted the shared
  conditional models and condition-blind control.

## Artifacts

- [Production authorization](production_authorization.json)
- [Production attestation](production_attestation.json)
- [Frozen gate result](production_gate_check.json)
- [Production aggregates](production_aggregates.json)
- [Production review summaries](production_review_summaries.json)
- [Long-form rows](../../../results/scalar_gaussian_translation_production/review/production_rows.csv)
- [Production figures](../../../results/scalar_gaussian_translation_production/review/figures)

The copied raw records and checkpoints are under
`results/scalar_gaussian_translation_production/{host0,host1}`. The
`combined` directory is a symlink index over the two verified shards.

## Reproduction

Run the post-run verifier from the repository root:

```bash
uv run python -m experiments.scalar_gaussian_translation.production_attestation \
  --manifest experiments/scalar_gaussian_translation/manifests/production_proposal.json \
  --authorization experiments/scalar_gaussian_translation/review/production_authorization.json \
  --records-dir results/scalar_gaussian_translation_production/combined/job_records \
  --gates experiments/scalar_gaussian_translation/review/frozen_gates.json \
  --rows results/scalar_gaussian_translation_production/review/production_rows.csv \
  --aggregates experiments/scalar_gaussian_translation/review/production_aggregates.json \
  --gate-result experiments/scalar_gaussian_translation/review/production_gate_check.json \
  --summaries experiments/scalar_gaussian_translation/review/production_review_summaries.json \
  --direct-rows experiments/scalar_gaussian_translation/review/direct_audit_rows.csv \
  --figure results/scalar_gaussian_translation_production/review/figures/production_scalar_profiles.pdf \
  --figure results/scalar_gaussian_translation_production/review/figures/production_scalar_profiles.png \
  --figure results/scalar_gaussian_translation_production/review/figures/production_scalar_tvd_by_condition.pdf \
  --figure results/scalar_gaussian_translation_production/review/figures/production_scalar_tvd_by_condition.png \
  --figure results/scalar_gaussian_translation_production/review/figures/production_scalar_aggregate.pdf \
  --figure results/scalar_gaussian_translation_production/review/figures/production_scalar_aggregate.png \
  --output experiments/scalar_gaussian_translation/review/production_attestation.json
```
