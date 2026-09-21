# Scalar Gaussian Translation Benchmark: review report

> This is the pre-production adversarial review. The author later approved and
> completed the n=8 campaign. See the
> [production report](PRODUCTION_REPORT.md) for the final replication results.

## Decision

The n=8, zero-ancilla benchmark is **scientifically positive after a post-audit evaluator correction**. A threshold-preserving re-evaluation of the saved checkpoints shows that both general affine RY encoders pass all 28 frozen locked-audit checks. The evidence supports a narrow proof-of-concept claim: one shared IQP unitary learns a substantial correction to predeclared affine product encoders that do not consume target statistics at runtime, and it retains controlled accuracy at unseen conditions.

The recommendation is **prepare the larger n=8 campaign for explicit author authorization, but do not execute it yet**. The remaining launch prerequisites are a clean source commit, a byte-for-byte source-equivalence bridge to the original audit manifest (or a clean audit rerun if equivalence fails), and regeneration of a launch-valid attestation and production manifest. This package does not authorize n=12. It does not support quantum advantage, general classical superiority, or a broad optimization-landscape claim.

## Scope and frozen design

The target translates a discrete Gaussian along Hamming weight on eight visible qubits. Every condition uses the same complete, untied one- and two-body IQP unitary with 36 parameters and no ancillas. The controlled comparison changes only the input encoder: broadcast affine RY, qubit-dependent affine RY, or the symmetry-informed anti-block RY construction. At the midpoint, anti-block assigns every qubit an angle of \(\pi/2\), preparing \(|+\rangle^{\otimes 8}\) before the shared IQP unitary. Away from the midpoint, paired angles \(\pi c\) and \(\pi(1-c)\) preserve one expected excitation per pair and transform by global bit complementation under \(c\mapsto1-c\). Its midpoint and complement-covariant behavior are intentional controls, not failed learning.

The affine coefficients were frozen before validation and audit, and neither affine encoder reads target moments at runtime. The tracked design record does not establish that the coefficients were derived independently of the target family. The review therefore calls them **predeclared runtime-target-agnostic affine encoders**, not target-independent encoders.

The two general encoders passed the predeclared development gate, so the exponential fallback was not invoked. The bounded eight-candidate search selected one common configuration for both encoders: 600 updates, two restarts, learning rate 0.03 with a tenfold decay, and Hamming-RBF bodyness targets 2 and 4. The [selected configuration](selected_config.json) records every candidate and its rejection reason.

The evidence comprises 2,298 fit jobs: 6 Stage-1 jobs, 48 bounded-search jobs, 936 validation jobs, 183 validation direct references, 936 locked-audit jobs, and 189 audit direct references. The controlled validation and audit blocks cross three encoders, three main protocols, four anchor counts, even plus three fixed random layouts, two budget modes, and three optimizer seeds. Mirrored near and far protocols provide a smaller directional diagnostic.

## Locked-audit result

The audit manifest was bound to gate hash `571246e7143cfc1c0b5cec19a512106d2670c8d45fdd338f95e46347d8214130` before audit execution. The table reports median seed-mean TVD, worst seed-mean TVD, and the worst individual condition over the complete controlled audit. Anti-block is comparative and does not determine whether a general encoder qualifies.

| Regime | Broadcast affine | Qubit affine | Anti-block |
| --- | ---: | ---: | ---: |
| Training | 0.264 / 0.267 / 0.474 | 0.263 / 0.265 / 0.465 | 0.076 / 0.081 / 0.183 |
| Held interpolation | 0.280 / 0.284 / 0.536 | 0.283 / 0.284 / 0.530 | 0.113 / 0.122 / 0.342 |
| Near extrapolation | 0.277 / 0.279 / 0.447 | 0.274 / 0.278 / 0.434 | 0.121 / 0.123 / 0.219 |
| Far extrapolation | 0.342 / 0.357 / 0.791 | 0.353 / 0.384 / 0.687 | 0.121 / 0.127 / 0.312 |

The first evaluator incorrectly allowed mirrored training rows into qualification summaries and pooled random layouts before taking their median. The corrected evaluator excludes every mirrored protocol and compares the maximum of the three layout-seed medians with the unchanged frozen caps. No threshold, checkpoint, or training result changed. Both general encoders pass all 28 checks in [the corrected gate result](audit_gate_check.json); [the complete observed-value-versus-threshold table](GATE_TABLE.md) records every check. Their corrected median grouped improvements over their own encoder-only distributions are 0.519 and 0.510 TVD; the worst category-seed improvements remain 0.375 and 0.374. Their median generated-to-target mean-movement ratios are 0.626 and 0.641, and every evaluated configuration moves in the correct direction. Neither general map shows a meaningful advantage over the other.

Accuracy degrades under far extrapolation, as required for an informative stress test. The worst individual far point reached 0.791 TVD for broadcast affine and 0.687 for qubit affine. The condition curves show the controlled \(k=3\), even, fixed-total slice; the global worst points remain visible in the encoder-by-protocol aggregate scatter.

## Baseline context

The general encoders alone are poor approximations to the target: their audit-grid mean TVDs are 0.773 and 0.780. The learned shared unitary therefore supplies a large, reproducible correction. The condition-blind shared model is also weaker, with median seed-mean TVDs of 0.440 for held interpolation, 0.578 for near extrapolation, and 0.709 for far extrapolation.

Independent per-condition IQP fits achieve mean audit TVD 0.207, median 0.232, and worst 0.293. They see target data at every evaluated condition and therefore serve as capacity references, not deployable conditional methods. Exact convex interpolation between specified target-anchor distributions is also strong: with three evenly spaced anchors, its mean TVD is 0.161 for full-interval interpolation, 0.095 inside the near interval, and 0.186 beyond the near boundary. Under far extrapolation, endpoint holding increases its mean TVD to 0.404. The shared general encoders need not beat these references, and the results do not establish that they do.

The anti-block construction is substantially more accurate than either general encoder across all main regimes. Because it also changes the midpoint state, excitation allocation, and complement covariance, this controlled result supports only the empirical statement that the **complete anti-block construction** performs better in this benchmark. It does not isolate complement symmetry as the cause or establish a general encoder-design principle.

[The reproducible review summaries](review_summaries.json) add the missing baseline strata. Exact even-anchor interpolation improves monotonically inside each training interval as the anchor count increases: its mean held TVD for \(k=2,3,4,5\) is 0.402/0.161/0.073/0.046 on the full interval, 0.291/0.095/0.049/0.026 on the near interval, and 0.161/0.046/0.021/0.012 on the far interval. Endpoint holding is independent of interior anchor count and gives mean TVD 0.186 beyond the near boundary and 0.404 beyond the far boundary. The same artifact reports every audit condition for this baseline, condition-resolved direct-fit summaries, and condition-blind results by protocol, anchor count, and category.

## Directional, anchor, and budget diagnostics

The mirrored diagnostic reveals a limitation that the qualification gates intentionally exclude. Mean mirror-near TVD is 0.233 for broadcast affine and 0.239 for qubit affine, but mean mirror-far TVD rises to 0.478 and 0.432. Anti-block remains nearly symmetric at 0.111 and 0.096. The directional asymmetry is reproducible; its cause is unresolved, so attributing it to encoder inductive bias remains a hypothesis rather than a result.

Increasing anchor count helps held interpolation, while near and far behavior is not monotone. Fixed-total and matched-exposure held-out interpolation medians differ by at most 0.0043 across the three encoders. The observed protocol differences therefore cannot be attributed solely to unequal updates per condition. The revised aggregate figure facets every anchor-count and budget comparison by encoder and protocol, so anti-block's lower error scale cannot conceal the general-encoder behavior.

The frozen gate estimand first averages raw condition rows within each optimizer seed, so configurations with more training anchors and protocols with more grid points receive more weight. As a sensitivity check, the review now averages conditions within each complete configuration and then weights configurations equally. The equal-configuration mean TVDs remain consistent with the main conclusion:

| Regime | Broadcast affine | Qubit affine | Anti-block |
| --- | ---: | ---: | ---: |
| Training | 0.270 | 0.268 | 0.076 |
| Held interpolation | 0.271 | 0.275 | 0.113 |
| Near extrapolation | 0.277 | 0.274 | 0.116 |
| Far extrapolation | 0.327 | 0.346 | 0.122 |

Complement-covariance error is now reported explicitly over every saved pair \((c,1-c)\) present within a checkpoint. The median errors are 0.206 for broadcast affine, 0.217 for qubit affine, \(2.2\times10^{-16}\) for anti-block, and 0.419 for condition-blind. The contributing protocols, categories, condition pairs, and stratified summaries are recorded in [the review summary artifact](review_summaries.json).

## Optimization evidence

The optimization diagnostic uses the controlled three-anchor, even, interpolation configuration and all six saved restarts per encoder. Its predeclared 0.32 success threshold equals the frozen held-interpolation median cap. Both general encoders succeed in 6/6 restarts; anti-block succeeds in 5/6 despite a much lower median held-grid TVD of 0.205. Matched-initialization median gradient norms are 0.059 for broadcast affine, 0.063 for qubit affine, and 0.048 for anti-block.

Anti-block reaches better selected checkpoints but does not dominate every reliability statistic. The results therefore support a benchmark-specific fidelity benefit for the complete anti-block construction, not a claim that its complement symmetry is the cause or that symmetry globally simplifies the optimization landscape. Time-to-threshold values in [the diagnostics](optimization_diagnostics.json) are linear estimates from total restart wall time because the runner does not save per-step timestamps.

## Finite-support qualification

The nominal target mean equals the actual truncated mean to numerical precision throughout the domain. Finite support contracts the width: the exponent width is 1.414, while the actual standard deviation decreases from 1.404 at the midpoint to 1.073 at either endpoint. Boundary-shell mass rises from 0.010 at the midpoint to 0.306 at the endpoints. Numerically, mean displacement is 4.46 and 8.46 times the standard-deviation contraction over the near and far segments, respectively. This ratio does not show that truncation is unimportant to distributional difficulty; width metrics must use the actual truncated target moments.

## Provenance and incident record

- Protocol SHA-256: `0093ef159236793ef10fa0a22646a5febaab115645cf3b408a23352e6f2f6ebc`
- Runtime snapshot SHA-256: `f11cfe51989367c699bbeaa7a6989b2254534a06b8ee09b692514c47bfaba4da`
- Paper repository revision: `f33ee7ae51948d5b3b58c74e869221fbcdf8bb71`
- Library revision: `ce6cd382e9fe62550ee18f1a9daeebf9d00f1eb4`
- Frozen audit gate SHA-256: `571246e7143cfc1c0b5cec19a512106d2670c8d45fdd338f95e46347d8214130`

The first validation launch exposed two provenance defects before its results were admitted: the source snapshot omitted an uncommitted library runtime change, and run IDs did not distinguish anchor count or random layout. The campaign now hashes every library runtime module and assigns a unique training-set label to every anchor configuration. All pre-fix outputs were quarantined in `results/scalar_gaussian_translation-pre-anchor-identity-fix-20260921-1740`; Stage 1 and Stage 2 then reproduced their decisions, and validation and audit ran from a clean namespace. A clean source commit remains necessary before production.

The adversarial review exposed two post-processing defects: mirrored rows entered qualification summaries, and the random-layout check pooled layouts. The repaired evaluator has regression tests with poisoned mirror rows and a deliberately bad layout. [The audit attestation](audit_attestation.json) binds the evaluator, manifest, frozen gates, long-form rows, gate result, all 936 job records, and all 936 checkpoints. It deterministically regenerates all 22,014 rows from those checkpoints, requires byte-identical canonical CSV, recomputes the gate result, and requires exactly 28 native-true checks for each general encoder. Its `launch_valid` field remains false while the paper worktree is dirty.

The original audit manifest recorded the pre-commit library revision even though its complete runtime-file map hashes the audited working-tree bytes. Committing those same bytes necessarily changes the library revision and therefore the snapshot digest. [The reviewed provenance root](audit_provenance_root.json) fixes the exact original manifest, frozen-gate, protocol, and source-snapshot hashes. The review code now provides a fail-closed source-equivalence bridge: from a clean commit it validates that root, compares the complete current runtime-file map against the untouched audit manifest, records both provenance states, and refuses any coordinated manifest, protocol, or source change. Production requires the author-approved root, that bridge, and a regenerated launch-valid attestation, or else a full audit rerun from a new clean manifest. The bridge has not been generated in the current dirty worktree.

## Unresolved risks

1. General-encoder absolute fidelity is moderate, and isolated far-extrapolation failures remain large.
2. The reflected far protocol exposes substantial directional bias in both general encoders.
3. Anti-block is much stronger, so the main positive result depends on accepting a sizable gap between the predeclared affine encoders and a structure-informed construction.
4. The classical specified-anchor and direct IQP references outperform the general shared model in several regimes; this benchmark is explanatory, not competitive.
5. Exact TVD and full distributions are available only because n=8 is simulated. The study does not test large-register measurement cost or sampling-only model selection.
6. Three optimizer seeds and three random layouts distinguish the main variance sources but do not support fine statistical claims.
7. Finite support changes the actual target width near the boundaries, even though it preserves the intended mean translation.
8. The saved training snapshot includes an uncommitted library API extension. Production must start from a clean, reviewed commit plus a verified source-equivalence bridge, or from a new clean audit if byte equivalence fails.
9. n=12, ancillary qubits, and circuit-depth selection remain outside this review. The disabled 3,120-job n=8 proposal is addressed only by the separate launch plan.

## Reproduction

From the repository root:

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run ruff format --check experiments/scalar_gaussian_translation \
  experiments/tests/test_scalar_gaussian_translation_*.py

# Re-run a complete manifest locally. Use the corresponding stage1, stage2,
# validation, direct_validation, audit, or direct_audit manifest.
uv run python -m experiments.scalar_gaussian_translation.campaign \
  --manifest experiments/scalar_gaussian_translation/manifests/audit.json \
  run-shard 0 1 --workers 1

# Two-host Strix execution of the same frozen manifest.
bash experiments/scalar_gaussian_translation/launch_strix.sh \
  experiments/scalar_gaussian_translation/manifests/audit.json 0 2 8
bash experiments/scalar_gaussian_translation/launch_strix.sh \
  experiments/scalar_gaussian_translation/manifests/audit.json 1 2 8

uv run python -m experiments.scalar_gaussian_translation.analysis extract \
  --manifest experiments/scalar_gaussian_translation/manifests/audit.json \
  --records-dir results/scalar_gaussian_translation/job_records \
  --output experiments/scalar_gaussian_translation/review/audit_rows.csv
uv run python -m experiments.scalar_gaussian_translation.analysis evaluate-gates \
  --manifest experiments/scalar_gaussian_translation/manifests/audit.json \
  --records-dir results/scalar_gaussian_translation/job_records \
  --gates experiments/scalar_gaussian_translation/review/frozen_gates.json \
  --output experiments/scalar_gaussian_translation/review/audit_gate_check.json
uv run python -m experiments.scalar_gaussian_translation.review_summaries \
  --audit-rows experiments/scalar_gaussian_translation/review/audit_rows.csv \
  --direct-rows experiments/scalar_gaussian_translation/review/direct_audit_rows.csv \
  --output experiments/scalar_gaussian_translation/review/review_summaries.json
uv run python -m experiments.scalar_gaussian_translation.analysis attest-audit \
  --manifest experiments/scalar_gaussian_translation/manifests/audit.json \
  --records-dir results/scalar_gaussian_translation/job_records \
  --provenance-root experiments/scalar_gaussian_translation/review/audit_provenance_root.json \
  --gates experiments/scalar_gaussian_translation/review/frozen_gates.json \
  --rows-csv experiments/scalar_gaussian_translation/review/audit_rows.csv \
  --gate-result experiments/scalar_gaussian_translation/review/audit_gate_check.json \
  --output experiments/scalar_gaussian_translation/review/audit_attestation.json
uv run python -m experiments.scalar_gaussian_translation.diagnostics \
  experiments/scalar_gaussian_translation/manifests/audit.json \
  --records-dir results/scalar_gaussian_translation/job_records \
  --success-threshold 0.32 \
  --output experiments/scalar_gaussian_translation/review/optimization_diagnostics.json
uv run python -m experiments.scalar_gaussian_translation.figures \
  --validation-manifest experiments/scalar_gaussian_translation/manifests/validation.json \
  --audit-manifest experiments/scalar_gaussian_translation/manifests/audit.json \
  --records-dir results/scalar_gaussian_translation/job_records \
  --rows-csv experiments/scalar_gaussian_translation/review/audit_rows.csv \
  --diagnostics-json experiments/scalar_gaussian_translation/review/optimization_diagnostics.json \
  --hyperparameter-id stage2-04 --split audit \
  --output experiments/scalar_gaussian_translation/review/figures
```

The commands above reproduce the present provisional artifacts while the audited runtime snapshot still matches directly. After the reviewed source is committed, use the source-equivalence workflow in [the production launch plan](../PRODUCTION_LAUNCH_PLAN.md); do not rewrite the old manifest around the saved records. The existing production proposal at `manifests/production_proposal.json` is populated but intentionally disabled, and the campaign runner rejects it by construction. The launch plan specifies the clean handoff, authorization, canary, two-host execution, and post-run verification gates without enabling the proposal.

## Reviewer self-check

- **Contribution:** pass for a narrow conditional-IQP proof of concept; no competitive or advantage claim.
- **Writing clarity:** pass after narrowing affine provenance, anti-block attribution, and truncation language.
- **Experimental strength:** pass for frozen multi-seed audit evidence; needs revision if the paper implies uniformly low error.
- **Evaluation completeness:** pass for the requested n=8 review package after adding equal-configuration sensitivity, complement covariance, and stratified baseline artifacts; larger-register and sampling-only claims still need new experiments.
- **Method soundness:** pass within the simulated n=8 scope after the evaluator correction; mirrored-far asymmetry remains unresolved, and a clean launch-valid attestation is required before production.
