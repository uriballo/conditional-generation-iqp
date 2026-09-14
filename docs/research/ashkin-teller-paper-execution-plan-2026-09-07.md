# Execution plan: symmetry-only and bond-informed conditional IQP

2026-09-07. Proposed staged work following the
[review plan](ashkin-teller-aware-unaware-review-plan-2026-09-07.md).
This document plans the next work; it is not an executable protocol freeze or
an authorization to interpret existing development fits as confirmation.
No experiments were launched while writing it.

## Scope to hold fixed

Primary comparison: two-block cat (symmetry-only) versus bond-informed input,
both with the same complete two-local IQP bank. At eight visible qubits this
has 30 independent phases after symmetry-sector deduplication. Keep no ancillas
initially, fixed kappa=.3, the cat map alpha(c)=.225*pi-.20*pi*(c-.1)/.6, and
bond g(c)=c with eta=pi/8. One learned unitary remains shared across conditions.

Aligned four-body gates remain a separate architecture ablation. Do not put
only the informed encoder on the richer circuit in the primary comparison.
The hypothesis concerns conditional generation with two different levels of
supplied information. Classical predictive superiority is not required.

## 1. Make reference access a reusable, reviewable implementation

The temporary prototype was deleted after capture; integrate a deliberately
reviewed implementation from the validated formulas, not a research script.
Keep changes limited to the bond reference, the sampling/ratio integration needed
by the existing trainer, and meaningful correctness tests.

Deliverables:

- Normalized post-Hadamard amplitude queries and sequential reference sampling
  via transfer contractions, with exact support/parity handling and stable
  normalization in the declared coupling/size range.
- Phase-gradient and unbiased MMD integration for fixed encoders. No learned
  encoder parameters or condition-dependent IQP phases.
- Tests against enumeration at small sizes for amplitudes, prefix laws, moments,
  gradients and normalization; sampling calibration and non-enumerating larger-size
  checks. Verify both layer flips and deduplicated phase equivalence.
- A short derivation and resource account: classical sampling/query complexity,
  numerical assumptions, and a concrete quantum input-preparation construction
  or clearly scoped preparation requirements. Classical sampling alone is not
  a demonstrated quantum preparation circuit.

Primary agent owns estimator mathematics, interfaces and final review. Bounded
implementation can be assigned to local_flash with explicit file ownership once
the implementation seam is chosen. Do not delegate theoretical decisions.

Exit: correctness checks pass and estimator/resource claims match the code.
No full manuscript-results rewrite yet.

## 2. Validate sampled training on development conditions

Use the existing 11-condition development set, including the old test corners.
All are available for fitting and tuning; none counts as fresh generalization.
Run 2 encoders x 3 paired seeds x 2 loss modes = 12 small fits: sampled MMD and paired
exact-population controls. Reconstruct or freshly freeze paired data and
initializations explicitly, so deleted pilot archives are not silently assumed
available. Reuse 1,000 training/1,000 independent validation examples per
condition, 500 updates, the previous Adam schedule, 512 masks per bandwidth and
2,048 reference samples as the starting specification.

Check whether sampled training reproduces the qualitative fitting and learned
interlayer-dependence behavior of exact training. Proposed engineering criterion:
for each encoder, median paired increase in mean TVD from exact to sampled is
at most .03, with no unexplained estimator failures. Also report encoder-only
joint-TVD gains, mean and worst correlation errors, and every condition's change.
A positive average does not justify concealing a condition that worsens.

The old worst-correlation <=.10 gate was an internal development decision, not
a requirement of the paper's contribution. Preserve its recorded failures.
This new stage explicitly assesses sampled-training fidelity; it must not be
represented as retroactively passing the old gate. Freeze the new criterion
before these runs, and treat it as an engineering tolerance, not significance.

If the sampled/exact gap fails, diagnose it on this development set. Allow one
predeclared reference-sample increase (2,048 to 4,096), applied to both encoders,
with the same paired controls and validation rules. Do not simultaneously change
map, circuit, loss, data and optimizer. If it remains unresolved, stop expansion
and report that sampled optimization is not established at the tested budget.
Any further architecture/loss research becomes a separate development decision.

Exit: both primary encoders have a technically sound sampled training path.
No requirement that the informed encoder wins, or that classical models lose.

## 3. Freeze and run one confirmatory conditional experiment

Proposed n=8 split, preserving the existing [.1,.7] domain and encoder maps:

L = {.10,.22,.38,.54,.70}.
Train at the 16 perimeter points of L x L. Test the 9 interior combinations
{.22,.38,.54} squared. Every individual test coordinate is seen on the perimeter;
all test combinations are new and inside the training convex hull. Exchanging
layers sends tests to tests, not to training points. The target laws at these
nine test pairs have not been evaluated in the work leading to this plan.

This is compositional interpolation, not extrapolation or a claim that the task
is intrinsically difficult. Some perimeter conditions were previously inspected;
the method/design was developed on related conditions. Disclose this history.
Keep the former extrapolation result as development/limitation evidence.

After stage 2, freeze architecture, maps, all training/validation choices,
classical baseline definitions, checkpoint rules, seeds and condition roles in
an immutable manifest. Use 10 paired seeds: 20 shared IQP fits for the two encoders.
Use equal per-condition data access and the settled development budget.
Validation and hyperparameter selection occur only at training conditions.
Store all selected parameters and test-coordinate predictions before evaluating
any test target. Do not optimize against those test scores afterward.

Mandatory outputs:

- Exact joint TVD at all 9 test points, mean/worst errors and paired seed summaries.
- Connected cross-layer bond errors and both layer-marginal errors.
- Encoder-only outputs for both families at the same test inputs.
- Data-matched symmetry-aware log-composition and pooled log-linear classical
  baselines, with regularization selected using training-condition validation
  only. They contextualize the task and supplied assumptions, not gate inclusion.
- Paired uncertainty estimates over independent seeds; test conditions are not
  additional independent replications. Report signed differences even if intervals
  cross zero or the informed encoder does not improve.

The outcome is reported regardless of which method wins. Decisions about paper
scope use accuracy, learned contribution, reproducibility and clarity of claims;
there is no requirement to search for a favorable classical comparison.

## 4. Expand only what supports a specific claim

If a second size materially strengthens the encoder demonstration and the
implementation supports it, add n=12 as a separately identified size extension.
Audit preparation, support and numerical behavior for six-site layers first;
freeze any size-specific development choices before its test evaluation.
Do not silently pool it into the n=8 protocol or require it merely for symmetry
with the existing Gaussian/Curie–Weiss campaign.

Keep the existing four-body 2x2 study as labelled development evidence. If we
want a confirmatory circuit-effect claim, predeclare both encoder/four-body
variants before evaluating their confirmation results; do not add only the
most favorable model after seeing the primary test outcomes. A wrong-topology
control is needed only for a narrower causal claim about correct topology.
An independent quantum bank is needed only if claiming an advantage over such
a bank; it is not essential to demonstrate a shared conditional construction.

## 5. Write the paper section around what is actually shown

Main presentation: symmetry-only versus bond-informed on the same circuit,
with encoder-only reference and per-condition interpolation/correlation plots.
Explain exactly what physics the informed input supplies and what the trained
IQP phases add. Separate exact development results, sampled-training results,
confirmatory interpolation, and failed extrapolation.

Methods/appendix: encoder derivation, post-Hadamard sampling and ratio algorithms,
preparation/resource account, full information-access inventory, optional circuit
ablation, and remaining correlation errors. Update the claim-to-evidence map
before promoting numerical statements to the manuscript. Scientific author
review remains distinct from automated verification.

## Immediate next work

Implement and verify the reusable bond reference and its sampled estimator
integration, then execute the 12-fit development comparison. Keep confirmatory
test targets unopened until development and the full protocol freeze are complete.
No large campaign, new benchmark search or additional encoder sweep is needed
at this stage.

| Claim/work product | Evidence now | Next evidence | Status |
| --- | --- | --- | --- |
| Reference construction and estimator identities | Reusable transfer implementation, correctness tests and preparation derivation | Scientific author review | Automated verification complete |
| Both encoders train using sampled estimates | Verified 12-fit development comparison; both median-gap gates passed | Scientific author review | Stages 1–2 complete |
| Both encoders interpolate to new joint conditions | Sealed 20-fit confirmation at nine interior conditions | Scientific author review | Stage 3 complete |
| Additional physics improves accuracy | Paired test-TVD difference −.03946; interval [−.04175,−.03703] | Author review; do not infer topology alone caused the gain | Verified at this split |

## Execution update: stages 1–2 authorized

Stage 1 is implemented and independently checked; see [reference method](bond-reference-method.md). The regression suite passed 214 tests, and the new runner checks verified loss/gradient equivalence and immutable paired data. Stage 2 is running under `artifacts/ashkin-teller-development-2026-09-07/manifest.json`, with a retained source archive. Both loss modes select checkpoints using the same fixed sampled development-validation objective. This is an explicit refinement of the planning document, frozen before fitting. Confirmatory targets remain unopened.

Runtime refinement: the first partial fits were superseded after replacing inefficient tiny matrix products and canceling common ratio normalizers. Both implementations agree on checked loss values; the restarted source freeze retains identical paired data. The original partial artifacts remain separate and are not included in the final comparison.

Final stage-2 outcome: all 12 paired fits completed and were independently checked; final suite 218 passed. Both median TVD-gap gates pass at 2,048 reference samples (cat .00922, bond .00240), so the fallback is unnecessary. See the [full result and claim-evidence map](ashkin-teller-sampled-development-2026-09-07.md). Correlation limitations remain. Stage 3 was subsequently authorized and launched as recorded below; stages 4–5 remain unlaunched.


## Execution update: stage 3 complete

The [confirmatory protocol](ashkin-teller-confirmation-protocol-2026-09-07.md)
was frozen with source and paired-data hashes. All 20 IQP fits and ten classical
bundles completed successfully. All 60 prediction/metadata files passed the
seal before test-target evaluation; subsequent fitting is closed. The regression
suite passed 228 tests. Independent prediction and metric checks passed, and
both PDF/PNG figures were visually inspected.

See the [result, figures and claim-evidence map](ashkin-teller-confirmation-result-2026-09-07.md).
Mean test TVD: symmetry-only IQP .13271, bond-informed IQP .09325, log composition
.09881, pooled log-linear .02595. The primary paired bond-minus-cat difference
is −.03946, with percentile bootstrap 95% interval [−.04175,−.03703]. Both trained
encoders improve joint TVD and connected-correlation error over their respective
inputs at every seed/condition pair. Correlation accuracy remains uneven: the
bond encoder is worse than cat at (.54,.54), and worst condition means are about
.15 for both. This supports compositional interpolation on the declared split,
not extrapolation, uniform dependence recovery, or quantum advantage.

Stages 1–3 are complete. Scientific author review and manuscript inclusion
remain pending; no stage-4 size or architecture extension has been launched.

## Paper-study planning follow-up

The [paper study plan](ashkin-teller-paper-study-plan-2026-09-07.md) proposes the motivation, saved-output structural diagnostics, a bounded n=12 replication and evidence-based discussion following stage 3. It does not launch stage 4 or insert manuscript text.

## Follow-up study completed

The authorized n=8 structural diagnostics and separately frozen n=12 replication are complete. All 20 n=12 IQP fits and ten classical bundles were sealed before evaluation, independently verified, and reported with all conditions/seeds. Four final figures passed PDF/PNG visual review. See the [completed study evidence and discussion](ashkin-teller-paper-study-plan-2026-09-07.md#completed-twelve-qubit-evidence). Author review and manuscript integration remain pending; no additional experiment or tuning is needed to close this execution stage.
