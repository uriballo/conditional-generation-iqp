# Adversarial closure review of protocol v0.2

> **Initial verdict:** approve with required changes
>
> **Final verification:** pass; no implementation blocker remains
>
> **Scope:** protocol closure only; no trained-model experiments

The closure review reassessed the redesigned benchmark against the blocking
findings in the v0.1 review. It found that the revised target, full-cross split,
encoder hierarchy, controls, estimands, and failure logic resolved the original
scientific objections. It did not approve the centered encoder for headline
use.

## Resolved scientific blockers

- The finite target search now contains feasible candidates and selects one by
  intrinsic target criteria rather than model behavior.
- The nine-point full cross supplies a shared midpoint and makes training-axis
  fit mandatory before held-out combinations can support a claim.
- The centered construction is a reduced-budget structural negative control.
  It cannot select a schedule, pass a headline gate, or become the headline
  model.
- Aligned and rotated general encoders have their own missing-channel
  ablations, and the rotated comparison discloses its unmatched product-state
  geometry.
- The classical reference is a fully specified multiplicative shell
  composition and is contextual rather than an advantage baseline.
- The claim is restricted to combination generalization under a supplied,
  family-aware representation.

## Required closure changes

The reviewer required the final protocol to make the following details
machine-evaluable before implementation:

1. State which gates apply to general encoders, ablations, the blind arm, and
   the centered negative control, including the consequence of an invalid
   control.
2. Freeze condition aggregation, response fits, quantiles, bootstrap
   resampling, failure scores, and the variance decomposition.
3. Separate screen, qualification, audit, diagnosis, and transfer seed blocks;
   define RNG algorithms and deterministic stream derivations.
4. Make the failure-only direct-fit and all-condition diagnosis quantitative
   and prohibit it from rescuing a failed protocol.
5. Define numerical invalidity mechanically rather than labeling finite poor
   fits as divergent.
6. Apply the complete Section 9 gate set, including absolute TVD, to the
   contingent $n=12$ strict-transfer study.
7. Freeze training constants and environment provenance before any protocol
   smoke or scientific run.

Protocol v0.2.1 incorporates these changes. A final bounded verification is
limited to checking this closure list; it is not an additional opportunity to
tune the scientific design.

## First verification

The first bounded verification found no new scientific defect and retained the
design, but it blocked implementation on three missing details:

- a sampled-test operator-mask stream and operator budget;
- a checkpoint rule and invalid-run treatment for the all-condition capacity
  diagnosis; and
- an explicit grid, split, ordering, and checkpoint rule for $n=12$ transfer.

Protocol v0.2.1 now specifies each item. It also writes the discrete-Gaussian
mass function and deterministic nested calibration procedure explicitly; this
clarification preserves the target construction used in the target-only
preflight.

## Final verification

The reviewer verified the corrections without running trained-model
experiments. The sampled-test RNG and budget, diagnostic checkpointing and
failure scoring, transfer grid and selection rule, invalid-score aggregation,
validation-stream names, and target calibration are explicit and internally
consistent. The final verdict is **pass**.
