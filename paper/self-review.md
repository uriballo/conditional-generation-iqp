# Adversarial self-review

This review applies to the local coauthor draft of *Conditional Generation
with Classically Trainable IQP Circuits*. The 1,020-fit Gaussian campaign and
80-fit Curie--Weiss v3 campaign are complete and must not be rerun. No
repository action submits or uploads the manuscript.

## Executive verdict

The contribution is coherent and appropriately narrow: input-state
conditioning adds conditional generation to an IQP model while preserving the
observable estimator used for classical MMD training. Product inputs support
the general second-moment result; the Curie--Weiss study uses a tractable
spin-flip two-branch input. The draft does not claim the first conditional
QCBM, universality, quantum advantage, thermal-state preparation, or optimizer
guarantees.

## Five-dimension review

| Dimension | Status | Reviewer-facing check |
|---|---|---|
| Contribution | Pass | The Introduction distinguishes the formulation from prior conditional QCBMs and from the underlying trainable IQP construction. |
| Writing clarity | Pass after proof | The main text has six sections and three figures; terminology is stable, and the preparation pipeline is not mislabeled as a strict IQP circuit. |
| Experimental strength | Pass with bounded scope | Gaussian v1 and Curie--Weiss v3 are complete. The v2 capacity comparison and two symmetry screens are supplementary diagnostics. |
| Evaluation completeness | Pass for the core campaigns | Profile and raw TVD, training and held-out conditions, paired seeds, and independent references are retained. |
| Method soundness | Pass with limitations | The estimator result bounds the second moment. The evidence does not isolate ancilla causality or guarantee optimization. |

## Claim--evidence audit

| Claim | Evidence | Status |
|---|---|---|
| Product inputs preserve the declared classical training estimator. | Derivation plus finite estimator-consistency audit. | Supported; no optimization guarantee. |
| Gaussian conditioning improves over no conditioning. | Twelve size--width cells, ten paired seeds, exact evaluation. | Supported in all 12 cells. |
| One shared Gaussian circuit matches independent circuits. | Training-condition reference comparison. | Unsupported; the fidelity loss is reported. |
| Denser condition coverage is more useful than more samples at existing conditions. | Matched fixed-per-condition and fixed-total tracks. | Supported descriptively; gains flatten after roughly five conditions. |
| Curie--Weiss v3 conditioning tracks temperature response. | Frozen 80-fit v3 report and active v3 claim registry. | Supported for the finite eight-visible-spin system. |
| Curie--Weiss v3 improves over no conditioning and has no resolved reference gap. | Paired v3 profile/raw effects and condition-specific references. | Supported for the preregistered aggregate comparisons; not a general equivalence claim. |
| Ancillas cause the v3 improvement. | No v3 matched-capacity control. | Unsupported and prohibited. |
| The v2 91-versus-92-phase comparison controls v3 capacity. | The diagnostic uses the v2 product input, not the v3 spin-flip input. | False; the result is labeled historical. |
| Symmetry matching alone explains the exploratory gains. | Two post-core screens; radial arms receive unequal condition information. | Unsupported; screens are feasibility evidence only. |

## Reproducibility and provenance

- Gaussian execution source is sealed under
  `results/gaussian-foundations-v1/evidence/execution-source-files` and checked
  against recorded hashes.
- Curie--Weiss v3 uses the spin-flip input and calibrated knot map selected in
  development screens and frozen before production.
- Active generated claims match the v3 campaign report; the v2 capacity
  summary remains separately identified.
- Exhaustive per-seed and per-condition outputs remain machine-readable.
- Final handoff requires tests, lint, a clean build, checksum verification, and
  visual inspection of every PDF page.

## Remaining rejection risks

1. Input-family selection used target-specific development evidence and is not
   demonstrated to transfer without retuning.
2. Raw-distribution errors remain materially larger than profile errors.
3. Both core benchmarks are exchangeable, finite, and one-dimensional in the
   condition.
4. The radial-shell screen does not match condition information across arms.
5. Final author metadata, contribution statements, and approval remain human
   review items.
