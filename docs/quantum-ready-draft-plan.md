# Plan for a Quantum-ready full draft

## Outcome

Produce a mature full draft titled *Conditional Generation with Classically
Trainable IQP Circuits* whose main contribution is a finite-resource account of
conditional response in fixed IQP circuits, supported by a controlled Ising
temperature-family benchmark and a complete reproducibility package. Quantum
is the design target. External expert review occurs only after this package is
complete and may reopen the experiment plan or venue choice.

This is a mechanism-and-boundary paper. It does not claim conditional
universality, quantum advantage, superiority to classical generators,
efficient full-probability evaluation, or general optimization-landscape
trainability.

## Binding claim hierarchy

1. A frozen affine product-state Condition map gives a fixed IQP
   circuit a finite, characterizable multivariate response spectrum.
2. The Pauli observables, sampled Conditional MMD loss, and gradients used for
   training admit efficient classical estimation. This is the paper's exact
   meaning of *classically trainable*.
3. Subject to the frozen empirical gates, a fixed compiled IQP circuit retains
   a substantial fraction of reprogrammed phase-conditioning performance on a
   2D Ising temperature family.
4. Ancilla marginalization is a bounded, secondary capacity extension.

The Abstract and first-page contribution statement must explicitly say that
all Conditions share one compiled gate sequence and one set of phase values,
and must give the operational definition and exclusions for *classically
trainable*.

## Stage 0 — Freeze the protocol

Deliverables:

- A machine-readable experiment manifest containing model families, coupling
  seeds, optimizer seeds, beta grid, data-chain seeds, hyperparameter candidate
  sets, checkpoint rule, metrics, bootstrap seed, and all gate thresholds.
- A claim-to-evidence registry linking each intended Abstract and Introduction
  statement to a theorem, experiment, or declared limitation.
- A run ledger enforcing the non-transferable 400-run allocation.
- Tests that Held-out betas cannot enter tuning, feature-map selection, or
  checkpoint selection.

No model-dependent choice may be changed after its relevant gate is released.

## Stage 1 — Pass the theorem novelty gate

Develop the two separate theoretical layers:

1. Full output probabilities: derive multivariate Condition-frequency support,
   bandwidth, smoothness, and finite-resource structural limits for fixed,
   fixed affine product-state encodings. This is an expressivity
   characterization, not an efficient probability-evaluation result.
2. Estimable observables: specialize the response analysis to the Pauli
   observables used by Conditional MMD and prove unbiased classical estimation
   and differentiation guarantees, including estimator cost.

The scalar broadcast encoding used experimentally is a corollary of the general
multidimensional result. Universality may appear only as context or a tightly
scoped corollary.

The mandatory Theorem novelty dossier contains:

- Exact statements, assumptions, and complete proofs.
- A closest-prior-result matrix based on primary sources.
- A precise novelty delta for every theorem.
- Symbolic or numerical checks and attempted counterexamples.
- The manuscript claim-to-theorem map.

Pass condition: at least one central finite-resource response theorem is
correct, is not a direct corollary of known IQP or Fourier results, and
materially supports fixed-circuit conditioning.

Failure action: do not begin the Ising training campaign. Redirect the
contribution or venue before spending the run budget.

## Stage 2 — Calibrate and certify the Ising targets

Use the official Xanadu/PennyLane periodic random-coupling \(4\times4\) Ising
generator prescription. Freeze eight coupling instances, including canonical
seed 666, before model training. For each instance:

1. Enumerate all \(2^{16}\) states to obtain exact target probabilities.
2. Select one shared beta interval using target properties only: adjacent
   target TVD, entropy, concentration, moments, and thermal-response
   diagnostics. The interval must contain \(\beta=1/3\), corresponding to the
   released \(T=3\) point.
3. Freeze five Training betas spanning the interval, including its endpoints
   and \(\beta=1/3\), plus the four adjacent beta midpoints as Held-out
   Conditions.
4. Map beta affinely to \([0,\pi/4]\) and prepare
   \(R_Y(\alpha(\beta))^{\otimes16}|0^{16}\rangle\). The map is fixed,
   broadcast, shared across coupling instances after target-family interval
   calibration, and unlearned.
5. Generate independent PennyLane-style MCMC chains for Training and
   Validation using the official burn-in and thinning approach.
6. Certify every instance-beta pair against exact enumeration using frozen
   effective-sample-size, chain-agreement, moment, correlation, and
   distributional checks. An ineligible beta triggers recalibration from target
   and sampler evidence only, never from model performance.
7. Freeze 5,000 Training strings and 2,000 disjoint Validation strings per
   instance and Training beta. Use Condition-balanced minibatches of 256.

Exact distributions, rather than an empirical test set, supply final
Bitstring TVD.

## Stage 3 — Runtime and representability gate (maximum 20 runs)

First benchmark one complete training run for runtime and peak memory using the
frozen sampled-MMD budget. Every MMD evaluation, for every family in Training
and Validation, uses 2,000 independently sampled Fourier operators per
bandwidth and 2,000 product-reference samples per operator and Condition. A
resource failure reopens the protocol explicitly; it does not permit a silent
reduction.

Then train sampled-MMD Separate models on the canonical and one frozen
secondary coupling instance at the low, anchor, and high Training betas.

Pass condition:

- Median exact Bitstring TVD is at most 0.25.
- Every screened cell has exact Bitstring TVD at most 0.35.

Failure action: run at most the one allowed exact-objective diagnostic to
classify estimator versus model-family limitation, then stop the shared Ising
campaign. Do not change topology, target range, objective, or thresholds.

## Stage 4 — Bounded tuning and phase-feature screen (maximum 36 runs)

Give fixed-input, Condition-modulated-phase, and No-condition shared families
equal, preregistered Training/Validation-only selection opportunity. They share
the same data and maximum update count but may use different optimizer
hyperparameters. No Held-out Condition may be evaluated.

For the primary parameter-matched phase baseline, screen exactly three fixed,
instance-shared 136-coefficient assignments:

- A common normalized affine multiplier.
- Deterministic signed affine multipliers across generators.
- A balanced constant/slope generator bank.

Select one assignment and one protocol per family, then freeze all optimizer,
initialization, stopping, checkpoint, and evaluation choices. Any
intercept-plus-slope higher-capacity phase model is descriptive only and cannot
define benefit retention.

## Stage 5 — Frozen visible-only primary matrix (320 runs)

Use a visible-only \(m=0\), Complete 2-local IQP topology at 16 qubits, with
136 trainable generators. The fixed-input model must use the identical compiled
gate sequence and phase values at every beta; only its product-state input
changes.

For each of eight coupling instances and five matched optimizer seeds, train:

- One fixed-circuit input-conditioned shared model.
- One parameter-matched Condition-modulated-phase shared model.
- One No-condition shared model.
- Five Separate-model references, one per Training beta.

This gives \(8\times5\times8=320\) full runs. Compare all models with exact
targets. Report Training-beta fidelity as well as all four untouched Held-out
betas.

Primary statistical analysis uses 10,000 paired hierarchical bootstrap
replicates with a frozen RNG seed, resampling coupling instances first and
optimizer seeds within selected instances while preserving model and Condition
pairing. Report 95% percentile intervals.

### Binding empirical gates

Fixed-circuit benefit retention is

\[
R=\frac{\mathrm{TVD}_{no}-\mathrm{TVD}_{fixed}}
        {\mathrm{TVD}_{no}-\mathrm{TVD}_{phase}}.
\]

It passes only if median \(R\ge0.5\) on Held-out Conditions and the lower 95%
hierarchical-bootstrap bound is strictly above zero.

Ising interpolation also requires all three outcomes against Oracle
adjacent-beta lookup:

- The paired 95% interval for Oracle-minus-fixed Bitstring TVD is strictly
  positive.
- Fixed evaluation wins at least 75% of coupling-instance by Held-out-Condition
  cells.
- No coupling instance has worse mean Held-out Bitstring TVD than Oracle.

Failure action: cancel the ancilla extension and write the empirical result as
a boundary or negative result. Do not relax thresholds or start a rescue sweep.

## Stage 6 — Conditional ancilla extension (24 runs)

Run only if the complete visible-only primary passes. Use the canonical and one
frozen secondary coupling instance, three matched optimizer seeds, and
\(m\in\{3,5\}\). At each rung train the ancilla-marginalized fixed-circuit model
and a visible-only Matched-capacity control, producing
\(2\times3\times2\times2=24\) new runs. Reuse the corresponding primary
\(m=0\) checkpoints.

An ancilla improvement may be stated only if at least one rung:

- Lowers paired median Held-out Bitstring TVD against both \(m=0\) and its
  Matched-capacity control.
- Beats each comparator in at least 75% of
  instance-seed-Held-out-Condition cells.
- Has no worse mean Held-out TVD than \(m=0\) on either selected instance.

Report both rungs regardless of outcome and label the two-instance evidence
descriptive rather than population-level.

## Stage 7 — Theory-aligned audits outside the training-run budget

### Conditional-response spectrum audit

Evaluate frozen checkpoints on a dense beta grid without retraining. For
selected full probabilities and Pauli observables:

- Measure Condition-frequency content.
- Compare it with theorem-predicted support.
- Quantify target spectral mass outside that support.
- Relate unresolved target content to interpolation error.
- Compare fixed-input and Condition-modulated-phase responses.

This audit must test a prediction of the theory rather than serve only as a
Fourier visualization.

### Estimator and gradient scaling benchmark

Use synthetic frozen parameters and no full training runs. Run two tracks:

- A bounded-degree \(O(n)\)-generator track to isolate qubit scaling.
- A Complete 2-local \(O(n^2)\)-generator track to measure the practical cost
  of the experimental ansatz.

Vary qubit count, generator count, Condition count, and Condition dimension.
At small sizes report exact estimator and gradient-estimator bias, variance,
and RMSE. At larger sizes report repeat-estimator consistency, runtime, and
peak memory. Raw gradient magnitude and barren-plateau behavior are separate
landscape questions and do not determine whether the estimator is classical.

## Stage 8 — Write the full manuscript and supplement

Main text:

1. Introduction, deployment problem, claim boundaries, and operational
   definition of *classically trainable*.
2. Fixed-circuit conditional IQP construction and sampled-MMD estimator.
3. Finite-resource multivariate response theory for probabilities and
   estimable observables.
4. Estimator and gradient scaling evidence.
5. Ising protocol, frozen baselines, and statistical rules.
6. Primary visible-only results and response-spectrum audit.
7. Boundary analysis, limitations, and deployment implications.

Include only one compact Gaussian motivating result in the main text,
contrasting Hamming-profile control with Bitstring failure. Move sparse grids,
the full ancilla ladder, permutation twirling, Condition-count scaling,
topology comparisons, and complete negative audits to the supplement. Twirling
must not appear in the Abstract or Conclusion.

Every result paragraph must separate observations from interpretation, name its
replication unit, and state which frozen gate it addresses. Related Work must
distinguish estimator tractability from classical simulation, sampling,
optimization convergence, and landscape trainability.

## Stage 9 — Reproducibility and internal full-draft audit

The full-draft package includes:

- Executable code and environment specification.
- Frozen configurations and all random seeds.
- Generated coupling instances and exact target tables.
- Calibration, MCMC sampler-audit, and finite-data artifacts.
- Run ledger, raw metrics, and immutable primary result tables.
- Analysis, statistics, and figure scripts.
- Theorem-check notebooks and the complete novelty dossier.
- Large checkpoints in a persistent archive if they do not fit the repository.

Before external review, perform an internal adversarial pass covering proof
correctness, claim-support alignment, baseline strength, leakage, statistical
reproduction, figure/table regeneration, citation completeness, and PDF layout.
The draft is not mature while any main claim depends on a TODO, provisional
figure, missing supplement proof, unavailable raw result, or untracked manual
analysis.

## Stage 10 — External submission gate

Give the mature full draft and reproducibility package to independent quantum
information or quantum-machine-learning experts. Ask them specifically to
assess theorem novelty, correctness, significance for Quantum, the meaning of
*classically trainable*, baseline fairness, and whether the fixed-circuit Ising
evidence supports the stated scope.

Submission requires this gate to pass. Major objections may reopen proofs,
experiments, claims, or venue choice; external review is not expected to invent
the basic study design.

## Training-run ledger

| Release | Maximum full runs | Transferable? |
|---|---:|---|
| Runtime feasibility and representability | 20 | No |
| Matched tuning and phase-feature screen | 36 | No |
| Frozen visible-only primary matrix | 320 | No |
| Conditional ancilla extension | 24 | No |
| **Total ceiling** | **400** | |

Target calibration and estimator-only scaling consume no full training runs.
Runs unused because a gate fails remain unused.

## Definition of ready

The artifact is a Quantum-ready full draft only when:

- The Theorem novelty dossier passes internally.
- Every executed gate is reported, including failures and stopped branches.
- The central empirical wording matches the frozen outcomes without threshold
  changes or post-hoc subgroup selection.
- All main figures and tables regenerate from archived raw results.
- Proofs, supplement, Related Work, limitations, code, data, and analysis are
  complete.
- The rendered PDF has no blank pages, placeholder text, broken references, or
  unresolved layout defects.
- The complete package is ready for independent external review.

Meeting these conditions produces a defensible full draft. It does not itself
guarantee acceptance by Quantum; that remains the purpose of the external
submission gate.
