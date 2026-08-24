# Abstract and Introduction claim-evidence registry

> **Historical registry.** This file records the frozen failed-Ising manuscript
> path and is not an active specification for the current Gaussian plus
> Curie--Weiss v3 manuscript. Active experimental eligibility is defined in
> `experiment-completion-audit.md`; active Curie--Weiss claims are mapped in
> `curie-weiss-ancilla-claim-evidence-registry-v3.md`.

Status: binding internal register for the planned Quantum draft  
Audit date: 2026-08-10  
Theorem gate: **failed**; affirmative primary-Ising claims remain ineligible

## Use of this registry

Every factual or contribution claim intended for the Abstract or Introduction
must have one row below. A sentence marked `BLOCKED` or `REJECTED` must not be
placed in the manuscript. `PROVEN-INTERNAL` means that an internal proof and
regression checks exist; it does not mean that novelty or external correctness
has been certified.

This register maps the planned theory-led manuscript required by the Quantum
campaign specification. The current legacy Gaussian/twirling Abstract is not
the target claim set. Its compact Gaussian result may remain as motivation,
but the positive Ising claims required by the new specification have no
released production evidence at the audit date. Target-only calibration,
eight exact target archives, and two failed sampler pilots are preparatory
artifacts, not a passed campaign gate.

## Status vocabulary

| Status | Meaning |
|---|---|
| `PROVEN-INTERNAL` | Exact assumptions and an internal proof exist; executable small-system checks pass |
| `KNOWN-SPECIALIZATION` | Correct internally, but the claimed result is a specialization or elementary consequence of cited primary work |
| `EMPIRICAL-VERIFIED` | A frozen result artifact, protocol decision, and reproducible statistic exist |
| `DECLARED-ONLY` | A manifest or protocol exists, but the required run/result artifact does not |
| `BLOCKED` | A predecessor gate failed or required evidence is absent |
| `REJECTED` | The wording is false, overbroad, or incompatible with the specification |
| `EXTERNAL-REVIEW` | Internal evidence exists, but independent specialist review is still mandatory |

## Evidence index

### Internal theorem and code evidence

| Evidence ID | Artifact | What it supports | Limitation |
|---|---|---|---|
| `TH-T1` | [Theorem dossier, T1](theorem-novelty-dossier.md#2-t1--full-probability-affine-input-frequency-support) | Full-probability finite signed-sum support for a fixed affine `RY` map | Known specialization; no efficient probability evaluation |
| `TH-T2` | [Theorem dossier, T2](theorem-novelty-dossier.md#3-t2--iqp-anticommutationspan-refinement-for-pauli-observables) | Active anticommuting set, binary span, real-input `Y` filter, refined observable support | Direct synthesis of close prior results; coefficients may vanish |
| `TH-T3` | [Theorem dossier, T3](theorem-novelty-dossier.md#4-t3--bandwidth-and-derivative-bounds) | Coordinate/directional bandwidth and derivative bounds | Regularity, not optimization-landscape trainability |
| `TH-T4` | [Theorem dossier, T4](theorem-novelty-dossier.md#5-t4--fixed-width-non-density-limitation) | Fixed-width non-density in continuous conditional distributions | Does not address growing width or re-upload depth |
| `TH-T5` | [Theorem dossier, T5](theorem-novelty-dossier.md#6-t5--product-reference-estimator-gradients-and-costs) | Unbiased Pauli estimator, second moments, active gradients, costs | Not full probabilities, sampling, convergence, or advantage |
| `CODE-R` | [`src/iqp_cond/response.py`](../src/iqp_cond/response.py) | Executable support, span, lightcone, and bandwidth enumeration | Exponential theorem-check utility, not a scalable training routine |
| `TEST-R` | [`tests/test_theorem_checks.py`](../tests/test_theorem_checks.py) | Small-system frequency and lightcone counterexample checks | Numerical regression evidence only |
| `TEST-E` | [`tests/test_simulator.py`](../tests/test_simulator.py) | Monte Carlo expectation and autodiff-gradient agreement with exact circuits | Finite tested instances, not a proof |
| `TEST-SCALE` | [`tests/test_estimator_scaling.py`](../tests/test_estimator_scaling.py) | Exact small-system bias/variance/RMSE and large-system repeated-estimator contracts across both growth tracks | Synthetic estimator evidence only; no model training or landscape claim |

### Primary-source evidence

| Evidence ID | Primary source | Supported background statement |
|---|---|---|
| `SRC-F1` | [Gil Vidal and Theis (2020)](https://doi.org/10.3389/fphy.2020.00297) | Pauli-encoding expectation functions have finite `{-1,0,1}` angle spectra; affine weights form signed-sum spectra; input redundancy limits Fourier rank |
| `SRC-F2` | [Schuld, Sweke, and Meyer (2021)](https://doi.org/10.1103/PhysRevA.103.032430) | Encoding Hamiltonian eigenvalue differences determine accessible Fourier frequencies, including multivariate inputs |
| `SRC-F3` | [Casas and Cervera-Lierta (2023)](https://doi.org/10.1103/PhysRevA.107.062612) | Multidimensional Fourier expressivity has degree-of-freedom limitations |
| `SRC-E1` | [Van den Nest (2010/2011)](https://arxiv.org/abs/0911.1624) | Probabilistic classical estimation can apply where strong probability simulation fails |
| `SRC-E2` | [Recio-Armengol and Bowles, IQPopt v2 (2026)](https://arxiv.org/abs/2501.04776) | Large IQP circuits admit efficient classical estimation/optimization for efficiently described Pauli-observable objectives; this is separated from output sampling |
| `SRC-E3` | [Recio-Armengol, Ahmed, and Bowles v2 (2026)](https://arxiv.org/abs/2503.02934) | Explicit active-anticommuting IQP expectation expansion, unbiased Pauli/MMD estimation, and autodifferentiated gradients |
| `SRC-C1` | [Ni and Van den Nest (2012)](https://arxiv.org/abs/1204.4570) | Classical simulation results and boundaries for commuting circuits/local measurements |
| `SRC-H1` | [Bremner, Jozsa, and Shepherd (2011)](https://doi.org/10.1098/rspa.2010.0301) | Exact classical simulation of suitable commuting computations would have unlikely complexity consequences |
| `SRC-H2` | [Bremner, Montanaro, and Shepherd (2016)](https://doi.org/10.1103/PhysRevLett.117.080501) | Approximate IQP sampling hardness is conditional on explicit average-case/complexity assumptions |

### Campaign evidence availability

| Evidence ID | Required artifact | Availability on 2026-08-10 | Consequence |
|---|---|---|---|
| `CAMP-M` | [`configs/quantum_campaign_v1.json`](../configs/quantum_campaign_v1.json) | Present; declares seeds, beta split, models, budgets, and thresholds | Protocol intent only; not result evidence |
| `CAMP-CAL` | [`results/quantum-campaign-v1/calibration.json`](../results/quantum-campaign-v1/calibration.json) and eight `targets/coupling_*.npz` archives | Present; SHA-256 `8d121e4a519766ff0234e00021815b3bcfd740e9fc9284dbf5b2aa448f29f6ae` matches the manifest; all eight exact target archives are present | Preparatory target-only evidence only. `TARGET_CALIBRATION` was not released because its predecessor theorem gate failed |
| `CAMP-GATE-T` | Passing theorem novelty decision | Failed in `TH-T1`--`TH-T5` audit | Major Ising compute is blocked by ADR 0007/0042 |
| `CAMP-S10` | [`sampler-attempts/thinning-10/summary.json`](../results/quantum-campaign-v1/sampler-attempts/thinning-10/summary.json) | Complete 40-cell pilot; overall `passed: false`; validation ESS failed for seeds/betas `(202, 0.55)` and `(404, 1/3)` | Not a certified dataset; negative preparatory evidence |
| `CAMP-S16` | [`sampler-attempts/thinning-16/summary.json`](../results/quantum-campaign-v1/sampler-attempts/thinning-16/summary.json) | Complete 40-cell pilot; overall `passed: false`; validation ESS failed for `(303, 0.15)` and `(505, 0.55)` | Not a certified dataset; negative preparatory evidence |
| `CAMP-S32` | Planned 32-thinning rerun | Interrupted immediately on theorem-novelty failure; no complete certification | Stopped branch; supplies no pass claim and must not be resumed as an implicit rescue |
| `CAMP-REP` | Runtime/representability result bundle and gate decision | Absent | No Separate-model representability claim |
| `CAMP-PRI` | Complete 320-run primary matrix and immutable ledger | Absent | No fixed-input/phase/no-condition/separate Ising comparison claim |
| `CAMP-STAT` | 10,000-replicate paired hierarchical bootstrap artifact | Absent | No interval, retention, or population statement |
| `CAMP-SPEC` | Dense-beta spectral audit linked to frozen checkpoints | A machine-readable blocked preflight exists at [`evidence/response-spectrum.json`](../results/quantum-campaign-v1/evidence/response-spectrum.json); no compatible primary checkpoints exist | No spectrum comparison or target-spectrum/error relationship claim |
| `CAMP-SCALE` | Sparse and complete estimator-scaling tracks | [`evidence/estimator-scaling.json`](../results/quantum-campaign-v1/evidence/estimator-scaling.json) contains 21 deterministic synthetic cases; configuration digest `d647f220e3791c4eb678934d25f2ab1557e9396055f3b0e7a4a6ffdd04b28674`, numerical-results digest `54b5a75c383db7d605efe971d185734c00b2b47adcd0d5438cf6f2a77eeddfb0` | Verifies estimator error/consistency and host resource measurements across declared axes; zero full training runs and no landscape inference |
| `CAMP-ANC` | Released 24-run `m=3,5` paired ancilla extension | Absent and predecessor primary gate unavailable | No campaign ancilla claim |
| `OLD-G` | [`results/summary/metrics.md`](../results/summary/metrics.md) and archived legacy runs | Present for the earlier Gaussian mechanism study | Compact motivation only; not evidence for the Ising campaign |

## Intended Abstract claim map

The “allowed wording” is deliberately conservative. It is not mandatory prose,
but any stronger version requires a new evidence row.

| ID | Intended claim | Evidence | Status | Allowed wording / action |
|---|---|---|---|---|
| `A-01` | One fixed IQP circuit can receive a condition solely through an affine product-state input | Model assumptions; `TH-T1`; `TH-T5` | `PROVEN-INTERNAL` | “We study a fixed `H D_Z H` IQP circuit whose condition enters only through a frozen affine `RY` product input.” The theorem treats the coefficients as fixed; this defines the mechanism and does not establish useful accuracy. |
| `A-02` | Every full output probability has a finite multivariate condition-frequency support | `TH-T1`, `SRC-F1`, `SRC-F2` | `KNOWN-SPECIALIZATION` | “Specializing established circuit Fourier theory, every output probability lies in the signed-sum support `Omega_full(W)`.” Do not call this theorem new. |
| `A-03` | The Pauli observables used in training have an IQP-specific refined support | `TH-T2`, `TEST-R`, `SRC-E3` | `PROVEN-INTERNAL`; novelty fail | “For a Pauli-`Z` response, only anticommuting diagonal generators enter; the active binary span is further filtered by zero `Y` moments of the real input.” State that this refines the known active-generator expansion. |
| `A-04` | Fixed resources impose bandwidth, smoothness, and approximation limits | `TH-T3`, `TH-T4`, `SRC-F1`--`SRC-F3` | `KNOWN-SPECIALIZATION` | “The fixed encoding imposes explicit derivative bounds and a finite-dimensional response space, so fixed width is not universally dense.” |
| `A-05` | The conditioned Pauli response has an unbiased amplitude-ratio estimator with bounded second moment | `TH-T5`, `TEST-E`, `SRC-E1`, `SRC-E3` | `PROVEN-INTERNAL`; known framework | “A product-reference amplitude-ratio estimator is unbiased and has second moment at most one.” Never say each sample is bounded by one. |
| `A-06` | Training gradients can be estimated by ordinary autodifferentiation of the sampled objective | `TH-T5`, `TEST-E`, `SRC-E2`, `SRC-E3` | `PROVEN-INTERNAL`; known framework | “Because the reference law is phase-parameter independent, differentiating the unbiased finite-sample expression yields unbiased phase gradients.” For squared MMD, name the independent-copy/U-statistic construction. |
| `A-07` | “Classically trainable” has a narrow operational meaning | `TH-T5`, `SRC-E1`--`SRC-E3` | `PROVEN-INTERNAL` | Define it on first use as efficient classical estimation of the selected Pauli moments, sampled MMD, and phase gradients for polynomial budgets. |
| `A-08` | The same method evaluates output probabilities or samples classically | `TH-T1`, `TH-T5`, `SRC-E2` | `REJECTED` | Explicitly state the opposite: the probability theorem is structural, and no efficient full-probability evaluator or classical output sampler is supplied. |
| `A-09` | On the frozen Ising family, fixed input retains at least half of the reprogrammed-phase benefit | `CAMP-PRI`, `CAMP-STAT` required | `BLOCKED` | No numerical or qualitative positive Ising sentence is eligible. The theorem gate failed and no primary matrix/statistical artifacts exist. |
| `A-10` | Fixed-input interpolation beats adjacent-beta lookup under all frozen rules | `CAMP-PRI`, `CAMP-STAT` required | `BLOCKED` | Omit unless all interval, 75% cell-win, and no-instance-regression rules are reproduced from archived outputs. |
| `A-11` | Observed unresolved target spectral mass explains held-out error | `CAMP-SPEC` required | `BLOCKED` | Omit. T1 predicts allowed support but no dense-beta checkpoint/target spectral artifact exists. |
| `A-12` | The paper introduces a novel central conditional-response theorem | Theorem dossier gate | `REJECTED` | Current audit finds no theorem that is both central and not a direct corollary/synthesis. “We prove for the first time” and equivalent wording are prohibited. |
| `A-13` | The results show conditional universality, generic trainability, classical superiority, hard sampling, or quantum advantage | Specification boundaries; `TH-T4`; `SRC-H1`, `SRC-H2` | `REJECTED` | State explicitly that none of these is claimed. Sampling-hardness context must be conditional and separate from the studied finite instances. |
| `A-14` | The earlier Gaussian study motivates the fixed-circuit question | `OLD-G` | `EMPIRICAL-VERIFIED` for the legacy study | At most one compact, clearly labeled motivating result. Do not let shell-profile or twirled fidelity substitute for primary untwirled Ising evidence. |
| `A-15` | Ancilla marginalization improves the campaign result | `CAMP-ANC` required | `BLOCKED` | Omit from the Abstract. The extension is unreleased and, by specification, cannot rescue a failed primary study. |

### Abstract disposition

With the gate and artifact state above, a Quantum-ready Abstract cannot contain
the planned positive Ising conclusion. A truthful current Abstract can only
describe the mechanism, state the known/specialized theory, define the
estimator boundary, and disclose that the novelty gate failed. Such an
Abstract is a research-stop/failure-path draft, not a submission-ready Quantum
paper.

## Intended Introduction claim map

| ID | Intended claim | Evidence | Status | Allowed wording / action |
|---|---|---|---|---|
| `I-01` | Conditional generation seeks one parameterized model for a family `q_theta(x|c)` | Definition | `PROVEN-INTERNAL` | Definition/background; no priority claim needed. |
| `I-02` | The deployment constraint is one compiled gate sequence and phase vector at every condition | Specification, model assumptions | `PROVEN-INTERNAL` | State before presenting results; distinguish it from condition-modulated phases and separate models. |
| `I-03` | Encoding choices constrain a quantum model's Fourier response | `SRC-F1`--`SRC-F3` | `KNOWN-SPECIALIZATION` | Cite the primary Fourier literature before T1. |
| `I-04` | IQP Pauli observables can be estimated classically even though strong probability simulation/sampling is a separate question | `SRC-E1`--`SRC-E3`, `SRC-H1`, `SRC-H2` | `KNOWN-SPECIALIZATION` | Use conditional language for hardness and say explicitly that efficient moments do not imply efficient sampling. |
| `I-05` | Existing observable-trained IQP work begins from one fixed reference input | `SRC-E2`, `SRC-E3` | `EXTERNAL-REVIEW` | Safer wording: “We consider the conditional extension in which the efficiently sampled reference product law varies with `c`.” Avoid an exhaustive “no prior work” claim. |
| `I-06` | The condition map is affine, shared across coupling instances, and fixed before training | Model assumptions; `CAMP-M`; preparatory `CAMP-CAL` | `PROVEN-INTERNAL` for theory; preparatory-only for campaign | State exact `alpha(c)=b+Wc`. The theorem fixes arbitrary coefficients; the campaign chooses one common map from target-family-only interval calibration. Do not call `TARGET_CALIBRATION` passed or released. |
| `I-07` | Full probabilities and estimable observables obey different results | `TH-T1`, `TH-T2`, `TH-T5` | `PROVEN-INTERNAL` | Give T1 as expressivity only; give T2/T5 as the observable-estimation route. |
| `I-08` | The active-generator span gives a topology-dependent observable lightcone | `TH-T2`, `TEST-R`, `SRC-E3` | `PROVEN-INTERNAL`; novelty fail | Call it a refinement/diagnostic derived from the known active-set expansion, not a wholly new lightcone theorem. |
| `I-09` | The response has explicit finite-resource bandwidth and smoothness | `TH-T3` | `KNOWN-SPECIALIZATION` | State exact dependence on `sum_i |W_ik|`; do not infer interpolation accuracy. |
| `I-10` | Fixed-width affine conditioning is not universally dense | `TH-T4`, `SRC-F1`--`SRC-F3` | `KNOWN-SPECIALIZATION` | Present as a limitation and prior-theory consequence, not the headline novelty. |
| `I-11` | The estimator is pointwise bounded | `TH-T5`, counterexample 10.6 | `REJECTED` | Only the second moment is bounded. Raw amplitude ratios can be arbitrarily large on rare reference outcomes. |
| `I-12` | Phase gradients outside the anticommuting active set vanish for a fixed Pauli response | `TH-T5`, `TEST-R` | `PROVEN-INTERNAL` | Qualify per observable. A loss mixture can activate the union of the sampled observables' sets. |
| `I-13` | Reverse-mode differentiation proves favorable landscape trainability | `TH-T5` | `REJECTED` | Autodiff proves estimator accessibility, not gradient magnitude, optimizer convergence, or lack of barren plateaus. |
| `I-14` | Contribution 1 is a new multivariate response theorem | `TH-T1`--`TH-T4`, `SRC-F1`--`SRC-F3` | `REJECTED` | Replace with an honest specialization/synthesis, or obtain a genuinely new theorem before retaining a theorem-led contribution claim. |
| `I-15` | Contribution 2 is an estimator preserving classical observable training under product-state conditioning | `TH-T5`, `TEST-E`, `SRC-E1`--`SRC-E3` | `PROVEN-INTERNAL`; modest delta | Accurate if framed as a conditioned specialization with explicit variance/cost accounting. |
| `I-16` | Contribution 3 is the frozen Ising mechanism benchmark | `CAMP-CAL`, `CAMP-S10`, `CAMP-S16`, `CAMP-REP`, `CAMP-PRI`, `CAMP-STAT` | `BLOCKED` | The Introduction may describe the preregistered design and stopped preparatory chronology only in a failure-path report. Calibration was not gate-released, both complete sampler pilots failed 2/40 validation ESS cells, and production results do not exist. |
| `I-17` | The fixed circuit empirically retains substantial reprogrammable benefit | `CAMP-PRI`, `CAMP-STAT` | `BLOCKED` | No evidence available. |
| `I-18` | The experiment connects target spectral mass to interpolation error | `CAMP-SPEC` | `BLOCKED` | No evidence available. |
| `I-19` | Estimator runtime and error are measured across qubits, generators, condition count, and condition dimension | `CAMP-SCALE`, `TEST-SCALE` | `EMPIRICAL-VERIFIED` | State that the 21 cases use frozen synthetic parameters, both $O(n)$ and $O(n^2)$ tracks, exact references only through $n=4$, and host-dependent runtime/RSS. Do not infer gradient magnitude or landscape trainability. |
| `I-20` | Ancillas are a secondary capacity extension | Specification; `CAMP-ANC` | `DECLARED-ONLY` | Describe only as a contingent planned extension unless the release and paired controls exist. |
| `I-21` | The legacy Gaussian/twirling program is the primary evidence line | Specification; `OLD-G` | `REJECTED` for the new draft | Retain one compact Gaussian motivation; move sparse grids, twirling, topology, scaling, and negative audits to the supplement. Target-specific twirling is barred from the Abstract and Conclusion. |
| `I-22` | Independent experts have certified significance and correctness | External-review gate | `BLOCKED` | No such claim until reviews are archived and major objections resolved. |

## Planned contribution paragraph that the evidence currently permits

The following content is accurate but does **not** satisfy the frozen novelty
gate:

1. We specialize established quantum-model Fourier results to a fixed affine
   `RY` product input and record the common finite support, derivative bounds,
   and fixed-width limitation of every output probability.
2. For Pauli-`Z` responses, we refine the known IQP active-generator expansion
   by grouping the active binary span and applying the real-input `Y` filter,
   which yields an observable-specific condition-frequency support.
3. We specialize the probabilistic IQP estimator to the condition-dependent
   product reference, give its second-moment and active-gradient bounds, and
   state the resource axes and non-implications precisely.

The paragraph must be followed by the adverse novelty conclusion unless a new
central result reopens the gate. It cannot currently be followed by a positive
Ising contribution.

## Rejected or mandatory-qualified phrases

| Phrase | Disposition | Replacement |
|---|---|---|
| “the full spectrum is `Omega_full`” | Reject unless “contained in” is used | “the structural support is contained in `Omega_full`; cancellations can shrink it” |
| “bounded estimator” | Ambiguous/reject | “an unbiased estimator with second moment at most one” |
| “Hoeffding gives logarithmic confidence” | Reject for the raw ratio | Use Chebyshev for the mean or a finite-variance median-of-means qualification |
| “efficient probability response” | Reject | “finite-support probability response; efficient selected-observable estimation” |
| “classically simulable conditional IQP model” | Reject | “classically estimable selected observables and sampled loss” |
| “trainable” without definition | Mandatory qualification | Define as estimator/loss/gradient accessibility, not landscape behavior |
| “smoothness guarantees interpolation” | Reject | “smoothness bounds response derivatives; accuracy remains target- and fit-dependent” |
| “observable lightcone equals the active-generator union” | Reject | “the union is an upper lightcone; the binary-span/`Y` filter may be tighter” |
| “first multivariate conditional-response theorem” | Reject | Cite the prior multivariate Fourier results and describe the specialization |
| “universal conditional generator” | Reject | T4 gives a fixed-width counterexample |
| “hard to sample, therefore quantum advantage” | Reject | State only the separate conditional complexity context with its assumptions |
| “the Ising benchmark demonstrates ...” | Blocked | No campaign result artifacts exist and the theorem predecessor gate failed |

## Manuscript reconciliation checklist

Before an Abstract/Introduction edit can be considered internally consistent:

- [x] Every theory sentence cites `SRC-F1`/`SRC-F2` before presenting T1 as a
  specialization.
- [x] T2 is described as an observable-specific refinement of the known
  active-generator expansion in `SRC-E3`.
- [x] “Classically trainable” is defined on first use with the exact T5 scope.
- [x] No sentence conflates T1 with efficient probability evaluation or
  sampling.
- [x] No sentence calls the raw amplitude-ratio samples bounded.
- [x] The fixed-width limitation is visible wherever conditional expressivity
  is summarized.
- [x] No positive Ising number or conclusion appears without `CAMP-PRI` and
  `CAMP-STAT`.
- [x] No spectral-explanation claim appears without `CAMP-SPEC`.
- [x] Ancilla claims remain absent unless the primary release and paired
  controls satisfy the frozen rule.
- [x] The failed theorem gate and missing external review are not presented as
  passed.
- [x] Target-specific twirling is absent from the Abstract and Conclusion.
- [x] Every current empirical sentence names its replication unit and frozen
  decision rule.

## Current readiness verdict

The claim registry is complete for the intended theory-led Abstract and
Introduction, but the evidence package is not submission-ready:

- mathematical correctness is provisionally supported;
- broad theory claims have close primary prior art;
- the central novelty requirement fails internally;
- exact targets and a hash-matched calibration artifact exist only as
  unreleased preparation; 10- and 16-thinning sampler pilots each failed
  validation ESS in 2/40 cells, and the 32-thinning branch stopped on the
  novelty verdict;
- representability, primary matrix, statistical, spectrum-comparison, and
  ancilla artifacts are unavailable; the separate zero-training-run synthetic
  estimator-scaling artifact is present;
- independent significance/correctness review is unavailable.

Therefore the only defensible current draft is an honest failure-path or
reframing draft. A positive Quantum submission narrative is blocked.
