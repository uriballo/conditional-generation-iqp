# Claim–evidence registry: arXiv foundations paper

Status: binding register for `arxiv_foundations_v1`  
Audit date: 2026-08-11  
Release status: **internal draft; representability gate failed**

This register is independent of the archived `quantum_campaign_v1` register.
Positive Ising wording is allowed in the Abstract or Introduction only when
both the conditional-control and interpolation gates have immutable `passed`
records. Neither gate exists because the preregistered representability screen
failed before tuning.

## Evidence index

| ID | Artifact | Status | Scope |
|---|---|---|---|
| `F-MANIFEST` | [`configs/arxiv_foundations_v1.json`](../configs/arxiv_foundations_v1.json) | frozen | Separate 12/12/36 compact protocol |
| `F-DATA` | [`results/arxiv-foundations-v1/evidence/data-ready.json`](../results/arxiv-foundations-v1/evidence/data-ready.json) | passed | Six active instances; exact categorical Training/Validation draws; zero Held-out draws |
| `F-EST` | [`results/arxiv-foundations-v1/evidence/estimator-calibration.json`](../results/arxiv-foundations-v1/evidence/estimator-calibration.json) | passed | Smallest passing equal budget is 128/128 |
| `F-RUNTIME` | [`results/arxiv-foundations-v1/evidence/runtime-feasibility.json`](../results/arxiv-foundations-v1/evidence/runtime-feasibility.json) | passed | 100-update preflight satisfies time and memory limits |
| `F-REP` | [`results/arxiv-foundations-v1/evidence/representability.json`](../results/arxiv-foundations-v1/evidence/representability.json) | **failed** | Median exact TVD 0.5122; maximum 0.8664 |
| `F-TUNE` | 12-run shared-family tuning matrix | blocked | Predecessor `F-REP` failed; zero tuning runs |
| `F-PRIMARY` | 36-run primary matrix | blocked | No tuning decision; zero primary runs |
| `F-CONTROL` | Ising conditional-control finalizer | blocked | No primary results |
| `F-INTERP` | Ising interpolation finalizer | blocked | No primary results |
| `G-PAIRED` | [`results/arxiv-foundations-v1/gaussian/paired-evidence.json`](../results/arxiv-foundations-v1/gaussian/paired-evidence.json) | verified | Ten paired optimizer seeds; 10,000 bootstrap replicates |
| `OLD-ARCHIVE` | [`archive/quantum-campaign-v1-failure-path/cc94836e/`](../archive/quantum-campaign-v1-failure-path/cc94836e/) | preserved | Content-addressed prior failure-path source and PDF |

## Eligible claims

| ID | Claim | Evidence | Status | Permitted wording |
|---|---|---|---|---|
| `M-01` | One fixed compiled IQP circuit receives the condition through a product-state input | model definition and deployment-signature tests | verified | “We define product-state-conditioned IQP generation using one condition-invariant compiled circuit and phase vector.” |
| `M-02` | “Classically trainable” means selected Pauli moments, Conditional MMD, and gradients are estimable | estimator derivations and executable checks | known specialization | State the operational definition and explicitly exclude classical full-law sampling. |
| `T-01` | Output probabilities have finite affine-input Fourier support | Gil Vidal–Theis; Schuld–Sweke–Meyer; local derivation | known specialization | “Specializing established circuit-Fourier theory…” Never claim theorem novelty. |
| `T-02` | Real product inputs filter the known IQP active-generator expansion | Recio-Armengol–Ahmed–Bowles; local derivation | synthesis | “Combining the active-generator expansion with the zero local (Y) moment…” |
| `G-01` | Product-state conditioning controls the Gaussian Hamming profile | `G-PAIRED` | empirical verified | Claim only for profile TVD; identify optimizer seed as the replication unit. |
| `G-02` | Dense conditioning interpolates in aggregate to held-out Gaussian conditions | `G-PAIRED` | empirical verified | State that paired mean profile TVD beats endpoint-only and adjacent lookup. |
| `G-03` | Adjacent lookup wins at one held-out Gaussian condition | `G-PAIRED` | empirical verified | Disclose (0.15625\pi) prominently with `G-02`. |
| `G-04` | Gaussian bitstring interpolation beats adjacent lookup | `G-PAIRED` | rejected | The paired 95% interval contains zero; report absence of advantage. |
| `I-01` | Compact 16-qubit Ising conditional generation works | `F-PRIMARY` required | blocked | No positive qualitative or numerical sentence. |
| `I-02` | Compact 16-qubit Ising interpolation/generalization works | `F-PRIMARY` required | blocked | No positive qualitative or numerical sentence. |
| `I-03` | The preregistered Ising representability screen failed | `F-REP` | empirical verified | Allowed in Methods/Limitations of the internal draft, not as a positive result. |

## Prohibited scope claims

The manuscript must not claim theorem novelty, quantum advantage, efficient
classical full-distribution sampling, extrapolation, distribution-shift
robustness, universal conditional expressivity, or optimizer/landscape
trainability. “Conditional generalization” may appear once only when defined
as interpolation to unseen conditions strictly inside the Training interval.

## Release checklist

- [x] New manifest and store do not alter `quantum_campaign_v1`.
- [x] Exact draws are role-separated and contain no Held-out samples.
- [x] Estimator and runtime gates are evidence-bound.
- [x] The failed representability decision stopped tuning and primary runs.
- [x] Gaussian summaries regenerate exactly and paired intervals are archived.
- [ ] Both Ising publication gates pass. **Blocked by representability failure.**
- [ ] Verified author names and affiliations supplied by the team.
- [ ] Contribution statement, competing interests, category, and license approved.
- [ ] Final human proof approval archived.

The `upload-ready` label is therefore prohibited.
