# Adversarial paper self-review

This review follows the five-dimension checklist in the
`research-paper-writing` skill. “Scoped pass” means the paper states the
limitation and does not make the stronger claim that would require new
evidence.

## 1. Contribution

| Question | Status | Evidence or revision |
|---|---|---|
| New knowledge? | Pass | Extends the IQP amplitude-ratio Pauli estimator to arbitrary real product-state inputs and identifies a condition-only input seam. |
| Meaningful failure case? | Pass | Tests whether one shared parameter vector can represent condition-indexed laws without condition-specific gates. |
| Non-obvious technical idea? | Pass | Product amplitudes preserve sampling/evaluation, but require an amplitude-ratio estimator and second-moment argument. |
| Surprising insight? | Pass | The basis control wins one shell comparison yet catastrophically fails the other because its outputs are exact complements. |
| Clear novelty type? | Pass | Method extension, unbiased conditional objective, and a negative controlled experimental finding. |

## 2. Writing clarity

| Question | Status | Evidence or revision |
|---|---|---|
| Reproducible from paper? | Pass | Circuit, estimator, kernels, targets, topology sizes, budgets, optimizer, seeds, baselines, and metrics are stated. |
| Key-module detail? | Pass | Product amplitudes, estimator, variance, gradient interchange, U-statistic behavior, and exact evaluation are described. |
| Motivation explicit? | Pass | Each module is tied to shared conditioning or preservation of tractable observable training. |
| Consistent terms/notation? | Pass | \(p_c\) is the target, \(q_\theta(\cdot\mid c)\) the shared model, and \(a,g,z\) have distinct roles. |
| Paragraph focus? | Pass | Reverse outline in `outline.md` assigns one rhetorical role to each paragraph group. |

## 3. Experimental strength

| Question | Status | Evidence or revision |
|---|---|---|
| Meaningful baseline effects? | Pass | Exact TVD differences are large; all medians and IQRs use ten seeds. |
| Strong absolute performance? | Scoped pass | Shell TVD is useful but bitstring TVD remains 0.27–0.49; the paper calls this incomplete recovery. |
| Consistent across settings/metrics? | Scoped pass | The strict rule fails on condition-A shell TVD; the abstract and discussion say so directly. |
| Strengths and failures honest? | Pass | Criterion failure, within-shell bias, sharing cost, and topology/capacity confounding are prominent. |

## 4. Evaluation completeness

| Question | Status | Evidence or revision |
|---|---|---|
| Key ablations? | Pass | Three native topologies, no condition, basis condition, and independent reference fits. |
| Strong/fair baselines? | Pass for scoped claim | Conditioning controls use the same complete ansatz; independent models expose sharing cost. |
| Standard/sufficient metrics? | Pass | Exact bitstring and shell TVD plus mean, standard-deviation, and held-out MMD diagnostics. |
| Challenging scenarios? | Scoped pass | Two 19-bit synthetic laws isolate the mechanism but do not establish broad conditional generalization. |
| Protocol documented? | Pass | Preregistered design and generated audit accompany machine-readable per-seed results. |

## 5. Method-design soundness

| Question | Status | Evidence or revision |
|---|---|---|
| Realistic practical setting? | Scoped pass | This is a controlled mechanism study; hardware/noise and large-scale verification are excluded. |
| Hidden technical defects? | Pass | Estimator values and gradients are checked against exact statevectors and brute-force kernel U-statistics. |
| Robust without per-case tuning? | Pass | One frozen estimator/training protocol is shared across models and seeds. |
| Benefits versus complexity? | Pass | Conditioning adds only input angles; no condition-specific trainable parameters or differentiation machinery. |
| Could net benefit be negative? | Scoped pass | Strict success fails, so the conclusion is conditional trainability and diagnostic insight—not superiority. |

## Remaining experiments, not paper defects hidden by wording

- Permutation-tied or permutation-augmented ansätze to address within-shell
  bias.
- Capacity-matched topology families.
- Held-out angle interpolation and more than two conditions.
- Noise-aware execution and scalable alternatives to exact TVD.
