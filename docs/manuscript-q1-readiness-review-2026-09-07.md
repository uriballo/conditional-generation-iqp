# Manuscript readiness review for a Q1 target

Reviewed 2026-09-07, repository HEAD 4fc4aa96. Scientific/editorial review of canonical Overleaf/main.tex, retained production summaries and claim ledger, project state, and selected primary literature. This is not a fresh raw-artifact audit, complete proof/code verification, or PDF layout review. No manuscript or experimental changes were made.

## Assessment

A credible focused methods paper is emerging, but the current draft is not submission-ready. The strongest asset is the completed matched empirical study. The largest publication risk is incremental novelty combined with narrow benchmarks, rather than too few runs. Q1 is index-, category-, and year-specific; the project's saved Quantum Machine Intelligence target must be checked against the institution's required ranking source before assuming eligibility. No journal quartile is certified by this review.

## Evidence already available

The retained production subset contains 2,600 fits across Gaussian and Curie–Weiss, n=8/12, five ancilla counts and ten matched optimization seeds. Median held-out shared raw TVD r=0 to r=10: Gaussian n8 0.219 to 0.102, n12 0.292 to 0.174; CW n8 0.190 to 0.047, n12 0.275 to 0.076 (CW averages include the landmark). Paired interpolation advantages at r=10: Gaussian n8 0.0198 [0.0173,0.0211]; n12 0.0049 [-0.0077,0.0137]. CW non-landmark advantages: n8 0.0119 [0.0080,0.0136], n12 0.0363 [0.0299,0.0433]. These are seed-level summaries conditional on frozen data archives, not uncertainty over independently resampled training datasets.

Canonical evidence: [claim table](paper_8_12/results_narrative_review.md), [summary](paper_8_12/summary.json), [provenance](paper_8_12/provenance.json).

## Submission blockers and reviewer risks

1. **Missing promised estimator argument.** main.tex around line 397 promises a derivation and finite-second-moment guarantee in Conditional IQP training. That section supplies character expansion and mask compression but no phase-weighted amplitude-ratio moment estimator, second-moment calculation, or explicit gradient argument. Supply these, explain support zeros and arithmetic assumptions, and show how independent reference draws/U-statistics avoid bias when squaring estimated moments. This is a missing exposition/proof, not a demonstrated invalid method.
2. **Novelty needs a precise comparison.** Conditional Born machines, input Fourier encodings, and classical IQP MMD training predate this work. Identify exactly what the fixed learned unitary and tractable conditional input enable beyond these ingredients; clearly attribute inherited results. Explain the practical benefit of keeping the unitary fixed, without implying a demonstrated deployment advantage.
3. **Both targets are exchangeable.** Their target distributions reduce to n+1 shell masses plus uniform within-shell sampling. This follows directly from the target definitions. Full-bitstring TVD is still valuable because the learned circuit need not be exchangeable, but these are weak evidence of general-purpose conditional modeling.
4. **Comparator scope is narrow.** Adjacent independent output mixtures answer a sharing/interpolation question, not competitive classical conditional learning. Add a data-trained shell-distribution model with condition dependence and a suitably small generic conditional classical comparator if broader empirical relevance is desired. Match training data, validation access, tuning effort, and report computational budgets. Analytic target samplers establish target simplicity; they are not fair learned baselines.
5. **Ancilla attribution remains combined.** Ancillas change phase count, encoded condition response, and marginalization together. The manuscript mostly acknowledges this correctly. Retain that wording or add a targeted fixed-versus-conditioned-ancilla control. Do not claim isolated purification benefit. Resource comparisons should include encoder preparation cost, especially for the cat-like state.
6. **Ablation remains exploratory.** Three development seeds and an unresolved immutable protocol/artifact pointer. Resolve provenance; show all four encoder/map arms in a table or factorial figure (the current figure shows only two). Confirm with fresh matched seeds if the interaction becomes a headline contribution. Do not label existing development seeds confirmatory.
7. **Scalable estimation versus selection.** The campaign selects checkpoints with full-distribution TVD; the runner explicitly computes exact model probabilities. Clarify empirical validation-target versus analytic test-target TVD and the exponential model-evaluation step. A retained-checkpoint comparison using validation MMD selection would test sensitivity without retraining, if required candidates/metrics remain available. Classical estimator efficiency is not an end-to-end scalable demonstrated workflow here.
8. **Unfinished paper.** Abstract and Introduction have TODOs, Discussion is preliminary, Conclusions is one sentence; author/affiliation, funding, and Zenodo placeholders remain. Gaussian Results line 827 calls held-out oracle gaps 'sharing costs', contradicting the earlier definition: use oracle gaps. Move the untested two-dimensional construction to an appendix or identify its unevaluated status explicitly. Complete a clean build and visual review after revision.

## Recommended order

First finish the estimator argument, sharpen novelty and scope, resolve ablation provenance, and reconcile the unfinished narrative. Next add the inexpensive classical comparison and checkpoint-selection sensitivity, reporting limitations even if unfavorable. For a stronger empirical submission, test one nonexchangeable target through a predeclared capacity gate before launching a campaign; the existing conditional bars-and-stripes proposal is a candidate, not evidence of success. Merely adding 16 visible qubits, another exchangeable target, hardware noise, or more seeds everywhere does not address the central weakness. A third benchmark is strategically useful but not a universal publication requirement.

For a selective quantum-methods venue, a stronger conceptual result could replace broader benchmarking; neither route guarantees acceptance. The present manuscript should be framed as a tractable conditioning interface with measured limits and selective interpolation gains, not quantum advantage, general scaling, or critical-point discovery.

## External context checked

- [Conditional Born machine for Monte Carlo event generation](https://link.aps.org/doi/10.1103/PhysRevA.106.022612): prior conditional QCBM work.
- [IQPopt](https://arxiv.org/abs/2501.04776): prior classical observable-based optimization and MMD framework.
- [Train on classical, deploy on quantum](https://arxiv.org/abs/2503.02934): existing large-scale IQP generative learning and classical comparisons; this paper's contribution must be conditional structure rather than scalable IQP training alone.
- [Quantum author criteria](https://quantum-journal.org/instructions/authors/): technical correctness, significance, clarity/reproducibility, and honest scope; correct but incremental work can fall below its threshold. Used as a selective-venue reference, not certification of a Q1 ranking.
