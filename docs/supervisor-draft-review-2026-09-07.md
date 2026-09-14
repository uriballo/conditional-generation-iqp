# First supervisor draft review — 2026-09-07

The draft is ready for a first scientific review by the supervisors. This is an assistant-reviewed handoff, not author approval or a journal-submission certification.

## Initial review figure arrangement (superseded below)

Removed the redundant Gaussian target heatmap and its references. The target remains specified by equations. Main figures now appear alongside the argument they support:

| Figure | Purpose | Page |
| --- | --- | --- |
| 1 | Conditional workflow, after the introduction | 2 |
| 2 | Gaussian accuracy and ancilla scaling | 11 |
| 3 | Curie–Weiss accuracy and ancilla scaling | 12 |
| 4 | Curie–Weiss conditioning ablation | 13 |
| 5 | Ashkin–Teller split and six-model comparison | 14 |

Appendix A contains detailed reconstruction, interpolation and condition-resolved profiles; Appendix B explains correlated reference access and preparation; Appendix C documents AT comparators, structural diagnostics and complete condition-level tables. Appendix profile headings are kept with their figures.

## Scientific consistency

- Distinguished exact-validation-TVD checkpoint selection for Gaussian/CW from sampled-validation-MMD selection for AT, including in the general training description.
- Corrected Gaussian “sharing costs” to “oracle gaps” for the paired statistics in question.
- Clarified that layer-exchange covariance describes the coupled-ring target and is not imposed on the learned circuit.
- Verified the CW development ablation's 24 evaluation identities, configuration/manifest linkage, stored probability and evaluation hashes, and complete temperature grid. Table II now includes all four input/map combinations. See [verification](research/curie-ablation-screen-verification-2026-09-07.json). This resolves the former missing artifact pointer without turning a three-seed development screen into production evidence.
- Retained interpolation-only scope, problem-informed input assumptions, finite-budget comparator limitations, separate training at each size, mean/median distinctions, and AT dependence/marginal/correlation exceptions. Classical wins are reported and no quantum advantage is claimed.
- Reviewed the estimator assumptions and moment/gradient argument. Existing numerical and test evidence remains recorded in the [claim-evidence ledger](paper_8_12/results_narrative_review.md); this pass introduced no new experiments and did not repeat the full 2,600-identity raw production audit.

## Editorial and administrative completion

Removed unused generated-content fallbacks, obsolete macros and commented draft material. Removed the unnamed author/affiliation and funding ellipses while preserving the four named authors and existing Eurecat fellowship acknowledgment. The author list and any additional funding should be confirmed by the authors before external submission. No identity or grant was invented.

Data availability now states the actual repository/artifact status and explicitly says public archival deposition is not yet complete. There is no fabricated Zenodo DOI. Scientific author review and release arrangements remain open.

## Validation and handoff

The 26-page PDF compiles with resolved references and citations, no overfull boxes and no stuck floats. All pages were visually reviewed; changed appendix heading pages were rendered again after the last layout edit. One benign `nameref` label-redefinition warning remains. Main figure/table text is readable, the former heatmap is absent, and numbering is continuous.

[Supervisor PDF](../output/pdf/supervisor-draft/conditional-iqp-supervisor-draft.pdf). Source and PDF hashes are recorded in [verification](research/supervisor-draft-verification-2026-09-07.json). Earlier integration snapshots are preserved. Nothing was sent, committed or pushed by this review.

## Body-figure revision requested by the author

The author requested that the main visual evidence appear in the body. The Results now include paired Gaussian reconstruction/midpoint panels (n=8), paired Curie–Weiss shape/baseline panels (n=12), and the Ashkin–Teller spatial/dependence diagnostic at both sizes. The existing accuracy, ablation and AT comparison figures remain in the body. The Gaussian target heatmap remains removed.

The complementary n=12 Gaussian and n=8 Curie–Weiss profiles remain in the appendix, together with condition-resolved and training/held-out checks, reference derivations and full condition tables. The main Curie–Weiss example deliberately retains the larger-size residuals. Captions distinguish reconstruction, held-out interpolation, supplied landmarks and post-training probes. The AT caption defines each transformation and distinguishes structural deltas from learned improvement over the input.

The paired figures are reproducible LaTeX compositions of the existing verified vector panels; no data were reselected, recomputed, smoothed or omitted from the manuscript. All 17 pre-existing figure asset hashes match the previous verification snapshot. See `research/body-figure-verification-2026-09-07.json` for the revised source and PDF identities and final layout checks.

Final validation: 26 pages, eight body figures (1–8) and seven supplemental figures (9–15). All captions are present and sequential; references resolve with no overfull boxes. Revised Results/appendix pages were rendered and visually checked, including a repaired ablation caption page break. [Current supervisor PDF](../output/pdf/body-figures/conditional-iqp-supervisor-draft.pdf). This revision supersedes the earlier figure map and PDF link above; the earlier snapshot is retained for provenance.

Figure 1 placement follow-up: the original full workflow is now a top-of-page float on page 2. The current 26-page PDF is [here](../output/pdf/figure-one-top/conditional-iqp-supervisor-draft.pdf); [verification](research/figure-one-top-verification-2026-09-07.json). Additional experiment priorities are proposals only, recorded separately; no runs were launched.

## Estimator complexity follow-up

Added main Methods Table I and explicit arithmetic, memory and sample-complexity bounds for product, symmetry-only and bond-informed reference estimators, moments/full phase Jacobians, scalar MMD/gradients, and exact enumeration. The text distinguishes attainable support-based algorithms from current dense/AD implementation costs. The compilation dip was checked independently in 18 fresh processes on both hosts; it persists with persistent caching disabled, with its precise compiler cause unresolved. See [technical audit](research/reference-estimator-complexity-2026-09-07.md).

The updated draft has 27 pages and retains Figure 1 at the top of page 2. Changed Methods pages and Figure 1 were rendered and visually checked; citations/references resolve and there are no overfull boxes. Existing empirical figures/results retain their scope. [Current supervisor PDF](../output/pdf/estimator-complexity/conditional-iqp-supervisor-draft.pdf). The completed estimator accuracy/cost figures are standalone candidates pending integration review. This update supersedes the preceding current-PDF pointer; historical snapshots remain available.

## Ongoing data/ancilla follow-up

The author expanded the authorized n12 data-efficiency experiment to test interaction with0/2/4ancillas. Main Experimental Setup now includes the motivation, symmetric fixed-ancilla construction and paired interaction contrast, explicitly labeled ongoing. No unobserved outcome was inserted. The 27-page PDF builds with resolved citations/references and no overfull boxes; new subsection pages11–12 were rendered and reviewed. [Current draft](../output/pdf/data-ancilla-plan/conditional-iqp-supervisor-draft.pdf). Empirical integration follows the frozen210-bundle gate; see [launch verification](research/ashkin-teller-data-efficiency-launch-2026-09-07.json).

## Completed data/ancilla integration, 2026-09-08

The ongoing subsection is now a completed Methods protocol, with new Results and Discussion and a six-panel main-body Figure 9. All 180 IQP fits and 30 classical bundles were sealed and independently audited before final interpretation. Both primary interaction intervals include zero; low-data bond variability, classical superiority with more data, fixed validation access and growing active phase count remain explicit. No seeds were excluded or scientific settings changed.

The current draft has 28 pages, nine body figures and seven supplemental figures. Figure 1 remains at the top of page 2. Figure PDF/PNG and changed manuscript pages were visually reviewed; the build resolves citations/references with no overfull boxes. The prior ongoing draft remains a historical snapshot. [Current supervisor PDF](../output/pdf/data-ancilla-results/conditional-iqp-supervisor-draft.pdf); [result](research/ashkin-teller-data-ancilla-result-2026-09-08.md); [verification](research/ashkin-teller-data-ancilla-integration-2026-09-08.json). Scientific author review, authorship/funding confirmation and public deposition remain open. No commit, push or external sending.

## Final adversarial review, 2026-09-08

A substantive resource-count inconsistency was corrected: ancilla-only phases are inactive for visible marginals, so stored coordinates must not be counted as added visible modeling capacity. Also corrected the classical denominator for variable training size, abstract/appendix scope, approximate-sampling assumptions and seed/ablation uncertainty wording. The finite-variance estimator argument and all current experimental conclusions survive the review. Independent algebra/numerical checks and complete data/ancilla artifact verification passed.

[Full adversarial findings](research/manuscript-adversarial-review-2026-09-08.md), [verification](research/manuscript-adversarial-verification-2026-09-08.json), and [current 29-page PDF](../output/pdf/adversarial-review/conditional-iqp-supervisor-draft.pdf). Figure 1 remains atop page 2; figure data are unchanged and main figures remain in the body. The corrected draft is ready for supervisor review, with novelty assessment, authorship/funding and public reproduction still requiring author action.
