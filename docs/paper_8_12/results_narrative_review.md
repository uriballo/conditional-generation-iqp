# Results narrative review

## Outline and paragraph roles

1. Opening: define the three evaluation questions and the complete evidence set.
2. Gaussian evidence: establish the translated target, quantify ancilla effects, compare interpolation and oracle fits, then use profiles to explain residual error.
3. Curie--Weiss evidence: describe the shape change, quantify the ancilla benefit, separate non-landmark interpolation from landmark realization, and interpret the direct-fit comparison.
4. Ablation: distinguish the input state, condition map, and ancilla contributions; identify the product--affine model's remaining bimodality error.
5. Scope: summarize which interpolation advantages the data support and explain why the held-out-minus-training gap is not a matched-condition causal test.

The revised paragraphs are in `Overleaf/main.tex`, Results. Each paragraph opens with its evidence or limitation, followed by the corresponding numbers or figure references. Existing profile figures remain in the appendix and are cited where they explain the main-text comparisons.

## Claim-evidence map

This is the canonical claim table for the current Results handoff, not a new numerical analysis. Production claims use the frozen manifest checksum in [provenance.json](provenance.json), which records 2,600 training/evaluation identities and their artifact hashes. Numerical comparisons come from [summary.json](summary.json) and [seed_summary.csv](seed_summary.csv); reproduction and raw-store locations are in [README.md](README.md). Development ablations and supplementary midpoint probes are separate evidence and must not inherit production verification merely by appearing here.

The author approved delivery of the Results narrative at commit `53d36a46`; that delivery approval is distinct from independent scientific validation. The table records the existing review's support and limitations. This documentation update does not change claims or manuscript prose.

| Claim | Result artifact or figure evidence | Limitation and evidence-review status |
| --- | --- | --- |
| ancillas reduce median held-out raw TVD for both sizes of both datasets. | five-point ladders in `gaussian_results.pdf` and `curie_weiss_results.pdf`; paired seed summaries in `summary.json`. | supported for aggregate medians, not every fit. |
| Gaussian shared interpolation beats the independent bank at n=8, r=10, but the n=12 advantage is unresolved. | paired advantages 0.0198 [0.0173, 0.0211] and 0.0049 [-0.0077, 0.0137]. | supported. |
| Gaussian residuals include endpoint reconstruction errors. | training-condition profiles at c=0 and c=1 and condition-resolved raw TVD. | supported; attribution to encoding versus optimization remains unresolved. |
| midpoint profiles illustrate differences between shared predictions and output mixtures. | three post-training probes at each Gaussian size. | supported as qualitative evidence, not a replacement headline test or proof of within-shell accuracy. |
| CW improves beyond the supplied landmark. | non-landmark r=10 paired advantages 0.0119 [0.0080, 0.0136] and 0.0363 [0.0299, 0.0433], positive in all ten seeds. | supported within the prescribed map and training range. |
| ancillas improve the CW shape, while twelve-qubit ordered residuals remain. | magnetization profiles at beta=0.6, 1, 1.4, especially alternating-shell errors and ordered edge/centre mass. | supported; profile aggregation does not certify raw accuracy. |
| the input symmetry and map interact. | retained three-seed crossed ablation; symmetry alone under the affine map does not improve the reported average. | supported for the tested constructions, not a universal resource equivalence. |
| shared CW circuits exceed an oracle capacity bound. | direct fits use a finite budget and different condition-data access. | unsupported; explicitly excluded. |

Before changing a claim, identify its exact protocol/result path and figure or table, retain the relevant limitations, and record the new author-review status. The retained three-seed crossed ablation's protocol/artifact pointer was resolved in the 2026-09-07 final supervisor review: `docs/research/curie-ablation-screen-verification-2026-09-07.json` records all 24 identities and hashes. It remains development evidence. Do not infer quantum advantage, extrapolation, or an oracle capacity bound from this evidence.

## Five-dimension self-review

- Contribution: pass for a conditional fixed-unitary construction and its tested response. No quantum advantage or general scaling claim is inferred.
- Writing: pass for progression from distributions to errors to baselines and limitations. Removed repeated Gaussian statistics; retained consistent raw/profile and landmark/non-landmark distinctions.
- Experimental strength: pass for complete matched production seeds and paired estimates. Ablation evidence remains limited to three development seeds.
- Evaluation completeness: needs a later research decision if the paper claims applicability beyond exchangeable targets or within-range interpolation. The present Results section makes neither claim.
- Method soundness: pass for information boundaries and finite-budget baseline interpretation. The experiments vary circuit capacity and ancilla encoding together, so they do not isolate those mechanisms.

The Abstract, Introduction evidence paragraph, and Conclusions still need a separate author-approved revision. No new experiments or changes to target definitions, figures, or numerical summaries accompany this prose edit.

## Methods and contribution revision, 2026-09-07

User requested readiness issues 1 and 2. `Overleaf/main.tex` now derives the nonuniform-reference Pauli estimator, support-aware second-moment bound, fixed-encoder phase derivatives, and explicit MMD U-statistic/gradient. Abstract and Introduction distinguish the conditional interface from inherited CT-state, IQPopt, input-encoding, and MMD machinery. Production fixed mask banks and frozen-data qualifications are explicit. These are assistant-reviewed manuscript revisions pending author scientific review; existing Results delivery status is unchanged.

| Claim | Evidence | Limitation / review status |
| --- | --- | --- |
| Conditional Pauli moments and phase derivatives have finite-variance additive estimators. | New `sec:ratio-estimator` derivation; implementation `src/iqp_cond/simulator.py`; exact enumeration check in `output/pdf/iqp-methods-review/check_ratio.py`. | Fixed encoder, efficient sampling/ratio access, polynomial diagonal, stated numerical accuracy. Assistant algebra/code review; author review pending. |
| U-statistic loss and phase gradient are unbiased at fixed parameters under the stated independent sampling. | New `sec:unbiased-conditional-mmd`; `src/iqp_cond/exact_tvd_runner.py::_weighted_loss_functions`, `src/iqp_cond/mmd.py`. | Full-kernel expectation averages over mask draws; production retains its bank and data. No adaptive population-gradient or convergence claim. |
| The contribution is a conditional sampling-and-ratio interface with fixed learned unitary and tractable constructions. | Revised Abstract/Introduction and cited primary works, including arXiv:2501.04776 and arXiv:2503.02934v2. | Positioning, not a proof of priority or a hardware speedup. Author review pending. |

Validation: 12 existing simulator/MMD tests passed. Exact checks covered 84 state/mask cases on N=2,3,4 with complex full, even-parity, and singleton support; maximum moment error 4.44e-16 and finite-difference gradient error 2.26e-10. Pair-U-statistic expectation identities and second-moment bounds passed. The PDF compiles with resolved citations/references; this remains a working draft with unrelated publication placeholders.

## Ashkin--Teller integration, 2026-09-07

The author approved integrating the completed third case. `Overleaf/main.tex` now includes correlated two-ring conditioning, the separate spatial two-input protocol, all-six-model results at n=8/12, and revised Abstract, Introduction, Discussion and Conclusions. The untested two-dimensional amplitude--phase construction and its ancillary references were removed. The original 2,600-fit Gaussian/Curie--Weiss identity, statistics and comparisons remain distinct from the 40-fit Ashkin--Teller confirmation/follow-up study. Earlier manuscript edits were retained.

| Claim | Protocol/result and manuscript evidence | Limitation / review status |
| --- | --- | --- |
| Correlated ring inputs realize the sampling-and-ratio interface | `docs/research/bond-reference-method.md`; `sec:two-ring-encoding`; `Overleaf/ashkin_teller_appendix.tex` | Controlled arithmetic and stated preparation/routing assumptions; assistant checked, author review pending |
| Shared IQP improves both supplied inputs at held-out coupling combinations | `docs/research/ashkin-teller-confirmation-result-2026-09-07.json` and `ashkin-teller-n12-result-2026-09-07.json`; `fig:at-summary`, `sec:at-results` | Compositional interpolation; n12 is a separate trained follow-up, no size transfer |
| Bond-informed input yields lower mean joint TVD under the matched within-size bank | Same frozen primary reports and paired contrasts | Complete encoder-family comparison, not topology-only causality; classical log-linear remains best |
| Output structure contributes beyond its own profiles/marginals | Both `ashkin-teller-*-structure-result` reports; `fig:at-structure` | Post hoc at n8, predeclared at n12; cat dependence negative in 13/90 n12 pairs; correlation and marginal exceptions retained |

Validation: compiled 26-page PDF at `output/pdf/ashkin-teller-integration/main.pdf`; no unresolved references, overfull boxes or stuck floats. Visually reviewed new Methods, protocol, Results/Discussion, combined figure, derivation, structural figures and tables. All 216 table entries match the source reports to displayed precision; 16 reference/preparation tests passed; table-generator Ruff and diff checks passed. Integration provenance is in `docs/research/ashkin-teller-manuscript-integration-2026-09-07.json`. The working draft retains pre-existing author/affiliation and release placeholders; this integration is not a submission-readiness certification. Scientific author review of the new prose remains pending.

## First supervisor draft review, 2026-09-07

[Final review and figure map](../supervisor-draft-review-2026-09-07.md). Removed the redundant Gaussian heatmap, placed the five main figures beside the corresponding narrative, and kept detailed profiles before the AT technical appendix. Corrected the generic checkpoint description (exact validation TVD for Gaussian/CW versus sampled validation MMD for AT), Gaussian oracle-gap terminology, and target-only layer-exchange covariance. Added the complete four-arm CW ablation table after verifying all 24 stored evaluation/probability hashes and protocol linkage.

| Claim / artifact | Evidence | Limitation / review status |
| --- | --- | --- |
| CW input/map interaction and ancilla benefit | `configs/curie_encoder_ablation_n8_screen_v1.json`; `docs/research/curie-ablation-screen-verification-2026-09-07.json`; Table II and Figure 4 | All 24 identities verified; three development seeds, not a production or causal symmetry-only claim. Assistant review complete, author review pending. |
| Complete first supervisor draft | `docs/research/supervisor-draft-verification-2026-09-07.json`; `output/pdf/supervisor-draft/conditional-iqp-supervisor-draft.pdf` | 26 pages compiled and visually reviewed; named authors retained, unknown-author/funding placeholders removed, archive status stated honestly. Scientific author review, authorship/funding confirmation and public deposition remain open. |

Earlier review entries describe historical states; the final review supersedes their unresolved-placeholder and ablation-pointer status. Frozen production/AT results and earlier provenance snapshots remain unchanged. No new experiments, commit, push or external sending accompanied this review.

## Main-body visual evidence revision, 2026-09-07

At the author's request, the main Results now include paired Gaussian reconstruction/interpolation panels (Figure 3), paired twelve-qubit CW shape/baseline panels (Figure 5), and both-size AT structure diagnostics (Figure 8). These complement the accuracy, ablation and primary AT comparison figures in the body. Complementary-size profiles and technical checks remain supplemental. Existing data, numerical claims and uncertainty definitions are unchanged; all 17 vector assets retain their verified hashes and occur once. The new compositions and captions were assistant-reviewed; scientific author review remains pending. Current artifact: `output/pdf/body-figures/conditional-iqp-supervisor-draft.pdf`; verification: `docs/research/body-figure-verification-2026-09-07.json`.

## Estimator precision/cost study, 2026-09-07

| Candidate claim | Frozen evidence | Scope / review |
| --- | --- | --- |
| Practical moment/phase-gradient error follows the expected sampling trend across the tested inputs | `artifacts/reference-estimator-study-2026-09-07/manifest.json`, `freeze.json`, `verification.json`; accuracy figure | 60 fixed cases, 8/12/16 qubits, no mask-bank variation; 54 synthetic and six trained checks kept separate. Assistant-verified; not yet added to manuscript. |
| The correlated reference has measurable computational overhead | Same raw timing records; cost figure | CPU, joint 16-moment/full-Jacobian calls, compilation separate, process RSS not estimator-only. No asymptotic or hardware-advantage claim. |

[Result and limitations](../research/reference-estimator-study-result-2026-09-07.md). The author selected this study before data efficiency; the latter remains deferred.

## Explicit estimator complexity and compilation audit, 2026-09-07

| Claim | Evidence | Scope / review |
| --- | --- | --- |
| All three references admit O(S A N^2) moment/full-phase-Jacobian arithmetic for the complete one-/two-body generator bank | Main Methods `sec:estimator-complexity`, `Overleaf/estimator_complexity.tex`; detailed derivation in `docs/research/reference-estimator-complexity-2026-09-07.md` | Attainable support-based arithmetic, controlled precision; current dense parities and vector-output AD have explicitly larger costs. Assistant algebra/code review; author review pending. |
| Product N=16 compilation dip is reproducible with persistent caching disabled | 18 fresh-process probes, both original hosts; `artifacts/reference-estimator-study-2026-09-07/compile-probe/summary.json` and HLO | Lowering/backend compilation separated; full Jacobian retained and estimated arithmetic rises. Exact compiler mechanism unresolved. No asymptotic compilation claim. |

The completed 240-bundle study remains frozen. The cost-axis label now explicitly includes lowering; no numerical observations changed. The 27-page supervisor draft includes the complexity section in the body, with Figure 1 still at the top of page 2. Empirical estimator figures remain standalone pending manuscript integration review; data efficiency is deferred.

## Data-efficiency × ancilla follow-up launched, 2026-09-07

| Claim or planned question | Evidence | Scope / review |
| --- | --- | --- |
| Fixed independent ancillas preserve both visible layer-flip symmetries and embed the r=0 model | `scripts/ashkin_teller_data_efficiency_model.py`; exact physical-input/zero-cross-coupling/flip and sampled moment/cross-gradient tests | New study-private reference construction; 78+12r active phases. Ancilla-only phases cancel under the trace. Assistant numerical/algebra review complete. |
| Does the benefit of ancillas change with training-set size? | `docs/research/ashkin-teller-data-efficiency-protocol-2026-09-07.md`; frozen manifest `artifacts/ashkin-teller-data-efficiency-2026-09-07/manifest.json`; launch record | Running: 180 IQP fits and30classical bundles, nvisible12, sizes100/300/1000 crossed withr0/2/4. No outcomes yet. |
| Paper includes the motivation and explicit paired interaction contrast | `sec:data-ancilla-followup`, `eq:data-ancilla-interaction`; `output/pdf/data-ancilla-plan/conditional-iqp-supervisor-draft.pdf` | Explicitly ongoing subsection; empirical plots/results/discussion will replace planned wording after complete verified evaluation. |

27 tests and all training-only engineering gates passed. Nested prefixes retain fixed1000validation examples/condition; initialization excludes unseen training suffixes. Both Strix hosts run four workers. Complete prediction seal precedes evaluation; a completion coordinator and hourly thread follow-up handle the remaining verified integration. Earlier frozen experiments remain reproducible from their original source archives; the new campaign records the small baseline shape and partial-epoch compatibility extensions. No commit, push or external sending.

## Data-size × ancilla study completed and integrated, 2026-09-08

This completion supersedes the preceding ongoing/deferred data-efficiency entries; their historical protocol and launch identities remain unchanged.

| Claim | Protocol/result evidence | Figure/table | Limitation / review |
| --- | --- | --- | --- |
| Neither encoder establishes increasing ancilla benefit with training data | Frozen data-efficiency protocol; all 420 sealed prediction files, 300 exact evaluation rows, report/summary.json and independent-interaction-check.json | Main Figure 9(e,f), Results V.E and Eq. (56) | Paired primary I: cat -0.0088 [-0.0219,0.0032], bond -0.0012 [-0.0768,0.0731]. Descriptive intervals; unresolved, not equivalence. Assistant-verified; author review pending. |
| Symmetry-only ancilla benefits are modest; bond means nearly coincide at 300/1,000 examples | Same complete ten-seed evidence; every fit retained | Figure 9(a,b,e,f) | Low-data bond failures occur in different seeds and widen interaction uncertainty. Phase count also grows; no latent-only attribution. |
| Classical log-linear improves more strongly with data and wins at 300/1,000 examples | Same paired data and validation access; exact joint and connected-bond metrics independently recomputed | Figure 9(a–d) | Different classical parameterization; no quantum advantage claim. Fixed 16,000 extra validation labels and 500 IQP updates; same-domain interpolation only. |

[Result and regeneration](../research/ashkin-teller-data-ancilla-result-2026-09-08.md). Empirical Methods/Results/Discussion replace the ongoing wording. Current PDF: `output/pdf/data-ancilla-results/conditional-iqp-supervisor-draft.pdf`; verification: `docs/research/ashkin-teller-data-ancilla-integration-2026-09-08.json`. The separate empirical estimator precision/cost figures remain standalone pending their own integration review.

## Adversarial review corrections, 2026-09-08

| Claim | Evidence | Manuscript location | Limitation / review |
| --- | --- | --- | --- |
| Ancilla-only phases do not affect any visible marginal | Partial-trace/commutation argument; 20 independent complex-reference checks, maximum change 1.39e-16 | Ancilla-assisted conditioning; scalar configuration table; CW resource comparison | Stored P_N differs from visible-affecting directions; at most n(n+1)/2+nr before extra symmetries. Historical frozen banks/results retained. Assistant-verified. |
| The ratio estimator requires controlled sampling and evaluation | Derivation, 300 moments and 6,300 derivatives against an independent dense oracle | Proposition 1 and precision paragraph | Exact sampling assumed; RMS arithmetic error controlled separately. No arbitrary approximate-sampler guarantee. |
| Statistical and comparator scope differs across campaigns | exact_tvd_runner.prepare_data_archive; frozen AT protocols; baseline runner | Scalar statistics, CW ablation, classical appendix | Fixed-data scalar seeds versus fresh-data AT seeds; three-seed ablation descriptive; histogram denominator is s+Kλ. |

[Review](../research/manuscript-adversarial-review-2026-09-08.md) and [verification](../research/manuscript-adversarial-verification-2026-09-08.json). The current 29-page supervisor PDF is `output/pdf/adversarial-review/conditional-iqp-supervisor-draft.pdf`; older snapshots retain their original evidence. No results or figure data changed. Novelty, author review and public deposition remain open; no extra experiment is required for the existing bounded claims.
