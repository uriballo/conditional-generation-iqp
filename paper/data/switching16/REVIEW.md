# Review of the conditional switching draft

The author selected the completed sixteen-qubit triangle study for the empirical symmetry section. The separate parity construction remains a methods draft without fitted Curie–Weiss results. The newer HPO results await author review and do not enter this revision. No new training or model evaluation was necessary.

## Paragraph roles and reverse outline

1. Motivation and target: define interacting triangle products and vertex switches.
2. Encoder: explain the shared-root RX/CNOT motif, final Hadamards, condition map and direct reference access.
3. Architectural guarantee: derive switching invariance and distinguish it from the two-angle simplification.
4. Finite-data evidence: specify training-pool size and exact selection, then report five fresh final fits and physical TVD.
5. Conditional response: define the two graph-order observables and identify the post-selection range extension.
6. Scope: report input sensitivity, saturation and fluctuation limitations, and the stronger fixed classical control.

## Claim and evidence map

| Claim | Evidence | Status |
|---|---|---|
| Exact conditional switching invariance | Even-incidence cycle reference, Hadamard conjugation and commutation with X-generated gates; stored encoder validation covers all 64 switches | Supported for arbitrary learned IQP phases |
| Tractable correlated reference access | Explicit ten-bit sampling and amplitude-ratio formula in the appendix | Supported for this construction |
| Finite-data conditional generation with two angles | Fifteen complete fit receipts, frozen configuration and five independent final fits; original twelve-condition mean TVD 0.1780 ± 0.0013 | Supported within the documented exploratory protocol |
| Response beyond the fitting interval | Frozen 121-point sweep through beta 2.4, ten added extrapolation conditions, separate range-extension summary | Supported as a post-selection diagnostic |
| Conditional input affects selected models | Frozen-input TVD 0.4886 ± 0.0011 compared with native 0.1780 ± 0.0013 | Supported sensitivity check; no optimized blind-model comparison |
| Recovery of full Gibbs thermodynamics | Large-beta fluctuation discrepancy in the appendix | Not established |
| Classical fidelity or computational advantage | Fixed classical control TVD 0.1134 and exact native quotient evaluation | Not claimed |
| Parity-input Curie–Weiss fitting performance | No sampled-training results for that construction | Not claimed; methods remain separate |

## Five-dimension self-review

- Contribution: the section demonstrates a conditional symmetry construction using correlated input phases and a simple learned layer. It does not claim a new commuting-symmetry theorem or an advantage over classical enforcement.
- Clarity: the circuit motif precedes the invariance argument; shared root wires and the single final Hadamard layer are explicit. Experimental settings appear in the appendix; numeric seed identifiers and software provenance remain in the research records.
- Experimental strength: all five final fits remain in the summaries. The main response visibly saturates below the target, and the text reports moderate TVD rather than describing complete recovery.
- Evaluation completeness: the original report, dense sweep and added extrapolation conditions have distinct roles. Exact development distributions contribute information beyond the finite target pools. Both plotted observables summarize the same triangle energy, rather than independent capabilities.
- Method soundness: the appendix describes the circuit, estimator and quotient validation. Detailed validation outputs, source identities and excluded interrupted runs remain in the research records. A new stored-array audit independently recomputes TVD and the plotted moments without fitting or evaluating circuits.

## Verification and layout

The primary agent reviewed the renderer sources and regenerated figure provenance. The stored-array audit passed sixteen checks, including normalization, original-grid preservation, per-fit fluctuations, reporting TVD, all table aggregates, endpoint rounding and figure/source hashes. Ruff passed for the figure sources. LaTeX resolved references and produced the section, full-manuscript and separate parity review PDFs. Visual inspection found and corrected the motif label and table width. The integrated manuscript retains nonfatal RevTeX deferred-float warnings; the affected figures appear legibly on the reviewed pages. Final journal layout remains a later editing step.

The input arrays and figure receipts reside in this directory. The detailed audit and separate review PDFs reside in `results/symmetry-draft-20261009/`. This is an author-review draft, not a publication or HPO adoption.

## Academic prose cleanup

The manuscript and active appendix no longer contain source hashes, package versions, numeric seed IDs, RNG offsets, precision types, run receipts or interrupted-run bookkeeping. The Gaussian architecture account now describes the scientific comparisons rather than the chronological fit history. Experimental budgets, independent-run counts, selection rules, aggregation, controls and limitations remain. Provenance stays in the original result records and ignored pre-cleanup source copies. Both PDFs were rebuilt with resolved references and no overfull boxes; affected appendix pages were visually reviewed. The existing nonfatal RevTeX float warnings remain. No experimental or figure data changed.
