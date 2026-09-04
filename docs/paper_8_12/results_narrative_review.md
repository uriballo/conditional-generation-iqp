# Results narrative review

## Outline and paragraph roles

1. Opening: define the three evaluation questions and the complete evidence set.
2. Gaussian evidence: establish the translated target, quantify ancilla effects, compare interpolation and oracle fits, then use profiles to explain residual error.
3. Curie--Weiss evidence: describe the shape change, quantify the ancilla benefit, separate non-landmark interpolation from landmark realization, and interpret the direct-fit comparison.
4. Ablation: distinguish the input state, condition map, and ancilla contributions; identify the product--affine model's remaining bimodality error.
5. Scope: summarize which interpolation advantages the data support and explain why the held-out-minus-training gap is not a matched-condition causal test.

The revised paragraphs are in `Overleaf/main.tex`, Results. Each paragraph opens with its evidence or limitation, followed by the corresponding numbers or figure references. Existing profile figures remain in the appendix and are cited where they explain the main-text comparisons.

## Claim-evidence map

- Claim: ancillas reduce median held-out raw TVD for both sizes of both datasets. | Evidence: five-point ladders in `gaussian_results.pdf` and `curie_weiss_results.pdf`; paired seed summaries in `summary.json`. | Status: supported for aggregate medians, not every fit.
- Claim: Gaussian shared interpolation beats the independent bank at n=8, r=10, but the n=12 advantage is unresolved. | Evidence: paired advantages 0.0198 [0.0173, 0.0211] and 0.0049 [-0.0077, 0.0137]. | Status: supported.
- Claim: Gaussian residuals include endpoint reconstruction errors. | Evidence: training-condition profiles at c=0 and c=1 and condition-resolved raw TVD. | Status: supported; attribution to encoding versus optimization remains unresolved.
- Claim: midpoint profiles illustrate differences between shared predictions and output mixtures. | Evidence: three post-training probes at each Gaussian size. | Status: supported as qualitative evidence, not a replacement headline test or proof of within-shell accuracy.
- Claim: CW improves beyond the supplied landmark. | Evidence: non-landmark r=10 paired advantages 0.0119 [0.0080, 0.0136] and 0.0363 [0.0299, 0.0433], positive in all ten seeds. | Status: supported within the prescribed map and training range.
- Claim: ancillas improve the CW shape, while twelve-qubit ordered residuals remain. | Evidence: magnetization profiles at beta=0.6, 1, 1.4, especially alternating-shell errors and ordered edge/centre mass. | Status: supported; profile aggregation does not certify raw accuracy.
- Claim: the input symmetry and map interact. | Evidence: retained three-seed crossed ablation; symmetry alone under the affine map does not improve the reported average. | Status: supported for the tested constructions, not a universal resource equivalence.
- Claim: shared CW circuits exceed an oracle capacity bound. | Evidence: direct fits use a finite budget and different condition-data access. | Status: unsupported; explicitly excluded.

## Five-dimension self-review

- Contribution: pass for a conditional fixed-unitary construction and its tested response. No quantum advantage or general scaling claim is inferred.
- Writing: pass for progression from distributions to errors to baselines and limitations. Removed repeated Gaussian statistics; retained consistent raw/profile and landmark/non-landmark distinctions.
- Experimental strength: pass for complete matched production seeds and paired estimates. Ablation evidence remains limited to three development seeds.
- Evaluation completeness: needs a later research decision if the paper claims applicability beyond exchangeable targets or within-range interpolation. The present Results section makes neither claim.
- Method soundness: pass for information boundaries and finite-budget baseline interpretation. The experiments vary circuit capacity and ancilla encoding together, so they do not isolate those mechanisms.

The Abstract, Introduction evidence paragraph, and Conclusions still need a separate author-approved revision. No new experiments or changes to target definitions, figures, or numerical summaries accompany this prose edit.
