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

Before changing a claim, identify its exact protocol/result path and figure or table, retain the relevant limitations, and record the new author-review status. The retained three-seed crossed ablation's immutable protocol/artifact pointer is not established by this table; resolve it before revising that claim. Do not infer quantum advantage, extrapolation, or an oracle capacity bound from this evidence.

## Five-dimension self-review

- Contribution: pass for a conditional fixed-unitary construction and its tested response. No quantum advantage or general scaling claim is inferred.
- Writing: pass for progression from distributions to errors to baselines and limitations. Removed repeated Gaussian statistics; retained consistent raw/profile and landmark/non-landmark distinctions.
- Experimental strength: pass for complete matched production seeds and paired estimates. Ablation evidence remains limited to three development seeds.
- Evaluation completeness: needs a later research decision if the paper claims applicability beyond exchangeable targets or within-range interpolation. The present Results section makes neither claim.
- Method soundness: pass for information boundaries and finite-budget baseline interpretation. The experiments vary circuit capacity and ancilla encoding together, so they do not isolate those mechanisms.

The Abstract, Introduction evidence paragraph, and Conclusions still need a separate author-approved revision. No new experiments or changes to target definitions, figures, or numerical summaries accompany this prose edit.
