# Scalar-Gaussian paper review

## Reverse outline

- The target family supplies a controlled moving-mean problem with exact
  complement structure.
- The paired encoder uses that structure without inserting the target mean or
  width.
- Complement covariance follows from the encoder relation and the commuting
  IQP generators.
- Three condition protocols distinguish anchor count from symmetry transfer.
- Condition-specific circuits measure the accuracy and resource cost avoided
  by sharing one parameter vector.
- An encoder-only diagnostic measures how much the learned IQP unitary adds to
  the fixed paired input map.

## Adversarial review

| Dimension | Assessment | Required evidence or revision |
| --- | --- | --- |
| Contribution | Pass for the formulation; empirical contribution pending. | The two-condition Gaussian and physics dataset must carry claims beyond a controlled validity test. |
| Writing clarity | Pass for the scalar formulation and protocol. | Define the eventual physics dataset before revising the abstract around all three datasets. |
| Experimental strength | Pending. | Complete all ten seeds and three restarts; report the direct supervised ceiling separately from anchor-only interpolation. |
| Evaluation completeness | Protocol complete. | Generate the result figure, resource table, ancilla appendix, and paired bootstrap intervals after the campaign. |
| Method soundness | Pass with a stated limitation. | Report within-shell residual because the paired encoder breaks full wire exchangeability. Do not claim that ancillas help unless the production matrix supports it. |

## Claim controls

- Treat `half3` upper-half performance as symmetry transfer, not extrapolation.
- Call the direct per-condition bank a supervised ceiling because it trains on
  held-condition target data.
- Call anchor interpolation a classical output mixture, not one quantum
  circuit.
- Use exact bitstring TVD for final accuracy and validation MMD only for model
  selection.
- Keep the anti-phase encoder fixed throughout the scalar campaign.
