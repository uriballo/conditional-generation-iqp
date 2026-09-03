# Revision ledger for the clean Overleaf manuscript

Reviewed source: `Classically_trainable_conditional_quantum_generative_models-2.pdf`
and its pasted LaTeX source, received 2026-09-01. This ledger does not modify the
canonical manuscript.

## Intended paper story

The paper asks whether a condition-indexed family can be represented by one
classically trainable IQP Born machine while keeping its learned diagonal
unitary fixed. The condition enters through a tractable input state. The
experiments should then answer three questions:

1. Can a shared circuit interpolate to unseen conditions?
2. Do ancillas reduce the cost of sharing one parameter vector across the
   family?
3. Does the method retain this behavior when the target changes shape near a
   critical point and on an applied MFC distribution?

## Stable terminology and baselines

- **Conditioned model:** one parameter vector trained jointly at the training
  conditions. A held-out condition is supplied through the input-state encoder;
  no data from that condition enter training or checkpoint selection.
- **Independent bank:** one circuit per training condition, initialized and
  trained only with samples from that condition.
- **Adjacent-independent interpolation:** for a held-out condition, linearly
  interpolate the output probabilities of the nearest independent circuits.
  This is the data-free generalization baseline.
- **Target-aware oracle:** an independent circuit trained directly at the
  held-out condition. It is a capacity reference, not a generalization method.
- **Generalization advantage:**
  `TVD(interpolated independent) - TVD(conditioned)`. Positive values favor the
  conditioned model.
- **Sharing cost:** at a training condition,
  `TVD(conditioned) - TVD(independent)`. This isolates the cost of sharing.
- **Gap to target-aware oracle:** at a held-out condition,
  `TVD(conditioned) - TVD(oracle)`. This includes both parameter sharing and the
  absence of direct training data, so it should not be called a pure sharing
  cost.

## Introduction

### Keep

- The opening question is sharp: retain classical IQP training while keeping
  the diagonal unitary fixed across conditions.
- The distinction between classical observable/loss/gradient estimation and
  classical output sampling is necessary and well placed.
- The scope limitations are responsible and should remain in the paper.

### Revise

- Repair the abstract comma splice after “commuting structure” and replace its
  provisional results only after the campaign is complete.
- Fix the unresolved `Figure ??` reference to the pipeline figure.
- Replace the old experimental paragraph. It currently refers to a small
  accuracy penalty, an unconditioned control, and the previous Curie--Weiss
  protocol. The new comparisons are ancilla effect, adjacent-independent
  interpolation, and target-aware oracle.
- Keep one short scope sentence in the introduction. Move the detailed list of
  non-claims to the Discussion so it does not interrupt the contribution.
- End with a compact contribution list or an equivalent paragraph that separates
  the model, the classical conditional-MMD estimator, and the empirical
  interpolation study.

## Conditional IQP model

### Core construction

Open with a direct execution path: encode `c` in a tractable state, apply the
fixed shared IQP unitary, and measure the visible register. Define the sharing
constraint immediately. Repair the malformed ket and amplitude expressions in
the paragraph following Eq. (3).

### Product-state encoder

- Keep the general product-state definition and the finite-Fourier-response
  observation in the Method.
- Move the exact affine angle map to Experimental Setup.
- Present the map as a fixed design choice, not an optimal encoding, unless an
  encoding sensitivity experiment supports that claim.

### Ancillas

Replace “Purified product-state conditioning” with “Ancilla-assisted
conditioning.” The production campaign prepares a joint product input, applies
a complete two-local IQP circuit to visible and ancilla qubits, measures only
the visible register, and sums over ancilla outcomes. It does not instantiate
the draft's general input purification with explicit weights `w_c(u)`.

The section should distinguish two effects without claiming to isolate them:
ancillas enlarge the joint circuit and visible--ancilla interactions allow the
visible marginal to become mixed.

### Symmetry-informed encoder

- Begin with the concrete Curie--Weiss spin-flip relation before introducing a
  general group action.
- Call Eq. (14) a spin-flip-symmetrized cat-like state, not a GHZ state in
  general. It is GHZ only in special cases.
- Move the general permutation discussion and the Hamming-class tractability
  example to an appendix unless they are used directly by the final encoder.
- State the actual sampling and amplitude-ratio formulas used by the frozen
  Curie--Weiss encoder. Symmetry alone does not establish classical
  tractability.

## Conditional IQP training

- Keep the population MMD and condition-balanced objective.
- Add the character expansion that maps the Hamming-RBF MMD to Pauli-Z moments.
- Explain the three bandwidths through their expected mask bodyness.
- State the nonidentity-mask normalization and exact duplicate compression.
- Correct the minibatch TODO. In the production campaign, every update uses 100
  examples from every training condition and averages their losses. Conditions
  are not visited one at a time.
- Separate optimization from evaluation explicitly: MMD is the differentiable
  training loss; exact visible-distribution TVD is the primary scientific
  metric.

## Experimental setup

This section is the largest missing block. Add four subsections:

1. **Targets and splits:** Gaussian, Curie--Weiss, and MFC conditions, with
   disjoint 1,000/1,000/1,000 train/validation/test sets.
2. **Models and baselines:** conditioned model, independent bank,
   adjacent-independent interpolation, and target-aware oracle.
3. **Training protocol:** `n={8,12,16}`, `r={0,1,3,6,10}`, ten seeds, complete
   two-local circuits, data initialization, 2,000 updates, 8,192 estimator
   samples, 2,000 character operators, three bandwidths, and Adam-cosine.
4. **Evaluation protocol:** exact validation-TVD checkpoint selection, exact
   test TVD, complementary profile/response metrics, and paired seed-level
   bootstrap intervals.

Update the target-family figure. The current version still shows Gaussian
`n=19` and a 16-bin MFC pseudorapidity target. Production uses Gaussian and
Curie--Weiss `n={8,12,16}`, plus the three-visible-qubit MFC energy-fraction
target with eight physical bins.

## Results

Replace the current provisional figures rather than editing their captions:

1. **Gaussian validity and interpolation:** exact-TVD ancilla ladders, held-out
   conditioned versus adjacent-independent interpolation and oracle, and one
   response plot showing the translated family.
2. **Curie--Weiss critical interpolation:** exact bitstring and magnetization
   results at `beta=1`, the `{0.8,1.0,1.2}` critical window, and the full
   held-out average. Show whether ancillas recover the bimodal transition.
3. **MFC applied case:** energy-fraction distributions and held-out 125 GeV
   comparison, with exact histogram TVD and Wasserstein distance.

The old `n=19` Gaussian panel, `r=5` Curie--Weiss panel, no-condition baseline,
and pseudorapidity MFC panel do not match the production campaign.

### Preliminary Gaussian n=8 evidence

The complete 700-fit `n=8` slice currently supports only a provisional claim.
Median held-out conditioned TVD decreases from 0.219 at `r=0` to 0.102 at
`r=10`. At `r=10`, the conditioned model beats adjacent-independent
interpolation by 0.020 TVD and remains 0.016 above the target-aware oracle. At
`r=0`, it is 0.070 worse than interpolation. This sign change is preliminary
evidence that ancillas improve shared conditional generalization.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| The condition can enter through a tractable input state while the IQP unitary remains fixed. | Model definition and circuit identity across conditions. | Supported theoretically. |
| Conditional MMD and its gradients remain classically estimable under the sampling/amplitude-ratio interface. | Character expansion and estimator derivation. | Needs complete derivation in the draft. |
| One shared circuit interpolates to unseen conditions. | Exact held-out TVD across all datasets and sizes. | Preliminary for Gaussian `n=8`; campaign pending. |
| Ancillas reduce the generalization and oracle gaps. | Paired ancilla ladders over ten seeds. | Preliminary for Gaussian `n=8`; campaign pending. |
| The method follows the Curie--Weiss transition near `beta=1`. | Exact critical profiles and critical-window TVD. | Pending. |
| The method handles the applied MFC conditional response. | Held-out 125 GeV energy-fraction experiment. | Pending. |

## Rendered-draft issues

- Red TODOs remain visible in the abstract, Methods, training section, and a
  result caption.
- Cross-references render as `Figure ??` and `Section ??`.
- Page 5 is largely empty because Experimental Setup, Results, Discussion, and
  Conclusions contain almost no prose.
- The bibliography appears before two deferred result figures. This should
  resolve once Results contains text and the obsolete floats are replaced.
- Figure 1 is a strong overview and should remain the conceptual anchor.
- Figure 2 is visually clear but scientifically stale relative to the frozen
  campaign.

## Recommended editing order

1. Correct the stable model definition, ancilla subsection, and training
   protocol now.
2. Write the complete Experimental Setup from the frozen manifest.
3. Define the three baselines and the two gap metrics before presenting results.
4. Add validated result blocks as each complete dataset-size slice arrives.
5. Rewrite the experimental paragraph in the Introduction, Abstract,
   Discussion, and Conclusion only after the full campaign is aggregated.
