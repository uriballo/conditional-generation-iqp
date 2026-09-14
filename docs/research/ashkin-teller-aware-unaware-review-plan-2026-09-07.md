# Review framing: symmetry-only and bond-informed IQP

2026-09-07. Reviewer assessment and revised comparison plan, following the
[bond-aware ablation](ashkin-teller-bond-redesign-ablation-2026-09-07.md).
The user requests both aware and unaware cases. No new training was run for
this assessment, and no existing experiment identity or result is changed.

## Assessment

The bond-aware construction is scientifically defensible as a physics-informed
input family within the conditional IQP framework. It is not yet supported as a
confirmed generalization experiment, and reviewer acceptance cannot be guaranteed.
The paper does not need quantum or classical predictive advantage. It does need
an accurate account of supplied information, what the trained unitary adds, and
the costs and assumptions behind the tractable input interface.

The central objection is substantive: with g(a)=a and g(b)=b, the input
measurement law is exactly the product of two uncoupled Ising-ring distributions.
This supplies topology, functional dependence on the couplings, and the entire
kappa=0 probability law. It is more information than just Z2 x Z2 symmetry.
At nonzero kappa these are not generally the true target layer marginals.

The input does not supply the cross-layer interaction and has zero connected
interlayer bond correlations. The existing gain from encoder-only TVD .2575
to trained TVD .0970 is evidence that the IQP layer contributes. However, the
conditional response is partly encoded, and some correlation errors worsen.
Do not describe this as recovering unknown physics, learning an arbitrary
conditional law from scratch, or obtaining uniformly better correlations.

## Primary comparison and terminology

Use **symmetry-only** and **bond-informed**, rather than an unqualified
"unaware" label. Both know the layer partition, independent-flip symmetry,
condition coordinates and their chosen condition maps. Neither is prior-free.

| Case | Encoder information | Shared circuit | Question |
| --- | --- | --- | --- |
| Symmetry-only | Two-block cat, fixed generic affine angle map; no ring adjacency or Gibbs factor | Same complete two-local bank | Can the method learn spatial joint structure using only symmetry in the encoder? |
| Bond-informed | Product of complex coherent Ising-ring states; supplied ring and g(c)=c; no kappa | Same complete two-local bank | What does additional physical structure in the input contribute? |

The two-local generator bank is permutation-covariant and does not single out
ring neighbors. Use the same 30 independent phase coordinates at n=8, data,
initialization, loss, optimization budget, and checkpoint selection. The input
states and condition maps differ intentionally; call the result a comparison of
encoder families/information, not an isolated effect of topology alone.

This clean comparison already exists in the development ablation: mean TVD
.1247 for symmetry-only and .1055 for bond-informed, approximately 15% lower for
the informed family. These are exact-population fits on all 11 development
conditions, not held-out scores or evidence of sampled training.

Retain the full 2x2 encoder/circuit table as a secondary ablation. Aligned
four-body generators supply problem-specific bond information even with the cat
input; that combination must not be labelled wholly bond-unaware. Comparing
cat/two-local .1247 directly with bond/four-body .0970 changes both components
and cannot isolate the encoder effect. The combined model can still be reported
as the best development configuration with that scope made explicit.

A completely generic product-input control is optional if the paper makes a
separate claim about the benefit of imposing symmetry. It is not required just
to compare bond-informed and bond-uninformed encoders. Any such new control must
use the same eventual protocol; the old axis-only experiment is not a matched
control for the all-condition redesign ablation.

## Evidence needed for the paper

- Give the encoder formula and an information inventory: layer symmetry and
  partition for both, ring adjacency and uncoupled Gibbs law for the informed
  case, aligned bond interactions additionally for the four-body circuit.
  Explain fixed eta=pi/8 as a design choice rather than a fitted physical quantity.
- Include each encoder at zero learned phases. Report joint TVD and connected
  cross-layer bond statistics before and after training, with per-condition
  results. These distinguish the supplied uncoupled structure from learned
  interlayer dependence. Keep kappa nonzero in the main task.
- Demonstrate actual sampled-MMD training for both primary cases. Successful
  moment/gradient checks at a few phase vectors do not establish that an entire
  optimization trajectory works. Preserve the current gate failures; a later
  protocol may use revised, scientifically justified criteria declared in advance.
- Freeze maps, phase choices, architecture and optimization rules before a new
  confirmatory split. The inspected interpolation, extrapolation and redesign
  points are development evidence. Do not relabel them as untouched tests.
- Derive sampling and amplitude ratios for the state after the initial Hadamards,
  state numerical/support assumptions, and describe physical state preparation
  and resource costs. Efficient classical reference sampling is not itself an
  implemented quantum preparation circuit. The existing finite-size checks are
  supportive, not a general precision or resource demonstration.
- Report classical baselines as context under the same data access. They need
  not lose. The claim is a conditional quantum construction with tested behavior,
  not superiority over all classical alternatives.

If the paper specifically attributes a gain to correct bond topology, add a
more tightly matched secondary control: the same bond-informed family, g(c),
eta and two-local circuit, but a predeclared incorrect labelled ring. This keeps
the encoder form and preparation resources similar while changing alignment
with target bonds. It is a wrong-topology control, not an unaware model; at four
sites different cycles necessarily overlap substantially, limiting its strength.
It is unnecessary if we only claim an encoder-family comparison.

## Proposed claim

"We demonstrate conditional IQP generation with symmetry-only and spatially
informed tractable inputs. The informed construction supplies uncoupled
single-layer structure, while one shared learned IQP unitary models interlayer
dependence. We report the accuracy and limitations of both choices."

This wording is a target framing, not a statement that sampled training and
confirmatory generalization have already been completed.

| Claim | Evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Both informed and symmetry-only configurations have development fits | Existing 2x2 exact-loss ablation | Linked ablation table | Not sampled training or fresh generalization | Verified development evidence |
| Correct topology alone causes the observed improvement | Encoder families differ in several ways | None | Requires a matched topology control for this narrower claim | Not isolated |
| Learned IQP phases add interlayer dependence beyond the informed input | Encoder-only vs trained diagnostics | Linked ablation table | Improvement not uniform across conditions | Supported at tested development settings |
| Bond-informed case is ready for a confirmatory paper claim | Pending sampled/confirmatory protocol and resource account | None | Acceptance and future results cannot be presumed | Incomplete |
