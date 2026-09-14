# Ashkin–Teller joint-coupling extrapolation design

Design proposed 2026-09-07 and subsequently executed in the
[joint-extrapolation screen](ashkin-teller-joint-extrapolation-result-2026-09-07.md).
The IQP gates failed in all three seeds; stronger classical baselines performed
better. The original proposed settings below are retained as design provenance.
This proposal follows the [first pilot](ashkin-teller-pilot-2026-09-07.md).
Numbers below describe the original design; the linked result contains the
frozen executable protocol and measured performance. Scientific review remains pending.

## What the next test should establish

Can one shared two-block IQP model predict a joint high-coupling regime after
seeing each coupling separately, and does its inductive bias help under a
finite sample budget? Distinguish representation/generalization ability from
predictive superiority over classical models. Neither requires or establishes
quantum advantage. Avoid choosing a split merely to disadvantage arithmetic
histogram interpolation.

There is an exact shortcut in the current target family. Write

p_ab(x) = exp[a A(x) + b B(x) + kappa C(x)] / Z_ab,

where A and B sum the layer bonds and C sums their products. At fixed kappa,
for any anchor (a0,b0), direct cancellation gives

p_ab(x) proportional to p_(a,b0)(x) p_(a0,b)(x) / p_(a0,b0)(x).

Normalizing the right side recovers the target exactly, including outside the
training convex hull. Similarly, normalized geometric combinations reproduce
any affine combination of natural parameters when the coefficients sum to one.
This is our algebraic deduction for the benchmark; the general exponential-family
form is covered by Wainwright and Jordan, section 3 of
[Graphical Models, Exponential Families, and Variational Inference](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf).
The exact identity requires exact positive distributions and fixed kappa;
finite-sample density ratios can be noisy. Increasing system size or widening
the coupling range does not remove the identity.

## Proposed split

Keep the four-site-per-layer ring, eight visible qubits, kappa=0.3, independent
layer-flip symmetry, zero ancillas, and complete 36-parameter IQP ansatz.
Use coupling levels {0.10, 0.30, 0.50, 0.70}, anchor a0=b0=0.10.

Train at seven points: (a,0.10) for all four levels and (0.10,b) for the other
three levels. Hold out exactly {(0.50,0.50), (0.50,0.70), (0.70,0.50),
(0.70,0.70)}. The other five grid combinations are unused, including for tuning.

| b / a | 0.10 | 0.30 | 0.50 | 0.70 |
| --- | --- | --- | --- | --- |
| 0.70 | train | unused | test | test |
| 0.50 | train | unused | test | test |
| 0.30 | train | unused | unused | unused |
| 0.10 | train | train | train | train |

Every test scalar level occurs in training. The training convex hull obeys
a+b <= 0.80, while every test point has a+b >= 1.00. Thus all four tests are
strictly outside that hull. Layer exchange maps training to training and test
to test; the split cannot be solved by exchanging layers of a training example.
Global layer flips do not change coupling coordinates, so exact architectural
symmetry is not itself evidence of learned extrapolation.

Call this **joint-coupling extrapolation with individually observed levels**.
It is not one-coordinate range extrapolation, unseen system-size transfer, or
a finite-system phase-transition demonstration. Geometric separation alone
does not prove distributional difficulty. The new condition pairs were not
used in the first pilot, but this design was informed by that pilot and is not
an independent replication of its original protocol.

## Maps, selection and data

Propose the same untuned angle range used previously, rescaled to this domain:
alpha(c)=0.225*pi - 0.20*pi*(c-0.10)/0.60. Feed alpha(a) to the first four
qubits and alpha(b) to the last four. Keep kappa and ring structure out of the
input map. Do not optimize map endpoints using the held-out corners.

The first screen uses three paired seeds, 1,000 training observations and
1,000 independent validation observations per training condition. Use the
existing pooled-data initialization, sampled-MMD settings (512 masks per
bandwidth, 2,048 model samples), 500 Adam updates, cosine learning rate .01 to
.001, batch100 per condition, and validation every25. Select among initial,
scheduled, best-validation-MMD and final candidates by mean empirical validation
TVD over the seven training conditions only. Pair samples and initialization
between the two IQP encoders. Record seed derivation and all protocol/source
hashes before execution. Do not reuse old pilot samples.

Freeze all baseline choices, fitted parameters and selected IQP checkpoints
before scoring any test target. A one-seed exact-population shared two-block
control may train only on the seven training targets. Target-aware fits at test
points are optional post-selection capacity diagnostics, explicitly labelled
and never used to retune this experiment. A poor shared result does not prove
an ansatz expressivity limit.

## Mandatory baselines

1. Shared product-input IQP, with the same input map, data and training budget.
   Also score its output averaged over the four layer-flip transformations.
   This separates the benefit of encoding symmetry from simple output correction;
   postprocessing has extra evaluation/sampling cost and is not a new trained model.
2. Symmetry-aware empirical log composition, using the exact formula above on
   estimated axis distributions. Represent the 64 four-state flip orbits with
   probabilities (n_orbit + lambda)/(N + 64*lambda). Use one global lambda from
   {0.1,1,10}, selected by mean held-out-sample log likelihood at training
   conditions only. Divide orbit mass equally among its four states, compose
   in log space, and normalize. Never clip a negative arithmetic extrapolation
   and call it interpolation. Exact axis laws give an analytic zero-TVD check,
   not a sample-matched competitive result.
3. A regularized symmetry-aware conditional log-linear categorical model:
   q(o|a,b)=softmax[u_o+a*v_o+b*w_o] over the 64 orbits. This pools all seven
   conditions rather than relying only on three histograms. It needs no ring
   Hamiltonian features, although affine log odds are a favorable assumption
   for this target. Fix one reference orbit, penalize all remaining coefficient
   vectors with L2, and select regularization from a predeclared grid using only
   training-condition validation likelihood. Freeze feature scaling, penalty
   convention, solver and convergence criteria in the executable protocol.
4. Nearest-condition symmetry-aware empirical histograms as a simple floor;
   Euclidean distance in the two coupling coordinates, averaging tied neighbors.
   Keep encoder-only and product-of-exact-layer-marginals diagnostics as before.

A fitted Ashkin–Teller Hamiltonian can be reported separately as a comparator
with supplied physical sufficient statistics. Do not omit a strong structural
baseline merely because it may win. The categorical model's 189 identifiable
coefficients and histogram storage differ from the 36 IQP parameters; report
parameter, sample, and compute budgets instead of calling them matched.

## Decision and follow-up

Report each test point and seed, mean/worst exact joint TVD, both layer marginal
TVDs, and connected cross-layer bond RMSE. Report training errors alongside test
errors; report symmetry defects only as implementation checks. No target-space
smoothing of reported predictions is permitted beyond the declared baselines.

For a quick three-seed screen, retain the provisional absolute criteria: mean
TVD <= .15, worst test TVD <= .25, worst connected-bond RMSE <= .10. These are
engineering gates, not statistical significance. To justify investigating a
predictive advantage, additionally require a mean paired TVD improvement of at
least .02 over each mandatory sample-trained classical baseline, with improvement
in every seed. The exact-law reference is excluded from this comparison.

If only the absolute gate passes, report extrapolation ability without claiming
superiority. If the classical baselines win, do not widen the gap repeatedly
until they fail: keep this as an encoding demonstration or choose a different
target family for a different scientific question. If the small screen supports
an advantage, separately freeze a 10-seed sample-efficiency study at N=100,500,
1000 per training condition, with nested training samples and all hyperparameter
selection restricted to training conditions. Its N=1000 setting is a replication;
it must not be described as untouched after the screen.

Axes do not identify arbitrary two-coordinate conditional families. For example,
an added energy term (a-a0)(b-b0)D(x), with flip-invariant D, is invisible on all
training axes and changes the test laws. Successful extrapolation therefore
reflects assumptions/inductive bias about the family, not recovery of arbitrary
unseen interactions. Introducing such a term without informative training
conditions would make a less identifiable task, not automatically a better one.

| Claim | Evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| All test points extrapolate jointly | a+b hull calculation above | Split table | Geometry does not guarantee predictive difficulty | Design, not run |
| Arithmetic interpolation is an insufficient sole baseline | Exact log-composition identity above | None | Exact inputs privileged; finite-sample performance unmeasured | Algebraic deduction |
| Two-block IQP generalizes efficiently on this split | No results yet | None | Strong baselines and capacity remain untested | Proposed, awaiting author review |
