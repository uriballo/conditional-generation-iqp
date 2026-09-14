# Two-block symmetry and compositional generalization

Proposed 2026-09-07; subsequently tested in the
[viability pilot](ashkin-teller-pilot-2026-09-07.md). The original design below
is retained as provenance. The pilot supports encoding viability, but a
symmetry-aware classical baseline beats the shared model on this split.
The purpose is to test unseen combinations of condition values,
not to collect another positive one-coordinate interpolation result.

## Candidate target

Use a classical Ashkin-Teller model on a short ring. Each site has spins
sigma_i and tau_i. Write dimensionless couplings a,b,kappa and define

p(sigma,tau | a,b) proportional to exp[
 a sum_edges sigma_i sigma_j + b sum_edges tau_i tau_j
 + kappa sum_edges sigma_i sigma_j tau_i tau_j ].

Keep kappa fixed and nonzero. Four sites use 8 visible qubits; six sites use
12. Ring geometry introduces named-neighbor correlations and does not reduce
the target to a single Hamming-weight histogram. These tiny finite systems
are not a phase-transition or asymptotic-scaling demonstration.

The Ashkin-Teller construction couples two Ising layers through a four-spin
interaction. A primary reference is Giuliani and Mastropietro,
[Anomalous universality in the anisotropic Ashkin-Teller model](https://arxiv.org/abs/cond-mat/0404701).
Their work concerns two-dimensional critical behavior; the small classical
ring and conditional benchmark proposed here are our design, not their result.

## Why the encoder matches the symmetry

The target is invariant under independent global flips of the sigma and tau
layers, hence has a Z2 x Z2 symmetry. Nonzero kappa couples their bond
statistics without breaking either flip symmetry. Equal layer couplings can
produce additional exchange symmetry, which is not the symmetry claim here.

Let F_A and F_B flip the two registers. Every X-string IQP generator commutes
with both. Thus an encoder invariant under F_A and F_B gives a joint output
invariant under both for arbitrary trained IQP phases. A product of two
condition-dependent cat-like states suffices:

|phi(a,b)> = |cat(alpha_A(a))>_A tensor |cat(alpha_B(b))>_B,
|cat(alpha)> proportional to |p_alpha>^m + X^m |p_alpha>^m.

This is an extension from one global Z2 symmetry to two independently acting
Z2 symmetries. It reuses the exact two_block_branch construction in
src/iqp_cond/encoding.py and the tested sampling/amplitude-ratio machinery.
The encoder's post-Hadamard amplitudes factorize across the blocks; this
retains tractable reference sampling and amplitude ratios. It does not
factorize the output after cross-block IQP generators act.

Use fixed generic affine angle maps and keep the encoder independent of the
cross-layer interaction. The target's joint bond correlations must then be
learned by the fixed IQP unitary. Ancilla placement must respect the declared
register partition if ancillas are introduced; do not rely on an accidental
half-register split. Start the design audit at r=0.

## Primary generalization test

Choose four levels for each coupling a and b. Train on the 12 perimeter
combinations of the 4x4 grid; hold out its four interior combinations.
Every individual coupling level is present in training, and every test point
lies strictly inside the training convex hull. All held-out points are also
held out under interchange a,b, so the split does not substitute a trained
layer-exchanged example for a missing combination.

This tests whether the shared model recombines familiar subsystem conditions
into a joint distribution it has never seen. It is still in-domain
interpolation, not evidence for general extrapolation or causal reasoning.
Do not call exact symmetry transformations learned generalization: those
transformations follow from the construction independently of training.

Compare the generic product encoder, single-global-flip encoder and the
two-block encoder using matched maps, parameter budgets and data. Include
independent joint fits interpolated in the two-dimensional condition plane
with a rule frozen before evaluation, and target-aware fits at held-out
conditions as a capacity reference. A separate product-of-layer-marginals
control tests whether apparent success ignores the interaction.

Raw joint TVD is the headline metric. Inspect connected cross-layer bond
correlations <sigma_i sigma_j tau_i tau_j> -
<sigma_i sigma_j><tau_i tau_j>. These expose errors that separate layer
magnetization profiles can hide. A symmetry-defect metric verifies the
construction but is not itself evidence of learned generalization.

## Limits and next decision

Isotropic MMD distinguishes each two- and four-body parity term separately;
this supplies a plausible loss signal but no optimization guarantee. First
inspect exact target dependence and run a small target-aware capacity check,
including an encoder-only baseline. Proceed only if the model improves on
that baseline and reproduces cross-layer dependence, not merely each layer's
marginal. Avoid kappa=0, where the task splits into two independent problems.

The failed grid-Ising and noisy-BAS screens make another failure plausible.
This recommendation follows the symmetry and generalization question, not a
claim that the model will fit. The target is classical and exactly enumerable
at these sizes. Strong classical baselines and careful comparator scope are
still required; no quantum advantage is implied. No campaign is authorized
by this design note alone.
