# Symmetry-preserving Ashkin–Teller model redesign

2026-09-07. Original architecture proposal, subsequently tested in the
[bond-redesign development ablation](ashkin-teller-bond-redesign-ablation-2026-09-07.md).
Joint fitting improves; correlation accuracy remains uneven. The proposal below
is retained as provenance. Follows the
[joint-extrapolation screen](ashkin-teller-joint-extrapolation-result-2026-09-07.md)
and the user's clarification: the paper demonstrates a new conditional quantum
method and does not require superiority over classical predictors. Classical
comparisons remain context; useful fitting, conditional behavior and a justified
tractable encoder are the inclusion criteria.

## What is actually missing

The current model already respects independent global flips of the two layers.
Its physical input is a product of two cat-like states, and every X-string IQP
generator commutes with both block flips. An X-string does not need even support
in each block to commute with those flips. Do not remove valid gates based on
that mistaken restriction.

The current input is additionally permutation-invariant within each layer: its
amplitudes depend only on Hamming weight. That is well matched to exchangeable
Curie–Weiss data, but it does not distinguish neighboring domain walls on a ring.
The complete two-local unitary can break this extra permutation symmetry, so
this is an input bias, not a claim that the full model is forced to be exchangeable.
The target has local bond structure and a four-spin interaction between aligned
bonds. These give a principled direction for changing the input and phase bank.
The failed screen does not prove that these changes will improve fitting.

## Recommended input family

Use a product of two bond-aware reference states, each defined on its ring:

|E(a,b)> = |B(g(a),eta)>_A tensor |B(g(b),eta)>_B,

|B(g,eta)> = Z(g)^(-1/2) sum_s
 exp[(g/2 + i*eta) sum_i s_i s_(i+1)] |s>.

Here Z(g)=sum_s exp[g sum_i s_i s_(i+1)]. Each state is exactly invariant under
a global layer flip and ring translations/reflections. It distinguishes different
bond configurations with the same magnetization. A first problem-informed
candidate is g(c)=c and fixed eta=pi/8; neither is a trained encoder parameter.
These are proposed development choices, not a frozen experiment.

The squared input amplitudes then reproduce two uncoupled Ising rings. That is
an explicit supplied inductive bias and must be disclosed. The cross-layer
coupling kappa is absent from the encoder, whose connected interlayer bond
correlations are identically zero. The shared unitary still has to learn the
coupled target, including changes to the marginals induced by the other layer.
Do not encode the full Ashkin–Teller square-root distribution: that would supply
the desired output before training.

The fixed bond phase is intentional. With eta=0 and g=0, the input is |+>^m,
an eigenstate of every X-string: an IQP phase bank cannot change its output
probabilities. A purely real, weak-coupling Gibbs input can therefore leave
little useful phase freedom. A fixed nonzero bond phase preserves the input
measurement law and symmetry while changing the state entering the diagonal
layer. It avoids that particular |+> degeneracy; it does not prove conditional
expressivity or successful optimization.

## Why this can fit the paper's estimator interface

Tractability must hold for H^m|B>, the reference entering the diagonal layer,
not merely for samples from the physical input's Ising distribution.
For bits x,y define the 2x2 matrix

T_xy = exp[(g/2+i*eta)(1-2x)(1-2y)],
D_z = diag(1,(-1)^z).

Its post-Hadamard amplitude at z is

phi_g(z) = Tr[product_i D_(z_i) T] / sqrt(2^m Z(g)).

This follows by expanding the trace over the ring spins and applying the
Hadamard signs. Products of fixed-size matrices provide amplitude evaluation.
Contracting the doubled tensor network over unobserved bits supplies prefix
probabilities and hence sequential sampling from |phi_g|^2. The two layers
factorize at this stage. Numerators and denominators for the bit-flip amplitude
ratio are evaluated by the same contractions. The tensors have constant bond
dimension; no exponential probability table is needed in the intended method.

Efficient direct sampling from MPS Born distributions is established, e.g.
[Han et al., Unsupervised Generative Modeling Using Matrix Product States](https://arxiv.org/abs/1709.01662).
The transfer-matrix construction above and its application as a fixed IQP
reference are our proposed specialization, not a result claimed by that paper.
A four-spin algebra check at g=.4, eta=pi/8 verified all 16 transfer amplitudes
against an explicit Hadamard transform and all 31 prefix probabilities against
enumerated sums (errors below 1e-12). This checks the contraction identities,
not a production sampler or general numerical stability.
An implementation must still check normalization, support zeros, precision and
sampling/ratio accuracy after Hadamards. Exact enumeration at eight qubits is a
verification oracle, not sufficient evidence of scalable reference access.
This would demonstrate an input family beyond the existing product/shell cases.

## Recommended circuit ablation

First retain the complete two-local IQP phase bank. Separately add aligned
four-body generators in the physical X frame:

P_i = X_(A,i) X_(A,i+1) X_(B,i) X_(B,i+1).

Equivalently, add the corresponding Z strings to D_G between the Hadamards.
Their phases are learned once and shared across conditions. All P_i commute
with each other, the existing X strings and both layer flips. The model remains
IQP and the moment estimator still uses the same sampling/ratio interface.
These gates allow direct four-body phase features motivated by the target's
bond coupling; their coefficients are not the classical Gibbs coupling kappa,
and their presence does not imply exact thermal-state preparation.

There is a small-system redundancy to handle before counting parameters. On the
current eight-qubit two-block reference's 64-state even/even Fourier support,
the existing 36 nominal generators have 30 distinct nonconstant characters.
Adding the four aligned bond-pair strings raises that count to 32, not 34:
opposite bond-pair strings are equivalent because each whole-block flip acts
as +1. These counts were checked by explicit restricted-character rank and
uniqueness calculations. Remove duplicate characters or state nominal and
independent counts separately; redundant parameters do not increase capacity.
This modest change alone may therefore be insufficient.

As a later ablation, tying gate coefficients under simultaneous ring translations
and reflections would preserve those spatial symmetries for every parameter
value. Do not tie arbitrary permutations or independent translations of the
two layers: the aligned cross-layer coupling does not have those symmetries.
Do not tie a and b by exchange within a condition when a!=b. Tying reduces the
hypothesis class; it is a regularization experiment, not added expressivity.

Repeating commuting IQP layers without an intervening noncommuting operation
only combines their phases and adds no expressivity. Interleaving noncommuting
trained gates would require revisiting the IQP/estimator argument. Neither is
the first proposed change.

## Small, interpretable next experiment

Use a 2x2 development comparison: current cat input versus the bond-aware input,
each with the current phase bank versus its four-body extension. This separates
input and circuit effects. Keep the unitary fixed across conditions, use exact
population MMD first to inspect fitting capacity, and repeat promising settings
with the sampled estimator. Include initial/encoder-only outputs and connected
cross-layer bond statistics. Use all already inspected conditions as development
data; do not present the old test points as fresh confirmation of the redesign.

Before training, verify both flip symmetries and the reference sampling/ratio
interface. Also audit input distinguishability: for a common unitary and
measurement, exact target fitting requires

TVD(p_c,p_d) <= sqrt(1 - |<E(c)|E(d)>|^2).

A violation rules out exact fitting by any common unitary for those inputs;
satisfying the bound does not establish feasibility. This identifies some input
map failures that adding phase gates cannot fix.

The primary question is whether the redesigned model improves joint fitting
and learns nonzero cross-layer dependence beyond its encoder-only baseline.
Classical performance is not a gate. Freeze a separate generalization experiment
only after the development comparison supports the new architecture. A positive
result would support a spatially structured, independently flip-symmetric input
family for the conditional IQP framework, with explicit supplied-physics and
state-preparation caveats.

| Claim | Evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Current model already has independent flips | Encoder construction and commuting X-string algebra | Existing encoder tests | Does not guarantee fit | Verified architecture |
| Four new nominal strings supply only two new phase directions at n=8 | Restricted character ranks 30 to 32 on 64 states | Calculation in this note | Specific to current four-site blocks | Algebraically/numerically checked |
| Bond-aware inputs preserve the tractable reference interface | Transfer-matrix construction, MPS sampling literature | Formula above | Sampler/ratio precision and implementation untested | Proposed |
| Redesign improves model accuracy or generalization | No training performed | None | Requires controlled ablation and fresh confirmation | Unestablished |
