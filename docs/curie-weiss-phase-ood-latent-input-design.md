# Ancilla-purified input encoding for phase-crossing generalization

## Result in one sentence

A target-independent ancilla encoder crosses the withheld Curie--Weiss
critical interval without multiplexing IQP phases: five ancillas reduce mean
profile/raw TVD from 0.262/0.341 for affine broadcast to 0.060/0.116, while an
equally sized visible-only circuit reaches only 0.144/0.294.

## What “no multiplexing” means

Tracing ancillas always produces a mathematical mixture in the visible
marginal. The relevant restriction is therefore not “no mixture,” which would
exclude every mixed visible state, but **no independently learned IQP phase
block per latent sector**.

The strict model has one complete two-local visible phase vector
\(\theta\in\mathbb R^{36}\). The diagonal acts only on the eight visible
qubits. It never acts on an ancilla, and \(\theta\) is identical for every
ancilla string. All latent variation enters before the IQP block through the
input state.

## Encoding

Normalize the declared condition interval as

\[
t=\frac{c-c_{\min}}{c_{\max}-c_{\min}}\in[0,1].
\]

Prepare \(r\) ancillas in the generic product state

\[
|\chi(t)\rangle=
\left(\sqrt{1-t}|0\rangle+\sqrt t|1\rangle\right)^{\otimes r}.
\]

Let \(L\in\mathbb R^{n\times r}\) contain learned input couplings. Conditional
on ancilla string \(z\), the visible product-state angles are

\[
\alpha_i(t,z)=\alpha_i^{(0)}(t)+
\sum_{k=1}^{r}L_{ik}(-1)^{z_k},
\qquad
\alpha_i^{(0)}(t)=\pi(0.35-0.70t).
\]

Same-axis ancilla-controlled \(R_Y\) rotations implement these shifts. The
joint input is

\[
|\Phi_L(t)\rangle=
\sum_z\sqrt{t^{|z|}(1-t)^{r-|z|}},|z\rangle_a
\otimes|\phi_L(t,z)\rangle_v.
\]

After one shared visible IQP circuit \(U_\theta^{(v)}\), tracing the ancillas
gives

\[
q_{\theta,L}(x|t)=\sum_z t^{|z|}(1-t)^{r-|z|}
\left|\langle x|U_\theta^{(v)}|\phi_L(t,z)\rangle\right|^2.
\]

The weights are fixed by the declared scalar condition. The learned quantities
are the one shared phase vector \(\theta\) and the input decoder \(L\). No
Curie--Weiss energy, mode, symmetry, critical-point location, or test target
enters the encoder.

## Why ancilla count matters

With \(r\) ancillas, the visible input has purification rank at most \(2^r\)
and gains \(nr\) learned input directions. For \(n=8\):

| ancillas \(r\) | shared IQP phases | input couplings | total parameters | test profile TVD | test raw TVD |
|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 8 | 44 | 0.108 | 0.225 |
| 3 | 36 | 24 | 60 | 0.074 | 0.161 |
| 5 | 36 | 40 | 76 | 0.060 | 0.116 |
| visible-only control | 76 | 0 | 76 | 0.144 | 0.294 |

The monotone ladder shows that the useful capacity scales with ancilla rank.
The matched 76-parameter control shows that adding visible diagonal parameters
does not reproduce the effect.

## Protocol

The primary critical-gap protocol trains at
\(\beta\in\{0.5,0.7,0.8,1.2,1.3,1.5\}\) and withholds
\(\{0.9,1.0,1.1\}\). Training therefore observes both phases but no target in
the transition interval. Each arm uses exact multiscale population MMD,
10,000 Adam updates, and ten optimizer seeds. Test targets are constructed only
after the minimum-training-objective checkpoint is fixed.

The one-sided stress test trains at \(\{0.5,0.6,0.7,0.8,0.9\}\) and evaluates
\(\{1.0,1.1,1.2,1.3,1.4,1.5\}\). The strict encoder improves affine
profile/raw TVD from 0.382/0.466 to 0.306/0.316, but this is not a strong
extrapolation result. No distribution-free method can infer an arbitrary
unseen phase from observations on only one side.

## Non-Curie control

On an eight-qubit Gaussian Hamming-weight location path with the same withheld
middle interval, the five-ancilla model improves profile/raw TVD from
0.110/0.241 to 0.071/0.103. The matched visible-only control gives
0.126/0.320. This five-seed screen supports a general input-rank mechanism
rather than a Curie--Weiss-specific symmetry or energy feature.

## Classical training interface

To estimate a visible Pauli observable, sample \(z\) from its Bernoulli product
law and then sample the visible product state conditioned on \(z\). A visible
bit-flip mask leaves \(z\) fixed, so the ancilla amplitude cancels from the
amplitude ratio and the remaining ratio factorizes over visible qubits. The
existing observable estimator therefore does not enumerate \(2^r\) branches.

## Evidence boundary

These are exact-population, eight-visible-qubit development experiments. The
encoder and the \(r=5\) endpoint were selected during exploration. The result
supports identifiable interpolation across a withheld transition interval; it
does not establish finite-sample robustness, large-system scaling,
distribution-free extrapolation, or quantum advantage. A confirmatory campaign
should freeze the encoder, use sampled MMD, increase system size, and reserve a
new condition grid and target family for evaluation.
