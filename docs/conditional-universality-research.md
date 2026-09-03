# Conditional universality of ancilla-assisted IQP generators

## Executive conclusion

There are two defensible conditional-universality statements, but neither says
that the present 2-local broadcast-angle model is universal.

1. **Finite Conditions:** for a finite set of \(K\) Conditions, exact
   conditional universality follows from \(n+1\) hidden qubits and
   \(r=\lceil\log_2 K\rceil\) additional selector qubits, provided the IQP
   diagonal is an arbitrary diagonal unitary. The selector is encoded only in
   the initial product state, and one global diagonal unitary multiplexes the
   \(K\) unconditional universal constructions.
2. **A continuous scalar Condition:** for every TVD-continuous path
   \(t\mapsto p(\cdot\mid t)\), \(t\in[0,1]\), the same \(n+1\) hidden qubits
   plus \(r\) parallel condition-encoding qubits give a vector-valued Bernstein
   approximation that converges uniformly in TVD as \(r\to\infty\).

Sequential data reuploading is therefore **not necessary** for either result.
The continuous construction does require condition-encoding capacity to grow,
but this can be supplied by parallel product-state encodings. Such encodings
retain the Reference-state estimator. What destroys scalable trainability in
the constructive universality proof is the arbitrary, exponentially large
many-body diagonal phase table, not the product-state condition encoding.

These results should be called **multiplexed** and **asymptotic continuous**
conditional universality, respectively. Strong universality for the present
overlapping broadcast Angle encoding and Complete 2-local topology remains
open.

## Source results and their scope

Kurkin et al. define an IQP-QCBM with \(n\) visible and \(m\) hidden qubits and
prove that \(m=n+1\) hidden qubits suffice to produce any distribution on
\(\{0,1\}^n\) exactly. Their construction uses an arbitrary diagonal phase
table and decomposes a target distribution into a uniform mixture of
\(2^{n+1}\) two-sparse distributions
([Theorem “Exact universality” and its appendix proof](https://arxiv.org/abs/2510.08476)).
They explicitly caution that the construction uses exponentially many
parameters, arbitrary phase range, and \(2n+1\) total qubits, and therefore is
a theoretical expressivity result rather than a practical training
prescription. They also state that \(n+1\) is only an upper bound; optimality
for \(n>2\) is open.

Schuld, Sweke, and Meyer show that expectation-valued quantum models with
encoding gates \(e^{-ixH}\) are partial Fourier series. Repeating Pauli
encodings \(r\) times in parallel or sequence expands the accessible degree to
\(r\). Their universal-function-approximation theorem additionally assumes
arbitrary trainable global unitaries (equivalently, an arbitrary state and
observable), potentially of exponential depth
([Sections II–IV](https://arxiv.org/abs/2008.08605)).
Consequently, that theorem does not itself establish universality for Born
probabilities, conditional distributions, or the restricted IQP diagonal
used here. It is useful for identifying the finite-frequency obstruction and
for showing that parallel repetition can replace sequential reuploading as a
source of Fourier bandwidth.

## Definitions

Let \(\mathcal X=\{0,1\}^n\), and let \(\mathcal C\) be a Condition space. A
conditional IQP family is **exactly conditionally universal on
\(\mathcal C\)** if, for every Markov kernel
\(p:\mathcal C\to\Delta(\mathcal X)\), there are shared circuit parameters
\(\theta\) such that

\[
q_\theta(x\mid c)=p(x\mid c)
\quad\text{for every }(x,c)\in\mathcal X\times\mathcal C.
\]

For a compact topological Condition space, it is **uniformly conditionally
universal for continuous kernels** if, for every TVD-continuous
\(p:\mathcal C\to\Delta(\mathcal X)\) and every \(\varepsilon>0\), some model
in the resource-scaled family satisfies

\[
\sup_{c\in\mathcal C}
\operatorname{TVD}\!\left(q_\theta(\cdot\mid c),p(\cdot\mid c)\right)
\leq\varepsilon.
\]

The encoder family must be fixed independently of the target. If arbitrary
target-dependent classical preprocessing is allowed, conditional
universality can be placed in the encoder and the claim becomes vacuous.

## Theorem 1: finite-Condition exact universality

**Theorem (multiplexed exact conditional universality).**  
Let \(\mathcal C\) contain \(K<\infty\) Conditions and let
\(r=\lceil\log_2K\rceil\). Consider:

- \(n\) visible qubits;
- \(m=n+1\) hidden qubits; and
- \(r\) selector qubits that are traced out with the hidden qubits.

Choose an injection \(s:\mathcal C\to\{0,1\}^r\). Before the first Hadamard
layer, prepare

\[
|\chi_c\rangle
=H^{\otimes r}|s(c)\rangle_S
 \otimes |0\rangle_H^{\otimes(n+1)}
 \otimes |0\rangle_V^{\otimes n}.
\]

If the shared IQP diagonal \(D_\theta\) may be an arbitrary diagonal unitary
on all \(2n+1+r\) qubits, then the model is exactly conditionally universal on
\(\mathcal C\).

**Proof.** After the first Hadamards, the selector is the computational-basis
state \(|s(c)\rangle\), while the hidden and visible registers have the
uniform amplitudes required by the unconditional construction. For every
Condition \(c\), apply the exact-universality theorem of Kurkin et al. to
choose a diagonal block \(D_c\) on the hidden and visible registers that
produces \(p(\cdot\mid c)\). Define one shared diagonal unitary

\[
D=\sum_{u\in\{0,1\}^r}|u\rangle\!\langle u|_S\otimes D_u,
\]

with \(D_{s(c)}=D_c\) and arbitrary blocks for unused selector strings. This
operator is diagonal and hence is an IQP diagonal. The encoded selector
chooses the required block. The final Hadamards on selector and hidden qubits
can be omitted because local unitaries on traced registers do not change the
visible reduced state. Therefore the visible output is
\(p(\cdot\mid c)\) for every \(c\). \(\square\)

### Interpretation

The Condition still enters only through a Tractable product-state encoding,
and \(\theta\) is formally one global parameter vector. However, different
Conditions address disjoint phase blocks. This is a direct-sum
multiplexing of separate universal generators, not the stronger scientific
notion in which every Condition constrains the same effective parameters.
It therefore conflicts with the intended shared-parameter interpretation,
even though the condition still enters only through the input state. It is best presented as an
expressivity upper bound or a deliberately weak definition of conditional
universality.

## Theorem 2: continuous conditional universality without sequential reuploading

**Theorem (uniform Bernstein conditional universality).**  
Let
\(p:[0,1]\to\Delta(\{0,1\}^n)\) be continuous in TVD. For every
\(\varepsilon>0\), there is an IQP-QCBM with:

- \(n\) visible qubits;
- \(n+1\) unconditional-universality hidden qubits; and
- \(r\) parallel condition-encoding selector qubits

such that a single shared arbitrary diagonal unitary produces

\[
\sup_{t\in[0,1]}
\operatorname{TVD}\!\left(q_r(\cdot\mid t),p(\cdot\mid t)\right)
\leq\varepsilon
\]

for sufficiently large \(r\).

**Construction.** Prepare each selector qubit so that its Reference-state
amplitudes after the first Hadamard are

\[
|\phi_t\rangle=\sqrt{1-t}|0\rangle+\sqrt t|1\rangle.
\]

This is a real product-state encoder. In the repository's Angle-encoding
notation it can be obtained with

\[
\alpha(t)=\arcsin(1-2t),
\qquad
\Pr_{\phi_t}(1)=\frac{1-\sin\alpha(t)}2=t.
\]

For every selector string \(s\) of Hamming weight \(j\), choose its diagonal
block to be the unconditional universal construction for the grid
distribution \(p(\cdot\mid j/r)\). Tracing the selectors gives

\[
q_r(x\mid t)
=\sum_{j=0}^r
\binom rj t^j(1-t)^{r-j}p(x\mid j/r),
\]

the vector-valued Bernstein polynomial of \(p\).

**Uniform TVD bound.** Let

\[
\omega_p(\delta)
=\sup_{|u-v|\leq\delta}
\operatorname{TVD}\!\left(p(\cdot\mid u),p(\cdot\mid v)\right).
\]

If \(J\sim\operatorname{Binomial}(r,t)\), convexity of TVD gives

\[
\begin{aligned}
\operatorname{TVD}(q_r(\cdot\mid t),p(\cdot\mid t))
&\leq
\mathbb E\!\left[
\operatorname{TVD}(p(\cdot\mid J/r),p(\cdot\mid t))
\right]\\
&\leq \omega_p(\delta)
+\Pr(|J/r-t|>\delta)\\
&\leq \omega_p(\delta)+\frac{1}{4r\delta^2}.
\end{aligned}
\]

Taking \(\delta=r^{-1/4}\) yields
\[
\sup_t\operatorname{TVD}(q_r(\cdot\mid t),p(\cdot\mid t))
\leq
\omega_p(r^{-1/4})+\frac1{4\sqrt r}\longrightarrow0.
\]

If \(p\) is \(L\)-Lipschitz in TVD, the sharper bound

\[
\sup_t\operatorname{TVD}(q_r(\cdot\mid t),p(\cdot\mid t))
\leq\frac{L}{2\sqrt r}
\]

follows from
\(\mathbb E|J/r-t|\leq\sqrt{\operatorname{Var}(J/r)}
\leq1/(2\sqrt r)\). \(\square\)

The same construction extends to a compact box
\([0,1]^d\) with tensor-product Bernstein weights and a separate bank of
parallel selector qubits for each coordinate.

### What this theorem does and does not say

- It is a genuine shared interpolation construction: interior Conditions mix
  the same bank of grid-point phase blocks rather than select a single one.
- It guarantees uniform approximation only for TVD-continuous conditional
  families. A continuous encoder and circuit cannot uniformly approximate a
  discontinuous target kernel.
- At finite \(r\), it is exact only for conditional families already in the
  resulting finite Bernstein-mixture class.
- It uses parallel repetition of the Condition. Sequential noncommuting
  data reuploading is unnecessary.
- The nonlinear map \(\alpha(t)=\arcsin(1-2t)\) is fixed classical
  preprocessing and must be disclosed as part of the model.

## Why the present fixed broadcast encoder is not continuously universal

Suppose a scalar angle \(\alpha\) is uploaded once in parallel to \(N\)
qubits as \(R_Y(\alpha)^{\otimes N}|0^N\rangle\), followed by any
\(\alpha\)-independent circuit and computational-basis measurement, with any
subset of qubits marginalized. Every pre-measurement amplitude is a
homogeneous degree-\(N\) polynomial in
\(\cos(\alpha/2)\) and \(\sin(\alpha/2)\). Every output probability is
therefore a trigonometric polynomial in \(\alpha\) of degree at most \(N\).

For fixed \(N\), these probabilities lie in a fixed finite-dimensional,
closed function space, which is not dense in
\(C([a,b])\). Hence no fixed-size version of the present broadcast Angle
encoding can uniformly approximate every continuous conditional
distribution, even if its IQP diagonal is arbitrary. Increasing parallel
encoding width or sequential upload depth expands the accessible spectrum,
in agreement with Schuld et al.

This obstruction is specific to the linear Pauli-angle encoder. An arbitrary
nonlinear classical feature map can inject additional or even infinite
Fourier content, but then the expressivity must be credited to the combined
classical preprocessing and quantum model.

## Classical-trainability audit

The finite-selector and Bernstein encoders are Tractable product-state
encodings. Sampling their Reference distributions and computing visible-mask
amplitude ratios remain efficient. Indeed, in the constructions above the
observable masks flip only uniform visible-register amplitudes, so the
selector amplitudes cancel from the ratio. The repository's bounded
second-moment estimator therefore survives.

This does **not** make the universal construction scalable. An arbitrary
diagonal unitary on \(N\) qubits has \(2^N-1\) independent non-global Pauli-\(Z\)
coefficients in general. The present Complete 2-local topology has only
\(N+\binom N2\) parameters and cannot implement that phase table. The
estimator is polynomial in the number of diagonal generators, so it becomes
exponential in \(n\) for the universal construction. This is the
expressivity–trainability trade-off already acknowledged by Kurkin et al.

Input-only noncommuting reuploading, such as the repository's Symmetric
reupload encoding, also remains a Tractable product-state encoding and does
not by itself invalidate direct backpropagation with respect to the shared
IQP phases. Reuploading interleaved with noncommuting trainable IQP layers
would be a different circuit family and would generally lose the present
estimator.

## Recommended claim boundary for the paper

A safe theorem section could contain:

1. the finite-Condition theorem, explicitly labeled a multiplexed upper bound;
2. the Bernstein theorem as the main positive continuous result;
3. the fixed-width Fourier obstruction for the present Angle encoding; and
4. the classical-trainability audit distinguishing Reference-state
   tractability from exponential diagonal description size.

The manuscript must not say that the experimentally tested Complete 2-local
model is universal, that \(n+1\) hidden qubits alone guarantee conditional
universality, or that Schuld et al.'s function-approximation theorem directly
applies to IQP Born distributions.

## Open problems

- Can the overlapping broadcast Angle encoder with an arbitrary shared
  diagonal, but without selector-addressable phase blocks, be conditionally
  universal for a finite number of Conditions?
- Is any fixed locality \(k\) sufficient for asymptotic conditional
  universality with polynomially many diagonal generators?
- Can one prove a useful target-class theorem for the Complete 2-local
  topology, such as universality for permutation-symmetric Hamming-weight
  families?
- Can the Bernstein phase blocks be compiled with useful symmetry or tensor
  structure so that phase evaluation and hardware depth avoid the generic
  exponential construction?
