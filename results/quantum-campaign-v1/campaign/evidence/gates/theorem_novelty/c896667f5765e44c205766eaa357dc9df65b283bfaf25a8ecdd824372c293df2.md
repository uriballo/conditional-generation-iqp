# Finite-resource conditional-response theorem dossier

Status: **internal correctness pass; theorem-novelty gate fail**  
Audit date: 2026-08-10  
Scope: affine `RY` product inputs, a fixed `H D_Z H` IQP circuit, output
probabilities, Pauli-`Z` responses, and the classical amplitude-ratio
estimator.  
External-review status: **not reviewed**. An independent quantum-information
or QML expert must reassess both the proofs and the prior-art search before any
submission decision.

## Executive decision

The mathematical statements below pass the present internal algebraic and
small-system checks. They do **not** pass the novelty condition in
[ADR 0042](adr/0042-require-a-written-theorem-novelty-dossier.md). In
particular:

- The multivariate affine-input frequency-support result is a direct
  specialization of established parameterized-circuit Fourier results. A
  probability is just the expectation of an output projector.
- The bandwidth, smoothness, and fixed-width non-density statements are useful
  finite-resource consequences, but are elementary corollaries of that finite
  spectrum.
- The IQP observable refinement adds the real-product-state `Y` filter and
  maps the surviving binary span to condition frequencies. However, the
  active anticommuting-generator expansion and subset/XOR cancellation
  mechanism already appear explicitly in recent primary work. The combined
  formula was not located verbatim, but it is a short synthesis rather than a
  defensible new central theorem.
- The estimator and gradient statements extend the standard IQP estimator to
  an efficiently sampleable product reference. They are covered in spirit by
  the general computationally-tractable-state/efficiently-computable-operator
  framework and by recent IQP estimator/autodifferentiation work.

Consequently, this dossier does not justify spending compute under the frozen
Quantum campaign. Target-only preparatory artifacts and sampler pilots were
created while the audit was in progress, but no `TARGET_CALIBRATION` gate was
released and production training was not authorized. The contribution must be
strengthened or the paper must be reframed before the theorem gate is
reconsidered.

## 1. Scope, assumptions, and notation

Let there be `n >= 1` qubits and a condition vector
`c in R^d`, `d >= 1`. The condition map is fixed independently of the target:

\[
  \alpha_i(c)=b_i+w_i^\mathsf{T}c,
  \qquad b_i\in\mathbb R,\quad w_i\in\mathbb R^d .
\]

Write `W in R^{n x d}` for the matrix with row `w_i^T`. The input is

\[
  |\psi(c)\rangle
    =\bigotimes_{i=1}^n R_Y(\alpha_i(c))|0\rangle .
\]

For binary masks `a,g,h in F_2^n`, define

\[
 X_a=\bigotimes_i X_i^{a_i},\qquad
 Z_g=\bigotimes_i Z_i^{g_i},\qquad
 \langle a,g\rangle_2=\sum_i a_i g_i\pmod 2.
\]

The fixed IQP circuit is

\[
 U_\theta=H^{\otimes n}D_\theta H^{\otimes n},\qquad
 D_\theta=\prod_{j=1}^m e^{i\theta_j Z_{g_j}}.
\]

All `g_j`, the ordering of gates, `theta`, `W`, and `b` are independent of the
condition at deployment. Only the input state changes with `c`. The results do
not assume that the rows of `W` are distinct, commensurate, or nonzero.

For a fixed output string `x`, let

\[
  q_\theta(x\mid c)
    =|\langle x|U_\theta|\psi(c)\rangle|^2.
\]

For an output Pauli observable `Z_a`, let

\[
  f_a(c)=\langle\psi(c)|U_\theta^\dagger Z_a U_\theta|\psi(c)\rangle.
\]

The Fourier convention in this dossier is `exp(i omega^T c)`; there is no
`2 pi` in the exponent. “Support” always means an upper bound on the
structurally allowed frequencies. A permitted coefficient can vanish because
of parameter choices, symmetry, or collisions between signed sums.

## 2. T1 — Full-probability affine-input frequency support

### Statement

For every fixed unitary `U` (and therefore for every fixed IQP unitary), every
output string `x`, and every affine `RY` product input above, there are complex
coefficients `Q_{x,s}` such that

\[
 q(x\mid c)=
 \sum_{s\in\{-1,0,1\}^n}
 Q_{x,s}\exp\!\left(i\left[\sum_{i=1}^n s_iw_i\right]^\mathsf{T}c\right).
\]

The bias vector `b` and the fixed unitary affect the coefficients, not the
frequency set. Thus

\[
 \operatorname{supp}\widehat q_x\subseteq
 \Omega_{\mathrm{full}}(W)
 :=\left\{\sum_i s_iw_i:s_i\in\{-1,0,1\}\right\}.
\]

If `r` rows of `W` are nonzero, then
`|Omega_full(W)| <= 3^r`. Equality is not guaranteed.

### Proof

The one-qubit input density operator is

\[
 \rho_i(\alpha_i)
 =R_Y(\alpha_i)|0\rangle\!\langle0|R_Y(\alpha_i)^\dagger
 =\tfrac12\left(I+\sin\alpha_i X+\cos\alpha_i Z\right).
\]

Both sine and cosine are linear combinations of `exp(+i alpha_i)` and
`exp(-i alpha_i)`. Hence each matrix entry of `rho_i` has angle-frequency
support contained in `{-1,0,1}`. Tensoring the local states gives

\[
 \rho(c)=\bigotimes_i\rho_i(\alpha_i(c))
   =\sum_{s\in\{-1,0,1\}^n} A_s
      \exp\!\left(i\sum_i s_i\alpha_i(c)\right)
\]

for fixed matrices `A_s`. Substitution of
`alpha_i(c)=b_i+w_i^T c` gives

\[
 \rho(c)=\sum_s
 \left[A_s e^{i\sum_i s_ib_i}\right]
 e^{i(\sum_i s_iw_i)^\mathsf{T}c}.
\]

Finally,

\[
 q(x\mid c)=\operatorname{Tr}
   \left(U^\dagger|x\rangle\!\langle x|U\,\rho(c)\right)
\]

is a fixed linear functional of `rho(c)`, so it cannot introduce a new
frequency. There are at most three choices for each nonzero row. Zero rows,
equal signed sums, and coefficient cancellation can only reduce the number of
distinct frequencies. This proves the statement.

### Interpretation and boundaries

- This theorem characterizes **every individual full output probability**.
  It does not provide an efficient algorithm for evaluating that probability.
- The IQP structure is unnecessary for T1; it holds for any fixed post-input
  unitary and measurement effect.
- Arbitrary real rows `w_i` can produce a quasiperiodic rather than a common
  scalar-periodic response. The finite real-vector frequency statement still
  holds.
- The theorem concerns a fixed `W`. Allowing target-dependent or
  condition-dependent changes to `W` changes the model class and invalidates
  the fixed-support conclusion.

### Novelty assessment

**No novelty credit.** Gil Vidal and Theis give the `{-1,0,1}^n` spectrum and
multiaffine sine/cosine form for Pauli encodings, then show that affine scalar
encoding maps it to signed sums of the encoding weights. Schuld, Sweke, and
Meyer give the general eigenvalue-difference Fourier theorem and an explicit
multidimensional extension. Taking the observable to be
`U^dagger |x><x| U` is immediate.

## 3. T2 — IQP anticommutation/span refinement for Pauli observables

### Statement

For output mask `a`, define the active generator set

\[
 \mathcal A(a)=\{j:\langle a,g_j\rangle_2=1\}
\]

and its binary row span

\[
 \mathcal H_a=\operatorname{span}_{\mathbb F_2}
 \{g_j:j\in\mathcal A(a)\}.
\]

For `h in H_a`, define

\[
 \kappa_h(\theta)=
 \sum_{\substack{S\subseteq\mathcal A(a)\\
                   \bigoplus_{j\in S}g_j=h}}
 i^{|S|}
 \prod_{j\in S}\sin(2\theta_j)
 \prod_{j\in\mathcal A(a)\setminus S}\cos(2\theta_j).
\]

Then the exact conditioned response is

\[
 f_a(c)=
 \sum_{\substack{h\in\mathcal H_a\\h\wedge a=0}}
 \kappa_h(\theta)
 \prod_{i:a_i=1}\cos\alpha_i(c)
 \prod_{i:h_i=1}\sin\alpha_i(c). \tag{T2}
\]

Here `h wedge a=0` means disjoint bit support. For every surviving `h`,
`kappa_h(theta)` is real. A coefficient may nevertheless vanish.

### Proof

Conjugating the measured operator through the final Hadamards gives
`H^n Z_a H^n=X_a`. Every diagonal generator with
`<a,g_j>_2=0` commutes with `X_a`. Because all diagonal generators commute,
those inactive gates cancel pairwise in `D^dagger X_a D`. For an active
generator, anticommutation gives

\[
 e^{-i\theta_jZ_{g_j}}X_a e^{i\theta_jZ_{g_j}}
 =X_a e^{2i\theta_jZ_{g_j}}.
\]

Therefore

\[
 D_\theta^\dagger X_aD_\theta
 =X_a\prod_{j\in\mathcal A(a)}
   \left(\cos(2\theta_j)I+i\sin(2\theta_j)Z_{g_j}\right).
\]

Expanding over subsets `S` of active generators produces the Pauli mask
`h=oplus_{j in S} g_j`. Grouping equal masks yields

\[
 D_\theta^\dagger X_aD_\theta
 =\sum_{h\in\mathcal H_a}\kappa_h(\theta)X_aZ_h.
\]

Conjugation through the first Hadamards maps `X_a Z_h` to `Z_a X_h`. The input
is a product state whose local expectations are

\[
 \langle X_i\rangle=\sin\alpha_i,\qquad
 \langle Z_i\rangle=\cos\alpha_i,\qquad
 \langle Y_i\rangle=0.
\]

If `h` and `a` overlap at any qubit, the local factor in `Z_a X_h` is
proportional to `Y`, so the product-state expectation is zero. If they are
disjoint, the expectation factors into the sine/cosine product in (T2).

It remains to show that a surviving coefficient is real. Every subset `S`
that produces mask `h` obeys

\[
 |S|\pmod2
 =\sum_{j\in S}\langle a,g_j\rangle_2
 =\left\langle a,\bigoplus_{j\in S}g_j\right\rangle_2
 =\langle a,h\rangle_2.
\]

For a disjoint survivor this parity is zero, so every `|S|` is even and
`i^{|S|}` is real. This completes the proof.

### Corollary T2.1 — Refined observable frequency support

For `I(h)=supp(a) union supp(h)`, (T2) and the exponential forms of sine and
cosine imply

\[
 \operatorname{supp}\widehat f_a\subseteq
 \Omega_a(W,G):=
 \bigcup_{\substack{h\in\mathcal H_a\\h\wedge a=0}}
 \left\{\sum_{i\in I(h)}\sigma_iw_i:
 \sigma_i\in\{-1,+1\}\right\}. \tag{T2.1}
\]

This is an upper bound: `kappa_h=0`, repeated frequencies, or cancellation
between different `h` values may shrink it.

### Corollary T2.2 — Observable lightcone

Let

\[
 L(a)=\operatorname{supp}(a)\cup
 \bigcup_{j\in\mathcal A(a)}\operatorname{supp}(g_j).
\]

Then `f_a(c)` is independent of every `alpha_i(c)` with `i notin L(a)`, and
therefore independent of the corresponding rows of `W`. The binary-span and
`Y`-filter support (T2.1) can be strictly smaller than this union lightcone.

### Novelty assessment

**Exact combined formula not located; central novelty still fails.** The
active anticommuting set and expansion over active-generator subsets already
appear in Eqs. (25)--(27) of Recio-Armengol, Ahmed, and Bowles. Their
unconditioned `|0>`, `X`-diagonal convention retains subsets whose XOR is the
identity. Equation (T2) replaces that final filter with the expectation on a
real affine `RY` product state, then applies established Fourier analysis.
This is useful bookkeeping and gives a sharper condition-response diagnostic,
but it is a direct Pauli-algebra/Fourier synthesis, not a credible standalone
Quantum-level theorem.

## 4. T3 — Bandwidth and derivative bounds

### Statement

Define the coordinate bandwidths

\[
 B_k=\sum_{i=1}^n|W_{ik}|,
 \qquad k=1,\ldots,d.
\]

Let `M` be any fixed effect or observable with operator norm at most one and

\[
 F(c)=\operatorname{Tr}\left[M U\rho(c)U^\dagger\right].
\]

For every multi-index `gamma in N_0^d`, with `B_k^0=1`,

\[
 |\partial^\gamma F(c)|
 \leq\prod_{k=1}^d B_k^{\gamma_k}. \tag{T3}
\]

For a direction `v in R^d` and integer `r >= 0`,

\[
 \left|\frac{d^r}{dt^r}F(c+tv)\bigg|_{t=0}\right|
 \leq\left(\sum_i|w_i^\mathsf{T}v|\right)^r. \tag{T3-dir}
\]

For the observable `f_a`, every sum may be restricted to the lightcone
`L(a)` from Corollary T2.2.

### Proof

Every derivative of a local state with respect to its angle has trace norm
one. For order zero this follows because `rho_i` is a density operator. For
every positive order the derivative is

\[
 \frac12\left(u_r(\alpha_i)X+v_r(\alpha_i)Z\right),
 \qquad u_r(\alpha_i)^2+v_r(\alpha_i)^2=1,
\]

whose eigenvalues are `+1/2` and `-1/2`.

Apply the requested coordinate derivatives one at a time to
`rho(c)=tensor_i rho_i(alpha_i(c))`. The product rule gives a sum indexed by
an ordered choice of a differentiated qubit for each derivative. Every tensor
term has trace norm one because trace norms multiply under tensor products.
For a coordinate derivative `partial/partial c_k`, the absolute coefficients
sum to `sum_i |W_ik|=B_k`. Repeating the argument for all entries of `gamma`
gives

\[
 \|\partial^\gamma\rho(c)\|_1
 \leq\prod_kB_k^{\gamma_k}.
\]

Unitary invariance of the trace norm and Holder's trace inequality give

\[
 |\partial^\gamma F(c)|
 \leq\|M\|_\infty\|\partial^\gamma\rho(c)\|_1,
\]

which proves (T3). Replacing each coordinate coefficient by
`w_i^T v` proves (T3-dir). T2 shows that angles outside `L(a)` do not occur in
the observable response, so they can be omitted from its bound.

### Consequences

- Every fixed-width response is real analytic and band-limited in the stated
  finite-frequency sense.
- Increasing a row norm of `W` raises the available bandwidth and the worst
  derivative bound together. A wider spectrum is not free smoothness.
- A small bound does not guarantee easy optimization, and a large bound does
  not imply a barren plateau. This is response regularity, not landscape
  trainability.

### Novelty assessment

**No central novelty credit.** The explicit trace-norm derivative bound is a
convenient finite-resource corollary. Analyticity and bandwidth limitation
follow directly from the known finite Fourier representation.

## 5. T4 — Fixed-width non-density limitation

### Statement

Fix `n`, `W`, and `b`. Let `C subset R^d` be compact with nonempty interior,
and allow the post-input unitary and its trainable parameters to vary while
retaining that condition map. The resulting full-probability responses are
not dense, in the uniform norm, in the continuous probability-valued maps on
`C`.

More explicitly, let

\[
 V_\Omega=\operatorname{span}
 \{c\mapsto e^{i\omega^\mathsf{T}c}:\omega\in\Omega_{\rm full}(W)\}
 \subset C(C;\mathbb C).
\]

There is a continuous two-outcome target family `p(.|c)` and a constant
`Delta>0` such that every model with the fixed condition map has

\[
 \sup_{c\in C}|q(x_0\mid c)-p(x_0\mid c)|\geq\Delta. \tag{T4}
\]

### Proof

`V_Omega` is finite dimensional and therefore closed in the uniform norm.
Choose `omega_star` such that neither `+omega_star` nor `-omega_star` belongs
to the finite set `Omega_full(W)`. Distinct real-vector exponentials are
linearly independent on any open set, so

\[
 g(c)=\cos(\omega_\star^\mathsf{T}c)\notin V_\Omega.
\]

Because `V_Omega` is closed,

\[
 \delta=\inf_{v\in V_\Omega}\|g-v\|_\infty>0.
\]

For any `0<epsilon<=1/2`, define

\[
 p(x_0\mid c)=\tfrac12+\epsilon g(c),\qquad
 p(x_1\mid c)=\tfrac12-\epsilon g(c),
\]

and assign zero probability to any remaining outcomes. This is a continuous
valid probability family. T1 puts every `q(x_0|.)` in `V_Omega`, and the
constant `1/2` also lies in `V_Omega` because zero is an allowed frequency.
Thus

\[
 \|q(x_0\mid\cdot)-p(x_0\mid\cdot)\|_\infty
 \geq\epsilon\delta.
\]

Taking `Delta=epsilon delta` proves the claim.

### Boundaries and novelty assessment

This is a **fixed-resource** statement, not a denial of approximation when
width, re-upload count, or encoding weights grow. It also says nothing about
whether a particular low-bandwidth target is learnable from finite data.

**No novelty credit.** It is the standard finite-dimensional closed-subspace
consequence of the known Fourier support and closely tracks input-redundancy
and multidimensional degree-of-freedom limitations in prior work.

## 6. T5 — Product-reference estimator, gradients, and costs

### Statement

Let

\[
 |\chi(c)\rangle=H^{\otimes n}|\psi(c)\rangle
 =\sum_z\phi_c(z)|z\rangle,
 \qquad \pi_c(z)=|\phi_c(z)|^2.
\]

For the `RY` product input, both `phi_c(z)` and `pi_c(z)` factor into local
terms, so `pi_c` is sampled and `phi_c(z)` is evaluated in linear time.
Define

\[
 \Phi_\theta(z)=\sum_j\theta_j(-1)^{g_j\cdot z},
\]

and, with the value set to zero whenever `phi_c(z)=0`,

\[
 R_{a,c,\theta}(z)=
 \frac{\phi_c(z\oplus a)}{\phi_c(z)}
 \exp\!\left(i[\Phi_\theta(z\oplus a)-\Phi_\theta(z)]\right).
\]

For `z ~ pi_c`,

\[
 \mathbb E\,\operatorname{Re}R_{a,c,\theta}(z)=f_a(c),
 \qquad
 \mathbb E|R_{a,c,\theta}(z)|^2\leq1. \tag{T5}
\]

Moreover,

\[
 \Phi_\theta(z\oplus a)-\Phi_\theta(z)
 =-2\sum_{j\in\mathcal A(a)}
 \theta_j(-1)^{g_j\cdot z}. \tag{T5-active}
\]

The sample law `pi_c` is independent of `theta`. Termwise differentiation is
therefore unbiased:

\[
 \partial_{\theta_j}f_a(c)
 =\mathbb E\,\operatorname{Re}
   \left[\partial_{\theta_j}R_{a,c,\theta}(z)\right],
\]

where

\[
 \partial_{\theta_j}R=
 \begin{cases}
 -2i(-1)^{g_j\cdot z}R,&j\in\mathcal A(a),\\
 0,&j\notin\mathcal A(a).
 \end{cases}
\]

Every active gradient component has second moment at most four.

### Proof

Expanding the expectation in the diagonal basis gives

\[
 f_a(c)=
 \sum_z\phi_c(z)^*\phi_c(z\oplus a)
 e^{i[\Phi_\theta(z\oplus a)-\Phi_\theta(z)]}.
\]

On the support of `pi_c`, multiplying the defined ratio by
`pi_c(z)=phi_c(z)^*phi_c(z)` reproduces this summand. Terms with
`phi_c(z)=0` vanish in the original sum, so the zero convention is harmless.
The quantum expectation is real, which proves the first identity in (T5).

For the second moment,

\[
 \mathbb E_{z\sim\pi_c}|R(z)|^2
 =\sum_{z:\phi_c(z)\neq0}|\phi_c(z\oplus a)|^2\leq1.
\]

For each generator character
`chi_j(z)=(-1)^{g_j dot z}`,

\[
 \chi_j(z\oplus a)=\chi_j(z)(-1)^{g_j\cdot a}.
\]

The phase difference is therefore zero for inactive generators and
`-2 theta_j chi_j(z)` for active ones, proving (T5-active). The sum is finite
and `pi_c` is independent of `theta`, so differentiation and expectation
commute. The derivative multiplier has magnitude two on active entries,
which combines with (T5) to give the gradient second-moment bound.

### Statistical guarantee

For `S` independent reference samples, the ordinary sample mean is unbiased
and has root-mean-square error at most `S^{-1/2}`. Chebyshev gives

\[
 \Pr(|\widehat f_a-f_a|\geq\varepsilon)
 \leq\frac{1}{S\varepsilon^2}.
\]

A median-of-means wrapper gives the usual
`O(epsilon^{-2} log(1/delta))` high-probability scaling under this finite
variance bound, but the median-of-means output is not generally unbiased.
There is **no** pointwise bound `|R|<=1`: a rare sample can have a large
amplitude ratio. Hoeffding's inequality is therefore unavailable without an
additional clipping/boundedness argument.

### Operation counts

Let `k_j=|supp(g_j)|`, `r=|supp(a)|`, and
`m_a=|A(a)|`. With local reference amplitudes precomputed:

- constructing affine angles costs `O(nd)` per condition;
- drawing one reference bitstring costs `O(n)` naively;
- its amplitude flip ratio costs `O(r)`;
- its active phase costs
  `O(sum_{j in A(a)} k_j)` without cached generator characters, or `O(m_a)`
  once those characters are available;
- all sparse active gradient entries reuse the same ratio and characters and
  cost `O(m_a)` to accumulate, in addition to the phase work;
- for `k`-local generators the observable phase work is `O(k m_a)` per
  sample; dense unstructured masks give `O(n m_a)`;
- for `C` conditions, `R` observables, and `S` shared reference samples per
  condition, a direct batched implementation costs
  `O(CS[n + sum_r(r_r + sum_{j in A(a_r)} k_j)])`, plus `O(Cnd)` angle
  construction. Streaming avoids storing all samples.

Writing a dense `m`-component gradient necessarily costs `Omega(m)` even when
many components are structurally zero. The structural support enumeration in
`response.py` is exponential and is a theorem-checking utility, not part of
the training algorithm.

For a Conditional MMD written as a mixture of Pauli moments, the same samples
can feed the required observables. A naive square of an unbiased moment
estimate is biased. Unbiased MMD estimation requires independent cross-products
or the corresponding U-statistic construction; differentiating that finite
sample expression remains unbiased because the reference law is independent
of `theta`.

### What T5 does not establish

- It does not evaluate all `2^n` output probabilities efficiently.
- It does not yield an efficient classical sampler from the final IQP output.
- It does not establish optimizer convergence, nonvanishing gradients, or
  absence of barren plateaus.
- It does not by itself make a loss efficient if that loss requires
  exponentially many observables or an inefficiently described mixture.
- It does not prove quantum advantage. Any sampling-hardness statement needs
  separate complexity assumptions and a separately specified circuit family.

### Novelty assessment

**No central novelty credit.** Van den Nest's tractable-state/operator
framework supplies the general probabilistic expectation-estimation route.
Recent IQP work gives the uniform-reference estimator, MMD construction, and
automatic-differentiation argument explicitly. The condition-dependent
product reference changes the sampling law and adds the amplitude ratio, but
does not presently clear the “not a direct corollary” requirement.

## 7. Probability response, observable estimation, and sampling are distinct

| Object | Result here | Classical cost established | Not implied |
|---|---|---|---|
| One full probability `q(x|c)` | T1 finite support and T3 smoothness | None beyond exponential exact checks | Efficient probability evaluation |
| One Pauli moment `f_a(c)` | T2 refined support; T5 estimator | Polynomial per sample for efficiently described generators | All moments at once |
| Sampled Conditional MMD | Mixture/U-statistic consequence of T5 | Polynomial only for a polynomial observable/sample budget | Likelihood evaluation |
| Parameter gradient | Differentiate the same sampled expression | Same active-character work plus gradient output | Favorable landscape or convergence |
| Final output sample | Run the quantum circuit | No classical sampler supplied | Classical simulability or quantum advantage |

This separation is mandatory in the Abstract, Introduction, theorem section,
and conclusion.

## 8. Closest-primary-result matrix

Only original papers and official journal/arXiv records were used. “Delta” is
the narrow difference from that source, not a claim of novelty.

| Candidate result | Closest primary result | What the source already establishes | Remaining delta | Internal verdict |
|---|---|---|---|---|
| T1 full-probability affine `RY` support | [Gil Vidal and Theis, *Input Redundancy for Parameterized Quantum Circuits* (2020)](https://doi.org/10.3389/fphy.2020.00297), Remark 1 and Theorem 4 | Pauli-rotation expectation functions have angle support in `{-1,0,1}^n`, are multiaffine in sines/cosines, and affine scalar encodings map to signed sums; the paper says the several-input generalization is straightforward | Use a projector expectation, write arbitrary vector rows `w_i`, and state the distribution-level consequence | Direct specialization; fail |
| T1 multivariate formulation | [Schuld, Sweke, and Meyer, *The effect of data encoding on the expressive power of variational quantum-machine-learning models* (2021)](https://doi.org/10.1103/PhysRevA.103.032430), especially Appendix A | Quantum model outputs are partial Fourier series; accessible frequencies are encoding-Hamiltonian eigenvalue differences; multivariate inputs are treated explicitly | Specialize to one affine `RY` per qubit and every output projector | Direct specialization; fail |
| T4 fixed multidimensional limitation | [Casas and Cervera-Lierta, *Multidimensional Fourier series with quantum circuits* (2023)](https://doi.org/10.1103/PhysRevA.107.062612) | Multidimensional Fourier expressivity and degree-of-freedom limitations for circuit models | Uniform non-density lemma for this fixed `W` family | Elementary corollary; fail |
| Active-generator cancellation in T2/T5 | [Recio-Armengol, Ahmed, and Bowles, *Train on classical, deploy on quantum* (2026 version)](https://arxiv.org/abs/2503.02934), Proposition 1 and Eqs. (25)--(27) | Only anticommuting IQP generators enter a Pauli expectation; the expectation expands over active subsets, with XOR/identity cancellations; gives an unbiased estimator and differentiable MMD | Evaluate those Pauli terms on a real affine product input and map surviving masks to condition frequencies | Short synthesis; fail as central novelty |
| T5 expectation estimator | [Van den Nest, *Simulating Quantum Computers with Probabilistic Methods* (2010/2011)](https://arxiv.org/abs/0911.1624), Theorem 3 | Polynomial classical estimation for suitable computationally tractable states and efficiently computable sparse operators, without strong simulation of probabilities | Write the specialized product-reference amplitude-ratio variable and its exact second moment | Specialization; fail |
| T5 implementation/gradient meaning | [Recio-Armengol and Bowles, *IQPopt* (2026 version)](https://arxiv.org/abs/2501.04776) | Efficient classical Pauli expectation estimation, automatic differentiation, MMD optimization, and explicit separation from final IQP sampling | Condition batching and active-set cost accounting | Engineering extension; not theorem novelty |
| Commuting-circuit observable locality context | [Ni and Van den Nest, *Commuting Quantum Circuits: Efficient Classical Simulations versus Hardness Results* (2012)](https://arxiv.org/abs/1204.4570) | Classical simulation results for commuting circuits and local measurements; establishes the broader observable-cancellation context | The exact real-product-state span/filter formula | Contextual prior; no rescue of novelty |

## 9. Precise novelty deltas

These are the strongest accurate deltas found. None currently meets the frozen
gate.

1. **Distribution-level packaging.** T1 states the known Fourier theorem for
   every output probability simultaneously and makes explicit that the same
   fixed `Omega_full(W)` controls the whole conditional distribution. This is
   exposition, not a new theorem.
2. **Real-input survivor filter.** T2 groups active-generator subsets by their
   `F_2` XOR mask and discards exactly those masks overlapping the measured
   mask because real `RY` inputs have zero local `Y` expectation.
3. **Condition-frequency lightcone.** T2.1 maps each surviving binary mask to
   signed sums of only the corresponding encoding rows. It can be much tighter
   than T1 for the observable mixture used in training.
4. **One-page finite-resource boundary.** T3 and T4 put bandwidth, derivative
   bounds, and fixed-width non-density beside the support statement.
5. **Conditioned estimator accounting.** T5 records the product-reference
   amplitude ratio, correct finite-variance rather than bounded-sample
   guarantee, active-gradient sparsity, and condition/observable cost axes.

The combined package materially explains the proposed mechanism, but
material explanatory value is not the same as theorem novelty.

## 10. Attempted counterexamples and failure probes

### 10.1 Structural support is not exact realized support

For one qubit with `D=exp(i theta Z)`, the IQP unitary is `exp(i theta X)`.
At `theta=pi/4`, the output `Z` expectation of a real `RY` input is zero for
all angles, so each output probability is constant even though T1 permits
frequencies `+w` and `-w`. Any equality claim for the full structural set is
false.

### 10.2 Repeated rows collide

For three scalar rows `w_1=w_2=w_3=1`, the nominal `3^3=27` signed choices
collapse to the seven frequencies `{-3,-2,-1,0,1,2,3}`. Frequency counts must
deduplicate signed sums.

### 10.3 The union lightcone can be too loose

Take `a=100` and one generator `g=110`. It is active, so the union lightcone
contains qubit 2. The nontrivial propagated term contains `Y` on qubit 1 and
has zero expectation on the real input. The exact response is
`cos(2 theta) cos(alpha_1)` and is independent of `alpha_2`. The `Y` filter is
essential.

### 10.4 Looking at active generators one at a time is too strict

Take `a=100`, `g_1=110`, and `g_2=101`. Each nonzero generator mask overlaps
`a` and is individually removed by the `Y` filter. Their XOR is `011`, which
is disjoint from `a`; the two-generator term is proportional to
`cos(alpha_1) sin(alpha_2) sin(alpha_3)`. Neighboring conditions can enter
jointly even when neither enters through a surviving one-generator term.

### 10.5 Inactive generators really cancel

With the preceding observable, a generator `g_3=011` is inactive because
`<a,g_3>_2=0`. Exact enumeration is unchanged when `theta_3` varies. Its
expectation gradient is identically zero, even though `g_3` overlaps qubits in
the broad geometric neighborhood.

### 10.6 The amplitude-ratio sample is not bounded

Near an input angle where one local post-Hadamard amplitude approaches zero,
the flip ratio from that rare bit value diverges. The rare event probability
shrinks so that the second moment remains at most one. This invalidates a
Hoeffding argument for the raw estimator while leaving the variance and
median-of-means statements intact.

### 10.7 Fixed width misses a continuous family

For any finite `Omega_full(W)`, choose `omega_star` outside it and use the
two-outcome family `1/2 +/- epsilon cos(omega_star^T c)`. T4 gives a positive
uniform approximation floor. Any unconditional fixed-width universality claim
is false.

## 11. Executable checks

The following checks live in
[`tests/test_theorem_checks.py`](../tests/test_theorem_checks.py):

- multivariate signed-sum enumeration, symmetry, and the `3^n` bound;
- repeated-row frequency collisions;
- a sampled exact output probability has no discrete Fourier component
  outside T1's bound;
- active-generator and union-lightcone parity;
- binary-span enumeration;
- the two-qubit real-input `Y` filter and its exact analytic response;
- the three-qubit joint-XOR survivor that imports two neighboring angles only
  together;
- the complete active-subset expansion against exact state-vector
  probabilities for seeded random systems from one through four qubits;
- exact independence from an inactive-generator parameter;
- global and lightcone-restricted bandwidth calculations;
- shape and binary-mask validation.

Existing estimator tests in
[`tests/test_simulator.py`](../tests/test_simulator.py) compare the Monte Carlo
expectations and their autodifferentiated gradients with exact PennyLane
circuits. These checks establish regression evidence, not a formal or external
proof review.

## 12. Internal theorem gate

| Requirement from ADR 0042 | Result | Reason |
|---|---|---|
| Exact statements and assumptions | Pass | Sections 1--6 |
| Complete internal proofs | Pass | Proofs given for T1--T5 |
| Primary-source closest-prior matrix | Pass, non-exhaustive | Section 8 |
| Precise delta for each theorem | Pass | Section 9 |
| Counterexamples and executable checks | Pass | Sections 10--11 |
| At least one correct central theorem | Provisional pass | T1--T5 pass internal checks |
| At least one central theorem not a direct corollary | **Fail** | T1/T3/T4/T5 are direct consequences; T2 is a short synthesis of explicit active-subset and Fourier results |
| Material significance for the fixed-circuit claim | **Not established** | Mechanism is clarified, but the mathematical delta is modest |
| Independent expert review | **Missing** | Required before submission regardless of internal verdict |

**Overall internal gate: FAIL.** Under ADR 0007 and ADR 0042, the major Ising
campaign must remain blocked. This finding should be reported, not overwritten
by changing the threshold after the audit.

### Gate chronology and stopped preparatory work

The repository chronology is consistent with the failure action when
preparatory generation is distinguished from gate release:

1. Target-only calibration and all eight exact target archives were generated
   while the theorem audit was open. The calibration artifact at
   `results/quantum-campaign-v1/calibration.json` has SHA-256
   `8d121e4a519766ff0234e00021815b3bcfd740e9fc9284dbf5b2aa448f29f6ae`,
   exactly the digest declared by the frozen manifest.
2. This matching artifact is preparatory evidence, **not** a passed or
   released `TARGET_CALIBRATION` gate, because the predecessor theorem gate did
   not pass.
3. Complete sampler pilots at 10 and 16 thinning sweeps each failed the frozen
   effective-sample-size rule in 2 of 40 validation cells. Their summary
   artifacts record overall `passed: false`.
4. A 32-thinning attempt was interrupted immediately when the novelty verdict
   became available. It did not produce a complete certification and cannot be
   interpreted as a rescue attempt or a passed sampler gate.
5. No representability, tuning, primary, spectral, scaling, statistical, or
   ancilla training stage was released from this preparatory work.

The failed sampler attempts are reportable negative operational evidence, but
they neither strengthen nor weaken the mathematical novelty verdict.

## 13. What could reopen the gate

The following are research directions, not claims that the gap can be closed:

1. Prove a genuinely IQP-specific quantitative approximation lower bound that
   uses the active-generator code/kernel structure, rather than only a finite
   support inherited from generic encodings.
2. Characterize the dimension or attainable coefficient variety of the
   conditioned observable response, including algebraic constraints between
   coefficients, and show a separation from generic fixed-unitary Fourier
   models.
3. Derive a target-spectral excess-risk lower bound that is both topology
   dependent and experimentally falsifiable, then show it predicts an
   observed interpolation boundary without post-hoc choices.
4. Establish a conditioned estimator result outside the existing
   tractable-state/operator framework, such as a provable variance or runtime
   separation unavailable to the general estimator.
5. Obtain an independent specialist search and proof review. A positive
   external opinion can correct an incomplete search, but should not merely
   relabel the current direct-corollary package.

## 14. Primary sources and search limits

Primary sources consulted:

1. Francisco Javier Gil Vidal and Dirk Oliver Theis,
   [*Input Redundancy for Parameterized Quantum Circuits*](https://doi.org/10.3389/fphy.2020.00297),
   Frontiers in Physics 8, 297 (2020); author version
   [arXiv:1901.11434](https://arxiv.org/abs/1901.11434).
2. Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer,
   [*The effect of data encoding on the expressive power of variational
   quantum-machine-learning models*](https://doi.org/10.1103/PhysRevA.103.032430),
   Physical Review A 103, 032430 (2021); [arXiv:2008.08605](https://arxiv.org/abs/2008.08605).
3. Berta Casas and Alba Cervera-Lierta,
   [*Multidimensional Fourier series with quantum circuits*](https://doi.org/10.1103/PhysRevA.107.062612),
   Physical Review A 107, 062612 (2023); [arXiv:2302.03389](https://arxiv.org/abs/2302.03389).
4. Maarten Van den Nest,
   [*Simulating Quantum Computers with Probabilistic Methods*](https://arxiv.org/abs/0911.1624),
   Quantum Information and Computation 11, 784--812 (2011).
5. Xiaotong Ni and Maarten Van den Nest,
   [*Commuting Quantum Circuits: Efficient Classical Simulations versus
   Hardness Results*](https://arxiv.org/abs/1204.4570) (2012).
6. Erik Recio-Armengol and Joseph Bowles,
   [*IQPopt: Fast optimization of instantaneous quantum polynomial circuits in
   JAX*](https://arxiv.org/abs/2501.04776), v2 (2026).
7. Erik Recio-Armengol, Shahnawaz Ahmed, and Joseph Bowles,
   [*Train on classical, deploy on quantum: scaling generative quantum machine
   learning to a thousand qubits*](https://arxiv.org/abs/2503.02934), v2
   (2026).

The search used exact theorem phrases and combinations of `IQP`,
`anticommuting generator`, `observable support`, `lightcone`, `affine input`,
`multidimensional Fourier`, `product state`, and `expectation estimator`, then
followed primary citations within the closest papers. It was not a systematic
review of every citation descendant, thesis, patent, or non-English record.
Failure to locate an exact formula is not evidence of novelty. The adverse
matches above are already sufficient for the internal fail decision.
