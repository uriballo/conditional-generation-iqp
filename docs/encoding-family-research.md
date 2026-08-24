# Classically tractable condition encodings beyond angle encoding

## Question and conclusion

The current estimator does not fundamentally require a real product state. It
requires a tractable description of the **reference state after the first
Hadamards**. If

\[
|\phi_c\rangle
=H^{\otimes n}E(c)|0^n\rangle
=\sum_z\phi_c(z)|z\rangle ,
\]

then a fixed encoding family remains compatible with the present direct-MMD
training whenever it provides:

1. exact or controlled-precision samples \(z\sim|\phi_c(z)|^2\); and
2. efficient complex-amplitude ratios
   \(\phi_c(z\oplus a)/\phi_c(z)\) for sampled \(z\) and MMD masks \(a\).

This admits a useful hierarchy beyond the present broadcast \(R_Y\) encoding:

1. noncommuting single-qubit reuploading (still a product state);
2. fixed-size entangled block-product encodings;
3. bounded-bond-dimension matrix-product-state (MPS) encodings.

The recommended next experiment is the first two steps, not a jump directly to
MPS, stabilizer, or matchgate states. Product reuploading tests a richer
condition-to-state curve without adding correlations. Two-qubit blocks then
test whether correlated reference amplitudes add value. Both are exact,
straightforward at 19 qubits, and retain the same downstream IQP parameters.

## Why the estimator generalizes

Write the diagonal IQP phase as

\[
D_\theta|z\rangle=e^{i\Lambda_\theta(z)}|z\rangle .
\]

After the final Hadamards, a Pauli-\(Z\) mask \(a\) is equivalent to \(X_a\)
before them. Therefore

\[
\begin{aligned}
\langle Z_a\rangle_{\theta,c}
&=\langle\phi_c|D_\theta^\dagger X_aD_\theta|\phi_c\rangle\\
&=\mathbb E_{z\sim|\phi_c|^2}
\left[
\frac{\phi_c(z\oplus a)}{\phi_c(z)}
e^{i[\Lambda_\theta(z\oplus a)-\Lambda_\theta(z)]}
\right].
\end{aligned}
\]

Terms with \(\phi_c(z)=0\) are never sampled. Moreover, if \(Y_a(z)\) denotes
the complex random variable inside the expectation,

\[
\mathbb E|Y_a|^2
=\sum_{z:\phi_c(z)\ne0}|\phi_c(z\oplus a)|^2
\le 1.
\]

Thus the real-part estimator is unbiased and has variance at most one for
*any* normalized reference state meeting the two computational requirements
above. Product structure is one sufficient representation, not a necessary
one. The present unbiased MMD U-statistic can use these samples unchanged.

This derivation is specific to this repository's circuit and estimator. The
source-backed facts below establish which state representations supply the
required operations efficiently.

## Encoding-family assessment

| Family | Sample and ratio cost | What it adds | Main limitation | Priority |
|---|---:|---|---|---:|
| Same-axis \(R_Y\) repeats | \(O(n)\) | Only a rescaled angle | \(R_Y(x_1)\cdots R_Y(x_L)=R_Y(\sum_l x_l)\); no genuine reuploading without noncommuting gates | Do not test as a new family |
| Noncommuting local reuploading | \(O(n)\) | Richer complex, nonlinear curve \(c\mapsto|\phi_c\rangle\) | Still has no inter-qubit correlations; a single-qubit pure state still has only two physical degrees of freedom at each fixed \(c\) | First |
| Explicit Fourier features in local rotations | \(O(n)\) | Controlled condition harmonics without deeper circuits | Gains come partly from classical preprocessing; current 19-fold broadcast encoding already repeats the feature in parallel | First, paired with reuploading |
| Blocks of \(b\) entangled qubits | Precompute \(O((n/b)2^b)\); then \(O(n/b)\) sampling/ratio lookups | Genuine within-block correlations and phases | Fixed blocks break full permutation symmetry; cost is exponential in \(b\) | Second, begin with \(b=2\) |
| MPS with bond dimension \(\chi\) | Polynomial; amplitudes \(O(n\chi^2)\), conservative sequential sampling \(O(n\chi^3)\) | Correlations extending across the chain | More code, more numerical conditioning, and a less direct hardware-preparation story | Third, only if blocks help |
| Stabilizer/Clifford states | Polynomial-time tableau/quadratic-form algorithms | Long-range entanglement with exact arithmetic | The \(n\)-qubit Clifford family is discrete up to global phase, so it is a poor continuous interpolation encoding | Negative/control study |
| Matchgate/fermionic Gaussian states | Polynomial determinant/Pfaffian machinery | Structured long-range fermionic correlations | Standard states have fixed parity, forcing all odd-weight \(X_a\) expectations to zero; \(H^{\otimes n}\) does not preserve the usual parity-preserving matchgate family | Defer |

### 1. Local data reuploading and Fourier features

Data reuploading alternates data-dependent gates with processing gates.
Pérez-Salinas et al. introduced repeated upload/process units and showed that
even a single qubit can produce a universal classifier when assisted by a
classical routine
([arXiv:1907.02085](https://arxiv.org/abs/1907.02085)). Schuld, Sweke, and
Meyer then showed that quantum models with common encoding gates are partial
Fourier series: encoding-Hamiltonian eigenvalues determine accessible
frequencies, while the surrounding circuit determines their coefficients.
Sequential or parallel repetitions enlarge the frequency spectrum
([arXiv:2008.08605](https://arxiv.org/abs/2008.08605)).

For this project, all upload/process gates can remain single-qubit gates. The
reference state is then

\[
|\phi_c\rangle=\bigotimes_i
H\,u_i(c)|0\rangle,\qquad
u_i(c)=\prod_{\ell=1}^{L}
R_Z(\beta_{i\ell})R_Y(s_{i\ell}c),
\]

so its complex local amplitudes, independent-bit sampling probabilities, and
flip ratios remain analytic. The \(R_Z\) gates matter: deleting them makes all
the \(R_Y\) uploads commute and collapse to one \(R_Y\).

There are two important controls.

- The current broadcast encoding already uploads the same scalar in parallel
  on 19 qubits. It is not a degree-one model in the Fourier sense. Sequential
  depth should therefore be described as increasing the available spectrum
  beyond an already parallel-repeated baseline.
- Any single-qubit sequence is one effective \(SU(2)\) operation at a fixed
  condition. Reuploading adds functional complexity across \(c\), not more
  state-space dimensions at each \(c\).

An explicit Fourier-feature alternative,

\[
u(c)=R_Z(\varphi_K(c))R_Y(\vartheta_K(c)),\qquad
\vartheta_K,\varphi_K
\in\operatorname{span}\{1,\sin(kc),\cos(kc):k\le K\},
\]

has the same tractability. It cleanly controls harmonic content, although the
extra expressivity should be credited partly to classical feature
preprocessing rather than quantum depth.

### 2. Fixed-size entangled blocks

Partition the qubits into blocks \(B\), each of size at most \(b\), and prepare

\[
|\phi_c\rangle=\bigotimes_B|\phi_{c,B}\rangle .
\]

For each condition, enumerate the \(2^b\) complex amplitudes in each block.
Sampling is categorical and independent across blocks, while

\[
\frac{\phi_c(z\oplus a)}{\phi_c(z)}
=\prod_B
\frac{\phi_{c,B}(z_B\oplus a_B)}{\phi_{c,B}(z_B)}.
\]

For fixed \(b\), preprocessing and sampling are linear in \(n\), with a
constant exponential only in \(b\). Unlike local reuploading, this is a genuine
increase in correlation complexity.

A minimal symmetric-within-pair circuit is

\[
u_{ij}(c)
=e^{-i\kappa Z_iZ_j/2}
\bigl(u(c)\otimes u(c)\bigr),
\]

with fixed \(\kappa\). At \(c=0\), the entangler contributes only a global
phase; away from that endpoint it generally entangles the pair. With 19
qubits, use nine pairs and one singleton, rotate which qubit is the singleton
across matched seeds, and report sensitivity to the pairing. That is necessary
because a disjoint block partition reduces the full permutation symmetry of
the Gaussian Hamming-weight targets.

### 3. Bounded-bond-dimension MPS

An open-boundary MPS represents

\[
\phi_c(z)=A_1^{z_1}(c)A_2^{z_2}(c)\cdots A_n^{z_n}(c).
\]

Contracting this chain gives any requested amplitude in polynomial time, and
canonical conditional norms give exact sequential samples. Local Hadamards act
on physical tensor indices and do not increase the bond dimension. Vidal's
classical simulation scheme scales linearly in system size and exponentially
in the entanglement, which is the relevant fixed-/bounded-\(\chi\) guarantee
([arXiv:quant-ph/0301063](https://arxiv.org/abs/quant-ph/0301063)).

MPS is therefore a valid generalization, but it is not the best first
experiment. A shallow nearest-neighbor encoding must be converted to and
maintained as an MPS; every MMD sample/mask then requires tensor contractions;
and an arbitrary MPS may need a sequential isometry/ancilla preparation rather
than the simple gate-only encoder used in the current paper.

### 4. Stabilizer encodings

Clifford circuits are efficiently classically simulable
([Aaronson and Gottesman, arXiv:quant-ph/0406196](https://arxiv.org/abs/quant-ph/0406196)).
More specifically for this estimator, stabilizer-state coefficients in the
computational basis have efficient binary quadratic-form descriptions
([Dehaene and De Moor, arXiv:quant-ph/0304125](https://arxiv.org/abs/quant-ph/0304125)).
Consequently sampling and complex-amplitude queries are both tractable, and
the first Hadamards preserve the family.

The drawback is scientific rather than computational: Clifford states form a
discrete family at fixed \(n\). Selecting one stabilizer state per condition
tests categorical lookup, not smooth conditional geometry. Adding generic
continuous rotations destroys exact stabilizer closure. Stabilizers are
therefore suitable as a discrete entangled control, not the main interpolation
encoding.

### 5. Matchgate/fermionic-Gaussian encodings

Nearest-neighbor matchgate circuits and fermionic linear optics admit efficient
classical simulation
([Jozsa and Miyake, arXiv:0804.4050](https://arxiv.org/abs/0804.4050);
[Terhal and DiVincenzo, arXiv:quant-ph/0108010](https://arxiv.org/abs/quant-ph/0108010)).
Fermionic Gaussian maps also have explicit Grassmann/Gaussian integral
representations
([Bravyi, arXiv:quant-ph/0404180](https://arxiv.org/abs/quant-ph/0404180)),
which supply determinant/Pfaffian-style amplitude calculations.

Two repo-specific obstacles make this a poor near-term choice.

1. Standard matchgates preserve fermion parity. A fixed-parity
   \(|\phi_c\rangle\) has
   \(\phi_c(z\oplus a)=0\) for every odd-weight mask \(a\), so the corresponding
   model Pauli expectations are identically zero. The Hamming-RBF MMD uses such
   masks, and the present asymmetric-weight targets need not respect this
   restriction.
2. The circuit's first \(H^{\otimes n}\) is not a parity-preserving matchgate
   layer. Tractability must be arranged for the state *after* it, for example
   by defining the physical encoder so that this Hadamard layer cancels before
   a matchgate state. It does not follow merely from choosing a matchgate
   encoder before the existing Hadamards.

## Fixed versus trainable encoders

For a fixed encoder, the reference samples and amplitude ratios do not depend
on the trainable IQP phases \(\theta\). Ordinary autodifferentiation through
the estimator therefore gives the same direct path used in the current code.
This remains true for every family above.

If encoder parameters \(\eta\) are trained and they change
\(p_\eta(z)=|\phi_\eta(z)|^2\), the samples themselves depend on \(\eta\).
Holding sampled bitstrings fixed and differentiating only the ratio is then
not the full gradient:

\[
\nabla_\eta\mathbb E_{p_\eta}[f_\eta(z)]
=\mathbb E_{p_\eta}
\left[\nabla_\eta f_\eta(z)
+f_\eta(z)\nabla_\eta\log p_\eta(z)\right].
\]

The second, score-function term is required for an unbiased gradient through a
discrete stochastic node; stochastic-computation-graph calculus gives the
general result
([Schulman et al., arXiv:1506.05254](https://arxiv.org/abs/1506.05254)).
The MMD U-statistic needs the corresponding term for every model sample in
each product. It is feasible, but likely higher variance and is a different
training algorithm from the current direct-backpropagation claim.

There are three exceptions/alternatives.

- Encoder parameters that change only \(\arg\phi_\eta(z)\), while leaving
  \(|\phi_\eta(z)|^2\) fixed, retain ordinary direct backpropagation because the
  sampling law is unchanged.
- Exact enumeration at \(n=19\) can differentiate the full finite sum, but
  gives up the polynomial classical-training claim.
- Sampling from a parameter-independent proposal and importance weighting
  also restores a direct derivative, but without a bounded-weight condition
  its variance can grow exponentially.

The clean first study should therefore keep all encoder coefficients fixed and
train only the shared IQP parameters. A later phase-only trainable encoder is
the safest extension if joint encoder training becomes important.

## Recommended experiment ladder

### Phase A: product-state geometry

Use the complete 2-local IQP ansatz first and retain the present broadcast
\(R_Y(c)\) model as the baseline.

1. Test local noncommuting reupload depths \(L\in\{2,3\}\), using a
   predeclared sequence such as alternating \(R_Y(s_Lc)\) and
   \(R_Z(\pi/2)\).
2. Choose \(s_L\) before training so that endpoint fidelity between the two
   local encoded states matches the baseline. This prevents improved
   condition distinguishability alone from masquerading as improved
   conditional geometry.
3. Add one explicit harmonic-path control whose extra sine terms vanish at
   the original endpoints. This changes only the interpolation path and tests
   whether higher curvature/harmonics improve held-out angles.

### Phase B: correlation complexity

1. Promote the best predeclared product sequence into two-qubit blocks.
2. Test fixed entangling strengths
   \(\kappa\in\{\pi/8,\pi/4\}\), reporting both rather than selecting on held-out
   test angles.
3. Use nine pairs plus one singleton at 19 qubits. Rotate the singleton and use
   matched pairings/seeds.
4. Only if \(b=2\) improves held-out fidelity, test \(b=3\) or a shallow
   bounded-\(\chi\) MPS. Do not begin with a generic MPS optimizer.

### Evaluation discipline

- Reuse the current endpoint task and five-train/four-held-out interpolation
  task. Changing the dataset at the same time would confound the encoding
  comparison.
- Keep IQP topology, trainable parameter count, optimizer, target samples, and
  matched seeds fixed.
- Re-run estimator calibration for every encoding family. The universal
  second-moment bound is unchanged, but actual variance and numerical ratio
  stability can differ substantially.
- Before training, report encoded-state endpoint fidelity, adjacent-angle
  fidelity/curvature, reference entropy, and empirical ratio variance. These
  identify whether a gain comes from condition separation, path geometry, or
  estimator behavior.
- Report exact endpoint and held-out shell/bitstring TVD, paired seed
  differences, wall-clock cost, and gradient variance. A compelling result is
  a held-out improvement that survives matched-seed comparison without an
  endpoint regression.
- If a family wins on the complete graph, then repeat it on lattice and
  heavy-hex. This avoids mixing an encoding question with the existing
  topology ablation in the initial screen.

## Suggested implementation boundary

Replace the angle-specific estimator dependency with a reference-state
interface exposing:

```text
sample(key, n_samples) -> bitstrings
amplitude_ratio(bitstrings, masks) -> complex ratios
```

Optionally add `statevector_small_n()` for exact verification and a
gate-description method for PennyLane checks. Product, block-product, and
later MPS implementations can then share the IQP phase-difference and MMD
code. The first implementation should support complex ratios from the outset;
otherwise noncommuting reuploading will be accidentally reduced back to the
current real-amplitude special case.
