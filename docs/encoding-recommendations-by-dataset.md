# Encoding recommendations by dataset

## Bottom line

The two datasets should not get the same next encoding experiment.

- For the Gaussian Hamming-weight family, try a permutation-symmetric shell-amplitude reference state. It can control the location and width of the reference law separately while retaining exact sampling and amplitude ratios. The current product encoding ties those quantities together.
- For the Curie--Weiss family, first try a role-tied product map that sends different fixed functions of temperature to visible and ancilla qubits. If that fails, try a spin-flip-symmetric two-branch state with bond dimension two. The local evidence points to latent branch capacity, especially near the transition, rather than a shortage of visible IQP terms.

These are hypotheses, not literature-backed performance promises. The primary papers establish expressivity, symmetry, preparation, or classical tractability. The performance ranking comes from the structure of these targets and this repository's experiments.

## The admissibility test

Let the state after the first Hadamards be

\[
|\phi_c\rangle=\sum_z\phi_c(z)|z\rangle.
\]

For the current estimator, a fixed encoding is admissible when it supplies both:

1. samples from \(|\phi_c(z)|^2\); and
2. the complex ratio \(\phi_c(z\oplus a)/\phi_c(z)\) for every sampled \(z\) and MMD mask \(a\).

Van den Nest's definition of a computationally tractable state asks for efficient sampling and efficient amplitude queries. It is a clean sufficient condition, although the ratio query above is the narrower requirement here ([Van den Nest, 2010](https://arxiv.org/abs/0911.1624)). A sampler without amplitude information does not pass this test.

All recommendations below keep the encoder fixed during each training run. If encoder parameters change the reference sampling law during training, differentiating only through the sampled estimator misses the score-function term. That would be a different optimization method and should not be folded into the first comparison.

## What the local evidence says

The existing Gaussian screen already rules out two easy stories.

- A calibrated noncommuting two-upload product map was worse than the angle baseline on held-out profile TVD by \(0.0106\), with a 95% paired interval of \([0.0038,0.0174]\).
- Two-qubit \(ZZ\) blocks were also worse on average. The \(\pi/8\) block lost by \(0.0086\), with interval \([0.0031,0.0146]\), and won only one of ten seeds.

See [the encoding benchmark](../results/encoding_benchmark/summary/metrics.md) and [the geometry study](../results/encoding/geometry/metrics.md). The first reupload path increased path length and reference entropy, but that extra geometry did not improve the profile fit. More local depth or arbitrary pair correlations are therefore weak next bets.

The Curie--Weiss evidence points elsewhere. Five ancillas beat a matched-parameter visible one-, two-, and three-local model by \(0.118\) profile TVD in the critical region and \(0.184\) bitstring TVD there. The cold-region profile effect was only \(0.007\), with an interval crossing zero. See [the matched-capacity control](../results/curie-weiss-capacity-control-v2/summary.md). This is consistent with a need for latent branch structure near \(\beta=1\), not simply more visible parameters. It does not prove that a particular encoding will supply that structure.

Both targets are uniform within Hamming-weight shells. The Gaussian symmetry intervention also showed that permutation averaging can sharply reduce bitstring TVD while preserving the shell profile, although it is a post-measurement protocol rather than a learned circuit result. See [the symmetry interpretation](../results/symmetry/summary/interpretation.md). Breaking visible-qubit permutation symmetry without a matched reason is hard to justify.

## A useful tractable family

For either dataset, define a radial reference state by a shell law \(q_c(k)\) and a shell phase \(\psi_c(k)\):

\[
\phi_c(z)=
\sqrt{\frac{q_c(|z|)}{\binom{n}{|z|}}}
e^{i\psi_c(|z|)}.
\]

Sampling is exact. Draw \(k\sim q_c\), then draw a uniformly random string of weight \(k\). If \(k'=|z\oplus a|\), then

\[
\frac{\phi_c(z\oplus a)}{\phi_c(z)}
=
\sqrt{
\frac{q_c(k')\binom{n}{k}}
     {q_c(k)\binom{n}{k'}}
}
e^{i[\psi_c(k')-\psi_c(k)]}.
\]

After tabulating \(q_c(0),\ldots,q_c(n)\), sampling and ratio evaluation are polynomial and cheap. The state preserves every visible-qubit permutation by construction. It also has a plausible circuit path. Dicke states are the equal superpositions within one shell, and Bärtschi and Eidenbenz give deterministic circuits for Dicke states and arbitrary symmetric pure states, including linear-depth constructions ([Bärtschi and Eidenbenz, 2019](https://arxiv.org/abs/1904.07358)).

This family is broad enough to cheat. Setting \(q_c\) from the exact target or its exact Hadamard transform turns the encoder into an oracle. Such a run is useful only as an upper bound. A fair primary experiment must freeze a low-dimensional, target-independent functional form before looking at test results.

## Gaussian Hamming-weight family

### Ranking

| Rank | Candidate | Why it may help | Tractability | Main risk |
|---:|---|---|---|---|
| 1 | Symmetric quadratic shell map | Separates reference location from dispersion while matching the target symmetry | Exact shell sampling; ratio from two table entries | Easy to inject the known target formula and make the comparison circular |
| 2 | Symmetry-tied nonlinear product map | Adds controlled condition frequencies at the lowest implementation cost | Independent-bit sampling and analytic complex ratios in \(O(nL)\) | A tested two-upload version already lost; gains are unlikely unless the new map changes the right geometry |
| 3 | Permutation-symmetric quadratic phase on a broadcast product | Adds a condition-dependent collective phase without changing the reference sampling law | Product sampling; ratio phase depends only on \(|z|\) and \(|z\oplus a|\) | It acts like a fixed condition-dependent two-local IQP term and may blur the shared-parameter claim |
| 4 | Fixed-size blocks or generic bounded-bond MPS | Adds correlations beyond a radial law | Blocks cost \(O((n/b)2^b)\); bounded-\(\chi\) MPS stays polynomial | Blocks break the target symmetry and already lost; MPS adds code and contractions without a clear missing mechanism |

### 1. Symmetric quadratic shell map

Use the generic exponential family

\[
q_c(k)\propto
\exp\left[\lambda_1(c)k+\lambda_2 k^2/n\right],
\qquad \psi_c(k)=0,
\]

with \(\lambda_1(c)=u_0+u_1c\) and one fixed \(\lambda_2\) per width campaign. The linear tilt moves the distribution through the shells. The quadratic coefficient changes its dispersion, something a broadcast product state cannot do independently of its mean.

The scientific control matters more than the implementation. Choose \((u_0,u_1,\lambda_2)\) from a target-free rule such as fixed endpoint reference moments, then freeze them for all optimizer seeds. Also report an oracle-calibrated version separately. Do not label the oracle version as evidence that the IQP model learned the Gaussian family.

Minimal fair experiment:

- Use \(n=19\), five training conditions, all three width factors, and ten matched seeds.
- Keep the IQP ansatz, optimizer, MMD masks, sample counts, and checkpoint rule unchanged.
- Compare angle encoding against three preregistered \(\lambda_2\) values: underdispersed, neutral, and overdispersed.
- Use held-out profile TVD as the primary metric. Keep bitstring TVD, mean error, and width error as co-diagnostics.
- Reject a candidate if it improves only endpoint fits or only the target-informed oracle calibration.

### 2. Symmetry-tied nonlinear product map

If implementation speed matters, use the same single-qubit state on every visible qubit but replace the affine angle by a small Chebyshev or Fourier basis. Repeated encoding gates enlarge the accessible frequency spectrum of quantum models ([Schuld, Sweke, and Meyer, 2021](https://arxiv.org/abs/2008.08605)). Noncommuting data reuploading can produce a richer single-qubit function of the input while preserving product-state tractability ([Pérez-Salinas et al., 2020](https://arxiv.org/abs/1907.02085)).

The previous reupload benchmark makes this a second choice. A fair retry should test a monotone two- or three-coefficient map, not another arbitrary oscillatory path. Match endpoint states and the mean adjacent-state fidelity to the angle baseline. This separates useful response curvature from the harmful extra path length already observed.

### 3. Collective quadratic phase

Take the existing broadcast product amplitudes and add

\[
\psi_c(k)=\gamma(c)(2k-n)^2/n
\]

with one fixed odd or centered function \(\gamma(c)\). Sampling remains unchanged. The ratio only needs the two weights \(k\) and \(k'\). This preserves permutation symmetry and supplies condition-dependent pairwise phase structure.

Use it as a mechanism test. Because the phase is diagonal in the reference basis, it is equivalent to putting a fixed condition-dependent collective quadratic term next to the trainable diagonal IQP unitary. The paper would need to say that clearly.

## Curie--Weiss temperature family

### Ranking

| Rank | Candidate | Why it may help | Tractability | Main risk |
|---:|---|---|---|---|
| 1 | Role-tied visible/ancilla product map | Uses the already successful latent register while preserving visible permutation symmetry | Independent-bit sampling and analytic ratios | More condition features may still fail to create the required two branches |
| 2 | Spin-flip-symmetric two-branch state, \(\chi=2\) | Represents the ordered pair of magnetization modes directly | Radial shell sampling or bounded-\(\chi\) MPS contraction | Target-aligned mean-field choices can bake in the transition |
| 3 | Tempered Curie--Weiss shell map | Supplies global, permutation-symmetric correlations with one strength parameter | Exact \(O(n)\) shell table and ratios | Too close to the known target if the strength is not attenuated and preregistered |
| 4 | Generic bounded-bond MPS | Can add longer-range correlations if the two-branch model is too rigid | Polynomial for fixed bond dimension | A chain ordering mismatches all-to-all symmetry; extra cost may buy the wrong correlations |

### 1. Role-tied visible/ancilla product map

The current five-ancilla campaign broadcasts the same piecewise-linear temperature angle across visible and ancilla qubits. The next cheap test is to retain a single shared visible angle but give the ancilla register its own fixed features:

\[
u_v(\beta)=R_Y[\theta_v(\beta)],
\qquad
u_{a,j}(\beta)=R_Z[\varphi_j(\beta)]R_Y[\theta_{a,j}(\beta)].
\]

Keep \(\theta_v\) equal to the frozen map. Use a small centered basis on the ancillas, for example \(1\), \(\beta-1\), and \((\beta-1)^2\), distributed across ancilla qubits with fixed coefficients. This preserves permutation symmetry on the measured spins while allowing the latent register to distinguish hot, critical, and cold regimes.

Minimal fair experiment:

- Use the existing eight-visible, five-ancilla campaign and its six training plus five held-out temperatures.
- Compare the current broadcast-all map with one role-tied linear map and one role-tied quadratic map.
- Match the IQP parameter count exactly. Do not train the encoding coefficients in this first screen.
- Keep critical-region profile TVD and held-out bitstring TVD as co-primary metrics. Report cold-region profile TVD separately because the existing ancilla advantage there is uncertain.
- Use ten paired seeds and the same validation checkpoints.

This is the strongest near-term recommendation because it follows the one positive local mechanism result and costs almost nothing in estimator complexity.

### 2. Spin-flip-symmetric two-branch state

Use an equal coherent superposition of two broadcast product states related by a global bit flip:

\[
|\phi_\beta\rangle\propto
|p_\beta\rangle^{\otimes n}
+X^{\otimes n}|p_\beta\rangle^{\otimes n}.
\]

Its amplitude depends only on Hamming weight, so the radial formulas above give exact sampling and ratios. It also has tensor rank two and an MPS representation with bond dimension at most two. Matrix-product states have canonical forms and polynomial classical contraction algorithms at fixed bond dimension ([Pérez-García et al., 2006](https://arxiv.org/abs/quant-ph/0608197)); Vidal's simulation analysis gives the corresponding bounded-entanglement rationale ([Vidal, 2003](https://arxiv.org/abs/quant-ph/0301063)).

The two branches encode the \(m\leftrightarrow-m\) symmetry of the zero-field target. Let their separation be zero for \(\beta\le1\) and increase smoothly above one. A mean-field choice based on \(m=\tanh(\beta m)\) is a useful physics-informed variant, but it should be labeled as such. The fairer primary screen uses two fixed post-critical slopes chosen without held-out results.

Compare this state with a one-branch radial state having the same shell mean and entropy. That control tests whether bimodality itself helps rather than merely changing reference concentration.

### 3. Tempered Curie--Weiss shell map

The exact target shell law is

\[
q_\beta(k)\propto
\binom nk
\exp\left[\frac{\beta}{2n}(2k-n)^2\right].
\]

This formula follows directly from the repository's Hamiltonian \(-M^2/(2n)\). Use a tempered reference with \(\beta\) replaced by \(\rho\beta\), where \(\rho\in\{0.25,0.5\}\) is frozen. This gives the reference state global exchangeable correlations without handing it the full target. Sampling and ratios use an \(n+1\)-entry shell table.

Include \(\rho=0\) as the independent reference and \(\rho=1\) only as a target-informed upper-bound diagnostic. The output distribution is not automatically the target because this is the state before the diagonal IQP layer and final Hadamards. Even so, \(\rho=1\) uses the exact target energy and should not be treated as a fair learned baseline.

### 4. Generic MPS

A condition-dependent MPS with bounded bond dimension passes the estimator test. An amplitude query is a chain contraction, and canonical conditional norms support sequential samples. Local Hadamards do not increase its bond dimension. For fixed \(\chi\), this remains polynomial rather than product-state cheap.

Do not start here. Curie--Weiss correlations are exchangeable and all-to-all, while a generic chain MPS privileges an ordering. The spin-flip two-branch construction already captures the main expected low-rank structure at \(\chi=2\). Only move to \(\chi=4\) after a two-branch win, and compare against a parameter-matched radial state.

## Candidates to defer or exclude

- Repeating only \(R_Y\) is not a richer encoding. Same-axis rotations collapse to one rotation.
- A black-box classical sampler fails the amplitude-ratio requirement.
- Fixed disjoint entangled blocks are a poor fit for these exchangeable targets unless the final prediction averages over pairings. The Gaussian block benchmark already gives negative evidence.
- A generic deep encoding circuit does not inherit tractability merely because it is easy to run on a quantum device.
- Training encoder parameters with the current backpropagation code is not valid when those parameters alter \(|\phi_c|^2\). Add the score term, exact enumeration, or a separate gradient estimator first.
- An exact target-derived symmetric state is an oracle control. It can measure headroom, but it cannot support a claim about learned conditional generation.

## Recommended order of work

1. Run the Curie--Weiss role-tied visible/ancilla product screen.
2. Run the Gaussian symmetric quadratic shell screen.
3. If Curie--Weiss still fails near \(\beta=1\), test the \(\chi=2\) spin-flip two-branch state.
4. Test collective shell phases only as a labeled mechanism study.
5. Move to generic MPS only after a simpler correlated state wins a matched comparison.

The largest uncertainty is not classical tractability. All five leading candidates pass that test. The uncertainty is whether changing the reference path helps a shared diagonal model after the final Hadamards. The earlier Gaussian reupload and block losses show that extra reference-state complexity can easily make interpolation worse.
