# Primary-source fact-check of the tractability claim

## Scope and source versions

This note checks the proposed tractability paragraph against three primary sources:

- Maarten Van den Nest, ["Simulating Quantum Computers with Probabilistic Methods"](https://arxiv.org/abs/0911.1624), especially Definition 1 and Theorem 3. The journal version appeared in *Quantum Information and Computation* 11, 784--812 (2011).
- Erik Armengol and Joseph Bowles, ["IQPopt: Fast Optimization of Instantaneous Quantum Polynomial Circuits in JAX"](https://arxiv.org/abs/2501.04776), arXiv v2 dated 7 May 2026, especially Proposition 1 and Sections 2.1, 2.4, and 3.3.1.
- Erik Recio-Armengol, Shahnawaz Ahmed, and Joseph Bowles, ["Train on Classical, Deploy on Quantum: Scaling Generative Quantum Machine Learning to a Thousand Qubits"](https://arxiv.org/abs/2503.02934), arXiv v2 dated 6 February 2026, especially Section 4, Propositions 1--3, and Appendix A.

The conclusion is favorable, with qualifications. Van den Nest supports the full computationally tractable state claim. The weaker "sampling plus amplitude ratios" condition is not a definition or theorem stated in that paper. It is a valid specialization for the one-sparse operator that occurs here, provided the manuscript gives the short derivation and states the efficiency, precision, and fixed-encoder assumptions.

## What Van den Nest actually assumes

Van den Nest defines a computationally tractable, or CT, state as a family of states with two classical algorithms. The first samples an \(n\)-bit string \(x\) from \(\lvert\langle x\vert\psi\rangle\rvert^2\) in polynomial time. The second computes \(\langle x\vert\psi\rangle\) in polynomial time for any supplied \(x\). The definition assumes exact coefficient evaluation for convenience, then says that evaluation to \(m\) significant bits in time polynomial in \(n\) and \(m\) also suffices. CT status is relative to a supplied classical description of the state, and the paper treats state families rather than isolated finite instances. See [Definition 1 and its precision and description remarks](https://arxiv.org/pdf/0911.1624#page=8).

Theorem 3 is more general than the estimator needed in this manuscript. It considers two CT states and an efficiently computable sparse, or ECS, operator \(A\) with \(\lVert A\rVert\leq 1\). ECS means that every row and column has only polynomially many nonzero entries and that an algorithm can enumerate their positions and values in polynomial time. Under those assumptions, the matrix element \(\langle\varphi\vert A\vert\psi\rangle\) has an inverse-polynomial additive approximation in polynomial time. The paper's convention also requires an exponentially small failure probability. See [Theorem 3 and its proof](https://arxiv.org/pdf/0911.1624#page=12).

This has three consequences for our wording.

1. "Efficient basis sampling and individual amplitude evaluation" accurately summarizes the CT definition.
2. CT is a sufficient condition, not the weakest possible condition for our particular observable estimator.
3. A citation to Van den Nest alone does not establish the weaker ratio-only interface. We must derive that specialization in our paper.

## The IQP specialization in the cited literature

IQPopt starts from the standard IQP reference state \(\lvert+\rangle^{\otimes n}\). After conjugating an output Pauli word \(Z_a\) through the final Hadamards, the observable becomes \(X_a\). Conjugating \(X_a\) through the diagonal phase operator yields a one-sparse operator. IQPopt then samples \(z\) uniformly and evaluates a cosine of a phase difference. Its Proposition 1 states that this produces an unbiased inverse-polynomial-precision estimator in polynomial time and space. See [IQPopt, Section 2.1 and Proposition 1](https://arxiv.org/pdf/2501.04776#page=4).

IQPopt also imposes a condition that should remain visible in our paper: the objective must have an efficient description in Pauli-\(Z\) expectation values. The fact that Pauli moments form a basis does not help if exponentially many moments are needed. IQPopt says this explicitly after Definition 2 and gives batched costs in Proposition 2. See [IQPopt, Definition 2 and Sections 2.1--2.2](https://arxiv.org/pdf/2501.04776#page=4).

The train-classical/deploy-quantum paper combines the Pauli-character decomposition of the Hamming Gaussian MMD with this IQP estimator. It constructs an unbiased MMD estimator by sampling both character masks and reference bit strings, and it uses automatic differentiation of the resulting finite-sample expression. See [the main construction](https://arxiv.org/pdf/2503.02934#page=9) and [Appendix A](https://arxiv.org/pdf/2503.02934#page=43). Both papers use the uniform reference state. Neither paper proves the nonuniform, condition-dependent amplitude-ratio formula proposed here.

## Why the weaker ratio interface is sufficient here

Let

\[
\psi_c(z)=\langle z\vert H^{\otimes n}\vert\phi(c)\rangle,
\qquad
D_{\boldsymbol\theta}\lvert z\rangle=e^{if_{\boldsymbol\theta}(z)}\lvert z\rangle.
\]

For an output character \(Z_a\), direct expansion gives

\[
\begin{aligned}
\mu_{q_{\boldsymbol\theta}}(a\mid c)
&=\langle\psi(c)\vert D_{\boldsymbol\theta}^{\dagger}X_aD_{\boldsymbol\theta}\vert\psi(c)\rangle\\
&=\operatorname{Re}\,\mathbb E_{z\sim\lvert\psi_c\rvert^2}
\left[
\frac{\psi_c(z\oplus a)}{\psi_c(z)}
e^{i[f_{\boldsymbol\theta}(z\oplus a)-f_{\boldsymbol\theta}(z)]}
\right].
\end{aligned}
\]

The denominator is nonzero almost surely under the sampling distribution. A zero numerator simply produces a zero sample. More importantly, the raw ratio need not be bounded, but it has a bounded second moment:

\[
\mathbb E_{z\sim\lvert\psi_c\rvert^2}
\left|\frac{\psi_c(z\oplus a)}{\psi_c(z)}\right|^2
=\sum_{z:\psi_c(z)\neq0}\lvert\psi_c(z\oplus a)\rvert^2\leq1.
\]

The sample mean therefore estimates one Pauli moment to additive error \(\epsilon\) with \(O(1/\epsilon^2)\) samples at constant success probability. A median-of-means construction raises the success probability with only a logarithmic factor. This recovers the polynomial-accuracy conclusion needed for training. It is a direct specialization of the CT--ECS method, but the ratio formula and the second-moment proof belong to this manuscript.

This point corrects a tempting overstatement. Hoeffding's inequality does not apply to the raw ratio estimator because an individual ratio can be arbitrarily large on a rare sampled string. The correct argument uses finite variance, Chebyshev, or median of means.

## Qualifications the manuscript should state

### Uniform and efficient descriptions

The sampling and ratio algorithms must run in polynomial time uniformly over the condition \(c\) and every mask \(a\) used by the loss. The classical description of \(c\), the encoder, and the diagonal generator list must have polynomial length. For continuous conditions, the requested numerical precision must also have a polynomial-size description.

The phase difference must be computable in polynomial time. This holds for a polynomial-size IQP diagonal. The complete one- and two-body circuit used in the experiments has \(O(n^2)\) generators, so it satisfies this requirement. It would not hold for an arbitrary diagonal specified by exponentially many phases.

### Ratio precision

The weaker interface should require direct evaluation of the ratio to sufficient numerical accuracy in polynomial time. It is unsafe to infer a numerically stable ratio oracle from approximate amplitude queries when the denominator may be extremely small. Van den Nest avoids this wording by taking exact amplitude computation in the formal CT definition and only then discussing an exponential-precision extension. Our theorem statement should make ratio accuracy an explicit assumption.

### Support

The ratio only needs to be defined when \(\psi_c(z)\neq0\), since the sampler never returns a zero-probability string. No full-support assumption is needed. The second-moment statement is an inequality rather than an equality when XOR by \(a\) moves some strings outside the support.

### Number of observables

Classical estimation of one Pauli moment does not make an exponentially large Pauli expansion efficient. The loss must use a polynomial number of explicitly listed moments or an efficiently sampled moment mixture with a polynomial sample budget. The character-sampled MMD meets this condition.

### Gradients and the encoder

IQPopt obtains phase gradients by differentiating a deterministic finite-sample computation after fixing the sampled reference strings. The same reasoning applies here because the encoder, its sampling law, and its amplitude ratios are fixed with respect to \(\boldsymbol\theta\). For the one- and two-body phases, each derivative of the phase difference is bounded by a constant, so individual gradient estimators retain finite variance.

This does not cover training encoder parameters that change \(\lvert\psi_c(z)\rvert^2\). Such parameters need an additional score-function, reparameterization, or analytic-gradient argument. It also does not prove optimizer convergence, nonvanishing gradients, or absence of barren plateaus.

For MMD, one must differentiate an unbiased finite-sample loss estimator. Squaring a single noisy moment estimate introduces bias. The independent cross-product or U-statistic construction used in the train-classical/deploy-quantum paper and in our derivation avoids that problem.

### Training is not output simulation

The primary sources support a sharp separation. They give classical additive estimation of selected observables and objectives built efficiently from them. They do not give exact output probabilities or a classical sampler from the final IQP distribution. IQPopt and the train-classical/deploy-quantum paper make this distinction explicitly. See [IQPopt's overview](https://arxiv.org/pdf/2501.04776#page=2) and [the train/deploy description](https://arxiv.org/pdf/2503.02934#page=3).

Sampling hardness is a separate, assumption-dependent statement about specified IQP families. A tractable conditional encoder does not by itself prove that every resulting conditioned circuit remains hard to sample. The manuscript should claim classical trainability of the stated loss and phase gradients. Any quantum-advantage wording should cite a separate hardness result and specify the applicable circuit family and assumptions.

## Recommended replacement paragraph

The proposed paragraph is sound after adding the uniformity and precision qualifications:

> To identify conditional encoders that preserve classical estimation, let \(\psi_c(z)=\langle z\vert H^{\otimes n}\vert\phi(c)\rangle\) denote the computational-basis amplitudes of the state entering the diagonal layer. Uniformly over the conditions and bit masks used by the loss, we require polynomial-time classical algorithms that sample \(z\) from \(\lvert\psi_c(z)\rvert^2\) and evaluate the ratio \(R_c(z,a)=\psi_c(z\oplus a)/\psi_c(z)\) to the precision required by the estimator. The ratio need only be defined on the support of \(\lvert\psi_c\rvert^2\). Computationally tractable states in the sense of Van den Nest, which permit efficient basis sampling and individual amplitude evaluation, provide a sufficient condition under exact or suitably controlled arithmetic. Our estimator uses the narrower sampling-and-ratio interface above. For a polynomial-size IQP diagonal, these operations yield polynomial-time additive estimators of the Pauli-\(Z\) moments entering the loss. They do not provide exact output probabilities or efficient classical samples from \(q_{\boldsymbol\theta}(x\mid c)\).

The training section should then show the displayed ratio estimator, its second-moment bound, and the fixed-encoder gradient qualification. Calling the ratio interface "weaker" is mathematically reasonable because it asks for less information than arbitrary amplitude queries. It should be presented as our sufficient interface for this one-sparse estimator, not as Van den Nest's formal definition.

## Remaining uncertainty

The primary papers do not analyze this condition-dependent nonuniform reference state. The generalized ratio estimator is supported by the direct calculation above and by Van den Nest's broader CT--ECS theorem, but its exact numerical implementation assumptions are ours to state. A fully formal complexity theorem would need to define the representation of \(c\), the precision model for the ratio oracle, and the requested failure probability. The main text can stay lighter if the supplement gives those details.
