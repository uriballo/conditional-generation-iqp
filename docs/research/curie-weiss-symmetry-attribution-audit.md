# Curie--Weiss symmetry encoder: attribution audit

## Question

Does Section 6 of Recio-Armengol, Ahmed, and Bowles already contain the central idea behind our Curie--Weiss spin-flip encoder, and how should the manuscript distinguish the two constructions?

## Primary source

Erik Recio-Armengol, Shahnawaz Ahmed, and Joseph Bowles, *Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits*, arXiv:2503.02934v2 (6 February 2026): <https://arxiv.org/abs/2503.02934v2>.

The comparison below uses Section 6, “Beyond IQP: incorporating symmetry into the ansatz,” and its 2D Ising experiment in Section 8.4.

## Verdict

The recollection is correct. Section 6 already establishes the core symmetry mechanism that we use: choose an input state that is a (+1) eigenstate of the global bit-flip operator (X^{\otimes N}), exploit the fact that the IQP unitary commutes with that operator, and thereby enforce

\[
q_\theta(x)=q_\theta(\bar x).
\]

The paper realizes this mechanism with the fixed GHZ input

\[
|\mathrm{GHZ}\rangle=\frac{|0^N\rangle+|1^N\rangle}{\sqrt 2},
\]

proves the resulting output invariance, and derives an unbiased classical expectation-value estimator for this modified input. It then uses the construction for one unconditional, zero-field 2D Ising distribution. We therefore must cite this paper for the symmetry principle, the GHZ construction, its invariance argument, and the precedent for retaining classical training after modifying the input state.

## Exact overlap

Both works:

- impose the same global \(\mathbb Z_2\) bit-flip invariance;
- impose it by modifying the input state rather than the shared commuting unitary;
- require the input to be an eigenstate of (X^{\otimes N});
- rely on commutation of the IQP unitary with (X^{\otimes N}) to transfer the input symmetry to the output probabilities; and
- retain classical estimation of the Pauli expectations used by the MMD objective.

This shared mechanism is prior work, not a new contribution of the conditional manuscript.

## Mathematical relation between the states

Our implemented Curie--Weiss input is the condition-dependent family

\[
|\phi_\beta\rangle
\propto
|p_{\alpha(\beta)}\rangle^{\otimes N}
+X^{\otimes N}|p_{\alpha(\beta)}\rangle^{\otimes N},
\qquad
|p_\alpha\rangle=R_Y(\alpha)|0\rangle.
\]

It is a (+1) eigenstate of (X^{\otimes N}) for every value of \(\beta\). The prior GHZ state is the special case \(\alpha(\beta)=0\). For a general angle, our input is therefore a spin-flip-symmetrized, condition-dependent cat-like product-state superposition, not a GHZ state.

After the first Hadamard layer, our reference amplitudes depend only on Hamming weight (k):

\[
A_\beta(k)\propto
u_\beta^{N-k}v_\beta^k\bigl[1+(-1)^k\bigr],
\]

where (u_\beta=(\cos(\alpha/2)+\sin(\alpha/2))/\sqrt2) and (v_\beta=(\cos(\alpha/2)-\sin(\alpha/2))/\sqrt2). The reference has support only on even-weight strings, but its probability is generally nonuniform across the even shells. At \(\alpha=0\), those even strings become uniform, recovering the post-Hadamard GHZ reference used in the Section 6 estimator.

## What is distinct in our work

The defensible differences are:

1. **Conditional family rather than one unconditional target.** The prior paper trains one fixed symmetric model for one 2D Ising distribution at a fixed temperature. We use one shared phase model across a family indexed by inverse temperature \(\beta\).
2. **Condition-dependent symmetric input.** The prior input is a fixed GHZ state. Our encoder moves within a family of (X^{\otimes N})-invariant cat-like states through the map \(\beta\mapsto\alpha(\beta)\). GHZ is one endpoint of this family.
3. **Nonuniform radial estimator.** The prior Section 6 estimator is specialized to the fixed GHZ state and a uniform bitstring expectation with a parity-dependent factor. Our training interface samples a condition-dependent radial reference and evaluates amplitude ratios from its (N+1) Hamming-shell table. This is the technical extension that must be derived explicitly rather than attributed to the prior paper.
4. **Ancilla-assisted visible marginal.** We apply the construction on visible and ancilla qubits and train the visible marginal. Global flip invariance of the joint law implies the same invariance of the visible marginal after relabelling the summed ancilla strings.
5. **Different scientific question.** Their experiment asks whether a known symmetry improves an unconditional fit. Ours asks whether one shared conditional circuit tracks temperature-dependent distributions, how much the supplied symmetry and supplied condition map contribute, and whether ancillas improve the generic and symmetry-informed encoders.

These are extensions and a different experimental setting. They do not justify presenting symmetry-through-input-state design itself as novel.

## Required manuscript corrections

The current symmetry subsection should not say that the Curie--Weiss transformation is \(\beta\mapsto-\beta\). For the zero-field Curie--Weiss family, global spin flip is a within-condition symmetry:

\[
p_\beta(x)=p_\beta(\bar x)
\quad\text{for each fixed }\beta.
\]

The current subsection also should not call the implemented family a GHZ state. It is GHZ only when \(\alpha=0\).

The general finite-group construction is broader than the experiment and overlaps the recipe already stated at the end of Section 6 of the prior paper. Unless it supports another result, it should be shortened or moved out of the main narrative. The main text should instead present the concrete fixed-\(\beta\) bit-flip symmetry, attribute the input-state mechanism immediately, define our conditional family, and then derive its radial sampling and amplitude-ratio evaluator.

## Recommended attribution boundary

A suitable opening is:

> Recio-Armengol, Ahmed, and Bowles showed that a GHZ input enforces global bit-flip invariance in an IQP-structured Born machine while retaining a classical estimator for the observables required during training. We use the same symmetry principle, but replace their fixed unconditional GHZ input by a condition-dependent family of spin-flip-symmetrized states.

The citation should appear after the first sentence and again near the statement that GHZ is the \(\alpha=0\) special case. The manuscript may claim the conditional family, its radial estimator, the ancilla-marginal formulation, and the conditional ablation as contributions, subject to a broader novelty search. It should not claim the global-bit-flip input construction or its basic commutation proof.

## Evidence boundary for the Curie--Weiss results

The planned ablation remains necessary. Because the encoder supplies the exact within-temperature \(\mathbb Z_2\) symmetry and the piecewise map supplies a landmark at \(\beta=1\), the present Curie--Weiss result cannot by itself show discovery of either structure. The (2\times2\) encoder/map ablation should determine whether performance comes from the symmetry, the informed condition map, their interaction, or ancilla capacity. Generalization claims should emphasize held-out non-landmark temperatures.
