# Revision plan: conditional IQP model and training

Status, 2026-09-07: this is the historical planning document. The implemented manuscript and remaining author decisions are documented in the [first supervisor draft review](../docs/supervisor-draft-review-2026-09-07.md).

This plan applies to the canonical Overleaf manuscript. It deliberately stops
short of rewriting the LaTeX: the first pass should agree on the argument and
section boundaries before changing scientific prose.

## Editorial objective

The Methods should let a quantum-machine-learning reader answer four questions
without consulting the experiments:

1. What object is learned and what remains fixed across conditions?
2. How does the classical condition enter the quantum model?
3. Under what interface is the conditional loss classically estimable?
4. What additional capacity do unmeasured ancillas provide?

The current draft contains most of these ingredients, but it mixes the abstract
model, the production encoding, estimator assumptions, and experimental
hyperparameters. The rewrite should move from definition to tractability to
capacity, and only then to optimization.

## Recommended section architecture

### 1. Conditional IQP model

#### 1.1 Task and sharing constraint

Open with the conditional target family
\(\{p(\cdot\mid c):c\in\mathcal C\}\). Then define a condition-dependent
input state \(\lvert\phi(c)\rangle\), one shared IQP unitary
\(U_{\boldsymbol\theta}\), and
\[
q_{\boldsymbol\theta}(x\mid c)
=\left|\langle x\lvert U_{\boldsymbol\theta}\rvert\phi(c)\rangle\right|^2.
\]
State the central constraint immediately: the graph and parameter vector are
identical for every condition; only the prepared input changes. End this
paragraph with the precise scope statement that the complete encoder-plus-IQP
pipeline need not itself be a strict IQP circuit.

#### 1.2 Shared IQP layer

Define \(G=(V,E)\), the one- and two-body diagonal generators, and the parameter
count. This material is already present, but the notation should distinguish
the number of visible qubits \(n\) from the joint number \(N=n+r\) once
ancillas are introduced. Avoid interrupting this definition with experimental
choices.

#### 1.3 Tractable conditioning interface

State the two operations required of the state immediately before the diagonal
layer: efficient sampling of computational-basis strings and efficient
evaluation of the amplitude ratios used by the Pauli estimator. This is more
precise than claiming that every admissible encoder has a fully queryable
statevector. Explain that these operations preserve classical estimation of the
selected observables, loss, and gradients; they do not imply classical output
sampling.

#### 1.4 Product-state encoder

Give the general product form with single-qubit rotations and retain the finite
Fourier-response observation. Move the specific affine angle map to
Experimental Setup. Unless an encoding ablation supports optimality, describe
that map as a fixed design choice rather than a superior encoding.

#### 1.5 Ancilla-assisted conditioning

Replace “Purified product-state conditioning” with “Ancilla-assisted
conditioning.” The production model prepares a tractable joint product input,
applies a complete two-local IQP circuit to visible and ancilla qubits, and
marginalizes the unmeasured ancillas:
\[
q_{\boldsymbol\theta}(x\mid c)
=\sum_{a\in\{0,1\}^r}
\left|\langle x,a\lvert U_{\boldsymbol\theta}^{(n+r)}
\rvert\Phi(c)\rangle\right|^2.
\]
Do not retain the general weighted purification with \(w_c(u)\) unless it is
actually instantiated. Attribute the capacity increase carefully to the larger
joint circuit and visible--ancilla interactions, which can induce mixed visible
marginals. The experiment does not isolate these two mechanisms.

#### 1.6 Symmetry-informed Curie--Weiss encoder

Begin with the concrete spin-flip relation
\(q(\bar x\mid-c)=q(x\mid c)\), then show the sufficient equivariance and
commutation conditions. Call the coherent two-branch construction
“spin-flip-symmetrized” or “cat-like,” not GHZ in general. State the actual
piecewise-linear condition map and the tractable sampling/amplitude-ratio
formulas used by the frozen encoder. Move the general finite-group,
permutation, and Hamming-class discussion to an appendix unless it is used in
the final analysis.

### 2. Conditional IQP training

#### 2.1 Condition-balanced population objective

Define the population MMD once, specialize the Gaussian kernel to Hamming
distance, and then average equally over conditions and bandwidths. Explain why
equal condition weighting matters: it prevents conditions with more samples
from dominating the shared parameter vector.

#### 2.2 Character expansion and Pauli moments

Derive the binary-character expansion of the Hamming-RBF kernel and show that
each MMD term becomes a weighted squared discrepancy of parity expectations,
equivalently Pauli-\(Z\) moments. This derivation is the central technical link
between conditional generation and classical IQP training; it should not be
compressed into the current one-sentence assertion.

#### 2.3 Stochastic estimator

Explain how nonidentity character masks are sampled, how the analytic
normalization preserves the original kernel objective, and how duplicate masks
are compressed into exact multiplicity weights. Introduce the three
bandwidths through their expected mask bodyness: local, intermediate, and
approximately half-system. Keep numerical values such as 2,000 operators and
8,192 state samples in Experimental Setup.

#### 2.4 Balanced minibatch gradient

Replace the existing TODO with the protocol actually used: every update draws
100 examples from every training condition, evaluates all three bandwidths,
and averages the per-condition losses before updating one shared parameter
vector. Deterministic shuffling yields 2,000 updates, or 200 effective epochs
per condition. This is not a schedule that visits one condition at a time.

#### 2.5 Optimization versus evaluation

Close the training section with a clean distinction. MMD is the differentiable
surrogate used for optimization. Exact visible-distribution TVD is the primary
selection and scientific evaluation metric. Complementary profile, moment,
thermodynamic, and Wasserstein metrics diagnose what kind of error remains but
do not replace exact TVD.

## Paragraph-level writing sequence

1. Write the model’s execution path in one paragraph: encode, apply the shared
   circuit, measure or marginalize.
2. Define the fixed-unitary constraint formally and state why it makes the
   problem conditional rather than a bank of separately programmed circuits.
3. State the classical estimator interface and its limitations.
4. Present product conditioning, ancillas, and Curie--Weiss symmetry as three
   increasingly structured encoders.
5. Define the balanced conditional MMD objective.
6. Derive its Pauli-moment representation.
7. Describe the unbiased/compressed stochastic estimator.
8. State the balanced update rule.
9. Separate training loss from exact evaluation.

## Material to relocate

| Current material | Destination | Reason |
|---|---|---|
| Affine rotation map | Experimental Setup | It is a campaign choice, not part of the model definition. |
| Numbers of samples, operators, updates, and bandwidth targets | Experimental Setup | They control the estimator implementation rather than its validity. |
| Optimizer and checkpoint schedule | Experimental Setup | These are reproducibility details. |
| General permutation symmetry | Appendix | Useful theory, but it delays the Curie--Weiss construction used in the paper. |
| Hamming-class tractability example | Appendix | It supports a broader claim than the main experiments require. |
| Detailed non-claims | Discussion | One short scope sentence is enough in Methods. |

## Claim--evidence checks before polishing

| Proposed claim | Required support | Safe wording now |
|---|---|---|
| The shared unitary is condition independent. | Circuit definition and implementation identity. | Direct claim. |
| Conditional loss and gradients are classically estimable. | Full character expansion and sampling/amplitude-ratio assumptions. | Claim only under the stated interface. |
| Ancillas improve conditional capacity. | Paired exact-TVD ancilla ladders. | Present as an empirical question until the campaign finishes. |
| The encoder generalizes between conditions. | Held-out conditioned model versus adjacent-independent interpolation. | Restrict to within-range interpolation. |
| The symmetry encoder captures the critical transition. | Exact profiles and response metrics near \(\beta=1\). | Defer until complete Curie--Weiss results. |

## Pending evidence: Curie--Weiss encoder ablation

The manuscript must not treat the held-out value \(\beta=1\) as blind interpolation or discovery of the critical region because the informed piecewise condition map identifies that landmark. The frozen local ablation crosses two input states, product and spin-flip-symmetrized, with two maps, a target-independent affine map and the informed piecewise map. It evaluates both \(r=0\) and \(r=10\), with zero-phase encoder-only controls.

The central contrast is the generic product-state arm at \(r=0\) versus \(r=10\). If ancillas rescue this arm, the strongest interpretation is generic ancilla-assisted conditional learning. If the product arms fail but the generic spin-flip arm succeeds, the evidence supports symmetry-informed learning without supplying the critical landmark. If success requires the informed spin-flip arm, restrict the claim to exploiting known symmetry and a known critical-region map. If the spin-flip arm succeeds at \(r=0\), do not attribute that gain to ancillas.

Report \(\beta=1\) separately as landmark-aware realization. Base interpolation and generalization claims on the non-landmark held-out set \(\{0.6,0.8,1.2,1.4\}\), as well as the full held-out average. The experiment is tracked by `configs/curie_encoder_ablation_v1.json`; manuscript claims remain pending until all matched seeds and exact evaluations are complete.

## Figures and notation

- Retain the current pipeline figure as the conceptual anchor, but label the
  unmeasured ancilla register if ancillas remain a headline contribution.
- Use \(n\) for visible qubits, \(r\) for ancillas, and \(N=n+r\) for the joint
  circuit throughout.
- Use one symbol for the condition set, preferably \(\mathcal C_{\mathrm{train}}\),
  and distinguish training from held-out conditions explicitly.
- Reserve \(\sigma\) for MMD bandwidth only if Gaussian target width uses a
  different symbol such as \(w\); otherwise the current notation is ambiguous.
- Define “exact TVD” as TVD between complete visible bitstring distributions,
  not only between Hamming or magnetization profiles.

## Acceptance criteria for the rewrite

- A reader can identify the fixed and condition-dependent components from the
  first model paragraph.
- Every theoretical claim states the assumptions under which it holds.
- The ancilla equations match the implemented joint-product-state model.
- The conditional MMD-to-Pauli reduction is explicit enough to reproduce.
- The minibatch description matches the production runner.
- Method definitions contain no campaign-specific numerical hyperparameters.
- MMD is never presented as the primary scientific metric.
- No result claim precedes completed exact-TVD evidence.
