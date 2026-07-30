# Conditional IQP experiment design

This document records the design agreed during the project grilling session.
Implementation begins only after the design is explicitly confirmed.

## Research question

Test whether a shared IQP circuit can learn two output distributions when an
exogenous class condition enters only through a real product-state angle
encoding. Reverse diffusion and denoising are out of scope.

The theoretical result covers

\[
|\psi(\boldsymbol{\alpha})\rangle
=\bigotimes_{i=1}^{n}R_Y(\alpha_i)|0\rangle
\]

with arbitrary real per-qubit angles. The experiment uses permutation-symmetric
condition prototypes:

\[
\alpha_A=0,\qquad \alpha_B=\pi/4,
\]

broadcast across all qubits. Both conditions share all trainable circuit
parameters.

The circuit convention is

\[
U_\theta=H^{\otimes n}
\left(\prod_g e^{i\theta_g Z_g}\right)
H^{\otimes n},
\]

applied after the condition encoding and followed by computational-basis
measurement.

## Targets

The output is a 19-bit string. Target mass is first assigned to Hamming-weight
shells using a discrete Gaussian, then distributed uniformly among bitstrings
within each shell.

Both targets have width

\[
\sigma_{\mathrm{target}}=\sqrt{n}/2.
\]

Their Gaussian location parameters are numerically calibrated after truncation
to weights \(0,\ldots,n\) so their exact expected Hamming weights are

\[
\mathbb E[K\mid A]=n/2,\qquad
\mathbb E[K\mid B]=n/4.
\]

Training uses 10,000 samples per condition. Validation uses 2,000 disjoint
samples per condition. Minibatches contain 256 samples from each condition.
Analytic target probabilities are reserved for exact final evaluation.

## Ansatz families

Every ansatz contains an independently parameterized one-qubit IQP generator
on every qubit and an independently parameterized two-qubit generator for each
edge of its interaction graph. Each generator appears once; repeated layers of
the same commuting generators would be algebraically redundant.

The primary angle-conditioned model is compared across:

1. The canonical distance-3 heavy-hex graph on 19 qubits.
2. A \(4\times5\) nearest-neighbor rectangular lattice with one corner removed.
3. The complete graph \(K_{19}\).

The comparison reports every variant. It compares native ansatz families, not
connectivity at a fixed parameter count.

## Objective

The class-balanced objective is

\[
\mathcal L(\theta)
=\frac12\operatorname{MMD}^2(p_A,q_\theta(\cdot\mid A))
+\frac12\operatorname{MMD}^2(p_B,q_\theta(\cdot\mid B)).
\]

Use an unbiased estimator of squared MMD without clipping or square roots.
Average the three Hamming-RBF kernels

\[
k_\sigma(x,y)
=\exp\left[-\frac{d_H(x,y)}{2\sigma^2}\right],
\qquad
\sigma\in\{1,2,4\}.
\]

Select Monte Carlo operator and IQP-sample budgets through a convergence study.
Use the smallest common training budget giving Pauli-expectation RMSE at most
0.01 against exact small-system calculations and Conditional-MMD coefficient
of variation at most 5% on representative 19-qubit parameters. Calibrate a
larger validation budget independently and then freeze both budgets.

The completed calibration selected 64 MMD observables and 8,192 IQP samples
per bandwidth for training. Validation uses 256 observables and 16,384 IQP
samples per bandwidth. See `calibration.md`.

## Training

Use Adam for 2,000 updates with a cosine learning-rate decay from \(10^{-2}\)
to \(10^{-3}\). Evaluate a fixed validation estimator every 25 updates and
select the checkpoint with the lowest validation Conditional MMD. Do not tune
hyperparameters per topology.

For an ansatz with \(m\) generators, initialize

\[
\theta_j\sim\mathcal N(0,0.1^2/m).
\]

Run 10 independent, matched seeds per reported configuration. For a given
seed, reuse the target data, minibatch order, MMD observables, and Monte Carlo
random streams across topologies. Report all runs, with median and interquartile
range as the primary aggregate.

## Baselines

Run the following on the complete 2-local topology:

- No-condition baseline: use \(\alpha=\pi/8\) for both conditions.
- Basis-condition baseline: use \(|0\rangle^{\otimes19}\) and
  \(|1\rangle^{\otimes19}\).
- Separate-model reference: train one independent model per target using the
  corresponding primary angle prototype. This is a capacity reference, not a
  conditional model.

## Evaluation and success criterion

At \(n=19\), evaluate all \(2^{19}\) bitstrings exactly. Report per condition:

- exact bitstring total-variation distance;
- exact Hamming-weight-shell total-variation distance;
- expected Hamming-weight error;
- Hamming-weight standard-deviation error;
- held-out multiscale MMD; and
- target and generated Hamming-weight histograms.

Angle conditioning succeeds on the complete topology only if its median exact
shell TVD is lower than both conditional baselines for each condition
separately and exact bitstring TVD shows the same ordering. The separate-model
reference provides context rather than a required threshold. The topology
comparison is exploratory.

## Verification

Validate the product-state simulator with analytic one- and two-qubit cases and
random 2–8-qubit circuits against exact PennyLane expectations. Per observable,
require error no greater than the maximum of five reported standard errors and
0.01. Check autodifferentiated gradients against exact PennyLane gradients
using repeated estimates to quantify uncertainty.

Validate Conditional MMD against brute-force small-system kernel MMD, covering
each condition, each bandwidth, the combined objective, loss gradients, a
zero-discrepancy case, and a label-swap case whose pooled marginal is unchanged.

## Theory and claim boundary

Present a proposition proving efficient sampling and amplitude evaluation for
the post-Hadamard reference state, unbiased Pauli-\(Z\) expectation estimation,
and polynomial runtime and sample complexity. State estimator variance
explicitly. Add a short corollary explaining that JAX can backpropagate directly
through the estimator because its sampled reference strings and MMD observables
do not depend on the trainable circuit parameters.

Claim classically tractable MMD training for this specified product-state
family. Do not claim that the trained 19-qubit circuits demonstrate hard
classical sampling or quantum advantage.

## Implementation and paper deliverables

Implement the estimator and Conditional MMD locally in JAX, use Optax for
training, PennyLane for exact verification and quantum sampling, SciPy for
target calibration, and NetworkX for topology representation. Do not add
IQPopt or Qiskit runtime dependencies. Reuse only narrowly relevant logic and
test patterns from the reference repository; do not migrate diffusion or
legacy experiment machinery.

Produce a generic two-column LaTeX draft with five pages of main text excluding
references. Include a two-column TikZ circuit figure and experimental results.
Retain and substantively cite all 15 unique supplied references, removing the
duplicated bibliography entry.
