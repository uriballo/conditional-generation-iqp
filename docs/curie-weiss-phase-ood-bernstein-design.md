# Bernstein selector encoding for phase-crossing generalization

Date: 2026-08-24
Status: validated exact-enumeration prototype; production evidence not yet run

## Decision

The next Curie--Weiss study should replace the spin-flip input and its
development-tuned temperature map with a target-independent Bernstein selector
encoding. Three selector ancillas encode only the normalized condition
(t=(\beta-0.5)/(1.5-0.5)). A fixed diagonal IQP unitary assigns one visible
phase block to each selector Hamming weight. Tracing out the selectors gives

\[
q_\theta(x\mid t)=\sum_{j=0}^{r}\binom rj
t^j(1-t)^{r-j}q_{\theta_j}(x),
\qquad r=3.
\]

This construction does not use the Curie--Weiss energy, global spin-flip
symmetry, permutation symmetry, a critical-point knot, or evaluation outcomes.
It is a general condition encoder for continuous families. The visible phase
blocks still determine which distributions the model can represent.

The prototype makes the previously failed design work at eight visible spins.
On a critical interval omitted from training, mean profile TVD falls from
0.262 for the affine broadcast encoder to 0.056. Mean raw TVD falls from 0.341
to 0.076. On the stricter one-sided hot-to-cold test, profile TVD falls from
0.382 to 0.131 and raw TVD from 0.466 to 0.235. These are exact-population,
exploratory results. They justify a fresh finite-sample campaign; they do not
belong in the paper as confirmatory evidence yet.

## Why the current Curie--Weiss result is insufficient

The current manuscript uses two pieces of Curie--Weiss structure. Its input is
an equal superposition of product states related by a global bit flip, and its
piecewise temperature map was selected after Curie--Weiss development screens.
That experiment is valid evidence for the frozen v3 protocol. It is weak
evidence for a general encoding principle because the encoder already contains
the target's two branches and exact symmetry.

The original affine broadcast encoder fails for a separate reason. One angle
controls every qubit, so one fixed phase vector must obtain both smooth
conditional transport and a change in modality from a narrow one-dimensional
reference path. At cold temperatures, the input also imposes a per-string
probability cap that can fall below the target's aligned-state mass. Ancillas
increase rank and parameter count, but broadcast conditioning does not give the
latent register a clear routing role.

The selector construction assigns that role explicitly. The condition controls
mixture weights, while fixed phase blocks control component shapes. A central
profile can therefore give way to a bimodal profile without placing the two
modes in the encoder.

## Method

### Condition preparation

Normalize the declared condition interval to (t\in[0,1]). Prepare each of
(r) selector qubits so that its state immediately before the diagonal unitary
is

\[
|\phi_t\rangle=\sqrt{1-t}|0\rangle+\sqrt{t}|1\rangle.
\]

Under the repository's pre-Hadamard (R_Y) convention, the required angle is

\[
\alpha(t)=\arcsin(1-2t).
\]

The selector string (z\) then has independent Bernoulli coordinates and
Hamming weight (J\sim\operatorname{Binomial}(r,t)). The map is fixed before
target access and contains no task-specific landmark.

### Selector-controlled diagonal

Let (G=(g_1,\ldots,g_m)) be the visible generator bank. For the present
eight-spin complete two-local ansatz, (m=36). The physical diagonal contains
(Z_{g_a}Z_S) for every visible generator (g_a) and selector subset
(S\subseteq[r]):

\[
D=\prod_{a=1}^{m}\prod_{S\subseteq[r]}
\exp\!\left(i c_{a,|S|}Z_{g_a}Z_S\right).
\]

Coefficients are tied across selector subsets of equal size. On a selector
string of weight (j), the effective visible phase is

\[
\theta_{a,j}=\sum_{\ell=0}^{r}c_{a,\ell}K_\ell(j;r),
\]

where (K_\ell(j;r)) is a binary Krawtchouk polynomial. The
((r+1)\times(r+1)) Krawtchouk matrix is invertible, so the tied coefficients
(c_{a,\ell}) can realize an independent phase (	heta_{a,j}) for every
selector weight. The implementation in
`src/iqp_cond/bernstein_selector.py` performs this inverse transform.

For (n=8), (r=3), and a complete two-local visible bank, the model has:

- 11 total qubits;
- 144 learned coefficients, stored as four 36-phase blocks;
- 288 compiled diagonal Pauli rotations after expanding selector subsets; and
- maximum Pauli locality five.

The gate count grows as (m2^r) for the full construction. This cost must
appear next to every expressivity claim. Low-order selector control reduces the
cost, but the first ablation indicates that it also removes much of the gain.

### Visible marginal

The diagonal is block diagonal in the selector basis and depends only on
selector Hamming weight. Local Hadamards on the selectors do not change the
visible reduced state after tracing them out. Therefore

\[
\begin{aligned}
q_\theta(x\mid t)
&=\sum_{z\in\{0,1\}^r}
(1-t)^{r-|z|}t^{|z|}q_{\theta_{|z|}}(x)\\
&=\sum_{j=0}^{r}\binom rj
(1-t)^{r-j}t^jq_{\theta_j}(x).
\end{aligned}
\]

The test suite compares this direct mixture with exact simulation of the
compiled ancilla circuit. They agree to (10^{-12}) on a random small-system
instance.

### Classical estimator

The state entering the diagonal remains a product state. A classical sampler
draws visible reference bits as before and selector bits independently from
(\operatorname{Bernoulli}(t)). Visible MMD masks are padded by identity on the
selectors, so an amplitude-ratio query never changes the selector string. The
existing ratio estimator and its bounded-second-moment argument therefore
apply without a Curie--Weiss exception. Gradients concern one fixed compiled
phase vector.

The full selector ansatz changes the resource claim, not the estimator claim.
For fixed (r), sampling and ratio evaluation remain polynomial. If (r)
grows, the compiled gate count is exponential unless the selector interaction
order is truncated.

## Prototype protocol

The prototype uses exact output probabilities and exact multiscale
Hamming-kernel MMD with bandwidths (0.5,1,2,4). Adam runs for 10,000 updates.
Ten initializations, numbered 0 through 9, supply replication units. Exact
test targets are first constructed after the minimum training-objective
checkpoint freezes. The encoding and phase bank never use test targets.

Three selectors were frozen for the reported comparison because a cubic
Bernstein family is the smallest one that can represent changing curvature
and remains overdetermined by the five one-sided training temperatures. The
earlier selector-count and ansatz screens are exploratory and cannot validate
that choice.

The critical-gap protocol trains at
(\beta\in\{0.5,0.7,0.8,1.2,1.3,1.5\}) and tests the omitted interval
(\{0.9,1.0,1.1\}). Training observes both regimes but no distribution inside
the gap. The one-sided protocol trains at
(\{0.5,0.6,0.7,0.8,0.9\}) and tests (\{1.0,1.1,1.2,1.3,1.4,1.5\}).

| Protocol | Encoding | Train profile TVD | Test profile TVD | Test raw TVD | Test absolute-magnetization error |
|---|---|---:|---:|---:|---:|
| Critical gap | affine broadcast | 0.214 | 0.262 | 0.341 | 0.055 |
| Critical gap | Bernstein selector | 0.052 | 0.056 | 0.076 | 0.008 |
| Hot to cold | affine broadcast | 0.087 | 0.382 | 0.466 | 0.263 |
| Hot to cold | Bernstein selector | 0.030 | 0.131 | 0.235 | 0.067 |

Positive effects below mean that the selector lowers error. For the critical
gap, the paired profile-TVD reduction is 0.206 with an exploratory 95% paired
bootstrap interval of [0.146, 0.262]. The raw-TVD reduction is 0.266
[0.219, 0.303]. For hot-to-cold evaluation, the corresponding reductions are
0.251 [0.210, 0.289] and 0.231 [0.172, 0.287]. These intervals quantify
optimizer-seed variation only.

## What the ablations say

The five-seed, 3,000-update ablation is deliberately secondary because it was
run during method development.

| Phase control | Learned coefficients | Critical-gap profile TVD | Critical-gap raw TVD |
|---|---:|---:|---:|
| Affine broadcast | 36 | 0.199 | 0.320 |
| Linear selector control | 72 | 0.116 | 0.200 |
| Quadratic selector control | 108 | 0.097 | 0.155 |
| Full selector-weight control | 144 | 0.091 | 0.138 |
| Affine plus full selector | 144 | 0.151 | 0.208 |

The full selector arm is best in this short screen. The hybrid arm is worse
than the pure selector, so the gain does not come from adding parameters to the
old affine path. Low-order selector control improves on affine broadcast but
does not reproduce the full result. The experiment therefore points to
distinct latent phase blocks as the active mechanism.

A separate one-sided Gaussian mean-shift screen is a useful negative control.
The pure selector extrapolates worse than affine broadcast there. The method
must not be described as a general solution to extrapolation. Its generality is
architectural: the encoder is target independent and Bernstein mixtures can
approximate continuous conditional paths as selector capacity grows. Whether a
finite model continues beyond observed conditions still depends on the target
path and inductive bias.

## Identifiability boundary

No distribution-free learner can infer an arbitrary unseen phase from data on
only one side of a boundary. Two conditional families can agree at every
training temperature and disagree everywhere beyond it. Encoding cannot remove
that ambiguity.

The critical-gap protocol is the stronger primary design because training
observes both regimes and withholds the transition region. The one-sided
hot-to-cold result is a stress test of the cubic Bernstein prior. It is
surprising and worth reporting if it survives a fresh campaign, but it cannot
support a universal OOD claim.

## Production protocol

The next campaign should treat every result above as development evidence.
The confirmatory design should be frozen before generating new model outputs.

1. Use (n=12) visible qubits, three selector ancillas, four complete
   two-local visible phase blocks, and the compiled Krawtchouk construction.
   This changes both system size and optimizer seeds relative to development.
2. Train with finite target samples and the existing classically estimable MMD
   pipeline. Keep exact enumeration for post-selection audit only.
3. Make a wide critical gap the primary protocol. Use the one-sided protocol
   as a secondary stress test.
4. Use new optimizer and data seeds. Freeze a shadow evaluation grid before
   running any fit. Do not reuse the development test grid for a confirmatory
   claim.
5. Compare against affine broadcast, the current spin-flip v3 encoder, an
   unconditioned four-block capacity control, low-order selector control, and a
   bank of separate circuits with matched stored phase count.
6. Report profile TVD and raw TVD as co-primary metrics. Add absolute
   magnetization, Binder cumulant, and calibration curves as diagnostics.
7. Require the selector to improve both TVDs over affine broadcast with paired
   intervals above zero. Also require mean critical-gap profile TVD below 0.10
   and raw TVD below 0.15. These thresholds are weaker than the exploratory
   means and leave room for finite-sample and size effects.
8. Run a non-Curie conditional family with the same encoder and selector count.
   This control may be negative, but it must reveal whether the claimed scope
   is phase-changing paths or conditional generation in general.

## Claim boundary for the paper

The current evidence supports this development statement:

> A target-independent selector encoding can turn a fixed IQP diagonal into a
> Bernstein mixture of latent phase blocks. In an exact eight-spin prototype,
> this construction crosses a Curie--Weiss transition region omitted from
> training and substantially improves one-sided hot-to-cold continuation over
> affine angle encoding.

The paper cannot yet claim finite-sample success, scaling beyond eight visible
spins, favorable gate complexity as selector count grows, or general OOD
generation. It should not replace the v3 results until the fresh campaign
passes its frozen criteria.

## Claim and evidence map

- Claim: the encoding does not contain Curie--Weiss structure. Evidence: the
  map uses only normalized condition and binomial selector weights. Status:
  supported by construction.
- Claim: one fixed diagonal circuit realizes the mixture. Evidence: the
  Krawtchouk compilation and exact physical-circuit equivalence test. Status:
  supported.
- Claim: the prototype crosses a withheld phase interval. Evidence:
  `critical-gap-r3.json`, ten seeds, both TVD metrics. Status: exploratory.
- Claim: the prototype continues from hot to cold conditions. Evidence:
  `hot-to-cold-r3.json`, ten seeds. Status: exploratory and target specific.
- Claim: the method works with the paper's finite-sample classical training
  protocol. Evidence: not run. Status: needs evidence.
- Claim: the method generalizes across target families or system sizes.
  Evidence: the theorem covers continuous paths with growing resources, but
  the finite empirical claim lacks a positive second benchmark. Status: needs
  evidence.

## Reviewer-style self-audit

- Contribution: pass for a distinct target-independent encoding and exact
  circuit construction; novelty relative to Bernstein mixtures and multiplexed
  quantum circuits still needs a focused literature check.
- Clarity: pass at the method level; the final paper needs a pipeline figure
  showing selector preparation, weight-indexed blocks, and marginalization.
- Experimental strength: needs a fresh finite-sample (n=12) campaign.
- Evaluation completeness: needs matched-capacity baselines, a new shadow grid,
  and a second target family.
- Method soundness: pass for exactness and estimator access; gate count and
  locality are the main costs and must stay prominent.
