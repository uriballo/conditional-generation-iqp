# Two-parameter Gaussian combination benchmark

> **Protocol version:** 0.2.1 closure revision
>
> **Status:** adversarially closed; implementation pending; no experiments
> authorized
>
> **Primary setting:** eight visible qubits, no ancillas
>
> **Scientific role:** compact bridge between the scalar proof of concept and
> the physics benchmark

## 1. Objective and claim boundary

This benchmark tests whether one shared conditional IQP circuit can recover two
condition effects at coordinate pairs absent from training. The target is a
discrete Gaussian over Hamming-weight shells. One coordinate controls its mean,
and the other controls its standard deviation.

The experiment supports at most the following claim:

> Given a supplied, family-aware two-coordinate product-state representation,
> a shared conditional IQP circuit can preserve mean and width responses
> learned from single-coordinate deviations at held-out combinations of
> coordinate values.

The experiment does not test learned disentanglement, generic compositionality,
prediction of arbitrary unseen interactions, competitive classical
performance, or quantum advantage. Benchmark 1 establishes shared conditional
fitting in one coordinate. Benchmark 3 will carry the paper's main empirical
weight by introducing physical structure and domain relevance. Benchmark 2
isolates condition dimensionality and held-out combination generalization.

The primary experiment uses $n=8$ visible qubits and no ancillas. A contingent
$n=12$, zero-ancilla strict-transfer study may run after the $n=8$ audit passes.
Ancilla experiments are excluded because they change circuit expressivity and
would confound the conditioning question.

## 2. Frozen target family

For a bitstring $z\in\{0,1\}^n$, let $w(z)$ denote its Hamming weight. The
target assigns a discrete Gaussian law to $w\in\{0,\ldots,n\}$ and distributes
each shell probability uniformly over its bitstrings. The implementation must
calibrate the exponent location and width so that the truncated law has the
prescribed mean and standard deviation.

For exponent location $\ell\in\mathbb R$ and width $\tau>0$, the shell and
bitstring laws are

\[
Q_k(\ell,\tau)
=
\frac{\exp[-(k-\ell)^2/(2\tau^2)]}
{\sum_{j=0}^n\exp[-(j-\ell)^2/(2\tau^2)]},
\qquad
q(z)=\frac{Q_{w(z)}(\ell,\tau)}{\binom{n}{w(z)}}.
\]

Calibration uses nested float64 bisection. For fixed $\tau$, bracket $\ell$
from $[-1,1]$, doubling both endpoints for at most 60 expansions until the
mean residual changes sign. The outer solve starts with
$\tau\in[10^{-9},1]$ and doubles its upper endpoint for at most 80 expansions
until the standard-deviation residual changes sign. Both solves use at most
200 bisection iterations and stop when the bracket width is below
$10^{-14}\max(1,|\text{midpoint}|)$. Log masses are stabilized by subtracting
their maximum before exponentiation. A target is invalid unless the absolute
residuals of both prescribed moments are at most $10^{-8}$.

For $c_\mu,c_\sigma\in[0,1]$, define

\[
\mu(c_\mu)=n\left[\frac12+0.06(2c_\mu-1)\right],
\qquad
\sigma(c_\sigma)=\sqrt n\left[0.35+0.125(2c_\sigma-1)\right].
\]

At $n=8$, the audit domain therefore spans

\[
\mu\in[3.52,4.48],
\qquad
\sigma\in[0.636396,1.343503].
\]

### 2.1 Target-only selection record

Protocol v0.1 proposed a range set whose mean-axis effect was too large
relative to its width-axis effect; no candidate passed its own balance rule.
The revised preflight evaluated

\[
\begin{aligned}
a&\in\{0.04,0.05,0.06,0.07,0.08\},\\
b&\in\{0.35,0.40\},\\
d&\in\{0.10,0.125,0.15,0.175\}.
\end{aligned}
\]

For a candidate $(a,b,d)$, it generated both complete $5\times5$ development
and audit target grids and applied three intrinsic criteria:

1. The maximum combined probability of the two boundary shells is at most
   0.05.
2. Every adjacent Hamming-profile TVD along either axis, over both grids, lies
   in $[0.04,0.30]$.
3. The endpoint-axis balance ratio is at most 1.5. Here
   $D_\mu$ is the mean over the five audit width levels of the TVD between
   $c_\mu=0$ and $c_\mu=1$, while $D_\sigma$ is the analogous mean over the
   five audit mean levels. The balance ratio is
   $\max(D_\mu,D_\sigma)/\min(D_\mu,D_\sigma)$.

Every prescribed mean and standard deviation must be reproduced within
$10^{-8}$ before a candidate is scored. Two of the 40 candidates fail this
calibration check and are rejected: $(0.06,0.35,0.175)$ and
$(0.07,0.35,0.175)$.

The deterministic ranking minimizes
$|\log(D_\mu/D_\sigma)|$, then maximizes the smallest adjacent TVD, then
minimizes the physical rectangle area proportional to $ad$, then resolves ties
lexicographically by $(a,b,d)$. Twenty-six candidates survive. The selected
triple is

\[
(a,b,d)=(0.06,0.35,0.125).
\]

Its maximum boundary mass is 0.011319; adjacent TVDs range from 0.055739 to
0.163627; $D_\mu=0.348781$; $D_\sigma=0.353324$; and the balance ratio is
1.013025. The implementation must regenerate and publish the complete 40-row
table before training.

Model-specific coincidences do not select the target. They are reported after
selection. The central target is 0.175938 bitstring TVD from the uniform
bitstring distribution. The exact multiplicative shell composition defined in
Section 6 has mean off-axis TVD 0.052125 on the development grid and 0.083775
on the audit grid.

## 3. Condition grids and combination split

Let

\[
G_{\mathrm{dev}}=\{0.1,0.3,0.5,0.7,0.9\},
\qquad
G_{\mathrm{audit}}=\{0,0.25,0.5,0.75,1\}.
\]

For phase $s\in\{\mathrm{dev},\mathrm{audit}\}$, the complete target grid is
$G_s\times G_s$, and the training set is the nine-point axis cross

\[
\mathcal C_{\mathrm{train},s}=
\{(c_\mu,0.5):c_\mu\in G_s\}
\cup
\{(0.5,c_\sigma):c_\sigma\in G_s\},
\]

where the shared midpoint is counted once. The two phases train separate
models. The 16 conditions in

\[
\mathcal C_{\mathrm{off},s}
=
\{(c_\mu,c_\sigma)\in G_s^2:
c_\mu\ne0.5,\ c_\sigma\ne0.5\}
\]

form the primary off-axis evaluation set. Every individual coordinate value
occurs during training, but no off-axis pair does.

Including the midpoint supplies a common baseline for the two learned axis
effects. It also exposes rather than rewards a centered encoder whose midpoint
output is fixed: the central target is nonuniform, and midpoint fit is a
mandatory training gate.

Audit condition pairs, sampled datasets, and trainer seeds are absent from
development. The normalized coordinate value 0.5 occurs in both phases because
it defines the common axis baseline.

## 4. Circuit and training boundary

Every quantum arm uses one shared conditional IQP circuit with the complete,
untied bank of all one- and two-body diagonal generators. At $n=8$, this gives
36 trainable parameters. The experiment does not tune generator content,
connectivity, parameter tying, depth, or ancilla count.

Training minimizes the established sampled conditional multiscale Hamming-RBF
MMD objective with Adam and a cosine learning-rate decay. The optimizer never
minimizes exact TVD. At this simulated scale, analytical target probabilities
at the nine training-cross conditions may select checkpoints. Off-axis
conditions never select audit checkpoints, initializations, trajectories, or
seeds.

Initial parameters are target-blind Gaussian draws with standard deviation
0.5. One trainer seed defines one trajectory; the primary analysis performs no
best-of-restart selection. Within a paired dataset–trainer cell, every quantum
arm receives the same initial parameter vector, target samples, condition
schedule, operator randomness, and estimator randomness.

## 5. Encoders and quantum arms

### 5.1 Common construction

Define centered coordinates

\[
x=2c_\mu-1,\qquad y=2c_\sigma-1,
\]

and constants

\[
\delta=\frac{\pi}{4},\qquad \lambda=\frac{\pi}{8}.
\]

At $n=8$, each encoder repeats a four-angle block twice. At $n=12$, it
repeats the block three times. Each complementary pair sums to $\pi$, so its
expected excitation is one and total expected input excitation is
condition-independent.

The constants encode a family-aware prior and are not presented as
target-agnostic. With zero IQP parameters, the aligned map has mean $n/2$ and
Hamming-weight variance

\[
\cos^2(\delta+\lambda x)+\cos^2(\delta+\lambda y).
\]

At the midpoint this variance equals 1, close to the central target variance
$8(0.35)^2=0.98$. This coincidence is disclosed and motivates the mandatory
encoder-only contribution gates. The constants were not fitted to exact target
probabilities or trained-model performance.

At $n=12$, three repeated blocks give zero-parameter central variance 1.5,
again close to the target value $12(0.35)^2=1.47$. The strict-transfer report
must disclose this analogous coincidence.

### 5.2 General aligned encoder

The four-angle block is

\[
\begin{aligned}
\theta_{\mu,+}&=\frac{\pi}{2}+\delta+\lambda x,&
\theta_{\mu,-}&=\frac{\pi}{2}-\delta-\lambda x,\\
\theta_{\sigma,+}&=\frac{\pi}{2}+\delta+\lambda y,&
\theta_{\sigma,-}&=\frac{\pi}{2}-\delta-\lambda y.
\end{aligned}
\]

This is the default headline candidate. Its angles lie in
$[0.125\pi,0.875\pi]$ over the full condition square, and no angle equals
$\pi/2$.

### 5.3 General rotated encoder

Define

\[
r_1=(x+y)/\sqrt2,\qquad r_2=(x-y)/\sqrt2.
\]

The rotated encoder replaces $x,y$ in the aligned block with $r_1,r_2$. The
linear angle map preserves rank and singular values under this orthogonal
coordinate change. It does not preserve every property of the product-state
geometry: the square maps to a diamond, coordinate excursions reach
$\sqrt2$, and the angle range expands to approximately
$[0.0732\pi,0.9268\pi]$. Encoder fidelities, amplitude margins, and angle
ranges must therefore accompany the comparison. Performance differences
cannot be attributed to axis alignment alone.

### 5.4 Matched ablations

The aligned primary has two matched ablations:

- Aligned mean-only: set $y=0$ before constructing the angles.
- Aligned width-only: set $x=0$ before constructing the angles.

The rotated comparator has separate matched ablations:

- Rotated mean-only: set $y=0$, then compute $r_1,r_2$.
- Rotated width-only: set $x=0$, then compute $r_1,r_2$.

A shared condition-blind arm sets $x=y=0$. Every ablation retains the complete
qubit layout and retrains independently under the common schedule. Masking
only during evaluation is prohibited.

The primary matrix therefore contains seven trained arms: general
aligned, general rotated, four matched missing-channel ablations, and
condition-blind.

### 5.5 Privileged-midpoint centered control

The centered control uses

\[
\left(
\frac{\pi}{2}+\lambda x,
\frac{\pi}{2}-\lambda x,
\frac{\pi}{2}+\lambda y,
\frac{\pi}{2}-\lambda y
\right).
\]

At the midpoint, its preparation is $|+\rangle^{\otimes n}$. The first
Hadamard layer maps this state to $|0\rangle^{\otimes n}$, the diagonal IQP
unitary contributes only a global phase, and the final distribution is uniform
for every parameter vector.

The degeneracy extends across the training cross. Whenever one coordinate is
zero in centered units, half of the wires become $|0\rangle$ after the first
Hadamard. Those wires finish as independent uniform output bits. Cross-group
two-body parameters also contain an exact unidentifiable subspace on the
training cross while changing off-axis outputs. Favorable off-axis behavior
would therefore depend partly on unconstrained parameters rather than learned
combination.

This encoder is a structural negative control. It cannot select the schedule,
pass headline gates, or become the headline model. It runs with a reduced
descriptive seed budget.

### 5.6 Encoder-contribution and sensitivity audit

Before training, the implementation must verify angle bounds, rank-two
coordinate Jacobians, and excitation invariants symbolically and numerically.
For each general encoder, both local post-Hadamard amplitudes on every qubit
must be nonzero over the continuous condition square, with minimum absolute
local amplitude at least 0.10. This establishes full computational-basis
support. For every nonidentity generator, that support must contain basis
states with both generator eigenvalues. These two checks define structural
parameter sensitivity; a probability Jacobian at zero parameters is not used.

The centered control must reproduce its analytically predicted support loss and
training-cross parameter aliasing.

For every full encoder, the evaluator computes the zero-IQP-parameter output.
An eligible headline model must improve meaningfully over this encoder-only
distribution through the contribution gates in Section 9. Exact target
probabilities, target TVD, and trained-model performance may not set encoder
coefficients.

The historically tuned scaled four-block map remains pilot evidence. It is not
part of the production matrix.

## 6. Multiplicative classical reference

The classical reference operates on shell probabilities and has no fitted
interaction term. At each training condition, shell counts $N_k$ receive a
Jeffreys pseudocount:

\[
\widehat p_k=\frac{N_k+1/2}{N+(n+1)/2}.
\]

For any off-axis pair $(c_\mu,c_\sigma)$ whose marginal coordinate values
appear on the training cross, define

\[
\widetilde p_k(c_\mu,c_\sigma)
\propto
\frac{
\widehat p_k(c_\mu,0.5)\,
\widehat p_k(0.5,c_\sigma)
}{
\widehat p_k(0.5,0.5)
},
\]

then normalize over shells and distribute each shell uniformly over its
bitstrings. The target-only diagnostic applies the same formula to exact shell
probabilities without pseudocounts.

This multiplicative shell reference is completely identified by the full cross
and requires no feature basis, optimizer, regularizer, or model selection. It
contextualizes the target's inherent interaction and the quantum result; it is
not part of the quantum pass/fail gate and supports no advantage claim.

The sampled reference is computed once per dataset seed because it has no
trainer randomness. Its summaries and intervals use the dataset-seed factor
only; values are never duplicated across trainer cells. The exact target-only
reference is one deterministic diagnostic.

## 7. Development stages

### 7.1 Frozen run constants and seed blocks

All configurations use:

- 1,000 training, 1,000 validation, and 1,000 sampled-test bitstrings per
  condition;
- target batch size 100;
- one optimization trajectory and no internal restart selection;
- target-blind normal initialization with standard deviation 0.5;
- fresh operator masks at every update;
- checkpoint evaluation every 25 updates, including step zero;
- float64 parameters, losses, and reported probabilities, with only the phase
  contraction cast to float32;
- reference-sample chunks of 256;
- 1,000 validation reference samples and validation operator draws equal to
  the active training operator budget;
- the same 1,000-reference-sample and active operator-draw budgets for the
  sampled test-MMD diagnostic.

The two Hamming-RBF bodyness targets are $t\in\{2,4\}$. For visible width $n$,
replace $t$ by $\min(t,0.49n)$ and set

\[
\gamma_t=
\left[-2\log\left(1-\frac{2t}{n}\right)\right]^{-1/2}.
\]

The optimizer samples conditions and bandwidths uniformly. Exact analytical
training-cross TVD, averaged equally over the nine conditions, selects the
checkpoint. A checkpoint replaces the incumbent only when its score is
strictly smaller; ties retain the earlier step. Sampled validation MMD is
recorded but does not select checkpoints.

For a grid $G_s$, condition ordering is fixed as follows:

1. $(c_\mu,0.5)$ for ascending $c_\mu\in G_s$;
2. $(0.5,c_\sigma)$ for ascending $c_\sigma\in G_s\setminus\{0.5\}$;
3. off-axis pairs in lexicographic order with $c_\mu$ outermost.

Dataset and trainer seeds are disjoint across every decision phase:

| Phase | Dataset seeds | Trainer seeds |
| --- | --- | --- |
| Schedule screen | 4101–4103 | 5101–5103 |
| Development qualification | 4201–4205 | 5201–5205 |
| Locked audit | 4301–4308 | 5301–5308 |
| Failure-only capacity diagnosis | 4401–4403 | 5401–5403 |
| $n=12$ strict transfer | 6101–6105 | 7101–7105 |

Every schedule uses the same screen cells. The centered negative control uses
the first three qualification seeds per factor and the first five audit seeds
per factor; this subset choice is frozen and never performance-dependent.
Complete seed-derivation formulas for initialization, batches, conditions,
operators, and estimator samples belong in the manifest. The following scheme
is frozen rather than left to implementation convention:

- Target samples use NumPy `PCG64DXSM`. For dataset seed $d$, split tag
  $s\in\{1,2,3\}$ for train, validation, or sampled test, and the zero-based
  condition index $k$ from the fixed ordering, instantiate
  `SeedSequence([20260922, d, s, k])`.
- Trainer randomness uses JAX `threefry2x32`. Instantiate
  `jax.random.key(t, impl="threefry2x32")` for trainer seed $t$, apply
  `fold_in` with the stream tag below, and then apply `fold_in` with the
  zero-based update or checkpoint index where applicable.

| Trainer stream | Tag | Final index |
| --- | ---: | --- |
| Parameter initialization | 101 | None |
| Training-condition draw | 201 | Update |
| Bandwidth draw | 202 | Update |
| Target-minibatch indices | 203 | Update |
| Training operator masks | 301 | Update |
| Training estimator samples | 401 | Update |
| Sampled-validation diagnostic operator masks | 302 | Checkpoint |
| Sampled-validation diagnostic estimator samples | 402 | Checkpoint |
| Sampled-test diagnostic operator masks | 303 | Condition |
| Sampled-test estimator samples | 403 | Condition |

These keys generate all stochastic arrays before a stochastic computation;
implicit framework RNG state is prohibited. The target-minibatch stream picks
indices from the dataset-seed-specific samples, so the same trainer seed uses
the same index schedule across dataset seeds and arms. The crossed bootstrap
uses NumPy `PCG64DXSM` with `SeedSequence([260922])`. The manifest records this
mapping and the resulting root-key fingerprints.

### 7.2 Bounded schedule screen

The schedule screen crosses:

- Updates: $\{1000,2000\}$.
- Initial learning rate: $\{0.005,0.01\}$, decayed to one tenth of its
  initial value.
- Estimator budget: either 1,024 reference samples and 2,000 operator draws, or
  2,048 reference samples and 4,000 operator draws.

The eight schedules use the frozen three dataset seeds crossed with three
trainer seeds for the general aligned and general rotated encoders only, giving
144 fits. All other choices follow Section 7.1.

A schedule qualifies only if both encoders have no numerical failure and
satisfy all of the following on the nine screen cells:

1. Median training-cross TVD at most 0.30.
2. Median off-axis TVD at most 0.35.
3. Median normalized off-axis mean and width MAE at most 0.20.
4. For both encoders, the median generated-mean coefficient $\beta_\mu$ and
   generated-width coefficient $\beta_\sigma$ each lie in $[0.6,1.4]$.

Among qualifying schedules, select the one with the smaller worst-encoder
empirical 90th-percentile off-axis TVD, then the smaller mean off-axis TVD
across both encoders, then fewer operator draws, then fewer updates, then lower
learning rate. If no schedule qualifies, development stops and a new protocol
version must revise the bounded schedule space. There is no fallback schedule.

### 7.3 Development qualification

The selected schedule runs on five dataset seeds crossed with five trainer
seeds for all seven primary experimental arms, giving 175 fits. The two general
encoders qualify independently. Each must pass every applicable development
gate in Section 9 to remain headline-eligible.

The centered negative control runs on a separate $3\times3$ descriptive matrix
and cannot affect any decision. Development off-axis results may qualify the
frozen design but may not change target ranges, encoders, grids, gates, or
candidate arms.

The audit opens if at least one general encoder qualifies. A development-failed
general encoder still runs in the audit as a prespecified descriptive arm, but
it is permanently ineligible for the headline; audit results cannot rescue it.
All seven primary arms run whenever the audit opens, so the audit arm set never
depends on relative performance.

If neither general encoder qualifies, the audit does not run. Section 10
defines a bounded diagnosis that distinguishes individual-target capacity
failure from combination failure. Any redesign then receives a new protocol
version and new development seeds.

Gate applicability is fixed:

| Arm | Completeness, provenance, numerical validity | Training, off-axis, and response gates | Encoder-only and blind contribution gates | Matched missing-channel gate | Headline eligibility |
| --- | --- | --- | --- | --- | --- |
| General aligned | Required | Required | Required | Required against aligned ablations | Yes, if development-qualified |
| General rotated | Required | Required | Required | Required against rotated ablations | Yes, if development-qualified |
| Four missing-channel ablations | Required | Not applicable | Not applicable | Supply the paired comparison | No |
| Condition-blind | Required | Not applicable | Not applicable | Supplies the paired comparison | No |
| Centered negative control | Descriptive completeness and provenance only | Not applicable | Not applicable | Not applicable | Never |

A matched ablation or condition-blind arm is numerically valid only if it has a
complete cell matrix and a numerical-failure rate at most 5%. An invalid
control invalidates the associated contribution gate; it never creates an
artificial pass for the full encoder.

## 8. Locked audit

The audit crosses eight dataset seeds with eight trainer seeds for each of the
seven primary experimental arms: 448 trained fits. The centered negative control
runs on a reduced $5\times5$ matrix, giving 25 additional descriptive fits.

The dataset seed controls sampled targets. The trainer seed controls
initialization, condition scheduling, operator draws, estimator draws, and
optimization. Separate configuration fields must represent these sources.

The audit manifest binds:

- protocol and target-preflight hashes;
- condition grids, encoders, gates, and seed lists;
- training configuration and expected run identifiers;
- code and library revisions;
- Python, JAX, PennyLane, NumPy, BLAS, accelerator, and operating-system
  versions;
- numerical precision and contraction dtype;
- RNG algorithms and seed derivations;
- artifact schema and all runtime-source hashes.

No audit aggregate may be generated until every expected artifact passes
completeness and provenance checks. An operational interruption may resume or
rerun the identical content-addressed job, while retaining the failure record.
A completed trajectory that is numerically invalid under Section 9.2 is a
scientific failure and may not be replaced. Performance-based exclusions and
post-audit seed additions are prohibited.

## 9. Machine-evaluable estimands and gates

### 9.1 Per-cell quantities

For each paired dataset–trainer cell, let

- $T_{\mathrm{train}}$ be mean exact bitstring TVD over the nine training-cross
  conditions;
- $T_{\mathrm{off}}$ be mean exact bitstring TVD over the 16 off-axis
  conditions;
- $T_q$ be mean TVD over the four conditions in off-axis quadrant $q$.

For a condition, let $p(z)$ and $q(z)$ be the generated and target bitstring
laws, and let $P_k=\sum_{w(z)=k}p(z)$ and
$Q_k=\sum_{w(z)=k}q(z)$. Exact bitstring TVD is

\[
\frac12\sum_z |p(z)-q(z)|.
\]

Hamming-profile TVD is

\[
\frac12\sum_{k=0}^n |P_k-Q_k|.
\]

The within-shell residual is the TVD between $p$ and its shell-uniform
projection,

\[
\frac12\sum_z
\left|p(z)-\frac{P_{w(z)}}{\binom{n}{w(z)}}\right|.
\]

Generated mean and standard deviation are computed from $P_k$ with population
normalization.

The four quadrants are
$q_{--}=\{c_\mu<0.5,c_\sigma<0.5\}$,
$q_{-+}=\{c_\mu<0.5,c_\sigma>0.5\}$,
$q_{+-}=\{c_\mu>0.5,c_\sigma<0.5\}$, and
$q_{++}=\{c_\mu>0.5,c_\sigma>0.5\}$.

Define generated normalized coordinates

\[
\widehat m=
\frac{\widehat\mu-\mu_{\min}}{\mu_{\max}-\mu_{\min}},
\qquad
\widehat s=
\frac{\widehat\sigma-\sigma_{\min}}{\sigma_{\max}-\sigma_{\min}}.
\]

Their targets are $c_\mu$ and $c_\sigma$. Mean absolute error always refers to
the arithmetic mean of $|\widehat m-c_\mu|$ or
$|\widehat s-c_\sigma|$ over the named condition set.
$\mu_{\min},\mu_{\max},\sigma_{\min},\sigma_{\max}$ always denote the physical
family endpoints at normalized coordinates 0 and 1, including in development.

On the 16 off-axis conditions, fit by ordinary least squares

\[
\widehat q=
\beta_0+\beta_\mu(c_\mu-0.5)
+\beta_\sigma(c_\sigma-0.5)
+\beta_{\mu\sigma}(c_\mu-0.5)(c_\sigma-0.5).
\]

For generated mean, the target coefficients are $(0.5,1,0,0)$. For generated
width, they are $(0.5,0,1,0)$. Training-axis slopes are separate ordinary
least-squares slopes over the five corresponding points of the full cross.

Empirical quantiles use the NumPy linear convention. A condition-level median
first fixes a condition and then takes the median across seed cells.

### 9.2 Numerical-failure encoding

A completed trajectory is numerically invalid if any recorded parameter,
training loss, validation loss, gradient, checkpoint score, or output
probability is nonfinite; if any output probability is below $-10^{-10}$; or
if a probability vector differs from unit sum by more than $10^{-8}$. A
missing checkpoint or an uncaught numerical exception is also invalid. A
finite but inaccurate trajectory is not “divergent” and is evaluated normally.

At every condition, an invalid trajectory receives TVD, Hamming-profile TVD,
within-shell residual, and normalized moment errors equal to 1 before any cell
aggregation. Its intended response slopes equal 0, and its absolute intercept,
cross-axis, and interaction errors equal 1. An arm fails numerical validity if
more than 5% of its cells are invalid.

### 9.3 Mandatory training gates

Each headline-eligible full encoder must satisfy:

1. Median $T_{\mathrm{train}}\leq0.25$.
2. Empirical 90th-percentile $T_{\mathrm{train}}\leq0.30$.
3. No training condition has median TVD above 0.35.
4. Midpoint median TVD is at most 0.30.
5. Median normalized training mean and width MAE are each at most 0.10; their
   empirical 90th percentiles are at most 0.15.
6. Let $T_{\mathrm{mid},ij}$ denote trained midpoint TVD in cell $(i,j)$ and
   $T_{\mathrm{mid}}^0$ the deterministic encoder-only midpoint TVD. Define

   \[
   \rho_{\mathrm{mid}}
   =
   1-\frac{\operatorname{median}_{ij}T_{\mathrm{mid},ij}}
   {T_{\mathrm{mid}}^0}.
   \]

   The denominator must exceed $10^{-12}$, $\rho_{\mathrm{mid}}\geq0.15$, and
   the one-sided 95% lower bound for
   $\operatorname{median}_{ij}(T_{\mathrm{mid}}^0-T_{\mathrm{mid},ij})$
   must be positive.

### 9.4 Primary off-axis gates

Each headline-eligible full encoder must satisfy:

1. Median $T_{\mathrm{off}}\leq0.25$.
2. Empirical 90th-percentile $T_{\mathrm{off}}\leq0.30$.
3. For every quadrant, median $T_q\leq0.30$ and empirical
   90th-percentile $T_q\leq0.35$.
4. No off-axis condition has median TVD above 0.35.
5. Within each cell, Hamming-profile TVD and within-shell residual are first
   averaged over the 16 off-axis conditions. The medians of those cell
   averages are each at most 0.15; their empirical 90th percentiles are at most
   0.20.
6. Median normalized off-axis mean and width MAE are each at most 0.10; their
   empirical 90th percentiles are at most 0.15.

### 9.5 Response and preservation gates

Compute all response coefficients separately in every seed cell. Across cells:

1. The median intended-axis slope lies in $[0.8,1.2]$, and the empirical
   90th percentile of its absolute error from one is at most 0.35.
2. The median absolute intercept error from 0.5 is at most 0.10, and its
   empirical 90th percentile is at most 0.20.
3. The median absolute cross-axis and interaction coefficients are each at
   most 0.15, and their empirical 90th percentiles are each at most 0.25.
4. The absolute difference between the median training-axis slope and the
   median matching off-axis intended slope is at most 0.20.

These requirements apply separately to generated mean and generated width.
Pointwise errors and endpoint movement ratios are always reported.

### 9.6 Contribution and matched-ablation gates

For a full encoder and a paired control, define aggregate relative improvement

\[
\rho=1-
\frac{\operatorname{median}(T_{\mathrm{off}}^{\mathrm{full}})}
{\operatorname{median}(T_{\mathrm{off}}^{\mathrm{control}})}.
\]

If the control median is at most $10^{-12}$, the relative gate fails.
The full encoder must achieve $\rho\geq0.15$ relative to its encoder-only
output and condition-blind. For either comparison, define paired cell
differences

\[
D_{ij}
=
T_{\mathrm{off},ij}^{\mathrm{control}}
-
T_{\mathrm{off},ij}^{\mathrm{full}}.
\]

For an encoder-only control, its deterministic value is repeated only to form
the paired difference; it does not gain artificial trainer variability. The
bootstrapped statistic is $\operatorname{median}_{ij}D_{ij}$, and its
one-sided 95% lower bound must be positive.

For each coordinate, replace $T_{\mathrm{off}}$ in the preceding definition
with its normalized MAE and compare with the matched missing-channel ablation.
Generated-mean MAE is compared with the width-only arm, which lacks the mean
coordinate. Generated-width MAE is compared with the mean-only arm, which
lacks the width coordinate. The control median must exceed $10^{-12}$; the
relative reduction must be at least 50%; and the one-sided 95% lower bound for
the median paired improvement must be positive. Aligned and rotated models use
their own matched ablations.

### 9.7 Crossed uncertainty and variance decomposition

The two-way cluster bootstrap independently resamples the complete dataset-seed
index and trainer-seed index with replacement, then evaluates the statistic on
their Cartesian product while preserving arm pairing. It uses 10,000 resamples,
and RNG seed 260922. A directional 95% lower bound is the 5th percentile of the
bootstrap distribution. A descriptive two-sided 95% interval uses the 2.5th
and 97.5th percentiles. Resampled cluster multiplicities are retained when the
Cartesian table and its statistic are computed.

Empirical 90th percentiles are fixed-grid qualification summaries and receive
no inferential interval. Variance components use the balanced two-way
random-effects decomposition

\[
Y_{ij}=\alpha+A_i+B_j+E_{ij},
\]

with method-of-moments estimates

\[
\widehat{\sigma}_A^2=\max\{0,(MS_A-MS_E)/n_B\},\quad
\widehat{\sigma}_B^2=\max\{0,(MS_B-MS_E)/n_A\},\quad
\widehat{\sigma}_E^2=MS_E.
\]

$E_{ij}$ combines dataset–trainer interaction and residual variation because
there is one observation per cell.

No gate may be relaxed after the first schedule-screen trajectory starts.

## 10. Failure-only capacity diagnosis

If neither general encoder passes development, no audit runs. A bounded
diagnosis then uses the frozen $3\times3$ capacity seed block.

For the direct reference, each of the 25 development targets receives an
independent IQP circuit with the same complete ansatz and a fixed
$|0\rangle^{\otimes n}$ preparation, equivalent to $R_Y(0)$ on every wire.
Each condition uses 1,000 training, validation, and test samples, the selected
schedule, one trajectory per trainer seed, and exact target TVD for checkpoint
selection. This gives $25\times3\times3=225$ direct fits.

For the all-condition reference, each general encoder trains one shared circuit
on all 25 development conditions under the same data and schedule, giving
$2\times3\times3=18$ fits. Mean exact TVD over all 25 conditions, with equal
condition weights, selects its checkpoint. Section 9.2 numerical-invalidity
rules and failure scores apply to every one of the 243 diagnostic fits.

Individual-target capacity is adequate only if the direct fits have overall
median TVD at most 0.15, empirical 90th-percentile TVD at most 0.25, and no
condition-level median above 0.25. Shared all-condition capacity is adequate
for an encoder only if the median cell-average full-grid TVD is at most 0.20,
its empirical 90th percentile is at most 0.25, and no condition-level median
exceeds 0.30.

If direct capacity fails, the diagnosis is individual-target
capacity/optimization failure. If direct capacity passes but both
all-condition models fail, the diagnosis is shared-model capacity failure. If
direct capacity and an encoder's all-condition capacity pass while its
axis-cross protocol failed, the diagnosis is held-out-combination failure for
that encoder. Mixed outcomes are reported as unresolved rather than forced
into one category.

These 243 fits cannot rescue the audit or alter the failed result; they inform
the next protocol version.

## 11. Headline hierarchy and outcomes

The headline hierarchy is fixed:

1. General aligned.
2. General rotated.

An encoder enters this hierarchy only if it passed every applicable development
gate. It becomes headline-eligible only by also passing every applicable audit
gate. Audit cannot rescue development failure. The first encoder in the fixed
order that passes both phases becomes the headline model. The rotated encoder
does not replace a passing aligned encoder because of a slightly better point
estimate. The centered negative control is never headline-eligible.

If no development-qualified encoder passes audit, the benchmark fails and
returns to design. A development-failed encoder's audit results remain
descriptive even if its audit point estimates happen to meet the numerical
thresholds. The manuscript must report the failed protocol if any result from
it is discussed.

## 12. Contingent n=12 strict-transfer study

The $n=12,r=0$ study runs only after the $n=8$ audit passes. Before launch, it
must reproduce the prescribed moments, boundary-mass checks, and target-only
diagnostics at $n=12$. It uses the selected $n=8$ schedule without retuning and
five dataset seeds crossed with five trainer seeds. It uses
$G_{\mathrm{audit}}$, the corresponding nine-point full training cross, and
the 16 off-axis conditions. Condition ordering and mean exact training-cross
TVD checkpoint selection are identical to the $n=8$ audit.

The arms are the headline encoder, the other general encoder selected by the
fixed hierarchy, the headline encoder's two matched ablations, and
condition-blind. The phrase “strongest alternative” is prohibited because it
would select on audit performance. If the other general encoder failed $n=8$
development, its $n=12$ results remain descriptive.

Strict transfer passes only if the $n=8$ headline encoder satisfies every
applicable Section 9 gate at $n=12$, including the absolute-TVD, normalized
moment, response, contribution, numerical-validity, and uncertainty
requirements. The five-by-five crossed matrix replaces the eight-by-eight
matrix when computing empirical summaries and the frozen bootstrap. Failure at
$n=12$ does not overturn the $n=8$ result; it rejects strict transfer at the
larger register.

## 13. Resource and reporting plan

Before implementation approval, the campaign generator must report prospective
operator draws, reference draws, wall time, peak memory, storage, and recovery
time for:

- 144 schedule-screen fits;
- 175 development-qualification fits;
- 9 development negative-control fits;
- 448 primary-arm audit fits;
- 25 audit negative-control fits;
- 243 failure-only capacity-diagnosis fits, if triggered;
- 125 contingent transfer fits.

The primary pre-transfer campaign contains 801 fits. Capacity diagnosis and
strict transfer are contingent additions, not part of a successful primary
path.

The benchmark's paper footprint remains one main figure and one compact table.
The figure shows the full-cross design and the headline model's generated mean,
generated width, and TVD surfaces. The table reports median and upper-tail TVD,
normalized moment errors, response coefficients, preservation, and matched
ablation effects.

The appendix reports every arm, encoder-only surfaces, the centered-control
degeneracy, target-only and sampled multiplicative references, quadrant results,
variance decomposition, development history, gate attestation, failure-only
capacity diagnostics if triggered, and the $n=12$ study if run. Figures use
condition-complete data rather than selected examples.

## 14. Adversarial-review disposition

Protocol v0.1 received a reject/redesign verdict. Version 0.2 resolved its
blocking findings as follows:

| Finding | Resolution |
| --- | --- |
| Empty target-range search | Replaced by a feasible 40-candidate intrinsic search and frozen selected range |
| Centered encoder degenerate on the cross | Demoted to a reduced-budget structural negative control; permanently ineligible |
| No training-axis gate and missing common baseline | Restored the full cross and added training, midpoint, and preservation gates |
| Undefined additive reference | Replaced by an exact, pseudocount-defined multiplicative shell composition |
| Unmatched rotated causal gates | Added physical-coordinate ablations before rotation |
| Undefined estimands and statistics | Added formulas, aggregation order, failure scores, bootstrap, and variance decomposition |
| Target-adaptive selection criteria | Removed encoder and classical-reference behavior from target selection |
| Undefined capacity interpretation | Added failure-only direct-fit and all-condition references |
| Ambiguous transfer claim | Recast $n=12$ as strict transfer with a fixed comparator rule |

The v0.2 closure review returned **approve with required changes** and found no
new scientific blocker. Version 0.2.1 freezes the requested gate-applicability
rules, estimand formulas, independent seed blocks and RNG derivations,
failure-only diagnosis, numerical-failure encoding, and strict-transfer
criterion. Its first verification identified three remaining specification
omissions: the sampled-test operator stream, all-condition diagnostic
checkpointing, and the $n=12$ grid. Version 0.2.1 now defines all three. The
final bounded verification returned **pass**, with no implementation blocker
remaining. The centered encoder remains permanently headline-ineligible.

## 15. Execution authorization

This document does not authorize model training, schedule screens, development
fits, audits, transfer studies, or manuscript claims. Before any scientific
run:

1. Implement the target-only table and verify its hash and selected range.
2. Build a new versioned harness rather than extending the perimeter protocol
   in place.
3. Implement and test separate dataset and trainer seeds, the full-cross split,
   every encoder and matched ablation, encoder-only and sensitivity checks,
   exact estimands, failure encoding, crossed analysis, manifests, and
   attestations.
4. Freeze the reviewed protocol, gates, seed lists, environment lock, and RNG
   and dependency versions before the first protocol smoke run. Any
   compatibility change requires a new protocol patch version.
5. Run unit, integration, smoke, provenance, and deterministic-regeneration
   tests without using smoke accuracy to change scientific choices.
6. Freeze the tested source hashes and campaign manifest.
7. Obtain explicit author approval for the development launch.

Audit authorization requires a second explicit approval after development
qualification passes.
