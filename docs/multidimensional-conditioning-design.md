# Multidimensional location/width conditioning

## Decision

The primary track is a two-coordinate Condition controlling the location and
width of one analytic Gaussian Hamming-weight Target family. This is the
smallest richer task that adds a capability not present in the existing
one-dimensional moving-mean path: the model must use two independently
encoded coordinates to recover two Target properties at once.

The study starts with an exact eight-qubit feasibility setting. It is frozen
here before substantive training. A successful feasibility result permits one
bounded multi-seed study at the same size; it does not authorize a larger
qubit or topology sweep.

## Diagnosis of prior explorations

The existing experiments answer different questions and leave a specific gap:

| Artifact | What it established | Actionable limitation for this study |
|---|---|---|
| `docs/hopfield-design.md`, `results/hopfield/summary/` | A structured bitstring target is distinguishable from its shell profile, but the shared model remains near the representability ceiling. | Hopfield targets are a poor first test here: the sampled objective has an estimator barrier, exact MMD has a localization barrier, and a product reference has a pattern-peaked envelope barrier. Stay with smooth, exact targets while testing multidimensional sharing. |
| `docs/hamming-balls-design.md`, `results/hamming/summary/` | Bitstring-level multimodality exposes failures hidden by shell metrics. | The Hamming-RBF objective converged while low-order characters failed to localize planted centers; effective center count collapsed and interpolation beat lookup only once. Do not reuse a center-mixture target or interpret loss convergence as structural fidelity. |
| `src/iqp_cond/iris.py`, `results/iris/` | Empirical class-conditional distributions can be supplied to the existing trainer. | Quantile-binned Iris Targets are sparse finite samples, class labels are discrete ancilla codes, and the evaluation is empirical rather than an exact analytic family. They do not isolate a two-coordinate parameter-sharing question. |
| `docs/encoding-study.md`, `results/encoding/summary/` | A more complicated one-dimensional reupload path did not improve the existing scalar interpolation task. | Reuploading changes the path but not the capability: it still has one scalar Condition and worsens shell/mean diagnostics. The new test must add a second independently controlled coordinate, not another scalar path. |

The failure categories are therefore separated as follows:

- **Representability:** Hopfield pattern peaks exceed the measured product
  envelope; Hamming balls require localized mode mixtures. The new Gaussian
  shell family is smooth, positive on every state, and screened one Condition
  at a time before shared training.
- **Objective/localization:** Hamming-RBF MMD emphasizes low-order characters
  and does not constrain planted centers; Hopfield MMD cannot localize peaks.
  The new primary metric is exact full-distribution TVD, with exact MMD only as
  a disclosed training objective and diagnostic.
- **Optimization:** sampled-MMD Hopfield training moved almost no pattern mass,
  while exact training reached the boundary. Feasibility uses the existing
  deterministic exact-MMD trainer, so a failure is not attributed to Monte
  Carlo noise.
- **Parameter sharing:** the prior endpoint/interpolation studies mainly use
  a scalar path, and separate models show the cost of sharing. The new split
  compares one shared vector-conditioned model with a no-Condition model and
  independently trained single-Condition references.
- **Evaluation/protocol:** shell-only readouts would miss the Hamming-ball and
  Hopfield failures, while empirical Iris Targets make exact generalization
  hard to audit. Every new point is evaluated by exact bitstring TVD and by
  the two quantities that define the family: target mean and target standard
  deviation.

## Capability claim and boundary

If the frozen tests pass, the defensible claim is:

> On an exact eight-qubit Gaussian Hamming-weight family, a shared IQP
> parameter vector can use a two-coordinate tractable product-state Condition
> to recover both Target location and Target width at held-out interior and
> coordinate-slice Conditions, subject to the measured single-Condition
> capacity and the stated ansatz.

This does not claim conditional universality, arbitrary per-qubit control,
structured bitstring-mode recovery, extrapolation, quantum advantage, or
performance beyond the tested qubit count and complete 2-local ansatz.

## Target and Condition space

For \(n=8\), let \(u,v\in[0,1]\). The Target is uniform within each Hamming
weight shell and has shell mass

\[
 q_{u,v}(k) \propto
   \exp\left[-\frac12\left(\frac{k-\ell(u,v)}{w(v)}\right)^2\right],
 \qquad k=0,\ldots,n,
\]

where the location is recalibrated after truncation so that

\[
 \mathbb E[K\mid u,v] = \mu(u)=3.5-1.5u,
 \qquad w(v)=0.75+0.75v.
\]

The binomial coefficient is not included: this is the same discrete Gaussian
over shell mass used by `GaussianHammingTarget`, with uniform probability
inside a shell. Thus the two Target properties are exactly the calibrated
mean and the standard deviation induced by the truncated shell law.

The feasibility grid is the 3-by-3 product grid
`{0, 0.5, 1} x {0, 0.5, 1}`. Training uses only the four corners:

\[
 \mathcal C_{\rm train} = \{(0,0),(0,1),(1,0),(1,1)\}.
\]

Held-out Conditions are the five cross points:

\[
 \mathcal C_{\rm heldout} =
 \{(0.5,0),(0.5,0.5),(0.5,1),(0,0.5),(1,0.5)\}.
\]

This includes the interior point and holds out both a constant-u slice and
a constant-`v` slice. Held-out Targets are never used for optimization or
checkpoint selection.

## Fixed product-state Condition map

Conditions enter only through the initial product state, as required by
ADR 0002. Split the eight data qubits into two blocks of four and define

\[
 a(u)=\frac{\pi}{6}u, \qquad
 b(v)=\frac{\pi}{16}(2v-1).
\]

The fixed angle vector is

\[
 \alpha(u,v)=
 \big(a(u)+b(v)\big)^{\otimes 4}\otimes
 \big(a(u)-b(v)\big)^{\otimes 4}.
\]

The common component carries the location coordinate. The antisymmetric
contrast carries the width coordinate while keeping the reference state's
first-order total-weight shift approximately independent of width. This is a
disclosed linear map, uses no Target-dependent learned preprocessing, and is
still a real `RY` product-state encoding with efficient sampling and amplitude
ratios. The signs are allowed because the tractable encoding is defined for
real local angles, not only the original endpoint interval.

## Model and feasibility screen

The model is the complete 2-local ansatz on eight qubits (36 shared generator
parameters), trained with the existing exact zero-variance multiscale MMD
implementation and the ordinary mean exact gradient. No Condition-specific
circuit parameters are introduced.

Before any shared result is interpreted, train one independent model at each
of the four training Conditions against its exact Target. The screen passes
only if every single-Condition fit has bitstring TVD at most 0.30 and the
median is at most 0.20, using 800 ADAM steps and the fixed initialization
protocol. If it fails, stop: the target is not a useful sharing test at this
capacity.

## Controls and evaluation

The bounded run contains:

1. **Dense shared model:** four corner Conditions, one parameter vector.
2. **No-condition control:** the same four Targets, but every training row
   receives the center angle vector for `(u,v)=(0.5,0.5)`.
3. **Separate-model reference:** one independently trained model per corner.
4. **Oracle all-training lookup:** at each held-out Condition, evaluate every
   stored training output against the held-out Target and retain the best
   output. This is a strong memorization baseline, not a trainable model.

For every model output and every Target, enumerate all `2**8` bitstrings and
report:

- **Primary:** bitstring TVD.
- **Structural diagnostics:** absolute error in exact Hamming-weight mean and
  standard deviation.
- **Secondary:** shell-profile TVD, to verify that the structural readout is
  not being hidden by aggregation.
- **Training diagnostic:** exact Conditional MMD at held-out points, disclosed
  separately from TVD.

The full-distribution metric remains primary even though this family is shell
structured. Mean/standard-deviation errors identify which of the two
coordinates is being recovered; shell TVD checks the rest of the Target law.

## Frozen decision rules

The single-seed feasibility gate passes only if all of the following hold:

- the single-Condition screen passes;
- at every training corner, dense bitstring TVD is lower than no-condition
  bitstring TVD;
- the dense training median bitstring TVD is no more than 0.10 above the
  separate-model median;
- at least four of five held-out points have dense bitstring TVD no worse than
  the oracle lookup by 0.02 TVD, and dense TVD is not more than 0.05 worse at
  any held-out point;
- on the held-out cross, dense mean absolute error and width absolute error are
  each lower than no-condition for at least three of five points, with at
  least one improvement on each coordinate-wise slice.

The two-seed feasibility gate repeats the same frozen configuration with seed
1. It passes only if the single-Condition screen passes for both seeds and the
median dense-versus-no-condition improvement is positive for bitstring TVD,
mean error, and width error. A failure of either gate terminates this avenue
for the current ansatz and objective; no unbounded variant ladder is allowed.

If both gates pass, the only permitted full study is ten matched seeds at the
same eight-qubit grid, controls, objective, and budget. It is a success only if
the median frozen rules pass and at least eight of ten seeds pass the
memorization and held-out TVD conditions. Any other result is reported as a
bounded negative or mixed feasibility result.

## Stopping conditions

Stop without scaling if the single-Condition screen fails, if the dense model
does not beat no-condition at the training corners, or if the two-seed gate
fails. A redesign is allowed at most once and must change one named mechanism
(target family or fixed encoding map) while preserving the same screen and
controls. If that redesign fails, terminate the multidimensional avenue. No
larger-qubit, extra-topology, extra-encoding, or ten-seed sweep is justified by
loss convergence alone.

## Bounded feasibility outcome

The frozen seed-0 run was stopped at the single-Condition screen, as required.
The four exact single-Condition bitstring TVDs were 0.3883, 0.3177, 0.2220,
and 0.0945; the screen threshold is 0.30 per Condition and 0.20 median. The
two narrow-width Conditions therefore fail the capacity screen even though
their mean errors are small: the corresponding standard-deviation errors are
0.4659 and 0.1058. No shared, no-condition, separate, or interpolation result
is interpreted from this run because doing so would confound conditional
sharing with single-Condition width recovery.

The machine-readable record is
results/multidimensional/feasibility.json. Its frozen recommendation is to
terminate this avenue for the current target range, ansatz, and objective.
