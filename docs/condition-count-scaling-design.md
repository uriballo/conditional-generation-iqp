# Condition-count scaling design

Status: frozen before scaling runs.

## Question and scope

This study tests whether one fixed-capacity, shared Angle-conditioned IQP
model can fit a growing finite family of Gaussian Hamming-weight Targets. The
independent variable is the number of supervised Conditions
`K in {2, 3, 5, 9}`. The experiment is a trained-Condition scaling study; it is
not the held-out interpolation experiment in `docs/interpolation-design.md`.

The model is the complete 2-local ansatz on 19 qubits. It has one independent
one-qubit generator and one independent two-qubit generator for every pair,
for 190 trainable circuit parameters. The Condition enters only through the
broadcast Angle product-state encoding. The estimator, three MMD bandwidths,
Adam/cosine optimizer family, and checkpoint rule are inherited from
`docs/experiment-design.md`.

## Condition grids

Write `f = alpha / pi`. For each K, the trained grid is

`f_j = j / (4 (K - 1))`, for `j = 0, ..., K - 1`.

Thus the exact grids are:

| K | `alpha / pi` | Target means for n=19 |
|---:|---|---|
| 2 | `0, 1/4` | `9.5, 4.75` |
| 3 | `0, 1/8, 1/4` | `9.5, 7.125, 4.75` |
| 5 | `0, 1/16, 1/8, 3/16, 1/4` | `9.5, 8.3125, 7.125, 5.9375, 4.75` |
| 9 | `0, 1/32, ..., 1/4` in steps of `1/32` | affine path values |

The grids are nested: the K=2 grid is contained in K=3, which is contained in
K=5, which is contained in K=9. The target at every trained angle is the
existing fixed-width discrete Gaussian with

`mu(alpha) = n/2 - n alpha/pi` and `width = sqrt(n)/2`.

No interleaved Held-out Conditions enter training, validation, checkpoint
selection, or the primary evaluation. This keeps the independent variable
Condition count and avoids conflating this track with conditional
interpolation.

## Data and optimization budgets

Two views are frozen:

1. **Fixed-total-data (primary):** 18,000 Training samples and 3,600
   validation samples per K, allocated evenly as 9,000/1,800 for K=2,
   6,000/1,200 for K=3, 3,600/720 for K=5, and 2,000/400 for K=9.
2. **Fixed-per-Condition-data (secondary):** 10,000 Training samples and
   2,000 validation samples per Condition, matching the existing experiment.
   Total samples therefore grow from 20,000/4,000 at K=2 to
   90,000/18,000 at K=9.

Every track uses exactly 2,000 optimizer updates and a balanced total
minibatch of 450 target samples per update. The per-Condition minibatch is
`450/K` (225, 150, 90, and 50 respectively), so every K consumes 900,000
sample draws during optimization. Sampling with replacement is allowed, as in
the existing training core. Validation is evaluated every 25 updates using
the fixed validation estimator. The lowest validation Conditional MMD is the
only checkpoint-selection criterion.

The fixed-total track is primary because it prevents larger K from receiving
more data. The fixed-per-Condition track is a secondary descriptive analysis
of the data-allocation trade-off; it does not replace the primary track.

## Compared models and parameter accounting

For each K, track, and seed, train:

- **Shared Angle:** one 190-parameter model with the K distinct grid angles.
- **No-condition shared:** one 190-parameter model receiving the same
  midpoint encoding `alpha = pi/8` for every Target. It sees the same K
  labeled Target arrays and balanced objective, but has no Condition signal.
- **Separate bank:** K independently trained 190-parameter models, one per
  Target and its grid angle.

Trainable-parameter accounting is explicit:

| Method | Models | Trainable parameters |
|---|---:|---:|
| Shared Angle | 1 | 190 |
| No-condition shared | 1 | 190 |
| Separate bank | K | `190 K` |

Optimizer state, sampled observables, and data are not counted as trainable
parameters. No grouped or capacity-matched control is added: a clean grouped
construction would change the question from Condition-count scaling to a
capacity ablation.

## Seeds and matched randomness

Seeds 0--9 are the frozen matched seed set. For a seed, all methods use the
same generated Target data, validation data, target order, and fixed
condition-balanced sampling schedule wherever shapes permit. The estimator
random streams are derived deterministically from the same top-level seed and
the existing training function. Runs are independent files and can be resumed
with `--skip-existing`.

## Evaluation and aggregation

For every trained Condition, enumerate all `2^19` bitstrings and record:

- Hamming-weight profile TVD;
- full Bitstring TVD;
- target/model Hamming-weight mean and standard deviation errors; and
- the complete target and model shell profiles.

For every K, method, and seed, retain the full per-Condition profile. Summaries
are computed from per-seed Condition summaries, never from a pooled sample:
mean-Condition, median-Condition, and worst-Condition TVD are all reported.
Across the ten seeds, report the median and interquartile range of each of
those summaries. The shared-model summary also reports the fidelity gap
(shared TVD minus Separate-bank TVD) for each aggregation and Condition.

The audit retains strict counts at the individual Condition level. In
particular, it reports how many of the `10 K` shared-vs-No-condition and
shared-vs-Separate comparisons fail for each metric, rather than allowing a
mean to hide a failed Condition.

## Frozen decision rule

The primary fixed-total track is said to scale usefully only if all statements
below hold separately for profile TVD and Bitstring TVD:

1. At each K in `{3, 5, 9}`, the shared model's median across seeds of
   mean-Condition TVD is at least 20% lower than No-condition's corresponding
   median, and its median worst-Condition TVD is no larger than the
   No-condition value plus 0.05.
2. Relative to K=2, the shared model's median mean-Condition TVD increases by
   no more than 0.10 and its median worst-Condition TVD by no more than 0.15
   at K=3, 5, and 9.
3. Strict Condition-level failure counts are disclosed; no K may have more
   than half of its Conditions fail the shared-vs-No-condition comparison at
   the seed-median threshold.
4. For K >= 3, the shared model uses at least 50% fewer trainable parameters
   than the Separate bank and its fidelity gap to the Separate bank is
   reported. Parameter savings alone cannot pass the rule.

The secondary fixed-per-Condition track receives the same audit tables but is
descriptive. A failure at any single Condition remains visible in the
per-Condition tables and strict counts even when all aggregate criteria pass.

## Artifacts and completion audit

Runs are stored as one JSON and one compressed NPZ per K/track/method/seed.
`scripts/summarize_condition_scaling.py` consumes only these audited files and
produces per-Condition metrics, aggregate tables with seed IQRs, strict audit
JSON/Markdown, and the fidelity-versus-K/parameter figure. It does not create
or modify manuscript files. The final `audit.md` records each frozen criterion
as PASS or FAIL and includes a claim-evidence assessment: strengthen the
compact paper only if the primary rule passes; otherwise treat the result as a
follow-up positive, mixed, or negative scaling study.
