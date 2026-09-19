# Gaussian ablation backlog

Status: **1D items run 2026-09-19** with `blocks:anti` as the winning encoder
(report: `prototypes/overnight_verify/REPORT.md`; staging plan in
`ABLATION_PLAN.md`).  Outcomes: three clustered anchors `{0, 1/4, 1/2}` meet
the headline at r = 0 (0.100 held, worst condition 0.164) and generalise
off-grid (`c = 0.45` TVD 0.03); non-commuting reuploading and per-qubit
conditioning are closed as negative results; best-of-3 random init is the
protocol (0.070 over ten seeds, zero failures).  2D items were not run.
Scope: the one-parameter (1D) and two-parameter (2D) Gaussian studies.
Sequencing: 1D done; 2D deferred to Phase 2 after the 1D protocol settles.

## Headline goal

The first paper experiment asks "does this approach work?".  Its target is
**exact bitstring TVD below 0.2 with 3 training conditions and 6 evaluation
conditions**.  Every item below reports the goal at r = 0 and then checks the
effect of ancillas (1D ladder `1, 3, 6, 10`; 2D `10`), so we can say whether
the encoding change survives marginalization.

## Current baselines to compare against

- 1D, library defaults, n = 8, r = 0, 3 seeds: production map `s = 0.70` gives
  edge TVD 0.426 (c = 0, 1), interior 0.203, held-out mean 0.222; the retuned
  slope `s = 0.55` gives 0.299 / 0.213 / 0.208.  n = 12 reproduces the edge
  gap.  This is comparable to the old campaign's r = 0 numbers; ancillas are
  the old campaign's main lever.
- 2D, library defaults, n = 8, r = 0, 3 seeds: production `broadcast` is
  exactly width-blind (held-out TVD 0.098); `bs:0.10` resolves width
  (held-out 0.109, resolution error 0.026); `two_block` resolves width but
  costs 0.217 with three times the within-shell residual.  Non-commuting
  phase layers did not help here.
- Prototype artifacts: `prototypes/gaussian_edge_map/` and
  `prototypes/gaussian2d_layers/` (gitignored).

## Ideas

### 1. Number of training conditions

How does held-out performance change as the number of training conditions
grows, and can 2-3 training conditions already work?  Build nested subsets of
the frozen grids (1D: `{0, ..., 1}` in eighths or sixteenths; 2D: 2x2 corners,
3x3, perimeter grid) and report held-out exact TVD per condition against the
count.  The question matters for labeling the benchmark as interpolation
rather than extrapolation.

Ancillas: repeat the winning count at each rung.

### 2. Unevenly spaced conditions

Train on only `c in {0, 0.25}` and evaluate at `0.45` and `0.9`: does the model
generalize when the training conditions are clustered and the evaluation
points are far or off-grid?  This tests whether performance depends on even
coverage of the condition range rather than on the number of points.  Report
per-condition TVD and the moment errors, and compare with the evenly spaced
case at the same count.  2D analog: train on one corner cluster and evaluate
on distant interior conditions.

Ancillas: repeat with ancillas to see whether marginalization widens the
effective training support.

### 3. Data reuploading of non-commuting gates

Does a layered encoder with non-commuting rotations (`RY`-`RZ`-`RY`, or
`RY`-`RX`-`RY`, with condition-dependent angles per layer) beat the single
commuting `RY` layer?  The extra local phase is a second condition-dependent
knob per qubit; in 2D it could carry the width coordinate without breaking
permutation symmetry.  Compare against the single-layer map at equal training
budgets, and include the commuting two-layer control (fixed intermediate
axis) so the gain is attributable to non-commutation, not depth.

Ancillas: test at r = 0 and r = 10 for both studies.

### 4. Condition not shared across qubits

What if `c` enters each qubit differently, i.e. the angle map is
qubit-dependent (`alpha_i(c)`) instead of broadcast?  More general conditioning
may help if the readout is not permutation symmetric, but it breaks input
exchangeability, so the learned circuit must restore within-shell uniformity.
Evaluate the cost directly: exact TVD, profile TVD, and (2D) within-shell
residual against the broadcast map.  The paper needs at least one control
showing whether per-qubit conditioning is worth the symmetry price.

Ancillas: with and without ancillas, since ancillas can restore some of the
lost symmetry.

### 5. Exponential / geometric per-qubit encoding (tested, parked)

Encode qubit `i` with a scaled angle `2^i alpha(c)` (exponential feature
encoding from the Fourier-expressivity / data-reuploading literature), or a
no-wrap geometric tilt `r^i alpha(c)` with `1 < r < 2`, instead of the same
value on every qubit.  Tested 2026-09-18 on the 1D Gaussian, 5 anchors:

- held-out exact TVD is 0.35-0.43 at r=10 (broadcast: 0.072) and 0.31-0.36 at
  r=0 (broadcast: 0.205) for `r = 1.3, 1.6, 2.0`;
- the r=0 oracle cannot even fit the family (train 0.31-0.34, held 0.24-0.27),
  so this is a representability problem, not just optimization;
- every heterogeneous per-qubit map tried (wedge, blocks, geometric) loses to
  the broadcast map.

Parked unless a different ansatz or target family makes exchangeability less
important.  Details: `prototypes/gaussian_1d/RESULTS.md`, spec `geometric`.

## Measurements common to every item

- Exact bitstring TVD per condition, split into training and held-out; profile
  TVD and moment errors; 2D additionally the width-resolution diagnostic.
- Library default training budgets unless the item is itself a budget
  ablation; screening at 3 seeds, headline numbers at the study seed set.
- Every run through `experiments.common.runner` with its own run directory.
