# Frozen stage-3 design: compositional interpolation

This protocol is specified before fitting and before evaluating any of the nine
interior target laws. The executable manifest, paired sample archive, and source
snapshot in `artifacts/ashkin-teller-confirmation-2026-09-07` are the immutable
execution record. This document itself is included in that source freeze.

## Question and scope

Compare symmetry-only and bond-informed inputs to one shared conditional IQP
unitary, under the same circuit and data budget. The primary contrast is
bond minus cat mean held-out joint TVD. Negative favors bond. Report either
sign and both classical predictors, without a requirement for quantum advantage.

Eight visible qubits, no ancillas, four sites per layer, kappa=.3. For levels
{.10,.22,.38,.54,.70}, train on the 16 perimeter pairs and test on the nine
interior pairs {.22,.38,.54} squared, both in lexicographic order. Each interior
coordinate appears on the training perimeter; its joint combination is new.
This tests compositional interpolation, not extrapolation. Methods were developed
on related conditions; several perimeter conditions were previously inspected.
Do not present the entire problem family as untouched development-independent data.

## IQP and data protocol

- Two-block cat with alpha(c)=.225*pi-.20*pi*(c-.1)/.6; bond input with g(c)=c,
  eta=pi/8. Both use the same 30 independent phases from the complete two-local
  bank on the even-even reference support. No aligned four-body addition.
- Ten new paired seeds 2701 through 2710. Each gets 1,000 training and 1,000
  independent validation samples per perimeter condition. RNG uses
  `default_rng(SeedSequence([seed,3,role,condition_index]))`. All models receive
  identical data for the same seed. There is no sampled held-out target archive.
- Initialize the original 36 phases from pooled training data, then add phases
  with identical support characters to get the same 30-coordinate initialization
  for both encoders. No data-dependent encoder or map selection.
- 500 Adam-cosine updates (.01 to .001), batches of 100 per condition,
  512 nonidentity masks per bandwidth/condition with multiplicity weights,
  2,048 fresh reference draws per update/bandwidth/condition. Use the same
  three expected-bodyness bandwidths as development.
- Validation uses a fixed sampled-MMD reference key and independent validation
  data at training conditions only. Choose minimum validation MMD among the
  initial checkpoint and evaluations every 25 updates. No test-based selection.

## Classical predictors, fixed before fitting

Both represent the 64 orbits of the independent layer flips; each orbit mass
is divided among its four states. They know the layer partition, flips and
condition coordinates, but receive no ring Hamiltonian features. Their favorable
log-additivity/affine-log-odds assumptions and larger parameter budgets must be
reported. Validation uses mean per-example orbit negative log likelihood; the
omitted log(4) is a common constant. No validation samples are used for fitting.

**Empirical log composition.** Fit one orbit histogram per perimeter condition,
with pseudocount lambda per orbit, selected from {.1,1,10}. For every prediction,
let lo=.1 and hi=.7 and compute

L(a,b) = .5 log q(a,lo) + .5 log q(a,hi)
       + .5 log q(lo,b) + .5 log q(hi,b)
       - .25 sum_(four corners) log q(corner).

Apply softmax over the 64 orbit logits. These are the actual supplied edge
histograms, not interpolation from only four corner histograms. This is the
average log of the four corner-based composition identities. Select the
pseudocount by this predictor's validation likelihood across all 16 perimeter
conditions. The baseline estimates 16*63 histogram degrees of freedom; it is
not parameter-matched to IQP.

**Pooled conditional log-linear model.** Orbit logits are
[1,(a-.4)/.3,(b-.4)/.3] times a (3,63) coefficient matrix, with the last orbit
fixed at zero. Fit mean training NLL plus lambda/2 times the sum of squared
coefficients, including intercepts. Grid: {1e-6,1e-5,1e-4,1e-3,1e-2,1e-1}.
Each candidate starts at zero and uses L-BFGS-B, maxiter=5000, ftol=1e-14,
gtol=1e-8. Require finite success and gradient infinity norm <=1e-6. A failed
candidate may retry the same objective from its last iterate with maxiter=20000;
persistent numerical failure stops evaluation for diagnosis, not grid expansion.
Select lowest unpenalized validation NLL; ties use first grid entry. Retain
candidate diagnostics, selected coefficients and counts. There are 189 fitted
coefficients; no hidden ring sufficient statistics.

The composition identity is checked using synthetic additive orbit logits,
including nonlinear level effects. This numerical check does not evaluate the
nine Ashkin-Teller held-out target laws.

## Prediction seal and evaluation boundary

First save selected parameters, training predictions, all nine test-coordinate
predictions, and encoder-only predictions for every IQP fit. Save predictions
and fitted information for both classical baselines for every seed. Verify all
20 IQP and ten classical bundles and write a hash seal. Only then create an
evaluation-started marker and evaluate the analytic target at the nine test
pairs. The runner refuses further fitting after that marker. Retries of
read-only evaluation must match the same seal. There is no outcome-based early
stopping or architecture/optimizer change after test access.

## Metrics and uncertainty

Report exact joint TVD, both layer-marginal TVDs, and the four-bond connected
cross-layer correlation RMSE at every test condition. Include encoder-only
comparisons, per-seed mean and worst errors, and train/test distinctions. Never
hide a condition where training or supplied physics hurts a metric.

The independent replication unit is the seed, not the test condition. Average
nine condition errors within each seed. For the primary bond-minus-cat mean-TVD
contrast, bootstrap the ten paired seed indices 10,000 times, RNG seed950701,
and report the mean difference and percentile 95% interval. Per-model uncertainty
and secondary metric contrasts use the same paired resampling and are descriptive,
without multiple-testing-adjusted significance claims. Report seed SDs, all seed
means and condition-level values so the small replication count is visible.

## Stopping and paper interpretation

Complete 20 IQP fits and ten paired baseline bundles, integrity checks and
reported evaluation regardless of which model wins. The development gate already
passed; no new performance gate is imposed. Correlation failures from development
remain documented. Stage 4 size/circuit expansion and stage 5 manuscript inclusion
are separate decisions after this result; this run does not authorize claiming
quantum advantage or uniformly accurate learned interlayer dependence.
