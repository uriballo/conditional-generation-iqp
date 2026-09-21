# Scalar-Gaussian training configuration review

The scientific settings below reflect the September 21 protocol discussion.
The implementation and two-host partition pass their local checks. The user
authorized the largest-register pilot and subsequent production launch on
September 21, subject to the pilot checks below.

## Accepted settings

| Component | Selected value |
| --- | --- |
| Optimizer | Adam with cosine decay from \(10^{-2}\) to \(10^{-3}\) |
| Base horizon | 2,000 updates |
| Shared budgets | 2,000 total updates and \(2,000|C|\) condition-matched updates |
| Condition sampling | One uniformly sampled training condition per update |
| Target batch | 100 samples from a fixed 1,000-sample training split |
| Hamming-RBF scales | Two scales targeting mask bodyness 2 and 4; the second target is capped at 3.92 for \(n=8\) |
| Reference samples | 1,024 per update |
| Operator masks | 128 per update at \(n=8\), 512 at \(n=12\) |
| Mask handling | Fresh nonidentity masks at each update, exact duplicate compression, reference samples shared across masks |
| Model selection | Mean exact TVD over the training conditions |
| Sample diagnostic | Held-out MMD from 1,000 target samples per condition |
| Initialization | Independent \(\mathcal N(0,0.5^2)\) parameters |
| Restarts | 3 per data seed, with target data and estimator randomness held fixed |
| Data seeds | 10, seeds 100 through 109 |
| Numeric path | Float32 phase contraction, float64 reductions, sample chunks of 256 |
| Ansatz | Complete one- and two-body graph, \(P=N(N+1)/2\) parameters |
| Baselines | Direct no-encoder circuits, anchor-output mixtures, and the selected encoder without a trained IQP unitary |
| Encoder ablations | Omitted from the scalar campaign |

Each condition-specific circuit receives 2,000 updates. The fixed-total shared
model therefore uses fewer updates per condition, while the condition-matched
shared model receives \(2,000|C|\) total updates. Reporting both comparisons
separates total-compute efficiency from matched condition exposure.

## Remaining launch checks

- Run one (n=12,r=10) pilot to verify the memory and wall-time cost of exact
  TVD every 25 updates before starting the full shards.
- If the pilot is acceptable, start modulo-two shards on the two Strix hosts.

The local checks confirm the two derived bandwidths, the 128/512-mask budgets,
exact-TVD checkpoint and restart selection, diagnostic-only held-out MMD, and a
balanced modulo-two manifest.
