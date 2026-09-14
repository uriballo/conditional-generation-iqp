# Reference-estimator precision and cost — completed 2026-09-07

All 240 production bundles completed and were checked against the frozen manifest, source hashes and NPZ digests. The study contains 7,680 independent repeated joint moment/Jacobian evaluations on 60 fixed fixtures. Ten runner/reference/preparation tests and two analysis tests passed. Independent physical-input moment/selected-gradient checks passed for all 60 fixtures, with maximum disagreement 2.69e-14. All 36 plotted aggregate moment/gradient points were independently recomputed from the saved arrays. PDF and PNG figures were visually checked.

## Findings

Across the nine synthetic encoder/width combinations, increasing the reference budget from 128 to 8,192 (64 times as many samples) reduced moment RMSE by 7.47–8.34 times and gradient RMSE by 7.57–8.28 times. These results are consistent with inverse-square-root Monte Carlo error in these fixtures. At S=8,192 the synthetic aggregate moment RMSEs are 0.00727–0.00773 and the gradient RMSEs are 0.00889–0.01045.

The bond-informed reference has comparable aggregate error but a larger computational cost. At N=16 and S=8,192, the median warmed call took 12.34 ms (product), 18.47 ms (symmetry-only), and 39.13 ms (bond-informed). Each call returns 16 moments and their full 136-phase Jacobian and includes reference sampling. These are CPU implementation measurements on the documented Strix setup, with compilation measured separately. Width also changes the number of gradient components; timings are not asymptotic complexity estimates. The compilation curve is not monotonic for product inputs and is reported without filtering. An 18-process follow-up with persistent caching disabled reproduced the N=16 dip on both hosts: backend compilation medians were 0.078–0.080 s at N=16 versus 0.220 s at N=12, despite increasing estimated arithmetic and warmed runtime. The responsible compiler mechanism is unresolved. [Complexity accounting and diagnostic evidence](reference-estimator-complexity-2026-09-07.md).

No large error is hidden by the aggregate at the final budget: over all 60 cases, including the six trained checkpoints, maximum case moment RMSE was 0.00912 and maximum gradient RMSE was 0.01096. Identity/parity-forbidden moment controls agree within 1.35e-14. Those controls are excluded from active error curves. The studied masks are fixed; mask-bank error and full-kernel loss accuracy are not measured.

## Trained checkpoints at S=8,192

These checks use fixed smallest-seed selected checkpoints, not a population sample of trained models. They are kept outside the main synthetic curves.

| Case | Moment RMSE | Gradient RMSE |
| --- | ---: | ---: |
| n12_bond_trained | 0.00742 | 0.01003 |
| n12_cat_trained | 0.00766 | 0.00966 |
| n12_product_trained | 0.00855 | 0.00953 |
| n8_bond_trained | 0.00721 | 0.00960 |
| n8_cat_trained | 0.00696 | 0.00896 |
| n8_product_trained | 0.00912 | 0.00813 |

Per-case and aggregate percentile intervals are in `summary.json`. They resample independent Monte Carlo repetition vectors within each fixed case; cases, masks and gradient components are not independent replications. Bias estimates are descriptive, not proofs of unbiasedness. Process RSS includes exact/JIT allocations and prior cases, so it is not an isolated estimator-memory measurement.

## Implication for the paper

This is a useful empirical companion to the estimator derivation: the correlated-reference construction retained comparable additive estimation accuracy at the tested widths, with measured overhead. It does not establish efficient optimization, generalization, large-width floating-point stability, full-distribution accuracy, or quantum advantage. The primary main-text candidate is the accuracy figure, with the cost figure or table accompanying the resource discussion. The explicit arithmetic/sample-complexity accounting has been added to the main Methods; integration of the empirical accuracy/cost figures remains pending author review. The data-efficiency study has not started.

[Protocol](reference-estimator-study-protocol-2026-09-07.md), [manifest](../../artifacts/reference-estimator-study-2026-09-07/manifest.json), [frozen sources](../../artifacts/reference-estimator-study-2026-09-07/freeze.json), [complete summary](../../artifacts/reference-estimator-study-2026-09-07/report/summary.json), [accuracy figure](../../output/pdf/estimator-study/reference_estimator_accuracy.pdf), [cost figure](../../output/pdf/estimator-study/reference_estimator_cost.pdf).
