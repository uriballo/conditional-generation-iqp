# Ashkin-Teller twelve-qubit interpolation result

Frozen n=12 experiment: 20 IQP fits, ten paired classical-baseline bundles, 16 perimeter training conditions and nine new interior test combinations. This follow-up was designed after n=8 results and trains new parameters from scratch. It tests compositional interpolation at a second size, not parameter transfer, extrapolation or asymptotic scaling.

The primary bond-minus-cat mean test-TVD difference is **-0.08437**, paired 95% bootstrap interval **[-0.08919, -0.08016]**. The average favors bond-informed encoding.
The interval excludes zero under this predeclared paired-seed analysis; it does not establish an advantage across other tasks or encoders.

## All predeclared models

| Model | Train TVD | Test TVD [95% CI] | Seed SD | Test C error | Mean worst-condition TVD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Symmetry-only input | 0.59283 | 0.63845 [0.63845, 0.63845] | 0.00000 | 0.19659 | 0.69902 |
| Symmetry-only IQP | 0.25778 | 0.27367 [0.26846, 0.28005] | 0.00991 | 0.11622 | 0.33901 |
| Bond-informed input | 0.27566 | 0.24999 [0.24999, 0.24999] | 0.00000 | 0.19659 | 0.28055 |
| Bond-informed IQP | 0.19902 | 0.18930 [0.18714, 0.19148] | 0.00370 | 0.11272 | 0.24199 |
| Log composition | 0.29078 | 0.31454 [0.30923, 0.31940] | 0.00863 | 0.04074 | 0.34393 |
| Pooled log-linear | 0.11468 | 0.10883 [0.10726, 0.11049] | 0.00277 | 0.01536 | 0.12217 |

TVDs and C errors are exact metrics of the saved output distributions. C error is the RMSE over the six connected cross-layer bond correlations. Means average nine conditions within each seed and then ten seeds. Encoder-only inputs are deterministic controls, not ten independent fitted models. Worst-condition TVD is first maximized within each seed, then averaged. Raw per-seed worst values and marginal errors are included in the JSON.

## Learned contribution and correlation limitations

- Symmetry-only IQP: mean encoder-only minus trained TVD **+0.36478**; 0 of 90 seed/condition pairs worsen TVD. Mean C-error gain **+0.08036**; worsened condition means: none (0 of 90 seed/condition pairs worsen C error).
- Bond-informed IQP: mean encoder-only minus trained TVD **+0.06069**; 0 of 90 seed/condition pairs worsen TVD. Mean C-error gain **+0.08387**; worsened condition means: none (0 of 90 seed/condition pairs worsen C error).

Bond-minus-cat connected-correlation error: -0.00350, descriptive paired interval [-0.00479, -0.00211]. This secondary comparison is not adjusted for multiple testing.
The average correlation comparison does not imply uniform improvement: bond-informed IQP has higher mean C error than symmetry-only IQP at [[0.22, 0.22], [0.22, 0.38], [0.38, 0.22], [0.38, 0.38]]. The largest condition-mean C errors are 0.18671 (cat) and 0.19378 (bond). These remaining errors limit claims of accurate interlayer-dependence recovery.

The classical predictors are contextual comparisons under the same sample access. They need not lose for this to demonstrate the conditional quantum method. They have different representations and supplied assumptions: 3069 pooled log-linear coefficients or 16 orbit histograms, compared with 78 IQP phases. The bond encoder already supplies the uncoupled Ising-ring law and its coupling map, while the cat encoder supplies the partition and both flips. Neither is prior-free.

## Every held-out condition

| (a,b) | Cat input TVD | Cat IQP TVD | Bond input TVD | Bond IQP TVD | Log composition TVD | Log-linear TVD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (0.22,0.22) | 0.69902 | 0.33901 | 0.28055 | 0.24199 | 0.27089 | 0.12213 |
| (0.22,0.38) | 0.69157 | 0.31327 | 0.26606 | 0.20824 | 0.29322 | 0.11785 |
| (0.22,0.54) | 0.64486 | 0.26789 | 0.25101 | 0.17992 | 0.32447 | 0.11432 |
| (0.38,0.22) | 0.69157 | 0.31309 | 0.26606 | 0.20858 | 0.29633 | 0.11748 |
| (0.38,0.38) | 0.66655 | 0.27945 | 0.23968 | 0.18156 | 0.30965 | 0.10988 |
| (0.38,0.54) | 0.59720 | 0.24081 | 0.21714 | 0.16706 | 0.32825 | 0.09927 |
| (0.54,0.22) | 0.64486 | 0.26639 | 0.25101 | 0.18025 | 0.33282 | 0.11431 |
| (0.54,0.38) | 0.59720 | 0.24002 | 0.21714 | 0.16728 | 0.33419 | 0.09945 |
| (0.54,0.54) | 0.51324 | 0.20310 | 0.26130 | 0.16881 | 0.34101 | 0.08474 |

| (a,b) | Input C error | Cat IQP C error | Bond IQP C error | Log composition C error | Log-linear C error |
| --- | ---: | ---: | ---: | ---: | ---: |
| (0.22,0.22) | 0.26034 | 0.18671 | 0.19378 | 0.07925 | 0.02666 |
| (0.22,0.38) | 0.23179 | 0.15079 | 0.15402 | 0.05215 | 0.01399 |
| (0.22,0.54) | 0.19185 | 0.11349 | 0.10783 | 0.02688 | 0.01057 |
| (0.38,0.22) | 0.23179 | 0.15051 | 0.15402 | 0.05943 | 0.01319 |
| (0.38,0.38) | 0.20312 | 0.11580 | 0.11606 | 0.02697 | 0.01013 |
| (0.38,0.54) | 0.16447 | 0.08170 | 0.07291 | 0.02108 | 0.01632 |
| (0.54,0.22) | 0.19185 | 0.11300 | 0.10782 | 0.02865 | 0.01095 |
| (0.54,0.38) | 0.16447 | 0.08148 | 0.07289 | 0.02188 | 0.01697 |
| (0.54,0.54) | 0.12961 | 0.05253 | 0.03516 | 0.05035 | 0.01950 |

## Reproducibility and evaluation boundary

All 20 selected IQP parameters and their test-coordinate predictions, both encoder-only predictions, and all ten classical bundles were saved and sealed before the first test-target evaluation. The sealed predictions were recomputed from fitted parameters before target access. Subsequent fitting is blocked by the evaluation-started marker. No test-based map, architecture, optimizer, regularization, checkpoint or seed selection was made.

Each model used the same seed-specific 1,000 training and 1,000 independent validation examples at each of 16 perimeter conditions. IQP used the validated 500-update Adam-cosine schedule, 2,048 reference draws and 512 masks per bandwidth/condition; selection used fixed sampled validation MMD at training conditions only. Both classical grids and convergence rules were frozen before fitting and selected on validation likelihood. See the [frozen protocol](ashkin-teller-n12-protocol-2026-09-07.md).

This report independently checked the source archive, all sealed prediction hashes, exact IQP probabilities through the NumPy evaluation path, every joint/marginal/correlation metric, normalization and both layer flips. Ten paired seeds are the uncertainty units; the nine conditions are not independent replications. Bootstrap intervals are approximate and conditional on this fixed split.

Manifest file SHA256: `09c993f3288b942c8420618b14338b0dd567819f7821a43c0c2ad7e216982cc1`. Prediction seal SHA256: `310bd888c5d57d6585fbd520d4a41f9e59f0e6081f78050d2162107b5b23b09c`. Evaluation SHA256: `bea9dad01397c8b532a5241ec942e6474c06a556128e41d0bc942117a59f710f`. Raw archives and source snapshot: `artifacts/ashkin-teller-n12-2026-09-07`.

## Claim-to-evidence map

| Claim | Evidence | Limitation | Review status |
| --- | --- | --- | --- |
| Conditional IQP predicts new joint input combinations | Frozen prediction seal, all nine test scores above | Within-domain interpolation; related development history disclosed | Numerically verified; author review pending |
| Additional bond information changes accuracy | Paired primary TVD contrast and full C-error comparison | Changes encoder family and supplied information, not topology alone | Predeclared comparison; author review pending |
| Learned phases add to the supplied input | Per-condition encoder-only gains above | Accuracy and dependence can improve differently | Verified at this split only |
| Classical or quantum advantage | Not claimed | Classical comparators use different parameterizations and priors | Outside scope |
| Generalization beyond this split or size | Not measured | Failed development extrapolation retained; no larger-size extension here | Open |

## Figures

[Paired summary](../../artifacts/ashkin-teller-n12-2026-09-07/figures/confirmation_paired_summary.pdf): mean test TVD and connected-correlation RMSE; faint points are ten seed means, bars are percentile bootstrap intervals over seeds, and grey inputs are deterministic controls. All six models are included.

[Condition grids](../../artifacts/ashkin-teller-n12-2026-09-07/figures/confirmation_test_grid.pdf): all nine condition means for both IQP encoders; each metric uses a shared color scale across encoders. The lower row is connected-correlation RMSE. PNG previews and input/file hashes accompany the PDFs. These are proposed experiment figures, not yet manuscript inserts.
