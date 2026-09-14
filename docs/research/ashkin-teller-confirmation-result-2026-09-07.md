# Ashkin-Teller confirmatory interpolation result

Frozen stage-3 experiment: 20 IQP fits, ten paired classical-baseline bundles, 16 perimeter training conditions and nine new interior test combinations. The method was developed on related conditions; this is compositional interpolation, not extrapolation.

The primary bond-minus-cat mean test-TVD difference is **-0.03946**, paired 95% bootstrap interval **[-0.04175, -0.03703]**. The average favors bond-informed encoding.
The interval excludes zero under this predeclared paired-seed analysis; it does not establish an advantage across other tasks or encoders.

## All predeclared models

| Model | Train TVD | Test TVD [95% CI] | Seed SD | Test C error | Mean worst-condition TVD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Symmetry-only input | 0.32782 | 0.33999 [0.33999, 0.33999] | 0.00000 | 0.19140 | 0.44575 |
| Symmetry-only IQP | 0.12748 | 0.13271 [0.12948, 0.13549] | 0.00522 | 0.06620 | 0.16592 |
| Bond-informed input | 0.26003 | 0.27707 [0.27707, 0.27707] | 0.00000 | 0.19140 | 0.30017 |
| Bond-informed IQP | 0.10685 | 0.09325 [0.09159, 0.09496] | 0.00289 | 0.06156 | 0.13829 |
| Log composition | 0.08146 | 0.09881 [0.09337, 0.10500] | 0.00993 | 0.03539 | 0.12241 |
| Pooled log-linear | 0.03193 | 0.02595 [0.02460, 0.02742] | 0.00241 | 0.00937 | 0.03394 |

TVDs and C errors are exact metrics of the saved output distributions. C error is the RMSE over the four connected cross-layer bond correlations. Means average nine conditions within each seed and then ten seeds. Encoder-only inputs are deterministic controls, not ten independent fitted models. Worst-condition TVD is first maximized within each seed, then averaged. Raw per-seed worst values and marginal errors are included in the JSON.

## Learned contribution and correlation limitations

- Symmetry-only IQP: mean encoder-only minus trained TVD **+0.20729**; 0 of 90 seed/condition pairs worsen TVD. Mean C-error gain **+0.12520**; worsened condition means: none (0 of 90 seed/condition pairs worsen C error).
- Bond-informed IQP: mean encoder-only minus trained TVD **+0.18382**; 0 of 90 seed/condition pairs worsen TVD. Mean C-error gain **+0.12984**; worsened condition means: none (0 of 90 seed/condition pairs worsen C error).

Bond-minus-cat connected-correlation error: -0.00464, descriptive paired interval [-0.00606, -0.00317]. This secondary comparison is not adjusted for multiple testing.
The smaller average correlation error does not imply uniform improvement: bond-informed IQP has higher mean C error than symmetry-only IQP at [[0.54, 0.54]]. The largest condition-mean C errors are 0.15634 (cat) and 0.15346 (bond). These remaining errors limit claims of accurate interlayer-dependence recovery.

The classical predictors are contextual comparisons under the same sample access. They need not lose for this to demonstrate the conditional quantum method. They have different representations and supplied assumptions: 189 pooled log-linear coefficients or 16 orbit histograms, compared with 30 IQP phases. The bond encoder already supplies the uncoupled Ising-ring law and its coupling map, while the cat encoder supplies the partition and both flips. Neither is prior-free.

## Every held-out condition

| (a,b) | Cat input TVD | Cat IQP TVD | Bond input TVD | Bond IQP TVD | Log composition TVD | Log-linear TVD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (0.22,0.22) | 0.44575 | 0.16592 | 0.24361 | 0.13829 | 0.11582 | 0.03394 |
| (0.22,0.38) | 0.39336 | 0.15064 | 0.26057 | 0.10268 | 0.11006 | 0.02844 |
| (0.22,0.54) | 0.33423 | 0.12617 | 0.27423 | 0.08113 | 0.10237 | 0.02748 |
| (0.38,0.22) | 0.39336 | 0.15038 | 0.26057 | 0.10181 | 0.10749 | 0.02928 |
| (0.38,0.38) | 0.33315 | 0.13592 | 0.28464 | 0.08736 | 0.10144 | 0.02354 |
| (0.38,0.54) | 0.28364 | 0.11475 | 0.29779 | 0.07993 | 0.08772 | 0.02233 |
| (0.54,0.22) | 0.33423 | 0.12591 | 0.27423 | 0.08084 | 0.09612 | 0.02805 |
| (0.54,0.38) | 0.28364 | 0.11465 | 0.29779 | 0.08032 | 0.09002 | 0.02158 |
| (0.54,0.54) | 0.25858 | 0.11003 | 0.30017 | 0.08687 | 0.07826 | 0.01889 |

| (a,b) | Input C error | Cat IQP C error | Bond IQP C error | Log composition C error | Log-linear C error |
| --- | ---: | ---: | ---: | ---: | ---: |
| (0.22,0.22) | 0.27933 | 0.15634 | 0.15346 | 0.04490 | 0.01087 |
| (0.22,0.38) | 0.24006 | 0.10595 | 0.09934 | 0.04076 | 0.01009 |
| (0.22,0.54) | 0.18325 | 0.06077 | 0.04430 | 0.02970 | 0.00959 |
| (0.38,0.22) | 0.24006 | 0.10598 | 0.09938 | 0.03821 | 0.01076 |
| (0.38,0.38) | 0.19967 | 0.05707 | 0.04492 | 0.03510 | 0.00888 |
| (0.38,0.54) | 0.14667 | 0.01942 | 0.00974 | 0.03503 | 0.00784 |
| (0.54,0.22) | 0.18325 | 0.06085 | 0.04436 | 0.03375 | 0.01072 |
| (0.54,0.38) | 0.14667 | 0.01945 | 0.00971 | 0.03145 | 0.00860 |
| (0.54,0.54) | 0.10366 | 0.00997 | 0.04884 | 0.02957 | 0.00702 |

## Reproducibility and evaluation boundary

All 20 selected IQP parameters and their test-coordinate predictions, both encoder-only predictions, and all ten classical bundles were saved and sealed before the first test-target evaluation. The sealed predictions were recomputed from fitted parameters before target access. Subsequent fitting is blocked by the evaluation-started marker. No test-based map, architecture, optimizer, regularization, checkpoint or seed selection was made.

Each model used the same seed-specific 1,000 training and 1,000 independent validation examples at each of 16 perimeter conditions. IQP used the validated 500-update Adam-cosine schedule, 2,048 reference draws and 512 masks per bandwidth/condition; selection used fixed sampled validation MMD at training conditions only. Both classical grids and convergence rules were frozen before fitting and selected on validation likelihood. See the [frozen protocol](ashkin-teller-confirmation-protocol-2026-09-07.md).

This report independently checked the source archive, all sealed prediction hashes, exact IQP probabilities through the NumPy evaluation path, every joint/marginal/correlation metric, normalization and both layer flips. Ten paired seeds are the uncertainty units; the nine conditions are not independent replications. Bootstrap intervals are approximate and conditional on this fixed split.

Manifest file SHA256: `e5272c250d9cb5203fade3486ffac4c1fc6d21c44f7a63d5a456024384e2c6cb`. Prediction seal SHA256: `61cde31dea9ab053262412a529be428ad747074c1577ce3096085b5a4a7ba6e4`. Evaluation SHA256: `e66b246ee175040a44086e8f0aa39fd22c6d433d82adaddc32dd401741ddaa97`. Raw archives and source snapshot: `artifacts/ashkin-teller-confirmation-2026-09-07`.

## Claim-to-evidence map

| Claim | Evidence | Limitation | Review status |
| --- | --- | --- | --- |
| Conditional IQP predicts new joint input combinations | Frozen prediction seal, all nine test scores above | Within-domain interpolation; related development history disclosed | Numerically verified; author review pending |
| Additional bond information changes accuracy | Paired primary TVD contrast and full C-error comparison | Changes encoder family and supplied information, not topology alone | Predeclared comparison; author review pending |
| Learned phases add to the supplied input | Per-condition encoder-only gains above | Accuracy and dependence can improve differently | Verified at this split only |
| Classical or quantum advantage | Not claimed | Classical comparators use different parameterizations and priors | Outside scope |
| Generalization beyond this split or size | Not measured | Failed development extrapolation retained; no n=12 extension here | Open |

## Figures

[Paired summary](../../artifacts/ashkin-teller-confirmation-2026-09-07/figures/confirmation_paired_summary.pdf): mean test TVD and connected-correlation RMSE; faint points are ten seed means, bars are percentile bootstrap intervals over seeds, and grey inputs are deterministic controls. All six models are included.

[Condition grids](../../artifacts/ashkin-teller-confirmation-2026-09-07/figures/confirmation_test_grid.pdf): all nine condition means for both IQP encoders; each metric uses a shared color scale across encoders. The lower row is connected-correlation RMSE. PNG previews and input/file hashes accompany the PDFs. These are proposed experiment figures, not yet manuscript inserts.
