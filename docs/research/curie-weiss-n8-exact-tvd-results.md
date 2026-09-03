# Curie--Weiss (n=8): complete Exact-TVD results

## Scope

This note reports the completed (n=8) Curie--Weiss production block: five ancilla counts, ten seeds, one shared conditional model, six independent training-condition models, and five target-aware held-out models per seed and ancilla count. All 600 fits and their exact evaluations are complete. The primary metric is exact visible-distribution total variation distance (TVD). We average conditions within each seed before computing medians, interquartile ranges, and paired bootstrap intervals.

## Main result

Ancillas substantially improve the shared conditional model. Its median held-out TVD decreases from (0.1903) without ancillas to (0.0470) with ten ancillas. Every nonzero-ancilla configuration improves on (r=0) in all ten matched seeds. Most of the gain appears between one and three ancillas; improvements beyond six ancillas are smaller.

| Ancillas (r) | Shared conditional | Adjacent independent interpolation | Target-aware independent | Generalization advantage | Sharing cost |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1903 | 0.1171 | 0.1528 | -0.0751 | +0.0377 |
| 1 | 0.1731 | 0.0859 | 0.1182 | -0.0873 | +0.0551 |
| 3 | 0.0705 | 0.0675 | 0.0939 | -0.0029 | -0.0225 |
| 6 | 0.0537 | 0.0630 | 0.0856 | +0.0101 | -0.0318 |
| 10 | 0.0470 | 0.0593 | 0.0831 | +0.0123 | -0.0357 |

Values are medians over ten seed-level averages across the five held-out temperatures. Generalization advantage is interpolation TVD minus shared-model TVD, so positive values favor the shared model. Sharing cost is shared-model TVD minus oracle TVD, so negative values favor the shared model.

The shared model overtakes interpolation at six ancillas. This conclusion is paired and consistent across all ten seeds at both (r=6) and (r=10). At (r=3), the two approaches are close, but interpolation remains better in all ten paired seed averages.

The shared model also outperforms the target-aware independent fits at (r=3,6,10), again in all ten paired seeds. This is not evidence that the oracle lacks access to the target condition: each oracle is trained directly on that condition. A more plausible interpretation is that parameter sharing across temperatures provides useful statistical regularization or improves optimization. We should retain the term *target-aware independent reference* and avoid calling it a strict performance upper bound.

## Temperature dependence

The median held-out TVDs of the shared model are:

| (r) | β=0.6 | β=0.8 | β=1.0 | β=1.2 | β=1.4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2235 | 0.1840 | 0.1752 | 0.1887 | 0.1903 |
| 1 | 0.1667 | 0.1839 | 0.2053 | 0.1683 | 0.1402 |
| 3 | 0.0532 | 0.0577 | 0.0864 | 0.0834 | 0.0739 |
| 6 | 0.0324 | 0.0537 | 0.0631 | 0.0708 | 0.0482 |
| 10 | 0.0261 | 0.0526 | 0.0484 | 0.0717 | 0.0351 |

Performance is not pointwise monotone in (r). In particular, one ancilla worsens the critical-point result, and (r=10) is slightly worse than (r=6) at β=1.2. The aggregate ancilla trend is nevertheless strong and reproducible. At the critical point, the median TVD falls from (0.1752) at (r=0) to (0.0484) at (r=10). The critical-window median falls from (0.1810) to (0.0572).

## Claims supported at this stage

- Ancillas materially improve the (n=8) conditional Curie--Weiss model.
- Six ancillas are sufficient for the shared model to outperform adjacent-bank interpolation on the held-out temperatures.
- The model captures the critical region much more accurately with three or more ancillas.
- Joint training across temperatures can outperform separately trained circuits at the same circuit size, suggesting a benefit from sharing data or optimization structure.

## Limits

- These results establish the trend only at eight visible qubits. The (n=12) and (n=16) blocks remain necessary for a scaling claim.
- The symmetry-informed condition map contains the known transition location. The held-out critical-point result must therefore be presented as conditional distribution fitting under an informed inductive bias, not as discovery of the critical point.
- Negative held-out-minus-training TVD does not by itself demonstrate unusually good interpolation because the training and held-out condition sets differ in intrinsic difficulty.
- The independent reference is target-aware but is not a mathematical oracle or guaranteed upper bound.

## Recommended manuscript emphasis

Lead with the ancilla ladder and the crossover against adjacent independent interpolation. Use the temperature-resolved plot to show that the gain extends through the transition rather than arising from a single easy condition. Use the β=1 profile plot to establish whether the improvement reflects recovery of the bimodal magnetization structure, not merely a lower scalar TVD.
