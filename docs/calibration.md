# Monte Carlo calibration

The calibration was run with 20 independent estimates using
`scripts/calibrate_estimators.py`. The complete results are stored in
`results/calibration.json`.

## Pauli expectation accuracy

Against exact six-qubit expectations, 8,192 IQP samples were the smallest
tested budget satisfying the preregistered RMSE threshold of 0.01:

| IQP samples | RMSE |
|---:|---:|
| 2,048 | 0.0160 |
| 4,096 | 0.0130 |
| **8,192** | **0.0085** |
| 16,384 | 0.0065 |

## Conditional-MMD stability

At representative initialized 19-qubit complete-ansatz parameters, 64
observables with 512 IQP samples already satisfied the 5% coefficient-of-
variation threshold:

| Observables | IQP samples | CV |
|---:|---:|---:|
| **64** | 512 | **4.49%** |
| 128 | 1,024 | 2.96% |
| 256 | 2,048 | 2.01% |
| 512 | 4,096 | 1.20% |

The shared training budget therefore combines the smallest passing observable
count with the 8,192 samples required by the stricter expectation criterion.
The larger validation budget of 256 observables and 16,384 samples yielded a
2.02% MMD coefficient of variation and expectation RMSE 0.0065.
