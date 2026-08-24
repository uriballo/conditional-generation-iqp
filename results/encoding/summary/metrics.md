# Encoding comparison

Matched converged seeds: 0, 1, 2. Lower is better.

| Metric | Angle | Symmetric reupload | Paired difference | Wins |
|---|---:|---:|---:|---:|
| heldout_mmd | 0.000381 | 0.000622 | +0.000240 | 0/3 |
| shell_tvd | 0.063898 | 0.091864 | +0.027966 | 0/3 |
| bitstring_tvd | 0.348079 | 0.345503 | -0.002576 | 2/3 |
| mean_error | 0.132590 | 0.262570 | +0.129980 | 0/3 |
| std_error | 0.186290 | 0.328968 | +0.142678 | 0/3 |

The encodings are identical at the original endpoint angles. Differences therefore measure the learned/interpolated conditional path, not endpoint distinguishability.
