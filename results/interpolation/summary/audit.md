# Held-out interpolation audit

## Frozen tests

| Test | Metric | Angle / pi | Dense | Baseline | Result |
|---|---|---:|---:|---:|---:|
| continuous conditioning vs oracle adjacent lookup | shell tvd | 0.03125 | 0.0526 | 0.1231 | PASS |
| continuous conditioning vs oracle adjacent lookup | shell tvd | 0.09375 | 0.0689 | 0.1005 | PASS |
| continuous conditioning vs oracle adjacent lookup | shell tvd | 0.15625 | 0.0732 | 0.0691 | FAIL |
| continuous conditioning vs oracle adjacent lookup | shell tvd | 0.21875 | 0.0461 | 0.0651 | PASS |
| dense supervision vs endpoint-only | shell tvd | mean | 0.0609 | 0.1596 | PASS |
| continuous conditioning vs oracle adjacent lookup | bitstring tvd | 0.03125 | 0.3877 | 0.3896 | PASS |
| continuous conditioning vs oracle adjacent lookup | bitstring tvd | 0.09375 | 0.3492 | 0.3635 | PASS |
| continuous conditioning vs oracle adjacent lookup | bitstring tvd | 0.15625 | 0.3185 | 0.3148 | FAIL |
| continuous conditioning vs oracle adjacent lookup | bitstring tvd | 0.21875 | 0.3328 | 0.3153 | FAIL |
| dense supervision vs endpoint-only | bitstring tvd | mean | 0.3474 | 0.3600 | PASS |
