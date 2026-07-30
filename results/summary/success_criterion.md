# Preregistered success-criterion audit

**Overall result: FAIL.**

Angle/complete must have lower median shell TVD and bitstring TVD than both conditional baselines for A and B separately.

| Condition | Metric | Angle/complete | No condition | Basis condition | Result |
|---:|---|---:|---:|---:|---:|
| A | shell tvd | 0.1210 | 0.4032 | 0.0066 | FAIL |
| A | bitstring tvd | 0.3516 | 0.4969 | 0.4054 | PASS |
| B | shell tvd | 0.0718 | 0.3273 | 0.7221 | PASS |
| B | bitstring tvd | 0.2746 | 0.4618 | 0.7777 | PASS |

Balanced averages are descriptive diagnostics, not substitutes for the condition-wise criterion.

| Model | Mean-condition shell TVD | Mean-condition bitstring TVD |
|---|---:|---:|
| Angle / complete | 0.0964 | 0.3140 |
| No condition | 0.3651 | 0.4815 |
| Basis condition | 0.3644 | 0.5916 |
