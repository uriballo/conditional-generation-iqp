# Tractable Noncommuting and Block Encoding Benchmark

Matched seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ($N=10$). Lower is better.

| Encoding | Family | Held-Out Profile TVD | Held-Out Bitstring TVD | $\Delta$ Profile TVD vs Angle | Win Rate |
|---|---|---:|---:|---:|---:|
| Commuting angle ($L=1$) | Commuting ($L=1$) | 0.0619 [0.0596, 0.0646] | 0.3478 [0.3441, 0.3517] | Baseline | - |
| Calibrated noncommuting ($L=2$) | Noncommuting ($L=2$) | 0.0725 [0.0672, 0.0776] | 0.3743 [0.3575, 0.3887] | +0.0106 [+0.0038, +0.0174] | 2/10 |
| 2-Qubit block $ZZ(\pi/8)$ ($b=2$) | Entangling Block ($b=2$) | 0.0705 [0.0660, 0.0750] | 0.3709 [0.3594, 0.3826] | +0.0086 [+0.0031, +0.0146] | 1/10 |
| 2-Qubit block $ZZ(\pi/4)$ ($b=2$) | Entangling Block ($b=2$) | 0.0705 [0.0613, 0.0782] | 0.3985 [0.3930, 0.4041] | +0.0086 [-0.0012, +0.0171] | 2/10 |
