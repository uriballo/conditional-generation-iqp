# Grid-Ising capacity pilot results

The frozen eight-fit pilot tested independent complete two-local IQP circuits on a periodic (2\times4) Ising lattice. It used four target conditions, one development seed, and (r\in\{0,3\}) ancillas. These results diagnose target-fitting capacity and optimization; they are not evidence for conditional generalization.

## Exact-TVD results

| \(\beta\) | \(h\) | TVD, \(r=0\) | TVD, \(r=3\) | \(r=0-r=3\) |
|---:|---:|---:|---:|---:|
| 0.20 | 0.00 | 0.1264 | 0.1148 | +0.0116 |
| 0.44 | 0.00 | 0.1858 | 0.1940 | -0.0082 |
| 0.65 | 0.00 | 0.4075 | 0.4075 | 0.0000 |
| 0.44 | 0.05 | 0.2032 | 0.2037 | -0.0005 |
| **Median** |  | **0.1945** | **0.1989** | **-0.0044** |

The pilot failed its preregistered gate. At (r=3), the median TVD exceeded 0.15 and the maximum exceeded 0.25. The ancilla non-regression check passed, but ancillas did not improve the median.

## Failure mode

The disordered condition is learnable, and three ancillas improve it modestly. Performance degrades near the transition and under a small field. Both models fail sharply in the ordered regime at \((\beta,h)=(0.65,0)\), where their TVD is approximately 0.408 and their absolute-magnetization expectation is too low by approximately 0.244.

This is not primarily a shortage of updates. For the ordered target, validation MMD falls during the first 50 updates and then plateaus through update 500. The selected (r=3) ancilla parameters move by an \(L_2\) norm of only 0.0074 from initialization, whereas they move by 0.99 for the learnable disordered condition. The optimizer therefore does not recruit the extra degrees of freedom in the difficult ordered regime.

The result does not invalidate the grid-Ising benchmark. It rules out proceeding directly to a conditional campaign with the standard product input and present initialization. A subsequent pilot should change one mechanism at a time, with the most informative candidates being a symmetry-preserving two-branch input or an initialization that gives ancilla-supported phases a usable gradient scale.

## Reproducibility

- Manifest: `configs/grid_ising_capacity_pilot_v1.json`
- Runner: `scripts/run_grid_ising_capacity_pilot.py`
- Summary: `results/grid-ising-capacity-pilot-v1/summary.json`
- Exact probabilities and immutable per-run records: `results/grid-ising-capacity-pilot-v1/`
