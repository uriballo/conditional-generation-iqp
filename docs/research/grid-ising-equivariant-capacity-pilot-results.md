# Grid-Ising equivariant capacity pilot results

This four-fit pilot tested a two-coordinate, globally spin-flip-equivariant reference state on the two failure conditions from the standard-product capacity screen. The reference uses a smooth temperature-dependent branch separation with no encoded critical point. Field reversal swaps the two globally flipped branch weights, which guarantees (E(\beta,-h)=X^{\otimes n}E(\beta,h)). The construction contains no lattice-edge information.

## Exact-TVD results

| Condition | Product baseline, \(r=0\) | Product baseline, \(r=3\) | Equivariant, \(r=0\) | Equivariant, \(r=3\) |
|---|---:|---:|---:|---:|
| \((\beta,h)=(0.65,0)\) | 0.4075 | 0.4075 | 0.3553 | 0.3540 |
| \((\beta,h)=(0.44,0.05)\) | 0.2032 | 0.2037 | 0.3772 | 0.3655 |

The pilot failed every substantive performance gate. The ordered TVD remained above 0.20, the field condition regressed by more than 0.02, and the ordered (r=3) advantage over (r=0) was only 0.0013 rather than 0.02. The ordered run did learn relative to its data-initialized checkpoint, but its selected output was numerically indistinguishable from the encoder-only zero-phase distribution. The extra ancilla-supported parameters moved by an (L_2) norm of only 0.0049.

The initial gradient diagnostic rules out a simple vanishing-gradient explanation. At the ordered condition, the mean ancilla-gradient norm was 0.0279, approximately 53% of the visible-gradient norm. At the field condition it was 0.0313, approximately 43% of the visible-gradient norm. The optimizer initially sees the ancilla directions, but the selected solutions do not retain them.

## Decision

The grid-Ising branch should stop here. The standard product reference and a symmetry-equivariant two-branch reference both fail at the target-aware capacity level, and the second construction damages the field-conditioned target. Further encoder search or broad hyperparameter optimization would be difficult to justify before the Gaussian and Curie--Weiss campaigns are complete.

The next experimental priority is therefore the remaining frozen campaign: finish the final 14 Curie--Weiss (n=8) identities, then run Gaussian (n=16), Curie--Weiss (n=12), and Curie--Weiss (n=16) as separate atomic waves.

## Reproducibility

- Manifest: `configs/grid_ising_equivariant_capacity_pilot_v1.json`
- Runner: `scripts/run_grid_ising_equivariant_capacity_pilot.py`
- Immutable summary: `results/grid-ising-equivariant-capacity-pilot-v1/summary.json`
- Checkpoints, exact probabilities, gradients, and per-run records: `results/grid-ising-equivariant-capacity-pilot-v1/`
