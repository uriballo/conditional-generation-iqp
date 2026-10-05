# Compact figure data

This directory contains the small numerical arrays used to regenerate the
paper figures without checkpoints, training pools, or host-specific paths.
Each JSON file records the source manifest hash and the aggregation convention
where available. Gaussian CSV files contain selected target/model profiles and
the corresponding profile-TVD diagnostics. Curie--Weiss data contain the
reviewed profile arrays and HPO per-seed metrics. Scaling data contain the
per-seed held-out MMD trajectories for the six reported system sizes.

The figures summarize finite numerical experiments; profile TVD is a shell
profile metric and does not imply full bitstring fidelity. Scaling MMD values
use size-dependent kernels and should not be compared as absolute errors across
system sizes. The compact files are provenance-bearing render inputs, not a
replacement for the full training archives.

## Claims and evidence

| Manuscript result | Compact evidence | Figure/table | Limitation |
| --- | --- | --- | --- |
| Gaussian translated profiles and aggregate TVD | `gaussian/results.json`, `gaussian/profile_*.csv` | Figure 2 | Exact shell-profile errors; no full-distribution claim |
| Shared and independently fitted Curie--Weiss comparison | `curie-weiss/figure_data.json` | Figure 3, Table 1 | Five paired seeds; independent temperatures are fitted; shared reporting temperatures have historical exposure |
| HPO-selected optimizer settings | `curie-weiss/hpo-summary.json` and per-run records in `figure_data.json` | Appendix tables | Development/fitted-temperature selection; no general optimizer superiority claim |
| Computational feasibility at 64--2048 qubits | `scaling/figure_data.json` | Figure 4 | Sampled MMD with size-dependent kernels; residual error persists |
| Input-induced bit-flip symmetry | `symmetry/symmetry.json`, analytic manuscript derivation | Figure 5 | Fixed three-qubit illustration, no fitting |

These numerical results were reviewed for the first draft. The current
comparison uses seeds 5400--5404, rather than the superseded three-seed fits.
The archived full research workspace retains historical analyses and pools.
