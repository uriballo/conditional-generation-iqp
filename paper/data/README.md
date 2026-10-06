# Compact figure data

This directory contains the small numerical arrays used to regenerate the
paper figures without checkpoints, training pools, or host-specific paths.
Each JSON file records the source manifest hash and the aggregation convention
where available. Gaussian CSV files contain selected target/model profiles and
the corresponding profile-TVD diagnostics; `gaussian/results.json` also retains exact full-distribution TVD for every seed and reporting condition. Curie--Weiss data contain the
reviewed profile arrays and HPO per-seed metrics. Scaling data contain the
per-seed held-out MMD trajectories for the six reported system sizes.

The sixteen-qubit figures and main tables prioritize exact full-distribution
TVD. Hamming profiles and profile TVD remain secondary diagnostics, since shell
aggregation can hide errors within shells. Existing models were tuned and
selected using profile TVD; this reporting revision does not change fitting,
selection, seeds, or checkpoints. Scaling MMD values
use size-dependent kernels and should not be compared as absolute errors across
system sizes. The compact files are provenance-bearing render inputs, not a
replacement for the full training archives.

## Claims and evidence

| Manuscript result | Compact evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Gaussian translated profiles and full-distribution accuracy | `gaussian/results.json`, `gaussian/profile_*.csv` | Gaussian translation figure; `tab:gaussian_tvd` | Exact full and profile errors on 80 held-out conditions; median across ten seed-wise means differs from pointwise plotted medians; checkpoint selection remains development-profile TVD | Existing fits retained; full-TVD reporting revision checked locally; author review pending |
| Shared and independently fitted Curie--Weiss full-distribution comparison | `curie-weiss/figure_data.json` | Comparison figure; `tab:curie_weiss_full_tvd`; appendix `tab:curie_weiss_profile_tvd` | Five paired seeds; independent temperatures are fitted; shared reporting temperatures have historical exposure; unequal aggregate budgets; selection remains profile TVD; uniform encoder outperforms all trained families at inverse temperature 0.4 | Existing fits retained; full-TVD reporting revision checked locally; author review pending |
| HPO-selected optimizer settings | `curie-weiss/hpo-summary.json` and per-run records in `figure_data.json` | Appendix tables | Development/fitted-temperature selection; no general optimizer superiority claim | Previously reviewed; retained |
| Approved full-TVD HPO protocol | `../../results/curie-weiss-hpo-redesign-20261006/APPROVAL_SPEC.md`, `../../results/curie-weiss-full-tvd-hpo-20261006/launch/source-manifest.json` | Appendix `sec:cw_full_tvd_hpo`, `tab:cw_full_tvd_hpo_ranges`, `tab:cw_full_tvd_sample_budgets` | New campaign; existing numerical tables still describe profile-selected fits. Reporting will reuse five confirmation seeds, without an independent final cohort | Protocol authorized; appendix checked locally; full comparison pending |
| Computational feasibility at 64--2048 qubits | `scaling/figure_data.json` | Scaling figure | Sampled MMD with size-dependent kernels; residual error persists | Previously reviewed; retained |
| Input-induced bit-flip symmetry | `symmetry/symmetry.json`, analytic manuscript derivation | Input-symmetry illustration | Fixed three-qubit illustration, no fitting | Previously reviewed; retained |
| Curie--Weiss input symmetry comparison | `curie-weiss-symmetry/{figure_data.json,winner.json,audit.json,additional-audit.json,hpo-audit.json}` | `curie_weiss_symmetry_comparison.pdf`; `tab:cw_symmetry_metrics`, `tab:cw_symmetry_settings`, `tab:cw_symmetry_secondary`, `tab:cw_symmetry_paired` | Five paired seeds 6400--6404; separate matched 48-trial searches; final selection uses development-profile TVD. Mean raw full TVD improves at all reporting temperatures; profile differences at inverse temperatures 1.0 and 1.6 are small. Classical averaging of the product output has lower profile and full TVD at inverse temperature 0.4. Reporting temperatures have historical exposure. | Independent replay and primary scientific review passed; author review pending |

The input comparison reports exact raw profiles and means with sample standard
deviations of per-seed errors. Full TVD precedes profile TVD in all result tables;
confirmation scores remain development-profile TVD. Same-temperature spin-flip error is distinct
from the paired-condition complement diagnostic in the original experiment.
`curie-weiss-symmetry/protocol.json`, `source-manifest.json` and `provenance.json`
freeze the fitting protocol, sources, dependencies and complete budgets.
Fitting and current export-source hashes are recorded separately.
`curie-weiss-symmetry/publication.json` preserves the hashes of the previous
integrated draft. `full-tvd-reporting.json` records the current reporting
revision, source and artifact hashes, and validation without changing the
frozen fitting or selection records.

The original comparison uses seeds 5400--5404, rather than the superseded
three-seed fits.
The archived full research workspace retains historical analyses and pools.
