# Compact figure data

This directory contains the small numerical arrays used to regenerate the
paper figures without checkpoints, training pools, or host-specific paths.
Each JSON file records the source manifest hash and the aggregation convention
where available. Gaussian `midpoint_profiles.json` contains exact per-run profiles at the four development midpoints. Its generated CSV files support the editable figure source. `gaussian/results.json` retains exact profiles and full-distribution TVD for every run and reporting condition. Curie--Weiss data contain the
reviewed profile arrays and HPO per-seed metrics. Scaling data contain the
per-seed held-out MMD trajectories for the six reported system sizes.

The sixteen-qubit figures and main tables prioritize exact full-distribution
TVD. Hamming profiles and profile TVD remain secondary diagnostics, since shell
aggregation can hide errors within shells. All sixteen-qubit Curie--Weiss comparisons now use completed 60-trial full-TVD searches and five winning confirmation runs reused for reporting. The Gaussian data use the simpler pre-tuning configuration: five runs, 1,024 updates, 512 training targets per condition, minibatch 128, initialization width 0.25, reference draws 1,024, and expected mask weight one. Selection uses development-profile TVD, checked every 32 updates. Scaling MMD values
use size-dependent kernels and should not be compared as absolute errors across
system sizes. The compact files are provenance-bearing render inputs, not a
replacement for the full training archives.

## Claims and evidence

| Manuscript result | Compact evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Gaussian translated profiles, conditional response and full-distribution accuracy | `gaussian/results.json`, `gaussian/midpoint_profiles.json` and generated CSVs | Gaussian translation figure; `tab:gaussian_tvd` | Mean profiles at development midpoints used for selection; response uses pointwise median of model mean Hamming weights on 80 separate reporting conditions; TVD table uses median of run-wise mean errors | Author restored the pre-tuning settings and five-run results; manual study archived |
| Shared and independently fitted Curie--Weiss full-distribution comparison | `curie-weiss/figure_data.json` | Comparison figure; `tab:curie_weiss_full_tvd`; appendix `tab:curie_weiss_profile_tvd` | Five paired seeds; independent temperatures are fitted; shared reporting temperatures have historical exposure; unequal aggregate budgets; selection uses full TVD and reporting reuses confirmation seeds; uniform encoder outperforms all trained families at inverse temperature 0.4 | Completed full-TVD HPO fits adopted and checkpoint metrics verified; author review pending |
| Shared Curie--Weiss magnetization levels off at high inverse temperature; energy fluctuations miss the target's heat-capacity peak and decline | `curie-weiss/response_sweep.json`, `response_sweep_validation.json`, independent records in `figure_data.json`; renderer `paper_experiments/physical_response.py` | `fig:cw_physical_response`; appendix `sec:cw_physical_observables` | Exact frozen-model sweep over 101 points, beta 0 to 2; five selected confirmation seeds reused; includes fitting/development conditions and extrapolation; independent markers are in-sample; energy variance is not a verified temperature derivative for learned distributions | Author requested manuscript inclusion; primary numerical and figure review passed |
| HPO-selected optimizer settings | `curie-weiss/hpo-summary.json` and per-run records in `figure_data.json` | Appendix tables | Development/fitted-temperature selection; no general optimizer superiority claim | Completed winner configurations adopted; author review pending |
| Approved full-TVD HPO protocol | `../../results/curie-weiss-hpo-redesign-20261006/APPROVAL_SPEC.md`, `../../results/curie-weiss-full-tvd-hpo-20261006/launch/source-manifest.json` | Appendix `sec:cw_hpo`, `tab:cw_hpo_ranges`, `tab:training_hyperparams`, `tab:cw_baseline_budgets` | All four studies completed. Reporting reuses five confirmation seeds, without an independent final cohort | Protocol authorized; all completed studies adopted; author review pending |
| Computational feasibility at 64--2048 qubits | `scaling/figure_data.json` | Scaling figure | Sampled MMD with size-dependent kernels; residual error persists | Previously reviewed; retained |
| Input-induced bit-flip symmetry | `symmetry/symmetry.json`, analytic manuscript derivation | Input-symmetry illustration | Fixed three-qubit illustration, no fitting | Previously reviewed; retained |
| Curie--Weiss input symmetry comparison | `curie-weiss-symmetry/figure_data.json`, `provenance.json`; `../../results/curie-weiss-full-tvd-hpo-20261006/shared-reporting-update.json` | `tab:cw_symmetry_metrics`, encoder-only appendix paragraph, `tab:cw_symmetry_paired` | Exact TVDs, five paired confirmation seeds reused for reporting; separate matched 60-trial full-TVD searches. Projection improves mean full TVD at inverse temperatures 0.4 and 1.0 but worsens it at 1.6. Reporting temperatures have historical exposure; confirmation reuse limits independent selection assessment | Checkpoint evaluation and primary scientific review passed; author review pending |

The input comparison reports exact raw profiles and means with sample standard
deviations of per-seed errors. Full TVD precedes profile TVD in all result tables;
confirmation scores for the current encoder comparison use median development full TVD across seeds. Same-temperature spin-flip error is distinct
from the paired-condition complement diagnostic in the original experiment.
`curie-weiss-symmetry/protocol.json`, `source-manifest.json` and `provenance.json`
retain historical fitting records. Current winner settings, checkpoint hashes and result provenance are recorded in `provenance.json` and the shared-reporting artifact.
Fitting and current export-source hashes are recorded separately.
`curie-weiss-symmetry/publication.json` preserves the hashes of the previous
integrated draft. `full-tvd-reporting.json` records the current reporting
revision, source and artifact hashes, and validation without changing the
frozen fitting or selection records.

The superseded comparison used a different five-seed cohort; its artifacts are retained in the ignored historical backup. Current Curie--Weiss reporting uses the winning five-seed confirmation cohorts.
The archived full research workspace retains historical analyses and pools.

Table III reports only the two trained input models. Classical output averaging is retained in historical numerical records but appears in the manuscript only as a possible postprocessing improvement.

The author-approved academic writing revision separates classical estimation from optimization difficulty, clarifies the spin-flip commutation statement, and adds explicit figure/table references. The paired symmetry table now contains only full-distribution TVD differences. Numerical results and fitting protocols are unchanged. The revision record is in `../../results/paper-writing-edit-20261007/APPLIED.md`.
