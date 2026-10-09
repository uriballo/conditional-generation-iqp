# Compact figure data

This directory contains the small numerical arrays used to regenerate the
paper figures without checkpoints, training pools, or host-specific paths.
Each JSON file records the source manifest hash and the aggregation convention
where available. Gaussian `midpoint_profiles.json` contains exact per-run profiles at the four development midpoints. Its generated CSV files support the editable figure source. `gaussian/results.json` retains exact profiles and full-distribution TVD for every run and reporting condition. Curie--Weiss data contain the
reviewed profile arrays and HPO per-seed metrics. Scaling data contain the
per-seed held-out MMD trajectories for the six reported system sizes.

The sixteen-qubit figures and main tables prioritize exact full-distribution
TVD. Hamming profiles and profile TVD remain secondary diagnostics, since shell
aggregation can hide errors within shells. The three Curie--Weiss model families retained in the manuscript use completed 60-trial full-TVD searches and five winning confirmation runs reused for reporting. The Gaussian data use the simpler pre-tuning configuration: five runs, 1,024 updates, 512 training targets per condition, minibatch 128, initialization width 0.25, reference draws 1,024, and expected mask weight one. Selection uses development-profile TVD, checked every 32 updates. Scaling MMD values
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
| Approved full-TVD HPO protocol | `../../results/curie-weiss-hpo-redesign-20261006/APPROVAL_SPEC.md`, `../../results/curie-weiss-full-tvd-hpo-20261006/launch/source-manifest.json` | Appendix `sec:cw_hpo`, `tab:cw_hpo_ranges`, `tab:training_hyperparams`, `tab:cw_baseline_budgets` | The retained studies completed. Reporting reuses five confirmation seeds, without an independent final cohort | Protocol authorized; all completed studies adopted; author review pending |
| Computational feasibility at 64--2048 qubits | `scaling/figure_data.json` | Scaling figure | Sampled MMD with size-dependent kernels; residual error persists | Previously reviewed; retained |
| Conditional signed-graph generation with exact switching symmetry | `switching16/response.npz`, `response-summary.json`, `reporting.npz`, `reporting-summary.json` and figure receipts | `fig:switching16-encoder`, `fig:switching16-response`, `tab:switching16`; appendix `app:switching16` and `fig:switching16-fluctuations` | Five independent final fits, 128 training bitstrings per temperature, two learned angles; moderate full TVD, post-selection dense sweep and substantial large-inverse-temperature fluctuation discrepancy; fixed classical switching control has lower TVD | Manuscript draft for author review; current data reproduced without new fitting |

The two-dimensional Gaussian section uses `gaussian2d/` and
`../tables/gaussian2d_tvd.tex`. The reviewed heatmaps compare separately and
jointly varied mean--width conditions against an encoder-only baseline on a
common withheld table domain. `gaussian2d/REVIEW.md` maps the claims to evidence
and records the limits of the adaptive study and width response.

The conditional switching section uses the frozen sixteen-qubit triangle study. The main figure shows triangle balance and fully balanced probability with an encoder-only baseline, fitting markers and a boundary at beta 1.2. The appendix records reference access, optimizer settings, selection, controls and energy fluctuations; detailed provenance remains in the research records. `switching16/README.md` documents deterministic figure regeneration from compact arrays. The separate parity methods draft has no fitted Curie--Weiss result and is outside the current manuscript. The newer HPO campaign remains outside this revision while the author reviews its results.
