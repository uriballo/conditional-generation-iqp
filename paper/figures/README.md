# Manuscript figures

These are the local manuscript PDF figures and PNG previews included in `main.tex`.
The current reporting revision prioritizes exact full-distribution TVD at
sixteen qubits and remains subject to author review.
The symmetry illustration remains inline in the manuscript.

| File | Source |
| --- | --- |
| `conditional_iqp_workflow.pdf` | Editable TikZ in `../figure_sources/conditional_iqp_overview.tex` |
| `gaussian_translation_profiles.pdf` | Canonical Matplotlib profiles and response from `../data/gaussian/`; compact editable CSV exports are `midpoint_profiles.csv`, `response.csv`, and `training_response.csv`. An editable TikZ source remains in `../figure_sources/gaussian_intro.tex`. |
| `curie_weiss_model_comparison.pdf` | Five-seed profiles in the restored 2x2 model layout from `../data/curie-weiss/figure_data.json`; generated `../tables/curie_weiss_{full,profile}_tvd.tex` |
| `curie_weiss_lattice_scaling_mmd.pdf` | Six-size per-seed trajectories in `../data/scaling/figure_data.json` |

Run `python run_experiments.py --mode figures` from the repository root to
regenerate numerical figures and TVD tables in the ignored results directory.
Use `--data-root results/paper/reported` to render completed reproduction runs.
Reviewed paper artifacts are never replaced automatically.

`curie_weiss_physical_response.pdf` and its PNG preview show a single-column,
two-panel response: mean absolute magnetization above and per-spin heat-capacity
fluctuation estimate below. The renderer `paper_experiments/physical_response.py`
uses `../data/curie-weiss/response_sweep.json` and the independent fitted-temperature
records in `figure_data.json`. The 101-point beta 0 to 2 sweep reuses five selected
confirmation circuits without refitting or changing selection. Curves and bands
show mean and sample SD; independent markers and error bars use the same summaries
at their three fitted temperatures. Compute the nonlinear energy variance within
each run before aggregation. The target fluctuation expression equals equilibrium
heat capacity; the generated-distribution value does not verify the learned
family's temperature derivative. The figure was added at the author's request.

Gaussian profiles average five runs at four development midpoints. The model
response reports pointwise median expected Hamming weight with interquartile
range across five runs over 80 reporting conditions; its TVD table takes the
median of seed-wise means across 80 conditions. Curie--Weiss curves average raw profiles across seeds
5400--5404; full-TVD panels and table errors average per-seed exact TVDs and
report sample standard deviations. These are errors of individual circuit
distributions, not errors reconstructed from mean profiles. Shared temperatures are excluded from current
fitting and selection but have historical exposure; independent fits are
in-sample. Scaling retains the original protocol and averages independent MMD estimates
and summarizes the seed averages by their median and interquartile range. Its size-dependent kernels
prevent direct fidelity comparisons between sizes.

The input comparison uses separate matched 48-trial searches for the two
encoders and selects final settings using development-profile TVD. The reporting
revision does not retune or reselect models. Figure curves average exact raw profiles across the paired seeds.
Tables report means and sample standard deviations of individual-seed errors,
including a distinct same-temperature spin-flip metric. The input comparison is presented only in tables. Classical output averaging is included in the main numerical comparison; encoder-only controls and paired differences are in the appendix. Reporting temperatures have historical exposure. Independent
checkpoint replay, selection reconstruction and primary scientific review
passed before these new artifacts were added. The claim-to-evidence map in
`../data/README.md` records the limits and review status.
