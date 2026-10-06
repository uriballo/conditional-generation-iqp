# Manuscript figures

These are the local manuscript PDF figures and PNG previews included in `main.tex`.
The current reporting revision prioritizes exact full-distribution TVD at
sixteen qubits and remains subject to author review.
The symmetry illustration remains inline in the manuscript.

| File | Source |
| --- | --- |
| `conditional_iqp_workflow.pdf` | Editable TikZ in `../figure_sources/conditional_iqp_overview.tex` |
| `gaussian_translation_profiles.pdf` | Selected profiles and per-seed full TVD in `../data/gaussian/`; generated `../tables/gaussian_tvd.tex`. Historical profile-only TikZ is in `../figure_sources/gaussian_intro.tex`. |
| `curie_weiss_model_comparison.pdf` | Five-seed profiles and full-TVD panels from `../data/curie-weiss/figure_data.json`; generated `../tables/curie_weiss_{full,profile}_tvd.tex` |
| `curie_weiss_lattice_scaling_mmd.pdf` | Six-size per-seed trajectories in `../data/scaling/figure_data.json` |
| `curie_weiss_symmetry_comparison.pdf` | Five paired seeds 6400--6404 in `../data/curie-weiss-symmetry/figure_data.json`; generated table fragments are `../tables/curie_weiss_symmetry_{main,settings,secondary,paired}.tex` |

Run `python run_experiments.py --mode figures` from the repository root to
regenerate numerical figures and TVD tables in the ignored results directory.
Use `--data-root results/paper/reported` to render completed reproduction runs.
Reviewed paper artifacts are never replaced automatically.

Gaussian accuracy curves show pointwise median full TVD and interquartile
ranges across ten seeds; its table takes the median of seed-wise means across
80 conditions. Curie--Weiss curves average raw profiles across seeds
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
including a distinct same-temperature spin-flip metric. Classical output averaging is included in the full-TVD figure panels and
reported in detail in the appendix. Reporting temperatures have historical exposure. Independent
checkpoint replay, selection reconstruction and primary scientific review
passed before these new artifacts were added. The claim-to-evidence map in
`../data/README.md` records the limits and review status.
