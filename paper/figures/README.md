# Manuscript figures

These are the reviewed PDF figures and PNG previews included in `main.tex`.
The symmetry illustration remains inline in the manuscript.

| File | Source |
| --- | --- |
| `conditional_iqp_workflow.pdf` | Editable TikZ in `../figure_sources/conditional_iqp_overview.tex` |
| `gaussian_translation_profiles.pdf` | Selected profiles in `../data/gaussian/`; editable TikZ in `../figure_sources/gaussian_intro.tex` |
| `curie_weiss_model_comparison.pdf` | Five-seed profiles and metrics in `../data/curie-weiss/figure_data.json` |
| `curie_weiss_lattice_scaling_mmd.pdf` | Six-size per-seed trajectories in `../data/scaling/figure_data.json` |

Run `python run_experiments.py --mode figures` from the repository root to
regenerate numerical figures and TVD tables in the ignored results directory.
Use `--data-root results/paper/reported` to render completed reproduction runs.
Reviewed paper artifacts are never replaced automatically.

Curie--Weiss curves average raw profiles across seeds 5400--5404; table errors
average per-seed exact TVDs. Shared temperatures are excluded from current
fitting and selection but have historical exposure; independent fits are
in-sample. Scaling retains the original protocol and averages independent MMD estimates
and summarizes the seed averages by their median and interquartile range. Its size-dependent kernels
prevent direct fidelity comparisons between sizes.
