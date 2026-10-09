# Manuscript figures

The PDF figures and PNG previews here are reviewed publication artifacts.
Regeneration writes to ignored results by default and never replaces these
files automatically.

| Manuscript figure | Source |
| --- | --- |
| `conditional_iqp_workflow.pdf` | `../figure_sources/conditional_iqp_overview.tex` |
| `gaussian_translation_profiles.pdf` | `../data/gaussian/` and `paper_experiments/figures.py`; editable TikZ source `../figure_sources/gaussian_intro.tex` |
| `gaussian2d_phase_heatmaps.pdf`, mean and width responses | `../data/gaussian2d_phase/` and `../figure_sources/gaussian2d_phase.py` |
| `curie_weiss_lattice_scaling_mmd.pdf` | Original product-input trajectories in `../data/scaling/figure_data.json` |
| `conditional_symmetry_current.pdf`, `conditional_symmetry_fluctuation.pdf` | `../data/switching16/` and its `render.py` |
| `switching16_graph.pdf`, `triangle_motif_draft.pdf` | `../data/switching16/draw_encoders.py` |

From the repository root, run `uv run python run_experiments.py --mode figures`
to replay the current numerical figures to `results/paper/figures/`. The Gaussian
controls and finite-validation tables have separate replay commands documented
in `../data/gaussian2d_phase/README.md`.

Gaussian translation profiles average five fits at four development conditions.
The response uses the pointwise median and interquartile range on eighty held-out
conditions; TVD takes the median of seed-wise condition averages. Phase-conditioned
heatmaps average five selected fits per design and include fitting and development
points. Mean and width curves use median and interquartile range. Scaling uses
median and interquartile range across seed-level bank averages, retaining negative
unbiased MMD estimates. Switching curves use means and sample standard deviations
of five per-fit observables, calculating energy variance before aggregation.

The current parity-input Curie–Weiss comparison is pending. Earlier
`curie_weiss_model_comparison.*` and `curie_weiss_physical_response.*` assets
remain historical evidence for the transitional appendix. Replay them only with
`--experiment curie-weiss-historical`. Their confirmation seeds were reused for
reporting; they are not the new comparison's final fits.
