# Editable illustrations

`conditional_iqp_overview.tex` is the standalone TikZ source for the reviewed
workflow figure. `gaussian_intro.tex` is an editable two-panel TikZ alternative
for Figure 2, rendered from compact CSV files in `../data/gaussian/`. The
canonical renderer is `paper_experiments/figures.py`; this source is intended
for standalone editing and inspection. The symmetry illustration is inline in
`../main.tex`. Numerical figures can also be rendered by the repository's
single reproduction command.

Build standalone sources from this directory with `pdflatex` and inspect the
output before replacing any reviewed figure. Generated PDFs and previews stay
in a temporary output directory:

Condition labels use concise decimals, retaining the precision needed to
identify each condition. Curie--Weiss legends use one decimal place; Gaussian
conditions are shown as `.125`, `.375`, `.625`, and `.875` in the shared
legend. The Gaussian alternative uses the renderer-generated
`midpoint_profiles.csv`, `response.csv`, and `training_response.csv` inputs;
it is no longer the earlier three-panel source.
Reported TVDs use four decimal places. Plot axes retain concise tick labels.
The inline symmetry plot uses DejaVu Serif for labels and ticks, matching the
numerical Curie--Weiss and scaling plots. Its font selection is scoped to the
plot; circuit and mathematical notation retain the manuscript math font.

```sh
mkdir -p /tmp/iqp-illustrations
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/iqp-illustrations conditional_iqp_overview.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/iqp-illustrations gaussian_intro.tex
```

The current two-dimensional Gaussian figures use `gaussian2d_phase.py` and
`../data/gaussian2d_phase/`. From the repository root, run
`.venv/bin/python paper/figure_sources/gaussian2d_phase.py` to replay into ignored
results. `gaussian2d_phase_controls.py` replays saved initial, selected and
terminal parameters and analytic controls without fitting; use `--tables`
to regenerate the two numerical tables. Both sources support explicit output
paths. `gaussian2d_phase_finite_validation.py` independently replays the finite-
validation follow-up and its direct calibration controls from compact saved
parameters, without fitting. The earlier `gaussian2d_heatmaps.py` source is
historical.
