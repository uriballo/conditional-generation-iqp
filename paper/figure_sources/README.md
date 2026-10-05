# Editable illustrations

`conditional_iqp_overview.tex` is the standalone TikZ source for the reviewed
workflow figure. `gaussian_intro.tex` renders the Gaussian panels from the
compact CSV files in `../data/gaussian/`. The symmetry illustration is inline
in `../main.tex`. Numerical figures can also be rendered by the repository's
single reproduction command.

Build standalone sources from this directory with `pdflatex` and inspect the
output before replacing any reviewed figure. Generated PDFs and previews stay
in a temporary output directory:

Condition labels use concise decimals, retaining the precision needed to
identify each condition. Curie--Weiss legends use one decimal place; Gaussian
panel labels retain three to distinguish the displayed evaluation points.
Reported TVDs use four decimal places. Plot axes retain concise tick labels.
The inline symmetry plot uses DejaVu Serif for labels and ticks, matching the
numerical Curie--Weiss and scaling plots. Its font selection is scoped to the
plot; circuit and mathematical notation retain the manuscript math font.

```sh
mkdir -p /tmp/iqp-illustrations
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/iqp-illustrations conditional_iqp_overview.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/iqp-illustrations gaussian_intro.tex
```
