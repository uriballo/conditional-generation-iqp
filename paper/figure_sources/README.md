# Editable illustrations and numerical replay

`conditional_iqp_overview.tex` is the standalone TikZ source for the workflow
figure. `gaussian_intro.tex` is the editable Gaussian translation figure using
compact CSV inputs from `../data/gaussian/`. Its condition labels retain the
precision needed to identify the four development midpoints.

Build standalone illustrations into a temporary directory and inspect them
before replacing reviewed figures:

```sh
mkdir -p /tmp/iqp-illustrations
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/iqp-illustrations conditional_iqp_overview.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/iqp-illustrations gaussian_intro.tex
```

`gaussian2d_phase.py` replays the historical phase-conditioned mean–width heatmaps
and response curves from `../data/gaussian2d_phase/`. The repository figure
command includes these plots. `gaussian2d_phase_controls.py` replays saved angles
and analytic controls; `gaussian2d_phase_finite_validation.py` replays the finite-
validation fits and direct calibration controls. These scripts perform no fitting.
Their data README documents output paths and table regeneration.

Signed-graph response and encoder illustration sources are in
`../data/switching16/`. The older amplitude-conditioned two-dimensional Gaussian
and three-qubit symmetry experiment have been removed from the active tree.

`gaussian_lattice.py` renders the current binomial-centered family from
`../data/gaussian_lattice/`. The main two-coordinate figure shows axis training
alone, with training conditions, their convex hull, and the fixed origin marked.
The heatmap is the unsmoothed mean full-distribution TVD across five selected
fits on the existing 41-by-41 diagnostic grid. It includes training and
development conditions; the sixteen mixed reporting pairs are separate from both.
The grid-training comparison table and response curves remain appendix context.
The renderer changes no fit, selection rule, or reviewed numerical evidence.
