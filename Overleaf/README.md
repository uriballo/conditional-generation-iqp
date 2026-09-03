# Canonical Overleaf manuscript

Upload the contents of this folder to a new Overleaf project and compile
`main.tex` with pdfLaTeX (the `revtex4-2` class). The bibliography file and
figure assets are included locally.

## Inventory

- `main.tex`: canonical manuscript source.
- `references.bib`: bibliography database.
- `figures/gaussian_results.pdf`
- `figures/gaussian_n8_training_profiles.pdf`
- `figures/gaussian_n12_training_profiles.pdf`
- `figures/gaussian_n8_generalization_profiles.pdf`
- `figures/gaussian_n12_generalization_profiles.pdf`
- `figures/curie_weiss_n8_magnetization_profiles.pdf`
- `figures/curie_weiss_n8_generalization_profiles.pdf`
- `figures/curie_weiss_n12_magnetization_profiles.pdf`
- `figures/curie_weiss_n12_generalization_profiles.pdf`
- `figures/curie_encoder_ablation_n8.pdf`

Generated campaign files referenced conditionally by the source are optional;
the manuscript supplies pending-value fallbacks when they are absent.

## Source-formatting convention

Keep each prose paragraph on one uninterrupted source line, with a blank line only between paragraphs. Displayed equations and structured environments may use multiple source lines.
