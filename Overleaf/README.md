# Canonical Overleaf manuscript

Upload the contents of this folder to a new Overleaf project and compile
`main.tex` with pdfLaTeX (the `revtex4-2` class). The bibliography file and
figure assets are included locally.

## Inventory

- `main.tex`: canonical manuscript source, with only compile-safety repairs.
- `references.bib`: bibliography database.
- `figures/all_datasets.pdf`
- `figures/gaussian_results.pdf`
- `figures/curie_weiss_results.pdf`: neutral placeholder; replace with the
  campaign figure when available.
- `figures/curie_weiss_results_placeholder.tex`: editable source for that
  placeholder.

Generated campaign files referenced conditionally by the source are optional;
the manuscript supplies pending-value fallbacks when they are absent.

## Source-formatting convention

Keep each prose paragraph on one uninterrupted source line, with a blank line only between paragraphs. Displayed equations and structured environments may use multiple source lines.
