# Manuscript figures

This directory is the canonical home for final manuscript figures. Each figure
is tracked as a vector PDF for LaTeX and as a 300 dpi PNG for quick review.

| Stem | Role | Provenance |
| --- | --- | --- |
| `fig1_pipeline_targets` | Conditional-generation pipeline and target families | Imported byte-for-byte from the read-only Overleaf snapshot on 2026-08-27 |
| `all_datasets` | Overview of the Gaussian, Curie--Weiss, and MFC datasets | Imported byte-for-byte from the read-only Overleaf snapshot on 2026-08-27 |
| `gaussian_results` | Main Gaussian confirmatory result | Imported byte-for-byte from the read-only Overleaf snapshot on 2026-08-27 |
| `curie_weiss_confirmatory_main` | Main Curie--Weiss confirmatory result | Generated from the complete 390-run campaign |

Regenerate the Curie--Weiss figure from the two reviewed result directories:

```bash
uv run python scripts/plot_curie_weiss_confirmatory.py
```

The plotting command verifies the immutable envelopes, rejects duplicate run
coordinates, and requires exact coverage of the frozen campaign manifest before
writing either output file.
