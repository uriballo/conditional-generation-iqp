# Manuscript figures

This directory contains only the five figures included in the handcrafted manuscript. PNGs are previews of the same figures. All numerical fit plots and metrics use raw distributions without probability postprocessing.

| PDF | Content |
| --- | --- |
| `conditional_iqp_workflow.pdf` | Conditional input encoder, shared IQP unitary, and training workflow. |
| `gaussian_translation_profiles.pdf` | Sixteen-qubit Gaussian translation with an open 4 by 4 lattice and 40 phases. |
| `curie_weiss_model_comparison.pdf` | Raw encoder-only, shared conditional, independent conditional, and independent ordinary-input profiles. New 2,048-mask/2,048-reference budget and expected weights 2, 4, 6. |
| `curie_weiss_ancilla_tvd.pdf` | Exact raw visible-bitstring TVD with sixteen visible lattice qubits and mixed visible-ancilla coupling. Prior accepted training protocol retained until replacement results are reviewed. |
| `curie_weiss_lattice_scaling_mmd.pdf` | Held-out sampled MMD trajectories for open lattices from 64 to 2048 qubits. Prior accepted training protocol retained until replacement results are reviewed. |

The symmetry illustration remains inline in the manuscript LaTeX. It describes actual output symmetry rather than postprocessing. Shared Curie--Weiss predictions are held out; independent comparisons are in-sample.

The [filename migration and checksums](../../figure_drafts/paper_figures/provenance.json) bind the renamed assets to the same reviewed PDF/PNG bytes. The [LaTeX filename list](../../figure_drafts/paper_figures/figure_paths.tex) supports manual manuscript integration. The Gaussian editable source is in `../../figure_drafts/gaussian_intro/`, the raw comparison export in `../../figure_drafts/curie_weiss_balanced/`, and the ancillary/scaling export in `../../figure_drafts/phase_lattice_scaling/`. Optional training/condition diagnostics are kept with that export; the old combined panel is in `../../figure_drafts/archived_ancilla_scaling/`.
