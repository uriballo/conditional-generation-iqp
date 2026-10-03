# Manuscript figures

This directory contains only the five figures included in the handcrafted manuscript. PNGs are previews of the same figures. All numerical fit plots and metrics use raw distributions without probability postprocessing.

| PDF | Content |
| --- | --- |
| `conditional_iqp_workflow.pdf` | Author-approved classical training and conditional quantum generation workflow; editable TikZ source and PNG preview included. |
| `gaussian_translation_profiles.pdf` | Sixteen-qubit Gaussian translation with an open 4 by 4 lattice and 40 phases. |
| `curie_weiss_model_comparison.pdf` | Raw encoder-only, shared conditional, independent conditional, and independent ordinary-input profiles. New 2,048-mask/2,048-reference budget and expected weights 2, 4, 6. |
| `curie_weiss_ancilla_tvd.pdf` | Exact raw Hamming-profile TVD with sixteen visible lattice qubits and mixed visible-ancilla coupling; condition mean over beta 0.4, 1.0, 1.6 within seed, then seed median/IQR. Revised at author request; primary numerical and visual review passed. |
| `curie_weiss_lattice_scaling_mmd.pdf` | Held-out sampled MMD trajectories for open lattices from 64 to 2048 qubits. Prior accepted training protocol retained until replacement results are reviewed. |

The symmetry illustration remains inline in the manuscript LaTeX. It describes actual output symmetry rather than postprocessing. Shared Curie--Weiss predictions are held out; independent comparisons are in-sample.

The [filename migration and checksums](../../figure_drafts/paper_figures/provenance.json) record the reviewed PDF/PNG bytes and the approved workflow replacement. The [LaTeX filename list](../../figure_drafts/paper_figures/figure_paths.tex) supports manual manuscript integration. The [workflow editable source and build instructions](../../figure_drafts/conditional_iqp_overview/README.md) accompany the approved workflow PDF and preview. The Gaussian editable source is in `../../figure_drafts/gaussian_intro/`, the raw comparison export in `../../figure_drafts/curie_weiss_balanced/`, and the ancillary/scaling export in `../../figure_drafts/phase_lattice_scaling/`. Optional training/condition diagnostics are kept with that export; the old combined panel is in `../../figure_drafts/archived_ancilla_scaling/`.

The current workflow revision incorporates adversarial-review clarifications: the reference state is explicit, classical phase evaluation is distinct from the complete output state, squared MMD losses are matched by condition before averaging, and the generation circuit is schematic on n qubits. The author retained the simpler Condition c_i=c_2 and Samples for chosen c_i output wording.

The author-approved layout refinement aligns the encoder columns and outer left edges, places generation headings on a common baseline, matches MMD/data card heights, and wraps Conditional samples onto two lines. The scientific labels and circuit gate spacing are retained.
