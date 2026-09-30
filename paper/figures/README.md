# Manuscript figures

These four PDFs are the assets referenced by the author-supplied manuscript,
with the September 30 lattice revision applied. The complete-graph Curie-Weiss
benchmark remains in the paper; the exploratory lattice comparison is separate.

| Asset | Content |
| --- | --- |
| `F1.pdf` | Conditional IQP workflow, retained unchanged. |
| `gaussian_intro.pdf` | Open 4x4 lattice, 40 phases; seed 17100 profiles and ten-seed pointwise median/IQR over 80 held-out conditions. |
| `curie_phase_comparison_flip.pdf` | Retained 136-phase complete graph; encoder, shared, independent conditional and independent ordinary-input panels. Curves average profiles from three seeds after equal global-flip mixing. Numerical TVDs in the manuscript use raw laws. |
| `curie_weiss_ancilla_scaling.pdf` | Sixteen-visible-qubit lattice ancilla comparison with product phase input, three seeds and exact raw bitstring TVD; archived independent MMD evaluation at 64/256/1000 qubits uses the informed input. |

The Gaussian editable TikZ/pgfplots source and audited CSVs are in
`../../figure_drafts/gaussian_intro/`. The Curie-Weiss JSON sidecars preserve
per-seed profile/metric or scaling-curve inputs. `provenance.json` binds the PDFs
to checksums and frozen campaign identities. Full checkpoints remain local run
artifacts and are not included in this figure commit.

Mixed ancilla banks have 40+16r parameters, while visible-only controls retain
40. The ancilla panel shows the median/IQR across three seeds of raw bitstring
TVD first averaged across the three evaluation temperatures. Scaling curves
show the median/IQR across their three archived seeds; MMD values are not
comparable between sizes. The inline three-qubit symmetry circuit and bar chart
remain part of the manuscript LaTeX rather than a separate PDF asset.
