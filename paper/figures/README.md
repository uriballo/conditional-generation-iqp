# Manuscript figures

These five PDFs are the current assets referenced by the handcrafted author draft. The complete-graph Curie-Weiss benchmark remains in the paper. The Gaussian uses its updated lattice; the ancilla and scaling panels are now separate and use the general product phase encoder.

| Asset | Content |
| --- | --- |
| `F1.pdf` | Conditional IQP workflow, retained unchanged. |
| `gaussian_intro.pdf` | Open 4 by 4 lattice, 40 phases; seed 17100 profiles at three held-out conditions, with no TVD panel. |
| `curie_phase_comparison_flip.pdf` | Retained 136-phase complete graph; encoder, shared, independent conditional and independent ordinary-input panels. Curves average profiles from three seeds after equal global-flip mixing. Manuscript TVDs use raw laws. |
| `curie_weiss_ancilla.pdf` | Sixteen visible qubits, mixed visible-ancilla phases, general product phase input on every wire; three seeds, exact raw bitstring TVD after ancillary marginalization. Only the mixed coupling curve is shown, with the visible-qubit count annotated inside the top right. |
| `curie_weiss_scaling.pdf` | General product phase input, open lattices, no ancillas; independent held-out MMD trajectories for 64, 128, 256, 512, 1024, 2048 qubits, three seeds per size, without a panel title. |
| `curie_weiss_scaling_train.pdf` | Optional training-grid counterpart with the same six sizes, seed median/IQR design, separate temperature-grid averages, automatic y-range, and no panel title. |

The former `curie_weiss_ancilla_scaling.pdf` and JSON sidecar remain archived. They are superseded by the two separate plots and their associated export; their informed-input scaling results are not used in the current manuscript.

Editable Gaussian TikZ/pgfplots source and audited CSVs are in `../../figure_drafts/gaussian_intro/`. The separate ancilla/scaling export, standalone renderer, per-condition diagnostics and train/held-out comparison are in `../../figure_drafts/phase_lattice_scaling/`. Curie-Weiss JSON sidecars preserve per-seed profile inputs. `provenance.json` binds publication PDFs to checksums and campaign identities. Full checkpoints remain local run artifacts.

Mixed ancilla banks have 40+16r phases. The ancilla plot shows the median/IQR across seeds 410, 411, 412 of exact raw bitstring TVD first averaged across the three evaluation temperatures. Scaling plots show the median/IQR across seeds 100, 101, 102 after averaging five independent evaluation banks within each seed. MMD values are not comparable between sizes; high-beta discrepancies remain. The inline three-qubit symmetry circuit and bar chart stay in the manuscript LaTeX.
