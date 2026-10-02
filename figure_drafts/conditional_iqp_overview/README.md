# Conditional IQP workflow

This author-approved revised workflow is the publication figure for 2026-10-02. The canonical PDF is `../../paper/figures/conditional_iqp_workflow.pdf`, with the matching PNG preview. The editable source remains `conditional_iqp_overview.tex`. The handcrafted manuscript is integrated manually; this publication updates the figure assets and provenance.

The author approved the adversarial-review revision while retaining the simpler previous output wording: Condition c_i=c_2 and Samples for chosen c_i. The output bitstrings stay unboxed and aligned near the wire ends. Each is a complete independent measurement outcome from the selected condition. The three drawn wires represent an n-qubit register schematically, as stated in the circuit title.

The fixed encoder varies with the condition. Classical estimation samples the reference state H^{tensor n}|phi(c_i)> and evaluates its amplitude ratios. The upper blue block evaluates the diagonal IQP phases and their derivatives; it does not depict construction of the full output state. Combining these operations estimates conditional Pauli-Z moments and their parameter gradients. Target and model statistics are matched at each condition before averaging squared MMD losses. The optimizer updates one shared phase vector, and the lower handoff reuses it for generation.

The lower encoder E(c_i), IQP unitary H D_G(theta) H, and computational-basis measurement are shown inside one Conditional IQP frame. Only the learned block must have IQP form. Gates have matching edge-to-edge gaps and large labels. Condition choices are vertically stacked, with c_2 highlighted. All bitstrings, wires and computer icons are schematic. The diagram describes the generation pathway and applies no outcome filtering or probability postprocessing.

All artwork is editable TikZ with Computer Modern typography and the paper palette. Earlier variants drew on the training/deployment organization of Figure 1 of Recio-Armengol, Ahmed and Bowles, arXiv:2503.02934v2. No image assets were copied.

## Proposed caption

Conditional IQP workflow. Classical training combines sampling and amplitude-ratio access to the reference state H^{tensor n}|phi(c_i)> with evaluation of diagonal IQP phase differences and derivatives to estimate conditional Pauli-Z moments and their parameter gradients. Model and target statistics are compared at the same condition; squared MMD losses are averaged across conditions to update one shared parameter vector. Generation prepares the condition-dependent input and applies the same learned IQP unitary, followed by computational-basis measurement. The circuit represents an n-qubit register schematically. Each displayed bitstring is a complete outcome from an independent shot at the selected condition.

## Build and edit

From the repository root:

```sh
mkdir -p /private/tmp/iqp-overview-build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/private/tmp/iqp-overview-build figure_drafts/conditional_iqp_overview/conditional_iqp_overview.tex
cp /private/tmp/iqp-overview-build/conditional_iqp_overview.pdf figure_drafts/conditional_iqp_overview/
pdftoppm -png -singlefile -r 300 figure_drafts/conditional_iqp_overview/conditional_iqp_overview.pdf figure_drafts/conditional_iqp_overview/conditional_iqp_overview
```

Copy the reviewed PDF and PNG to their canonical paper paths. The PDF is vector artwork with embedded fonts. Build details, author approval and checksums are recorded in provenance.json.

The author-approved layout aligns upper and lower encoder columns and outer left edges. The generation headings share a baseline; the MMD and training-data cards share top and bottom edges. Conditional samples wraps onto two lines. Scientific labels, raw schematic outcomes, circuit gates, and gate spacing are retained. The preceding publication is commit 69ead98d.
