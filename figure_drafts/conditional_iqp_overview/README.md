# Conditional IQP overview

This author-approved workflow is the publication figure for 2026-10-02. The canonical PDF is `../../paper/figures/conditional_iqp_workflow.pdf`, with preview `../../paper/figures/conditional_iqp_workflow.png`; the editable source is `figure_drafts/conditional_iqp_overview/conditional_iqp_overview.tex`. The fixed encoder, classical-computer estimator and combined Conditional IQP block are bundled with the source, preview, PDF and provenance. The hand-crafted manuscript `.tex` remains manually integrated; only figure assets are promoted.

The generation panel uses plain vertically stacked condition boxes. Only the highlighted $c_2$ has an outgoing arrow, labelled $c_i$, into the combined model. Measurement yields several samples carrying that same condition label. The diagram uses no selection dialog, explanatory call labels, grouping brackets or multiplexers. The illustrated condition boxes are schematic; the encoder may accept other conditions, including continuous values within its domain.

The encoder map is fixed while its prepared state varies with $c_i$. Classical training uses tractable reference-state sampling and amplitude ratios, together with the current shared parameters; the encoder picture does not imply quantum execution during training. Target data and estimated model moments enter MMD separately. Its loss averages squared discrepancies over the training conditions. The optimizer changes one parameter vector, and the lower arrow uses that vector after training. There is no direct MMD-to-model handoff.

The combined model displays $U_{\boldsymbol\theta}\ket{\phi(c_i)}$. Conditional IQP names the composite model; only its learned unitary is required to have IQP form. The output card labels raw samples with the supplied condition and applies no filtering or postprocessing. Bitstrings and statistic bars are schematic. The quantum panel describes the generation pathway rather than claiming hardware execution.

All artwork is native, editable TikZ with Computer Modern typography and the paper palette. Earlier variants drew on the training/deployment organization of [Figure 1 of Recio-Armengol, Ahmed and Bowles, arXiv:2503.02934v2](https://arxiv.org/html/2503.02934v2#S1.F1). No image assets were copied.

## Proposed caption

Conditional IQP training and generation with shared parameters. (a) For a training condition $c_i$, the fixed encoder supplies tractable reference sampling and amplitude ratios to the classical estimator. The estimated model moments and condition-labelled target data determine the MMD loss averaged over the training conditions. Optimization updates one shared phase vector, which also enters the estimator. (b) After training, a selected condition determines the input state while the learned unitary remains fixed. The outgoing arrow from the highlighted $c_2$ box illustrates the input $c_i=c_2$ and the matching raw measurement samples. Condition labels, bitstrings and statistic bars are schematic; the lower panel illustrates the quantum-generation pathway rather than a hardware experiment.

## Build and edit

Edit the palette, labels and coordinates in `conditional_iqp_overview.tex`. Generation uses plain condition boxes and one matching sample card. Training data use one card with compact condition-tagged rows. The parameter node separates optimization from the handoff to generation. The measurement gauge is separate from the combined model. No external images are needed.

From the repository root, build outside the source directory:

```sh
mkdir -p /private/tmp/iqp-overview-build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/private/tmp/iqp-overview-build figure_drafts/conditional_iqp_overview/conditional_iqp_overview.tex
cp /private/tmp/iqp-overview-build/conditional_iqp_overview.pdf figure_drafts/conditional_iqp_overview/
pdftoppm -png -singlefile -r 300 figure_drafts/conditional_iqp_overview/conditional_iqp_overview.pdf figure_drafts/conditional_iqp_overview/conditional_iqp_overview
```

The PDF is vector artwork with embedded Computer Modern fonts; the PNG is the 300 dpi preview. Build, dimensions and visual inspection are recorded in `provenance.json`.
