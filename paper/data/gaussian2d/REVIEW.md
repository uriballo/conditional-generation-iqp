# Two-dimensional Gaussian section review

The main section asks whether a shared model generalizes to unseen combinations
of two conditioning variables. It explains the target, encoder and local learned
layer before defining the training designs and reporting full-distribution TVD.
Appendix B contains condition lists, seeds, optimization, selection, estimator
precision, reference access, preparation resources and simplification details.

| Claim | Evidence | Scope |
| --- | --- | --- |
| Mean and width enter distinct encoder components. | Main-text input equations and Appendix B shell-amplitude transform. | Fixed task-specific correlated Gaussian prior; encoder-only probabilities depend on width but not mean. |
| Sixteen local learned rotations approximate unseen mixed conditions after separate-coordinate training. | `results.json` fixed reporting metrics; no mixed fitting or development pairs. | Joint extrapolation outside a convex hull, within the marginal training ranges. |
| Both designs improve on the encoder alone over a common withheld domain. | `DENSE_GRID.npz`, `DENSE_GRID.json`, and Table II. | 1,652 pairs exclude both designs' fitting and development conditions. The heatmaps include all 1,681 points. |
| The results support two-input sensitivity, but imperfect width calibration. | Original combined campaign clamp diagnostics and endpoint width-response ratios, described in Appendix B. | Frozen-model interventions; no retrained condition-blind optimum. |
| A single learned angle weakens width response despite similar TVD. | Full repeated selection/training in the tied-angle campaign, summarized in Appendix B. | Adaptive architectural choice; the correlated preparation remains unchanged. |
| The reference state supports tractable classical access. | Explicit shell transform, weight sampling, and amplitude-ratio construction in Appendix B. | A physical preparation construction is described separately; this campaign did not execute the counter circuit. |

The dense grid is descriptive after selection, and some reporting coordinates
have historical exploratory exposure. Final fits use seeds separate from
configuration selection. Five fits quantify training variation and do not
establish a general ordering of training designs. TVDs are computed for each
individual fit, then averaged; they are not errors of a model mixture. The
study makes no claim of encoder necessity or quantum advantage.

The primary review corrected two compact-summary standard deviations against
the audited original campaign and corrected a documentation sentence that had
mistakenly described the full heatmap grid as withheld. No experimental values,
condition sets, model selection or checkpoints changed.

The section follows academic-writing and research-paper-writing guidance and
uses active voice. It separates the scientific design and outcome from the
implementation protocol. The prose review removes conversational history,
unsupported rankings and claims of fully calibrated width control. Across the
review dimensions, the argument, definitions, paragraph flow, figure/table
consistency and claim scope are ready for author review. Full numerical records
and architecture history remain in ignored research outputs; the committed
bundle supports portable figure replay, not a complete fit replay.

Validation: the portable renderer passes Ruff and reproduces the committed
figure metadata and table values exactly. Dense arrays match the approved
artifact byte for byte, and committed data hashes verify. The compiled paper
has no unresolved references or overfull boxes. Visual review covers the main
section, heatmap/table page and appendix. Existing Curie--Weiss/scaling float
deferral warnings also occur when compiling the previous manuscript.
