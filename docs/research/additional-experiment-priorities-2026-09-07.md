# Additional experiment priorities — 2026-09-07

Status update: the author selected study 1, which is now complete and verified; see [results](reference-estimator-study-result-2026-09-07.md) and [complexity/compilation audit](reference-estimator-complexity-2026-09-07.md). Study 2 is now authorized, expanded before freeze to test the data-size × ancilla interaction, and complete; see [frozen protocol](ashkin-teller-data-efficiency-protocol-2026-09-07.md) and [launch record](ashkin-teller-data-efficiency-launch-2026-09-07.json). See the [data/ancilla result](ashkin-teller-data-ancilla-result-2026-09-08.md) for verified outcomes and manuscript integration. The recommendations below preserve the original proposal, not the final frozen protocol. The existing supervisor draft is complete for its current claims. Two free Strix Halo machines permit independent studies; runtime and device placement require a local timing pilot, not assumed speedups.

## 1. Estimator precision and computational cost

This is the closest experimental test of the methodological contribution: conditional reference sampling, amplitude-ratio moments and phase gradients.

Use fixed condition/parameter sets for product, cat-like and correlated bond encoders, including initial and already-trained configurations where available. At total widths 8, 12 and 16, compare estimates with exact enumeration. Vary reference samples on a logarithmic grid, for example 128, 512, 2048 and 8192; use independent estimator repetitions. Report absolute bias estimates with uncertainty, RMSE, componentwise gradient error, runtime and peak memory. Avoid relative gradient error when exact gradients are near zero. Condition and phase choices must be fixed before inspecting estimator error.

Keep mask-bank error separate from reference-sampling error: first fix the observable/mask set and evaluate against its exact reference expectation; then measure mask subsampling separately if needed. A numerical bias estimate is not a proof of unbiasedness. Separate compilation, steady-state compute and reference preparation; synchronize asynchronous accelerator work before timing. Larger-width timing-only points can be added after a smoke run, but do not label agreement between two Monte Carlo estimates as exact accuracy or full-distribution learning. Do not infer trainability from a bounded estimator variance.

Suggested output: one figure with error versus reference samples and cost versus width. This complements the proof and checks how practical the correlated-reference extension is.

## 2. Data-efficiency follow-up on the spatial case

Use n=12 Ashkin–Teller, both IQP encoders, both classical predictors and the two deterministic input controls. Keep the existing perimeter/interior coordinates and architecture. Compare 100, 300 and 1000 training examples per perimeter condition, with ten fresh paired seeds and nested training subsets within each seed. All learned models receive the same training/validation data for each arm. Keep an independently sampled fixed validation set across training-size arms and disclose its size; the x-axis is training-set size, not total labeled-data access. Keep quantum optimizer/mask/reference budgets fixed and baseline selection rules frozen. Fit all arms anew so comparisons share the new seed population (60 IQP fits plus 30 paired classical bundles).

Report exact interior joint TVD, connected-bond error, input-relative improvement and paired seed uncertainty; retain all nine conditions and all outcomes. This tests dependence on empirical data and supplied input structure. It does not isolate topology or establish out-of-domain generalization. Existing test coordinates have already been examined: call this a predeclared follow-up, not an untouched replication or newly discovered split. Freeze choices and seal predictions before the follow-up evaluation, without retroactively relabeling old data.

Suggested output: learning curves for all four learned models with encoder-only horizontal controls. Classical wins remain scientifically useful.

## Lower priority

A bounded n=12 optimization-budget study (500 versus 2000 updates, both encoders, otherwise matched) could clarify whether larger-size residuals shrink with more optimization. It would show budget sensitivity, not prove a capacity or noise mechanism. More sizes, four-body gates, wrong-topology inputs, extrapolation redesigns or another dataset should be motivated by an additional claim; available hardware alone is insufficient.

## Literature context

The inherited large-scale classical IQP training framework already demonstrates very large circuits: [Recio-Armengol et al., arXiv:2503.02934](https://arxiv.org/abs/2503.02934). This makes a practical evaluation of the new reference interface more informative than a width record alone.

A recent preprint reports that low moment-matching loss does not reliably imply generative generalization on its constrained/genomic benchmarks: [Raj et al., arXiv:2608.31117](https://arxiv.org/abs/2608.31117), submitted 31 August 2026. Its tasks differ from conditional interpolation here; it does not establish a failure of our model. It reinforces keeping exact held-out distribution and structural evaluation, rather than presenting loss convergence alone as success. Only the abstract was inspected for this recommendation; no manuscript citation or theoretical conclusion was added from it.
