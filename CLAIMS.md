# Claim-to-evidence map

This table separates analytic claims from claims that remain contingent on the
production campaign. Exact metrics and sampled training objectives are never
treated as interchangeable.

| Claim | Protocol or evidence artifact | Figure or table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Conditional product-state encoders retain the sampling-and-ratio interface required by the classical estimator. | Derivation in `paper/main.tex`; encoder checks in the `iqp-cond` library. | Pipeline figure and estimator equations. | Applies only to encoders that implement the stated interface. | Supported analytically; author review pending. |
| The paired scalar encoder has constant expected input excitation and exact complement covariance, including odd joint widths and marginalized ancillas. | `experiments/tests/test_encoders.py`; complement metric tests; the complement-covariance proposition in the manuscript. | `scalar_gaussian_formulation.pdf`. | Uses the known target complement relation and does not enforce full wire exchangeability. | Supported analytically and by unit tests; author review pending. |
| One shared conditional IQP model represents held scalar-Gaussian conditions at (n=8,12). | `results/gaussian/campaign.json`, groups `shared`, protocols `full3`, `half3`, and `full5`; exact bitstring TVD. | Scalar results Fig. 2. | Controlled synthetic validity test; the physics dataset must support broader claims. | Needs production evidence. |
| The learned unitary improves on the selected encoder alone. | Exact encoder-only and trained metrics at $n=8$, $r=0$, and `full5`. | Profile panel in the scalar results figure. | This diagnostic does not compare alternative encoder designs. | Needs production evidence. |
| Sharing parameters reduces parameter instances and optimization jobs relative to condition-specific IQP circuits at an acceptable accuracy cost. | Campaign groups `shared`, `shared_matched`, and `direct`; anchor-mixture analysis; recorded reference draws, jobs, parameters, and wall time. | Resource panel in the scalar results figure. | Direct fits see target data at test conditions and are a supervised ceiling, not a generalization baseline. The two shared budgets separate fixed total compute from matched condition exposure. | Needs production evidence. |
| Ancillas improve the scalar conditional model. | Full Cartesian shared and direct campaigns over (r\in\{0,1,3,6,10\}). | Appendix ancilla grid. | Earlier exploratory runs were inconsistent and cannot support this claim. | Open; report the direction found by the production study. |

## Manuscript review qualifications, 2026-09-21

The [focused novelty comparison](notes/novelty-boundary-2026-09-21.md)
maps the draft's concrete reference-state families and individual phase derivatives
to Van den Nest's Theorem 3. Prior IQP work already supplies unbiased MMD
training, and Kiss et al. already use input conditioning with shared learned
parameters. Proposed positioning therefore concerns their combination in a
classically trainable conditional IQP construction, not a new general simulation
principle or input-conditioning architecture. The author accepted this positioning on 2026-09-21. The introduction now
explicitly attributes the simulation foundation to Van den Nest; no priority
claim over the entire literature is asserted.

See [the reference-guided review](notes/manuscript-review-bowles-2026-09-21.md).
These findings qualify interpretation of the frozen protocol; they do not change
its configuration or establish production outcomes.

| Claim or correction | Evidence | Figure or table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| The paired encoder fixes the midpoint visible distribution to uniform for all IQP parameters and ancillary widths. | Analytic eigenstate argument and independent four-qubit calculation in the review. | Annotate condition-resolved accuracy with the structural midpoint floor. | Applies to the stated paired map; not every admissible encoder. | Author accepted disclosure and retention of the current experiments; manuscript now states the restriction. |
| `full3` has one independent trainable anchor; `half3` and `full5` have two each. | Midpoint invariance, complement covariance, and population-gradient identities in the review. | Condition-protocol comparison. | Finite datasets and stochastic optimization still differ; nominal anchor counts are not independent information counts. | Author agreed to retain all protocols and disclose their distinct interpretations; manuscript updated. |
| Ancilla-only commuting gates do not affect the visible marginal. | Partial-trace argument and independent generic-input numerical check in the review. | Report nominal and potentially active parameter counts separately. | Requires the commuting IQP block and unconditioned visible marginal; visible-ancilla interactions and ancilla inputs remain relevant. Potentially active counts need not equal identifiable dimension. | Author accepted with explicit scope and cancellation proof; manuscript now distinguishes both counts. |
| The scalar Gaussian holds its exponent width fixed, not its actual standard deviation. | `experiments/common/datasets.py`, location-only calibration; exact moment calculations in the review. | Target definition and standard-deviation metrics. | Does not apply automatically to the separately calibrated two-coordinate target. | Author agreed; manuscript defines the fixed exponent width and condition-dependent target standard deviation. Metric implementation already uses actual target moments. |
| Logarithmic confidence overhead requires an explicit robust estimation argument for unbounded ratio samples. | Proposition 1 and the optional confidence-amplification appendix in `paper/main.tex`. | Main estimator proposition and Appendix A. | The ordinary training sample mean retains its mean-square guarantee; no new training estimator is claimed. | Author approved simple main guarantee; explicit appendix argument added. |

## Agreed estimator presentation

The author approved presenting the estimation guarantee before the encoder examples.
`paper/main.tex` now states Proposition 1 with independent reference sampling,
on-support ratio evaluation, fixed phase-independent input, unbiasedness and
second-moment bounds of 1 and 4. The relocated proof gives fixed-confidence
additive estimation via Chebyshev's inequality and credits the prior sparse-operator
framework. Exact sampling/arithmetic are explicit. The author approved keeping the main
guarantee simple; an optional median-of-means appendix now proves simultaneous
confidence for predetermined finite queries and distinguishes it from the ordinary
training estimator. The agreed numerical-accuracy qualification appears immediately after the proposition.

The author approved a brief numerical-accuracy qualification: the estimator guarantee assumes exact sampling and arithmetic, including ratio and phase evaluation. Approximate implementations require separate error control; no numerical-stability analysis is claimed. Experimental comparator terminology, including “supervised ceiling,” is deferred pending possible experimental redesign.


## Agreed preparation and MMD scope

The author approved a brief distinction between classical reference-state access and efficient quantum input preparation. Product rotations provide the latter directly; entangled preparation remains a separate requirement. The MMD derivation now cites Recio-Armengol et al., specifies independent target/reference draws, and distinguishes population expectations at fixed parameters from gradients of the empirical objective with fixed target data. The target-only U-statistic changes the empirical loss by a parameter-independent constant.

The author retained the existing classical-trainability terminology as efficient classical access to the training objective. Broader introductory loss-access and sampling-hardness qualifications are deferred, alongside experimental comparator terminology.

## Scalar Gaussian translation review, n=8

The new review package is documented in
[`experiments/scalar_gaussian_translation/review/REPORT.md`](experiments/scalar_gaussian_translation/review/REPORT.md).
Its locked audit uses no ancillas and does not authorize the disabled production
proposal or any n=12 run.

| Claim | Evidence | Limitation | Status |
| --- | --- | --- | --- |
| A shared complete IQP unitary learns a useful correction to either predeclared affine RY encoder, neither of which consumes target statistics at runtime. | Corrected frozen audit gate `encoder_only_improvement`: median grouped TVD improvements 0.519 (broadcast) and 0.510 (qubit-dependent); worst category-seed improvements exceed 0.374. | The tracked record does not establish that the affine coefficients were derived independently of the target family. Exact full distributions and TVD are available only in the simulated n=8 study. | Supported in this benchmark with narrowed encoder provenance. |
| The general encoders retain controlled accuracy at unseen interpolation and near-extrapolation conditions, with weaker far extrapolation. | Corrected audit median seed-mean TVD: 0.280/0.277/0.342 for broadcast and 0.283/0.274/0.353 for held/near/far; both pass all 28 frozen checks after excluding mirrored diagnostics and evaluating layouts separately. | Worst far-condition TVD reaches 0.791 and 0.687; reflected far extrapolation is worse. | Supported with explicit boundary failures and a recorded post-audit evaluator correction. |
| Qubit-dependent affine RY materially improves on broadcast affine RY. | Their gate outcomes and condition-resolved TVDs are similar; neither dominates across regimes. | The comparison covers one predefined qubit-dependent formula, not the full design space. | Not supported; omit the improvement claim. |
| The symmetry-informed anti-block construction achieves lower TVD than the two predeclared affine encoders. | Anti-block audit median seed-mean TVD is 0.113 for held interpolation and about 0.12 for both extrapolation regimes, substantially below both general encoders; its complement-covariance error is numerical zero. | Anti-block jointly changes complement covariance, paired excitation structure, and the midpoint state (`|+>^8` at `c=0.5`). The comparison does not isolate symmetry as the cause, and one restart misses the 0.32 diagnostic threshold. | Supported as a benchmark-specific comparison of complete encoder constructions. |
| The conditional model is generally superior to classical or direct per-condition alternatives. | Exact anchor interpolation and independent direct IQP fits are strong references; direct audit mean TVD is 0.207. | These references answer different questions and the shared general model does not uniformly beat them. | Not claimed. |
| The benchmark demonstrates quantum advantage or large-register trainability. | No corresponding evidence was collected. | n=12, production, hardware sampling, and sampling-only selection remain outside scope. | Prohibited by the review evidence. |

## Author-approved raw comparison, 2026-10-01

| Claim or decision | Protocol or evidence artifact | Figure | Limitation | Review status |
| --- | --- | --- | --- | --- |
| The main Curie--Weiss comparison presents exact raw circuit output, without global-bit-flip mixing or other probability postprocessing. | `figure_drafts/curie_weiss_balanced/figure_data.json`, `run_provenance.json`, and `provenance.json`; manifest e48a6304b06fadec89096386186d12cf3b5d4cb7b367da9060f2d8c2a1960cab. | `paper/figures/curie_phase_comparison_raw.pdf`; identical legacy flip filename alias. | Profiles average three seed distributions. Shared evaluation conditions are held out; independent fits are in-sample. Checkpoint selection uses exact raw profile TVD. New training masks/reference budgets and bandwidths changed together, so causal attribution is not isolated. Earlier manuscript statistics require manual replacement. Ancilla/scaling assets still refer to the prior campaign. | Figure visually approved by the author; raw-only presentation applies to all numerical fit figures and metrics. |
