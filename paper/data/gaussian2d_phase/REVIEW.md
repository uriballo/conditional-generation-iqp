# Phase-conditioned Gaussian section review

The author approved promotion as a conditional-capability and generalization
example after independent adversarial review. The section replaces the earlier
width-amplitude encoder; it retains the author's other manuscript revisions.

The narrative progresses from single-coordinate product-state translation to
two-coordinate phase encoding. It defines the fixed Gaussian prior, the linear
and quadratic phases, the shared output angle, the training geometries and the
reported errors. Main text keeps scientific assumptions and both selection protocols
visible; the appendix contains training budgets, optimizer settings, controls,
resource accounting and response diagnostics.

| Claim | Evidence | Status |
| --- | --- | --- |
| Both conditions modify phases while encoder probabilities remain fixed | Diagonal linear/quadratic phase law and fixed prior | Supported exactly |
| Classical reference access is tractable | Seventeen shell amplitudes, Krawtchouk transform, shell sampling and lookup ratios | Supported for the symmetric family; estimator variance is a separate issue |
| The shared model approximates pairs absent from its fitting and development | Five selected fits/design; 32 joint and 16 separate mixed reporting pairs | Supported descriptively; conditions were previously explored |
| Separate fitting transfers to mixed pairs outside the joint hull | Twelve pairs outside the diamond, within both marginal ranges | Supported; joint-domain extrapolation, not extrapolation beyond marginal ranges |
| Both response coordinates are useful | Mean/width sweeps, monotone width on tested slices | Supported on the diagnostic sweeps; not perfect independent calibration |
| Training discovers the conditional law or is necessary | Analytic controls match or improve fidelity | Not claimed |
| Gaussian amplitude preparation improves accuracy | Product-prior analytic diagnostic is stronger | Not claimed |
| The original plotted fits use only finite target information for selection | Exact population development access improves terminal errors | Not claimed for those plots; finite-validation follow-up reported separately |
| This is an independent confirmatory or advantage benchmark | Encoder, initialization branch and reporting conditions were explored | Not claimed |

The numerical bundle preserves frozen diagnostics and independently replays
initial, selected, terminal and fixed-control results. Full TVD equals profile
TVD because both input and learned rotation preserve permutation symmetry. The
control replay does not optimize or select parameters.

The existing counter construction supports polynomial prior preparation, with
five clean workspace qubits and 296 high-level operations at n=16. The pilot
used classical shell access and explicit state-vector circuit validation. The
appendix counts 120 conditional pair gates and distinguishes high-level costs
from compiled depth or hardware performance.

Self-review finds the contribution scoped to capability, the encoder defined
before use, and the generalization geometry explicit. Exact development access,
previous exposure, analytic controls, initialization dependence and the
production estimator-tail qualification remain visible. No generic learning,
preparation benefit, sample-only selection, or quantum advantage claim remains.

## Revised section outline

1. Extend the one-coordinate demonstration to center and width.
2. Define the fixed input and its linear and quadratic phase features.
3. Explain shared readout and tractable symmetric reference access.
4. Distinguish separate-axis generalization from joint-grid interpolation.
5. Report the original benchmark and the finite-validation follow-up separately.
6. Present direct finite-mean calibration and product-prior controls.
7. Limit the conclusion to structured conditional generation and calibration.

## Finite-validation claim checks

| Claim | Evidence | Status |
|---|---|---|
| Useful mean/width response survives finite-target checkpoint selection | Five raw-estimator fits per design, 128 validation strings per development condition; TVD 0.0288/0.0280 | Supported with exact model moments and previously chosen configuration |
| Both response coordinates remain useful | Mean displacement approximately100%; width endpoint ratios89–93%, no reversals on checked slices; width clamping increasesTVD | Supported on diagnostic slices, not globally independent calibration |
| Iterative fitting is necessary | Finite-mean calibration achievesTVD0.0263/0.0259 | Not claimed |
| Gaussian amplitude preparation is necessary | Product-prior finite calibration achievesTVD0.0241/0.0230 | Not claimed; conditional quadratic gates remain |
| Pair averaging improves fidelity | Five paired fits perdesign with mixed accuracy differences | Not claimed |

The revised main text contains no seed identifiers or optimizer configuration
lists. The existing figures remain tied to the original exact-selection
experiment; the finite-validation table describes its own campaign. Fresh
pools do not undo earlier oracle-informed encoder and optimizer choices.

The five-dimensional self-review finds the contribution limited to conditional
capability; the encoder and readout defined before interpretation; finite data
budgets and selection access explicit; terminal results and strongest calibration
controls available; and task-informed features distinguished from target-law
learning. No additional Gaussian experiments are required for the agreed scope.
