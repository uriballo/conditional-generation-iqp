# Ashkin–Teller viability pilot, 2026-09-07

The two-block model learns the held-out combinations consistently, but a simple
symmetry-aware histogram baseline performs better. Keep this as evidence that
the encoding is viable; this split does not establish a demanding generalization
benchmark. Exploratory eight-qubit screen, not an author-reviewed paper result
or canonical campaign.
Source commit: `4fc4aa96f5a82f16fdb21fb93d69acb02122ee29`.
Frozen protocols and numerical results: [JSON](ashkin-teller-pilot-2026-09-07.json).
Design: [proposal](ashkin-teller-compositional-generalization-proposal.md).

## Capacity result

The predeclared seed-0 sampled two-block gate passed. Across three independent
conditions, median exact joint TVD was 0.0997, worst TVD 0.1547, median improvement
over the untrained encoder 0.3534, and worst connected-bond RMSE 0.0462.
All three seed-0 exact-population fits also reached TVD below 0.15.

| Couplings (a,b) | Product, sampled | Global flip, sampled | Two block, sampled | Two block, population |
| --- | ---: | ---: | ---: | ---: |
| (.15,.15) | .2682 | .2248 | .1547 | .1134 |
| (.15,.60) | .1440 | .2669 | .0997 | .0589 |
| (.60,.60) | .1116 | .0996 | .0843 | .0833 |

These are independent fits using the same constant input angle pi/8, data,
initialization and 500-update budget across encoders. Exact raw TVD is evaluated
against the enumerated target; selection uses independent empirical validation
TVD. Population mode has analytic target access for training/candidate generation.
All fits have zero ancillas and 36 complete-ansatz parameters.

Two-block replication seeds 1 and 2 gave median TVDs 0.1404 and 0.1006, with worst
TVDs 0.1730 and 0.1766. However, seed 2 at (.15,.15) had connected-bond RMSE
0.1316, exceeding the original 0.10 threshold. The initial gate passes but the
full result is not uniformly robust. This warrants an exploratory conditional
pilot, not a claim of reliable capacity across seeds.

## Conditional protocol

Before accessing any held-out target, freeze the 12-perimeter/4-interior split
and affine map alpha(c)=pi/8 - .20*pi*(c-.375)/.45. The first four qubits receive
alpha(a), the last four alpha(b). Coupling levels are .15,.30,.45,.60 and kappa=.3.
Compare shared product and two-block models. The existing global-flip encoder
uses only one scalar angle and is excluded from this two-coordinate comparison.

Independent two-block perimeter fits form a bank. At each interior point, average
the two linearly interpolated distributions bracketed along the a and b axes.
This is a fixed convex probability interpolation. Also evaluate encoder-only
outputs and independent fits at the test conditions, explicitly target-aware.
The JSON records the frozen thresholds and stopping rule. No condition-map or
optimizer tuning on held-out outcomes is allowed.

## Generalization result and classical diagnostic

Across three seeds, sampled two-block mean held-out TVDs are 0.0665, 0.0629,
and 0.0677. The shared product model gives 0.1734, 0.2021, and 0.1925.
The one-seed exact-population two-block control gives 0.0621. No held-out
samples enter training or checkpoint selection. All four interior conditions
are genuinely new combinations of scalar levels already seen in training.
The worst connected-bond RMSE per seed is 0.0778, 0.0788, and 0.0714.

The interpolated independent two-block bank gives mean TVDs 0.0989, 0.0930,
and 0.0964. The predeclared generalization gate passes in all three seeds:
shared TVD improves by at least 0.02, while absolute TVD, encoder-only improvement,
and correlation criteria also pass. This gate did not include the stronger
classical diagnostic below and must not be interpreted as passing that comparison.

| Method | Mean test TVD, averaged over seeds | Seed means, min–max |
| --- | ---: | ---: |
| Shared two-block | 0.0657 | 0.0629–0.0677 |
| Shared product | 0.1893 | 0.1734–0.2021 |
| Interpolated independent two-block bank | 0.0961 | 0.0930–0.0989 |
| Raw training-histogram interpolation (post-hoc) | 0.0830 | 0.0792–0.0871 |
| Symmetrized training-histogram interpolation (post-hoc) | 0.0436 | 0.0393–0.0477 |

Each seed mean covers the same four interior conditions. These ranges are not
confidence intervals. The classical diagnostics receive the same training
samples as the IQP models.

After those results, an **untuned post-hoc diagnostic** applied the already
frozen interpolation formula directly to training histograms, with and without
averaging the four known block-flip transformations. Neither version uses
validation or held-out samples. Symmetrized histogram interpolation achieves
mean TVDs 0.0393, 0.0477, and 0.0439, beating the shared quantum model in every
seed. Raw histogram interpolation gives 0.0792, 0.0871, and 0.0826.

Interpolating exact boundary target laws gives TVD 0.0109. This reference has
analytic target access and is not a sample-budget-matched baseline, but it
shows that the chosen interior distributions are close to simple mixtures of
boundary distributions. The shared model's positive result is therefore an
encoding/parameter-sharing demonstration, not evidence that this is a difficult
compositional task or that IQP outperforms classical prediction.

The product of exact target layer marginals has mean test TVD 0.1880 and
connected-bond RMSE 0.156–0.245. The shared two-block model captures substantial
cross-layer dependence rather than only matching each layer separately. This
marginal-product reference is not the optimal product approximation under TVD.
The four target-aware independent sampled fits at the interior points average
0.1489 in seed 0: a finite-budget optimization reference, not a capacity bound.

Recommendation: do not launch a larger paper campaign on the strength of this
split. The target is viable for showcasing independent Z2 x Z2 symmetry and
its encoder, but a stronger generalization claim needs a separately frozen,
more discriminating split and the symmetry-aware classical baseline retained.
Do not tune on these now-consumed test points and relabel them as unseen.

## Verification and claim scope

The target vector was checked against an explicit spin-loop calculation at the
three development points, normalized and checked for both block-flip symmetries.
The kappa=0 target factorizes across layers; kappa=.3 has nonzero connected bond
statistics. The existing two-block encoding tests passed (2 tests).

| Proposed claim | Evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Two-block capacity merits a conditional pilot | Frozen capacity protocol and 24 fit records in linked JSON | Capacity table above | One seed misses correlation threshold; n=8 only | Exploratory, awaiting author review |
| Independent flips are exact architectural symmetries | Encoder construction and two-block tests | None | Not learned generalization | Implementation verified |

The disposable exact-population runner initially applied the one-vector NumPy
Walsh transform to a matrix. Verification caught this; that diagnostic run was
discarded and rerun with an explicit per-condition transform. No map, split,
optimizer setting or gate changed. The sampled objective never used that array.
The corrected target transforms agree with explicit parity sums.

Shared models use 36 parameters, while the 12-member independent bank uses
432 total. Both receive the same 12,000 training observations and matched
per-condition update budgets; sharing pools statistical information across
conditions. Histogram baselines have a different representation and training
cost. These comparisons do not establish quantum advantage or scaling.

| Proposed claim | Evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Shared two-block model learns held-out coupling combinations | Three-seed shared fits and frozen split in JSON | Numerical results above | One size, one untuned map, in-domain interpolation | Exploratory, awaiting author review |
| Current split establishes a strong predictive advantage | Symmetry-aware histogram diagnostic in JSON | Numerical results above | Classical comparator is better; exact boundary interpolation is already accurate | Not supported |

All 71 valid fit archives were checked for normalized probabilities, empirical
validation checkpoint selection, and two-block symmetries where applicable.
Shared held-out outputs were independently re-evaluated at the correct interior
angles. Data, maps and initial parameters match between shared encoders; the
bank uses identical per-condition training/validation data and maps. All 16
target laws agree with explicit scalar energies, and their Fourier vectors agree
with explicit parity sums. Numerical outcomes and frozen protocols are retained
in JSON; disposable scripts and checkpoints were deleted after evidence capture.
No production implementation, canonical campaign, or manuscript was changed
by this pilot.
