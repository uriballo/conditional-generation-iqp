# Ashkin–Teller joint-coupling extrapolation screen

2026-09-07. **The two-block IQP model fails this screen in all three seeds.**
Strong classical predictors work substantially better. Stop this setup rather
than expand it or adjust the split using these results. The result is exploratory
and awaits author scientific review; no manuscript or canonical campaign was
modified by this screen.

[Design](ashkin-teller-joint-extrapolation-design-2026-09-07.md) ·
[Frozen protocol, per-condition metrics, selection diagnostics and hashes](ashkin-teller-joint-extrapolation-result-2026-09-07.json)

## What was run

Eight visible qubits, no ancillas, two four-site ring layers, fixed kappa=0.3.
Train at seven axis conditions (a,.1) and (.1,b), with a,b in {.1,.3,.5,.7}.
Test all four combinations of {.5,.7} squared, strictly outside the training
convex hull. The five unused grid conditions were never generated. Each seed
has 1,000 training and 1,000 independent validation samples per training point.

Three paired seeds compare shared product and two-block IQP, with one additional
seed-0 two-block exact-population control. Each has 36 nominal parameters and
500 Adam updates. The map alpha(c)=.225*pi-.20*pi*(c-.1)/.6 and the sampled-MMD
settings were fixed before fitting. Checkpoint selection uses only the mean
empirical validation TVD over the seven training conditions. At this 500-step
budget, the existing training driver supplies initial, scheduled final, and
lowest-validation-MMD candidates; validation occurs every25 steps, but not every
validation evaluation becomes a retained checkpoint. All selected fits use step500.

Classical baselines are symmetry-aware empirical log composition, regularized
conditional log-linear orbit regression, and nearest-condition histograms.
All use the same training/validation archives. Histogram smoothing and regression
regularization are selected using validation likelihood at training conditions
only. Product-IQP predictions are additionally averaged over both independent
layer flips without checkpoint reselection.

All seven IQP fits and all three classical model sets, including predicted test
probabilities, were saved and SHA256-sealed before the first test-target computation.
No target-aware test fits, parameter changes or extra optimizer steps followed
scoring. Source commit: `4fc4aa96f5a82f16fdb21fb93d69acb02122ee29`.
Protocol SHA256: `51f2088f8dca7ef97be5755861ce7f2399f91f7832cdeaf71b3971588de6343a`.

## Results

Each seed mean covers the four test conditions. The across-seed mean is descriptive;
these three-seed ranges are not confidence intervals. All TVDs are exact
comparisons of normalized model distributions against enumerated targets.

| Method | Mean training TVD | Mean test TVD | Test seed means, min–max |
| --- | ---: | ---: | ---: |
| Shared two-block IQP | .1776 | .3305 | .3196–.3499 |
| Shared product IQP | .2719 | .4520 | .4047–.5331 |
| Symmetrized shared product IQP | .2298 | .4314 | .3882–.5013 |
| Symmetry-aware empirical log composition | .0834 | .1004 | .0671–.1215 |
| Symmetry-aware conditional log-linear model | .0511 | .0659 | .0556–.0780 |
| Nearest-condition symmetry-aware histogram | .0828 | .3813 | .3747–.3877 |

The two-block mean test TVDs by seed are .3499, .3221 and .3196, all above the
predeclared .15 threshold. The worst individual test TVD is .4417 (threshold .25).
Worst connected-bond RMSE per seed is .2273, .2294 and .2438, all above .10.
The absolute and predictive-advantage gates therefore fail. The model beats
nearest-condition lookup, but that does not compensate for losing to both
stronger sample-trained classical predictors.

The untrained two-block encoder averages .2329 test TVD, better than the trained
shared model in every seed. Its training TVD is .3871, so fitting the axes improves
training accuracy while making these test predictions worse. The one-seed
exact-population control gives training TVD .1709 and test TVD .3332, with worst
connected-bond RMSE .2470. Thus eliminating MMD estimation noise does not resolve
the failure at this map, ansatz, initialization and update budget.

Because training errors remain substantial, this is not a clean isolation of
extrapolation error after near-perfect fitting. It is also not a proof of an
expressivity limit: no global optimum or impossibility result was established.

## Classical baseline checks

All 18 log-linear regularization candidates converged with gradient infinity norm
below 1e-6. The largest was 4.64e-7. Histogram smoothing selected lambda=1 for all
seeds. The pooled regression selected regularization 1e-6, 1e-5 and 1e-5. The
first choice is the lower edge of the fixed grid; it was not expanded after test
access. Finite-difference checks of the analytic gradient passed before fitting.

Reconstructing the test laws from exact axis distributions via
p(a,b) proportional to p(a,.1)*p(.1,b)/p(.1,.1) gives maximum test TVD
3.73e-16. This confirms the algebraic identity and its implementation. It is an
analytic-data diagnostic, not a sample-budget-matched competitive result.

The pooled categorical model has 189 identifiable coefficients and a favorable
affine-log-odds assumption for this family, while each IQP model has 36 nominal
parameters. Histogram storage, optimization and postprocessing costs also differ.
All receive identical sample archives, but these are not parameter- or compute-
matched comparisons and support no quantum-advantage claim.

## Verification, decision and evidence

Verified all seven IQP archives against independently reconstructed validation
histograms, checkpoint argmins, pooled initializations and angle maps. Re-evaluated
the selected model predictions at the correct train/test inputs. Verified all
classical choices against their saved training/validation orbit counts, probability
normalization, orbit expansion, and block-flip symmetries. All prediction hashes
were unchanged after evaluation. All 11 evaluated target laws agree with scalar
energy calculations; training Fourier vectors agree with explicit parity sums.

The frozen experiment completed with no protocol deviations. Disposable runners,
data and checkpoints were deleted after the outcome was captured; the linked
JSON retains protocol, numerical results, solver diagnostics and provenance hashes.
The proposed ten-seed sample-efficiency expansion was not triggered.

Recommendation: do not promote Ashkin–Teller to the paper's third generalization
benchmark on this evidence. The previous interpolation pilot remains a demonstration
of a viable independent-flip encoding; this harder split shows that the current
shared model does not extrapolate reliably, while classical structural models do.
If the paper needs a third generalization case, reconsider the target/question
rather than repeatedly changing this split in pursuit of a positive result.

| Claim | Evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Shared two-block IQP reliably extrapolates on the axis/corner split | Frozen three-seed results in JSON | Results table | All absolute gates fail | Not supported by this screen |
| Sampling noise alone accounts for failure | Seed-0 exact-population control | Diagnostic above | One seed, finite budget; no optimum established | Not supported by this control |
| Strong classical predictors outperform this IQP setup | Sample-paired regression and log-composition results | Results table | Different parameter budgets and inductive biases | Exploratory, awaiting author review |
