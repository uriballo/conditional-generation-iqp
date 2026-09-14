# Bond-aware Ashkin–Teller development ablation

2026-09-07. **The proposed input improves joint fitting; adding four-body phases
helps it further. Correlation accuracy remains uneven.** This is a development
result, not evidence of fresh interpolation or extrapolation. Classical superiority
is not an inclusion criterion for this paper and was not a gate in this screen.

[Design](ashkin-teller-symmetry-preserving-redesign-2026-09-07.md) ·
[Frozen protocol, all results and verification](ashkin-teller-bond-redesign-ablation-2026-09-07.json)

## Controlled comparison

All four models fit the same 11 previously inspected conditions: the seven axis
conditions and four former extrapolation corners. Every one now enters training
and validation. These are **not held-out results**, and the numbers must not be
compared directly to the previous .331 extrapolation error as an improvement in
generalization. The shared circuit is still identical across conditions.

Use eight visible qubits, no ancillas, fixed kappa=.3, and three paired seeds.
Each seed supplies 1,000 training and 1,000 independent validation observations at
every condition. All 12 fits use exact-population multiscale MMD, 500 Adam updates,
the same schedule and pooled-data initialization rule. Candidate selection uses
mean empirical validation TVD over all 11 development conditions. Exact target
laws enter the population loss; this is a capacity/development study.

The cat encoder uses the existing affine angle map over [.1,.7]. The new bond
encoder uses g(c)=c and a fixed bond phase eta=pi/8. It supplies uncoupled Ising
ring amplitudes, but no cross-layer kappa term. Its encoder-only output has zero
connected cross-layer bond correlations. The supplied single-layer physics must
be disclosed in any paper use.

Duplicate generator characters on the even/even Fourier support were removed.
The complete two-local bank has 30 independent phases. Its aligned four-body
extension has 32. Original 36-parameter initial phases were collapsed by summing
duplicate coefficients, preserving the initial unitary on the reference support;
the two extra parameters start at zero. Removing duplicate parameters changes the
optimizer parametrization, so both cat controls were rerun in this comparison.

Source commit: `4fc4aa96f5a82f16fdb21fb93d69acb02122ee29`.
Protocol SHA256: `5e175f7c1309290269b93d555ea334801e4cbc73e5fd9f6aff71ff453ad2cf0c`.

## Results

Means average the 11 conditions within each seed and then the three seeds.
Correlation error is RMSE over the four connected corresponding-bond observables
at a condition; the table averages that RMSE across conditions/seeds.

| Input | Circuit | Mean joint TVD | Worst joint TVD | Mean correlation RMSE | Worst correlation RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Cat | Two-local | .1247 | .1866 | .0881 | .2293 |
| Cat | Two-local + four-body | .1502 | .2744 | .0839 | .2243 |
| Bond-aware | Two-local | .1055 | .2066 | .0946 | .2195 |
| Bond-aware | Two-local + four-body | **.0970** | .1849 | .0905 | .2072 |

The combined redesign lowers mean joint TVD by about 22% relative to the cat
baseline. The four-body extension helps the bond input, but does not consistently
help the cat input: one cat-extension seed is worse (.2108 mean TVD), while its
other seeds reach .1218 and .1181. A larger hypothesis class need not optimize
better within this finite budget.

The bond encoder alone has mean joint TVD .2575; learned phases lower it to
.0970. Its mean connected-bond RMSE drops from .1716 to .0905. Thus the circuit
learns substantial cross-layer dependence rather than merely returning the
uncoupled input law. Nevertheless, mean correlation accuracy is not better than
the trained cat baseline, and the improvement is not uniform across conditions.

For the combined model, seed 0 at (.1,.1) has correlation RMSE .2072, versus
encoder-only .3074. At (.7,.7), correlation RMSE worsens from encoder-only .0408
to .0913. The joint objective trades accuracy between conditions and observables.
The bond-extension seed mean TVDs are .096977, .096984 and .096977. These very
similar population-loss fits are not a sampled-training robustness demonstration;
the seeds vary data-dependent initialization and empirical validation selection.

## Reference-access verification

Implemented a disposable transfer-matrix amplitude routine and a sequential
Born sampler for the post-Hadamard bond reference. The sampler uses constant 4x4
doubled transfer contractions, not a categorical draw from an enumerated state
probability table. Two independent ring draws give the two-block reference.
Matrix products and prefix/suffix contractions are rescaled for stability in the
tested range. These routines were not integrated into production code.

- All amplitudes at m=4,6 and g=.1,.3,.5,.7 agree with explicit physical-state
  construction plus Hadamards to below 8e-16. Odd-parity amplitudes are exactly zero.
- Joint reference sampling at three coupling pairs, 50,000 draws each, has TVD
  .0122–.0148 against the exact reference probabilities.
- At a fixed nonzero phase vector, all 256 moment estimates have maximum error
  .0074–.0117. Four checked phase gradients across those observables agree with
  exact finite differences to within .0190. These are finite Monte Carlo checks,
  not exact equalities or guarantees for every parameter value.
- A 32-site-per-ring sampler/amplitude smoke test returns even-parity samples and
  finite amplitudes without enumerating the basis. It is not a runtime-scaling
  study or a numerical-stability proof for arbitrary lengths/couplings.
- The exact expectation of the amplitude-ratio estimator agrees with all 256
  Fourier moments across all 11 development conditions to 2.45e-15.

The generic input-distinguishability audit found one necessary exact-fit
constraint violated by the bond input: between (.3,.1) and (.5,.5), target TVD
exceeds the input trace distance by .00627. For any shared unitary and measurement,
the two fitting errors must sum to at least this gap. This rules out simultaneous
exact fitting of that pair with these inputs. The lower bound is far below the
observed errors and does not explain their magnitude or rule out useful
approximation. No violation was found for the cat input on the checked pairs.

All 12 fit archives were checked for checkpoint argmins, normalization, symmetries,
and recorded metrics. The collapsed cat initial phases reproduce the existing
implementation. Exact spectral MMD was checked independently against explicit
kernel double expectations. Frozen source hashes remained unchanged for fitting.

## Gate, interpretation and next decision

Both bond variants pass the predeclared mean-TVD, worst-TVD and encoder-only-gain
criteria, but miss worst connected-bond RMSE <=.10. The screen therefore did not
trigger sampled-MMD training. Sampling and gradient **validation** above should
not be confused with training using those sampled estimates. No map, phase,
circuit, budget or threshold was tuned after seeing the outcomes.

The narrower scientific conclusion is positive: a spatially structured tractable
reference is implementable, and its combination with four-body IQP phases improves
shared joint fitting on these development conditions. It does not yet establish
uniformly accurate coupling reproduction, sampled training, or generalization.
The remaining issue is model/objective behavior in the low-coupling regime, not
whether a classical baseline wins. Retain the combined model as a candidate;
inspect the remaining correlation errors before a fresh confirmatory protocol.

Protocol, numerical outcomes and verification were captured in the linked JSON
and the project evidence log. Disposable runners/data/checkpoints were removed.
No canonical campaign or manuscript was modified by this ablation.

| Claim | Evidence | Figure/table | Limitation | Review status |
| --- | --- | --- | --- | --- |
| Bond input plus four-body phases improves shared fitting | Paired 2x2 exact-loss fits | Results table | Development data, finite budget, no fresh generalization | Exploratory; author review pending |
| Tractable reference sampler/ratio construction works numerically | Transfer sampler, amplitude, moment and gradient checks in JSON | Verification above | Tested sizes/range; no production integration or broad precision proof | Prototype verified |
| Coupling reproduction meets every provisional tolerance | Per-condition correlation diagnostics | Results table | Both bond variants fail worst-correlation threshold | Not established |
| Sampled training or new generalization succeeds | Not run | None | Subsequent experiment required | Unestablished |
