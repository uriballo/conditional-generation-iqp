# Symmetry-aligned full-distribution fidelity experiment

## Status and question

This protocol is frozen before evaluating the symmetry intervention. It asks
whether the gap between Hamming-weight profile TVD and Bitstring TVD is caused
primarily by nonuniform probabilities within Hamming-weight shells, and whether
averaging over the Gaussian Hamming-weight Target's permutation symmetry closes
that gap without weakening conditional control.

The compact manuscript at `paper/main.tex` remains frozen until the audit in
this document has been generated.

## Mechanism-level hypothesis

The broadcast Angle encoding and Gaussian Hamming-weight Target are invariant
under every permutation of the 19 output coordinates, but the independently
parameterized Complete 2-local topology need not produce a permutation-invariant
distribution. The hypothesis is that most of its residual Bitstring TVD is
therefore symmetry-breaking mass within otherwise approximately correct
Hamming-weight shells.

For a model distribution \(q(x\mid\alpha)\), define the permutation twirl

\[
q_{\mathrm{perm}}(x\mid\alpha)
=\frac{1}{|S_{19}|}\sum_{\pi\in S_{19}}
q(\pi^{-1}x\mid\alpha).
\]

Operationally, draw \(x\) from the unchanged Conditional IQP model, draw a
uniform random permutation \(\pi\), and return \(\pi x\). This preserves the
mass of every Hamming-weight shell exactly and makes the distribution uniform
within each shell. Consequently, against a Target that is uniform within each
shell, its exact Bitstring TVD equals its exact Hamming-weight profile TVD.

## Intervention and matched control

Three protocols use each frozen Complete Angle-conditioned checkpoint:

1. **Unaligned:** return the model bitstring unchanged.
2. **Permutation twirl:** apply one uniformly sampled output-coordinate
   permutation to each generated bitstring. Exact evaluation uses the equivalent
   shell-uniform distribution, not Monte Carlo permutation samples.
3. **XOR-pair control:** with probability one half return the bitstring unchanged
   and otherwise XOR it with the fixed alternating mask
   \((1,0,1,0,\ldots,1)\) in the project's little-endian coordinate order.
   Exact evaluation averages the original distribution with its XOR-reindexed
   copy. This is a parameter-free, Condition-independent random output action
   with the same asymptotic generation and exact-evaluation cost, but it does
   not project onto the Target's permutation-invariant distributions.

The primary intervention is the permutation twirl. The XOR-pair protocol is a
mechanism control, not a proposed generator. Its mask is fixed here and is not
selected from results.

The intervention is applied only after computational-basis measurement. The
Condition still enters only through the initial Tractable product-state
encoding, all Conditions retain the same 190 trainable parameters, and the
classical Pauli-observable estimator and efficient-sampling claim boundary are
unchanged.

## Capacity and compute budget

| Protocol | Trainable parameters | Training updates | Checkpoint rule | Random output actions per sample | Exact-evaluation cost |
|---|---:|---:|---|---:|---:|
| Unaligned | 190 shared | 2,000 | Frozen Validation minimum | 0 | \(O(2^{19})\) |
| Permutation twirl | 190 shared | 2,000 | Same frozen checkpoint | 1 | \(O(2^{19}+19)\) |
| XOR-pair control | 190 shared | 2,000 | Same frozen checkpoint | 1 | \(O(2^{19})\) |

No protocol is retrained. All three reuse identical parameters, Training data,
Validation data, optimizer histories, estimator budgets, and checkpoints. At
generation time both randomized protocols require one Condition-independent
random action and \(O(19)\) bit operations. At exact evaluation time both are
linear in the enumerated state-space size.

## Split, seeds, and checkpoint selection

The endpoint study reuses `results/runs/angle_complete_seed{0..9}.npz`, trained
on Conditions \(\alpha=0\) and \(\alpha=\pi/4\). The Held-out study reuses
`results/interpolation/runs/interpolation_complete_seed{0..9}.npz`, trained on
the five angles in `docs/interpolation-design.md` and evaluated at its four
interleaved Held-out angles.

- Training sets remain the original 10,000 samples per Training Condition.
- Validation sets remain the original disjoint 2,000 samples per Training
  Condition and are used only by the already-frozen checkpoint rule.
- Analytic Target probabilities are used only for exact evaluation.
- Held-out-angle Targets never enter optimization or checkpoint selection.
- Seeds 0--9 are matched across all three protocols because each protocol is
  evaluated from the same checkpoint.
- The original 2,000-update Adam schedule and lowest-Validation-Conditional-MMD
  checkpoint are unchanged. There is no new early stopping or model selection.

## Metrics

The primary metric is exact Bitstring TVD, reported per endpoint Condition and
per Held-out angle. For every protocol, summaries report the median and
interquartile range over ten matched seeds. The per-seed reduction relative to
the unaligned model is also retained in machine-readable output.

Guardrail metrics are:

- exact Hamming-weight profile TVD;
- exact conditional Hamming-weight mean and standard deviation errors;
- the endpoint audit against the frozen No-condition and Basis-condition
  controls;
- at Held-out angles, the dense-grid comparison against endpoint-only
  interpolation and Oracle adjacent-angle lookup, with the same output protocol
  applied to both sides of each comparison;
- normalization, nonnegativity, shell-mass preservation, and the identity
  \(\mathrm{TVD}(p,q_{\mathrm{perm}})=
  \mathrm{TVD}(p_K,q_K)\) for shell-uniform Targets.

The stochastic construction does not require estimator recalibration because
training is unchanged and exact evaluation analytically marginalizes the
output action. Sampling calibration is therefore out of scope for the primary
numbers; unit tests verify that empirical permutation averaging converges to
the exact twirl on a small system.

## Frozen decision rule

The symmetry hypothesis passes only if all of the following hold:

1. At each endpoint Condition and each of the four Held-out angles, the median
   permutation-twirled Bitstring TVD is at least **0.15 lower in absolute TVD**
   and at least **40% lower relatively** than the unaligned median.
2. At every evaluated Condition, median Hamming-weight profile TVD degrades by
   no more than **0.01 absolute**, median mean absolute error by no more than
   **0.05 Hamming-weight units**, and median standard-deviation absolute error
   by no more than **0.05 Hamming-weight units**. Exact invariance is expected;
   these tolerances only guard implementation and numerical errors.
3. Applying the permutation twirl introduces no new failed cell in either the
   strict endpoint control audit or the Held-out dense-versus-baseline audits.
   Existing failures remain failures and cannot be hidden by aggregation.
4. The XOR-pair control does not meet the complete primary success rule. If it
   does, the result supports generic output randomization rather than the
   proposed permutation-symmetry mechanism.

The 0.15 and 40% thresholds define a material change relative to the already
reported 0.275--0.352 endpoint and approximately 0.35 Held-out Bitstring TVDs;
they are fixed before computing any twirled full-matrix result.

## Exact evaluation and persistence

The evaluator loads a saved checkpoint, recomputes the exact model
probabilities at the declared Condition angles, applies each output protocol,
and writes one independently resumable JSON file per study, seed, and protocol.
Each record contains source checkpoint identity, parameter count, protocol,
Condition, raw and transformed metrics, and the frozen threshold values.
Summarization refuses incomplete seed sets and writes both Markdown and JSON
audits, including the capacity/budget table and per-seed medians/IQRs.

## Falsification and interpretation

The mechanism-level hypothesis is falsified if the permutation twirl misses
the material Bitstring-TVD threshold at any endpoint or Held-out Condition,
damages a guardrail beyond tolerance, creates a new conditional-control audit
failure, or is not specific relative to the XOR-pair control. A failure would
show that the fidelity gap is not resolved by Target-symmetry projection alone
under the frozen protocol.

A pass establishes only that a parameter-free symmetry-aligned generation
protocol resolves most within-shell error for this permutation-invariant
Gaussian Hamming-weight family and broadcast Angle-to-mean path. It does not
show that the underlying untwirled circuit learned uniform within-shell mass,
that arbitrary Target symmetries can be averaged efficiently, or that sampling
from the trained IQP circuit is classically efficient.
