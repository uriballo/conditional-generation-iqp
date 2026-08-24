# Sparse-grid interpolation with ancillas

## Question

The dense five-angle study trains at every other \(\pi/32\) grid point and
evaluates four interleaved midpoints. It established that five-angle
supervision beats endpoint-only supervision at \(m=0\). This experiment asks
two follow-up questions:

1. Does the model interpolate between trained peak positions when between-peak
   supervision is sparse or absent, at every tested ancilla count?
2. Does adding between-peak supervision improve interpolation at a matched
   ancilla count, and does the ancilla benefit persist at sparse supervision?

## Target path

Identical to the dense study: for \(n=19\) and a broadcast angle
\(\alpha\in[0,\pi/4]\), the target is a discrete Gaussian over Hamming-weight
shells, uniform within each shell, with width \(\sqrt{n}/2\) and truncated
location calibrated so that

\[
\mathbb E[K\mid\alpha]
= \frac n2-\frac{n\alpha}{\pi}.
\]

The endpoints are preserved: \(\mathbb E[K\mid0]=n/2\) and
\(\mathbb E[K\mid\pi/4]=n/4\).

## Grids

- **3-peak grid:** training angles \(0,\pi/8,\pi/4\) (fractions
  \(0,4/32,8/32\)); held-out angles \(\pi/16,3\pi/16\) (fractions
  \(2/32,6/32\)), each the midpoint of two adjacent training angles.
- **2-peak grid:** training angles \(0,\pi/4\) (fractions \(0,8/32\));
  held-out angles \(\pi/16,\pi/8,3\pi/16\) (fractions \(2/32,4/32,6/32\)).
  \(\pi/8\) is the true midpoint; the other two held-out angles are closer to
  one endpoint and test interpolation away from the center.
- Training uses 10,000 samples and validation uses 2,000 disjoint samples per
  training angle, with the same seed offsets as the dense study. Held-out
  samples do not enter optimization or checkpoint selection.

The two grids share the held-out angles \(\pi/16\) and \(3\pi/16\), which
enables a matched dense-supervision comparison: the 3-peak model and the
2-peak model trained with the same seed and ancilla count are evaluated at
identical held-out targets.

## Model and protocol

- \(n=19\) data qubits plus \(m\in\{0,2,3,5,7\}\) ancilla qubits; complete
  two-local ansatz on \(N=19+m\) qubits; ancillas are traced out at evaluation.
- Ancilla conditioning scheme: `broadcast_all` (ancillas receive the same
  broadcast angle as the data qubits), matching the existing ancilla study.
- The frozen optimizer, kernel bandwidths, estimator budgets, update count
  (2,000 steps), checkpoint selection, and ten matched seeds from the primary
  experiment are reused without tuning.
- Exact evaluation marginalizes the \(2^{19+m}\) joint probabilities down to
  \(2^{19}\) data probabilities.

## Comparisons

For every ancilla count \(m\):

1. **Interpolated evaluation:** evaluate the trained model at each exact
   held-out input angle.
2. **Oracle adjacent-angle lookup:** for each held-out angle, evaluate the
   stored model outputs at the adjacent training angles and retain the lower
   TVD for each metric. Beating it requires doing better than either stored
   neighboring output.
3. **Matched dense supervision:** the 3-peak grid versus the 2-peak grid at
   the two shared held-out angles, same seed and same \(m\).

The \(m=0\) rows of the 2-peak grid reuse the existing endpoint-only primary
model (`results/runs/angle_complete_seed0-9.npz`); all other rows are trained
in this experiment.

## Metrics and frozen decision rules

Exact enumeration provides Hamming-weight profile TVD, bitstring TVD, mean
error, and standard-deviation error for every evaluated angle. Both TVD
resolutions are audited separately; a pass on one resolution does not override
a failure on the other.

- **Continuous-conditioning test:** for each grid, ancilla count,
  resolution, and held-out angle, the median interpolated TVD must be lower
  than the median oracle adjacent-angle lookup TVD.
- **Matched dense-supervision test:** for each ancilla count and resolution,
  the median 3-peak held-out TVD over the shared held-out angles must be lower
  than the median 2-peak held-out TVD.
- **Ancilla-scaling report:** held-out TVD versus \(m\) is reported
  descriptively per grid, without a frozen pass/fail rule.

## Claim boundary

Success would establish interpolation along the one-dimensional broadcast-angle
path between Gaussian Hamming-weight targets at sparse supervision, and the
effect of ancillas on that capacity. It would not establish extrapolation,
arbitrary per-qubit conditioning, or generalization to another target family.
