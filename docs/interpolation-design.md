# Held-out angle interpolation experiment

## Question

Does the product-state condition define a usable conditional geometry between
the two original prototypes, or does the circuit merely memorize angles seen
during training?

## Target path

For \(n=19\) and a broadcast angle \(\alpha\in[0,\pi/4]\), the target remains a
discrete Gaussian over Hamming-weight shells, uniform within each shell, with
width \(\sqrt{n}/2\). Its truncated location is calibrated so that

\[
\mathbb E[K\mid\alpha]
= \frac n2-\frac{n\alpha}{\pi}.
\]

This preserves the original endpoints:
\(\mathbb E[K\mid0]=n/2\) and
\(\mathbb E[K\mid\pi/4]=n/4\).

## Split

- Training angles:
  \(0,\pi/16,\pi/8,3\pi/16,\pi/4\).
- Held-out angles:
  \(\pi/32,3\pi/32,5\pi/32,7\pi/32\).
- Each held-out angle is the midpoint of two adjacent training angles.
- Training uses 10,000 samples and validation uses 2,000 disjoint samples per
  training angle.
- Held-out samples do not enter optimization or checkpoint selection.

The model is the 190-parameter complete ansatz with the frozen optimizer,
kernel bandwidths, estimator budgets, update count, and ten matched seeds from
the primary experiment.

## Comparisons

1. **Dense-grid interpolation:** evaluate the five-angle model at the exact
   held-out input angle.
2. **Oracle adjacent-angle lookup:** because every test angle is exactly
   equidistant from two training angles, evaluate both adjacent stored
   conditions and retain the lower TVD for each metric. This deliberately
   strengthens the memorization baseline; dense interpolation must beat both
   neighboring outputs.
3. **Endpoint-only interpolation:** evaluate the original model trained only at
   \(0\) and \(\pi/4\) at the exact held-out angle. This tests whether denser
   conditional supervision improves interpolation.

## Metrics and frozen decision rule

Exact enumeration over all \(2^{19}\) bitstrings provides Hamming-weight profile TVD, bitstring
TVD, mean error, and standard-deviation error. Held-out Conditional MMD is a
secondary diagnostic.

Hamming-weight profile and bitstring resolution are audited separately:

- **Continuous-conditioning test:** dense-grid interpolation must have lower
  median TVD than oracle adjacent-angle lookup at every held-out angle.
- **Dense-supervision test:** the per-seed mean over held-out angles must have
  lower median TVD than endpoint-only interpolation.

Both tests are reported independently for Hamming-weight profile and bitstring TVD. Failure on
one resolution is not overridden by success on the other or by an aggregate
over both metrics.

## Claim boundary

Success would establish interpolation only along this one-dimensional,
broadcast-angle path between the two Gaussian targets. It would not establish
extrapolation, arbitrary per-qubit conditioning, or generalization to another
target family.
