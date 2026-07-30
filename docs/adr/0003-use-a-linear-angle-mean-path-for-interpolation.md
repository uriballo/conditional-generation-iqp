# ADR 0003: Use a linear angle-to-mean path for interpolation

- Status: Accepted
- Date: 2026-07-30

## Context

The original experiment defines only two condition prototypes:
\(\alpha=0\) targets mean Hamming weight \(n/2\), and \(\alpha=\pi/4\) targets
\(n/4\). A held-out interpolation study needs a target distribution at every
intermediate angle. Without a rule, those intermediate labels are undefined
and any favorable mapping could be chosen after observing results.

## Decision

On \([0,\pi/4]\), define the target mean by the unique affine map through the
original endpoints:

\[
\mu(\alpha)=\frac n2-\frac{n\alpha}{\pi}.
\]

Keep the discrete-Gaussian width fixed at \(\sqrt n/2\), recalibrate its
truncated location at every angle, and keep probability uniform within each
Hamming-weight shell.

Train on five evenly spaced angles and test the four interleaved midpoints, as
specified in `docs/interpolation-design.md`.

## Consequences

- The original two targets remain unchanged.
- Every intermediate condition is fixed before training and has an
  interpretable monotone target mean.
- The study tests interpolation along one broadcast-angle path only.
- A successful result does not imply that another angle-to-target mapping,
  arbitrary per-qubit angles, or extrapolation outside the interval will
  generalize.
