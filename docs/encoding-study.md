# Encoding-complexity study

## Question

Does a more complex but still classically tractable condition path improve
conditional interpolation for the shared complete-graph IQP model?

The first comparison changes product-state geometry without adding trainable
parameters or changing the two endpoint states. It therefore isolates
interpolation geometry from endpoint distinguishability.

## Encoders

The baseline local encoder is

\[
u_{\mathrm{angle}}(a)=R_Y(a).
\]

The first noncommuting reupload encoder is

\[
u_{\mathrm{reupload}}(a)
=R_Y(a/2)R_Z\!\left[\frac{\pi}{2}\sin(4a)\right]R_Y(a/2),
\]

where operators are listed in circuit application order. The middle phase is
a fixed harmonic feature, not a trainable coefficient. It vanishes at
\(a=0\) and \(a=\pi/4\), so both encoders prepare exactly the same states at
the original endpoint conditions. At intermediate angles it produces a
complex, noncommuting path. Repeating only \(R_Y\) would not be a valid richer
control because the rotations collapse to one effective angle.

Both encoders remain products over qubits. Sampling from the post-Hadamard
reference distribution and evaluating complex amplitude ratios cost
\(O(n)\). Only the shared diagonal IQP parameters are optimized, so ordinary
backpropagation through the MMD estimator remains valid.

## Controlled comparison

- complete two-local IQP ansatz, including all single-qubit generators;
- the existing five training and four interleaved held-out angles;
- identical targets, target samples, optimizer, estimator sizes, and seeds;
- 2,000 optimization steps with validation selection;
- exact 19-qubit shell and bitstring TVD evaluation;
- paired seed differences, with lower values preferred.

The first screen uses seeds 0--2. It is exploratory until all ten preregistered
interpolation seeds are complete. A useful outcome need not be an overall win:
a shell/bitstring tradeoff would indicate which distributional structure the
extra phase geometry changes.

## Next rung

If a product path is promising, test two-qubit block-product encoders with
fixed \(ZZ\) strengths \(\pi/8\) and \(\pi/4\). Use nine pairs plus one
singleton, rotate the singleton and pairings across matched seeds, and retain
exact categorical block sampling and amplitude-ratio lookup. This is the first
rung that adds genuine reference-state correlations.

Only test bounded-bond-dimension MPS encodings if the two-qubit blocks improve
the matched comparison. The wider rationale and source review are in
`docs/encoding-family-research.md`.
