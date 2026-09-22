# Adversarial review of protocol v0.1

> **Verdict:** reject and redesign the protocol; retain the benchmark concept
>
> **Scope:** scientific and protocol review only; no trained-model experiments

The review assessed construct validity, encoder leakage, target selection,
comparators, statistics, audit integrity, transfer, and claim scope.

## Blocking findings

1. **The proposed target search was empty.** None of its 27 ranges passed the
   endpoint-axis balance rule. The smallest balance ratio was approximately
   1.95, above the threshold of 1.5.
2. **The centered encoder was degenerate on the complete hollow cross.** At
   every training point, one coordinate block became $|0\rangle$ after the
   first Hadamard. The corresponding output wires were uniform, and a
   cross-group parameter subspace was unidentifiable from all training
   distributions while still changing off-axis outputs.
3. **The hollow cross lacked a common baseline and compulsory training-fit
   gates.** Off-axis success could not establish that the circuit had first
   learned the two axis effects.
4. **The additive classical reference was undefined.** The draft specified no
   feature basis, identifiability convention, pseudocount, regularization, or
   optimizer.
5. **The rotated encoder lacked matched ablations.** Aligned missing-channel
   controls could not support causal gates for the rotated candidate.
6. **Several gates lacked machine-evaluable definitions.** Missing details
   included normalization, aggregation, failed-run encoding, bootstrap
   procedure, schedule-screen contingencies, and relative-improvement formulas.

## Major findings

- The claim had to be narrowed to held-out combination generalization under a
  supplied factorized representation.
- Target selection could not depend on the centered encoder or classical
  reference.
- The aligned encoder's central zero-parameter variance nearly matched the
  target and required explicit disclosure and encoder-only gates.
- The coordinate rotation did not preserve the complete product-state
  geometry and could not be interpreted as a single-factor causal comparison.
- A failure-only individual-fit and all-condition capacity reference was
  needed.
- The $n=12$ experiment was a strict-transfer test rather than a replication.
- The new protocol required a dedicated harness because the existing code
  retained perimeter splitting, old encoders, conflated randomness, and
  restart selection.

## Invariant-control decision

The reviewer rejected headline eligibility for the centered encoder. Its
off-axis behavior can depend on parameters that the training cross cannot
identify. It remains useful only as a structural negative control.

## Target-only follow-up

A revised finite search evaluated 40 ranges. Two were rejected because the
requested moments were infeasible at extreme audit points; 26 passed the
intrinsic boundary, adjacency, and axis-balance criteria. The deterministic
ranking selected

\[
(a,b,d)=(0.06,0.35,0.125).
\]

Its endpoint TVDs are $D_\mu=0.348781$ and $D_\sigma=0.353324$, its maximum
boundary mass is 0.011319, and its adjacent TVDs range from 0.055739 to
0.163627. The central target lies 0.175938 TVD from the uniform distribution.

## Disposition

Protocol v0.2 must restore the full axis cross, demote the centered construction,
add matched rotated ablations, freeze a feasible intrinsic target range, define
all estimands and failure rules, add training and midpoint gates, and undergo a
second adversarial review before implementation.
