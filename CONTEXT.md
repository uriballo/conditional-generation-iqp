# Conditional IQP Generation

This project studies conditional generative distributions represented by IQP
circuits. The current publication context combines the immutable
`gaussian_foundations_v1` campaign with the completed
`curie_weiss_m5_foundations_v3` spin-flip campaign. The v2 matched-capacity
comparison is retained only as a historical diagnostic.
Earlier research lines are preserved in content-addressed archives and
versioned result stores but do not define the current manuscript vocabulary.

## Language

**Condition**:
An externally supplied scalar (c\in[0,1]) that selects a target output
distribution.
_Avoid_: Noisy state, diffusion timestep, class label unless it is genuinely
discrete

**Conditional generation**:
Learning or sampling a family of output laws (q_\theta(x\mid c)) indexed by a
Condition.
_Avoid_: Conditional denoising, reverse diffusion

**Fixed-circuit conditional generation**:
Conditional generation in which every Condition uses the identical compiled
IQP gate sequence, ordered generator bank, topology, and trained phase vector;
only the efficiently prepared tractable input state changes.
_Avoid_: Shared-parameter conditioning, Condition-programmed phases,
Condition-specific compilation

**Product-state-conditioned IQP model**:
The fixed-circuit model

\[
q_\theta(x\mid c)=\left|\langle x|H^{\otimes n}D_Z(\theta)H^{\otimes n}
\bigotimes_iR_Y[\alpha_i(c)]|0\rangle\right|^2.
\]

The Gaussian campaign uses one broadcast angle on every visible qubit.
_Avoid_: Trainable encoder, phase-conditioned circuit

The complete preparation-and-circuit pipeline is not itself called a strict
IQP circuit: product-state preparation precedes the fixed compiled IQP
unitary.

**Spin-flip input-state-conditioned IQP model**:
The Curie--Weiss v3 model uses a normalized two-branch input proportional to
\(|p(\beta)\rangle^{\otimes n}+X^{\otimes n}|p(\beta)\rangle^{\otimes n}\).
It preserves global spin-flip symmetry and supports exact reference sampling
and amplitude-ratio evaluation.
_Avoid_: Product-state Curie--Weiss v3 model, learned symmetry

**Notation (r) and (m)**:
(r) is the number of marginalized ancillas. (m) is the number of learned
IQP generators or phases.
_Avoid_: (m) for ancilla count

**Broadcast Gaussian Condition map**:
The frozen map \(\alpha_i(c)=\pi(0.35-0.70c)\), identical for every qubit and
fixed before campaign execution.
_Avoid_: Per-qubit learned features, target-metric-selected encoding

**Complete 2-local IQP ansatz**:
The generator bank containing all one-qubit and every distinct two-qubit
diagonal Pauli generator, with (n(n+1)/2) trainable phases.
_Avoid_: All-to-all layer without one-body generators

**Classically trainable**:
A narrow operational statement: the declared selected Pauli observables,
Hamming-kernel Conditional MMD estimator, and phase gradients can be estimated
classically from the declared tractable reference at the frozen budgets.
_Avoid_: Efficient classical output sampling, strong simulation, guaranteed
optimization, absence of barren plateaus

**Gaussian Hamming-profile Target**:
A distribution whose total probability over weight-(k) strings follows a
truncated discrete Gaussian and is uniform within each Hamming shell. Its
requested mean is \(n(0.15+0.70c)\); the untruncated location is recalibrated so
the truncated shell law has that exact mean.
_Avoid_: Gaussian distribution over bitstrings, uncalibrated location

**Width factor**:
The dimensionless value (s\in\{0.35,0.50,0.75\}) defining nominal target
width (s\sqrt n).
_Avoid_: Realized truncated standard deviation

**Profile TVD**:
Total variation distance between model and Target Hamming-shell mass vectors.
It is the primary task metric because the benchmark asks for Hamming-profile
control.
_Avoid_: Full-distribution TVD, fidelity

**Raw bitstring TVD**:
Total variation distance over all (2^n) strings. It reveals redistribution
within Hamming shells that Profile TVD intentionally quotients out and is a
principal limitation metric.
_Avoid_: Profile error

**Training Condition**:
A grid coordinate whose finite Target samples may enter optimization or
checkpoint selection. The main grid is \(\{0,1/4,1/2,3/4,1\}\).
_Avoid_: Seen test condition

**Validation data**:
An independent finite Target sample role at Training Conditions, used only for
minimum-Conditional-MMD checkpoint selection.
_Avoid_: Held-out interpolation data, exact Target evaluation

**Held-out interpolation Condition**:
One of the eight common odd-sixteenth coordinates strictly inside the Training
interval and absent from all Training grids. No finite sample at such a
coordinate enters training, validation, tuning, or checkpoint selection.
_Avoid_: Extrapolation condition, test sample used during selection

**Conditional generalization**:
Used once in the paper to mean interpolation to unseen Conditions strictly
inside the Training interval. The manuscript uses *interpolation* thereafter.
_Avoid_: Distribution-shift generalization, extrapolation

**No-condition control**:
A parameter-matched circuit trained across the same Target Conditions while
receiving the neutral (c=1/2) input for every target. It shares initialization,
data seed, and estimator streams with the Conditioned model.
_Avoid_: Unconditional dataset, untrained baseline

**Single-peak circuit**:
One independent circuit trained and frozen at exactly one Training Condition.
Its estimator stream is deterministically tagged by peak.
_Avoid_: Shared model, per-condition fine-tuning of the shared checkpoint

**Single-peak bank**:
The collection of independent Single-peak circuits over a declared Training
grid. Its stored parameters and optimization work grow linearly with the number
of peaks.
_Avoid_: Free oracle, parameter-matched shared model

**Target-aware adjacent lookup**:
At a Held-out coordinate, the more accurate of the two adjacent Single-peak
outputs, selected by Target Profile TVD. This is an intentionally generous
evaluation-time oracle.
_Avoid_: Target-blind interpolation, nearest-neighbor lookup

**Target-blind barycentric mixture**:
The convex mixture of the two adjacent Single-peak output distributions with
weights determined only by the Held-out coordinate. Target metrics never enter
its weights.
_Avoid_: Oracle mixture, metric-fitted interpolation

**Factorial Gaussian study**:
The (4\times3\times10) size–width–seed surface. Each cell contains one
Conditioned model, one No-condition control, and five Single-peak circuits,
for 840 atomic model identities.
_Avoid_: Pooled Gaussian benchmark

**Condition-count characterization**:
The nested (K\in\{2,3,5,9\}) study at (n=19,s=0.50), comparing denser
Training grids under Fixed-per-peak and Fixed-total data allocation.
_Avoid_: Hyperparameter sweep, extrapolation test

**Fixed-per-peak regime**:
Every Training Condition has 10,000 Training and 2,000 Validation strings, so
total finite data grows with (K).
_Avoid_: Fixed-data study

**Fixed-total regime**:
The 18,000 Training and 3,600 Validation totals are divided evenly across
Training Conditions using exact prefixes of the same target-specific master
pools.
_Avoid_: Independently redrawn reduced datasets

**Atomic model identity**:
One immutable training identity containing study, size, width factor, peak
count, data regime, family, optional bank-peak index, optimizer/data seed, and
manifest digest. The campaign declares exactly 1,020 identities.
_Avoid_: Evaluation coordinate, optimizer step, bank as one identity

**Paired seed effect**:
A method difference computed after averaging Conditions within one matched
optimizer/data seed. Ten such values form the replication units for the
deterministic paired bootstrap.
_Avoid_: Pooled Condition count, unpaired run average

**Evidence status**:
For a paired 95% interval, *supported* means the interval is strictly positive
in the declared favorable direction, *unsupported* means strictly negative,
and *mixed* means it contains zero. These labels describe evidence and are not
a publication gate.
_Avoid_: Passed cell, failed paper

**Immutable Gaussian campaign**:
The manifest-bound store under `results/gaussian-foundations-v1`, including
role-tagged datasets, resource preflight, reservations, result envelopes,
checkpoints, exact probabilities, report, and claim registry.
_Avoid_: Mutable results directory, reuse of an earlier campaign

**Canonical Gaussian PDF**:
The PDF promoted only after all 1,020 identities and evaluations complete and
the clean extracted-source verification succeeds. Promotion does not imply
upload readiness.
_Avoid_: Working PDF, upload-ready manuscript

**Curie--Weiss Target**:
The finite-size zero-field Gibbs law
\(p_\beta(x)\propto\exp[\beta(\sum_i s_i)^2/(2n)]\), with
\(s_i=2x_i-1\). It is invariant under qubit permutations and global spin
flip, and its condition is inverse temperature \(\beta\).
_Avoid_: Generic Ising target, spatial lattice model

**Symmetry-aligned conditioned IQP**:
The complete two-local IQP generator sequence with one learned coefficient
shared by all one-body phases and a second shared by all two-body phases. The
two coefficients expand to the full compiled sequence.
_Avoid_: Two-gate model, reduced topology

**Temperature reference**:
One independently trained symmetry-aligned circuit frozen at exactly one
Training temperature. It is a capacity reference, not a conditional model.
_Avoid_: Temperature-conditioned reference, shared checkpoint

**Curie--Weiss interpolation Condition**:
One of the five unseen midpoints between adjacent Training temperatures,
including the deliberately unseen \(\beta=1\) critical-point probe.
_Avoid_: Extrapolation, critical training point

**Immutable Curie--Weiss v3 campaign**:
The completed 80-fit store under `results/curie-weiss-m5-foundations-v3`.
It uses eight visible qubits, (r=5) marginalized ancillas, 91 complete
two-local phases, six Training temperatures, five held-out temperatures, and
the spin-flip input with calibrated knot map frozen before production.
_Avoid_: Generic linear-map campaign, product-input v3 campaign, running
multi-rung campaign

**Historical v2 matched-capacity diagnostic**:
An eight-qubit visible-only model with all one-, two-, and three-body diagonal
generators, totaling 92 learned phases. It is paired against the 91-phase
(r=5) product-input model using the v2 data, condition streams, optimizer,
checkpoint rule, and exact evaluation. It is not matched to the v3 spin-flip
model.
_Avoid_: v3 control, proof that ancillas or mixed visible states are necessary

**Author-review package**:
A local source-and-PDF bundle for coauthor inspection. It may contain draft
metadata but no upload script or submission action. The authors retain sole
responsibility for approval and submission.
_Avoid_: Upload-ready, submitted, arXiv release
