# Conditional IQP generation

This file defines the vocabulary for the current manuscript and experiments.
It supersedes the terminology of the discarded foundations manuscript.

## Model

**Conditioned IQP model**

A fixed IQP unitary receives a condition-dependent input state. The trained IQP
phases do not change with the condition.

**Classically trainable**

The declared Pauli moments, Hamming-kernel MMD estimator, and parameter
gradients can be estimated classically from the chosen tractable input family.
This does not imply efficient classical sampling from the model output, strong
simulation, easy optimization, or universality.

**Visible qubits and ancillas**

The model returns a distribution on `n` visible qubits. It may use `r`
additional qubits that are marginalized after measurement. Ancillas can add
expressivity and can also enrich how the condition enters the circuit.

**Complete two-local circuit**

The IQP generator bank contains every one-body and distinct two-body diagonal
Pauli generator over the visible and ancillary qubits.

**Product-state encoder**

Each qubit receives a single-qubit state whose angle depends on the condition.
The Gaussian benchmark uses the same broadcast angle on every qubit.

**Spin-flip-symmetric encoder**

The Curie--Weiss informed encoder uses a two-branch input related by global bit
flip. It imposes the target's known global spin-flip symmetry but does not set
the target magnetization profile.

## Objectives and metrics

**Training MMD**

Training minimizes a multiscale Hamming-RBF MMD estimated from sampled
character operators. Three fixed bandwidths cover low-, intermediate-, and
higher-body correlations. MMD is an optimization loss, not the headline
performance metric.

**Exact raw TVD**

Total variation distance between the exact visible bitstring distribution and
the target. This is the primary metric.

**Profile TVD**

TVD after aggregating bitstrings by Hamming weight or magnetization. It shows
whether the model captures the intended macroscopic shape but can hide errors
within a shell.

**Ancilla effect**

The paired difference between a conditioned model with `r > 0` and the same
family at `r = 0`.

**Independent-model bank**

Separate circuits trained at each training condition. At a held-out condition,
the adjacent outputs are interpolated without using the held-out target. This
is the non-shared generalization baseline.

**Target-aware oracle**

An independent circuit trained directly at a held-out condition. It measures
single-condition capacity and is not a deployable interpolation baseline.

**Generalization advantage**

The interpolated independent-bank TVD minus the shared conditioned model TVD
at held-out conditions. Positive values favor the conditioned model.

**Sharing cost**

The conditioned model TVD minus the target-aware oracle TVD. Positive values
measure the price of representing several conditions with one parameter set.

## Benchmarks

**Gaussian Hamming-profile target**

A truncated discrete Gaussian over Hamming weight, uniform within each shell.
The campaign uses visible sizes 8, 12, and 16 and fixes the width to 0.5. This
is the simple validity test for a translated conditional family.

**Curie--Weiss target**

The finite-size zero-field Gibbs law
`p_beta(x) proportional to exp[beta (sum_i s_i)^2 / (2n)]`, where
`s_i = 2x_i - 1`. It is invariant under qubit permutations and global spin
flip. The inverse temperature crosses the finite-size transition near
`beta = 1`.

**Grid Ising pilot**

A spatial Ising target under evaluation as a third benchmark. It must supply a
conditional response and symmetry structure that differ materially from the
exchangeable Curie--Weiss example while remaining compatible with the
isotropic Hamming-kernel training objective.

## Data roles

**Training condition**

A condition whose training samples enter optimization. Validation samples at
the same conditions may select checkpoints.

**Held-out condition**

An interleaved condition excluded from initialization, training, tuning, and
checkpoint selection. It tests interpolation inside the training range.

**Matched seed**

One shared data and optimization replication used across model comparisons.
Statistical summaries average conditions within each seed before paired
bootstrap resampling.

**Campaign identity**

One immutable fit defined by dataset, visible size, ancilla count, model family,
condition where applicable, and seed.
