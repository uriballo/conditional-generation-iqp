# Conditional IQP generation

Vocabulary for the experiments in this repository.  The manuscript lives
elsewhere; this file covers the model, the Gaussian studies, the metrics, and
the run structure.

## Model

**Conditioned IQP model**

A fixed IQP unitary receives a condition-dependent input state.  The trained IQP
phases do not change with the condition.

**Classically trainable**

The declared Pauli moments, the Hamming-kernel MMD estimator, and its parameter
gradients can be estimated classically from the chosen input family.  This does
not imply efficient classical sampling from the trained model output.

**Visible qubits and ancillas**

The model returns a distribution on `n` visible qubits and may use `r`
additional qubits that are marginalized after measurement.  Ancillas can add
expressivity and enrich how the condition enters the circuit.

**Complete two-local circuit**

The generator bank contains every one-body and distinct two-body diagonal Pauli
generator over the visible and ancillary qubits.

**Angle encoder**

A product-state input built from stacked single-qubit rotations, provided by
the library.  A single all-`Y` layer is the plain `RY` product state; the
studies use one layer with condition-dependent angles.

**Anti-phase two-block encoder**

The current choice for the one-parameter Gaussian study
(`experiments/common/encoders.py:anti_block_encoder`).  One all-`Y` layer
where even-indexed qubits get angle `pi c` and odd-indexed qubits `pi (1 - c)`,
so the condition sets the contrast between the two qubit halves while the
expected total excitation stays at `n/2`.  It uses no target statistics, and
with random initialization it reaches 0.088 held-out / 0.083 endpoints over
ten seeds at `r = 0`.

**Encoder arm**

One choice of condition-to-angle map, trained on the same circuit and data as
the other arms of a study.  The two-parameter study compares a two-block arm
with location and width channels against a single-channel broadcast ablation.

## Objectives and metrics

**Training MMD**

Training minimizes a multiscale Hamming-RBF MMD estimated from sampled character
operators.  MMD is an optimization loss, not the headline metric.

**Exact bitstring TVD**

Total variation distance between the exact visible bitstring distribution and
the target distribution of the same condition.  Computed from
`iqp_cond.simulation.probabilities`; this is the primary metric.

**Profile TVD**

TVD after aggregating bitstrings by Hamming weight.  It shows whether the model
captures the macroscopic shape but can hide errors within a shell.

**Moment errors**

The error in the mean weight divided by `n`, and the error in the weight
standard deviation divided by `sqrt(n)`.

**Codification gap**

The profile TVD between the encoder's sampling law and the target law at a
condition.  A zero gap would mean the learned circuit only has to undo a fixed
transformation, so the two-parameter benchmark keeps a positive gap.

## Benchmarks

**Gaussian Hamming-profile target**

A truncated discrete Gaussian over Hamming weight, uniform within each shell.
The scalar condition sets the mean weight, `n (0.15 + 0.70 c)`, with width
`0.5 sqrt(n)`.  This is the translated-family validity test.

**Two-parameter Gaussian target**

The same shell family with the prescribed mean and standard deviation both
varying with a two-dimensional condition `(u, v) = (mean / n, sd / sqrt(n))` on a
frozen 5x5 grid.  The 16 perimeter conditions train the shared circuit; the nine
interior conditions are held out.

**Two-block encoder map**

Affine maps send the two condition coordinates to a sum and a difference angle,
which are assigned to alternating qubits, so the width channel survives ancilla
marginalization.

## Data roles

**Training condition**

A condition whose samples enter optimization; validation samples at the same
conditions may select checkpoints.

**Held-out condition**

A condition excluded from training and checkpoint selection.  It tests
interpolation inside the training range.

**Matched seed**

One shared data and optimization replication across model comparisons, so paired
summaries average conditions within a seed.

## Runs

**Run config**

The frozen budgets and seeds of a run (`experiments/common/config.py`).  Its
canonical digest names the run directory.

**Run directory**

`results/<study>/<run_id>/` with `config.json`, `run.json` (paper and library
revisions), `metrics.json`, and `checkpoint.npz`.  `results/index.jsonl` indexes
completed runs.
