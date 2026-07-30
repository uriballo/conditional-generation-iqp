# Conditional IQP Generation

This project studies conditional generative distributions represented by IQP
circuits. Its domain excludes reverse diffusion and denoising processes.

## Language

**Condition**:
An externally supplied value that selects the target output distribution.
_Avoid_: Noisy state, diffusion state, timestep

**Conditional generation**:
Learning or sampling an output distribution conditioned on a Condition,
written \(p(x \mid c)\).
_Avoid_: Conditional denoising, reverse diffusion

**Target distribution**:
The desired distribution over outputs for a particular Condition.
_Avoid_: Reverse target, posterior target

**Conditional IQP model**:
An IQP generative model whose Condition is represented by an initial
product-state encoding. All Conditions share one set of trainable circuit
parameters.
_Avoid_: Conditional diffusion model

**Tractable product-state encoding**:
A real product state
\(\bigotimes_i R_Y(\alpha_i)|0\rangle\) for which the sampling and amplitude
quantities required by the classical expectation estimator are efficiently
computable. The general family permits distinct angles on each qubit.
_Avoid_: Arbitrary tractable distribution, arbitrary tractable state

**Angle encoding**:
The primary Tractable product-state encoding, in which a Condition determines
the rotation angles of the initial product state.
_Avoid_: Soft basis encoding

**Basis encoding**:
A baseline encoding in which a binary Condition selects a computational-basis
product state. Its conditional distributions are restricted to XOR
translations of a shared base distribution.
_Avoid_: Primary encoding

**Interaction topology**:
The graph specifying which pairs of qubits receive two-qubit IQP generators.
Every Interaction topology also includes one-qubit IQP generators on all
qubits.
_Avoid_: Circuit condition, encoding topology

**Topology ablation**:
A controlled comparison of heavy-hex, lattice, and complete 2-local
Interaction topologies in which all non-topological experimental choices are
held fixed and every variant is reported. Because topology determines the
number of independent parameters, it compares native ansatz families rather
than isolating connectivity at fixed capacity.
_Avoid_: Ansatz selection

**Lattice topology**:
A near-square two-dimensional rectangular grid with horizontal and vertical
nearest-neighbor interactions.
_Avoid_: Chain topology, arbitrary grid

**Heavy-hex topology**:
An uncropped canonical heavy-hex interaction graph.
_Avoid_: Cropped heavy-hex graph

**Complete 2-local topology**:
The complete interaction graph, containing a two-qubit generator for every
pair of qubits.
_Avoid_: All-to-all layer

**Topology ansatz**:
An IQP ansatz containing one independently parameterized one-qubit generator
per qubit and one independently parameterized two-qubit generator per edge of
an Interaction topology.
_Avoid_: Layered ansatz, tied-parameter ansatz

**Conditional MMD**:
The equally weighted mean of unbiased per-Condition squared MMD estimates
between the model and its corresponding Target distribution. Estimates may be
negative because they are unbiased.
_Avoid_: Pooled MMD, sample-weighted conditional loss

**Target width**:
The standard deviation of a Gaussian Hamming-weight target.
_Avoid_: MMD bandwidth, kernel width

**MMD bandwidth**:
The scale parameter \(\sigma\) of the Hamming-RBF kernel
\(k_\sigma(x,y)=\exp[-d_H(x,y)/(2\sigma^2)]\) used by Conditional MMD.
_Avoid_: Target width

**Training set**:
A finite collection of condition-labeled samples from the Target
distributions used to estimate Conditional MMD.
_Avoid_: Analytic target, oracle distribution

**Validation set**:
A condition-balanced collection of target samples disjoint from the Training
set and used only for model selection and training diagnostics.
_Avoid_: Training batch, exact evaluation

**No-condition baseline**:
A shared model given the same Angle encoding for both Conditions.
_Avoid_: Unconditional target

**Basis-condition baseline**:
A shared model that replaces Angle encoding with two computational-basis
product states.
_Avoid_: Primary conditional model

**Separate-model reference**:
Two independently trained IQP models, one per Target distribution, used as a
capacity reference rather than as a conditional model.
_Avoid_: Conditional baseline

**Classically trainable**:
Having an efficient classical estimator for the Pauli-\(Z\) expectations
required by Conditional MMD training.
_Avoid_: Classically simulable, classically sampleable

**Bitstring fidelity**:
Agreement between model and Target distributions over complete bitstrings,
measured by total-variation distance.
_Avoid_: Weight fidelity

**Weight fidelity**:
Agreement between the model and Target probability masses aggregated by
Hamming-weight shell.
_Avoid_: Bitstring fidelity

**Condition prototype**:
The single fixed angle representing one Condition in the initial experiment,
broadcast across all qubits to preserve permutation symmetry.
_Avoid_: Condition range, encoding interval

**Conditional interpolation**:
Evaluation at an Angle encoding strictly between trained condition angles,
using a Target distribution fixed by the Angle-to-mean path.
_Avoid_: Extrapolation, unseen target family

**Angle-to-mean path**:
The affine interpolation
\(\mu(\alpha)=n/2-n\alpha/\pi\) on \([0,\pi/4]\), which assigns a Gaussian
Hamming-weight target mean to every broadcast encoding angle while preserving
the original endpoint Conditions.
_Avoid_: Learned target mapping, post-hoc target curve

**Held-out angle**:
An intermediate Angle encoding whose Target distribution is excluded from
training and checkpoint selection and used only for Conditional interpolation
evaluation.
_Avoid_: Validation condition, extrapolation angle

**Oracle adjacent-angle lookup**:
A memorization diagnostic that evaluates both training conditions adjacent to
a midpoint Held-out angle and retains the lower TVD for each metric. Beating
it requires doing better than either stored neighboring output.
_Avoid_: Continuous interpolation, nearest-angle interpolation

**Gaussian Hamming-weight target**:
A Target distribution whose probability mass over output Hamming-weight
shells has a discrete Gaussian shape. Conditional on a weight, probability is
uniform across all bitstrings in that shell. Its location is calibrated so
that its expected Hamming weight equals the prescribed condition-specific
mean despite truncation to valid weights.
_Avoid_: Binomial target, Gaussian bitstring
