# Two-ring bond reference: construction and scope

The fixed encoder prepares two independent zero-field Ising rings, each with
m sites, coupling g in [.1,.7], and fixed phase eta=pi/8:

B_g(x) = exp[(g/2 + i eta) sum_i s_i s_(i+1)] / sqrt(Z_g),
where s_i=1-2x_i and Z_g=(2 cosh g)^m+(2 sinh g)^m.

The input supplies nearest-neighbor correlations within each layer. It supplies
no interlayer coupling: the connected cross-layer bond correlations are zero.
The condition changes g, while one trained IQP phase vector is shared across
all conditions. The implementation uses two equal rings with at least three
sites each; eight qubits is the present training setting. The public estimator's
legacy `angles` array stores g_A in its first half and g_B in its second half
for the explicit `two_ring_bond` encoding. These entries are couplings, not
rotation angles. The reference module itself accepts the two couplings directly.

## Reference access

Let T_xy=exp[(g/2+i eta)(1-2x)(1-2y)] and
A_z=diag(1,(-1)^z) T. The post-Hadamard amplitude is

phi_g(z) = trace(A_z0 ... A_z(m-1)) / sqrt(2^m Z_g).

This follows by expanding the matrix trace over cyclic physical spins. Global
spin flip leaves B_g unchanged, so phi_g is exactly zero at odd parity. For two
rings the amplitude factorizes. The implementation enforces these zeros rather
than dividing by numerical leakage outside the support.

Define E_z=A_z tensor conjugate(A_z), and S=E_0+E_1. The unnormalized probability
of a prefix z_0...z_(k-1) is trace(E_z0 ... E_z(k-1) S^(m-k)). The two possible next
prefix weights give the conditional Bernoulli probability. Sequential sampling
therefore requires 4 by 4 contractions, not a table over 2^m strings. Common
positive rescaling of suffixes and prefixes cancels in the conditional ratio.
Amplitudes similarly use scaled products and a logarithmic normalization.
For ratios, the common partition/Hadamard normalizers cancel. The ratio path
divides T by its absolute row sum, 2 cosh(g/2), then contracts these bounded
matrices directly. This avoids per-query rescaling/logarithms in training;
small-denominator cancellation remains a finite-precision limitation. Tiny
2 by 2 and batched 4 by 4 contractions use explicit elementwise sums to avoid
backend overhead from many small matrix products.

For fixed matrix dimension, an amplitude query costs O(m) arithmetic; suffix
preparation costs O(m), followed by O(m) work per independent reference draw.
Batch implementation memory also depends on the number of simultaneous queries
and masks. This is an arithmetic complexity statement, not a bound on arbitrary
precision: cancellation can affect very small amplitudes. Tests cover small
rings against enumeration and a 32-site numerical smoke check. This does not
establish uniform floating-point accuracy at arbitrary size or coupling.

## IQP moments and phase gradients

Write f_theta(z)=sum_j theta_j chi_j(z). With z sampled from |phi(z)|^2,

W_a(z)=Re[(phi(z xor a)/phi(z)) exp(i[f_theta(z xor a)-f_theta(z)])]

has expectation equal to the output Pauli-Z moment for mask a. A zero numerator
is legitimate; a sampled zero denominator is an error. The second moment of
the complex ratio estimator is at most one, since its weighted squared modulus
sums |phi(z xor a)|^2 over the sampled support. Differentiation in theta multiplies
the complex integrand by i[chi_j(z xor a)-chi_j(z)], whose modulus is at most two.
The encoder is fixed, so neither its sampling distribution nor its ratios need
to be differentiated. No encoder-gradient estimator is claimed.

The existing MMD estimator uses the off-diagonal sample-pair average for the
squared model moment, and the independent target-sample average for its target
terms. Mask sampling gives an unbiased population-kernel estimator when averaged
over the random mask bank. Freezing a finite mask bank yields its empirical
weighted objective; it is not identical to the full Fourier objective in each
individual run. Samples may be shared across masks without changing this
unbiasedness, though this changes covariance.

Both physical layer flips commute with the IQP unitary and fix the input.
Consequently the output has both symmetries for every parameter vector.
The unrestricted phase bank does not enforce ring translations/reflections or
exchange of the layers; those additional target symmetries are not claimed as
architectural guarantees. At four
sites per ring, identical generator characters on the even-even reference
support reduce the complete one/two-body bank from 36 to 30 independent phases.
Adding the original parameters belonging to each identical character preserves
the unitary action on this support. Both encoder comparisons use this same bank.

## Concrete quantum input preparation

Classical reference sampling is not itself a quantum preparation circuit. A
separate constructive preparation of B_g is available as follows. Let the
positive transfer matrix V_xy=exp[g(1-2x)(1-2y)]. Prepare the first spin in |+>.
For each new site i=1,...,m-1, apply a rotation to its initially zero qubit with

P(x_i=b | x_0=a, x_(i-1)=u)
 = V_ub (V^(m-i))_ba / (V^(m-i+1))_ua.

Choose RY angle 2 asin(sqrt(P(b=1))). For i=1 the first and previous spins are
the same control; later rotations are multiplexed by the first and previous
spins. The transfer suffix includes the closing bond. The conditional factors
telescope to the normalized Ising law with P(x_0)=1/2, preparing its positive
square-root amplitudes coherently. Finish with the m commuting diagonal gates
exp(i eta Z_i Z_(i+1)). Repeat independently for the second ring.

This uses O(m) bounded-control rotations and O(m) ZZ gates, with no additional
logical ancilla. It is a preparation construction and verified amplitude
identity, not an executed hardware demonstration. Gate synthesis precision,
hardware connectivity and routing costs are separate. Keeping the first spin
as a control at every step can require nonlocal gates on a nearest-neighbor
layout; the O(m) count assumes access to those bounded-control operations.

## Evidence and limitation

`tests/test_bond_reference.py` checks reference access;
`tests/test_bond_estimator.py` checks independent physical-state simulation,
ratio moments, phase gradients, both flips, and phase-bank equivalence.
`tests/test_bond_preparation.py` verifies the coherent preparation probabilities.
Stage-2 development results are separate evidence for optimization. Neither
these identities nor a successful optimizer establishes quantum advantage,
efficient output sampling, or generalization to unseen conditions.
