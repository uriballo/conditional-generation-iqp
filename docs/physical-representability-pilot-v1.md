# Preliminary physical-data representability pilot

Date: 2026-08-13  
Status: exploratory evidence; not manuscript evidence  
Immutable result: `results/physical-representability-pilot-v1/results.json`  
Result SHA-256: `467e197316e2cec8cd7d2d9f27d266c41a0447ae2c998db3823e17ecf338e5e5`

## Question and interpretation

Before training one conditional model, this pilot asks whether an independently
optimized complete two-local IQP circuit can reproduce representative members
of each proposed target family. The screen is deliberately optimistic: it uses
the exact target law, exact population Hamming-RBF MMD, exact gradients, exact
output probabilities, 2,000 Adam updates, and three random initializations.
There is no finite-data or random-estimator noise.

A failure is evidence against proceeding to conditional training with the
current architecture and objective. A success establishes only a necessary
condition; it does not show that one shared circuit can learn or interpolate a
conditional family.

## Protocol

- Complete two-local IQP circuit with one- and two-body phases.
- Standard zero-angle IQP product input for every independently trained model.
- Hamming-RBF bandwidths `(1, 2, 4)`.
- Adam with cosine decay `0.01 -> 0.001` for 2,000 updates.
- Checkpoint selected by the lowest exact population MMD every 25 updates.
- Three optimizer initializations per target.
- Exact bitstring TVD is the primary representability diagnostic.
- Three representative physical coordinates are screened per family.

The incoming-energy targets use the public Muonic Force Carrier tensors archived
by Galon et al. at Zenodo record 6768686, as cited by Kiss et al. The screened
variable is the same 16-bin outgoing-muon-energy marginal used for their
one-dimensional demonstration.

## Results

| Dataset | Physical coordinate | Best TVD | Median TVD | Worst TVD | Assessment |
|---|---:|---:|---:|---:|---|
| Ferromagnetic spin law | beta = 0.20 | 0.5425 | 0.5478 | 0.5478 | unsupported |
| Ferromagnetic spin law | beta = 0.4406868 | 0.4691 | 0.4691 | 0.4698 | unsupported |
| Ferromagnetic spin law | beta = 0.70 | 0.4971 | 0.6284 | 0.6284 | unsupported |
| Attractive lattice gas | mu = -2.5 | 0.0038 | 0.0041 | 0.0042 | supported endpoint |
| Attractive lattice gas | mu = -2.0 | 0.4753 | 0.4975 | 0.4975 | coexistence unsupported |
| Attractive lattice gas | mu = -1.5 | 0.5130 | 0.6666 | 0.6666 | optimizer failure; representable by symmetry |
| MFC outgoing energy | 50 GeV | 0.0307 | 0.0463 | 0.0811 | supported |
| MFC outgoing energy | 125 GeV | 0.0313 | 0.0428 | 0.1065 | supported |
| MFC outgoing energy | 200 GeV | 0.0306 | 0.0470 | 0.1046 | supported |

The best MFC fits also reproduce the first two scaled-energy moments to absolute
errors below `0.0014`. The critical ferromagnetic fit underestimates absolute
magnetization by `0.239` and errs in energy per bond by `0.283`. The best
lattice-gas coexistence fit errs in occupied-pair fraction by `0.097` despite a
small density error, showing that density alone would hide a serious structural
failure.

## Diagnostics and caveats

### Hamming MMD can conceal poor bitstring fidelity

At beta `0.20`, the ferromagnetic fit reaches exact multiscale MMD-squared
`0.00264` while its bitstring TVD is `0.542`. The isotropic loss therefore does
not supply a reliable spatial-correlation training signal for this target. This
is the anticipated kernel-geometry limitation, now observed directly rather
than assumed.

### Separate optimization failure is not always non-representability

At beta `2` and attraction `1`, the lattice-gas laws at chemical potentials
`-2.5` and `-1.5` are exact particle-hole complements. Adding `pi/2` to every
one-body IQP phase transforms the learned sparse-endpoint distribution into its
bitwise complement. The transformed dense-endpoint TVD is exactly `0.00385` up
to numerical precision. Thus, the dense endpoint is representable even though
all three random-start fits fail. The failure is optimization, which would
still make a shared conditional campaign risky.

The coexistence law at `mu=-2` is not rescued by this symmetry. A supplementary
broadcast-input sweep at angle fractions `1/8`, `1/4`, `3/8`, and `1/2` of pi
also failed to improve the hard cases: the best tested nonzero-angle TVDs were
`0.528` for the critical ferromagnet and `0.606` for coexistence.

### The MFC screen is narrower

The MFC target is a four-qubit, 16-bin marginal, not the full multivariate event
record. Its successful screen supports an incoming-energy-conditioned marginal
experiment. It does not establish representability of the joint outgoing
energy, transverse-momentum, and pseudorapidity distribution. A faithful joint
encoding would also require a kernel whose geometry respects the physical
coordinates rather than raw binary place value.

## Decision

1. **Proceed first with incoming-energy conditioning of the 16-bin outgoing
   energy marginal.** Each screened target is individually learnable, including
   the held-out energy used in the closest prior conditional-QCBM study.
2. **Do not proceed with temperature-conditioned ferromagnetic configurations
   under the current Hamming-MMD protocol.** Individual-target learning already
   fails, and physical observables confirm the failure.
3. **Do not use the attractive lattice gas as the next headline experiment.**
   The single-phase endpoints are representable, but the physically interesting
   coexistence region is not learned and random-start optimization cannot reach
   even the symmetry-related dense solution.

The next experiment should train one product-state-conditioned fixed IQP circuit
on MFC energies `50, 75, 100, 150, 175, 200` GeV and reserve `125` GeV for
within-range interpolation, matching the closest prior study's split. It should
compare against independently trained per-energy references and an unconditioned
model, and it should retain exact 16-bin TVD as the primary metric.
