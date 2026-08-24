# PennyLane-Ising provenance and reconstruction audit

## Frozen source

The campaign follows the first-party PennyLane dataset generator, not a
generic square-lattice approximation.

- Dataset page: <https://pennylane.ai/datasets/ising>
- Dataset generator: <https://github.com/XanaduAI/gqml_datasets/blob/de96fc2647928c755d15e434c6c8293a0c5b3b60/generate_ising.py>
- Metropolis kernel: <https://github.com/XanaduAI/qml-benchmarks/blob/95e5a07e8e9e75ba7e24e67fb32b030112a1309a/src/qml_benchmarks/data/ising.py>
- Dataset release used for the reconstruction check: Ising version 0.1,
  updated 2026-04-23, CC BY-SA 4.0.

The public generator uses a periodic NetworkX grid, no local fields, directed
edge draws from NumPy's legacy random stream in \([0,2]\), and arithmetic
symmetrisation of the two directions. It resets the RNG to seed 666 for the
released \(4\times4\) instance. Sampling uses single-spin Metropolis proposals;
one kernel transition performs 16 proposals, followed by declared burn-in and
thinning.

## Reconstruction check

`iqp_cond.ising.periodic_lattice_couplings(666)` has SHA-256

```text
58cd62927e3be95a827e98efb7282af5a62d2232e15953da6362599017989194
```

The released 5,000 Training and 50,000 test strings at \(T=3\) were scored
under the exactly enumerated reconstructed Hamiltonian. Their observed mean
energy is -18.7718575999; the exact target value is -18.7697295208. The
difference is -0.00022065 exact energy standard deviations. This numerical
agreement protects the coupling ordering, energy sign, factor of one half,
bit-to-spin map, and \(\beta=1/T\) convention simultaneously.

The release itself is retained only as a provenance check. The conditional
campaign generates independent Training and Validation chains at every frozen
Training beta and uses exact enumeration, not the released finite strings, as
final distributional truth.

## Conditional-family extension

The source dataset contains only \(T=3\). The campaign preserves its Hamiltonian
prescription and evaluates the same exact energy function over a target-only
calibrated inverse-temperature grid. Seven additional coupling seeds are
frozen in the campaign manifest; they use the identical source construction.

At colder grid points, local Metropolis sweeps can remain in one of the two
zero-field ferromagnetic sectors. The campaign therefore adds an independent
global-spin-flip transition to each retained state. Its energy difference is
exactly zero because the Hamiltonian has no local field, so it is always
accepted and leaves the exact target invariant. This disclosed sampler-only
mixing step prevents a known symmetry sector from becoming apparent model
error; it does not alter target probabilities or model outputs.

## Finite-data schedule audit

Two preparatory pilots used eight independent chains and 2,000 burn-in sweeps.
Intervals of 10 and 16 thinning sweeps each missed the preregistered per-chain
energy-ESS threshold in 2 of 40 Validation cells, while their exact-reference
moment, correlation, and spectral-character checks passed. Those failed audits
are retained under `results/quantum-campaign-v1/sampler-attempts/`.

The manifest's frozen distributional check is the RMSE of the complete declared
bank of one- and two-body parity characters against exact enumeration, with
threshold 0.08. This is a finite Fourier discrepancy for the moments used
downstream. The plug-in empirical Bitstring TVD over all 65,536 states is
archived descriptively but is not a pass criterion: with only 7,000
Training-plus-Validation strings, its unseen-cell contribution is dominated by
finite-support undersampling rather than chain mixing. No full-distribution
certification is inferred from the character rule.

A conservative candidate with 4,000 burn-in and 32 thinning sweeps was then
specified, but its full audit was interrupted as soon as the predecessor
theorem-novelty gate failed. It is therefore an unaccepted protocol candidate,
not a certified campaign dataset. No model training consumed any of these
samples.
