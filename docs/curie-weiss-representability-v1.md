# Curie--Weiss single-circuit representability screen

Date: 2026-08-13  
Status: preliminary; not manuscript evidence

## Question

Can one independently trained, regular 16-qubit complete two-local IQP circuit
learn representative finite Curie--Weiss laws before any conditional model is
attempted?

The target is

\[
H(s)=-\frac{J}{2n}\left(\sum_i s_i\right)^2,
\qquad
p_\beta(s)=\frac{e^{-\beta H(s)}}{Z(\beta)},
\]

with `n=16`, `J=1`, and zero field. The thermodynamic critical scale is
`beta_c=1`. We screen `beta=0.5, 1.0, 1.5`, corresponding to disordered,
critical, and ordered regimes.

This is not the spatial nearest-neighbour ferromagnet screened previously. The
Curie--Weiss law depends only on total magnetization and is exactly
permutation-symmetric, so Hamming geometry is aligned with the physical target.

## Primary training screen

Each target was fit independently using:

- one ordinary complete two-local IQP circuit with 136 independent phases;
- the standard zero-angle IQP product input;
- exact population Hamming-RBF MMD with bandwidths `(1,2,4)`;
- exact gradients and exact checkpoint evaluation;
- 2,000 Adam updates with cosine decay `0.01 -> 0.001`;
- five random initializations;
- exact raw bitstring and magnetization-profile TVD after selection.

Result: `results/curie-weiss-representability-v1/results.json`  
SHA-256: `01eb3903695ee3c158d0610eff360fb80f84f3c59e6a383f636c5e3211b645e7`

| beta | Regime | Best raw TVD | Median raw TVD | Best profile TVD | Median profile TVD |
|---:|---|---:|---:|---:|---:|
| 0.5 | disordered | 0.3991 | 0.4013 | 0.1568 | 0.1580 |
| 1.0 | critical | 0.3863 | 0.5532 | 0.0668 | 0.3931 |
| 1.5 | ordered | 0.3728 | 0.3729 | 0.3728 | 0.3729 |

The unrestricted optimizer does not reliably learn these targets. At the
critical point, one initialization learns the magnetization profile reasonably
well but distributes probability incorrectly within equal-magnetization shells,
giving raw TVD `0.386`. This is spontaneous symmetry breaking by the optimizer,
not a property of the target.

## Exchange-symmetric MMD diagnostic

The same regular circuit contains a two-dimensional exchange-symmetric
subfamily: all one-body phases are equal and all two-body phases are equal. No
gate or generator is removed; only parameter equality is enforced. A global
search of this subfamily under the same exact Hamming-MMD objective gives:

| beta | Raw/profile TVD | Exact MMD-squared |
|---:|---:|---:|
| 0.5 | 0.1143 | 1.35e-7 |
| 1.0 | 0.2332 | 7.73e-7 |
| 1.5 | 0.0774 | 1.65e-7 |

Result: `results/curie-weiss-representability-v1/symmetry-diagnostic.json`  
SHA-256: `e0c64eeb1fe2cfb73f6a730200c5e12c034ee6e5e5837f2219889477da99d608`

This shows that respecting target symmetry dramatically improves training, but
also exposes an objective mismatch: an extremely small Hamming-MMD can coexist
with critical-point TVD `0.233`. A small sweep of narrower bandwidth sets did
not repair this critical mismatch; it is not enough merely to retune sigma.

## Capacity diagnostic

To separate circuit expressivity from the Gaussian-MMD objective, we globally
optimized exact TVD inside the same two-parameter symmetric subfamily. This is
not a proposed training method: exact TVD requires the full probability vector
and does not satisfy the paper's scalable classical-training premise.

| beta | Best exact TVD |
|---:|---:|
| 0.5 | 0.00724 |
| 1.0 | 0.04255 |
| 1.5 | 0.06275 |

Result: `results/curie-weiss-representability-v1/capacity-diagnostic.json`  
SHA-256: `c2cc8a5b173872a2391e3afdde14d010f3be5978d83ffa964369344cc3769f68`

The circuit family therefore has enough capacity to represent all three
regimes to useful fidelity. The negative primary result is predominantly a
training-objective and optimization problem, not a fundamental IQP
representability failure.

## Decision

The answer depends on the meaning of "can learn":

- **Circuit capacity: yes.** A regular complete two-local IQP circuit contains
  high-fidelity solutions across disordered, critical, and ordered regimes.
- **Current classically estimable Gaussian-MMD pipeline: not yet.** Ordinary
  136-parameter random-start training is unreliable, and even symmetry-aligned
  MMD optimization does not give adequate critical-point TVD.

Do not begin the conditional campaign yet. The next bounded study should improve
the trainable loss while preserving classical estimability and exchange
symmetry. Candidate objectives should weight magnetization-shell discrepancies
or higher-order symmetric Pauli characters explicitly. Only after independently
trained circuits reach stable low profile and raw TVD at all three regimes
should a product-state-conditioned shared circuit be planned.
