# Hopfield-beta conditional experiment

## Question

Can the shared-parameter conditional IQP model (i) memorize the beta-classes
of a two-pattern Hopfield Boltzmann family and (ii) interpolate to held-out
beta, when the classes are structured at the bitstring level rather than
uniform within Hamming-weight shells?

A preregistered representability analysis (Section "Representability
boundary") fixes the beta range in which this question is answerable for the
present model family and records the stacked barriers measured outside it.

## Target family

For \(n=19\) qubits and \(p=2\) patterns, the coupling matrix is

\[
J=\frac{p_1p_1^\top+p_2p_2^\top}{n},\qquad
\operatorname{diag}(J)=0,
\]

with spins \(s\in\{-1,+1\}^n\) and energy
\(E(s)=-\tfrac12 s^\top Js\). The target for a Condition \(\beta>0\) is the
Boltzmann distribution

\[
p(x\mid\beta)\propto\exp[-\beta E(s(x))],
\]

where \(x\in\{0,1\}^{19}\) is the little-endian bitstring of \(s\).

### Frozen pattern constants

The two patterns are fixed by `np.random.default_rng(0).choice([-1, 1], (2, 19))`
and frozen as constants:

- \(p_1=(1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1)\)
- \(p_2=(1,-1,1,1,-1,-1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1)\)

Their overlap is \(\rho=p_1\cdot p_2/n=1/19\approx0.0526\). The four states
\(\{p_1,-p_1,p_2,-p_2\}\) share the minimal energy
\(E=-(19+\rho n)/2=-162/19\approx-8.526\). The global flip symmetry
\(E(-s)=E(s)\) forces \(\mathbb E[K\mid\beta]=n/2\) for every \(\beta\);
Hamming-weight means are therefore uninformative here and are not reported.

### Calibration of the feasible beta range (computed at design time)

Pattern mass on \(\{p_1,-p_1,p_2,-p_2\}\), maximal bitstring probability, and
distance from the uniform distribution:

| \(\beta\) | pattern mass | \(\max_x p(x)\) | TVD from uniform |
|---:|---:|---:|---:|
| 0.5 | 0.0005 | 0.0001 | 0.2227 |
| 0.6124 | 0.0012 | 0.0003 | 0.2966 |
| 0.75 | 0.0030 | 0.0008 | 0.3774 |
| 0.8660 | 0.0068 | 0.0017 | 0.4602 |
| 1.0 | 0.0160 | 0.0040 | 0.5548 |
| 1.1180 | 0.0318 | 0.0080 | 0.6365 |
| 1.25 | 0.0627 | 0.0157 | 0.7235 |
| 1.4142 | 0.1265 | 0.0316 | 0.8176 |
| 2.0 | 0.5163 | 0.1291 | 0.9674 |
| 4.0 | 0.9840 | 0.2460 | 0.9994 |

Adjacent-target bitstring TVD along the study grid (bitstring /
Hamming-weight shell):

| pair | bitstring TVD | shell TVD |
|---|---:|---:|
| 0.5 \(\to\) 0.6124 | 0.0753 | 0.0042 |
| 0.6124 \(\to\) 0.75 | 0.0839 | 0.0048 |
| 0.75 \(\to\) 0.8660 | 0.0859 | 0.0054 |
| 0.8660 \(\to\) 1.0 | 0.1009 | 0.0077 |
| 1.0 \(\to\) 1.1180 | 0.0857 | 0.0083 |
| 1.1180 \(\to\) 1.25 | 0.0919 | 0.0107 |

The classes are distinguishable at bitstring resolution, and shell TVD hides
most of the structure: targets are far from uniform within weight shells.

## Angle-to-beta path

The Condition is the broadcast Angle encoding angle

\[
\alpha(\beta)=\frac{\pi}{4}\cdot
\frac{\log(\beta/0.5)}{\log(1.25/0.5)},
\qquad \beta\in[0.5,1.25].
\]

This is log-linear in \(\beta\): \(\alpha(0.5)=0\),
\(\alpha(\sqrt{0.5\cdot1.25})=\pi/8\) at the geometric center, and
\(\alpha(1.25)=\pi/4\). The map is frozen and disclosed; it is never re-fit.

## Split

- Training betas: \(0.5,0.75,1.0,1.25\).
- Held-out betas: \(\sqrt{0.5\cdot0.75},\sqrt{0.75\cdot1.0},\sqrt{1.0\cdot1.25}\)
  (log-midpoints of adjacent training betas).
- Held-out targets enter neither optimization nor checkpoint selection.

## Model, objective, and training budget

The model is the 190-parameter Complete 2-local ansatz on 19 qubits with the
frozen optimizer (ADAM, cosine decay \(10^{-2}\to10^{-3}\), 2,000 steps),
kernel bandwidths \(\sigma=(1,2,4)\), and ten matched seeds.

Training uses the **exact zero-variance Conditional MMD** (the
infinite-budget limit of the repository's classical estimator), evaluated
over all \(2^{19}\) bitstrings:

\[
\operatorname{MMD}^2(p_m,p_t)
=\sum_m q_\sigma(m)\bigl(F_m[m]-F_t[m]\bigr)^2,
\]

where \(F=\operatorname{FWHT}(p)\) are the full Fourier coefficients and
\(q_\sigma\) the product Bernoulli mask weights. This is a disclosed design
choice forced by the estimator barrier: with the frozen Monte Carlo budgets
(64 operators, 8,192 IQP samples per step), sampled-MMD training moves
essentially no probability mass onto the pattern states even where
representable (measured model pattern mass \(\lesssim0.02\) at every beta);
the exact objective at the same budgets reaches the representability
ceiling. The Monte Carlo estimator and its barrier remain part of the
boundary finding below.

## Models and comparisons

1. **Dense model:** one shared 4-condition model trained at the four training
   angles, evaluated exactly at all seven betas.
2. **Oracle adjacent-beta lookup:** because every held-out beta is the
   geometric midpoint of two training betas, evaluate both neighboring stored
   conditions and retain the lower TVD for each metric. Dense interpolation
   must beat this memorization baseline.
3. **No-condition baseline:** one shared model with all four conditions set to
   the same broadcast angle \(\alpha=\pi/8\) (the geometric-center condition,
   \(\beta=\sqrt{0.625}\)), trained against the same four targets. It measures
   what conditioning buys beyond a single shared distribution.
4. **Separate-model reference:** four independently trained single-condition
   IQP models, one per training beta, used as a capacity reference. They cannot
   interpolate.

## Metrics and frozen decision rule

Exact enumeration over all \(2^{19}\) bitstrings provides, at every beta:

- **Bitstring TVD** (primary; captures the within-shell structure),
- **Hamming-weight profile (shell) TVD** (secondary),
- **Pattern-mass error**: \(|p_{\text{target}}-p_{\text{model}}|\) mass on
  \(\{p_1,-p_1,p_2,-p_2\}\) (structural diagnostic).

Held-out exact Conditional MMD is a secondary diagnostic at held-out betas.

The two resolution levels are audited separately; failure on one is not
overridden by success on the other:

- **Memorization test:** the dense model must have lower median bitstring and
  shell TVD than the no-condition baseline at every training beta.
- **Continuous-conditioning test:** at every held-out beta, dense
  interpolation must have lower median bitstring and shell TVD than oracle
  adjacent-beta lookup.
- **Pattern-mass diagnostic:** report median pattern-mass error at every beta;
  this is diagnostic, not a pass/fail test.

## Representability boundary (companion finding)

Measured at design time with single-condition exact-gradient optimization at
\(\beta=4\) (target pattern mass 0.984), ADAM \(10^{-2}\), 3,000 steps, and the
reproducible script `scripts/analyze_hopfield_boundary.py`:

| configuration | objective | final pattern mass |
|---|---:|---:|
| 2-local, m=0 ancillas | sampled MMD (frozen budget) | \(\lesssim0.01\) |
| 2-local, m=0 | exact MMD (any \(\sigma\)) | 0.079 |
| 2-local, m=1 | exact MMD | 0.160 |
| 2-local, m=2 | exact MMD | 0.160 |
| 2-local, m=0 | exact KL | 0.096 |
| 2-local, m=1 | exact KL | 0.240 |
| 2-local, m=2 | exact KL | 0.478 |
| 2+3-local (1,159 params), m=0 | exact KL | 0.101 |

Three stacked barriers are identified:

1. **Estimator barrier.** With the frozen Monte Carlo budgets the sampled
   Conditional MMD moves essentially no mass onto the pattern states,
   regardless of ancillas or sigma; the exact objective at the same budgets
   reaches its ceiling. The feasibility runs for m=1,2 at the original
   \(\beta\in[0.5,4]\) grid measured model pattern mass 0.006--0.019 at
   \(\beta=2,4\).
2. **MMD localization barrier.** Even the zero-variance MMD caps pattern mass
   at 0.08 (m=0) or 0.16 (m\(\ge\)1), independent of \(\sigma\). KL is
   strictly better, placing the kernel objective between representability and
   trainability.
3. **Product-envelope representability barrier.** With a product reference,
   \(|w(z)|\equiv|\varphi(z)|\) is fixed by the smooth reference envelope,
   while a pattern-peaked target requires \(w\) to be a sparse sum of
   characters. The exact-KL ceiling is 0.10 at m=0; traced-out ancillas
   multiply the effective envelope budget by \(2^m\) and lift it to 0.48 at
   m=2; a complete 3-local diagonal does not lift it further.

The feasible beta grid above keeps every target's pattern mass below the
lowest measured ceiling (0.079), so the memorization and interpolation tests
are answerable within the family.

## Claim boundary

Success would establish conditional memorization of, and interpolation along,
this one-dimensional log-linear path in \(\beta\) for one fixed pair of
patterns on 19 qubits, trained with an exact objective. It would not
establish the same for sampled budgets, for extrapolation beyond
\(\beta\in[0.5,1.25]\), for other pattern sets or \(p>2\), or for the peaked
regime \(\beta\gtrsim1.5\), which the boundary finding places outside the
representable family. No quantum-sampling claim is made.
