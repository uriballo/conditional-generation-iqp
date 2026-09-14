# Estimator complexity and the product-input compilation dip

The manuscript now separates reference-interface cost, arithmetic cost of moments/gradients, statistical sample complexity, and implementation timings. The derivation is in `Overleaf/estimator_complexity.tex`, included in the main Methods. No frozen estimator source or production result was changed.

## Arithmetic model and notation

One condition; N total qubits including ancillas; P diagonal generators; Lg = sum of their support sizes; A distinct observable masks; S reference draws; B target examples on n visible bits. C conditions and H bandwidths multiply per-bank costs in the absence of reuse. Arithmetic is counted at fixed or controlled precision; bit complexity, compiler passes, and physical quantum preparation are separate questions. The finite-precision qualification in the estimator proposition still applies.

| Reference used in the paper | Preprocessing | One draw | One arbitrary-mask ratio |
| --- | --- | --- | --- |
| Product | O(N) | O(N) | O(N) |
| Symmetry-only, one or two radial blocks | O(N) | O(N log N) with the current sorts; O(N) attainable | O(N) |
| Bond-informed, two rings | O(N) | O(N) | O(N) |

Product sampling uses N Bernoulli draws. Symmetry-only sampling uses a categorical shell draw and two sorts of N random coordinates (summed over a fixed number of blocks). A sequential fixed-weight draw instead chooses each bit with probability remaining-ones / remaining-sites, so O(N) is attainable without sorting. Shell weights/amplitudes can be precomputed with linear recurrences at the arithmetic level. Bond sampling contracts constant 4-by-4 transfer matrices, with suffix preprocessing; each amplitude ratio uses constant 2-by-2 matrices along the ring. It never enumerates 2^N states. Product and radial ratios can use O(1+w) work for a weight-w mask, after caching original block weights and representing mask support explicitly; all reported implementation bounds use dense O(N) queries. Identity/parity-forbidden operators have shortcuts but do not determine worst-case cost.

## Moments, gradients and MMD

Precompute flip indicators f_aj = (a dot g_j) mod 2 in O(A Lg). Per string, generator characters h_sj = (-1)^(z_s dot g_j) cost O(Lg), and all phase differences cost O(AP). Let W_sa be the complex ratio times the phase exponential. The derivative is

`J_aj = (1/S) sum_s Re(i W_sa) [-2 f_aj h_sj]`.

Thus the whole A-by-P Jacobian can be accumulated as one A-by-S times S-by-P product and an elementwise multiplication by the flip indicators. It costs O(SAP), reuses every reference draw, and does not require P separately sampled estimators. Including sampling/ratios, attainable moment/full-Jacobian work is

`T = O(N + A Lg + S [Ts(N) + Lg + A(N+P)])`.

An attainable streaming-memory bound is O(N + Lg + AN + AP). It includes the mask bank, flip indicators/Jacobian accumulators, and per-string work. This is not the memory footprint of the batched JAX code.

For one moment (A=1), any single derivative has the same leading cost as the moment, and accumulating its P derivatives also has that order. For complete one- and two-body generators, P=N(N+1)/2 and Lg=N^2. Consequently the support-based bound is O(SAN^2) for **all three** reference families. Their differing reference costs and constants remain relevant at finite width.

For a fixed mask bank, target Fourier characters cost O(ABn). Both target and model U-statistics use squared sums with a diagonal subtraction: O(A(B+S)), not quadratic pair loops. A scalar MMD and its full phase gradient therefore cost O(T+ABn). There is no need to materialize an observable-by-parameter Jacobian just to differentiate one scalar loss. Across the condition/bandwidth banks, the bound is O(CH(T+ABn)), taking A as their maximum width. Fixed masks and fixed validation target terms permit reuse; training minibatches change target terms.

The finite second moments bound scalar additive sample complexity by O(epsilon^-2 log(1/delta)) using median-of-means. For simultaneous accuracy of a fixed bank's A moments and AP phase derivatives, use O(epsilon^-2 log(A(P+1)/delta)) draws, with the same sample strings shared across components. This is **componentwise** accuracy, not fixed total Euclidean error. For loss/gradients, apply the same finite-variance argument to independent pairs and the desired mask/condition law; the U-statistic avoids forming pair arrays. Production uses ordinary sample means and its empirical bootstrap intervals, not median-of-means. Increasing S alone never eliminates the error from a frozen finite mask bank.

## What the implementation actually computes

`simulator._phase_differences` computes sample/generator and mask/generator parities with dense matrix products. Its forward arithmetic is conservatively

`O(N + (S+A)NP + SAP + SAN + S Ts(N))`.

The ANP term can be cached/constant-folded for fixed banks. Scalar reverse-mode differentiation has the same leading arithmetic order as the forward differentiable graph. At fixed A and a complete one-/two-body bank, the dense parity term gives O(SN^3); claiming the current code already implements support-based O(SN^2) would be misleading.

The benchmark harness uses `jax.jacrev` on A outputs. Its product-input optimized HLO explicitly retains an [A,A,1,S] intermediate and a [A^2,S] times [S,P] contraction, costing O(SA^2P). This matches reverse-mode batching over an output basis; it is an implementation overhead, not an intrinsic cost of the phase-derivative estimator. The attainable analytic O(SAP) accumulator above is not implemented in the frozen study. No optimization was made as part of this audit.

A conservative explicit-batch forward memory bound is O(NP + AN + SN + SP + SAN + AP), before compiler fusion. The full-Jacobian path additionally exposes O(SA^2 + A^2P) intermediates in the inspected HLO. Actual allocation can differ through fusion and scheduling. The production RSS is cumulative process high-water memory including exact checks, compilation and previous cases; it cannot validate these isolated estimator-memory bounds.

Exact enumeration provides an exponential verification baseline. For these references, amplitudes cost O(N2^N), diagonal phases O(Lg 2^N), and a Walsh–Hadamard transform O(N2^N). A further transform of output probabilities gives all characters, followed by O(A) lookups. One can stream the P phase derivative states, transform their probability derivatives, and select A characters in O(2^N[Lg+PN]+AP) time and O(2^N+Lg+AN+AP) memory. The exact oracle's current dense/automatic-differentiation implementation need not achieve these streaming bounds. The benchmark harness constructs exact arrays outside the measured sampled call; the sampled estimator itself does not depend on enumeration.

## Compilation dip: fresh-process follow-up

The original S=8192 product compilation curve decreases from N=12 to N=16. We repeated the central-condition, initial-scale fixture at each N=8/12/16 in three fresh processes per host, on both original Strix CPUs: **18 probes**. Each used the frozen runner and simulator, JAX 0.11.0 CPU, x64, CPU affinity 0–15, OMP_NUM_THREADS=16 and OPENBLAS_NUM_THREADS=1. Persistent compilation caching was explicitly disabled with JAX_ENABLE_COMPILATION_CACHE=false. Probes ran sequentially within each host. These are diagnostic follow-ups, not replacements for the frozen timing observations.

Median seconds across three processes per cell:

| Host | N | Lowering | Backend compilation | Lowering + compilation |
| --- | ---: | ---: | ---: | ---: |
| Strix 1 | 8 | 0.08853 | 0.09911 | 0.1882 |
| Strix 1 | 12 | 0.08859 | 0.2205 | 0.3084 |
| Strix 1 | 16 | 0.04253 | 0.08009 | 0.1228 |
| Strix 2 | 8 | 0.08977 | 0.09943 | 0.1897 |
| Strix 2 | 12 | 0.09055 | 0.2197 | 0.3105 |
| Strix 2 | 16 | 0.04275 | 0.07788 | 0.1211 |

The production field `sampled_compile_seconds` includes both lowering and backend compilation; the probe times these separately. Import, device initialization and exact-fixture construction occur before the probe's timed lowering. The original production process also had prior compilation activity, so absolute timings need not match fresh-process measurements.

The dip persists in **both** lowering and backend compilation, ruling out a persistent-cache hit or random one-off timing fluctuation as a sufficient explanation. Optimized HLO retains the full [16,P] Jacobian with P=36,78,136, the same 22 fusion instructions and comparable program text sizes (approximately 78 kB). The compiler's arithmetic estimates **increase** from 0.191 to 0.396 to 0.683 billion FLOPs; its estimated bytes accessed increase from 69.0 to 85.6 to 107.6 MB. These are compiler estimates, not measured hardware instruction or memory-traffic counts. Warm execution also increases with width. The original independent oracle checks already verified all three widths numerically.

**Conclusion:** retain the nonmonotonic observation. It concerns compilation behavior of these concrete shapes, not an easier 16-qubit estimation problem or improved asymptotic scaling. The exact responsible compiler pass/code-generation decision has not been isolated; a power-of-two/vectorization explanation would be speculative. No asymptotic compilation-time claim is justified by three widths.

Reproduce the diagnostic with `scripts/probe_reference_estimator_compile.py`; validate/summarize with `scripts/report_reference_estimator_compile.py`. [All probe metadata, HLO and summary](../../artifacts/reference-estimator-study-2026-09-07/compile-probe/summary.md), [main production result](reference-estimator-study-result-2026-09-07.md), [frozen protocol](reference-estimator-study-protocol-2026-09-07.md).
