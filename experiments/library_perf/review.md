# Performance review and step-mode optimisation

Date: 2026-09-17 · Subject: `packages/iqp-cond`. Production problem unless
stated: 12 visible qubits, 5 conditions, complete two-local ansatz,
`TrainingConfig` defaults (`n_ops=64`, `n_iqp_samples=8192`, `sigmas=(1,2,4)`).

Hosts: `strix-halo2` (32-core x86, 124 GiB, idle, JAX 0.11 CPU, float64) for all
headline numbers; Apple M3 Max for cross-checks (the laptop runs background load
and drifts by tens of percent, so only interleaved ratios are used from it).
Harness: `review.py`, `step_improve.py`, `pipeline.py`, `bench.py`; raw outputs
in `results/`.

## Optimisations landed

| change | measurement (interleaved, idle host) |
|---|---|
| Product ratios in float64 instead of complex128 | ratio stage 1.64 → 0.33 ms (5×); validation pass 94 → 55 ms |
| Batched numpy bank build (one `np.unique` for all entries) | bank build 2.77 → 1.10 ms (2.5×) |
| Rejection-free NumPy sampler (inverse CDF + order statistic) | 1.6-3.8× per bandwidth |
| uint32 rank draws for all conditions at once, plus scalar `math.exp` instead of a JAX probability call | bank build 3.83 → 2.26 ms at 2000 ops (1.7×) |
| Shared reference samples per condition across bandwidths | step loss+grad 30.7 → 22.9 ms (25%); e2e 77.8 → 59.8 s |

Bank-build numbers in the current experiment regime (1000-2000 operators):
1.24 ms at 1000 ops and 2.26 ms at 2000 ops, against 2.09/3.83 ms before the
uint32 rewrite.  Mixed precision is worth 5.8% (r=0) to 16.4% (r=10) of a
full 2000-step fit at 2000 ops / 1000 samples, with identical validation and
held-out scores.  The shipped-defaults figures below are the preliminary
campaign settings and remain only as historical context.

The first is exact: the product encoder's amplitudes are real, and the float
components extracted from the pinned complex expression are bit-for-bit the
golden values (all 74 tests pass unchanged, including the campaign-equivalence
loss). The third is the paper's estimator form — every mask is evaluated on the
same reference draw — and is exposed as `banked_mmd(..., shared_samples=True)`;
the trainer uses it for `mask_refresh="step"`, while `"cache"` keeps the
campaign's per-entry layout bit for bit.

Rejected after measurement: a multi-bank pilot to tighten `max_masks` (6%
slower and shape-sensitive), `jax.vmap` over conditions or bandwidths (below),
pure-JAX mask sampling for the default path (below), and sharing the sample
characters across bandwidths (1.00-1.03x and bit-identical: XLA already
common-subexpresses them).

## Precision levers

Inspired by the sibling `iqpopt-gpu` repository, whose benchmark shows that
mixed precision is the dominant lever on GPUs (`fp32-mix`: 22-24x on forward +
backward, <0.1% error, zero sign flips; plain fp64 is 1/64 rate on consumer
cards).  The shipped knob is `gemm_dtype` on `estimate_pauli_z_*`,
`banked_mmd`, `sampled_mmd`, and `TrainingConfig`: only the phase contraction
runs in the reduced dtype, and the result is cast back to float64 so
exponentiation and every reduction stay precise.

Idle `strix-halo2`, shared-sample step unless noted:

| case | float64 | mixed `gemm_dtype=float32` | full float32 |
|---|---|---|---|
| production step (n=12, C=5, `n_ops=64`, S=8192) | 22.45 ms | **20.76 ms** | 21.9 ms |
| paper scale (22 qubits, C=10, `n_ops=256`, S=16384) | 319.1 ms | **250.5 ms** | — |
| Gaussian fit, 2000 steps (`n_ops=1000`, S=500) | 27.39 s | 26.62 s | — |
| gradient relative error vs float64 | — | 4e-7 (production), 7e-8 (paper) | 1.5e-2 |
| loss deviation vs float64 | — | ~1e-11 | 8e-4 |

Reading:

- Mixed precision is profitable even on CPU once the estimator is large
  (+7.5% production, +21.5% paper scale) and costs nothing statistically: the
  reductions accumulate in float64, so the loss is unchanged to ~1e-11 and the
  gradient to ~1e-7.
- Casting *everything* to float32 is strictly worse: no extra speed and a 1.5%
  gradient error, which is why the knob is a GEMM dtype and not a global one.
- On GPUs the gap is far larger; the sibling benchmark is the reference for
  that regime.  Default remains float64 so golden fixtures and published
  numbers stay bit-identical.
- Trained outcomes are unaffected: three seeds each at r=0 and r=10 reach the
  same best step, with validation, held-out MMD and final loss within ~1e-8
  relative between float64 and mixed precision (seed spread is ~6%).

## Sample-axis streaming

The sibling repository streams its operator axis with `lax.scan` to cap memory;
here the sample axis is the one that matters, because the product-encoder ratio
tensor is `(samples, masks, model qubits)`.  `sample_chunk` streams that axis and
accumulates the per-mask sum and sum of squares, which is the same U-statistic
up to summation order.

Interleaved A/B on the idle host, forward + gradient:

| case | monolithic | chunk 2048 | speedup |
|---|---|---|---|
| r=10, S=8192, 64 ops | 37.87 ms | 34.22 ms | 1.11x |
| r=0, S=8192, 64 ops | 26.68 ms | 23.41 ms | 1.14x |
| r=10, S=16384, 1000 ops | 300.65 ms | 242.52 ms | 1.24x |

Full 2000-step fits at the shipped defaults (`n_ops=64`, S=8192):

| arm | wall | per step | peak RSS | validation | held-out |
|---|---|---|---|---|---|
| r=0, monolithic | 61.75 s | 30.9 ms | 1306 MiB | 0.023541 | 5.01e-3 |
| r=0, chunk 2048 | 59.56 s | 29.8 ms | 1199 MiB | 0.023541 | 5.01e-3 |
| r=10, monolithic | 97.82 s | 48.9 ms | 1462 MiB | 0.022461 | 4.03e-3 |
| r=10, chunk 2048 | 83.20 s | 41.6 ms | 1349 MiB | 0.022461 | 4.03e-3 |

Small budgets pay for the extra program structure (S=500: +13% step time), so
the default is monolithic.

Rejected: `jax.checkpoint` on the condition body.  It was 32% slower and did not
reduce the peak, because XLA already reuses buffers across the sequential
per-condition subgraphs and RSS counts allocator-retained pages, not live
tensors.  The memory that scales with the condition count is backward
activations, not the forward ratio tensor.

## Stage attribution (isolated calls, not additive)

| stage | ms |
|---|---|
| `sample` (one condition) | 0.85 |
| `amplitude_ratios` (one condition, first bandwidth entry) | 3.48 |
| `_phase_differences` (same) | 1.38 |
| full single-entry estimate | 6.77 |
| full single-entry `_condition_mmd` | 6.94 |

The full 15-entry step costs ~23 ms, well below 15 x 6.9 ms: XLA runs the
independent per-entry subgraphs concurrently. Stage rows say where the flops
live, not where wall clock goes.

## Variants, forward + gradient, same problem

| variant | ms | vs baseline |
|---|---|---|
| `banked_mmd`, one reference draw per condition | 22.9 | 0.75× |
| `banked_mmd`, per-entry draws (campaign layout) | 30.7 | 1.00× |
| `jax.vmap` over conditions, padded widths | 38.5 | 1.25× |
| `jax.vmap` over bandwidths, padded widths | 46.1 | 1.50× |
| pure-JAX `sampled_mmd` (masks in JAX) | 47.0 | 1.53× |

Conclusions:

- **Sharing samples across bandwidths wins** (25%): the sampler is expensive
  enough on this host that resampling per bandwidth is the largest avoidable
  cost, and one draw per condition is closer to the paper.
- **vmap does not pay here.** Both axes were measured; padding every entry to a
  common width (the per-bandwidth widths cannot survive `vmap`) and the batched
  dot lowering cost more than the Python loop, because the loop's subgraphs
  already overlap and each one stays narrow. vmap is the right tool for a
  GPU-resident batched workload, or for a much larger condition count.
- **The pure-JAX path remains the whole-loop-on-device option** at ~1.5× the
  shared-sample banked step: it cannot collapse duplicate masks, so it evaluates
  ~1.8× more mask columns at equal draws.

## Bank build

| component | ms |
|---|---|
| `operator_bank` (all 15 entries) | 1.10 |
| `sample_nonidentity_masks` entry-wise | 2.53 |

The batched path packs each mask into one integer code and groups by condition,
so compression is a single `np.unique` over 15 entries instead of one
lexicographic row sort per entry. The pinned `sample_nonidentity_masks` sampler
is untouched (golden fixtures) and remains the per-entry redraw path.

## Full pipeline

`pipeline.py`: shipped trainer at `mask_refresh="step"` and `"cache"`, plus a
loop mirroring it with `sampled_mmd`, on identical data, seed, schedule, and
optimizer; every arm scored on a common independent bank; the trainer's own
validation bank is rebuilt to check the checkpoint.

500 steps, 2 seeds:

| arm | wall s (seed 0 / 1) | independent score (seed 0 / 1) |
|---|---|---|
| `mask_refresh="step"` (default) | 16.2 / 16.4 | 2.30e-4 / -1.18e-4 |
| `mask_refresh="cache"` | 18.4 / 18.9 | 1.91e-4 / -1.02e-4 |
| pure-JAX `sampled_mmd` loop | 27.3 / 27.3 | 2.98e-4 / -1.06e-4 |

2000 steps, seed 0, production case (`bench.py e2e`):

| arm | wall | per step |
|---|---|---|
| step | **59.8 s** | 29.9 ms |
| cache | 71.0 s | 35.5 ms |
| pure-JAX (pipeline extrapolation) | ~109 s | 54.6 ms |

- The default per-step refresh is now the **fastest** arm: shared sampling saves
  more than the per-step bank build costs. It is also the unbiased one.
- Checkpoint integrity is exact: `best_params` re-evaluated on the trainer's
  validation bank and key reproduces `best_validation_loss` bit-for-bit for both
  trainer arms and both seeds.
- No convergence difference is detectable at 500 steps; arm spread (~1e-5 to
  1e-4) is the order of seed noise. This is not a convergence study.

## Correctness note found during the review

At encoder angles near the two-branch deterministic point (`alpha = pi/2`), the
reference law concentrates on a few strings but the amplitude ratios on rare
strings are huge; the estimator stays unbiased with bounded second moment, yet
the per-sample standard deviation computed from a finite sample can be ~0 while
the mean is still moving. Sample-standard-error tolerances are therefore not
valid there. The PennyLane cross-checks pick angles in the spread regime and
document this in `tests/test_pennylane_crosscheck.py`.

## Reproduction

On the idle Strix host (`uribagi@192.168.1.67`), with the package and harness
rsynced to `/home/uribagi/iqp-cond-perf`:

```bash
cd experiments/library_perf
export PYTHONPATH=../../packages/iqp-cond/src
../../.venv/bin/python review.py
../../.venv/bin/python step_improve.py
../../.venv/bin/python pipeline.py
../../.venv/bin/python bench.py e2e --label strix --refresh step   # and cache
```
