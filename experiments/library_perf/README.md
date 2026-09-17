# `iqp-cond` performance harness

Measures the shipped library at the default `TrainingConfig` so the cost of the
mask-refresh protocol is a number, not a preference.

```bash
PYTHONPATH=packages/iqp-cond/src .venv/bin/python experiments/library_perf/bench.py step
PYTHONPATH=packages/iqp-cond/src .venv/bin/python experiments/library_perf/bench.py validate
PYTHONPATH=packages/iqp-cond/src .venv/bin/python experiments/library_perf/bench.py e2e --steps 2000 --refresh step
PYTHONPATH=packages/iqp-cond/src .venv/bin/python experiments/library_perf/bench.py memory
```

Each command writes `results/<label>-<command>.json`; timing is the median of
warm, fully blocking calls, memory is peak `ru_maxrss` of a dedicated
subprocess (includes the JAX runtime).

## Experiment regime (1000-2000 operators, 500-1000 samples)

The shipped `TrainingConfig` defaults (64 operators, 8192 samples) were the
preliminary campaign settings.  The current experiments use 1000-2000
operators and 500-1000 samples; bank construction is a much larger share
there, and mixed precision pays more.  Per-step breakdown (idle host):

| case | bank build | step f64 | step f32-GEMM | validation |
|---|---|---|---|---|
| r=0, 1000 ops, S=500 | 1.24 ms | 8.42 ms | 8.06 ms | 2.75 ms |
| r=10, 1000 ops, S=500 | 1.24 ms | 10.85 ms | 8.40 ms | 4.95 ms |
| r=0, 2000 ops, S=1000 | 2.26 ms | 20.48 ms | 17.58 ms | 8.00 ms |
| r=10, 2000 ops, S=1000 | 2.26 ms | 26.58 ms | 20.51 ms | 14.13 ms |

Full 2000-step fits at 2000 ops / 1000 samples:

| arm | wall | per step | validation | held-out MMD | peak RSS |
|---|---|---|---|---|---|
| r=0, float64 | 58.86 s | 29.4 ms | 0.022653 | 4.98e-3 | 1221 MiB |
| r=0, float32 GEMM | 55.44 s | 27.7 ms | 0.022653 | 4.98e-3 | 1162 MiB |
| r=10, float64 | 70.93 s | 35.5 ms | 0.021473 | 3.85e-3 | 1250 MiB |
| r=10, float32 GEMM | 59.30 s | 29.7 ms | 0.021473 | 3.85e-3 | 1251 MiB |

Mixed precision is worth 5.8% (r=0) to 16.4% (r=10) end to end with
bit-comparable validation and held-out scores; ten ancillas cost 7-20% wall
and improve the held-out MMD by ~23%.  `sample_chunk` is neutral in this
regime (S <= 1000) and the bank build is down to 1.2-2.3 ms after the
rejection-free uint32 sampler rewrite.

## Headline numbers (2026-09-17, idle `strix-halo2`)

Idle 32-core x86, 124 GiB, JAX 0.11 CPU backend, float64. Problem: 12 visible
qubits, 5 conditions, complete two-local ansatz (78 generators), defaults
`n_ops=64`, `n_iqp_samples=8192`, validation `n_ops=256`,
`n_iqp_samples=16384`, `sigma in {1, 2, 4}`.

| measurement | value |
|---|---|
| training step, loss + gradient (`mask_refresh="step"`) | 21.6 ms + 1.1 ms bank build |
| training step, loss + gradient (per-entry draws) | 30.7 ms |
| training step, loss + gradient (pure-JAX masks) | 47.0 ms |
| validation pass, fixed bank | 55.4 ms |
| 2000-step fit, `step` | **59.8 s** (29.9 ms/step) |
| 2000-step fit, `cache` | 71.0 s (35.5 ms/step) |
| 2000-step fit, 8 visible / 3 conditions, `step` | 25.4 s |
| peak RSS, training step (compressed bank) | 616 MiB |
| peak RSS, training step (pure-JAX masks) | 873 MiB |
| peak RSS, validation pass | 675 MiB |
| peak RSS, paper scale (22 qubits, 10 conditions, spin-flip) | 1152 MiB |

Reading:

- **The default per-step refresh is now the fastest arm.** Sharing one
  reference draw per condition across bandwidths (the paper's estimator form)
  saves more than the per-step bank build costs, so `step` beats `cache` by
  ~16% while keeping each update unbiased for the true kernel MMD.
- **Mixed-precision phase contraction** (`gemm_dtype=jnp.float32`, either on the
  loss functions or `TrainingConfig`) is another 7.5% at production and 21.5% at
  paper scale on this CPU host, with gradient error ~1e-7; on GPUs the same
  estimator structure gains 15-28x.  See `review.md` and `precision.py`.
- **Sample-axis streaming** (`sample_chunk`, `chunking.py`) bounds the live
  `(samples, masks, qubits)` ratio tensor.  At the 8192-sample defaults it is
  3.5% faster with no ancillas and 14.9% faster with ten, at 8% lower peak
  RSS, and identically valued; at small sample budgets it costs a little, so
  it is off by default.
- The fresh-in-JAX mask path (`iqp_cond.mmd` with `sigmas`/`n_ops`) costs
  ~1.5–2x the compressed step at the same draw budget: it cannot collapse
  duplicates, so it evaluates ~1.8x more mask columns. It remains for
  whole-loop-on-device use.
- Product ratios run in float64 (real amplitudes), the batched numpy bank build
  uses one `np.unique` for all entries, and the bank path can share samples; see
  `review.md` for the interleaved A/B measurements and the rejected variants
  (vmap over conditions/bandwidths, multi-bank cap calibration).

Earlier M3 Max numbers (`results/pre-refactor-*.json`,
`results/post-refactor-*.json`) are kept for provenance but were taken under
background load; compare ratios within a single run, not absolute times across
machines.

## 12-qubit Gaussian fit: library vs the campaign implementation

`gaussian_fit.py` (library) and `legacy_fit.py` (research `src/iqp_cond`, the
code path the published experiments ran) on the archived 12-qubit Gaussian data
(5 training conditions, 1000 samples each; 8 held-out conditions), 2000
updates, batch 100, `n_ops=1000` per sigma per condition, 500 Monte Carlo
samples per case, `sigmas=(1,2,4)`, data-based initialization from condition 0.
Idle `strix-halo2`.

| implementation | wall | per step | best step | validation | held-out MMD |
|---|---|---|---|---|---|
| research campaign (retained bank, per-entry draws) | 37.97 s | 19.0 ms | 550 | 0.02356 | 4.82e-3 |
| library `mask_refresh="cache"` | 26.51 s | 13.3 ms | 425 | 0.02261 | 5.14e-3 |
| library `mask_refresh="step"` (default) | 33.05 s | 16.5 ms | 1250 | 0.02239 | 4.83e-3 |
| library `mask_refresh="step"`, seed 1 | 32.37 s | 16.2 ms | 250 | 0.02327 | 4.84e-3 |

- The library's retained-bank arm is **30% faster** than the campaign
  implementation at the same budget (batched bank build, real float64 ratios,
  jitted validation); the default per-step refresh is still **13% faster** than
  the campaign run while removing the retained-bank bias.
- Quality is comparable across all arms; the held-out spread (~3e-4) is single-
  seed mask noise, not a ranking.
- Protocol differences to keep in mind: the campaign shuffles epochs without
  replacement (`shuffled_batches`), the library resamples with replacement per
  update; the campaign's validation pass runs eagerly, the library's is jitted.

## Same fit with 0 and 10 ancillas (current library)

`--ancillas r` puts `r` identity-marginalized qubits under the same product
encoder and shared IQP unitary; the target stays 12 visible bits. Everything
else is unchanged (2000 updates, batch 100, `n_ops=1000`, 500 samples,
`sigmas=(1,2,4)`, per-step refresh). `--gemm-dtype float32` selects the mixed
precision phase contraction. Idle `strix-halo2`, fresh bank sampler.

| arm | wall | per step | validation | held-out MMD | peak RSS |
|---|---|---|---|---|---|
| r=0, float64 | 27.21 s | 13.61 ms | 0.022492 | 4.58e-3 | 1200.8 MiB |
| r=0, float32 GEMM | 26.50 s | 13.25 ms | 0.022492 | 4.58e-3 | 1173.3 MiB |
| r=10, float64 | 30.13 s | 15.06 ms | 0.021034 | 3.92e-3 | 1306.8 MiB |
| r=10, float32 GEMM | 27.62 s | 13.81 ms | 0.021034 | 3.92e-3 | 1253.0 MiB |

- Ten ancillas cost **+10.7% wall and +8.8% peak RSS** at float64, while
  improving the held-out MMD by ~14% (4.58e-3 → 3.92e-3), matching the paper's
  ancilla-scaling finding.
- Mixed precision helps more with ancillas (−8.3% vs −2.6% wall) because the
  ancilla-dependent tensors are the ones the cast halves; quality is unchanged.
- Training equivalence across dtypes: three seeds each at r=0 and r=10 reach
  the **same best step**, and validation, held-out MMD and final train loss
  agree to ~1e-8 relative (e.g. held-out 4.58295435e-3 vs 4.58295429e-3),
  against a seed-to-seed spread of ~6%. The cast only moves noise far below the
  statistics the fit is decided on.
- The full-run peak is dominated by runtime and evaluation buffers, so the
  ancilla memory effect is easier to see on the step alone (same settings,
  training step in a subprocess, per-entry evaluation):

| samples | r=0 peak | r=10 peak | difference |
|---|---|---|---|
| 500 | 505.3 MiB | 518.9 MiB | +13.6 MiB (+2.7%) |
| 16384 | 2520.9 MiB | 2855.1 MiB | +334 MiB (+13.3%) |

The step's ancilla cost grows with the sample budget because the product-encoder
ratio tensor is `(samples, masks, model qubits)`: masks stay visible-width but
the qubit axis grows from 12 to 22.

Ancilla initialization is the campaign rule: visible parameters from data,
seeded Gaussian noise on ancilla-support generators
(`iqp_cond.parameters_from_data` / `iqp_cond.random_parameters`), identical in both
implementations.

## Files

| File | Role |
|---|---|
| `common.py` | problem builder, blocking timer, result writer |
| `bench.py` | `step` / `validate` / `e2e` / `memory` subcommands |
| `review.py`, `review.md` | jit/vmap variant review and its written findings |
| `step_improve.py` | interleaved A/B of bank build, ratio dtype, cap widths |
| `precision.py` | float64 vs mixed-GEMM float32 vs full float32, with accuracy |
| `chunking.py` | sample-axis streaming vs monolithic, memory and time |
| `regime.py` | per-step breakdown in the 1000-2000 operator regime |
| `pipeline.py` | full-pipeline arm comparison and checkpoint consistency |
| `gaussian_fit.py` | full 12-qubit Gaussian fit through the library |
| `legacy_fit.py` | the same fit through the research campaign implementation |
| `results/strix-*.json` | idle-host measurements |
| `results/review.json`, `results/step_improve.json`, `results/pipeline.json` | raw numbers behind the docs |
| `results/pre-refactor-*.json`, `results/post-refactor-*.json` | earlier M3 history |
