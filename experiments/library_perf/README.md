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
- The pure-JAX mask path (`loss.sampled_mmd`) costs ~1.5–2x the compressed
  step at the same draw budget: it cannot collapse duplicates, so it evaluates
  ~1.8x more mask columns. `sampled_mmd` remains for whole-loop-on-device use.
- Product ratios run in float64 (real amplitudes), the batched numpy bank build
  uses one `np.unique` for all entries, and `banked_mmd` can share samples; see
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

## Same fit with 10 ancillas (22 model qubits)

`--ancillas 10` puts ten identity-marginalized qubits under the same product
encoder and shared IQP unitary; the target stays 12 visible bits. Everything
else is unchanged (2000 updates, batch 100, `n_ops=1000`, 500 samples,
`sigmas=(1,2,4)`).

| implementation | wall | per step | best step | validation | held-out MMD |
|---|---|---|---|---|---|
| research campaign (retained bank) | 48.71 s | 24.4 ms | 1975 | 0.02179 | 3.74e-3 |
| library `mask_refresh="cache"` | 31.16 s | 15.6 ms | 1750 | 0.02117 | 4.15e-3 |
| library `mask_refresh="step"` (default) | 35.36 s | 17.7 ms | 1750 | 0.02091 | 3.72e-3 |
| library `mask_refresh="step"`, seed 1 | 33.87 s | 16.9 ms | 1800 | 0.02174 | 3.85e-3 |

- The library stays ~27% faster than the campaign implementation with ancillas
  (step) and ~36% with the retained bank.
- Held-out MMD improves over the ancilla-free runs (4.8e-3 → 3.7–4.2e-3),
  matching the paper's ancilla-scaling finding; the ancilla overhead itself is
  small at this budget (~7% per step for 253 vs 78 generators).
- Ancilla initialization is the campaign rule: visible parameters from data,
  seeded Gaussian noise on ancilla-support generators
  (`initialize_parameters_with_ancillas` / `data_initialize`), identical in both
  implementations.

## Files

| File | Role |
|---|---|
| `common.py` | problem builder, blocking timer, result writer |
| `bench.py` | `step` / `validate` / `e2e` / `memory` subcommands |
| `review.py`, `review.md` | jit/vmap variant review and its written findings |
| `step_improve.py` | interleaved A/B of bank build, ratio dtype, cap widths |
| `pipeline.py` | full-pipeline arm comparison and checkpoint consistency |
| `gaussian_fit.py` | full 12-qubit Gaussian fit through the library |
| `legacy_fit.py` | the same fit through the research campaign implementation |
| `results/strix-*.json` | idle-host measurements |
| `results/review.json`, `results/step_improve.json`, `results/pipeline.json` | raw numbers behind the docs |
| `results/pre-refactor-*.json`, `results/post-refactor-*.json` | earlier M3 history |
