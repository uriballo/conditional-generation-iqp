# Condition-count scaling completion audit

This status file is deliberately separate from the frozen design. It records
what has been verified in the current workspace without turning a short smoke
run into evidence for the 2,000-update, ten-seed study.

| Completion item | Status | Evidence |
|---|---|---|
| Frozen design written before training | PASS | `docs/condition-count-scaling-design.md` |
| Nested K=2/3/5/9 grids | PASS | `tests/test_scaling.py` |
| Fixed-total and fixed-per-Condition budgets | PASS | `tests/test_scaling.py`; JSON `budget` fields |
| Balanced Condition sampling | PASS | `tests/test_scaling.py` |
| Shared/Separate parameter accounting | PASS | `tests/test_scaling.py` and summary parameter panel |
| Per-Condition mean/median/worst aggregation | PASS | `tests/test_scaling.py`; `per_condition.md`/`per_seed.md` |
| Strict failure counts | PASS | `strict_comparison_counts`; generated `audit.json` |
| Resumable machine-readable runner | PASS | `scripts/run_condition_scaling.py` |
| All-K integration smoke matrix | PASS | 2 seeds, 25 updates under `tmp/scaling-smoke-full/` |
| Full-budget runtime benchmark | PASS | K=2 fixed-total Shared Angle, seed 0: 2,000 updates in 106.8 s; isolated artifacts under `tmp/scaling-benchmark/` |
| Frozen ten-seed, 2,000-update matrix | PASS | 240 JSON and 240 NPZ records under `results/scaling/runs/`; 14.53 h artifact span |
| Frozen scaling success criteria | FAIL | Profile criteria pass at K=3/5/9; Bitstring K=9 fails the strict Condition-level rule with 49/90 failures |

The frozen matrix contains 240 resumable top-level configurations and is now
complete. Two workers covered matched seed halves. The first and last artifact
timestamps span 14.53 hours; both workers exited successfully and the strict
summarizer accepted the full Cartesian matrix.

Run the frozen matrix with:

```bash
uv run python scripts/run_condition_scaling.py \
  --conditions 2,3,5,9 \
  --tracks fixed_total,fixed_per_condition \
  --methods shared_angle,no_condition,separate_bank \
  --seeds 0-9 \
  --skip-existing

uv run python scripts/summarize_condition_scaling.py
```

`results/scaling/summary/audit.md` is the authoritative claim-evidence audit.
The shared model passes every frozen profile criterion and the K=3 and K=5
Bitstring criteria. At K=9, however, 49 of 90 shared-vs-No-condition Bitstring
comparisons fail, exceeding the allowed half (45). The overall primary result
is therefore FAIL. The fixed-per-Condition track is descriptive and does not
override that verdict. The defensible manuscript role is a mixed stress-test
boundary, not a claim that full-distribution fidelity scales with Condition
count.
