# Exact-TVD ancilla campaign v1

This campaign is defined by
[`configs/exact_tvd_ancilla_v1.json`](../configs/exact_tvd_ancilla_v1.json).
It does not modify or extend any frozen historical manifest.

## Gates and waves

Run the stages in this order:

1. Prepare the seven fixed data archives with `run_exact_tvd_campaign.py
   prepare-data`.
2. Run all 90 estimator-calibration cells with
   `run_exact_tvd_development.py calibration`. Production is blocked unless
   one `n_samples` candidate passes every threshold in all 18 matched cells.
3. Run all 90 optimizer cells with `run_exact_tvd_development.py optimizer`.
   Create `protocol-selection.json` with
   `run_exact_tvd_optimizer_pilot.py`; this file is the production gate.
4. Run the retained 3,900 Gaussian and Curie--Weiss fits as a training-only
   wave. The discarded 400-fit MFC block remains in the frozen manifest but is
   not executed.
5. After training has stopped, run the 3,900 exact evaluations as a separate
   wave. The evaluator computes one condition at a time and marginalizes
   ancillas before continuing.
6. Merge both host shards, then run `process_exact_tvd_results.py` and
   `plot_exact_tvd_results.py`.

The sharded launcher `run_exact_tvd_strix_wave.py` supports the calibration,
optimizer, training, and evaluation waves. On each 32-thread Strix host, use
six workers; the launcher pins each worker to a disjoint CPU range.

## Production invariants

- `run_exact_tvd_campaign.py plan` reports the original 4,300 unique fits:
  2,100 Gaussian, 1,800 Curie--Weiss, and 400 MFC. Current execution stops at
  identity 3,900, before the obsolete MFC block.
- The production manifest remains unresolved until the immutable protocol
  selection passes both development gates.
- Shared-model initialization, training, validation-MMD checkpointing, and
  exact-TVD checkpoint selection use training conditions only.
- Independent held-out fits are target-aware capacity oracles and are never
  inputs to the shared fit.
- Result envelopes store relative artifact paths so shards remain readable
  after transfer from either host.
- The canonical handwritten paper is outside this workflow and must not be
  edited by campaign scripts.

## Verification

Use `preflight_exact_tvd_campaign.py` on an idle host. The required case is 16
visible plus 10 ancilla qubits and must remain below 96 GiB peak RSS. Use
`benchmark_exact_tvd_workers.py` to compare two, four, and six workers while
reading the AMD `k10temp` CPU sensor; unrelated NVMe sensors are deliberately
excluded from the thermal gate.
