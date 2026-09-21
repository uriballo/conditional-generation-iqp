# n=8 production launch plan

> Status: executed successfully on 2026-09-21 after explicit author approval.
> The frozen 3,120-job matrix, results, and post-run integrity checks are
> documented in [the production report](review/PRODUCTION_REPORT.md). This file
> remains the pre-launch plan and authorization record.

## Decision and scope

The corrected locked audit supported the full n=8 campaign. At the time this
plan was frozen, it did not authorize execution. The author later granted
explicit approval, and the campaign ran through the separate fail-closed
authorization path. The production run is a 3,120-job replication and
precision campaign over optimizer seeds 501–510; it is not a second blind
audit and does not expand the claim to n=12, ancillas, hardware sampling, or a
new circuit family.

The planned matrix preserves the reviewed scientific choices:

| Component | Production scope |
| --- | --- |
| General encoders | broadcast affine and qubit affine |
| Structured comparison | anti-block |
| Baseline | condition-blind, even placement, fixed-total budget |
| Main protocols | interpolation, near, and far |
| Directional diagnostics | mirror-near and mirror-far, even \(k=3,5\) only |
| Anchor counts | 2, 3, 4, and 5 |
| Placements | even and random layout seeds 1101–1103 |
| Budgets | fixed total and matched exposure |
| Optimizer seeds | 501–510 |
| Selected training configuration | `stage2-04`: 600 base updates, 2 restarts, learning rate 0.03 to 0.003, bodyness targets 2 and 4 |

This gives 1,000 jobs for each trained encoder and 120 condition-blind jobs.
The full matrix contains 1,680 fixed-total and 1,440 matched-exposure jobs.

## Hard authorization gates

Production may start only after all of the following are true:

1. Commit the reviewed library changes in the `iqp-cond` submodule, then commit
   the reviewed benchmark source, report, tests, lockfile, submodule pointer,
   and `review/audit_provenance_root.json` as paper source commit C1. The author
   must explicitly accept that provenance root as the trust anchor for the
   original audit manifest. Both repositories must be clean; do not sweep
   unrelated user changes into either commit.
2. Run the complete tests, Ruff lint, and task-scope format checks from clean
   C1.
3. From clean C1, create a source-equivalence bridge against the untouched
   audit manifest:

   ```bash
   uv run python -m experiments.scalar_gaussian_translation.analysis \
     attest-source-equivalence \
     --manifest experiments/scalar_gaussian_translation/manifests/audit.json \
     --provenance-root experiments/scalar_gaussian_translation/review/audit_provenance_root.json \
     --output experiments/scalar_gaussian_translation/review/source_equivalence.json
   ```

   The pinned root has content digest
   `61e01f81018599943f2e735d388dcb275c761373da50703ca8e2b86a6907a905`
   and binds audit-manifest SHA-256
   `9a5e89a3a18deeaca98413e040b258de1e853db77eef4555c511d6a465371ea4`.
   Bridge creation must fail unless those reviewed identities, every
   runtime-source hash, the frozen-gate hash, and the protocol hash match. It
   records both the original dirty-manifest provenance and the clean C1
   provenance; it does not rewrite or relabel the old manifest. If any identity
   or runtime hash differs, stop and rerun the audit from a new clean manifest
   instead. Commit only the bridge as artifact commit C2, and verify that C2
   changes no runtime source.
4. From clean C2, regenerate the long-form rows, gate result, and attestation
   into a temporary directory outside the repository so the clean-state check
   remains meaningful. Pass `--source-equivalence` to every read of the old
   audit manifest. For example, with `audit_tmp` set to a fresh external
   directory:

   ```bash
   audit_tmp="$(mktemp -d)"
   uv run python -m experiments.scalar_gaussian_translation.analysis extract \
     --manifest experiments/scalar_gaussian_translation/manifests/audit.json \
     --records-dir results/scalar_gaussian_translation/job_records \
     --source-equivalence experiments/scalar_gaussian_translation/review/source_equivalence.json \
     --provenance-root experiments/scalar_gaussian_translation/review/audit_provenance_root.json \
     --output "$audit_tmp/audit_rows.csv"
   uv run python -m experiments.scalar_gaussian_translation.analysis evaluate-gates \
     --manifest experiments/scalar_gaussian_translation/manifests/audit.json \
     --records-dir results/scalar_gaussian_translation/job_records \
     --source-equivalence experiments/scalar_gaussian_translation/review/source_equivalence.json \
     --provenance-root experiments/scalar_gaussian_translation/review/audit_provenance_root.json \
     --gates experiments/scalar_gaussian_translation/review/frozen_gates.json \
     --output "$audit_tmp/audit_gate_check.json"
   uv run python -m experiments.scalar_gaussian_translation.analysis attest-audit \
     --manifest experiments/scalar_gaussian_translation/manifests/audit.json \
     --records-dir results/scalar_gaussian_translation/job_records \
     --source-equivalence experiments/scalar_gaussian_translation/review/source_equivalence.json \
     --provenance-root experiments/scalar_gaussian_translation/review/audit_provenance_root.json \
     --gates experiments/scalar_gaussian_translation/review/frozen_gates.json \
     --rows-csv "$audit_tmp/audit_rows.csv" \
     --gate-result "$audit_tmp/audit_gate_check.json" \
     --output "$audit_tmp/audit_attestation.json"
   ```

   The attestation must report `launch_valid: true`, bind all 936 records and
   checkpoints, verify the bridge and pinned root, require the supplied CSV to
   equal deterministic regeneration from those checkpoints, recompute the
   supplied gate result from those rows, and confirm exactly 28 native-true
   checks plus `pass=true` for each general encoder. Install the three
   byte-identical temporary artifacts
   under `review/`, commit them alone as artifact commit C3, and verify that C3
   changes no evaluator or runtime source. This staged sequence avoids requiring
   an artifact to contain the hash of the commit that first adds that artifact.
5. Record explicit author acceptance of the moderate general-encoder error,
   far-boundary failures, unresolved directional asymmetry, anti-block's
   coupled design differences, and the explanatory rather than competitive
   role of the benchmark.
6. From clean C3, generate a new production proposal. Do not authorize the
   current `split=dev` proposal. Use the locked audit grid so production repeats
   the reviewed estimand with fresh optimizer seeds:

   ```bash
   uv run python -m experiments.scalar_gaussian_translation.campaign \
     --manifest experiments/scalar_gaussian_translation/manifests/production_proposal.json \
     generate production --split audit \
     --selected-config experiments/scalar_gaussian_translation/review/selected_config.json \
     --frozen-gates experiments/scalar_gaussian_translation/review/frozen_gates.json
   ```

   Commit the proposal as a generated-artifact-only commit C4, and verify that
   its source snapshot still matches the runtime at C4.

7. Bind explicit authorization to the exact proposal SHA-256, provenance-root
   SHA-256, source-equivalence bridge SHA-256, audit-attestation SHA-256, clean
   paper revision, library revision, and runtime source snapshot.
   The current runner deliberately rejects every production proposal. Add and
   review an authorization artifact and fail-closed verification path only
   after the author approves this plan; do not enable production by manually
   flipping JSON booleans.

## Operational canary

Run a small subset of the authorized manifest on each Strix host before the
full shards. The canary must include each trained encoder, condition-blind, a
fixed-total job, and the maximum-cost matched-exposure \(k=5\) job. Canary
selection is by predeclared manifest identity, not performance.

The canary is operational only. It passes when:

- every selected job writes one matching job record and checkpoint;
- configs, manifest identity, gate hash, and source snapshot match;
- all recorded metrics are finite and both restarts complete;
- peak memory leaves adequate headroom for eight workers;
- logs contain no compilation, I/O, or numerical errors; and
- observed throughput and disk growth support the full launch.

Do not retune the model from canary accuracy. Any scientific change requires a
new versioned protocol and a new review. An operational failure pauses the
launch without invalidating the locked audit.

## Capacity and launch shape

The completed campaign used about 133 KiB per run directory. At the same rate,
3,120 jobs need roughly 420 MiB for run artifacts; reserve at least 1 GiB per
host for artifacts, logs, temporary directories, and compilation caches.

Audit timings predict approximately 3.46 serial worker-hours for the proposed
mix of 400-, 600-, 800-, and 1,000-update jobs. Sixteen perfectly scaling
workers would take about 13 minutes, but the canary measurement—not this ideal
estimate—sets the operational expectation.

After the canary passes, launch modulo-two shards with eight workers on each
host. The authorization-aware launcher must receive the same manifest and
authorization artifact on both hosts. The exact command syntax should be added
with the fail-closed authorization implementation; the present launcher
correctly refuses production proposals.

## Monitoring, recovery, and completion

- Monitor completed job-record counts, failed logs, free disk, memory, and
  per-job wall time. Do not inspect interim scientific rankings to stop or
  extend selected encoders.
- Re-running a shard safely reuses complete runs. An incomplete run directory
  fails closed; quarantine it with its log before retrying the same job ID.
- Preserve host-local logs and caches until record collection and verification
  finish.
- After both shards finish, collect records and run the same manifest-to-record
  and checkpoint validation used by `load_records`. Require exactly 3,120
  unique, matching jobs and no unexpected production records.
- Produce a production attestation that hashes the authorized manifest,
  authorization record, evaluator, extracted rows, every job record, every
  checkpoint, and all final summary outputs.
- Analyze the ten fresh seeds using both the frozen row-weighted estimand and
  the equal-configuration sensitivity estimand. Keep audit and production
  results separate. Report condition-resolved failures, complement covariance,
  encoder/protocol/anchor/budget strata, and uncertainty across optimizer
  seeds without inferential claims unsupported by the design.

Production is complete only when the 3,120-job attestation verifies, the final
figures and tables reproduce from the authorized artifacts, and the report
records failures as well as aggregate performance.
