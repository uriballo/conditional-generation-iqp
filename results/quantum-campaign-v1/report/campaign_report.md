# Frozen Quantum campaign report

- Campaign: `quantum_campaign_v1`
- Manifest digest: `8cf5380b7a658381c8b61b3069f921c938f92be814c091418e46d4d905f6f0f3`
- Report digest: `67ebcbed110f1f311e43152ffe61af523e1a15d33f9a712ed1f7250d4e70d1fa`
- Status: `stopped_theorem_novelty_failed`

## Provenance checks

| check | status | expected | actual | details |
|---|---|---|---|---|
| manifest_content_digest | pass | 8cf5380b7a658381c8b61b3069f921c938f92be814c091418e46d4d905f6f0f3 | 8cf5380b7a658381c8b61b3069f921c938f92be814c091418e46d4d905f6f0f3 | — |
| immutable_campaign_snapshot | pass | — | — | snapshot parsed and matched by CampaignStore |
| calibration_artifact_digest | pass | 8d121e4a519766ff0234e00021815b3bcfd740e9fc9284dbf5b2aa448f29f6ae | 8d121e4a519766ff0234e00021815b3bcfd740e9fc9284dbf5b2aa448f29f6ae | — |
| exact_target_archives | pass | 8 | 8 | — |
| zero-run_estimator_scaling_evidence | pass | 8cf5380b7a658381c8b61b3069f921c938f92be814c091418e46d4d905f6f0f3 | 8cf5380b7a658381c8b61b3069f921c938f92be814c091418e46d4d905f6f0f3 | — |
| blocked_response_spectrum_evidence | pass | blocked_zero_run_preflight | blocked | canonical production evidence |
| immutable_gate_records | pass | — | 1 | all present gate envelopes, archived evidence paths, SHA-256 digests, terminal decisions, and manifest bindings must agree |
| reservation_result_compatibility | pass | — | — | — |

## Gate decisions

| gate | status | decision | evidence_status | evidence_path | evidence_digest | reason |
|---|---|---|---|---|---|---|
| theorem_novelty | failed | failed | pass | evidence/gates/theorem_novelty/c896667f5765e44c205766eaa357dc9df65b283bfaf25a8ecdd824372c293df2.md | c896667f5765e44c205766eaa357dc9df65b283bfaf25a8ecdd824372c293df2 | Internal correctness passed, but no central finite-resource response theorem was more than a direct specialization or short synthesis of the closest primary results; ADR 0042 novelty criterion failed. |
| target_calibration | blocked | — | not_applicable | — | — | terminally blocked by failed theorem_novelty |
| sampling_quality | blocked | — | not_applicable | — | — | terminally blocked by failed theorem_novelty |
| runtime_feasibility | blocked | — | not_applicable | — | — | terminally blocked by failed theorem_novelty |
| representability | blocked | — | not_applicable | — | — | terminally blocked by failed theorem_novelty |
| tuning_protocol | blocked | — | not_applicable | — | — | terminally blocked by failed theorem_novelty |
| benefit_retention | blocked | — | not_applicable | — | — | terminally blocked by failed theorem_novelty |
| interpolation | blocked | — | not_applicable | — | — | terminally blocked by failed theorem_novelty |
| ancilla_extension | blocked | — | not_applicable | — | — | terminally blocked by failed theorem_novelty |

## Non-transferable budget ledger

| pool | ceiling | reserved | completed | failed | remaining | status |
|---|---|---|---|---|---|---|
| feasibility_representability | 20 | 0 | 0 | 0 | 20 | blocked_unspent |
| tuning | 36 | 0 | 0 | 0 | 36 | blocked_unspent |
| primary | 320 | 0 | 0 | 0 | 320 | blocked_unspent |
| ancilla | 24 | 0 | 0 | 0 | 24 | blocked_unspent |
| total | 400 | 0 | 0 | 0 | 400 | blocked_unspent |

## Declared run matrix

| stage | declared | reserved | completed | failed | status | reason |
|---|---|---|---|---|---|---|
| runtime_feasibility | 1 | 0 | 0 | 0 | blocked | terminally blocked by failed theorem_novelty |
| representability_screen | 6 | 0 | 0 | 0 | blocked | terminally blocked by failed theorem_novelty |
| tuning | 36 | 0 | 0 | 0 | blocked | terminally blocked by failed theorem_novelty |
| primary | 320 | 0 | 0 | 0 | blocked | terminally blocked by failed theorem_novelty |
| ancilla | 24 | 0 | 0 | 0 | blocked | terminally blocked by failed theorem_novelty |
| representability_diagnostic | 1 | 0 | 0 | 0 | blocked | terminally blocked by failed theorem_novelty |

## Leakage checks

| check | status | violations | details |
|---|---|---|---|
| target_only_calibration_inputs | pass | [] | model/checkpoint performance keys are forbidden |
| dataset_split_key_schema | pass | [] | every finite-data input must use matched training_beta_i/validation_beta_i content-digest keys |
| training_validation_hash_disjointness | pass | [] | Training and Validation hashes must differ within every run |
| held_out_preprimary_access | pass | [] | Held-out hashes are forbidden before frozen primary evaluation |
| validation_only_checkpoint_selection | pass | [] | lowest_validation_conditional_mmd |

## Statistical decisions

```json
{
  "benefit_retention": null,
  "interpolation": null,
  "paired_cells": 0,
  "reason": "theorem novelty failed before primary evaluation",
  "status": "blocked"
}
```

## Figure-source inventory

| figure | output | output_exists | status |
|---|---|---|---|
| ising_calibration | paper/figures/ising_calibration.pdf | yes | pass |
| response_support | paper/figures/response_support.pdf | yes | pass |
| sampler_pilots | paper/figures/sampler_pilots.pdf | yes | pass |
| weight_fit | results/summary/weight_fit.pdf | yes | pass |

## Claim-support registry

| claim_id | section | status | evidence |
|---|---|---|---|
| A-01 | abstract | PROVEN-INTERNAL | Model assumptions; `TH-T1`; `TH-T5` |
| A-02 | abstract | KNOWN-SPECIALIZATION | `TH-T1`, `SRC-F1`, `SRC-F2` |
| A-03 | abstract | PROVEN-INTERNAL; novelty fail | `TH-T2`, `TEST-R`, `SRC-E3` |
| A-04 | abstract | KNOWN-SPECIALIZATION | `TH-T3`, `TH-T4`, `SRC-F1`--`SRC-F3` |
| A-05 | abstract | PROVEN-INTERNAL; known framework | `TH-T5`, `TEST-E`, `SRC-E1`, `SRC-E3` |
| A-06 | abstract | PROVEN-INTERNAL; known framework | `TH-T5`, `TEST-E`, `SRC-E2`, `SRC-E3` |
| A-07 | abstract | PROVEN-INTERNAL | `TH-T5`, `SRC-E1`--`SRC-E3` |
| A-08 | abstract | REJECTED | `TH-T1`, `TH-T5`, `SRC-E2` |
| A-09 | abstract | BLOCKED | `CAMP-PRI`, `CAMP-STAT` required |
| A-10 | abstract | BLOCKED | `CAMP-PRI`, `CAMP-STAT` required |
| A-11 | abstract | BLOCKED | `CAMP-SPEC` required |
| A-12 | abstract | REJECTED | Theorem dossier gate |
| A-13 | abstract | REJECTED | Specification boundaries; `TH-T4`; `SRC-H1`, `SRC-H2` |
| A-14 | abstract | EMPIRICAL-VERIFIED for the legacy study | `OLD-G` |
| A-15 | abstract | BLOCKED | `CAMP-ANC` required |
| I-01 | introduction | PROVEN-INTERNAL | Definition |
| I-02 | introduction | PROVEN-INTERNAL | Specification, model assumptions |
| I-03 | introduction | KNOWN-SPECIALIZATION | `SRC-F1`--`SRC-F3` |
| I-04 | introduction | KNOWN-SPECIALIZATION | `SRC-E1`--`SRC-E3`, `SRC-H1`, `SRC-H2` |
| I-05 | introduction | EXTERNAL-REVIEW | `SRC-E2`, `SRC-E3` |
| I-06 | introduction | PROVEN-INTERNAL for theory; preparatory-only for campaign | Model assumptions; `CAMP-M`; preparatory `CAMP-CAL` |
| I-07 | introduction | PROVEN-INTERNAL | `TH-T1`, `TH-T2`, `TH-T5` |
| I-08 | introduction | PROVEN-INTERNAL; novelty fail | `TH-T2`, `TEST-R`, `SRC-E3` |
| I-09 | introduction | KNOWN-SPECIALIZATION | `TH-T3` |
| I-10 | introduction | KNOWN-SPECIALIZATION | `TH-T4`, `SRC-F1`--`SRC-F3` |
| I-11 | introduction | REJECTED | `TH-T5`, counterexample 10.6 |
| I-12 | introduction | PROVEN-INTERNAL | `TH-T5`, `TEST-R` |
| I-13 | introduction | REJECTED | `TH-T5` |
| I-14 | introduction | REJECTED | `TH-T1`--`TH-T4`, `SRC-F1`--`SRC-F3` |
| I-15 | introduction | PROVEN-INTERNAL; modest delta | `TH-T5`, `TEST-E`, `SRC-E1`--`SRC-E3` |
| I-16 | introduction | BLOCKED | `CAMP-CAL`, `CAMP-S10`, `CAMP-S16`, `CAMP-REP`, `CAMP-PRI`, `CAMP-STAT` |
| I-17 | introduction | BLOCKED | `CAMP-PRI`, `CAMP-STAT` |
| I-18 | introduction | BLOCKED | `CAMP-SPEC` |
| I-19 | introduction | EMPIRICAL-VERIFIED | `CAMP-SCALE`, `TEST-SCALE` |
| I-20 | introduction | DECLARED-ONLY | Specification; `CAMP-ANC` |
| I-21 | introduction | REJECTED for the new draft | Specification; `OLD-G` |
| I-22 | introduction | BLOCKED | External-review gate |
