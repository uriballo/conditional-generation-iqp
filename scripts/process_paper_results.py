"""Verify and summarize the complete 8/12-qubit paper subset of the frozen campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json, sha256_file
from iqp_cond.exact_tvd_campaign import load_manifest, production_identities
from iqp_cond.exact_tvd_results import (
    aggregate,
    load_records,
    summarize_seed_values,
    write_summary_tables,
)


def process(campaign: Path, manifest: Path, output: Path) -> None:
    config = load_manifest(manifest)
    expected = {
        r.run_id: r
        for r in production_identities(config)
        if r.dataset in ("gaussian", "curie_weiss") and r.visible_qubits in (8, 12)
    }
    if len(expected) != 2600:
        raise ValueError("paper subset must contain 2600 identities")
    records = [
        r
        for r in load_records(campaign / "evaluations", config, require_complete=False)
        if r.identity["dataset"] in ("gaussian", "curie_weiss")
        and r.identity["visible_qubits"] in (8, 12)
    ]
    actual = {r.envelope_path.stem for r in records}
    if actual != set(expected):
        raise ValueError(
            f"incomplete paper subset: missing={len(set(expected) - actual)}"
        )
    provenance = []
    for record in records:
        run_id = record.envelope_path.stem
        training_path = campaign / "training" / f"{run_id}.json"
        training = read_immutable_json(training_path)
        if training["run_identity"] != expected[run_id].to_dict():
            raise ValueError(f"training identity mismatch: {run_id}")
        checkpoint = campaign / "checkpoints" / Path(training["checkpoint_path"]).name
        if sha256_file(checkpoint) != training["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch: {run_id}")
        evaluation = read_immutable_json(record.envelope_path)
        provenance.append(
            {
                "run_id": run_id,
                "training_sha256": sha256_file(training_path),
                "evaluation_sha256": sha256_file(record.envelope_path),
                "checkpoint_sha256": training["checkpoint_sha256"],
                "probabilities_sha256": evaluation["probabilities_sha256"],
            }
        )
    summary = aggregate(records, config)
    nonlandmark = []
    metrics = (
        "conditioned_tvd",
        "interpolation_tvd",
        "oracle_tvd",
        "generalization_advantage",
        "sharing_cost",
        "profile_tvd",
    )
    for n in (8, 12):
        for r in (0, 1, 3, 6, 10):
            for seed in range(100, 110):
                rows = [
                    v
                    for v in summary["condition_rows"]
                    if v["dataset"] == "curie_weiss"
                    and v["visible_qubits"] == n
                    and v["ancillas"] == r
                    and v["seed"] == seed
                    and v["split"] == "held_out"
                    and not np.isclose(v["condition"], 1)
                ]
                if len(rows) != 4:
                    raise ValueError("non-landmark comparison requires four conditions")
                nonlandmark.append(
                    dict(
                        dataset="curie_weiss",
                        visible_qubits=n,
                        ancillas=r,
                        seed=seed,
                        group="curie_nonlandmark",
                        **{m: float(np.mean([v[m] for v in rows])) for m in metrics},
                    )
                )
    summary["nonlandmark_seed_rows"] = nonlandmark
    for n in (8, 12):
        for r in (0, 1, 3, 6, 10):
            rows = [
                v
                for v in nonlandmark
                if v["visible_qubits"] == n and v["ancillas"] == r
            ]
            summary["summaries"][f"curie_weiss/{n}/{r}/curie_nonlandmark"] = {
                m: summarize_seed_values([v[m] for v in rows], seed=20260904 + i)
                for i, m in enumerate(metrics)
            }
    output.mkdir(parents=True, exist_ok=True)
    write_summary_tables(summary, output / "summary.json", output / "seed_summary.csv")
    audit = {
        "manifest_sha256": sha256_file(manifest),
        "record_count": len(records),
        "scope": {"datasets": ["gaussian", "curie_weiss"], "visible_qubits": [8, 12]},
        "records": sorted(provenance, key=lambda v: v["run_id"]),
    }
    (output / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verified": len(records),
                "provenance_sha256": hashlib.sha256(
                    (output / "provenance.json").read_bytes()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, default=Path("results/exact-tvd-ancilla-v1")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/exact_tvd_ancilla_v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/paper_complete_20260904")
    )
    args = parser.parse_args()
    process(args.campaign, args.manifest, args.output)
