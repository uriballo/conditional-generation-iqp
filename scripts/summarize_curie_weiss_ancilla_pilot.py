#!/usr/bin/env python3
"""Summarize the immutable three-seed ancilla-count development study."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json, sha256_file, write_immutable_json
from iqp_cond.protocol import content_digest


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    store = repository / "results/curie-weiss-ancilla-pilot-v1/store"
    manifest_path = store / "manifest.json"
    manifest = read_immutable_json(manifest_path)
    manifest_digest = content_digest(manifest)
    values: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    seed_values: dict[tuple[int, str, str, str, int], list[float]] = defaultdict(list)
    for path in sorted((store / "results").glob("*.json")):
        result = read_immutable_json(path)
        run = result["run"]
        if result["manifest_digest"] != manifest_digest:
            raise RuntimeError("pilot result belongs to another manifest")
        if run["ancillas"] not in (0, 1, 3) or run["optimizer_seed"] not in (0, 1, 2):
            raise RuntimeError("pilot contains an undeclared development coordinate")
        if run["family"] not in ("conditioned", "no_condition"):
            raise RuntimeError("pilot contains a non-development family")
        for row in result["metrics"]["evaluation_rows"]:
            for metric in ("profile_tvd", "bitstring_tvd"):
                key = (run["ancillas"], run["family"], row["role"], metric)
                values[key].append(float(row[metric]))
                seed_values[(*key, run["optimizer_seed"])].append(float(row[metric]))
    if sum(len(v) for v in values.values()) != 396:
        raise RuntimeError("pilot requires exactly 18 fits and 198 evaluation rows")

    rows = []
    for ancillas in (0, 1, 3):
        for family in ("conditioned", "no_condition"):
            for role in ("training", "held_out"):
                rows.append(
                    {
                        "ancillas": ancillas,
                        "family": family,
                        "role": role,
                        "mean_profile_tvd": float(
                            np.mean(values[(ancillas, family, role, "profile_tvd")])
                        ),
                        "mean_bitstring_tvd": float(
                            np.mean(values[(ancillas, family, role, "bitstring_tvd")])
                        ),
                    }
                )
    gains = []
    for ancillas in (1, 3):
        for role in ("training", "held_out"):
            for metric in ("profile_tvd", "bitstring_tvd"):
                paired = []
                for seed in (0, 1, 2):
                    zero = np.mean(seed_values[(0, "conditioned", role, metric, seed)])
                    latent = np.mean(
                        seed_values[(ancillas, "conditioned", role, metric, seed)]
                    )
                    paired.append(float(zero - latent))
                gains.append(
                    {
                        "comparison": f"m{ancillas}_vs_m0_conditioned_{metric}_{role}",
                        "mean_gain": float(np.mean(paired)),
                        "seed_gains": paired,
                    }
                )
    summary = {
        "schema_version": 1,
        "development_only": True,
        "optimization_claim": False,
        "pilot_manifest_path": str(manifest_path.relative_to(repository)),
        "pilot_manifest_sha256": sha256_file(manifest_path),
        "pilot_manifest_digest": manifest_digest,
        "completed_fits": 18,
        "tested_ancillas": [0, 1, 3],
        "tested_seeds": [0, 1, 2],
        "rows": rows,
        "paired_gains": gains,
        "selection_statement": (
            "The pilot supports adding latent capacity; m=5 was selected "
            "prospectively as a compute-capacity compromise and was not tested here."
        ),
    }
    output = repository / "results/curie-weiss-ancilla-pilot-v1/pilot-summary.json"
    if output.exists() and read_immutable_json(output) != summary:
        raise RuntimeError("immutable pilot summary differs")
    if not output.exists():
        write_immutable_json(output, summary)

    held_out = [row for row in rows if row["role"] == "held_out"]
    tex = (
        "\n".join(
            f"{row['ancillas']} & {row['family'].replace('_', ' ')} & "
            f"{row['mean_profile_tvd']:.3f} & {row['mean_bitstring_tvd']:.3f} \\\\"
            for row in held_out
        )
        + "\n"
    )
    tex_path = repository / "paper/generated/curie_weiss_ancilla_pilot.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    if tex_path.exists() and tex_path.read_text() != tex:
        raise RuntimeError("immutable pilot table differs")
    if not tex_path.exists():
        tex_path.write_text(tex)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
