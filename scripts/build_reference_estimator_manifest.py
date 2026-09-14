#!/usr/bin/env python3
"""Build deterministic fixtures for the reference-estimator precision study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from iqp_cond.ansatz import complete_ansatz


def observable_masks(n: int) -> list[list[int]]:
    """Keep identity/forbidden controls and distinct even-block probes."""
    m = n // 2
    subsets = [
        [],
        [0],
        [0, m],
        [0, 1],
        [m, m + 1],
        [0, 1, m, m + 1],
        list(range(m)),
        list(range(n)),
    ]
    rng = np.random.default_rng(61000 + n)
    masks = []
    for ids in subsets:
        row = np.zeros(n, dtype=int)
        row[ids] = 1
        if row.tolist() not in masks:
            masks.append(row.tolist())
    while len(masks) < 16:
        row = np.zeros(n, dtype=int)
        for start in [0, m]:
            k = int(rng.choice([0, 2, 4]))
            row[start + rng.choice(m, size=k, replace=False)] = 1
        if row.tolist() not in masks:
            masks.append(row.tolist())
    return masks


def build() -> dict:
    cases = []
    for n in [8, 12, 16]:
        g = np.asarray(complete_ansatz(n).generators, dtype=int)
        for ci, c in enumerate([[0.1, 0.7], [0.4, 0.4], [0.7, 0.1]]):
            coords = np.repeat(c, n // 2)
            for family, encoding in [
                ("product", "angle"),
                ("cat", "two_block_branch"),
                ("bond", "two_ring_bond"),
            ]:
                if family == "product":
                    angles = np.pi * (0.35 - 0.7 * coords)
                elif family == "cat":
                    angles = np.pi * (0.225 - 0.20 * (coords - 0.1) / 0.6)
                else:
                    angles = coords
                for pi, scale in enumerate([0.1, 0.4]):
                    params = np.random.default_rng(72000 + n * 10 + pi).normal(
                        0, scale, len(g)
                    )
                    cases.append(
                        dict(
                            case_id=f"n{n}_{family}_c{ci}_p{pi}",
                            encoder=family,
                            n_qubits=n,
                            encoding=encoding,
                            angles=angles.tolist(),
                            generators=g.tolist(),
                            params=params.tolist(),
                            ops=observable_masks(n),
                            phase_regime=["initial_scale", "dispersed"][pi],
                            condition=c,
                        )
                    )
    return dict(
        study_id="reference_estimator_precision_v1",
        sample_budgets=[128, 512, 2048, 8192],
        repetitions=32,
        root_seed=20260907,
        cases=cases,
        notes=[
            "No ancillas; complete one/two-body phase bank in synthetic fixtures.",
            "Shared phases/masks across encoder families, matched within width.",
            "cat is the two-block symmetry-only input used in the spatial study.",
            "16 fixed masks; identity/parity-forbidden controls reported separately.",
            "All phase derivatives retained; error metrics are absolute.",
            "No mask sampling; no MMD or optimization/generalization claim follows.",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trained", type=Path)
    args = parser.parse_args()
    manifest = build()
    if args.trained:
        manifest["cases"].extend(json.loads(args.trained.read_text()))
        manifest["trained_fixture_sha256"] = hashlib.sha256(
            args.trained.read_bytes()
        ).hexdigest()
    payload = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    if args.output.exists() and args.output.read_text() != payload:
        raise RuntimeError("Refusing to overwrite a different manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(len(manifest["cases"]), "cases", hashlib.sha256(payload.encode()).hexdigest())


if __name__ == "__main__":
    main()
