"""Summarize matched angle and reupload interpolation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _seed(path: Path) -> int:
    return int(path.stem.rsplit("seed", maxsplit=1)[1])


def _metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    heldout = [point for point in payload["points"] if point["role"] == "heldout"]
    return {
        "heldout_mmd": float(payload["heldout_mmd"]),
        "shell_tvd": float(np.mean([point["shell_tvd"] for point in heldout])),
        "bitstring_tvd": float(
            np.mean([point["bitstring_tvd"] for point in heldout]),
        ),
        "mean_error": float(
            np.mean([point["mean_absolute_error"] for point in heldout]),
        ),
        "std_error": float(
            np.mean([point["std_absolute_error"] for point in heldout]),
        ),
    }


def main() -> None:
    """Write paired aggregate metrics for all completed seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--angle-runs",
        type=Path,
        default=Path("results/interpolation/runs"),
    )
    parser.add_argument(
        "--reupload-runs",
        type=Path,
        default=Path("results/encoding/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/encoding/summary"),
    )
    args = parser.parse_args()

    angle_paths = {
        _seed(path): path
        for path in args.angle_runs.glob("interpolation_complete_seed*.json")
    }
    reupload_paths = {
        _seed(path): path
        for path in args.reupload_runs.glob(
            "interpolation_symmetric_reupload_complete_seed*.json",
        )
    }
    seeds = sorted(angle_paths.keys() & reupload_paths.keys())
    if not seeds:
        raise ValueError("no matched angle/reupload seeds found")

    per_seed = []
    for seed in seeds:
        angle = _metrics(angle_paths[seed])
        reupload = _metrics(reupload_paths[seed])
        per_seed.append(
            {
                "seed": seed,
                "angle": angle,
                "symmetric_reupload": reupload,
                "reupload_minus_angle": {
                    name: reupload[name] - angle[name] for name in angle
                },
            },
        )

    names = list(per_seed[0]["angle"])
    aggregate = {}
    for model in ("angle", "symmetric_reupload", "reupload_minus_angle"):
        aggregate[model] = {}
        for name in names:
            values = np.asarray([row[model][name] for row in per_seed])
            aggregate[model][name] = {
                "mean": float(values.mean()),
                "sample_std": (
                    float(values.std(ddof=1)) if values.size > 1 else None
                ),
            }
    wins = {
        name: int(
            sum(
                row["symmetric_reupload"][name] < row["angle"][name]
                for row in per_seed
            ),
        )
        for name in names
    }

    payload = {
        "matched_seeds": seeds,
        "lower_is_better": names,
        "wins_by_symmetric_reupload": wins,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Encoding comparison",
        "",
        f"Matched converged seeds: {', '.join(map(str, seeds))}. Lower is better.",
        "",
        "| Metric | Angle | Symmetric reupload | Paired difference | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        angle = aggregate["angle"][name]["mean"]
        reupload = aggregate["symmetric_reupload"][name]["mean"]
        difference = aggregate["reupload_minus_angle"][name]["mean"]
        lines.append(
            f"| {name} | {angle:.6f} | {reupload:.6f} | "
            f"{difference:+.6f} | {wins[name]}/{len(seeds)} |",
        )
    lines.extend(
        [
            "",
            "The encodings are identical at the original endpoint angles. "
            "Differences therefore measure the learned/interpolated conditional "
            "path, not endpoint distinguishability.",
            "",
        ],
    )
    (args.output / "metrics.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
