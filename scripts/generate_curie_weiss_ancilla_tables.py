#!/usr/bin/env python3
"""Generate deterministic supplement tables for the ancilla campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"immutable generated table differs: {path}")
    if not path.exists():
        path.write_text(content)


def _effect(interval: dict) -> str:
    return (
        f"{interval['estimate']:+.3f} "
        f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "results/curie-weiss-m5-foundations-v2/report/campaign-report.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/generated"))
    args = parser.parse_args(argv)
    report = read_immutable_json(args.report)
    if report.get("completed_runs") != 80 or report.get("headline_ancillas") != 5:
        raise RuntimeError("tables require all 80 immutable fixed-r=5 results")

    comparisons = []
    for record in report["comparisons"]:
        label = record["comparison"].replace("m5", "r5").replace("_", r"\_")
        comparisons.append(
            f"\\texttt{{{label}}} & "
            f"{_effect(record['interval'])} & {record['interval']['status']} \\\\"
        )
    _write(
        args.output / "curie_weiss_ancilla_effects.tex",
        "\n".join(comparisons) + "\n",
    )

    rows = report["response_rows"]
    response = []
    for beta in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        selected = [
            row
            for row in rows
            if row["ancillas"] == 5 and np.isclose(row["beta"], beta)
        ]
        role = "train" if selected[0]["role"] == "training" else "held out"
        values = {}
        for family in ("conditioned", "no_condition"):
            for metric in ("profile_tvd", "bitstring_tvd"):
                values[(family, metric)] = np.mean(
                    [row[family][metric] for row in selected]
                )
        response.append(
            f"{beta:.1f} & {role} & "
            f"{values[('conditioned', 'profile_tvd')]:.3f} & "
            f"{values[('conditioned', 'bitstring_tvd')]:.3f} & "
            f"{values[('no_condition', 'profile_tvd')]:.3f} & "
            f"{values[('no_condition', 'bitstring_tvd')]:.3f} \\\\"
        )
    _write(
        args.output / "curie_weiss_ancilla_response.tex",
        "\n".join(response) + "\n",
    )

    exhaustive = []
    for row in sorted(
        rows, key=lambda item: (item["ancillas"], item["seed"], item["beta"])
    ):
        conditioned = row["conditioned"]
        no_condition = row["no_condition"]
        mmd = row["conditioned_exact_mmd_by_bandwidth"]
        exhaustive.append(
            f"{row['ancillas']} & {row['seed']} & {row['beta']:.1f} & "
            f"{row['role'].replace('_', ' ')} & "
            f"{conditioned['profile_tvd']:.4f} & "
            f"{conditioned['bitstring_tvd']:.4f} & "
            f"{no_condition['profile_tvd']:.4f} & "
            f"{no_condition['bitstring_tvd']:.4f} & "
            f"{mmd['1.0']:.5f} & {mmd['2.0']:.5f} & {mmd['4.0']:.5f} \\\\"
        )
    _write(
        args.output / "curie_weiss_ancilla_all_rows.tex",
        "\n".join(exhaustive) + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
