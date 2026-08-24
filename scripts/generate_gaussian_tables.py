"""Generate deterministic LaTeX tables from the complete Gaussian report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/gaussian-foundations-v1/report/complete-report.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/generated"))
    return parser


def _interval(interval: dict[str, float]) -> str:
    return (
        f"{interval['estimate']:+.4f} "
        f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}]"
    )


def _status(interval: dict[str, float]) -> str:
    if interval["lower"] > 0.0:
        return "supported"
    if interval["upper"] < 0.0:
        return "unsupported"
    return "mixed"


def _regime_label(regime: str) -> str:
    return "fixed per condition" if regime == "fixed_per_peak" else "fixed total"


def _comparison_label(name: str) -> str:
    return (
        name.replace("_", " ")
        .replace("single peak", "independent reference")
        .replace("lookup", "neighbor oracle")
    )


def _mean_metric(cell: dict[str, object], role: str, metric: str) -> float:
    return float(
        np.mean(
            [
                row["conditioned"][metric]
                for seed in cell["seed_rows"]
                for row in seed[role]
            ]
        )
    )


def _worst_condition(cell: dict[str, object], role: str, metric: str) -> float:
    rows = cell["seed_rows"]
    count = len(rows[0][role])
    return max(
        float(np.mean([seed[role][index]["conditioned"][metric] for seed in rows]))
        for index in range(count)
    )


def _write(path: Path, content: str) -> None:
    encoded = (content.rstrip() + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"immutable generated table differs: {path}")
    if not path.exists():
        path.write_bytes(encoded)


def _factorial(cells: list[dict[str, object]]) -> str:
    lines = []
    keys = [
        "training_no_condition_minus_conditioned_profile_tvd",
        "heldout_lookup_minus_conditioned_profile_tvd",
        "heldout_mixture_minus_conditioned_profile_tvd",
    ]
    for cell in cells:
        intervals = [cell["paired_intervals"][key] for key in keys]
        lines.append(
            f"{cell['n_qubits']} & {float(cell['width_factor']):.2f} & "
            + " & ".join(_interval(interval) for interval in intervals)
            + " & "
            + "/".join(_status(interval)[0].upper() for interval in intervals)
            + r" \\"
        )
    return "\n".join(lines)


def _factorial_wins(cells: list[dict[str, object]]) -> str:
    keys = (
        "training_vs_no_condition_profile_tvd",
        "heldout_vs_lookup_profile_tvd",
        "heldout_vs_mixture_profile_tvd",
    )
    return "\n".join(
        f"{cell['n_qubits']} & {float(cell['width_factor']):.2f} & "
        + " & ".join(
            f"{cell['cell_wins'][key]['wins']}/{cell['cell_wins'][key]['total']}"
            for key in keys
        )
        + r" \\"
        for cell in cells
    )


def _condition_count(cells: list[dict[str, object]]) -> str:
    lines = []
    for cell in cells:
        count = int(cell["peak_count"])
        regime = str(cell["data_regime"])
        if regime == "fixed_per_peak":
            train, validation = 10_000, 2_000
        else:
            train, validation = 18_000 // count, 3_600 // count
        accounting = cell["parameter_accounting"]
        lines.append(
            f"{count} & {_regime_label(regime)} & {train:,} & {validation:,} & "
            f"{_mean_metric(cell, 'held_out', 'profile_tvd'):.4f} & "
            f"{_worst_condition(cell, 'held_out', 'profile_tvd'):.4f} & "
            f"{accounting['shared_parameters']} & "
            f"{accounting['shared_optimizer_updates']:,} & "
            f"{accounting['single_peak_bank_optimizer_updates']:,}" + r" \\"
        )
    return "\n".join(lines)


def _condition_count_per_condition(cells: list[dict[str, object]]) -> str:
    lines = []
    for cell in cells:
        for condition_index, first_row in enumerate(cell["seed_rows"][0]["held_out"]):
            means = {
                method: float(
                    np.mean(
                        [
                            seed["held_out"][condition_index][method]["profile_tvd"]
                            for seed in cell["seed_rows"]
                        ]
                    )
                )
                for method in (
                    "conditioned",
                    "no_condition",
                    "adjacent_lookup",
                    "barycentric_mixture",
                )
            }
            lines.append(
                f"{cell['peak_count']} & "
                f"{_regime_label(str(cell['data_regime']))} & "
                f"{float(first_row['condition']):.4f} & "
                f"{means['conditioned']:.4f} & {means['no_condition']:.4f} & "
                f"{means['adjacent_lookup']:.4f} & "
                f"{means['barycentric_mixture']:.4f}" + r" \\"
            )
    return "\n".join(lines)


def _condition_count_intervals(cells: list[dict[str, object]]) -> str:
    keys = (
        "training_no_condition_minus_conditioned_profile_tvd",
        "training_conditioned_minus_single_peak_profile_tvd",
        "heldout_lookup_minus_conditioned_profile_tvd",
        "heldout_mixture_minus_conditioned_profile_tvd",
    )
    return "\n".join(
        f"{cell['peak_count']} & {_regime_label(str(cell['data_regime']))} & "
        + " & ".join(_interval(cell["paired_intervals"][key]) for key in keys)
        + r" \\"
        for cell in cells
    )


def _raw_bitstring(cells: list[dict[str, object]]) -> str:
    lines = []
    for cell in cells:
        intervals = cell["paired_intervals"]
        training = intervals["training_no_condition_minus_conditioned_bitstring_tvd"]
        heldout = intervals["heldout_lookup_minus_conditioned_bitstring_tvd"]
        lines.append(
            f"{cell['n_qubits']} & {float(cell['width_factor']):.2f} & "
            f"{_mean_metric(cell, 'training', 'bitstring_tvd'):.4f} & "
            f"{_mean_metric(cell, 'held_out', 'bitstring_tvd'):.4f} & "
            f"{_interval(training)} & {_interval(heldout)}" + r" \\"
        )
    return "\n".join(lines)


def _secondary_diagnostics(cells: list[dict[str, object]]) -> str:
    lines = []
    for cell in cells:
        values = [
            _mean_metric(cell, role, metric)
            for metric in (
                "mean_absolute_error",
                "width_absolute_error",
                "conditional_mmd",
            )
            for role in ("training", "held_out")
        ]
        lines.append(
            f"{cell['n_qubits']} & {float(cell['width_factor']):.2f} & "
            + " & ".join(f"{value:.4f}" for value in values)
            + r" \\"
        )
    return "\n".join(lines)


def _all_intervals(cells: list[dict[str, object]]) -> str:
    lines = []
    for cell in cells:
        cell_id = (
            f"$n={cell['n_qubits']}$, $s={float(cell['width_factor']):.2f}\\sqrt n$"
        )
        for name, interval in sorted(cell["paired_intervals"].items()):
            readable_name = _comparison_label(name)
            lines.append(
                f"{cell_id} & {readable_name} & "
                f"{_interval(interval)} & {_status(interval)}" + r" \\"
            )
    return "\n".join(lines)


def _all_intervals_paginated(cells: list[dict[str, object]]) -> str:
    blocks = []
    for cell in cells:
        n_qubits = int(cell["n_qubits"])
        width = float(cell["width_factor"])
        rows = []
        for name, interval in sorted(cell["paired_intervals"].items()):
            readable_name = _comparison_label(name)
            rows.append(
                f"{readable_name} & {_interval(interval)} & {_status(interval)}"
                + r" \\"
            )
        blocks.append(
            "\n".join(
                [
                    r"\clearpage",
                    r"\begin{table}[p]",
                    r"\centering",
                    r"\scriptsize",
                    r"\setlength{\tabcolsep}{4pt}",
                    (
                        r"\caption{Paired intervals for $n="
                        f"{n_qubits}"
                        r"$ and $s="
                        f"{width:.2f}"
                        r"\sqrt n$. Positive effects favor the conditioned "
                        r"model under the named subtraction.}"
                    ),
                    f"\\label{{tab:all-intervals-n{n_qubits}-w{width:.2f}}}",
                    r"\begin{tabular}{@{}p{0.48\textwidth}"
                    r"p{0.29\textwidth}p{0.15\textwidth}@{}}",
                    r"\toprule",
                    r"Comparison & Effect [95\% interval] & Status\\",
                    r"\midrule",
                    *rows,
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\end{table}",
                ]
            )
        )
    return "\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = read_immutable_json(args.report)
    if report.get("completed_atomic_models") != 1_020:
        raise RuntimeError("tables require the complete 1,020-model report")
    factorial = report["factorial_cells"]
    _write(args.output / "factorial_cells.tex", _factorial(factorial))
    _write(args.output / "factorial_wins.tex", _factorial_wins(factorial))
    _write(
        args.output / "condition_count_cells.tex",
        _condition_count(report["condition_count_cells"]),
    )
    _write(
        args.output / "condition_count_per_condition.tex",
        _condition_count_per_condition(report["condition_count_cells"]),
    )
    _write(
        args.output / "condition_count_intervals.tex",
        _condition_count_intervals(report["condition_count_cells"]),
    )
    _write(args.output / "raw_bitstring_cells.tex", _raw_bitstring(factorial))
    _write(
        args.output / "secondary_diagnostics.tex",
        _secondary_diagnostics(factorial),
    )
    _write(args.output / "all_intervals.tex", _all_intervals(factorial))
    _write(
        args.output / "all_intervals_paginated.tex",
        _all_intervals_paginated(factorial),
    )
    print(json.dumps({"factorial_cells": 12, "condition_count_cells": 8}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
