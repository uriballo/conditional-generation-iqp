"""Summarize and audit the frozen symmetry-aligned fidelity experiment."""

# ruff: noqa: E501 -- Markdown table rows are intentionally kept as single strings.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iqp_cond.interpolation import HELDOUT_ANGLE_FRACTIONS
from iqp_cond.symmetry import (
    OUTPUT_PROTOCOLS,
    PERMUTATION_TWIRL,
    UNALIGNED,
    XOR_PAIR_CONTROL,
)
from iqp_cond.symmetry_fidelity import FROZEN_THRESHOLDS

PROTOCOL_LABELS = {
    UNALIGNED: "Unaligned",
    PERMUTATION_TWIRL: "Permutation twirl",
    XOR_PAIR_CONTROL: "XOR-pair control",
}
GUARDRAILS = {
    "shell_tvd": "maximum_shell_tvd_degradation",
    "mean_absolute_error": "maximum_mean_error_degradation",
    "std_absolute_error": "maximum_std_error_degradation",
}


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def _load(directory: Path) -> list[dict[str, object]]:
    records = [
        json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))
    ]
    for study in ("endpoint", "heldout"):
        for protocol in OUTPUT_PROTOCOLS:
            seeds = {
                record["seed"]
                for record in records
                if record["study"] == study and record["protocol"] == protocol
            }
            if seeds != set(range(10)):
                raise SystemExit(
                    f"expected {study}/{protocol} seeds 0-9; found {sorted(seeds)}",
                )
    return records


def _by_key(
    records: list[dict[str, object]],
) -> dict[tuple[str, str, int], dict[str, object]]:
    return {
        (record["study"], record["protocol"], record["seed"]): record
        for record in records
    }


def _endpoint_values(
    indexed: dict[tuple[str, str, int], dict[str, object]],
    protocol: str,
    condition: str,
    metric: str,
) -> list[float]:
    values = []
    for seed in range(10):
        record = indexed[("endpoint", protocol, seed)]
        row = next(
            item for item in record["conditions"] if item["condition"] == condition
        )
        values.append(float(row["angle"][metric]))
    return values


def _heldout_values(
    indexed: dict[tuple[str, str, int], dict[str, object]],
    protocol: str,
    fraction: float,
    model: str,
    metric: str,
) -> list[float]:
    values = []
    for seed in range(10):
        record = indexed[("heldout", protocol, seed)]
        row = next(
            item
            for item in record["points"]
            if np.isclose(item["angle_fraction_pi"], fraction)
        )
        values.append(float(row[model][metric]))
    return values


def _material_comparison(
    label: str,
    unaligned: list[float],
    aligned: list[float],
) -> dict[str, object]:
    unaligned_median = float(np.median(unaligned))
    aligned_median = float(np.median(aligned))
    absolute = unaligned_median - aligned_median
    relative = absolute / unaligned_median
    return {
        "label": label,
        "unaligned_median": unaligned_median,
        "aligned_median": aligned_median,
        "absolute_reduction": absolute,
        "relative_reduction": relative,
        "per_seed_reductions": [
            raw - transformed
            for raw, transformed in zip(unaligned, aligned, strict=True)
        ],
        "passed": (
            absolute >= FROZEN_THRESHOLDS["absolute_bitstring_tvd_reduction"]
            and relative >= FROZEN_THRESHOLDS["relative_bitstring_tvd_reduction"]
        ),
    }


def _guardrail_comparison(
    label: str,
    metric: str,
    unaligned: list[float],
    aligned: list[float],
) -> dict[str, object]:
    degradation = float(np.median(aligned) - np.median(unaligned))
    threshold = FROZEN_THRESHOLDS[GUARDRAILS[metric]]
    return {
        "label": label,
        "metric": metric,
        "unaligned_median": float(np.median(unaligned)),
        "aligned_median": float(np.median(aligned)),
        "degradation": degradation,
        "maximum_degradation": threshold,
        "passed": degradation <= threshold,
    }


def _endpoint_audit_cells(
    indexed: dict[tuple[str, str, int], dict[str, object]],
    protocol: str,
) -> list[dict[str, object]]:
    cells = []
    for condition in ("A", "B"):
        for metric in ("shell_tvd", "bitstring_tvd"):
            angle = _endpoint_values(indexed, protocol, condition, metric)
            controls = defaultdict(list)
            for seed in range(10):
                record = indexed[("endpoint", protocol, seed)]
                row = next(
                    item
                    for item in record["conditions"]
                    if item["condition"] == condition
                )
                for name, values in row["frozen_unaligned_controls"].items():
                    controls[name].append(float(values[metric]))
            angle_median = float(np.median(angle))
            control_medians = {
                name: float(np.median(values)) for name, values in controls.items()
            }
            cells.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "angle": angle_median,
                    "controls": control_medians,
                    "passed": all(
                        angle_median < value for value in control_medians.values()
                    ),
                },
            )
    return cells


def _heldout_audit_cells(
    indexed: dict[tuple[str, str, int], dict[str, object]],
    protocol: str,
) -> list[dict[str, object]]:
    cells = []
    for metric in ("shell_tvd", "bitstring_tvd"):
        for fraction in HELDOUT_ANGLE_FRACTIONS:
            dense = _heldout_values(indexed, protocol, fraction, "dense", metric)
            oracle = _heldout_values(
                indexed,
                protocol,
                fraction,
                "oracle_adjacent",
                metric,
            )
            cells.append(
                {
                    "test": "continuous_conditioning",
                    "metric": metric,
                    "angle_fraction_pi": float(fraction),
                    "dense": float(np.median(dense)),
                    "baseline": float(np.median(oracle)),
                    "passed": float(np.median(dense)) < float(np.median(oracle)),
                },
            )
        dense_means = []
        endpoint_means = []
        for seed in range(10):
            record = indexed[("heldout", protocol, seed)]
            dense_means.append(
                np.mean([point["dense"][metric] for point in record["points"]])
            )
            endpoint_means.append(
                np.mean([point["endpoint"][metric] for point in record["points"]]),
            )
        cells.append(
            {
                "test": "dense_supervision",
                "metric": metric,
                "angle_fraction_pi": None,
                "dense": float(np.median(dense_means)),
                "baseline": float(np.median(endpoint_means)),
                "passed": float(np.median(dense_means))
                < float(np.median(endpoint_means)),
            },
        )
    return cells


def build_audit(records: list[dict[str, object]]) -> dict[str, object]:
    indexed = _by_key(records)
    primary = []
    xor_primary = []
    for condition in ("A", "B"):
        raw = _endpoint_values(indexed, UNALIGNED, condition, "bitstring_tvd")
        primary.append(
            _material_comparison(
                f"endpoint Condition {condition}",
                raw,
                _endpoint_values(
                    indexed,
                    PERMUTATION_TWIRL,
                    condition,
                    "bitstring_tvd",
                ),
            ),
        )
        xor_primary.append(
            _material_comparison(
                f"endpoint Condition {condition}",
                raw,
                _endpoint_values(
                    indexed,
                    XOR_PAIR_CONTROL,
                    condition,
                    "bitstring_tvd",
                ),
            ),
        )
    for fraction in HELDOUT_ANGLE_FRACTIONS:
        raw = _heldout_values(indexed, UNALIGNED, fraction, "dense", "bitstring_tvd")
        primary.append(
            _material_comparison(
                f"Held-out alpha/pi={fraction:.5f}",
                raw,
                _heldout_values(
                    indexed,
                    PERMUTATION_TWIRL,
                    fraction,
                    "dense",
                    "bitstring_tvd",
                ),
            ),
        )
        xor_primary.append(
            _material_comparison(
                f"Held-out alpha/pi={fraction:.5f}",
                raw,
                _heldout_values(
                    indexed,
                    XOR_PAIR_CONTROL,
                    fraction,
                    "dense",
                    "bitstring_tvd",
                ),
            ),
        )

    guardrails = []
    for condition in ("A", "B"):
        for metric in GUARDRAILS:
            guardrails.append(
                _guardrail_comparison(
                    f"endpoint Condition {condition}",
                    metric,
                    _endpoint_values(indexed, UNALIGNED, condition, metric),
                    _endpoint_values(
                        indexed,
                        PERMUTATION_TWIRL,
                        condition,
                        metric,
                    ),
                ),
            )
    for fraction in HELDOUT_ANGLE_FRACTIONS:
        for metric in GUARDRAILS:
            guardrails.append(
                _guardrail_comparison(
                    f"Held-out alpha/pi={fraction:.5f}",
                    metric,
                    _heldout_values(indexed, UNALIGNED, fraction, "dense", metric),
                    _heldout_values(
                        indexed,
                        PERMUTATION_TWIRL,
                        fraction,
                        "dense",
                        metric,
                    ),
                ),
            )

    endpoint_unaligned = _endpoint_audit_cells(indexed, UNALIGNED)
    endpoint_aligned = _endpoint_audit_cells(indexed, PERMUTATION_TWIRL)
    heldout_unaligned = _heldout_audit_cells(indexed, UNALIGNED)
    heldout_aligned = _heldout_audit_cells(indexed, PERMUTATION_TWIRL)
    new_failures = []
    for raw, aligned in zip(
        endpoint_unaligned + heldout_unaligned,
        endpoint_aligned + heldout_aligned,
        strict=True,
    ):
        if raw["passed"] and not aligned["passed"]:
            new_failures.append(aligned)

    primary_passed = all(item["passed"] for item in primary)
    guardrails_passed = all(item["passed"] for item in guardrails)
    xor_complete_rule = all(item["passed"] for item in xor_primary)
    return {
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "overall_passed": (
            primary_passed
            and guardrails_passed
            and not new_failures
            and not xor_complete_rule
        ),
        "primary_passed": primary_passed,
        "guardrails_passed": guardrails_passed,
        "no_new_audit_failures": not new_failures,
        "xor_control_complete_rule": xor_complete_rule,
        "primary_comparisons": primary,
        "xor_control_comparisons": xor_primary,
        "guardrail_comparisons": guardrails,
        "endpoint_unaligned_audit": endpoint_unaligned,
        "endpoint_aligned_audit": endpoint_aligned,
        "heldout_unaligned_audit": heldout_unaligned,
        "heldout_aligned_audit": heldout_aligned,
        "new_audit_failures": new_failures,
    }


def _metrics_markdown(records: list[dict[str, object]]) -> str:
    indexed = _by_key(records)
    lines = [
        "# Symmetry-aligned fidelity metrics",
        "",
        "Median [interquartile range] over ten matched seeds.",
        "",
        "## Endpoint Conditions",
        "",
        "| Protocol | Condition | Profile TVD | Bitstring TVD | Mean error | Std. error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for protocol in OUTPUT_PROTOCOLS:
        for condition in ("A", "B"):
            values = {
                metric: _median_iqr(
                    _endpoint_values(indexed, protocol, condition, metric)
                )
                for metric in (
                    "shell_tvd",
                    "bitstring_tvd",
                    "mean_absolute_error",
                    "std_absolute_error",
                )
            }
            cells = [
                f"{item[0]:.4f} [{item[1]:.4f}, {item[2]:.4f}]"
                for item in values.values()
            ]
            lines.append(
                f"| {PROTOCOL_LABELS[protocol]} | {condition} | "
                + " | ".join(cells)
                + " |",
            )
    lines.extend(
        [
            "",
            "## Held-out angles",
            "",
            "| Protocol | Angle / pi | Profile TVD | Bitstring TVD | Mean error | Std. error |",
            "|---|---:|---:|---:|---:|---:|",
        ],
    )
    for protocol in OUTPUT_PROTOCOLS:
        for fraction in HELDOUT_ANGLE_FRACTIONS:
            values = {
                metric: _median_iqr(
                    _heldout_values(indexed, protocol, fraction, "dense", metric),
                )
                for metric in (
                    "shell_tvd",
                    "bitstring_tvd",
                    "mean_absolute_error",
                    "std_absolute_error",
                )
            }
            cells = [
                f"{item[0]:.4f} [{item[1]:.4f}, {item[2]:.4f}]"
                for item in values.values()
            ]
            lines.append(
                f"| {PROTOCOL_LABELS[protocol]} | {fraction:.5f} | "
                + " | ".join(cells)
                + " |",
            )
    return "\n".join(lines) + "\n"


def _audit_markdown(audit: dict[str, object]) -> str:
    lines = [
        "# Frozen symmetry-aligned fidelity audit",
        "",
        f"**Overall result: {'PASS' if audit['overall_passed'] else 'FAIL'}.**",
        "",
        "## Material Bitstring-TVD reduction",
        "",
        "| Evaluation | Unaligned | Permutation twirl | Absolute reduction | Relative reduction | Result |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in audit["primary_comparisons"]:
        lines.append(
            f"| {item['label']} | {item['unaligned_median']:.4f} | "
            f"{item['aligned_median']:.4f} | {item['absolute_reduction']:.4f} | "
            f"{100 * item['relative_reduction']:.1f}% | "
            f"{'PASS' if item['passed'] else 'FAIL'} |",
        )
    lines.extend(
        [
            "",
            "## Decision-rule components",
            "",
            f"- Primary thresholds: {'PASS' if audit['primary_passed'] else 'FAIL'}.",
            f"- Profile/moment guardrails: {'PASS' if audit['guardrails_passed'] else 'FAIL'}.",
            f"- No new conditional-control audit failures: {'PASS' if audit['no_new_audit_failures'] else 'FAIL'}.",
            f"- XOR-pair control does not pass the complete primary rule: {'PASS' if not audit['xor_control_complete_rule'] else 'FAIL'}.",
            "",
            "## Held-out permutation-twirl audit",
            "",
            "| Test | Metric | Angle / pi | Dense | Baseline | Result |",
            "|---|---|---:|---:|---:|---:|",
        ],
    )
    for item in audit["heldout_aligned_audit"]:
        angle = (
            "mean"
            if item["angle_fraction_pi"] is None
            else f"{item['angle_fraction_pi']:.5f}"
        )
        lines.append(
            f"| {item['test'].replace('_', ' ')} | {item['metric'].replace('_', ' ')} | "
            f"{angle} | {item['dense']:.4f} | {item['baseline']:.4f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |",
        )
    return "\n".join(lines) + "\n"


def _capacity_markdown() -> str:
    return """# Symmetry intervention capacity and budget

| Protocol | Trainable parameters | Training updates | Checkpoint | Random actions/sample | Exact evaluation |
|---|---:|---:|---|---:|---:|
| Unaligned | 190 shared | 2,000 | Frozen Validation minimum | 0 | O(2^19) |
| Permutation twirl | 190 shared | 2,000 | Identical checkpoint | 1 | O(2^19 + 19) |
| XOR-pair control | 190 shared | 2,000 | Identical checkpoint | 1 | O(2^19) |

All protocols reuse the same Training data, Validation data, random seeds,
optimizer history, and selected parameter vector. The Condition enters only
through the initial Tractable product-state encoding.
"""


def _paper_table_tex(audit: dict[str, object]) -> str:
    """Render the compact manuscript table from the frozen primary audit."""
    lines = []
    for item in audit["primary_comparisons"]:
        label = str(item["label"])
        if label.startswith("endpoint Condition "):
            display = label.replace("endpoint Condition ", "Endpoint ")
        else:
            angle = label.removeprefix("Held-out alpha/pi=")
            display = rf"Held-out ${angle}\pi$"
        lines.append(
            f"{display} & {item['unaligned_median']:.3f} & "
            f"{item['aligned_median']:.3f} & "
            f"{100 * item['relative_reduction']:.1f}\\% \\\\%",
        )
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def _interpretation_markdown(audit: dict[str, object]) -> str:
    status = "passes" if audit["overall_passed"] else "fails"
    return f"""# Symmetry-aligned fidelity interpretation

## Result

The frozen decision rule **{status}**. Permutation averaging is interpreted as
a generation protocol layered on the unchanged shared-parameter Conditional
IQP model, not as evidence that its untwirled circuit learned uniform
within-shell probabilities. The strict Held-out audit remains separately
reported; a material fidelity reduction does not erase any inherited
conditional-interpolation failure.

The XOR-pair control {"does" if audit["xor_control_complete_rule"] else "does not"}
pass the complete primary rule. This {"weakens" if audit["xor_control_complete_rule"] else "supports"}
the mechanism-specific attribution to the Target's permutation symmetry.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Permutation averaging materially reduces exact Bitstring TVD at every endpoint and Held-out Condition. | Frozen per-Condition 0.15 absolute and 40% relative audit. | {"Supported" if audit["primary_passed"] else "Rejected"} |
| The intervention preserves conditional Hamming-weight control. | Exact profile, mean, and width guardrails plus no-new-failure audit. | {"Supported" if audit["guardrails_passed"] and audit["no_new_audit_failures"] else "Rejected"} |
| The gain is specific to Target-symmetry projection rather than arbitrary random output processing. | Capacity/budget-matched XOR-pair control. | {"Supported" if not audit["xor_control_complete_rule"] else "Needs further evidence"} |
| The untwirled circuit recovers the full Bitstring distribution. | The intervention is post-measurement and the unaligned TVDs remain unchanged. | Rejected |
| The result extends beyond permutation-invariant Gaussian Hamming-weight Targets. | No additional Target family is evaluated here. | Not claimed |

## Five-dimension self-review

- **Contribution:** isolates a symmetry mechanism with identical parameters,
  checkpoints, and training budgets.
- **Writing clarity:** distinguishes circuit fidelity from fidelity after the
  stochastic output protocol.
- **Experimental strength:** reports ten matched seeds, exact distributions,
  a target-misaligned randomization control, and inherited failures.
- **Evaluation completeness:** includes endpoints, Held-out Conditions,
  profile/Bitstring metrics, moments, and strict control audits.
- **Method soundness:** the intervention preserves input-only Condition
  encoding and does not alter the classical observable estimator or sampling
  claim boundary.

The compact manuscript remains frozen. A manuscript update should be proposed
only as a scoped post-measurement symmetry result, with the unchanged
untwirled-fidelity and Held-out-audit limitations stated explicitly.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("results/symmetry/runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/symmetry/summary"),
    )
    args = parser.parse_args()
    records = _load(args.runs)
    audit = build_audit(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.md").write_text(_metrics_markdown(records))
    (args.output / "audit.md").write_text(_audit_markdown(audit))
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "capacity_budget.md").write_text(_capacity_markdown())
    (args.output / "interpretation.md").write_text(_interpretation_markdown(audit))
    (args.output / "paper_table.tex").write_text(_paper_table_tex(audit))


if __name__ == "__main__":
    main()
