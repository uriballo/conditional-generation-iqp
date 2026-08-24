"""Frozen analysis for the role-separated Curie--Weiss encoding comparison."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_immutable_json, sha256_file, write_immutable_json
from .curie_weiss_ancilla_campaign import CurieWeissAncillaCampaignStore
from .curie_weiss_ancilla_protocol import (
    ENCODING_COMPARISON_VERSION,
    plan_curie_weiss_ancilla_runs,
)
from .gaussian_statistics import paired_bootstrap_interval

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_820
BITSTRING_MARGIN = 0.01
METRICS = (
    "profile_tvd",
    "bitstring_tvd",
    "exact_mmd",
    "absolute_magnetization",
    "energy_per_spin",
    "susceptibility",
    "heat_capacity_per_spin",
    "binder_cumulant",
)


def _interval(values: dict[int, float]) -> dict[str, Any]:
    return paired_bootstrap_interval(
        values,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    ).to_dict()


def _verify_baseline(store: CurieWeissAncillaCampaignStore) -> dict[str, Any]:
    repository = store.root.parents[2]
    reference = store.manifest.raw["baseline_reference"]
    manifest_path = repository / reference["manifest_path"]
    report_path = repository / reference["report_path"]
    if sha256_file(manifest_path) != reference["manifest_sha256"]:
        raise RuntimeError("baseline manifest digest differs")
    if sha256_file(report_path) != reference["report_sha256"]:
        raise RuntimeError("baseline report digest differs")
    report = read_immutable_json(report_path)
    if (
        report.get("campaign_version") != reference["campaign_version"]
        or report.get("completed_runs") != 80
    ):
        raise RuntimeError("baseline report is incomplete or incompatible")
    return report


def _baseline_rows(report: dict[str, Any]) -> dict[tuple[int, float], dict[str, Any]]:
    rows = {}
    for row in report["response_rows"]:
        if int(row["ancillas"]) == 5:
            metrics = dict(row["conditioned"])
            metrics["exact_mmd"] = float(
                np.mean(list(row["conditioned_exact_mmd_by_bandwidth"].values()))
            )
            rows[(int(row["seed"]), float(row["beta"]))] = metrics
    if len(rows) != 110:
        raise RuntimeError("baseline must contain ten seeds at eleven temperatures")
    return rows


def _new_rows(
    store: CurieWeissAncillaCampaignStore,
) -> dict[tuple[str, int, float], dict[str, Any]]:
    rows = {}
    for run in plan_curie_weiss_ancilla_runs(store.manifest):
        bundle = store.load_result(run)
        if bundle.metrics.get("source_tree_digest") != bundle.environment.get(
            "source_tree_digest"
        ):
            raise RuntimeError("result source provenance differs")
        for row in bundle.metrics["evaluation_rows"]:
            key = (run.family.value, run.optimizer_seed, float(row["beta"]))
            rows[key] = dict(row)
    if len(rows) != 220:
        raise RuntimeError("new arms must contain ten seeds at eleven temperatures")
    return rows


def _seed_means(
    response_rows: list[dict[str, Any]], arm: str, metric: str, role: str
) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in response_rows:
        if row["role"] == role:
            values[int(row["seed"])].append(float(row[arm][metric]))
    return {seed: float(np.mean(items)) for seed, items in values.items()}


def _difference(left: dict[int, float], right: dict[int, float]) -> dict[int, float]:
    if set(left) != set(right) or set(left) != set(range(10)):
        raise RuntimeError("paired comparison requires optimizer seeds 0 through 9")
    return {seed: left[seed] - right[seed] for seed in sorted(left)}


def encoding_decision(
    primary: dict[str, Any],
    bitstring_guardrail: dict[str, Any],
    margin: float = BITSTRING_MARGIN,
) -> tuple[str, bool, bool]:
    """Apply the frozen superiority and non-inferiority decision rule."""
    profile_pass = float(primary["lower"]) > 0.0
    guardrail_pass = float(bitstring_guardrail["upper"]) < margin
    if profile_pass and guardrail_pass:
        decision = "supported"
    elif (
        float(primary["upper"]) <= 0.0 or float(bitstring_guardrail["lower"]) >= margin
    ):
        decision = "unsupported"
    else:
        decision = "mixed"
    return decision, profile_pass, guardrail_pass


def build_encoding_report(store: CurieWeissAncillaCampaignStore) -> dict[str, Any]:
    """Build the preregistered report after all twenty new fits complete."""
    if store.manifest.campaign_version != ENCODING_COMPARISON_VERSION:
        raise RuntimeError("encoding report requires the comparison manifest")
    planned = plan_curie_weiss_ancilla_runs(store.manifest)
    if len(store.completed_runs()) != len(planned):
        raise RuntimeError("encoding report requires all 20 immutable results")
    baseline_report = _verify_baseline(store)
    baseline = _baseline_rows(baseline_report)
    new = _new_rows(store)
    response_rows = []
    for seed in range(10):
        for beta in store.manifest.evaluation_betas:
            role = "training" if beta in store.manifest.training_betas else "held_out"
            response_rows.append(
                {
                    "seed": seed,
                    "beta": beta,
                    "role": role,
                    "broadcast_baseline": baseline[(seed, beta)],
                    "role_linear": new[("role_linear", seed, beta)],
                    "role_quadratic": new[("role_quadratic", seed, beta)],
                }
            )

    heldout = {
        arm: {
            metric: _seed_means(response_rows, arm, metric, "held_out")
            for metric in METRICS
        }
        for arm in ("broadcast_baseline", "role_linear", "role_quadratic")
    }
    primary = _interval(
        _difference(
            heldout["broadcast_baseline"]["profile_tvd"],
            heldout["role_quadratic"]["profile_tvd"],
        )
    )
    bitstring_guardrail = _interval(
        _difference(
            heldout["role_quadratic"]["bitstring_tvd"],
            heldout["broadcast_baseline"]["bitstring_tvd"],
        )
    )
    quadratic_vs_linear = _interval(
        _difference(
            heldout["role_linear"]["profile_tvd"],
            heldout["role_quadratic"]["profile_tvd"],
        )
    )
    linear_vs_baseline = _interval(
        _difference(
            heldout["broadcast_baseline"]["profile_tvd"],
            heldout["role_linear"]["profile_tvd"],
        )
    )
    decision, profile_pass, guardrail_pass = encoding_decision(
        primary,
        bitstring_guardrail,
    )

    per_temperature = []
    for beta in store.manifest.evaluation_betas:
        selected = [row for row in response_rows if np.isclose(row["beta"], beta)]
        per_temperature.append(
            {
                "beta": beta,
                "role": selected[0]["role"],
                "arms": {
                    arm: {
                        metric: float(np.mean([row[arm][metric] for row in selected]))
                        for metric in METRICS
                    }
                    for arm in (
                        "broadcast_baseline",
                        "role_linear",
                        "role_quadratic",
                    )
                },
            }
        )
    return {
        "schema_version": 1,
        "campaign_version": store.manifest.campaign_version,
        "manifest_digest": store.manifest.digest,
        "completed_new_fits": 20,
        "reused_baseline_fits": 10,
        "baseline_reference": store.manifest.raw["baseline_reference"],
        "decision": decision,
        "success_rule": {
            "profile_improvement_lower_strictly_above_zero": profile_pass,
            "bitstring_degradation_upper_below_margin": guardrail_pass,
            "bitstring_noninferiority_margin": BITSTRING_MARGIN,
        },
        "primary_profile_baseline_minus_quadratic": primary,
        "bitstring_quadratic_minus_baseline": bitstring_guardrail,
        "secondary_profile_linear_minus_quadratic": quadratic_vs_linear,
        "secondary_profile_baseline_minus_linear": linear_vs_baseline,
        "held_out_seed_means": {
            arm: {
                metric: {str(seed): value for seed, value in seed_values.items()}
                for metric, seed_values in metrics.items()
            }
            for arm, metrics in heldout.items()
        },
        "per_temperature": per_temperature,
        "response_rows": response_rows,
        "claim": (
            "At eight visible qubits and five ancillas, a fixed quadratic "
            "temperature feature on the ancilla register improved held-out "
            "Hamming-profile fidelity relative to broadcast conditioning without "
            "a material loss in bitstring fidelity."
            if decision == "supported"
            else None
        ),
        "prohibited_claims": [
            "quadratic encoding is generally better",
            "the encoding captures critical scaling",
            "the result extends to other datasets",
            "quantum advantage",
        ],
    }


def write_encoding_report(path: Path, report: dict[str, Any]) -> Path:
    if path.exists():
        if read_immutable_json(path) != report:
            raise RuntimeError("immutable encoding report differs")
    else:
        write_immutable_json(path, report)
    return path


def render_encoding_summary(report: dict[str, Any]) -> str:
    primary = report["primary_profile_baseline_minus_quadratic"]
    guardrail = report["bitstring_quadratic_minus_baseline"]
    secondary = report["secondary_profile_linear_minus_quadratic"]
    lines = [
        "# Role-separated Curie--Weiss encoding comparison",
        "",
        f"Decision: **{report['decision']}**.",
        "",
        "The primary effect is baseline minus quadratic held-out profile TVD. "
        "Positive values favor the quadratic ancilla map.",
        "",
        "| Comparison | Estimate | 95% paired interval |",
        "|---|---:|---:|",
        f"| Baseline minus quadratic, profile TVD | {primary['estimate']:+.4f} | "
        f"[{primary['lower']:+.4f}, {primary['upper']:+.4f}] |",
        f"| Quadratic minus baseline, bitstring TVD | {guardrail['estimate']:+.4f} | "
        f"[{guardrail['lower']:+.4f}, {guardrail['upper']:+.4f}] |",
        f"| Linear minus quadratic, profile TVD | {secondary['estimate']:+.4f} | "
        f"[{secondary['lower']:+.4f}, {secondary['upper']:+.4f}] |",
        "",
        "Success requires the first interval to lie above zero and the upper "
        "bitstring bound to remain below 0.01.",
        "",
    ]
    if report["claim"] is not None:
        lines.extend(["Supported claim:", "", f"> {report['claim']}", ""])
    else:
        lines.extend(
            [
                "The preregistered claim is not supported. The result must be "
                "reported as mixed or unsupported according to the decision above.",
                "",
            ]
        )
    return "\n".join(lines)
