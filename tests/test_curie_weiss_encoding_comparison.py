from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iqp_cond.ancilla import ancilla_ansatz
from iqp_cond.artifacts import read_immutable_json
from iqp_cond.curie_weiss_ancilla_protocol import (
    CurieWeissAncillaFamily,
    CurieWeissAncillaManifest,
    broadcast_angle,
    plan_curie_weiss_ancilla_runs,
    role_separated_angle,
)
from iqp_cond.curie_weiss_ancilla_runner import condition_angles, evaluate_run
from iqp_cond.curie_weiss_encoding_report import (
    build_encoding_report,
    encoding_decision,
    render_encoding_summary,
)

ROOT = Path(__file__).parents[1]
MANIFEST = CurieWeissAncillaManifest.from_json(
    ROOT / "configs/curie_weiss_encoding_comparison_v1.json"
)


def _run(family: CurieWeissAncillaFamily, seed: int = 0):
    return next(
        run
        for run in plan_curie_weiss_ancilla_runs(MANIFEST)
        if run.family is family and run.optimizer_seed == seed
    )


def test_frozen_role_separated_angles() -> None:
    amplitude = 0.14 * math.pi
    linear = [
        role_separated_angle(MANIFEST, beta, CurieWeissAncillaFamily.ROLE_LINEAR)
        for beta in (0.5, 1.0, 1.5)
    ]
    quadratic = [
        role_separated_angle(
            MANIFEST,
            beta,
            CurieWeissAncillaFamily.ROLE_QUADRATIC,
        )
        for beta in (0.5, 1.0, 1.5)
    ]
    assert [angles[-1] for angles in linear] == pytest.approx(
        [-amplitude, 0.0, amplitude]
    )
    assert [angles[-1] for angles in quadratic] == pytest.approx(
        [amplitude, -0.5 * amplitude, amplitude]
    )
    for beta, angles in zip((0.5, 1.0, 1.5), linear, strict=True):
        assert len(angles) == 13
        assert angles[:8] == pytest.approx(broadcast_angle(MANIFEST, beta, 8))
        assert len(set(angles[8:])) == 1


def test_manifest_plans_twenty_paired_fits() -> None:
    runs = plan_curie_weiss_ancilla_runs(MANIFEST)
    assert len(runs) == len({run.run_id for run in runs}) == 20
    assert {(run.optimizer_seed, run.family.value) for run in runs} == {
        (seed, family)
        for seed in range(10)
        for family in ("role_linear", "role_quadratic")
    }
    assert {run.parameter_count for run in runs} == {91}


def test_training_and_evaluation_use_the_same_full_angles(monkeypatch) -> None:
    run = _run(CurieWeissAncillaFamily.ROLE_QUADRATIC)
    expected = condition_angles(MANIFEST, run, evaluation=True)
    captured = None

    def fake_probabilities(
        params,
        generators,
        full_angles,
        n_data_qubits,
        n_ancillas,
        encoding=None,
    ):
        nonlocal captured
        captured = np.asarray(full_angles)
        return np.full((len(full_angles), 2**n_data_qubits), 1.0 / 2**n_data_qubits)

    monkeypatch.setattr(
        "iqp_cond.curie_weiss_ancilla_runner."
        "exact_ancilla_marginal_probabilities_from_full_angles",
        fake_probabilities,
    )
    ansatz = ancilla_ansatz(8, 5)
    _, arrays = evaluate_run(
        MANIFEST,
        run,
        np.zeros(len(ansatz.generators)),
        ansatz.generators,
    )
    assert captured is not None
    assert np.array_equal(captured, expected)
    assert np.array_equal(arrays["condition_angles"], expected)


@pytest.mark.parametrize(
    ("primary", "guardrail", "expected"),
    [
        (
            {"lower": 0.001, "upper": 0.02},
            {"lower": -0.01, "upper": 0.009},
            "supported",
        ),
        ({"lower": 0.0, "upper": 0.02}, {"lower": -0.01, "upper": 0.009}, "mixed"),
        ({"lower": 0.001, "upper": 0.02}, {"lower": -0.01, "upper": 0.01}, "mixed"),
        (
            {"lower": -0.02, "upper": 0.0},
            {"lower": -0.01, "upper": 0.009},
            "unsupported",
        ),
        (
            {"lower": 0.001, "upper": 0.02},
            {"lower": 0.01, "upper": 0.02},
            "unsupported",
        ),
    ],
)
def test_frozen_success_rule(primary, guardrail, expected) -> None:
    assert encoding_decision(primary, guardrail)[0] == expected


def test_complete_synthetic_report_applies_preregistered_rule() -> None:
    planned = plan_curie_weiss_ancilla_runs(MANIFEST)
    baseline_report = read_immutable_json(
        ROOT / "results/curie-weiss-m5-foundations-v2/report/campaign-report.json"
    )
    baseline = {
        (int(row["seed"]), float(row["beta"])): row
        for row in baseline_report["response_rows"]
        if int(row["ancillas"]) == 5
    }

    class FakeStore:
        manifest = MANIFEST
        root = ROOT / "results/curie-weiss-encoding-comparison-v1/store"

        @staticmethod
        def completed_runs():
            return planned

        @staticmethod
        def load_result(run):
            rows = []
            factor = 0.9 if run.family is CurieWeissAncillaFamily.ROLE_LINEAR else 0.8
            for beta in MANIFEST.evaluation_betas:
                source = baseline[(run.optimizer_seed, beta)]
                metrics = dict(source["conditioned"])
                metrics["profile_tvd"] *= factor
                metrics["exact_mmd"] = float(
                    np.mean(list(source["conditioned_exact_mmd_by_bandwidth"].values()))
                )
                metrics.update(
                    {
                        "beta": beta,
                        "role": "training"
                        if beta in MANIFEST.training_betas
                        else "held_out",
                    }
                )
                rows.append(metrics)
            return SimpleNamespace(
                metrics={"source_tree_digest": "same", "evaluation_rows": rows},
                environment={"source_tree_digest": "same"},
            )

    report = build_encoding_report(FakeStore())
    assert report["decision"] == "supported"
    assert report["success_rule"] == {
        "profile_improvement_lower_strictly_above_zero": True,
        "bitstring_degradation_upper_below_margin": True,
        "bitstring_noninferiority_margin": 0.01,
    }
    assert len(report["response_rows"]) == 110
    assert "Supported claim" in render_encoding_summary(report)
