"""Independent shell-law fixtures for validation and reporting tests."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments import cw_classical_report as report
from paper_experiments.cw_classical_config import (
    FAMILIES,
    betas,
    limits,
    model,
    protocol,
    target_stream,
)
from paper_experiments.cw_followup_metrics import curve_scores
from paper_experiments.target import curie_weiss_target


def _curves(grid, repeats, n):
    m = (2 * np.arange(n + 1) - n) / n
    energy = -n * m**2 / 2
    p = np.stack([curie_weiss_target(n, beta).probabilities for beta in grid])
    q = np.stack(
        [
            [
                curie_weiss_target(n, beta + 0.03 * (repeat + 1)).probabilities
                for beta in grid
            ]
            for repeat in range(repeats)
        ]
    )
    p = np.broadcast_to(p, q.shape)
    heat = np.asarray(grid) ** 2 / n * (q @ energy**2 - (q @ energy) ** 2)
    target_heat = np.asarray(grid) ** 2 / n * (p @ energy**2 - (p @ energy) ** 2)
    profile = abs(q - p).sum(-1) / 2
    return {
        key: value.tolist()
        for key, value in dict(
            tvd=profile,
            profile_tvd=profile,
            shells=q,
            target_shells=p,
            magnetization=q @ abs(m),
            magnetization_error=abs((q - p) @ abs(m)),
            heat=heat,
            target_heat=target_heat,
            heat_error=abs(heat - target_heat),
            signed_magnetization=q @ m,
            flip_defect=np.zeros(profile.shape),
        ).items()
    }


def _bundle(smoke=True):
    budget = limits(smoke)
    repeats, n = budget["repeats"]["final"], budget["n"]
    grid = betas("final", smoke)
    curves = _curves(grid, repeats, n)
    marker = _curves((0.4, 1.0, 2.0), repeats, n)
    pools = []
    for repeat in range(repeats):
        records = []
        for beta in protocol()["training"]["shared_anchors"]:
            stream = target_stream("final", repeat, beta)
            x = curie_weiss_target(n, beta).sample(
                budget["n_train"], np.random.default_rng(stream["seed"])
            )
            records.append(
                dict(
                    beta=beta,
                    stream=stream,
                    draws=len(x),
                    array_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                )
            )
        pools.append(records)
    rows = {}
    diagnostic = dict(nodes=128, errors={64: 1e-7, 128: 1e-8}, tolerance=1e-5)
    for family in FAMILIES:
        rows[family] = dict(
            architecture=model(family).architecture(n, "small"),
            beta=grid,
            curves=copy.deepcopy(curves),
            marker_beta=(0.4, 1.0, 2.0),
            marker_curves=copy.deepcopy(marker),
            repeat_scores=[
                curve_scores(
                    curves["tvd"][i],
                    curves["magnetization_error"][i],
                    curves["heat_error"][i],
                    grid,
                )
                for i in range(repeats)
            ],
            repeat_ids=list(range(repeats)),
            target_pools=copy.deepcopy(pools),
            probability_evaluation=[
                [copy.deepcopy(diagnostic) for _ in grid] for _ in range(repeats)
            ],
            marker_probability_evaluation=[
                [copy.deepcopy(diagnostic) for _ in range(3)] for _ in range(repeats)
            ],
        )
    return dict(
        kind="cw_classical_final",
        stage="final",
        smoke=smoke,
        capacity="small",
        repeats=repeats,
        starts=budget["starts"]["final"],
        families=rows,
        provenance=dict(
            manifest=dict(protocol=protocol(smoke), capacity="small", smoke=smoke)
        ),
    )


def test_smoke_requires_opt_in_and_validates_known_shell_laws():
    data = _bundle()
    with pytest.raises(ValueError, match="opt-in"):
        report._validate(data)
    assert report._validate(data, allow_smoke=True)


@pytest.mark.parametrize(
    "corruption", ["moment", "score", "pool", "repeat", "quadrature"]
)
def test_report_rejects_inconsistent_evidence(corruption):
    data = _bundle()
    row = data["families"]["vae"]
    if corruption == "moment":
        row["curves"]["magnetization"][0][0] += 0.1
    elif corruption == "score":
        row["repeat_scores"][0]["J"] += 0.1
    elif corruption == "pool":
        row["target_pools"][0][0]["array_sha256"] = "wrong"
    elif corruption == "repeat":
        row["repeat_ids"].reverse()
    else:
        row["probability_evaluation"][0][0]["errors"][128] = 0.1
    with pytest.raises(ValueError):
        report._validate(data, allow_smoke=True)


def test_summary_aggregates_metrics_before_averaging_curves():
    row = dict(beta=(0.0, 1.0), marker_beta=(0.4, 1.0, 2.0))
    row["curves"] = dict(
        tvd=[[0.1, 0.9], [0.9, 0.1]],
        magnetization_error=[[0.0, 0.0]] * 2,
        heat_error=[[0.0, 0.0]] * 2,
    )
    row["marker_curves"] = dict(tvd=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    summary, per_fit = report._summary(
        "test", row, parameters=1, generation_parameters=1, probability_kind="exact"
    )
    assert summary["max_tvd_mean"] == 0.9
    assert summary["max_tvd_mean"] != max(summary["tvd_curve"])
    assert summary["J_mean"] == np.mean([r["J"] for r in per_fit])


def test_paired_bootstrap_uses_repeat_differences():
    result = report._bootstrap([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], seed=4)
    assert result["mean_difference"] == 1
    assert result["ci95"] == [1.0, 1.0]
    assert result["repeats"] == 3
    with pytest.raises(ValueError, match="matched"):
        report._bootstrap([1.0, 2.0], [1.0], seed=4)


def test_smoke_render_and_output_boundary(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-smoke.json"
    source.write_text(json.dumps(_bundle()))
    monkeypatch.setattr(report, "RESULTS_ROOT", tmp_path / "outputs")
    with pytest.raises(ValueError, match="ignored"):
        report.render(source, tmp_path / "wrong", allow_smoke=True)
    monkeypatch.setattr(report.shutil, "which", lambda _: None)
    paths = report.render(source, tmp_path / "outputs" / "smoke", allow_smoke=True)
    assert all(path.is_file() for path in paths)
    summary = json.loads(paths[0].read_text())
    assert summary["smoke"] is True
    assert summary["paired_against_twisted_iqp"] == {}
    assert len(summary["rows"]) == 3


DATA = Path(__file__).parents[1] / "data"


def _iqp_root(tmp_path, independent=True):
    """Bundled IQP data, optionally without the per-temperature twisted model."""
    root = tmp_path / "iqp"
    for name in ("curie-weiss-followup", "curie-weiss-twisted"):
        (root / name).mkdir(parents=True)
        data = json.loads((DATA / name / "figure_data.json").read_text())
        if name == "curie-weiss-twisted" and not independent:
            data["families"] = {
                k: v for k, v in data["families"].items() if k.startswith("shared")
            }
        (root / name / "figure_data.json").write_text(json.dumps(data))
    return root


def test_full_render_includes_audited_iqp_and_both_paired_metrics(
    tmp_path, monkeypatch
):
    variant, label = "twisted", "TP"
    source = tmp_path / "synthetic-final.json"
    source.write_text(json.dumps(_bundle(False)))
    monkeypatch.setattr(report, "RESULTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(report.shutil, "which", lambda _: None)
    iqp_root = _iqp_root(tmp_path, independent=False)
    paths = report.render(
        source, tmp_path / "outputs" / "synthetic-final", iqp_data_root=iqp_root
    )
    summary = json.loads(paths[0].read_text())
    assert len(summary["rows"]) == 5
    assert summary["rows"][-2]["parameter_count"] == 136
    assert [r["model"] for r in summary["rows"][-2:]] == [
        f"C-IQP ({label})",
        "C-IQP (P)",
    ]
    assert f"C-IQP ({label})" in summary["paired_difference_direction"]
    assert any(f"curie-weiss-{variant}" in key for key in summary["input_sha256"])
    paired_all = summary[f"paired_against_{variant}_iqp"]
    assert set(paired_all) == set(FAMILIES)
    for paired in paired_all.values():
        assert set(paired) == {"J", "mean_tvd"}
        assert paired["J"]["repeats"] == 10


def test_paper_artifacts_compare_against_main_encoder(tmp_path):
    iqp_root = _iqp_root(tmp_path, independent=False)
    paths = report.paper_artifacts(
        DATA / "curie-weiss-classical",
        iqp_root / "curie-weiss-followup",
        tmp_path / "out",
    )
    table = paths[0].read_text()
    assert table.count("C-IQP (TP)") == 1 and "C-IQP (SP)" not in table
