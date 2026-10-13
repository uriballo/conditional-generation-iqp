import copy
import json

import numpy as np
import pytest

from paper_experiments import cw_followup_reporting as reporting
from paper_experiments.cw_followup_config import (
    FAMILIES,
    MARKERS,
    protocol,
    stage_betas,
)
from paper_experiments.cw_followup_metrics import curve_scores
from paper_experiments.target import curie_weiss_target


def _curves(beta):
    beta = np.asarray(beta)
    m = (2 * np.arange(17) - 16) / 16
    energy = -8 * m**2
    targets = np.stack([curie_weiss_target(16, float(b)).probabilities for b in beta])
    p = np.repeat(targets[None], 10, axis=0)
    q = p.copy()
    delta = 0.1 * np.minimum(p[0, :, 7], p[0, :, 8])
    for repeat in range(10):
        sign = 1 if repeat % 2 else -1
        q[repeat, :, 7] += sign * delta
        q[repeat, :, 8] -= sign * delta
    heat = beta**2 / 16 * (q @ energy**2 - (q @ energy) ** 2)
    target_heat = beta**2 / 16 * (p @ energy**2 - (p @ energy) ** 2)
    profile = np.abs(q - p).sum(-1) / 2
    return {
        k: v.tolist()
        for k, v in dict(
            tvd=profile + 0.1,
            shells=q,
            target_shells=p,
            magnetization=q @ np.abs(m),
            magnetization_error=np.abs((q - p) @ np.abs(m)),
            signed_magnetization=q @ m,
            heat=heat,
            target_heat=target_heat,
            heat_error=np.abs(heat - target_heat),
            flip_defect=np.abs(q - q[:, :, ::-1]).sum(-1) / 2,
        ).items()
    }


def _payload():
    families = {}
    for name in FAMILIES:
        beta = (
            list(stage_betas(protocol(), "final"))
            if name.startswith("shared_")
            else list(MARKERS)
        )
        curves = _curves(beta)
        families[name] = dict(
            trial_id=44,
            config=dict(
                learning_rate=0.001,
                init_scale=0.05,
                b1=0.9,
                b2=0.99,
                schedule="constant",
                expected_weights=[2, 5, 7],
                **({} if name == "independent_unconditional" else dict(beta_scale=0.6)),
            ),
            beta=beta,
            curves=curves,
            marker_beta=list(MARKERS),
            marker_curves=_curves(MARKERS),
            repeat_scores=[
                curve_scores(
                    curves["tvd"][r],
                    curves["magnetization_error"][r],
                    curves["heat_error"][r],
                    beta,
                )
                for r in range(10)
            ],
        )
    return dict(
        kind="cw_followup_final",
        stage="final",
        smoke=False,
        repeats=10,
        starts=3,
        C_star=reporting.C_STAR,
        families=families,
    )


def test_mean_sd_uses_sample_standard_deviation():
    mean, sd = reporting._mean_sd([[1.0], [3.0], [5.0]], 3)
    np.testing.assert_allclose(mean, [3.0])
    np.testing.assert_allclose(sd, [2.0])


def test_validation_checks_final_scope_shapes_moments_and_scores():
    data = _payload()
    assert reporting._validate(data) is data
    corruptions = [
        lambda d: d.update(smoke=True),
        lambda d: d["families"]["shared_parity"]["beta"].pop(),
        lambda d: d["families"]["shared_parity"]["marker_curves"]["shells"].pop(),
        lambda d: d["families"]["shared_parity"]["curves"]["magnetization"][
            0
        ].__setitem__(0, 0.9),
        lambda d: d["families"]["independent_parity"]["repeat_scores"][0].update(J=0.5),
    ]
    for corrupt in corruptions:
        bad = copy.deepcopy(data)
        corrupt(bad)
        with pytest.raises(ValueError):
            reporting._validate(bad)


def test_errors_are_computed_before_repeat_averaging_and_use_markers(tmp_path):
    data = reporting._validate(_payload())
    row = data["families"]["shared_parity"]
    profiles = reporting._profile_tvd(row)
    shells = np.asarray(row["marker_curves"]["shells"])
    targets = np.asarray(row["marker_curves"]["target_shells"])
    assert profiles.shape == (10, 3)
    assert np.all(profiles.mean(0) > 0)
    assert np.max(np.abs(shells.mean(0) - targets.mean(0))) < 1e-15
    np.testing.assert_array_equal(reporting._full_tvd(row), row["marker_curves"]["tvd"])
    reporting.export_tables(data, tmp_path)
    summary = json.loads((tmp_path / "curie_weiss_summary.json").read_text())
    np.testing.assert_allclose(
        summary["families"]["shared_parity"]["marker_profile_tvd"], profiles
    )
    raw_m = np.asarray(row["marker_curves"]["magnetization_error"])
    raw_heat = np.asarray(row["marker_curves"]["heat_error"])
    np.testing.assert_allclose(
        summary["families"]["shared_parity"]["marker_physics_errors"]["e_m"], raw_m
    )
    np.testing.assert_allclose(
        summary["families"]["shared_parity"]["marker_physics_errors"]["e_c"],
        raw_heat / reporting.C_STAR,
    )
    assert "ten independent fits" in (tmp_path / "curie_weiss_full_tvd.tex").read_text()
    assert "0.4" in (tmp_path / "curie_weiss_full_tvd.tex").read_text()
    assert (
        "tab:cw_encoder_comparison"
        in (tmp_path / "curie_weiss_encoder_comparison.tex").read_text()
    )
    physics_table = (tmp_path / "curie_weiss_physics_errors.tex").read_text()
    assert "tab:cw_physics_errors" in physics_table
    assert "$e_m$" in physics_table and "$e_c$" in physics_table
    expected_cell = reporting._cell(raw_heat[:, 0] / reporting.C_STAR)
    assert expected_cell in physics_table
    assert (
        summary["families"]["independent_parity"]["marker_repeat_scores"]
        == data["families"]["independent_parity"]["repeat_scores"]
    )
    shared = summary["families"]["shared_parity"]
    assert shared["marker_repeat_scores"][0]["J"] != shared["repeat_scores"][0]["J"]


def test_current_figure_alias_dispatches_final_data_without_historical_substitution(
    tmp_path, monkeypatch
):
    from paper_experiments.figures import render_all

    folder = tmp_path / "curie-weiss-followup"
    folder.mkdir()
    (folder / "figure_data.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        reporting, "render", lambda data, out: calls.append((data, out)) or ["final"]
    )
    assert render_all(tmp_path, tmp_path / "out", experiment="curie-weiss") == ["final"]
    assert render_all(
        tmp_path, tmp_path / "out", experiment="curie-weiss-followup"
    ) == ["final"]
    assert calls[0][0] == folder


def _squeezed_payload():
    published = _payload()["families"]
    families = {}
    for name, source in (
        ("shared_squeezed_parity", "shared_parity"),
        ("independent_squeezed_parity", "independent_parity"),
    ):
        row = copy.deepcopy(published[source])
        row["config"]["kappa"] = 0.4
        families[name] = row
    return dict(
        kind="cw_squeezed_final",
        stage="final",
        smoke=False,
        repeats=10,
        starts=3,
        families=families,
    )


def test_squeezed_bundle_merges_and_renders(tmp_path):
    followup = tmp_path / "curie-weiss-followup"
    squeezed = tmp_path / "curie-weiss-squeezed"
    followup.mkdir()
    squeezed.mkdir()
    (followup / "figure_data.json").write_text(json.dumps(_payload()))
    (squeezed / "figure_data.json").write_text(json.dumps(_squeezed_payload()))
    data = reporting.load(followup)
    assert set(data["families"]) == set(reporting.ORDER)
    paths = reporting.render(followup, tmp_path / "out")
    names = {p.name for p in paths}
    assert "curie_weiss_encoder_comparison.pdf" in names
    table = (tmp_path / "out" / "curie_weiss_full_tvd.tex").read_text()
    assert "squeezed parity encoder" in table
    assert reporting.NAMES["independent_squeezed_parity"] in table
    bad = _squeezed_payload()
    bad["families"]["shared_parity"] = bad["families"].pop("shared_squeezed_parity")
    (squeezed / "figure_data.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        reporting.load(followup)
