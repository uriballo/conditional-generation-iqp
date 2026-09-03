#!/usr/bin/env python3
"""Generate manuscript fragments for the strict latent-input prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _mean(payload: dict[str, object], key: str, role: str, metric: str) -> float:
    return float(payload["summary"][key][role][metric])


def _seed_means(
    payload: dict[str, object],
    encoding: str,
    role: str,
    metric: str,
    *,
    ancillas: int | None = None,
) -> dict[int, float]:
    rows = [
        row
        for row in payload["rows"]
        if row["encoding"] == encoding
        and row["role"] == role
        and (ancillas is None or int(row.get("ancillas", ancillas)) == ancillas)
    ]
    return {
        seed: float(
            np.mean([float(row[metric]) for row in rows if int(row["seed"]) == seed])
        )
        for seed in sorted({int(row["seed"]) for row in rows})
    }


def _paired_interval(
    baseline: dict[str, object],
    model: dict[str, object],
    metric: str,
    *,
    seed: int = 20_260_824,
) -> tuple[float, float, float]:
    affine = _seed_means(baseline, "affine_broadcast", "test", metric)
    latent = _seed_means(
        model,
        "latent_input_linear",
        "test",
        metric,
        ancillas=5,
    )
    seeds = sorted(set(affine) & set(latent))
    differences = np.asarray([affine[value] - latent[value] for value in seeds])
    random = np.random.default_rng(seed)
    indices = random.integers(0, len(differences), size=(10_000, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return float(differences.mean()), float(lower), float(upper)


def _command(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _claims(
    gap_baseline: dict[str, object],
    one_sided_baseline: dict[str, object],
    rungs: dict[int, dict[str, object]],
    one_sided: dict[str, object],
    visible: dict[str, object],
    gaussian: dict[str, object],
) -> str:
    lines = [
        "% Generated from exploratory strict latent-input artifacts.",
        r"\newif\ifphaseoodresults",
        r"\phaseoodresultstrue",
    ]
    gap_models = {
        "Affine": (gap_baseline, "affine_broadcast"),
        "Rone": (rungs[1], "latent_input_linear_r1"),
        "Rthree": (rungs[3], "latent_input_linear_r3"),
        "Rfive": (rungs[5], "latent_input_linear_r5"),
        "Visible": (visible, "visible_capacity_r5"),
    }
    for name, (payload, key) in gap_models.items():
        for role, role_name in (("training", "Train"), ("test", "Test")):
            for metric, metric_name in (
                ("profile_tvd", "Profile"),
                ("raw_tvd", "Raw"),
                ("absolute_magnetization_error", "Magnetization"),
            ):
                lines.append(
                    _command(
                        f"PhaseGap{name}{role_name}{metric_name}",
                        f"{_mean(payload, key, role, metric):.3f}",
                    )
                )
    for metric, metric_name in (
        ("profile_tvd", "Profile"),
        ("raw_tvd", "Raw"),
        ("absolute_magnetization_error", "Magnetization"),
    ):
        mean, lower, upper = _paired_interval(gap_baseline, rungs[5], metric)
        lines.append(
            _command(
                f"PhaseGap{metric_name}Reduction",
                f"{mean:.3f} [{lower:.3f}, {upper:.3f}]",
            )
        )
    for name, payload, key in (
        ("Affine", one_sided_baseline, "affine_broadcast"),
        ("Latent", one_sided, "latent_input_linear_r5"),
    ):
        for role, role_name in (("training", "Train"), ("test", "Test")):
            for metric, metric_name in (
                ("profile_tvd", "Profile"),
                ("raw_tvd", "Raw"),
                ("absolute_magnetization_error", "Magnetization"),
            ):
                lines.append(
                    _command(
                        f"PhaseOneSided{name}{role_name}{metric_name}",
                        f"{_mean(payload, key, role, metric):.3f}",
                    )
                )
    gaussian_models = {
        "Affine": "affine_broadcast_r0",
        "Latent": "latent_input_linear_r5",
        "Visible": "visible_capacity_r5",
    }
    for name, key in gaussian_models.items():
        for metric, metric_name in (
            ("profile_tvd", "Profile"),
            ("raw_tvd", "Raw"),
        ):
            lines.append(
                _command(
                    f"PhaseGaussian{name}Test{metric_name}",
                    f"{_mean(gaussian, key, 'test', metric):.3f}",
                )
            )
    lines.extend(
        (
            _command("PhaseLatentVisibleQubits", "8"),
            _command("PhaseLatentAncillas", "5"),
            _command("PhaseLatentTotalQubits", "13"),
            _command("PhaseLatentPhases", "36"),
            _command("PhaseLatentInputCouplings", "40"),
            _command("PhaseLatentParameters", "76"),
            _command("PhaseLatentDiagonalRotations", "36"),
            _command("PhaseLatentDiagonalLocality", "2"),
        )
    )
    return "\n".join(lines) + "\n"


def _results_rows(
    gap_baseline: dict[str, object],
    one_sided_baseline: dict[str, object],
    rungs: dict[int, dict[str, object]],
    one_sided: dict[str, object],
    visible: dict[str, object],
) -> str:
    rows = ["% Generated strict latent-input headline rows."]
    models = (
        ("Critical gap", "Affine broadcast", 36, gap_baseline, "affine_broadcast"),
        ("Critical gap", "$r=1$ latent input", 44, rungs[1], "latent_input_linear_r1"),
        ("Critical gap", "$r=3$ latent input", 60, rungs[3], "latent_input_linear_r3"),
        ("Critical gap", "$r=5$ latent input", 76, rungs[5], "latent_input_linear_r5"),
        ("Critical gap", "Visible-only capacity", 76, visible, "visible_capacity_r5"),
        ("Hot to cold", "Affine broadcast", 36, one_sided_baseline, "affine_broadcast"),
        ("Hot to cold", "$r=5$ latent input", 76, one_sided, "latent_input_linear_r5"),
    )
    for protocol, label, parameters, payload, key in models:
        rows.append(
            " & ".join(
                (
                    protocol,
                    label,
                    str(parameters),
                    f"{_mean(payload, key, 'training', 'profile_tvd'):.3f}",
                    f"{_mean(payload, key, 'test', 'profile_tvd'):.3f}",
                    f"{_mean(payload, key, 'test', 'raw_tvd'):.3f}",
                )
            )
            + r"\\"
        )
    return "\n".join(rows) + "\n"


def _ablation_rows(
    gap_baseline: dict[str, object],
    rungs: dict[int, dict[str, object]],
    visible: dict[str, object],
    gaussian: dict[str, object],
) -> str:
    rows = ["% Generated distributional and physical-metric control rows."]
    for label, parameters, payload, key in (
        ("Curie--Weiss affine", 36, gap_baseline, "affine_broadcast"),
        ("Latent input, $r=1$", 44, rungs[1], "latent_input_linear_r1"),
        ("Latent input, $r=3$", 60, rungs[3], "latent_input_linear_r3"),
        ("Latent input, $r=5$", 76, rungs[5], "latent_input_linear_r5"),
        ("Visible-only capacity", 76, visible, "visible_capacity_r5"),
    ):
        rows.append(
            f"{label} & {parameters} & "
            f"{_mean(payload, key, 'test', 'profile_tvd'):.3f} & "
            f"{_mean(payload, key, 'test', 'raw_tvd'):.3f} & "
            f"{_mean(payload, key, 'test', 'absolute_magnetization_error'):.3f}"
            r"\\"
        )
    rows.append(r"\midrule")
    for label, parameters, key in (
        ("Gaussian affine", 36, "affine_broadcast_r0"),
        ("Gaussian latent input, $r=5$", 76, "latent_input_linear_r5"),
        ("Gaussian visible-only capacity", 76, "visible_capacity_r5"),
    ):
        rows.append(
            f"{label} & {parameters} & "
            f"{_mean(gaussian, key, 'test', 'profile_tvd'):.3f} & "
            f"{_mean(gaussian, key, 'test', 'raw_tvd'):.3f} & "
            f"{_mean(gaussian, key, 'test', 'absolute_magnetization_error'):.3f}"
            r"\\"
        )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/curie-weiss-phase-ood-prototype-v1"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/curie_weiss_phase_ood")
    )
    args = parser.parse_args()
    gap_baseline = _read(args.results / "critical-gap-r3.json")
    one_sided_baseline = _read(args.results / "hot-to-cold-r3.json")
    rungs = {
        r: _read(args.results / f"latent-input-linear-r{r}-critical-gap-10seed.json")
        for r in (1, 3, 5)
    }
    one_sided = _read(args.results / "latent-input-linear-r5-hot-to-cold-10seed.json")
    visible = _read(args.results / "visible-capacity-r5-critical-gap-10seed.json")
    gaussian = _read(args.results / "latent-input-gaussian-critical-gap-control.json")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "curie_weiss_phase_ood_claims.tex").write_text(
        _claims(
            gap_baseline,
            one_sided_baseline,
            rungs,
            one_sided,
            visible,
            gaussian,
        )
    )
    (args.output / "curie_weiss_phase_ood_results.tex").write_text(
        _results_rows(
            gap_baseline,
            one_sided_baseline,
            rungs,
            one_sided,
            visible,
        )
    )
    (args.output / "curie_weiss_phase_ood_ablation.tex").write_text(
        _ablation_rows(gap_baseline, rungs, visible, gaussian)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
