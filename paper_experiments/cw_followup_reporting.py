"""Publication reporting for completed Curie-Weiss follow-up figure data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cw_followup_config import MARKERS, protocol, stage_betas
from .cw_followup_metrics import curve_scores
from .figures import COLORS, MM, _save

C_STAR = 0.8231097389772882
FAMILIES = (
    "shared_parity",
    "shared_product",
    "independent_parity",
    "independent_unconditional",
)
METRICS = ("tvd", "magnetization", "heat")
NAMES = {
    "shared_parity": "Shared parity",
    "shared_product": "Shared product",
    "independent_parity": "Independent parity",
    "independent_unconditional": "Independent unconditional",
}


def _mean_sd(values, repeats):
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[0] != repeats or not np.isfinite(array).all():
        raise ValueError("metric repeats are incomplete or non-finite")
    return array.mean(axis=0), array.std(axis=0, ddof=1)


def _validate(data):
    if data.get("kind") != "cw_followup_final" or data.get("stage") != "final":
        raise ValueError("unexpected follow-up reporting scope")
    if (
        data.get("smoke") is not False
        or data.get("repeats") != 10
        or data.get("starts") != 3
    ):
        raise ValueError("final reporting requires ten repeats and three starts")
    if data.get("C_star") != C_STAR or set(data.get("families", {})) != set(FAMILIES):
        raise ValueError("incomplete follow-up reporting payload")
    for name in FAMILIES:
        row = data["families"][name]
        if not isinstance(row.get("trial_id"), int) or not isinstance(
            row.get("config"), dict
        ):
            raise ValueError("invalid family selection")
        beta = np.asarray(row.get("beta"), dtype=float)
        marker = np.asarray(row.get("marker_beta", []), dtype=float)
        expected = (
            MARKERS
            if name.startswith("independent")
            else stage_betas(protocol(), "final")
        )
        if (
            beta.ndim != 1
            or not np.array_equal(marker, np.array([0.4, 1.0, 2.0]))
            or not np.array_equal(beta, expected)
        ):
            raise ValueError("invalid beta grid")
        for scope, grid in (("curves", beta), ("marker_curves", marker)):
            curves = row.get(scope, {})
            for key in (
                "tvd",
                "magnetization",
                "magnetization_error",
                "heat",
                "target_heat",
                "heat_error",
                "flip_defect",
                "signed_magnetization",
                "shells",
                "target_shells",
            ):
                arr = np.asarray(curves.get(key), dtype=float)
                shape = (
                    (10, len(grid), 17)
                    if key in ("shells", "target_shells")
                    else (10, len(grid))
                )
                if arr.shape != shape or not np.isfinite(arr).all():
                    raise ValueError("invalid curve repeats/shape")
                if key != "signed_magnetization" and np.any(arr < 0):
                    raise ValueError("negative error/probability")
                if key in ("shells", "target_shells") and not np.allclose(
                    arr.sum(-1), 1, atol=1e-8, rtol=0
                ):
                    raise ValueError("shell probabilities are not normalized")
            q, p = np.asarray(curves["shells"]), np.asarray(curves["target_shells"])
            profile = np.abs(q - p).sum(-1) / 2
            if np.any(profile > np.asarray(curves["tvd"]) + 1e-8) or np.any(
                np.asarray(curves["tvd"]) > 1 + 1e-8
            ):
                raise ValueError("inconsistent profile/full TVD")
            m = (2 * np.arange(17) - 16) / 16
            energy = -8 * m**2
            heat = grid**2 / 16 * (q @ energy**2 - (q @ energy) ** 2)
            target_heat = grid**2 / 16 * (p @ energy**2 - (p @ energy) ** 2)
            checks = dict(
                magnetization=q @ np.abs(m),
                signed_magnetization=q @ m,
                magnetization_error=np.abs((q - p) @ np.abs(m)),
                heat=heat,
                target_heat=target_heat,
                heat_error=np.abs(heat - target_heat),
            )
            for key, values in checks.items():
                if not np.allclose(curves[key], values, atol=1e-8, rtol=1e-8):
                    raise ValueError(f"per-repeat shell moments differ: {key}")
        scores = row.get("repeat_scores")
        if (
            not isinstance(scores, list)
            or len(scores) != 10
            or not np.isfinite(
                [[s[k] for k in ("J", "P", "max_tvd", "mean_tvd")] for s in scores]
            ).all()
        ):
            raise ValueError("invalid repeat scores")
        curves = row["curves"]
        for repeat, scores in enumerate(row["repeat_scores"]):
            expected_scores = curve_scores(
                curves["tvd"][repeat],
                curves["magnetization_error"][repeat],
                curves["heat_error"][repeat],
                beta,
            )
            if any(
                not np.isclose(scores.get(k, np.nan), v, atol=1e-12, rtol=0)
                for k, v in expected_scores.items()
            ):
                raise ValueError("nonlinear repeat score differs")
    return data


def _table(path, header, rows, caption, label):
    path.write_text(
        "\\begin{table*}[t]\n\\centering\n"
        + "\\small\n\\caption{"
        + caption
        + "}\n\\label{"
        + label
        + "}\n"
        + "\\begin{tabular}{l"
        + "r" * (len(header) - 1)
        + "}\n"
        + "\\toprule\n"
        + " & ".join(header)
        + " \\\\\n\\midrule\n"
        + "\n".join(" & ".join(str(x) for x in row) + " \\\\" for row in rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def _marker(row, key):
    return _mean_sd(row["marker_curves"][key], 10)


def _full_tvd(row):
    return np.asarray(row["marker_curves"]["tvd"], dtype=float)


def _profile_tvd(row):
    shells = np.asarray(row["marker_curves"]["shells"], dtype=float)
    targets = np.asarray(row["marker_curves"]["target_shells"], dtype=float)
    return 0.5 * np.abs(shells - targets).sum(axis=-1)


def render(data_dir, output_dir):
    data = _validate(json.loads((Path(data_dir) / "figure_data.json").read_text()))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    families = data["families"]
    colors = {
        "shared_parity": COLORS[0],
        "shared_product": COLORS[1],
        "independent_parity": COLORS[1],
        "independent_unconditional": COLORS[2],
    }
    paths = []
    fig, ax = plt.subplots(1, 3, figsize=(170 * MM, 65 * MM), layout="constrained")
    marker_families = (
        "shared_parity",
        "independent_parity",
        "independent_unconditional",
    )
    for index, beta in enumerate((0.4, 1.0, 2.0)):
        target = np.asarray(
            families["shared_parity"]["marker_curves"]["target_shells"]
        )[0, index]
        m = (2 * np.arange(17) - 16) / 16
        ax[index].plot(m, target, "k--", label="Exact target")
        for name in marker_families:
            values = np.asarray(families[name]["marker_curves"]["shells"], float)[
                :, index
            ]
            mean, sd = _mean_sd(values, 10)
            style = (
                "-"
                if name == "shared_parity"
                else "-."
                if name == "independent_parity"
                else ":"
            )
            ax[index].plot(
                m, mean, color=colors[name], linestyle=style, label=NAMES[name]
            )
            ax[index].fill_between(
                m, mean - sd, mean + sd, color=colors[name], alpha=0.15
            )
        ax[index].set(
            xlabel="Magnetization",
            ylabel="Probability mass",
            title=rf"$\beta={beta}$",
        )
        ax[index].spines[["top", "right"]].set_visible(False)
        ax[index].set_ylim(bottom=0)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    paths += _save(fig, output, "curie_weiss_model_comparison")
    fig = plt.figure(figsize=(170 * MM, 65 * MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(4.5, 1.5), hspace=0.05)
    ax = np.asarray([fig.add_subplot(grid[0, index]) for index in range(2)])
    for a, ylabel, key in zip(
        ax,
        (r"$\langle |m|\rangle$", r"$c_{\mathrm{fluc}}$ per spin"),
        ("magnetization", "heat"),
        strict=True,
    ):
        shared = families["shared_parity"]
        beta = np.asarray(shared["beta"])
        target = (
            np.asarray(shared["curves"]["target_shells"])[0]
            @ np.abs((2 * np.arange(17) - 16) / 16)
            if key == "magnetization"
            else shared["curves"]["target_heat"][0]
        )
        a.plot(beta, target, "k--", label="Exact target")
        for name in marker_families:
            if name.startswith("independent"):
                mean, sd = _marker(families[name], key)
                a.errorbar(
                    families[name]["marker_beta"],
                    mean,
                    yerr=sd,
                    fmt="o",
                    color=colors[name],
                    label=NAMES[name],
                    capsize=2,
                )
            else:
                mean, sd = _mean_sd(families[name]["curves"][key], 10)
                x = families[name]["beta"]
                a.plot(x, mean, color=colors[name], label=NAMES[name])
                a.fill_between(x, mean - sd, mean + sd, color=colors[name], alpha=0.15)
        a.set(xlabel=r"$\beta$", ylabel=ylabel)
        a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(-0.02, 2.05)
        a.set_ylim(bottom=0)
        a.plot(
            [0.1, 0.7, 1.3, 1.95],
            [0.015] * 4,
            "v",
            color=".5",
            transform=a.get_xaxis_transform(),
            markersize=3,
        )
    handles, labels = ax[0].get_legend_handles_labels()
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        fontsize=8,
    )
    paths += _save(fig, output, "curie_weiss_physical_response")
    fig, ax = plt.subplots(1, 2, figsize=(170 * MM, 65 * MM), layout="constrained")
    for name in ("shared_parity", "shared_product"):
        row = families[name]
        x = np.asarray(row["beta"])
        for a, key, label in zip(
            ax,
            ("tvd", "heat_error"),
            ("Full TVD", "Normalized fluctuation\n" + r"error $e_c$"),
            strict=True,
        ):
            mean, sd = _mean_sd(row["curves"][key], 10)
            if key == "heat_error":
                mean, sd = mean / C_STAR, sd / C_STAR
            a.plot(
                x,
                mean,
                color=colors[name],
                linestyle="-" if name == "shared_parity" else "--",
                label=NAMES[name],
            )
            a.fill_between(x, mean - sd, mean + sd, color=colors[name], alpha=0.15)
            a.set(xlabel=r"$\beta$", ylabel=label)
            a.spines[["top", "right"]].set_visible(False)
    for a in ax:
        a.set_ylim(bottom=0)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    paths += _save(fig, output, "curie_weiss_product_ablation")
    paths += export_tables(data, output)
    provenance = output / "curie_weiss_followup.provenance.json"
    provenance.write_text(
        json.dumps(
            dict(
                input_sha256=hashlib.sha256(
                    (Path(data_dir) / "figure_data.json").read_bytes()
                ).hexdigest(),
                renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                stage="final",
                repeats=10,
                aggregation=(
                    "within-repeat metrics; means and sample SD ddof=1; "
                    "all repeats retained"
                ),
                selection_changed=False,
            ),
            indent=2,
        )
        + "\n"
    )
    paths.append(provenance)
    return paths


def _cell(values):
    mean, sd = _mean_sd(values, 10)
    return rf"${float(mean):.4f}\pm{float(sd):.4f}$"


def export_tables(data, output):
    """Write exact marker tables and per-repeat summaries without plotting."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    families = data["families"]
    main = ("shared_parity", "independent_parity", "independent_unconditional")
    header = ["Model", *[rf"$\beta={b:g}$" for b in MARKERS]]
    paths = []
    for kind, extractor in (("full", _full_tvd), ("profile", _profile_tvd)):
        path = output / f"curie_weiss_{kind}_tvd.tex"
        rows = [
            [NAMES[name], *[_cell(extractor(families[name])[:, j]) for j in range(3)]]
            for name in main
        ]
        _table(
            path,
            header,
            rows,
            (
                "Exact "
                + ("full-distribution" if kind == "full" else "Hamming-weight profile")
                + " TVD at sixteen qubits. Values are means $\\pm$ sample standard "
                "deviations over ten fresh final repeats, computed within each repeat "
                "before aggregation. Lower is better. Shared parity excludes markers "
                "from fitting and selection; independent models fit directly "
                "at each marker."
            ),
            f"tab:curie_weiss_{kind}_tvd",
        )
        paths.append(path)
    path = output / "curie_weiss_product_ablation.tex"
    rows = []
    for name in ("shared_parity", "shared_product"):
        row = families[name]
        scores = row["repeat_scores"]
        rows.append(
            [
                NAMES[name],
                *[_cell(_full_tvd(row)[:, j]) for j in range(3)],
                _cell([s["J"] for s in scores]),
                _cell([s["P"] for s in scores]),
            ]
        )
    _table(
        path,
        [*header, "$J$", "$P$"],
        rows,
        (
            "Matched shared-encoder comparison. Marker columns give exact "
            "full-distribution TVD; $J$ and $P$ use the same 101-point reporting "
            "grid for both encoders. Values are means $\\pm$ sample standard "
            "deviations over ten fresh final repeats. Lower is better for all errors. "
            "The comparison changes input correlations and encoder phase generators as "
            "well as enforced spin-flip symmetry."
        ),
        "tab:cw_product_ablation",
    )
    paths.append(path)
    path = output / "curie_weiss_selected_settings.tex"
    rows = []
    for label, key in (
        ("Initial learning rate", "learning_rate"),
        (r"Width $s_\theta$ (radians)", "init_scale"),
        (r"\texttt{Adam} $b_1$", "b1"),
        (r"\texttt{Adam} $b_2$", "b2"),
        ("Schedule", "schedule"),
        ("Encoder scale $s$", "beta_scale"),
        ("Expected kernel weights", "expected_weights"),
    ):
        cells = [label]
        for name in FAMILIES:
            value = families[name]["config"].get(key)
            cells.append(
                "--"
                if value is None
                else f"{value:.6g}"
                if isinstance(value, (int, float))
                else str(value)
            )
        rows.append(cells)
    _table(
        path,
        ["Parameter", *[NAMES[n] for n in FAMILIES]],
        rows,
        (
            "Configurations selected by fresh confirmation. Independent-family "
            "settings apply jointly to three circuits with separately learned phases. "
            "Optimizer and unlisted settings follow the frozen protocol. Displayed "
            "continuous values are rounded; machine-readable settings retain "
            "full precision."
        ),
        "tab:cw_selected_settings",
    )
    paths.append(path)
    path = output / "curie_weiss_physics_errors.tex"
    rows = []
    for name in FAMILIES:
        row = families[name]
        magnetization = np.asarray(
            row["marker_curves"]["magnetization_error"], dtype=float
        )
        heat = np.asarray(row["marker_curves"]["heat_error"], dtype=float) / C_STAR
        rows.append(
            [NAMES[name], "$e_m$", *[_cell(magnetization[:, j]) for j in range(3)]]
        )
        rows.append([NAMES[name], "$e_c$", *[_cell(heat[:, j]) for j in range(3)]])
    _table(
        path,
        ["Model", "Error", *[rf"$\beta={b:g}$" for b in MARKERS]],
        rows,
        (
            "Exact marker mean absolute magnetization errors and normalized "
            "energy-fluctuation errors. Values are means $\\pm$ sample standard "
            "deviations over ten fresh repeats; energy errors use "
            "$c_\\star\\approx0.823110$. Lower is better. Shared markers are "
            "excluded from fitting and selection; independent models fit directly."
        ),
        "tab:cw_physics_errors",
    )
    paths.append(path)
    summary = dict(kind=data["kind"], repeats=10, families={})
    for name, row in families.items():
        summary["families"][name] = dict(
            trial_id=row["trial_id"],
            config=row["config"],
            marker_full_tvd=_full_tvd(row).tolist(),
            marker_profile_tvd=_profile_tvd(row).tolist(),
            marker_physics_errors={
                "e_m": np.asarray(
                    row["marker_curves"]["magnetization_error"], dtype=float
                ).tolist(),
                "e_c": (
                    np.asarray(row["marker_curves"]["heat_error"], dtype=float) / C_STAR
                ).tolist(),
            },
            marker_repeat_scores=[
                curve_scores(
                    row["marker_curves"]["tvd"][repeat],
                    row["marker_curves"]["magnetization_error"][repeat],
                    row["marker_curves"]["heat_error"][repeat],
                    row["marker_beta"],
                )
                for repeat in range(10)
            ],
            repeat_scores=row["repeat_scores"],
            curve_summary={
                k: dict(
                    zip(
                        ("mean", "sd"),
                        [a.tolist() for a in _mean_sd(v, 10)],
                        strict=True,
                    )
                )
                for k, v in row["curves"].items()
            },
        )
    path = output / "curie_weiss_summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    paths.append(path)
    return paths
