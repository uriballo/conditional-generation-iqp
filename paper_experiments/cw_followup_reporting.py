"""Publication reporting for completed Curie-Weiss figure data.

The audited follow-up bundle holds the parity, product and standard-input
families. The squeezed bundle, when present beside it, adds the conditional
and temperature-specific models with the squeezed parity encoder, which the
paper reports as its main Curie--Weiss models.
"""

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
SQUEEZED = ("shared_squeezed_parity", "independent_squeezed_parity")
METRICS = ("tvd", "magnetization", "heat")
NAMES = {
    "shared_squeezed_parity": "Conditional IQP",
    "shared_parity": "Conditional IQP, parity input",
    "shared_product": "Conditional IQP, product input",
    "independent_squeezed_parity": "Temperature-specific IQP",
    "independent_parity": "Temperature-specific IQP, parity input",
    "independent_unconditional": "Temperature-specific IQP, standard input",
}
ORDER = (
    "shared_squeezed_parity",
    "shared_parity",
    "shared_product",
    "independent_squeezed_parity",
    "independent_parity",
    "independent_unconditional",
)
COLOR = {
    "shared_squeezed_parity": COLORS[0],
    "shared_parity": COLORS[5],
    "shared_product": COLORS[3],
    "independent_squeezed_parity": COLORS[1],
    "independent_parity": COLORS[4],
    "independent_unconditional": COLORS[2],
}
STYLE = {
    "shared_squeezed_parity": "-",
    "shared_parity": "--",
    "shared_product": ":",
    "independent_squeezed_parity": "-.",
    "independent_parity": (0, (5, 1, 1, 1)),
    "independent_unconditional": ":",
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
        _validate_family(name, data["families"][name])
    return data


def _validate_squeezed(data):
    if data.get("kind") != "cw_squeezed_final" or data.get("stage") != "final":
        raise ValueError("unexpected squeezed reporting scope")
    if (
        data.get("smoke") is not False
        or data.get("repeats") != 10
        or data.get("starts") != 3
    ):
        raise ValueError("final reporting requires ten repeats and three starts")
    if not data.get("families") or not set(data["families"]) <= set(SQUEEZED):
        raise ValueError("unexpected squeezed families")
    for name, row in data["families"].items():
        _validate_family(name, row)
    return data


def load(data_dir):
    """Audited follow-up families merged with the squeezed bundle, if present."""
    data_dir = Path(data_dir)
    data = _validate(json.loads((data_dir / "figure_data.json").read_text()))
    families = dict(data["families"])
    squeezed = data_dir.parent / "curie-weiss-squeezed" / "figure_data.json"
    if squeezed.exists():
        families.update(
            _validate_squeezed(json.loads(squeezed.read_text()))["families"]
        )
    return dict(data, families=families)


def _validate_family(name, row):
    if not isinstance(row.get("trial_id"), int) or not isinstance(
        row.get("config"), dict
    ):
        raise ValueError("invalid family selection")
    beta = np.asarray(row.get("beta"), dtype=float)
    marker = np.asarray(row.get("marker_beta", []), dtype=float)
    expected = (
        MARKERS if name.startswith("independent") else stage_betas(protocol(), "final")
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


def _table_name(name):
    return NAMES[name]


def _present(families, names):
    return [name for name in names if name in families]


def _input_tvd(row, beta):
    """TVD of the squeezed input probabilities from uniform."""
    from .squeezed_parity_encoder import SqueezedParityEncoder

    config = row["config"]
    encoder = SqueezedParityEncoder.from_betas(
        beta, 16, scale=config["beta_scale"], kappa=config["kappa"]
    )
    table = np.abs(np.asarray(encoder.input_amplitudes())) ** 2
    counts = np.asarray(encoder.counts)[None, None]
    return 0.5 * (counts * np.abs(table - 2.0**-16)).sum((1, 2))


def render(data_dir, output_dir):
    data = load(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    families = data["families"]
    squeezed = "shared_squeezed_parity" in families
    main = "shared_squeezed_parity" if squeezed else "shared_parity"
    m = (2 * np.arange(17) - 16) / 16
    paths = []

    # Magnetization profiles at the three test temperatures.
    shown = _present(
        families,
        (
            ("shared_squeezed_parity", "shared_parity", "independent_squeezed_parity")
            if squeezed
            else ("shared_parity", "independent_parity")
        )
        + ("independent_unconditional",),
    )
    fig, ax = plt.subplots(1, 3, figsize=(170 * MM, 78 * MM), layout="constrained")
    for index, beta in enumerate(MARKERS):
        target = np.asarray(families[main]["marker_curves"]["target_shells"])[0, index]
        ax[index].plot(m, target, "k--", label="Exact target")
        for name in shown:
            values = np.asarray(families[name]["marker_curves"]["shells"], float)
            mean, sd = _mean_sd(values[:, index], 10)
            ax[index].plot(
                m, mean, color=COLOR[name], linestyle=STYLE[name], label=NAMES[name]
            )
            ax[index].fill_between(
                m, mean - sd, mean + sd, color=COLOR[name], alpha=0.15
            )
        ax[index].set(title=rf"$\beta={beta:g}$")
        ax[index].spines[["top", "right"]].set_visible(False)
        ax[index].set_ylim(bottom=0)
    fig.supxlabel(r"Magnetization $m$", fontsize=10, y=0.11)
    fig.supylabel("Probability mass", fontsize=10)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    paths += _save(fig, output, "curie_weiss_model_comparison")

    # Mean absolute magnetization and energy fluctuations across temperature.
    curves = _present(families, ("shared_squeezed_parity", "shared_parity"))
    points = _present(
        families,
        ("independent_squeezed_parity", "independent_unconditional")
        if squeezed
        else ("independent_parity", "independent_unconditional"),
    )
    fig = plt.figure(figsize=(170 * MM, 68 * MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(4.5, 1.6), hspace=0.05)
    ax = np.asarray([fig.add_subplot(grid[0, index]) for index in range(2)])
    beta = np.asarray(families[main]["beta"])
    targets = (
        np.asarray(families[main]["curves"]["target_shells"])[0] @ np.abs(m),
        np.asarray(families[main]["curves"]["target_heat"])[0],
    )
    for a, ylabel, key, target in zip(
        ax,
        (r"$\langle |m|\rangle$", r"$c_{\mathrm{fluc}}$ per spin"),
        ("magnetization", "heat"),
        targets,
        strict=True,
    ):
        a.plot(beta, target, "k--", label="Exact target")
        for name in curves:
            mean, sd = _mean_sd(families[name]["curves"][key], 10)
            a.plot(
                beta, mean, color=COLOR[name], linestyle=STYLE[name], label=NAMES[name]
            )
            a.fill_between(beta, mean - sd, mean + sd, color=COLOR[name], alpha=0.15)
        for name in points:
            mean, sd = _marker(families[name], key)
            a.errorbar(
                families[name]["marker_beta"],
                mean,
                yerr=sd,
                fmt="o",
                color=COLOR[name],
                label=NAMES[name],
                capsize=2,
            )
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
    inset = ax[1].inset_axes([0.08, 0.46, 0.42, 0.42])
    inset.plot(beta, targets[1], "k--")
    for name in curves:
        mean, sd = _mean_sd(families[name]["curves"]["heat"], 10)
        inset.plot(beta, mean, color=COLOR[name], linestyle=STYLE[name])
        inset.fill_between(beta, mean - sd, mean + sd, color=COLOR[name], alpha=0.15)
    for name in points:
        mean, sd = _marker(families[name], "heat")
        inset.errorbar(
            families[name]["marker_beta"],
            mean,
            yerr=sd,
            fmt="o",
            color=COLOR[name],
            capsize=1.5,
            markersize=3,
        )
    inset.set(xlim=(0, 2), ylim=(0, 1), xticks=(0, 1, 2), yticks=(0, 0.5, 1))
    inset.set_title(r"Zoom: $0\leq c_{\mathrm{fluc}}\leq1$", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.spines[["top", "right"]].set_visible(False)
    handles, labels = ax[0].get_legend_handles_labels()
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=3, fontsize=8)
    paths += _save(fig, output, "curie_weiss_physical_response")

    # Shared models with different encoders on the test grid.
    shared = _present(
        families, ("shared_squeezed_parity", "shared_parity", "shared_product")
    )
    panels = 3 if squeezed else 2
    fig, ax = plt.subplots(1, panels, figsize=(170 * MM, 62 * MM), layout="constrained")
    for name in shared:
        row = families[name]
        x = np.asarray(row["beta"])
        for a, key in zip(ax[:2], ("tvd", "heat_error"), strict=True):
            mean, sd = _mean_sd(row["curves"][key], 10)
            if key == "heat_error":
                mean, sd = mean / C_STAR, sd / C_STAR
            a.plot(x, mean, color=COLOR[name], linestyle=STYLE[name], label=NAMES[name])
            a.fill_between(x, mean - sd, mean + sd, color=COLOR[name], alpha=0.15)
        if squeezed:
            values = (
                _input_tvd(row, x)
                if name == "shared_squeezed_parity"
                else np.zeros_like(x)
            )
            ax[2].plot(x, values, color=COLOR[name], linestyle=STYLE[name])
    ax[0].set(xlabel=r"$\beta$", ylabel="Full TVD")
    ax[1].set(xlabel=r"$\beta$", ylabel="Normalized fluctuation\n" + r"error $e_c$")
    if squeezed:
        ax[2].set(xlabel=r"$\beta$", ylabel="Input-probability TVD\nfrom uniform")
        ax[2].text(
            0.97,
            0.05,
            "Parity and product inputs: 0",
            ha="right",
            transform=ax[2].transAxes,
            fontsize=7,
        )
    for index, a in enumerate(ax):
        a.set_ylim(bottom=0)
        a.spines[["top", "right"]].set_visible(False)
        a.set_title(f"({chr(97 + index)})", loc="left", fontsize=9)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    paths += _save(fig, output, "curie_weiss_encoder_comparison")
    paths += export_tables(data, output)
    provenance = output / "curie_weiss_followup.provenance.json"
    inputs = [Path(data_dir) / "figure_data.json"]
    extra = Path(data_dir).parent / "curie-weiss-squeezed" / "figure_data.json"
    if extra.exists():
        inputs.append(extra)
    provenance.write_text(
        json.dumps(
            dict(
                input_sha256={
                    str(path.parent.name): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in inputs
                },
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
    squeezed = "shared_squeezed_parity" in families
    header = ["Model", *[rf"$\beta={b:g}$" for b in MARKERS]]
    paths = []
    main = _present(
        families,
        (
            "shared_squeezed_parity",
            "shared_parity",
            "independent_squeezed_parity",
            "independent_parity",
            "independent_unconditional",
        ),
    )
    inputs = (
        "Conditional and temperature-specific IQP use the squeezed parity encoder; "
        "the parity-input models use the parity encoder, and the standard-input "
        r"model uses $\ket{0}^{\otimes16}$. "
        if squeezed
        else "Conditional IQP uses the parity encoder; the temperature-specific "
        r"models use the parity encoder or the standard input $\ket{0}^{\otimes16}$. "
    )
    for kind, extractor in (("full", _full_tvd), ("profile", _profile_tvd)):
        path = output / f"curie_weiss_{kind}_tvd.tex"
        rows = [
            [
                _table_name(name),
                *[_cell(extractor(families[name])[:, j]) for j in range(3)],
            ]
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
                "deviations over ten independent fits, computed within each fit "
                "before aggregation. Lower is better. "
                + inputs
                + "Conditional models exclude these test temperatures from training "
                "and selection, whereas temperature-specific models are trained "
                "directly at each temperature."
            ),
            f"tab:curie_weiss_{kind}_tvd",
        )
        paths.append(path)
    path = output / "curie_weiss_encoder_comparison.tex"
    rows = []
    for name in _present(
        families, ("shared_squeezed_parity", "shared_parity", "shared_product")
    ):
        row = families[name]
        scores = row["repeat_scores"]
        rows.append(
            [
                _table_name(name),
                *[_cell(_full_tvd(row)[:, j]) for j in range(3)],
                _cell([s["J"] for s in scores]),
                _cell([s["P"] for s in scores]),
            ]
        )
    _table(
        path,
        [*header, r"$E_{\mathrm{TVD}}$", r"$E_{\mathrm{phys}}$"],
        rows,
        (
            "Conditional IQP models with different encoders and the same IQP "
            "ansatz, training temperatures and budgets. The three temperature "
            "columns report exact full-distribution TVD. The aggregate scores "
            "$E_{\\mathrm{TVD}}$ and $E_{\\mathrm{phys}}$, defined in the "
            "hyperparameter-optimization protocol, use the $101$-point test grid. "
            "Values are means $\\pm$ sample standard deviations over ten "
            "independent fits. Lower values indicate smaller errors."
        ),
        "tab:cw_encoder_comparison",
    )
    paths.append(path)
    path = output / "curie_weiss_selected_settings.tex"
    rows = []
    for name in _present(families, ORDER):
        config = families[name]["config"]
        cells = [_table_name(name)]
        for key in (
            "learning_rate",
            "init_scale",
            "b1",
            "b2",
            "schedule",
            "beta_scale",
            "expected_weights",
        ):
            value = config.get(key)
            cells.append(
                "--"
                if value is None
                else f"{value:#.3g}"
                if isinstance(value, (int, float))
                else str(value)
            )
        rows.append(cells)
    _table(
        path,
        [
            "Model",
            "Learning rate",
            r"Width $s_\theta$",
            r"$b_1$",
            r"$b_2$",
            "Schedule",
            "Encoder scale",
            "Kernel weights",
        ],
        rows,
        (
            "Hyperparameter configurations selected by confirmation. The encoder "
            "scale is $a$ for the squeezed parity encoder, whose squeezing "
            "strength is fixed at $\\kappa=0.4$, and $\\lambda$ for the parity and "
            "product encoders. Each temperature-specific family's settings apply "
            "jointly to three circuits with separate learned phases. Widths are in "
            "radians, and $b_1$ and $b_2$ are the \\texttt{Adam} moment "
            "parameters. Continuous values are rounded to three significant digits."
        ),
        "tab:cw_selected_settings",
    )
    paths.append(path)
    path = output / "curie_weiss_physics_errors.tex"
    rows = []
    for name in _present(families, ORDER):
        row = families[name]
        magnetization = np.asarray(
            row["marker_curves"]["magnetization_error"], dtype=float
        )
        heat = np.asarray(row["marker_curves"]["heat_error"], dtype=float) / C_STAR
        rows.append(
            [
                _table_name(name),
                "$e_m$",
                *[_cell(magnetization[:, j]) for j in range(3)],
            ]
        )
        rows.append(
            [_table_name(name), "$e_c$", *[_cell(heat[:, j]) for j in range(3)]]
        )
    _table(
        path,
        ["Model", "Error", *[rf"$\beta={b:g}$" for b in MARKERS]],
        rows,
        (
            "Physical-observable errors for sixteen-qubit Curie--Weiss models at "
            "three test temperatures. The mean absolute magnetization "
            "error is $e_m=|\\langle|m|\\rangle_q-\\langle|m|\\rangle_p|$, and the "
            "normalized energy-fluctuation error is "
            "$e_c=|c_{\\mathrm{fluc},q}-c_{\\mathrm{fluc},p}|/c_\\star$, with "
            "$c_\\star\\approx0.823$ the maximum target value for "
            "$0\\leq\\beta\\leq2$. Values are means $\\pm$ sample standard "
            "deviations over ten independent fits, with errors computed "
            "within each fit before averaging. Lower values "
            "indicate smaller errors. Conditional models exclude these "
            "temperatures from training and selection, whereas "
            "temperature-specific models are trained directly at each temperature."
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
