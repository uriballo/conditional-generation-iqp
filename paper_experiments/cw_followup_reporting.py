"""Publication reporting for completed Curie-Weiss figure data.

The follow-up bundle holds the parity, product and standard-input
families. The main-encoder bundle beside it adds the conditional and
temperature-specific models that the paper reports as its main Curie--Weiss
models: the twisted parity encoder (``curie-weiss-twisted``). A
``kappa0_control.json`` beside that bundle is summarized as the twisting
ablation.
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
TWISTED = ("shared_twisted_parity", "independent_twisted_parity")
# Main-encoder bundle.
VARIANTS = {
    "twisted": dict(
        directory="curie-weiss-twisted",
        kind="cw_twisted_final",
        families=TWISTED,
        tag="TP",
        encoder="twisted parity encoder",
        strength="twisting strength",
    ),
}
# Diagonal-conditioning baseline bundle (cw_encoder_hpo --campaign diagonal_k4).
DIAGONAL = "shared_diagonal"
DIAGONAL_DIRECTORY = "curie-weiss-diagonal"
# The diagonal baseline varies strongly across fits; a lighter band keeps the
# other models visible.
BAND_ALPHA = {DIAGONAL: 0.07}
METRICS = ("tvd", "magnetization", "heat")
KAPPA0_CURVES = ("tvd", "magnetization_error", "heat_error")
NAMES = {
    "shared_twisted_parity": "C-IQP (TP)",
    "independent_twisted_parity": r"Per-$\beta$ IQP (TP)",
    "shared_parity": "C-IQP (P)",
    "shared_product": "C-IQP (Prod)",
    DIAGONAL: "D-IQP",
    "independent_parity": r"Per-$\beta$ IQP (P)",
    "independent_unconditional": r"Per-$\beta$ IQP",
}
COLOR = {
    "shared_twisted_parity": COLORS[0],
    "independent_twisted_parity": COLORS[1],
    "shared_parity": COLORS[5],
    "shared_product": "#882255",
    DIAGONAL: "#882255",
    "independent_parity": COLORS[4],
    "independent_unconditional": COLORS[2],
}
STYLE = {
    "shared_twisted_parity": "-",
    "independent_twisted_parity": "-.",
    "shared_parity": "--",
    "shared_product": ":",
    DIAGONAL: ":",
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


def _validate_main(data, variant="twisted"):
    spec = VARIANTS[variant]
    if data.get("kind") != spec["kind"] or data.get("stage") != "final":
        raise ValueError(f"unexpected {variant} reporting scope")
    if (
        data.get("smoke") is not False
        or data.get("repeats") != 10
        or data.get("starts") != 3
    ):
        raise ValueError("final reporting requires ten repeats and three starts")
    if not data.get("families") or not set(data["families"]) <= set(spec["families"]):
        raise ValueError(f"unexpected {variant} families")
    if spec["families"][0] not in data["families"]:
        raise ValueError(f"{variant} bundle lacks its conditional model")
    for name, row in data["families"].items():
        _validate_family(name, row)
    return data


def main_bundle(data_dir):
    """Variant name and figure-data path of the main-encoder bundle, if any."""
    for variant, spec in VARIANTS.items():
        path = Path(data_dir).parent / spec["directory"] / "figure_data.json"
        if path.exists():
            return variant, path
    return None, None


def main_encoder(families):
    """Spec of the main encoder present in ``families``; ``None`` if absent."""
    for variant, spec in VARIANTS.items():
        if spec["families"][0] in families:
            shared, independent = spec["families"]
            return dict(
                spec,
                variant=variant,
                shared=shared,
                independent=independent if independent in families else None,
                missing=() if independent in families else (independent,),
            )
    return None


def diagonal_bundle(data_dir):
    """Figure-data path of the diagonal-conditioning baseline, if present."""
    path = Path(data_dir).parent / DIAGONAL_DIRECTORY / "figure_data.json"
    return path if path.exists() else None


def load(data_dir):
    """Follow-up families merged with the main-encoder bundle.

    The diagonal-conditioning baseline
    is added as ``shared_diagonal`` when its bundle is present.
    """
    data_dir = Path(data_dir)
    data = _validate(json.loads((data_dir / "figure_data.json").read_text()))
    families = dict(data["families"])
    variant, path = main_bundle(data_dir)
    if variant is not None:
        families.update(
            _validate_main(json.loads(path.read_text()), variant)["families"]
        )
    diagonal = diagonal_bundle(data_dir)
    if diagonal is not None:
        row = json.loads(diagonal.read_text())
        _validate_family(DIAGONAL, row)
        families[DIAGONAL] = row
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


def _table(path, header, rows, caption, label, wide=True, breaks=()):
    env = "table*" if wide else "table"
    path.write_text(
        f"\\begin{{{env}}}[{'t' if wide else '!htbp'}]\n\\centering\n"
        + ("\\small\n" if wide else "\\footnotesize\\setlength{\\tabcolsep}{2pt}\n")
        + "\\caption{"
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
        + "\n".join(
            " & ".join(str(x) for x in row)
            + " \\\\"
            + ("\n\\midrule" if index in breaks else "")
            for index, row in enumerate(rows)
        )
        + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n"
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


def _order(families):
    spec = main_encoder(families)
    shared, independent = spec["families"] if spec else TWISTED
    return (
        shared,
        "shared_parity",
        "shared_product",
        independent,
        "independent_parity",
        "independent_unconditional",
    )


def render(data_dir, output_dir):
    data = load(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    families = data["families"]
    spec = main_encoder(families)
    has_main = spec is not None
    main = spec["shared"] if has_main else "shared_parity"
    independent = spec["families"][1] if has_main else None
    m = (2 * np.arange(17) - 16) / 16
    paths = []

    # Magnetization profiles at the three test temperatures.
    baseline = DIAGONAL if DIAGONAL in families else "shared_product"
    shown = _present(
        families,
        (
            (main, "shared_parity", baseline)
            if has_main
            else ("shared_parity", baseline)
        ),
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
                m,
                mean - sd,
                mean + sd,
                color=COLOR[name],
                alpha=BAND_ALPHA.get(name, 0.15),
            )
        ax[index].set(title=rf"$\beta={beta:g}$")
        ax[index].spines[["top", "right"]].set_visible(False)
        ax[index].set_ylim(bottom=0)
    ax[1].set_xlabel(r"Magnetization $m$", fontsize=10)
    fig.supylabel("Probability mass", fontsize=10)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    paths += _save(fig, output, "curie_weiss_model_comparison")

    # Mean absolute magnetization and energy fluctuations across temperature.
    curves = _present(families, (main, "shared_parity", DIAGONAL))
    points = _present(
        families,
        (independent, "independent_unconditional")
        if has_main
        else ("independent_parity", "independent_unconditional"),
    )
    fig = plt.figure(figsize=(170 * MM, 68 * MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(4.5, 0.7), hspace=0.02)
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
            a.fill_between(
                beta,
                mean - sd,
                mean + sd,
                color=COLOR[name],
                alpha=BAND_ALPHA.get(name, 0.15),
            )
        for name in points:
            mean, sd = _marker(families[name], key)
            a.errorbar(
                families[name]["marker_beta"],
                mean,
                yerr=sd,
                fmt="s" if name == "independent_unconditional" else "o",
                color=COLOR[name],
                label=NAMES[name],
                capsize=2,
            )
        a.set(xlabel=r"$\beta$", ylabel=ylabel)
        a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(-0.03, 2.08)
        a.set_ylim(bottom=0)
        a.plot(
            [0.1, 0.7, 1.3, 1.95],
            [0.015] * 4,
            "v",
            color=".5",
            transform=a.get_xaxis_transform(),
            markersize=3,
        )
    # The temperature-specific IQP circuit overshoots at beta = 2, so the
    # full range is shown with a zoom on the physical range of the target.
    inset = ax[1].inset_axes([0.08, 0.46, 0.42, 0.42])
    inset.plot(beta, targets[1], "k--")
    for name in curves:
        mean, sd = _mean_sd(families[name]["curves"]["heat"], 10)
        inset.plot(beta, mean, color=COLOR[name], linestyle=STYLE[name])
        inset.fill_between(
            beta,
            mean - sd,
            mean + sd,
            color=COLOR[name],
            alpha=BAND_ALPHA.get(name, 0.15),
        )
    for name in points:
        mean, sd = _marker(families[name], "heat")
        inset.errorbar(
            families[name]["marker_beta"],
            mean,
            yerr=sd,
            fmt="s" if name == "independent_unconditional" else "o",
            color=COLOR[name],
            capsize=1.5,
            markersize=3,
        )
    inset.set(xlim=(0, 2), ylim=(0, 1), xticks=(0, 1, 2), yticks=(0, 0.5, 1))
    inset.set_title(r"Zoom: $0\leq c_{\mathrm{fluc}}\leq1$", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.spines[["top", "right"]].set_visible(False)
    handles, labels = ax[0].get_legend_handles_labels()
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=3, fontsize=8)
    paths += _save(fig, output, "curie_weiss_physical_response")

    # Shared models with different encoders on the test grid.
    shared = _present(families, (main, "shared_parity", baseline))
    fig, a = plt.subplots(figsize=(85 * MM, 66 * MM), layout="constrained")
    for name in shared:
        row = families[name]
        x = np.asarray(row["beta"])
        mean, sd = _mean_sd(row["curves"]["tvd"], 10)
        a.plot(x, mean, color=COLOR[name], linestyle=STYLE[name], label=NAMES[name])
        a.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=COLOR[name],
            alpha=BAND_ALPHA.get(name, 0.15),
        )
    # Per-temperature circuits exist only at the three test temperatures.
    for name in _present(families, (independent, "independent_unconditional")):
        row = families[name]
        mean, sd = _mean_sd(row["marker_curves"]["tvd"], 10)
        a.errorbar(
            row["marker_beta"],
            mean,
            yerr=sd,
            fmt="o" if name == independent else "s",
            color=COLOR[name],
            label=NAMES[name],
            capsize=2,
            markersize=4,
        )
    a.set(xlabel=r"$\beta$", ylabel="Full TVD")
    a.set_xlim(-0.03, 2.08)
    a.set_ylim(bottom=0)
    a.spines[["top", "right"]].set_visible(False)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        fontsize=8,
        columnspacing=1.2,
        handlelength=1.8,
    )
    paths += _save(fig, output, "curie_weiss_encoder_comparison")

    # Combined main-text figure: (a) full TVD on the test grid, (b) energy
    # fluctuations, both with the per-beta circuits as markers.
    fig, (a_tvd, a_c) = plt.subplots(
        1, 2, figsize=(170 * MM, 66 * MM), layout="constrained"
    )
    marker_rows = _present(families, (independent, "independent_unconditional"))
    target_c = np.asarray(families[main]["curves"]["target_heat"])[0]
    beta_grid = np.asarray(families[main]["beta"])
    a_c.plot(beta_grid, target_c, "k--", label="Exact target")
    for name in shared:
        row = families[name]
        x = np.asarray(row["beta"])
        for ax_, key in ((a_tvd, "tvd"), (a_c, "heat")):
            mean, sd = _mean_sd(row["curves"][key], 10)
            ax_.plot(
                x, mean, color=COLOR[name], linestyle=STYLE[name], label=NAMES[name]
            )
            ax_.fill_between(
                x,
                mean - sd,
                mean + sd,
                color=COLOR[name],
                alpha=BAND_ALPHA.get(name, 0.15),
            )
    for name in marker_rows:
        row = families[name]
        for ax_, key in ((a_tvd, "tvd"), (a_c, "heat")):
            mean, sd = _marker(row, key)
            betas_ = row["marker_beta"]
            ax_.errorbar(
                betas_,
                mean,
                yerr=sd,
                fmt="s" if name == "independent_unconditional" else "o",
                color=COLOR[name],
                label=NAMES[name] if ax_ is a_tvd else None,
                capsize=2,
                markersize=4,
            )
    a_tvd.set(xlabel=r"$\beta$", ylabel="Full TVD")
    a_c.set(xlabel=r"$\beta$", ylabel=r"$c_{\mathrm{fluc}}$ per spin")
    for ax_, tag in ((a_tvd, "(a)"), (a_c, "(b)")):
        ax_.set_xlim(-0.03, 2.08)
        ax_.set_ylim(bottom=0)
        ax_.spines[["top", "right"]].set_visible(False)
        ax_.set_title(tag, loc="left", fontsize=9)
    # Zoom on the range of the target fluctuations, kept in the top-left corner.
    inset = a_c.inset_axes([0.08, 0.46, 0.42, 0.42])
    inset.plot(beta_grid, target_c, "k--")
    for name in shared:
        row = families[name]
        x = np.asarray(row["beta"])
        mean, sd = _mean_sd(row["curves"]["heat"], 10)
        inset.plot(x, mean, color=COLOR[name], linestyle=STYLE[name])
        inset.fill_between(
            x, mean - sd, mean + sd, color=COLOR[name], alpha=BAND_ALPHA.get(name, 0.15)
        )
    for name in marker_rows:
        mean, sd = _marker(families[name], "heat")
        inset.errorbar(
            families[name]["marker_beta"],
            mean,
            yerr=sd,
            fmt="s" if name == "independent_unconditional" else "o",
            color=COLOR[name],
            capsize=1.5,
            markersize=3,
        )
    inset.set(xlim=(0, 2.05), ylim=(0, 1), xticks=(0, 1, 2), yticks=(0, 0.5, 1))
    inset.set_title(r"Zoom: $0\leq c_{\mathrm{fluc}}\leq1$", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.spines[["top", "right"]].set_visible(False)
    handles, labels = a_tvd.get_legend_handles_labels()
    h_c, l_c = a_c.get_legend_handles_labels()
    handles, labels = [h_c[0], *handles], [l_c[0], *labels]
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=8)
    paths += _save(fig, output, "curie_weiss_tvd_fluctuations")
    paths += export_tables(data, output)
    paths += export_main_table(data, output)
    provenance = output / "curie_weiss_followup.provenance.json"
    inputs = [Path(data_dir) / "figure_data.json"]
    variant, extra = main_bundle(data_dir)
    if extra is not None:
        inputs.append(extra)
    if DIAGONAL in families:
        inputs.append(diagonal_bundle(data_dir))
    control = None if extra is None else extra.parent / "kappa0_control.json"
    if control is not None and control.exists():
        inputs.append(control)
        paths.append(export_kappa0(data, json.loads(control.read_text()), output))
    record = dict(
        input_sha256={
            (
                str(path.parent.name)
                if path.name == "figure_data.json"
                else f"{path.parent.name}/{path.name}"
            ): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage="final",
        repeats=10,
        aggregation=(
            "within-repeat metrics; means and sample SD ddof=1; all repeats retained"
        ),
        selection_changed=False,
    )
    if variant == "twisted":
        record["main_encoder"] = dict(
            variant=variant,
            families=[name for name in TWISTED if name in families],
            omitted=list(spec["missing"]),
            note=(
                ", ".join(spec["missing"])
                + " is not bundled, so its rows and curves are omitted"
                if spec["missing"]
                else "all main-encoder families are bundled"
            ),
        )
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    paths.append(provenance)
    return paths


def _cell(values):
    mean, sd = _mean_sd(values, 10)
    return rf"${float(mean):.4f}\pm{float(sd):.4f}$"


def _bold(cell):
    mean, sd = cell[1:-1].split(r"\pm")
    return rf"$\mathbf{{{mean}}}\pm{sd}$"


def _bold_best(rows, start, group=None):
    """Bold the lowest rounded mean per column, within rows sharing ``group``."""
    groups = {}
    for row in rows:
        groups.setdefault(None if group is None else row[group], []).append(row)
    for members in groups.values():
        for j in range(start, len(members[0])):
            means = [float(r[j][1:].split(r"\pm")[0]) for r in members]
            for r, m in zip(members, means, strict=True):
                if m == min(means):
                    r[j] = _bold(r[j])
    return rows


def export_main_table(data, output):
    """Compact Curie--Weiss comparison of the IQP models for Sec. VI B."""
    families = data["families"]
    spec = main_encoder(families)
    if spec is None:
        return []
    shared, independent = spec["shared"], spec["independent"]
    rows_spec = [
        (shared, "136"),
        ("shared_parity", "136"),
        (DIAGONAL, "544"),
        (independent, r"$3\times136$"),
        ("independent_unconditional", r"$3\times136$"),
    ]
    rows = []
    for name, params in rows_spec:
        if name is None or name not in families:
            continue
        row = families[name]
        if name.startswith("independent"):
            grid = "--"
        else:
            beta = np.asarray(row["beta"], float)
            tvd = np.asarray(row["curves"]["tvd"], float)
            grid = _cell(np.trapezoid(tvd, beta, axis=1) / (beta[-1] - beta[0]))
        marker = row["marker_curves"]
        test = _cell(np.asarray(marker["tvd"], float).mean(axis=1))
        e_m = _cell(np.asarray(marker["magnetization_error"], float).mean(axis=1))
        e_c = _cell(np.asarray(marker["heat_error"], float).mean(axis=1) / C_STAR)
        rows.append([NAMES[name], params, grid, test, e_m, e_c])
    # Per-temperature models have no grid value, so the grid column is compared
    # among the shared models only.
    shared_rows = [r for r in rows if r[2] != "--"]
    grid_means = [float(r[2][1:].split(r"\pm")[0]) for r in shared_rows]
    for r, mean in zip(shared_rows, grid_means, strict=True):
        if mean == min(grid_means):
            r[2] = _bold(r[2])
    _bold_best(rows, 3)
    path = output / "curie_weiss_main.tex"
    groups = (
        r"& & \multicolumn{1}{c}{$101$-point grid} & "
        r"\multicolumn{3}{c}{$\beta=0.4,1,2$} \\" + "\n"
        r"\cmidrule(lr){3-3}\cmidrule(lr){4-6}" + "\n"
    )
    _table(
        path,
        [groups + "Model", "Parameters", "TVD", "TVD", "$e_m$", "$e_c$"],
        rows,
        "Curie--Weiss comparison of the IQP models at sixteen qubits. The grid "
        "column averages the full-distribution TVD over the $101$-point "
        r"evaluation grid; the other columns average over $\beta=0.4,1,2$, where "
        "the per-temperature models train one circuit each. The errors $e_m$ and "
        r"$e_c$ are defined in Eq.~\eqref{eq:cw_selection_errors}. Values are "
        "means $\\pm$ standard deviations over ten independent fits, and "
        "bold marks the lowest mean in each column.",
        "tab:cw_main",
    )
    return [path]


def export_tables(data, output):
    """Write exact marker tables and per-repeat summaries without plotting."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    families = data["families"]
    spec = main_encoder(families)
    has_main = spec is not None
    order = _order(families)
    shared_main, independent_main = order[0], order[3]
    header = ["Model", *[rf"$\beta={b:g}$" for b in MARKERS]]
    paths = []
    main = _present(
        families,
        (
            shared_main,
            "shared_parity",
            independent_main,
            "independent_parity",
            "independent_unconditional",
        ),
    )
    if has_main and spec["independent"] is not None:
        users = f"C-IQP ({spec['tag']}) and Per-$\\beta$ IQP ({spec['tag']}) use"
    elif has_main:
        users = f"C-IQP ({spec['tag']}) uses"
    inputs = (
        f"{users} the {spec['encoder']}, C-IQP (P) and "
        r"Per-$\beta$ IQP (P) the parity encoder, and Per-$\beta$ IQP the "
        "standard input "
        r"$\ket{0}^{\otimes16}$. "
        if has_main
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
        _bold_best(rows, 1)
        _table(
            path,
            header,
            rows,
            (
                "Exact "
                + ("full-distribution" if kind == "full" else "Hamming-weight profile")
                + " TVD at sixteen qubits. Values are means $\\pm$ standard "
                "deviations over ten independent fits, computed within each fit "
                "before aggregation. Lower is better, and bold marks the lowest "
                "mean in each column. Bold does not imply a significant "
                "difference, because some entries lie within one standard "
                "deviation of each other. "
                + inputs
                + "Conditional models exclude these evaluation temperatures from "
                "training and selection, whereas temperature-specific models are "
                "trained directly at each temperature."
            ),
            f"tab:curie_weiss_{kind}_tvd",
            wide=False,
        )
        paths.append(path)
    path = output / "curie_weiss_encoder_comparison.tex"
    rows = []
    for name in _present(families, (shared_main, "shared_parity", "shared_product")):
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
    _bold_best(rows, 1)
    _table(
        path,
        [*header, r"$E_{\mathrm{TVD}}$", r"$E_{\mathrm{phys}}$"],
        rows,
        (
            "Conditional models with different encoders and the same IQP "
            "ansatz, training temperatures and budgets. The three temperature "
            "columns report exact full-distribution TVD. The aggregate scores "
            "$E_{\\mathrm{TVD}}$ and $E_{\\mathrm{phys}}$, defined in the "
            "hyperparameter-optimization protocol, use the $101$-point evaluation "
            "grid. "
            "Values are means $\\pm$ standard deviations over ten "
            "independent fits. Lower values indicate smaller errors, and bold "
            "marks the lowest mean in each column."
        ),
        "tab:cw_encoder_comparison",
    )
    paths.append(path)
    path = output / "curie_weiss_selected_settings.tex"
    rows = []
    for name in _present(families, order):
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
            r"Bodyness $\bar\ell$",
        ],
        rows,
        (
            "Hyperparameter configurations selected by confirmation. The encoder "
            f"scale is $\\eta$ for the {(spec or VARIANTS['twisted'])['encoder']}, "
            f"whose {(spec or VARIANTS['twisted'])['strength']} "
            "is fixed at $\\kappa=0.4$, and $\\lambda$ for the parity and "
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
    for name in _present(families, order):
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
    _bold_best(rows, 2, group=1)
    _table(
        path,
        ["Model", "Error", *[rf"$\beta={b:g}$" for b in MARKERS]],
        rows,
        (
            "Physical-observable errors for sixteen-qubit Curie--Weiss models at "
            "three evaluation temperatures. The mean absolute magnetization "
            "error is $e_m=|\\langle|m|\\rangle_q-\\langle|m|\\rangle_p|$, and the "
            "normalized energy-fluctuation error is "
            "$e_c=|c_{\\mathrm{fluc},q}-c_{\\mathrm{fluc},p}|/c_\\star$, with "
            "$c_\\star\\approx0.823$ the maximum target value for "
            "$0\\leq\\beta\\leq2$. Values are means $\\pm$ standard "
            "deviations over ten independent fits, with errors computed "
            "within each fit before averaging. Lower values "
            "indicate smaller errors, and bold marks the lowest mean of each "
            "error in each column. Bold does not imply a significant "
            "difference, because some entries lie within one standard "
            "deviation of each other. Conditional models exclude these "
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


def _kappa0_stats(scores):
    out = {}
    for key in ("J", "P", "mean_tvd", "max_tvd"):
        mean, sd = _mean_sd([s[key] for s in scores], 10)
        out[key] = dict(mean=float(mean), sd=float(sd))
    return out


def _paired(control, reference):
    out = {}
    for key in ("J", "mean_tvd", "max_tvd"):
        difference = np.asarray([c[key] for c in control]) - np.asarray(
            [r[key] for r in reference]
        )
        out[key] = dict(
            mean_difference=float(difference.mean()),
            sd_difference=float(difference.std(ddof=1)),
            control_higher=int((difference > 0).sum()),
            repeats=len(difference),
        )
    return out


def export_kappa0(data, control, output):
    """Per-scale scores of the kappa=0 control and paired fit-by-fit comparisons.

    Each control fit pairs with the final fit of the same repeat in the main
    conditional model and in C-IQP (P). Differences are control minus reference.
    """
    families = data["families"]
    spec = main_encoder(families)
    if spec is None or control.get("kind") != "cw_kappa0_control":
        raise ValueError("kappa0 control requires its main-encoder bundle")
    main = families[spec["shared"]]
    if (
        control.get("family") != spec["shared"]
        or control.get("trial_id") != main["trial_id"]
        or control.get("smoke") is not False
        or not np.array_equal(control.get("beta"), main["beta"])
    ):
        raise ValueError("kappa0 control does not match the selected model")
    beta = np.asarray(main["beta"])
    references = {
        NAMES[spec["shared"]]: main["repeat_scores"],
        NAMES["shared_parity"]: families["shared_parity"]["repeat_scores"],
    }
    scales = {}
    for key, entry in sorted(control["scales"].items(), key=lambda x: float(x[0])):
        config = entry["config"]
        expected = dict(main["config"], kappa=0.0, beta_scale=float(key))
        if config != expected or entry["repeats"] != list(range(10)):
            raise ValueError("kappa0 control configuration or repeats differ")
        scores = [
            curve_scores(*(entry["curves"][k][r] for k in KAPPA0_CURVES), beta)
            for r in range(10)
        ]
        if any(
            not np.isclose(stored[k], scores[r][k], atol=1e-12, rtol=0)
            for r, stored in enumerate(entry["repeat_scores"])
            for k in scores[r]
        ):
            raise ValueError("kappa0 control repeat score differs")
        scales[key] = dict(
            beta_scale=float(key),
            selected=bool(np.isclose(float(key), main["config"]["beta_scale"])),
            scores=_kappa0_stats(scores),
            paired={
                label: _paired(scores, reference)
                for label, reference in references.items()
            },
        )
    payload = dict(
        kind="cw_kappa0_control_summary",
        main_encoder=spec["variant"],
        family=spec["shared"],
        trial_id=main["trial_id"],
        repeats=10,
        reference=dict(
            scores={
                label: _kappa0_stats(reference)
                for label, reference in references.items()
            },
            paired={
                f"{NAMES[spec['shared']]} vs C-IQP (P)": _paired(
                    main["repeat_scores"], references[NAMES["shared_parity"]]
                )
            },
        ),
        scales=scales,
        difference_direction=(
            "first minus second (control minus reference); control_higher counts "
            "repeats with a positive difference"
        ),
        aggregation="per-fit scores on the 101-point grid; mean and sample SD ddof=1",
    )
    path = Path(output) / "curie_weiss_kappa0_control.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path
