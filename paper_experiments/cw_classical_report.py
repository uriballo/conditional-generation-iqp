"""Validated per-fit summaries and figures for the classical CW comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .cw_classical_config import (
    CAPACITIES,
    FAMILIES,
    betas,
    classical_stream,
    limits,
    model,
    protocol,
    target_stream,
)
from .cw_followup_fit import write_json
from .cw_followup_metrics import curve_scores
from .cw_followup_reporting import NAMES as IQP_NAMES
from .cw_followup_reporting import load as load_iqp
from .cw_followup_reporting import main_bundle, main_encoder
from .target import curie_weiss_target

RESULTS_ROOT = Path(__file__).parents[1] / "results"
MM = 1 / 25.4
STYLE = {
    "C-IQP (TP)": ("#0072B2", "-"),
    "C-IQP (SP)": ("#0072B2", "-"),
    "C-IQP (P)": ("#56B4E9", "--"),
    "rbm": ("#D55E00", "-."),
    "transformer": ("#009E73", ":"),
    "vae": ("#CC79A7", (0, (5, 1))),
}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _iqp_references(iqp):
    """Main-encoder C-IQP model (twisted parity) and C-IQP (P)."""
    spec = main_encoder(iqp["families"])
    shared = spec["shared"] if spec else "shared_twisted_parity"
    variant = spec["variant"] if spec else "twisted"
    return variant, (
        (shared, IQP_NAMES[shared]),
        ("shared_parity", IQP_NAMES["shared_parity"]),
    )


def _validate_curves(curves, grid, repeats, n):
    arrays = {}
    keys = (
        "tvd",
        "profile_tvd",
        "magnetization",
        "magnetization_error",
        "heat",
        "target_heat",
        "heat_error",
        "signed_magnetization",
        "flip_defect",
        "shells",
        "target_shells",
    )
    for key in keys:
        array = np.asarray(curves.get(key), dtype=float)
        shape = (
            (repeats, len(grid), n + 1)
            if key.endswith("shells")
            else (repeats, len(grid))
        )
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"invalid curve shape/values: {key}")
        if key != "signed_magnetization" and np.any(array < -1e-10):
            raise ValueError(f"negative metric: {key}")
        arrays[key] = array
    q, p = arrays["shells"], arrays["target_shells"]
    if not np.allclose(q.sum(-1), 1, atol=1e-8, rtol=0) or not np.allclose(
        p.sum(-1), 1, atol=1e-8, rtol=0
    ):
        raise ValueError("shell probabilities are not normalized")
    target = np.stack([curie_weiss_target(n, beta).probabilities for beta in grid])
    if not np.allclose(p, target, atol=1e-8, rtol=0):
        raise ValueError("target shell probabilities differ")
    profile = abs(q - p).sum(-1) / 2
    if (
        not np.allclose(arrays["profile_tvd"], profile, atol=1e-8, rtol=0)
        or np.any(profile > arrays["tvd"] + 1e-8)
        or np.any(arrays["tvd"] > 1 + 1e-8)
    ):
        raise ValueError("inconsistent profile/full TVD")
    m = (2 * np.arange(n + 1) - n) / n
    energy = -n * m**2 / 2
    heat = np.asarray(grid) ** 2 / n * (q @ energy**2 - (q @ energy) ** 2)
    target_heat = np.asarray(grid) ** 2 / n * (p @ energy**2 - (p @ energy) ** 2)
    checks = dict(
        magnetization=q @ abs(m),
        signed_magnetization=q @ m,
        magnetization_error=abs((q - p) @ abs(m)),
        heat=heat,
        target_heat=target_heat,
        heat_error=abs(heat - target_heat),
    )
    for key, expected in checks.items():
        if not np.allclose(arrays[key], expected, atol=1e-8, rtol=1e-8):
            raise ValueError(f"per-fit shell moments differ: {key}")
    if np.any(arrays["flip_defect"] > 1e-8) or np.any(
        abs(arrays["signed_magnetization"]) > 1e-8
    ):
        raise ValueError("global spin-flip symmetry differs")


def _validate_quadrature(rows, repeats, points):
    settings = protocol()["evaluation"]["vae_quadrature"]
    if len(rows) != repeats or any(len(row) != points for row in rows):
        raise ValueError("missing VAE integration diagnostics")
    for row in rows:
        for record in row:
            errors = sorted((int(k), float(v)) for k, v in record["errors"].items())
            if (
                len(errors) < 2
                or errors[-1][0] != record["nodes"]
                or record["nodes"] > settings["max_nodes"]
                or record["tolerance"] != settings["tvd_tolerance"]
                or any(not np.isfinite(error) or error < 0 for _, error in errors)
                or any(error > settings["tvd_tolerance"] for _, error in errors[-2:])
                or errors[-1][0] != 2 * errors[-2][0]
            ):
                raise ValueError("VAE integration did not pass two refinements")


def _settings(spec):
    """Protocol without its hash fields, which changed when its text was edited."""
    hashes = ("protocol_sha256", "parent_protocol_sha256")
    return {k: v for k, v in (spec or {}).items() if k not in hashes}


def _validate(data, allow_smoke=False):
    if data.get("kind") != "cw_classical_final" or data.get("stage") != "final":
        raise ValueError("expected completed classical final bundle")
    if (
        not isinstance(data.get("smoke"), bool)
        or data.get("capacity") not in CAPACITIES
    ):
        raise ValueError("invalid bundle scope")
    smoke = data["smoke"]
    if smoke and not allow_smoke:
        raise ValueError("smoke bundles require explicit opt-in")
    budget = limits(smoke)
    if (
        data.get("repeats") != budget["repeats"]["final"]
        or data.get("starts") != budget["starts"]["final"]
    ):
        raise ValueError("final repeat/start counts differ")
    manifest = data.get("provenance", {}).get("manifest", {})
    if (
        _settings(manifest.get("protocol")) != _settings(protocol(smoke))
        or manifest.get("capacity") != data["capacity"]
        or manifest.get("smoke") != smoke
    ):
        raise ValueError("bundle protocol identity differs")
    if set(data.get("families", {})) != set(FAMILIES):
        raise ValueError("incomplete classical family set")
    repeats = data["repeats"]
    grid = betas("final", smoke)
    pool_hashes = {}
    for repeat in range(repeats):
        for beta in protocol()["training"]["shared_anchors"]:
            stream = target_stream("final", repeat, beta)
            draws = curie_weiss_target(budget["n"], beta).sample(
                budget["n_train"], np.random.default_rng(stream["seed"])
            )
            pool_hashes[repeat, beta] = hashlib.sha256(draws.tobytes()).hexdigest()
    for family, row in data["families"].items():
        if tuple(row.get("beta", ())) != grid or tuple(row.get("marker_beta", ())) != (
            0.4,
            1.0,
            2.0,
        ):
            raise ValueError("reporting grid differs")
        if row.get("repeat_ids") != list(range(repeats)):
            raise ValueError("missing or unordered final repeat IDs")
        architecture = model(family).architecture(budget["n"], data["capacity"])
        if row.get("architecture") != architecture:
            raise ValueError("reported architecture differs")
        for scope, points in (("curves", grid), ("marker_curves", (0.4, 1.0, 2.0))):
            _validate_curves(row.get(scope, {}), points, repeats, budget["n"])
        scores = row.get("repeat_scores", [])
        if len(scores) != repeats:
            raise ValueError("missing repeat scores")
        for repeat in range(repeats):
            curves = row["curves"]
            got = curve_scores(
                curves["tvd"][repeat],
                curves["magnetization_error"][repeat],
                curves["heat_error"][repeat],
                grid,
            )
            if any(
                not np.isclose(
                    scores[repeat].get(key, np.nan), value, atol=1e-8, rtol=0
                )
                for key, value in got.items()
            ):
                raise ValueError("stored per-fit score differs")
        pools = row.get("target_pools", [])
        if len(pools) != repeats:
            raise ValueError("missing paired pool identities")
        for repeat, records in enumerate(pools):
            if [record["beta"] for record in records] != protocol()["training"][
                "shared_anchors"
            ]:
                raise ValueError("paired pool anchors differ")
            for record in records:
                beta = record["beta"]
                if (
                    record["stream"] != target_stream("final", repeat, beta)
                    or record["draws"] != budget["n_train"]
                    or record["array_sha256"] != pool_hashes[repeat, beta]
                ):
                    raise ValueError("paired target pool differs")
        if family == "vae":
            _validate_quadrature(
                row.get("probability_evaluation", []), repeats, len(grid)
            )
            _validate_quadrature(
                row.get("marker_probability_evaluation", []), repeats, 3
            )
    return smoke


def _mean_sd(values):
    array = np.asarray(values, dtype=float)
    if array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("mean/sample SD requires complete finite repeated values")
    return array.mean(0), array.std(0, ddof=1)


def _bootstrap(a, b, seed, count=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite(a - b).all():
        raise ValueError("paired bootstrap requires matched finite repeat vectors")
    differences = a - b
    indices = np.random.default_rng(seed).integers(0, len(a), (count, len(a)))
    distribution = differences[indices].mean(1)
    return dict(
        mean_difference=float(differences.mean()),
        ci95=np.percentile(distribution, [2.5, 97.5]).tolist(),
        repeats=len(a),
        bootstrap_draws=count,
    )


def _summary(name, row, *, parameters, generation_parameters, probability_kind):
    grid = row["beta"]
    curves = row["curves"]
    per_fit = [
        curve_scores(t, m, c, grid)
        for t, m, c in zip(
            curves["tvd"],
            curves["magnetization_error"],
            curves["heat_error"],
            strict=True,
        )
    ]
    result = dict(
        model=name,
        parameter_count=parameters,
        generation_parameter_count=generation_parameters,
        probability_kind=probability_kind,
    )
    for key in ("J", "P", "mean_tvd", "max_tvd"):
        mean, sd = _mean_sd([record[key] for record in per_fit])
        result[key + "_mean"], result[key + "_sd"] = float(mean), float(sd)
    for scope in ("curves", "marker_curves"):
        means, sds = _mean_sd(row[scope]["tvd"])
        if scope == "curves":
            result["tvd_curve"], result["tvd_sd"] = means.tolist(), sds.tolist()
        else:
            for beta, mean, sd in zip(row["marker_beta"], means, sds, strict=True):
                result[f"tvd_beta_{beta:g}_mean"] = float(mean)
                result[f"tvd_beta_{beta:g}_sd"] = float(sd)
    result["training_seconds_including_compile_mean"] = None
    result["warm_samples_per_second_mean"] = None
    if "training_seconds_including_compile" in row:
        result["training_seconds_including_compile_mean"] = float(
            _mean_sd(row["training_seconds_including_compile"])[0]
        )
        result["warm_samples_per_second_mean"] = float(
            _mean_sd([record["samples_per_second"] for record in row["sampling"]])[0]
        )
    return result, per_fit


def _style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "legend.frameon": False,
            "axes.xmargin": 0.02,
            "axes.ymargin": 0.02,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            **(
                {
                    "text.usetex": True,
                    "text.latex.preamble": (
                        r"\usepackage{amsmath} \usepackage{amssymb} "
                        r"\usepackage{lmodern} "
                        r"\renewcommand{\familydefault}{\sfdefault} "
                        r"\usepackage{sansmath} \sansmath"
                    ),
                }
                if shutil.which("latex")
                else {"text.usetex": False, "mathtext.fontset": "cm"}
            ),
        }
    )


def render(data_path, output, iqp_data_root=None, allow_smoke=False):
    source = Path(data_path)
    data = json.loads(source.read_text())
    smoke = _validate(data, allow_smoke)
    out = Path(output)
    if not out.resolve().is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError("classical reports must write under ignored results/")
    rows, per_fits, source_hashes = [], {}, {str(source): _hash(source)}
    for family in FAMILIES:
        row = data["families"][family]
        architecture = row["architecture"]
        summary, per_fit = _summary(
            family,
            row,
            parameters=architecture["parameter_count"],
            generation_parameters=architecture.get(
                "generation_parameter_count", architecture["parameter_count"]
            ),
            probability_kind="numerically integrated" if family == "vae" else "exact",
        )
        rows.append(summary)
        per_fits[family] = per_fit
    paired = {}
    if smoke and iqp_data_root is not None:
        raise ValueError("IQP comparison is unavailable for four-spin smoke data")
    if not smoke:
        iqp_root = (
            Path(iqp_data_root)
            if iqp_data_root is not None
            else Path(__file__).parents[1] / "data"
        )
        followup = (
            iqp_root / "curie-weiss-followup"
            if (iqp_root / "curie-weiss-followup").is_dir()
            else iqp_root
        )
        iqp = load_iqp(followup)
        variant, references = _iqp_references(iqp)
        main_path = main_bundle(followup)[1] or (
            followup.parent / "curie-weiss-twisted" / "figure_data.json"
        )
        for path in (followup / "figure_data.json", main_path):
            source_hashes[str(path)] = _hash(path)
        for family, label in references:
            row = iqp["families"][family]
            if tuple(row["beta"]) != tuple(data["families"]["rbm"]["beta"]):
                raise ValueError("IQP/classical reporting grids differ")
            summary, per_fit = _summary(
                label,
                row,
                parameters=136,
                generation_parameters=136,
                probability_kind="exact",
            )
            rows.append(summary)
            per_fits[label] = per_fit
        seed = classical_stream("report", "paired-bootstrap")["seed"]
        for family in FAMILIES:
            paired[family] = {
                key: _bootstrap(
                    [row[key] for row in per_fits[family]],
                    [row[key] for row in per_fits[references[0][1]]],
                    seed,
                )
                for key in ("J", "mean_tvd")
            }
    out.mkdir(parents=True, exist_ok=True)
    if smoke:
        variant, reference_label = "twisted", "C-IQP (TP)"
    else:
        reference_label = references[0][1]
    summary = dict(
        kind="cw_classical_summary",
        smoke=smoke,
        capacity=data["capacity"],
        repeats=data["repeats"],
        beta=data["families"]["rbm"]["beta"],
        rows=rows,
        **{f"paired_against_{variant}_iqp": paired},
        paired_difference_direction=(
            f"classical minus {reference_label}; positive means larger classical error"
        ),
        pairing_basis=(
            "same parent final stage/repeat target-pool derivation; "
            "ordered IQP repeats in audited compact bundles"
        ),
        uncertainty=(
            "sample SD across fits; paired bootstrap percentile 95% CI "
            "across ten fits, not across temperatures"
        ),
        input_sha256=source_hashes,
    )
    write_json(out / "summary.json", summary)
    fields = [key for key in rows[0] if key not in ("tvd_curve", "tvd_sd")]
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _style()
    fig, ax = plt.subplots(figsize=(170 * MM, 85 * MM), layout="constrained")
    grid = np.asarray(summary["beta"])
    plotted = rows[-2:] + rows[:-2] if not smoke else rows
    for row in plotted:
        color, linestyle = STYLE[row["model"]]
        mean, sd = np.asarray(row["tvd_curve"]), np.asarray(row["tvd_sd"])
        label = {
            "rbm": "Conditional RBM",
            "transformer": "Transformer",
            "vae": "Conditional VAE",
        }.get(row["model"], row["model"])
        ax.plot(grid, mean, color=color, linestyle=linestyle, label=label)
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.12)
    ax.set(
        xlabel=r"Inverse temperature $\beta$",
        ylabel="Full-distribution TVD",
        ylim=(0, None),
        title=(
            "Smoke validation, four spins"
            if smoke
            else f"Curie--Weiss, {data['capacity']} capacity"
        ),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    pdf, png = out / "tvd_vs_beta.pdf", out / "tvd_vs_beta.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    write_json(
        out / "figure_provenance.json",
        dict(
            input_sha256=source_hashes,
            aggregation="per-fit metrics, arithmetic mean, sample SD (ddof=1)",
            uncertainty="shading is one sample standard deviation",
            filtering="all fixed final repeats; no mixtures of fitted models",
            smoke=smoke,
        ),
    )
    return [
        out / "summary.json",
        out / "summary.csv",
        pdf,
        png,
        out / "figure_provenance.json",
    ]


PAPER_NAMES = {"rbm": "RBM", "transformer": "Transformer", "vae": "VAE"}
PAPER_STYLE = {
    "C-IQP (TP)": ("#0072B2", "-"),
    "C-IQP (SP)": ("#0072B2", "-"),
    "C-IQP (P)": ("#56B4E9", "--"),
    "rbm": ("#E69F00", "-."),
    "transformer": ("#117733", ":"),
    "vae": ("#AA4499", (0, (5, 1))),
}


def paper_artifacts(classical_root, iqp_root, output):
    """Manuscript table and two-capacity TVD figure from the bundled finals."""
    from .cw_followup_reporting import _bold_best, _table
    from .figures import _save

    classical_root, iqp_root, out = Path(classical_root), Path(iqp_root), Path(output)
    if out.resolve().is_relative_to((RESULTS_ROOT.parent / "data").resolve()):
        raise ValueError("refusing to write under data/")
    iqp = load_iqp(iqp_root)
    references = _iqp_references(iqp)[1]
    main = references[0][0]
    reference = {}
    for family, label in references:
        reference[label] = _summary(
            label,
            iqp["families"][family],
            parameters=136,
            generation_parameters=136,
            probability_kind="exact",
        )[0]
    capacities, inputs = {}, {}
    for capacity in CAPACITIES:
        path = classical_root / capacity / "figure_data.json"
        data = json.loads(path.read_text())
        if _validate(data) or data["capacity"] != capacity:
            raise ValueError("paper artifacts require full final bundles")
        if tuple(data["families"]["rbm"]["beta"]) != tuple(
            iqp["families"][main]["beta"]
        ):
            raise ValueError("IQP/classical reporting grids differ")
        inputs[f"{capacity}/figure_data.json"] = _hash(path)
        capacities[capacity] = {}
        for family in FAMILIES:
            architecture = data["families"][family]["architecture"]
            capacities[capacity][family] = _summary(
                family,
                data["families"][family],
                parameters=architecture.get(
                    "training_parameter_count", architecture["parameter_count"]
                ),
                generation_parameters=architecture.get(
                    "generation_parameter_count", architecture["parameter_count"]
                ),
                probability_kind="exact",
            )[0]

    def cell(row, key):
        return rf"${row[key + '_mean']:.4f}\pm{row[key + '_sd']:.4f}$"

    def params(row):
        total, generation = row["parameter_count"], row["generation_parameter_count"]
        return f"{total}" if total == generation else f"{total} ({generation})"

    rows = [
        [label, params(row), *(cell(row, k) for k in ("J", "mean_tvd", "P"))]
        for label, row in reference.items()
    ]
    for capacity in CAPACITIES:
        for family in FAMILIES:
            row = capacities[capacity][family]
            rows.append(
                [
                    f"{capacity.capitalize()} {PAPER_NAMES[family]}",
                    params(row),
                    *(cell(row, k) for k in ("J", "mean_tvd", "P")),
                ]
            )
    _bold_best(rows, 2)
    out.mkdir(parents=True, exist_ok=True)
    table = out / "curie_weiss_classical_comparison.tex"
    _table(
        table,
        [
            "Model",
            "Parameters",
            r"$E_{\mathrm{TVD}}$",
            "Mean TVD",
            r"$E_{\mathrm{phys}}$",
        ],
        rows,
        (
            "Comparison of C-IQP models with classical conditional models on the "
            "$101$-point Curie--Weiss evaluation grid. All models share "
            "the training temperatures, target samples, update budget, "
            "hyperparameter search and selection rule. C-IQP models train on the "
            "sampled MMD objective, whereas the classical models train on their "
            "native likelihood or evidence lower bound. Parameters count all learned "
            "parameters used in training; for the VAEs, parentheses give the "
            "generator parameters alone. Mean TVD is the temperature-averaged "
            "full-distribution TVD, and Appendix~\\ref{sec:cw_hpo} defines the "
            "scores $E_{\\mathrm{TVD}}$ and $E_{\\mathrm{phys}}$. Values are "
            "means $\\pm$ standard "
            "deviations over ten independent fits. Lower is better, and bold "
            "marks the lowest mean in each column; it is not a significance "
            "claim."
        ),
        "tab:cw_classical",
        breaks=(1, 4),
    )
    settings = out / "curie_weiss_classical_settings.tex"
    setting_rows = []
    for capacity in CAPACITIES:
        data = json.loads((classical_root / capacity / "figure_data.json").read_text())
        for family in FAMILIES:
            config = data["families"][family]["config"]
            setting_rows.append(
                [
                    f"{capacity.capitalize()} {PAPER_NAMES[family]}",
                    *(
                        f"{config[key]:#.3g}"
                        for key in ("learning_rate", "init_scale", "b1", "b2")
                    ),
                    config["schedule"].capitalize(),
                ]
            )
    _table(
        settings,
        [
            "Model",
            "Learning rate",
            "Width multiplier",
            r"$b_1$",
            r"$b_2$",
            "Schedule",
        ],
        setting_rows,
        (
            "Hyperparameter configurations of the classical models selected by "
            "confirmation. The width multiplier scales the Xavier initialization, "
            "and $b_1$ and $b_2$ are the \\texttt{Adam} moment parameters. "
            "Values are rounded to three significant digits."
        ),
        "tab:cw_classical_settings",
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(170 * MM, 62 * MM), sharey=True, layout="constrained"
    )
    grid = np.asarray(iqp["families"][main]["beta"])
    for index, (ax, capacity) in enumerate(zip(axes, CAPACITIES, strict=True)):
        plotted = [*reference.items(), *capacities[capacity].items()]
        for name, row in plotted:
            color, linestyle = PAPER_STYLE[name]
            mean, sd = np.asarray(row["tvd_curve"]), np.asarray(row["tvd_sd"])
            ax.plot(
                grid,
                mean,
                color=color,
                linestyle=linestyle,
                label=PAPER_NAMES.get(name, name),
            )
            ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.12)
        ax.set(xlabel=r"$\beta$", ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            f"({chr(97 + index)}) {capacity.capitalize()} classical models",
            loc="left",
            fontsize=9,
        )
    axes[0].set_ylabel("Full TVD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    paths = _save(fig, out, "curie_weiss_classical_tvd")
    provenance = out / "curie_weiss_classical.provenance.json"
    write_json(
        provenance,
        dict(
            input_sha256=inputs,
            aggregation="per-fit metrics, arithmetic mean, sample SD (ddof=1)",
            uncertainty="shading is one sample standard deviation",
            filtering="all fixed final repeats; no mixtures of fitted models",
        ),
    )
    return [table, settings, *paths, provenance]
