"""Review figures for the frozen scalar Gaussian translation benchmark.

The CLI reads validation/audit manifests, their job records, extracted long-form
CSV rows, and optional optimization diagnostics JSON. It writes four compact
PDF+PNG figure classes without selecting runs by score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import plotting
from experiments.common.metrics import hamming_profile

from . import analysis, protocol


def _finish(fig: plt.Figure, output: Path, stem: str) -> tuple[Path, Path]:
    base = output / stem
    plotting.save_figure(fig, base)
    return base.with_suffix(".pdf"), base.with_suffix(".png")


def _axes(fig: plt.Figure, count: int, cols: int = 2) -> list[plt.Axes]:
    rows = (count + cols - 1) // cols
    return [fig.add_subplot(rows, cols, index + 1) for index in range(count)]


def _clean(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)


def build_profile_figure(profiles: list[dict[str, Any]]) -> plt.Figure:
    """Plot median-nearest and worst eligible profiles for four regimes."""
    plotting.apply_style()
    if len(profiles) != 8:
        raise ValueError("profile figure requires two examples for four regimes")
    fig = plt.figure(
        figsize=(170 * plotting.MM, 174 * plotting.MM), layout="constrained"
    )
    axes = _axes(fig, 8, cols=2)
    for axis, item in zip(axes, profiles, strict=True):
        weights = np.arange(protocol.N_VISIBLE + 1)
        target = hamming_profile(np.asarray(item["target"]), protocol.N_VISIBLE)
        generated = hamming_profile(np.asarray(item["generated"]), protocol.N_VISIBLE)
        axis.plot(
            weights, target, color="0.25", linestyle="--", marker="o", label="Target"
        )
        axis.plot(
            weights,
            generated,
            color=plotting.OKABE_ITO_COLORS[0],
            marker="s",
            label="Model",
        )
        axis.set_title(
            f"{item['label']}\n"
            f"seed {item['seed']} | c={item['condition']:.3f} | "
            f"TVD={item['tvd']:.3f}",
            fontsize=8,
            linespacing=1.15,
        )
        axis.set_xlabel("Hamming weight")
        axis.set_ylabel("Shell probability")
        axis.set_xticks((0, 2, 4, 6, 8))
        axis.legend(fontsize=8)
        _clean(axis)
    return fig


def build_tvd_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    """Show every available seed curve with training anchors and regime shading."""
    plotting.apply_style()
    protocols = ("interpolation", "near", "far")
    encoders = ("broadcast_affine", "qubit_affine", "anti_block")
    fig = plt.figure(
        figsize=(170 * plotting.MM, 72 * plotting.MM), layout="constrained"
    )
    axes = _axes(fig, len(protocols), cols=3)
    colors = dict(zip(encoders, plotting.OKABE_ITO_COLORS[:3], strict=True))
    for axis, name in zip(axes, protocols, strict=True):
        subset = [
            r
            for r in rows
            if r.get("protocol") == name
            and r.get("category") != "training"
            and int(r.get("anchor_count", 0)) == 3
            and r.get("placement") == "even"
            and r.get("budget_mode") == "fixed_total"
        ]
        if not subset:
            axis.text(0.5, 0.5, "No rows", transform=axis.transAxes, ha="center")
            continue
        low, high = protocol.protocol_interval(name)
        axis.axvspan(low, high, color="0.9", zorder=0, label="Training interval")
        if high < 1:
            axis.axvspan(high, 1, color="#f4e7dc", zorder=0, label="Extrapolation")
        for encoder in encoders:
            for seed in sorted(
                {int(r["optimizer_seed"]) for r in subset if r["encoder"] == encoder}
            ):
                curve = sorted(
                    (
                        r
                        for r in subset
                        if r["encoder"] == encoder and int(r["optimizer_seed"]) == seed
                    ),
                    key=lambda r: float(r["condition"]),
                )
                if curve:
                    axis.plot(
                        [float(r["condition"]) for r in curve],
                        [float(r["bitstring_tvd"]) for r in curve],
                        color=colors[encoder],
                        alpha=0.28,
                        linewidth=0.9,
                    )
            med = []
            conditions = sorted(
                {float(r["condition"]) for r in subset if r["encoder"] == encoder}
            )
            for condition in conditions:
                points = [
                    float(r["bitstring_tvd"])
                    for r in subset
                    if r["encoder"] == encoder and float(r["condition"]) == condition
                ]
                med.append(float(np.median(points)))
            if conditions:
                axis.plot(
                    conditions, med, color=colors[encoder], marker="o", label=encoder
                )
        anchor_rows = [
            r
            for r in rows
            if r.get("protocol") == name
            and r.get("category") == "training"
            and int(r.get("anchor_count", 0)) == 3
            and r.get("placement") == "even"
            and r.get("budget_mode") == "fixed_total"
        ]
        anchors = sorted({float(r["condition"]) for r in anchor_rows})
        if anchors:
            axis.scatter(
                anchors,
                np.zeros(len(anchors)),
                marker="|",
                s=50,
                color="black",
                label="Training anchors",
            )
        axis.set(
            title=name.replace("_", " "),
            xlabel="Condition $c$",
            ylabel="Exact TVD",
            xlim=(0, 1),
        )
        _clean(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    return fig


def build_optimization_figure(diagnostics: dict[str, Any]) -> plt.Figure:
    """Compare general RY and anti-block loss traces and restart outcomes."""
    plotting.apply_style()
    fig = plt.figure(
        figsize=(170 * plotting.MM, 72 * plotting.MM), layout="constrained"
    )
    loss_ax, restart_ax, gradient_ax = fig.subplots(1, 3)
    colors = {
        "broadcast_affine": plotting.OKABE_ITO_COLORS[0],
        "qubit_affine": plotting.OKABE_ITO_COLORS[1],
        "anti_block": plotting.OKABE_ITO_COLORS[2],
    }
    display = {
        "broadcast_affine": "Broadcast",
        "qubit_affine": "Qubit affine",
        "anti_block": "Anti-block",
    }
    traces = diagnostics.get("traces", [])
    if not traces and diagnostics.get("restart_results"):
        traces = [
            {
                "encoder": row["encoder"],
                "steps": row["train_curve"]["steps"],
                "loss": row["train_curve"]["loss"],
            }
            for row in diagnostics["restart_results"]
        ]
    for encoder in colors:
        selected = [trace for trace in traces if trace["encoder"] == encoder]
        for trace in selected:
            loss_ax.plot(
                np.asarray(trace["steps"], dtype=float),
                np.asarray(trace["loss"], dtype=float),
                color=colors[encoder],
                alpha=0.18,
                linewidth=0.7,
            )
        if selected:
            steps = np.asarray(selected[0]["steps"], dtype=float)
            losses = np.stack([np.asarray(trace["loss"]) for trace in selected])
            loss_ax.plot(
                steps,
                np.median(losses, axis=0),
                color=colors[encoder],
                linewidth=1.8,
                label=display[encoder],
            )
    loss_ax.set(
        xlabel="Update", ylabel="Training MMD loss", title="Optimization traces"
    )
    _clean(loss_ax)
    restart_data = diagnostics.get("restarts", {})
    if not restart_data and diagnostics.get("restart_results"):
        restart_data = {
            encoder: [
                row["final_parameters"]["training_tvd_mean"]
                for row in diagnostics["restart_results"]
                if row["encoder"] == encoder
            ]
            for encoder in colors
        }
    encoders = [encoder for encoder in colors if restart_data.get(encoder)]
    values = [restart_data[name] for name in encoders]
    if values:
        box = restart_ax.boxplot(
            values,
            tick_labels=[display[name] for name in encoders],
            patch_artist=True,
            showfliers=True,
        )
        for patch, encoder in zip(box["boxes"], encoders, strict=True):
            patch.set_facecolor(colors.get(encoder, "0.7"))
            patch.set_alpha(0.55)
    else:
        restart_ax.text(
            0.5, 0.5, "No restart metrics", transform=restart_ax.transAxes, ha="center"
        )
    restart_ax.set(title="Restart outcomes", ylabel="Final exact training TVD")
    restart_ax.tick_params(axis="x", rotation=18, labelsize=8)
    _clean(restart_ax)
    gradient_rows = diagnostics.get("matched_initialization_gradients", {}).get(
        "rows", []
    )
    gradient_values = {
        encoder: [
            row["gradient_norm"] for row in gradient_rows if row["encoder"] == encoder
        ]
        for encoder in colors
    }
    if any(gradient_values.values()):
        names = [name for name, values in gradient_values.items() if values]
        grad_box = gradient_ax.boxplot(
            [gradient_values[name] for name in names],
            tick_labels=[display[name] for name in names],
            patch_artist=True,
            showfliers=True,
        )
        for patch, encoder in zip(grad_box["boxes"], names, strict=True):
            patch.set_facecolor(colors[encoder])
            patch.set_alpha(0.55)
    elif diagnostics.get("gradient_norms"):
        grad = diagnostics["gradient_norms"]
        gradient_ax.plot(grad["steps"], grad["values"], color="0.35", marker="o")
    else:
        gradient_ax.text(
            0.5,
            0.5,
            "No gradient metrics",
            transform=gradient_ax.transAxes,
            ha="center",
        )
    gradient_ax.set(title="Matched initial gradients", ylabel="Gradient norm")
    gradient_ax.tick_params(axis="x", rotation=18, labelsize=8)
    _clean(gradient_ax)
    handles, labels = loss_ax.get_legend_handles_labels()
    if handles:
        loss_ax.legend(handles, labels, fontsize=7)
    return fig


def build_aggregate_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    """Show anchor-count and budget effects separately by encoder/protocol."""
    plotting.apply_style()
    fig = plt.figure(
        figsize=(180 * plotting.MM, 150 * plotting.MM), layout="constrained"
    )
    encoders = ("broadcast_affine", "qubit_affine", "anti_block")
    protocols = ("interpolation", "near", "far")
    colors = dict(
        zip(
            ("fixed_total", "matched_exposure"),
            plotting.OKABE_ITO_COLORS[:2],
            strict=True,
        )
    )
    axes = fig.subplots(3, 3, sharey=True)
    for row_index, encoder in enumerate(encoders):
        for col_index, name in enumerate(protocols):
            axis = axes[row_index, col_index]
            category = {
                "interpolation": "held_out_interpolation",
                "near": "near_extrapolation",
                "far": "far_extrapolation",
            }[name]
            subset = [
                r
                for r in rows
                if r.get("encoder") == encoder
                and r.get("protocol") == name
                and r.get("category") == category
            ]
            for mode, color in colors.items():
                mode_rows = [r for r in subset if r.get("budget_mode") == mode]
                counts = sorted({int(r["anchor_count"]) for r in mode_rows})
                medians = []
                for count in counts:
                    group = [r for r in mode_rows if int(r["anchor_count"]) == count]
                    values = np.asarray([float(r["bitstring_tvd"]) for r in group])
                    median = float(np.median(values))
                    medians.append(median)
                    # Keep seed/layout points visible; summarize with the line.
                    jitter = (
                        np.linspace(-0.055, 0.055, len(group))
                        if len(group) > 1
                        else [0.0]
                    )
                    axis.scatter(
                        np.full(len(group), count, dtype=float) + jitter,
                        values,
                        color=color,
                        alpha=0.28,
                        s=11,
                        linewidths=0,
                        zorder=2,
                    )
                if counts:
                    axis.plot(
                        counts,
                        medians,
                        color=color,
                        marker="o",
                        label=mode.replace("_", " "),
                    )
            axis.set_title(f"{encoder.replace('_', ' ')} · {name}", fontsize=8)
            axis.set_xlabel("Anchor count")
            if col_index == 0:
                axis.set_ylabel("Exact TVD")
            axis.grid(axis="y", color="0.9", linewidth=0.5)
            _clean(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    return fig


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric = (
        "condition",
        "bitstring_tvd",
        "optimizer_seed",
        "anchor_count",
        "layout_seed",
    )
    for row in rows:
        for key in numeric:
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def _select_profile_rows(
    rows: list[dict[str, Any]], category: str, name: str, split: str, hp_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """Select the median-nearest and worst TVD rows with deterministic ties."""
    candidates = [
        row
        for row in rows
        if row.get("phase") == split
        and row.get("encoder") == "broadcast_affine"
        and row.get("protocol") == name
        and row.get("category") == category
        and int(row.get("anchor_count", 0)) == 3
        and row.get("placement") == "even"
        and row.get("budget_mode") == "fixed_total"
        and row.get("hyperparameter_id") == hp_id
    ]
    if not candidates:
        raise ValueError(f"no eligible {category} rows for {split}, hp={hp_id}")
    median_tvd = float(np.median([float(row["bitstring_tvd"]) for row in candidates]))
    median_nearest = min(
        candidates,
        key=lambda row: (
            abs(float(row["bitstring_tvd"]) - median_tvd),
            int(row["optimizer_seed"]),
            float(row["condition"]),
        ),
    )
    worst = min(
        candidates,
        key=lambda row: (
            -float(row["bitstring_tvd"]),
            int(row["optimizer_seed"]),
            float(row["condition"]),
        ),
    )
    return [("Median-nearest", median_nearest), ("Worst TVD", worst)]


def _profiles(
    rows: list[dict[str, Any]], entries: list[dict[str, Any]], split: str, hp_id: str
) -> list[dict[str, Any]]:
    requests = (
        ("training", "interpolation", 0.5),
        ("held_out_interpolation", "interpolation", 0.25),
        ("near_extrapolation", "near", 0.9),
        ("far_extrapolation", "far", 0.75),
    )
    display_categories = {
        "training": "Training",
        "held_out_interpolation": "Held interpolation",
        "near_extrapolation": "Near extrapolation",
        "far_extrapolation": "Far extrapolation",
    }
    profiles = []
    for category, name, _desired in requests:
        selected = _select_profile_rows(rows, category, name, split, hp_id)
        for label, row in selected:
            matching_entries = [
                entry
                for entry in entries
                if entry["job"].phase == split
                and entry["job"].encoder == row["encoder"]
                and entry["job"].protocol == row["protocol"]
                and entry["job"].anchor_count == int(row["anchor_count"])
                and entry["job"].placement == row["placement"]
                and entry["job"].budget_mode == row["budget_mode"]
                and entry["job"].hyperparameter_id == row["hyperparameter_id"]
                and entry["job"].optimizer_seed == int(row["optimizer_seed"])
                and entry["job"].layout_seed
                == (
                    int(row["layout_seed"])
                    if row.get("layout_seed") not in (None, "")
                    else None
                )
            ]
            if len(matching_entries) != 1:
                raise ValueError(
                    "expected one checkpoint for selected row, "
                    f"found {len(matching_entries)}"
                )
            entry = matching_entries[0]
            with np.load(entry["checkpoint"]) as data:
                if category == "training":
                    conditions = data["training_conditions"]
                    generated_values = data["train_probabilities"]
                    targets = data["target_probabilities"]
                else:
                    conditions = data["evaluation_conditions"]
                    generated_values = data["held_out_probabilities"]
                    targets = data["held_out_target_probabilities"]
                condition = float(row["condition"])
                indices = np.flatnonzero(
                    np.isclose(conditions, condition, rtol=0, atol=1e-12)
                )
                if len(indices) != 1:
                    raise ValueError(
                        f"expected one checkpoint condition {condition}, "
                        f"found {len(indices)}"
                    )
                index = int(indices[0])
                profiles.append(
                    {
                        "label": (f"{display_categories[category]} — {label.lower()}"),
                        "condition": condition,
                        "tvd": float(row["bitstring_tvd"]),
                        "seed": int(row["optimizer_seed"]),
                        "generated": generated_values[index],
                        "target": targets[index],
                    }
                )
    return profiles


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path)
    parser.add_argument("--hyperparameter-id", required=True)
    parser.add_argument("--split", choices=("validation", "audit"), default="audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = _load_csv(args.rows_csv)
    validation = analysis.load_records(args.validation_manifest, args.records_dir)[1]
    audit = analysis.load_records(args.audit_manifest, args.records_dir)[1]
    entries = validation + audit
    artifacts = [
        _finish(
            build_profile_figure(
                _profiles(rows, entries, args.split, args.hyperparameter_id)
            ),
            args.output,
            "scalar_profiles",
        ),
        _finish(build_tvd_figure(rows), args.output, "scalar_tvd_by_condition"),
        _finish(
            build_optimization_figure(
                json.loads(args.diagnostics_json.read_text())
                if args.diagnostics_json
                else {}
            ),
            args.output,
            "scalar_optimization",
        ),
        _finish(build_aggregate_figure(rows), args.output, "scalar_aggregate"),
    ]
    print("\n".join(str(path) for pair in artifacts for path in pair))


if __name__ == "__main__":
    main()
