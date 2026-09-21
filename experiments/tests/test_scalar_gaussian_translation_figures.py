"""Fast synthetic render checks for the benchmark review figures."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from experiments.scalar_gaussian_translation import figures


def test_profile_figure_renders_four_regimes() -> None:
    profiles = [
        {
            "label": f"{label} · {example}",
            "condition": c,
            "seed": seed,
            "tvd": tvd,
            "target": np.full(2**8, 1 / 2**8),
            "generated": np.full(2**8, 1 / 2**8),
        }
        for label, example, c, seed, tvd in (
            ("training", "Median-nearest", 0.5, 501, 0.2),
            ("training", "Worst TVD", 0.5, 503, 0.8),
            ("held interpolation", "Median-nearest", 0.25, 501, 0.2),
            ("held interpolation", "Worst TVD", 0.25, 503, 0.8),
            ("near extrapolation", "Median-nearest", 0.9, 501, 0.2),
            ("near extrapolation", "Worst TVD", 0.9, 503, 0.8),
            ("far extrapolation", "Median-nearest", 0.75, 501, 0.2),
            ("far extrapolation", "Worst TVD", 0.75, 503, 0.8),
        )
    ]
    fig = figures.build_profile_figure(profiles)
    assert len(fig.axes) == 8
    assert all(ax.lines for ax in fig.axes)
    titles = [axis.get_title() for axis in fig.axes]
    assert all(len(title.splitlines()) == 2 for title in titles)
    assert all(
        "seed " in title and "c=" in title and "TVD=" in title for title in titles
    )
    assert any("held interpolation" in title.lower() for title in titles)
    assert any("near extrapolation" in title.lower() for title in titles)
    assert any("far extrapolation" in title.lower() for title in titles)
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    plt.close(fig)


def test_tvd_and_aggregate_figures_keep_seed_points_visible() -> None:
    rows = []
    for protocol, category, c in (
        ("interpolation", "held_out_interpolation", 0.25),
        ("near", "near_extrapolation", 0.9),
        ("far", "far_extrapolation", 0.75),
    ):
        for encoder in ("broadcast_affine", "qubit_affine", "anti_block"):
            for seed in (401, 402, 403):
                for placement, layout_seed in (("even", None), ("random", 1101)):
                    rows.append(
                        {
                            "protocol": protocol,
                            "category": category,
                            "condition": c,
                            "encoder": encoder,
                            "optimizer_seed": seed,
                            "anchor_count": 3,
                            "placement": placement,
                            "layout_seed": layout_seed,
                            "budget_mode": "fixed_total",
                            "bitstring_tvd": 0.1 + seed / 10000,
                        }
                    )
    tvd = figures.build_tvd_figure(rows)
    aggregate = figures.build_aggregate_figure(rows)
    assert len(tvd.axes) == 3
    assert len(aggregate.axes) == 9
    assert sum(len(ax.lines) for ax in tvd.axes) >= 3
    FigureCanvasAgg(tvd)
    tvd.canvas.draw()
    FigureCanvasAgg(aggregate)
    aggregate.canvas.draw()
    plt.close(tvd)
    plt.close(aggregate)


def test_profile_selection_is_median_nearest_and_worst_without_seed_argument() -> None:
    rows = [
        {
            "phase": "audit",
            "encoder": "broadcast_affine",
            "protocol": "interpolation",
            "category": "training",
            "anchor_count": 3,
            "placement": "even",
            "budget_mode": "fixed_total",
            "hyperparameter_id": "selected",
            "optimizer_seed": seed,
            "condition": 0.5,
            "bitstring_tvd": tvd,
        }
        for seed, tvd in ((501, 0.10), (503, 0.20), (505, 0.90), (402, 0.99))
    ]
    # Seed 402 is ineligible through the wrong hyperparameter slice.
    rows[-1]["hyperparameter_id"] = "other"
    selected = figures._select_profile_rows(
        rows, "training", "interpolation", "audit", "selected"
    )
    assert [(label, row["optimizer_seed"]) for label, row in selected] == [
        ("Median-nearest", 503),
        ("Worst TVD", 505),
    ]
    assert [row["bitstring_tvd"] for _, row in selected] == [0.20, 0.90]


def test_aggregate_figure_facets_every_encoder_protocol_pair() -> None:
    rows = []
    for encoder_index, encoder in enumerate(
        ("broadcast_affine", "qubit_affine", "anti_block")
    ):
        for protocol_index, (name, category) in enumerate(
            (
                ("interpolation", "held_out_interpolation"),
                ("near", "near_extrapolation"),
                ("far", "far_extrapolation"),
            )
        ):
            for mode, offset in (("fixed_total", 0.0), ("matched_exposure", 0.1)):
                rows.append(
                    {
                        "encoder": encoder,
                        "protocol": name,
                        "category": category,
                        "budget_mode": mode,
                        "anchor_count": 3,
                        "bitstring_tvd": 0.05
                        + encoder_index * 0.2
                        + protocol_index * 0.02
                        + offset,
                    }
                )
    fig = figures.build_aggregate_figure(rows)
    assert len(fig.axes) == 9
    assert "anti block · far" in fig.axes[8].get_title()
    assert len(fig.axes[8].lines) == 2
    assert all(len(axis.lines) == 2 for axis in fig.axes)
    # Each facet retains its encoder/protocol-specific values; none is pooled.
    assert np.isclose(fig.axes[0].lines[0].get_ydata()[0], 0.05)
    assert np.isclose(fig.axes[8].lines[0].get_ydata()[0], 0.49)
    plt.close(fig)


def test_optimization_figure_renders_traces_restarts_and_gradient() -> None:
    diagnostics = {
        "traces": [
            {"encoder": name, "steps": [0, 1, 2], "loss": [1.0, 0.7, 0.5]}
            for name in ("broadcast_affine", "qubit_affine", "anti_block")
        ],
        "restarts": {
            "broadcast_affine": [0.3, 0.4, 0.35],
            "anti_block": [0.25, 0.5, 0.4],
        },
        "gradient_norms": {"steps": [0, 2], "values": [2.0, 0.5]},
    }
    fig = figures.build_optimization_figure(diagnostics)
    assert len(fig.axes) == 3
    assert fig.axes[0].lines
    assert fig.axes[1].patches
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    plt.close(fig)
