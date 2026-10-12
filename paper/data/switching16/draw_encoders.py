"""Draw the triangle encoder motif used in the manuscript."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import pennylane as qml

MM = 1 / 25.4
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/figures"
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "savefig.dpi": 300,
        "mathtext.fontset": "cm",
    }
)


def motif_tape():
    tape = qml.tape.QuantumTape()
    with tape:
        qml.RX(0.7, wires="ij")
        qml.CNOT(wires=["ij", "0i"])
        qml.CNOT(wires=["ij", "0j"])
        qml.Barrier(wires=["0i", "0j", "ij"], only_visual=True)
        for wire in ["0i", "0j", "ij"]:
            qml.Hadamard(wires=wire)
    return tape


def draw(tape, wires, filename, size, position, fontsize):
    fig = plt.figure(figsize=(size[0] * MM, size[1] * MM))
    _, ax = qml.drawer.tape_mpl(
        tape,
        wire_order=wires,
        fig=fig,
        style="black_white",
        fontsize=fontsize,
        wire_options={"linewidth": 0.7},
        label_options={"fontsize": fontsize},
        active_wire_notches=False,
    )
    fig.set_size_inches(size[0] * MM, size[1] * MM, forward=True)
    ax.set_position(position)
    for text in ax.texts:
        if filename == "triangle_motif" and text.get_text() in wires:
            text.set_visible(False)
        if text.get_text() == "RX":
            text.set_text(r"$R_X$" + "\n" + r"$t_\beta$")
            text.set_fontsize(7)
    if filename == "triangle_motif":
        for y, wire in zip((0.75, 0.5, 0.25), wires, strict=True):
            ax.text(
                0.015,
                y,
                rf"${wire}:\ |0\rangle$",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=7,
            )
        fig.text(
            0.27,
            0.90,
            "Reference-state preparation",
            ha="center",
            va="center",
            fontsize=8,
        )
        fig.text(
            0.79,
            0.90,
            "Final Hadamards",
            ha="center",
            va="center",
            fontsize=8,
        )
    for patch in ax.patches:
        patch.set_linewidth(0.8)
    pdf = OUT / f"{filename}.pdf"
    png = OUT / f"{filename}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def draw_signed_graph():
    vertices = list(range(7))
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 4),
    ]
    root_edges = [edge for edge in edges if edge[0] == 0]
    chords = [edge for edge in edges if edge[0] != 0]
    edge_set = set(edges)
    triangles = [
        triple
        for triple in combinations(vertices, 3)
        if all(edge in edge_set for edge in combinations(triple, 2))
    ]
    cycle_rank = len(edges) - len(vertices) + 1
    assert len(vertices) == 7 and len(edges) == 16
    assert len(triangles) == 16 and cycle_rank == 10
    assert len(root_edges) == 6 and len(chords) == 10
    positions = {
        0: (0.0, 1.9),
        1: (-0.95, 0.55),
        2: (0.95, 0.55),
        3: (-0.45, -0.85),
        4: (0.45, -0.85),
        5: (-1.95, -0.45),
        6: (1.95, -0.45),
    }
    root_color, chord_color = "#0072B2", "#D55E00"
    fig, ax = plt.subplots(figsize=(85 * MM, 70 * MM))
    ax.set_position((0.05, 0.18, 0.90, 0.77))
    for edge in chords + root_edges:
        is_root = edge in root_edges
        curvature = {
            (0, 5): 0.14,
            (0, 6): -0.14,
            (2, 5): -0.42,
            (1, 6): 0.42,
        }.get(edge, 0.0)
        ax.add_patch(
            FancyArrowPatch(
                positions[edge[0]],
                positions[edge[1]],
                arrowstyle="-",
                connectionstyle=f"arc3,rad={curvature}",
                shrinkA=7,
                shrinkB=7,
                color=root_color if is_root else chord_color,
                linewidth=1.1 if is_root else 0.9,
                linestyle="-" if is_root else "--",
                zorder=2 if is_root else 1,
            )
        )
    ax.scatter(
        [positions[v][0] for v in vertices],
        [positions[v][1] for v in vertices],
        s=180,
        facecolors="white",
        edgecolors="#222222",
        linewidths=0.8,
        zorder=3,
    )
    for vertex in vertices:
        ax.text(
            *positions[vertex],
            str(vertex),
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
        )
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.7, 2.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D([], [], color=root_color, linewidth=1.1, label="Root edges (6)"),
            Line2D(
                [],
                [],
                color=chord_color,
                linewidth=0.9,
                linestyle="--",
                label="Chords (10)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        columnspacing=1.2,
    )
    pdf = OUT / "switching16_graph.pdf"
    png = OUT / "switching16_graph.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    metadata = {
        "vertices": vertices,
        "edges": edges,
        "root_edges": root_edges,
        "chords": chords,
        "triangles": triangles,
        "n_vertices": len(vertices),
        "n_edges": len(edges),
        "n_triangles": len(triangles),
        "cycle_rank": cycle_rank,
        "vertex_positions": positions,
        "root_edge_color": root_color,
        "chord_color": chord_color,
        "edge_signs_assigned": False,
    }
    return pdf, png, metadata

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    motif = motif_tape()
    motif_pdf, motif_png = draw(
        motif,
        ["0i", "0j", "ij"],
        "triangle_motif",
        (85, 60),
        (0.08, 0.18, 0.84, 0.66),
        9,
    )
    graph_pdf, graph_png, graph_metadata = draw_signed_graph()
    metadata = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "motif_operations": {
            name: sum(op.name == name for op in motif.operations)
            for name in ("RX", "CNOT", "Hadamard")
        },
        "motif_outputs": [
            str(motif_pdf.relative_to(ROOT)),
            str(motif_png.relative_to(ROOT)),
        ],
        "motif_output_sha256": {
            "pdf": hashlib.sha256(motif_pdf.read_bytes()).hexdigest(),
            "png": hashlib.sha256(motif_png.read_bytes()).hexdigest(),
        },
    }
    metadata.update(
        {
            "graph": graph_metadata,
            "graph_outputs": [
                str(graph_pdf.relative_to(ROOT)),
                str(graph_png.relative_to(ROOT)),
            ],
            "graph_output_sha256": {
                "pdf": hashlib.sha256(graph_pdf.read_bytes()).hexdigest(),
                "png": hashlib.sha256(graph_png.read_bytes()).hexdigest(),
            },
        }
    )
    (ROOT / "paper/data/switching16/ENCODER_DIAGRAM.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
