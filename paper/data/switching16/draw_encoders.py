"""Draw the triangle encoder motif used in the manuscript."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
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
        if text.get_text() == "RX":
            text.set_text(r"$R_X$" + "\n" + r"$t_\beta$")
            text.set_fontsize(7)
    for patch in ax.patches:
        patch.set_linewidth(0.8)
    pdf = OUT / f"{filename}.pdf"
    png = OUT / f"{filename}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    motif = motif_tape()
    motif_pdf, motif_png = draw(
        motif,
        ["0i", "0j", "ij"],
        "triangle_motif_draft",
        (85, 48),
        (0.08, 0.22, 0.84, 0.62),
        9,
    )
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
    (ROOT / "paper/data/switching16/ENCODER_DIAGRAM.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
