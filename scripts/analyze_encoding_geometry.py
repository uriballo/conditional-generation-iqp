"""Report state-path diagnostics for the fixed product encoders."""

from __future__ import annotations

import json
from math import pi
from pathlib import Path

import numpy as np

from iqp_cond.encoding import ENCODINGS, post_hadamard_amplitudes
from iqp_cond.interpolation import (
    HELDOUT_ANGLE_FRACTIONS,
    TRAIN_ANGLE_FRACTIONS,
)

N_QUBITS = 19


def _binary_entropy(probability: float) -> float:
    terms = [
        -value * np.log2(value)
        for value in (probability, 1.0 - probability)
        if value > 0.0
    ]
    return float(sum(terms))


def _geometry(encoding: str) -> dict[str, object]:
    fractions = np.sort(
        np.concatenate([TRAIN_ANGLE_FRACTIONS, HELDOUT_ANGLE_FRACTIONS]),
    )
    angles = pi * fractions
    zero, one = post_hadamard_amplitudes(angles, encoding)
    states = np.stack([np.asarray(zero), np.asarray(one)], axis=1)

    local_fidelities = np.abs(
        np.sum(np.conjugate(states[:-1]) * states[1:], axis=1),
    ) ** 2
    local_distances = np.arccos(np.sqrt(np.clip(local_fidelities, 0.0, 1.0)))
    global_fidelities = local_fidelities**N_QUBITS
    probabilities_one = np.abs(np.asarray(one)) ** 2
    entropies = N_QUBITS * np.asarray(
        [_binary_entropy(float(value)) for value in probabilities_one],
    )
    max_single_flip_ratio = np.maximum(
        np.abs(np.asarray(one) / np.asarray(zero)),
        np.abs(np.asarray(zero) / np.asarray(one)),
    )

    chord_distances = []
    excess_lengths = []
    for index in range(1, len(states) - 1):
        chord_fidelity = np.abs(np.vdot(states[index - 1], states[index + 1])) ** 2
        chord_distance = np.arccos(np.sqrt(np.clip(chord_fidelity, 0.0, 1.0)))
        chord_distances.append(float(chord_distance))
        excess_lengths.append(
            float(
                local_distances[index - 1]
                + local_distances[index]
                - chord_distance,
            ),
        )

    endpoint_fidelity = float(np.abs(np.vdot(states[0], states[-1])) ** 2)
    return {
        "angle_fractions_pi": fractions.tolist(),
        "local_endpoint_fidelity": endpoint_fidelity,
        "global_endpoint_fidelity": endpoint_fidelity**N_QUBITS,
        "mean_adjacent_local_fubini_study_distance": float(
            local_distances.mean(),
        ),
        "mean_adjacent_global_fidelity": float(global_fidelities.mean()),
        "mean_path_excess": float(np.mean(excess_lengths)),
        "max_path_excess": float(np.max(excess_lengths)),
        "reference_entropy_bits": entropies.tolist(),
        "max_single_flip_amplitude_ratio": max_single_flip_ratio.tolist(),
        "mean_reference_entropy_bits": float(entropies.mean()),
        "max_single_flip_ratio_over_path": float(max_single_flip_ratio.max()),
        "chord_distances": chord_distances,
    }


def main() -> None:
    """Write machine-readable and compact Markdown geometry diagnostics."""
    output = Path("results/encoding/geometry")
    output.mkdir(parents=True, exist_ok=True)
    payload = {encoding: _geometry(encoding) for encoding in ENCODINGS}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Product-encoding geometry",
        "",
        "Diagnostics use the nine interleaved interpolation angles and 19 "
        "broadcast qubits.",
        "",
        "| Encoding | Local endpoint fidelity | Global endpoint fidelity | "
        "Mean adjacent global fidelity | Mean path excess | Mean reference "
        "entropy (bits) | Max one-flip ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for encoding, metrics in payload.items():
        lines.append(
            f"| {encoding} | {metrics['local_endpoint_fidelity']:.6f} | "
            f"{metrics['global_endpoint_fidelity']:.6f} | "
            f"{metrics['mean_adjacent_global_fidelity']:.6f} | "
            f"{metrics['mean_path_excess']:.6f} | "
            f"{metrics['mean_reference_entropy_bits']:.3f} | "
            f"{metrics['max_single_flip_ratio_over_path']:.3f} |",
        )
    lines.extend(
        [
            "",
            "Path excess is the two-step Fubini--Study length minus its chord; "
            "zero indicates a locally geodesic single-qubit path.",
            "",
        ],
    )
    (output / "metrics.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
