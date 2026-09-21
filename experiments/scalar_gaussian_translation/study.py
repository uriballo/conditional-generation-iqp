"""Runner-compatible descriptor for the scalar Gaussian benchmark."""

from __future__ import annotations

import iqp_cond as ciqp
import numpy as np

from experiments.common.datasets import gaussian_target
from experiments.common.runner import Study

from .protocol import (
    N_VISIBLE,
    WIDTH_FACTOR,
    anchor_conditions,
    evaluation_grid,
    target_probabilities,
    training_set_label,
)
from .protocol import (
    make_encoder as construct_encoder,
)


def make_targets(conditions, n_visible, n_samples, rng):
    """Sample condition-indexed bitstrings from the shared Gaussian target."""
    if n_visible != N_VISIBLE:
        raise ValueError("the scalar benchmark fixes n_visible=8")
    values = np.asarray(conditions, dtype=np.float64).reshape(-1)
    return np.stack(
        [
            gaussian_target(n_visible, WIDTH_FACTOR, float(c))
            .sample(rng, n_samples)
            .astype(np.float64)
            for c in values
        ]
    )


def build_study(
    protocol: str = "interpolation",
    anchor_count: int = 3,
    placement: str = "even",
    layout_seed: int = 1101,
    encoder: str = "broadcast_affine",
    evaluation_split: str = "dev",
) -> Study:
    """Create a Study with selected anchors and one immutable evaluation split."""
    anchors = anchor_conditions(
        protocol,
        anchor_count,
        placement,
        layout_seed if placement == "random" else None,
    )

    def make_encoder(conditions, config):
        if config.n_visible != N_VISIBLE or config.n_ancillas != 0:
            raise ValueError(
                "the scalar benchmark requires n_visible=8 and n_ancillas=0"
            )
        return construct_encoder(conditions, encoder)

    def make_ansatz(config):
        if config.n_visible != N_VISIBLE or config.n_ancillas != 0:
            raise ValueError(
                "the scalar benchmark requires n_visible=8 and n_ancillas=0"
            )
        return ciqp.Ansatz.complete(N_VISIBLE)

    return Study(
        name="scalar_gaussian_translation",
        training_conditions=anchors,
        held_out_conditions=evaluation_grid(evaluation_split),
        make_targets=make_targets,
        target_probabilities=target_probabilities,
        make_encoder=make_encoder,
        make_ansatz=make_ansatz,
        condition_protocols={
            training_set_label(
                protocol,
                anchor_count,
                placement,
                layout_seed if placement == "random" else None,
            ): anchors
        },
        condition_dimension=1,
    )
