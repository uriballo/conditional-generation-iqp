"""The paper's targets, fixed encoders and phase banks."""

from __future__ import annotations

import iqp_cond as ciqp
import numpy as np

from . import gaussian
from .ansatz import rectangular_lattice_ansatz, square_lattice_ansatz
from .engine import Study
from .phase_encoder import BroadcastPhaseEncoder
from .target import curie_weiss_target

TRAIN = (0.6, 0.8, 1.2, 1.4)
DEVELOPMENT = (0.5, 0.7, 1.3, 1.5)
REPORT = (0.4, 1.0, 1.6)
SCALING_REPORT = (0.4, 0.95, 1.0, 1.05, 1.6)


def make_encoder(conditions, width, kind):
    values = np.asarray(conditions, dtype=np.float64).ravel()
    if kind == "phase_075":
        return BroadcastPhaseEncoder.from_betas(values, width)
    if kind == "blind":
        return ciqp.AngleEncoder(np.zeros((len(values), 1, width)), axes=("Y" * width,))
    raise ValueError("unsupported paper encoder")


def make_targets(conditions, n_visible, n_samples, rng):
    return np.stack(
        [
            curie_weiss_target(n_visible, float(beta)).sample(n_samples, rng)
            for beta in np.asarray(conditions).ravel()
        ]
    ).astype(np.float64)


def target_probabilities(conditions, n_visible):
    if n_visible > 16:
        raise ValueError("exact probabilities are restricted to small paper models")
    indices = np.arange(2**n_visible, dtype=np.int64)
    states = ((indices[:, None] >> np.arange(n_visible)) & 1).astype(np.int8)
    return np.stack(
        [
            curie_weiss_target(n_visible, float(beta)).bitstring_probabilities(states)
            for beta in np.asarray(conditions).ravel()
        ]
    )


def build_study(job, config):
    kind = job["experiment"]
    if kind == "gaussian":
        return Study(
            name=config.study,
            training_conditions=np.asarray([0, 0.25, 0.5, 0.75, 1]),
            held_out_conditions=(np.arange(80) + 1 / 3) / 80,
            development_conditions=np.asarray([0.125, 0.375, 0.625, 0.875]),
            make_targets=gaussian.make_targets,
            target_probabilities=gaussian.target_probabilities,
            make_encoder=gaussian.make_encoder,
            make_ansatz=lambda c: square_lattice_ansatz(16),
            condition_protocols={"frozen5": np.asarray([0, 0.25, 0.5, 0.75, 1])},
        )
    reporting = SCALING_REPORT if kind == "scaling" else REPORT
    if kind in ("conditional", "unconditional"):
        reporting = (job["beta"],)
    elif job["stage"] in ("screen", "confirm"):
        reporting = DEVELOPMENT
    return Study(
        name=config.study,
        training_conditions=np.asarray(TRAIN),
        held_out_conditions=np.asarray(reporting),
        development_conditions=None if kind == "scaling" else np.asarray(DEVELOPMENT),
        make_targets=make_targets,
        target_probabilities=target_probabilities,
        make_encoder=lambda cond, c: make_encoder(cond, c.n_total, c.encoder),
        make_ansatz=(lambda c: rectangular_lattice_ansatz(c.n_total))
        if kind == "scaling"
        else (lambda c: ciqp.Ansatz.complete(c.n_total)),
    )
