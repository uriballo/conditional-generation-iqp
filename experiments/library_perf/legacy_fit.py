"""Same 12-qubit Gaussian fit through the pre-extraction research implementation.

The old campaign trained with a bank drawn once per fit (``cache`` in the new
library) through ``exact_tvd_runner._weighted_loss_functions`` and
``exact_tvd_training.train_campaign``.  This script calls those exact code paths
on the archived dataset with the same budget, so the new library can be compared
against the implementation the published experiments used.

Run with the research tree importable and *not* the library package, e.g. from
the repo root:

    PYTHONPATH=src .venv/bin/python experiments/library_perf/legacy_fit.py \
        --steps 2000 --n-ops 1000 --samples 500 --label legacy-gaussian-n12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.encoding import ANGLE_ENCODING
from iqp_cond.exact_tvd_campaign import data_initialize
from iqp_cond.exact_tvd_runner import _weighted_loss_functions
from iqp_cond.exact_tvd_training import CampaignTrainingConfig, train_campaign
from iqp_cond.gaussian_data import condition_angle_matrix
from iqp_cond.initialization import initialize_parameters_from_data

TRAINING_CONDITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
DATA_CANDIDATES = (
    "results/exact-tvd-ancilla-v1/data/gaussian-n12.npz",
    "data/gaussian-n12.npz",
)


def find_data(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for candidate in DATA_CANDIDATES:
        path = REPO / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"gaussian-n12 dataset not found under {REPO}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-ops", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ancillas", type=int, default=0)
    parser.add_argument("--label", default="legacy-gaussian-n12")
    args = parser.parse_args()

    with np.load(find_data(args.data)) as archive:
        data = {name: archive[name] for name in archive.files}
    conditions = np.asarray(data["conditions"], dtype=np.float64)
    n_visible = int(data["visible_qubits"])
    n_model = n_visible + args.ancillas
    selected = np.asarray(
        [int(np.argmin(np.abs(conditions - value))) for value in TRAINING_CONDITIONS],
    )
    held_out = np.asarray(
        [index for index in range(conditions.size) if index not in set(selected)],
    )
    training = np.asarray(data["train"], dtype=np.float64)[selected]
    validation = np.asarray(data["validation"], dtype=np.float64)[selected]
    held_out_targets = np.asarray(data["test"], dtype=np.float64)[held_out]

    ansatz = complete_ansatz(n_model)
    generators = np.asarray(ansatz.generators)
    angles = condition_angle_matrix(conditions[selected], n_model)
    held_out_angles = condition_angle_matrix(conditions[held_out], n_model)
    ancilla_support = np.any(generators[:, n_visible:] != 0, axis=1)
    if args.ancillas:
        initial_params = np.asarray(
            data_initialize(training[0], generators, ancilla_support, args.seed),
        )
    else:
        initial_params = np.asarray(
            initialize_parameters_from_data(training[0], generators),
        )

    training_loss, validation_mmd, _banks = _weighted_loss_functions(
        generators=generators,
        angles=angles,
        validation_targets=validation,
        sigmas=(1.0, 2.0, 4.0),
        n_ops=args.n_ops,
        n_samples=args.samples,
        seed=args.seed,
        encoding=ANGLE_ENCODING,
    )
    _held_out_loss, held_out_mmd, _ = _weighted_loss_functions(
        generators=generators,
        angles=held_out_angles,
        validation_targets=held_out_targets,
        sigmas=(1.0, 2.0, 4.0),
        n_ops=args.n_ops,
        n_samples=args.samples,
        seed=args.seed,
        encoding=ANGLE_ENCODING,
    )

    config = CampaignTrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        n_samples=args.samples,
        validation_interval=25,
        optimizer="adam_cosine",
        scheduled_checkpoints=(),
    )
    start = time.perf_counter()
    result = train_campaign(
        initial_params,
        training,
        validation_mmd,
        training_loss,
        seed=args.seed,
        config=config,
    )
    wall = time.perf_counter() - start
    held_out_score = float(held_out_mmd(jnp.asarray(result.best_params)))

    payload = {
        "label": args.label,
        "implementation": "research src/iqp_cond",
        "data": str(find_data(args.data)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "n_ops": args.n_ops,
        "n_iqp_samples": args.samples,
        "sigmas": [1.0, 2.0, 4.0],
        "mask_refresh": "retained (campaign)",
        "seed": args.seed,
        "ancillas": args.ancillas,
        "n_parameters": ansatz.n_parameters,
        "training_conditions": [float(value) for value in conditions[selected]],
        "held_out_conditions": [float(value) for value in conditions[held_out]],
        "wall_s": wall,
        "per_step_ms": 1e3 * wall / args.steps,
        "best_step": int(result.best_step),
        "best_validation_loss": float(result.best_validation_mmd),
        "final_train_loss": float(np.mean(result.train_losses[-20:])),
        "held_out_mmd": held_out_score,
    }
    for name, value in payload.items():
        print(f"{name}: {value}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.label}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[wrote {out}]")


if __name__ == "__main__":
    main()
