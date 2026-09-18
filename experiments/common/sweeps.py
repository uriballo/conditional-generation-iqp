"""Sweeps over run configurations; Optuna is planned.

Start with the standard-library grid and random sweeps, which run through
:func:`experiments.common.runner.run` and write the same run directories.  The
:func:`optuna_objective` hook marks where the planned Optuna integration goes
once ``optuna`` is added as an optional dependency: sample the swept fields,
build a :class:`RunConfig`, call the runner, and return the validation metric
to minimise.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .config import RunConfig
from .runner import Study, run_all_seeds

__all__ = [
    "grid_points",
    "optuna_objective",
    "random_points",
    "run_sweep",
]


def grid_points(axes: dict[str, Iterable[object]]) -> list[dict[str, object]]:
    """Return the Cartesian product of the swept fields."""
    names = sorted(axes)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(list(axes[name]) for name in names))
    ]


def random_points(
    axes: dict[str, Iterable[object]],
    n_points: int,
    *,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Return ``n_points`` random combinations of the swept fields."""
    if n_points < 1:
        raise ValueError("n_points must be positive")
    names = sorted(axes)
    pools = [list(axes[name]) for name in names]
    rng = np.random.default_rng(seed)
    return [
        {
            name: pool[int(rng.integers(len(pool)))]
            for name, pool in zip(names, pools, strict=True)
        }
        for _ in range(n_points)
    ]


def run_sweep(
    base: RunConfig,
    study: Study,
    points: Iterable[dict[str, object]],
    *,
    output_root: Path | None = None,
) -> list[dict[str, object]]:
    """Run every point of a sweep and return the metric records."""
    records = []
    for point in points:
        config = dataclasses.replace(base, **point)
        records.extend(run_all_seeds(config, study, output_root=output_root))
    return records


def optuna_objective(trial, study: Study) -> float:  # pragma: no cover
    """Return the validation objective of a trial (planned, not implemented).

    The integration will map ``trial.suggest_*`` calls to :class:`RunConfig`
    fields, call :func:`experiments.common.runner.run`, and return
    ``best_validation_loss`` so Optuna can minimise it.
    """
    raise NotImplementedError(
        "Optuna sweeps are planned; use grid_points/random_points for now",
    )


def _iter_configs(records: Iterator[dict[str, object]]) -> Iterator[object]:
    return iter(records)
