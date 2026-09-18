"""Run configuration: budgets, seeds, and content addressing.

A run is fully described by its :class:`RunConfig`; the canonical JSON digest
of the config identifies the run directory and the tracked metadata, so a
result can always be traced back to the exact settings and library commit that
produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import iqp_cond as ciqp

__all__ = ["RunConfig"]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class RunConfig:
    """Training and data budgets of one experiment run."""

    study: str
    encoder: str = "broadcast"
    n_visible: int = 12
    n_ancillas: int = 0
    width_factor: float = 0.5
    n_train: int = 1000
    n_validation: int = 1000
    n_test: int = 1000
    steps: int = 1000
    batch_size: int = 100
    n_ops: int = 2000
    mc_samples: int = 8192
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-3
    validation_interval: int = 25
    mask_refresh: str = ciqp.STEP
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.n_visible < 2:
            raise ValueError("n_visible must be at least two")
        if self.n_ancillas < 0:
            raise ValueError("n_ancillas must be nonnegative")
        if self.n_train < 2 or self.n_validation < 2 or self.n_test < 2:
            raise ValueError("every data split needs at least two samples")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        # Validate the training budgets through the library config.
        self.training_config()

    @property
    def n_total(self) -> int:
        """Joint register size."""
        return self.n_visible + self.n_ancillas

    def training_config(self) -> ciqp.TrainingConfig:
        """Return the library training configuration for this run."""
        return ciqp.TrainingConfig(
            steps=self.steps,
            batch_size=self.batch_size,
            n_ops=self.n_ops,
            mc_samples=self.mc_samples,
            learning_rate=self.learning_rate,
            final_learning_rate=self.final_learning_rate,
            validation_interval=self.validation_interval,
            mask_refresh=self.mask_refresh,
        )

    def as_dict(self) -> dict:
        """Return the config as a JSON-serialisable dictionary."""
        return asdict(self)

    def digest(self) -> str:
        """Return the canonical content digest of the config."""
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()
