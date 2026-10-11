"""Frozen paper budgets and content-addressed run configuration.

The runtime supplies every experiment-specific setting. Numerical contraction
precision and reference chunking are recorded explicitly to preserve the
reported training protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import iqp_cond as ciqp
import jax.numpy as jnp

__all__ = [
    "BUDGET_MODES",
    "MODEL_KINDS",
    "SELECTION_MODES",
    "RunConfig",
]

#: Accepted ``gemm_dtype`` values: ``"float64"`` leaves the contraction in
#: float64 (the library default) and ``"float32"`` casts only the contraction.
CONTRACTION_DTYPES = ("float32", "float64")
MODEL_KINDS = (
    "conditional",
    "encoder_only",
    "independent_direct",
)
SELECTION_MODES = (
    "exact_training_tvd",
    "exact_training_profile_tvd",
    "exact_development_profile_tvd",
    "exact_development_tvd",
    "validation_mmd",
)
BUDGET_MODES = (
    "fixed_total",
    "condition_matched",
    "direct",
    "encoder_only",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class RunConfig:
    """Training and data budgets of one experiment run."""

    study: str
    encoder: str = "broadcast"
    condition_protocol: str = "default"
    model_kind: str = "conditional"
    selection_mode: str = "exact_training_tvd"
    budget_mode: str = "fixed_total"
    fit_condition: tuple[float, ...] | None = None
    n_visible: int = 12
    n_ancillas: int = 0
    width_factor: float = 0.5
    n_train: int = 1000
    n_validation: int = 1000
    n_test: int = 1000
    steps: int = 1000
    batch_size: int = 100
    n_ops: int = 2000
    bodyness_targets: tuple[float, ...] = (2.0, 4.0)
    mc_samples: int = 8192
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-3
    validation_interval: int = 25
    mask_refresh: str = "step"
    gemm_dtype: str = "float64"
    sample_chunk: int | None = None
    validation_n_ops: int = 512
    validation_mc_samples: int = 1000
    initialization: str = "random"
    init_scale: float = 0.5
    encoder_scale: float = 0.75
    validation_banks: int = 1
    validation_scales: tuple[float, ...] = ()
    shared_restart_streams: bool = False
    restarts: int = 3
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "validation_scales", tuple(self.validation_scales))
        if self.n_visible < 2:
            raise ValueError("n_visible must be at least two")
        if self.n_ancillas < 0:
            raise ValueError("n_ancillas must be nonnegative")
        if self.model_kind not in MODEL_KINDS:
            raise ValueError(
                f"model_kind must be one of {MODEL_KINDS}, got {self.model_kind!r}",
            )
        if self.selection_mode not in SELECTION_MODES:
            raise ValueError(
                f"selection_mode must be one of {SELECTION_MODES}, "
                f"got {self.selection_mode!r}",
            )
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(
                f"budget_mode must be one of {BUDGET_MODES}, got {self.budget_mode!r}",
            )
        if self.model_kind == "independent_direct" and self.fit_condition is None:
            raise ValueError("independent_direct runs require fit_condition")
        if self.fit_condition is not None and not self.fit_condition:
            raise ValueError("fit_condition cannot be empty")
        if self.n_train < 2 or self.n_validation < 2 or self.n_test < 2:
            raise ValueError("every data split needs at least two samples")
        if self.gemm_dtype not in CONTRACTION_DTYPES:
            raise ValueError(
                f"gemm_dtype must be one of {CONTRACTION_DTYPES}, "
                f"got {self.gemm_dtype!r}",
            )
        if self.sample_chunk is not None and self.sample_chunk < 1:
            raise ValueError("sample_chunk must be positive or None")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.initialization != "random":
            raise ValueError("only random initialization is supported")
        if self.init_scale <= 0.0:
            raise ValueError("init_scale must be positive")
        if not math.isfinite(self.encoder_scale) or self.encoder_scale <= 0.0:
            raise ValueError("encoder_scale must be positive and finite")
        if (
            isinstance(self.validation_banks, bool)
            or not isinstance(self.validation_banks, int)
            or self.validation_banks < 1
        ):
            raise ValueError("validation_banks must be a positive integer")
        if any(
            not math.isfinite(scale) or scale <= 0.0 for scale in self.validation_scales
        ):
            raise ValueError("validation_scales must be positive and finite")
        if self.restarts < 1:
            raise ValueError("restarts must be positive")
        if not self.bodyness_targets or any(
            not math.isfinite(target) or target <= 0.0
            for target in self.bodyness_targets
        ):
            raise ValueError("bodyness_targets must be positive and finite")
        # Validate the training budgets through the library config.
        self.training_config()

    @property
    def n_total(self) -> int:
        """Joint register size."""
        return self.n_visible + self.n_ancillas

    def training_config(self) -> ciqp.TrainingConfig:
        """Return the library training configuration for this run."""
        cap = self.n_visible / 2.0
        targets = [float(target) for target in self.bodyness_targets]
        if any(
            not math.isfinite(target) or not 0.0 < target < cap for target in targets
        ):
            raise ValueError(
                "every bodyness target must be finite and satisfy "
                "0 < expected_weight < n_visible / 2"
            )
        sigmas = tuple(
            float(
                (-2.0 * math.log(1.0 - 2.0 * target / self.n_visible)) ** -0.5,
            )
            for target in targets
        )
        return ciqp.TrainingConfig(
            steps=self.steps,
            batch_size=self.batch_size,
            n_ops=self.n_ops,
            mc_samples=self.mc_samples,
            validation_n_ops=self.validation_n_ops,
            validation_mc_samples=self.validation_mc_samples,
            gemm_dtype=jnp.float32 if self.gemm_dtype == "float32" else None,
            sample_chunk=self.sample_chunk,
            sigmas=sigmas,
            learning_rate=self.learning_rate,
            final_learning_rate=self.final_learning_rate,
            validation_interval=self.validation_interval,
            mask_refresh=self.mask_refresh,
        )

    def as_dict(self) -> dict:
        """Return the config as a JSON-serialisable dictionary."""
        result = asdict(self)
        return result

    def digest(self) -> str:
        """Return the canonical content digest of the config."""
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()
