"""Preregistered estimator calibration decision for the foundations study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .ansatz import complete_ansatz
from .foundations_protocol import FoundationsManifest
from .ising import BetaGrid as IsingBetaGrid
from .ising import beta_to_broadcast_angles
from .mmd import conditional_mmd, exact_conditional_mmd, exact_target_fouriers
from .protocol import content_digest
from .training import initialize_parameters


class EstimatorSelectionError(RuntimeError):
    """Raised when the foundations estimator calibration has no valid budget."""


@dataclass(frozen=True, slots=True)
class EstimatorAuditRow:
    """All frozen n=8 and n=16 diagnostics for one equal-budget candidate."""

    operator_budget: int
    reference_budget: int
    n8_absolute_loss_bias: float
    n8_mean_gradient_cosine: float
    n8_normalized_gradient_rmse: float
    n16_repeated_loss_cv: float
    n16_median_pairwise_gradient_cosine: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.operator_budget, bool)
            or not isinstance(self.operator_budget, int)
            or isinstance(self.reference_budget, bool)
            or not isinstance(self.reference_budget, int)
            or self.operator_budget < 1
            or self.reference_budget < 1
        ):
            raise ValueError("estimator budgets must be positive integers")
        metrics = (
            self.n8_absolute_loss_bias,
            self.n8_mean_gradient_cosine,
            self.n8_normalized_gradient_rmse,
            self.n16_repeated_loss_cv,
            self.n16_median_pairwise_gradient_cosine,
        )
        if any(not math.isfinite(value) for value in metrics):
            raise ValueError("estimator audit metrics must be finite")
        if (
            self.n8_absolute_loss_bias < 0.0
            or self.n8_normalized_gradient_rmse < 0.0
            or self.n16_repeated_loss_cv < 0.0
            or not -1.0 <= self.n8_mean_gradient_cosine <= 1.0
            or not -1.0 <= self.n16_median_pairwise_gradient_cosine <= 1.0
        ):
            raise ValueError("estimator audit metrics are outside their domains")

    def passes(self, manifest: FoundationsManifest) -> bool:
        protocol = manifest.estimator
        return (
            self.operator_budget == self.reference_budget
            and self.n8_absolute_loss_bias <= protocol.exact_loss_bias_max
            and self.n8_mean_gradient_cosine >= protocol.exact_gradient_cosine_min
            and self.n8_normalized_gradient_rmse <= protocol.exact_gradient_nrmse_max
            and self.n16_repeated_loss_cv <= protocol.large_loss_cv_max
            and self.n16_median_pairwise_gradient_cosine
            >= protocol.large_gradient_cosine_min
        )

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "operator_budget": self.operator_budget,
            "reference_budget": self.reference_budget,
            "n8_absolute_loss_bias": self.n8_absolute_loss_bias,
            "n8_mean_gradient_cosine": self.n8_mean_gradient_cosine,
            "n8_normalized_gradient_rmse": self.n8_normalized_gradient_rmse,
            "n16_repeated_loss_cv": self.n16_repeated_loss_cv,
            "n16_median_pairwise_gradient_cosine": (
                self.n16_median_pairwise_gradient_cosine
            ),
        }


@dataclass(frozen=True, slots=True)
class EstimatorSelection:
    """Smallest preregistered budget passing all estimator criteria."""

    manifest_digest: str
    operator_budget: int
    reference_budget: int
    passing_budgets: tuple[int, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        for name, digest in (
            ("manifest_digest", self.manifest_digest),
            ("evidence_digest", self.evidence_digest),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            self.operator_budget != self.reference_budget
            or self.operator_budget not in self.passing_budgets
            or tuple(sorted(set(self.passing_budgets))) != self.passing_budgets
        ):
            raise ValueError("estimator selection budgets are inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_digest": self.manifest_digest,
            "operator_budget": self.operator_budget,
            "reference_budget": self.reference_budget,
            "passing_budgets": list(self.passing_budgets),
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EstimatorSelection:
        expected = {
            "manifest_digest",
            "operator_budget",
            "reference_budget",
            "passing_budgets",
            "evidence_digest",
        }
        if set(value) != expected:
            raise ValueError("estimator selection schema differs")
        return cls(
            manifest_digest=str(value["manifest_digest"]),
            operator_budget=int(value["operator_budget"]),
            reference_budget=int(value["reference_budget"]),
            passing_budgets=tuple(int(item) for item in value["passing_budgets"]),
            evidence_digest=str(value["evidence_digest"]),
        )


@dataclass(frozen=True, slots=True)
class EstimatorAudit:
    """Complete deterministic calibration table and its binding selection."""

    manifest_digest: str
    repetitions: int
    target_sha256: str
    rows: tuple[EstimatorAuditRow, ...]
    selection: EstimatorSelection

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "manifest_digest": self.manifest_digest,
            "repetitions": self.repetitions,
            "target_sha256": self.target_sha256,
            "rows": [row.to_dict() for row in self.rows],
            "selection": self.selection.to_dict(),
        }


def select_estimator_budget(
    manifest: FoundationsManifest,
    rows: tuple[EstimatorAuditRow, ...],
) -> EstimatorSelection:
    """Apply the frozen all-criteria rule without weakening any threshold."""
    budgets = tuple(row.operator_budget for row in rows)
    if budgets != manifest.estimator.candidate_budgets or len(set(budgets)) != len(
        budgets
    ):
        raise ValueError("audit must contain exactly the preregistered candidates")
    if any(row.operator_budget != row.reference_budget for row in rows):
        raise ValueError("foundations audit requires equal operator/reference budgets")
    passing = tuple(row.operator_budget for row in rows if row.passes(manifest))
    if not passing:
        raise EstimatorSelectionError(
            "no estimator candidate passes every frozen n=8 and n=16 criterion"
        )
    evidence = {
        "manifest_digest": manifest.digest,
        "thresholds": manifest.estimator.to_dict(),
        "rows": [row.to_dict() for row in rows],
    }
    selected = passing[0]
    return EstimatorSelection(
        manifest_digest=manifest.digest,
        operator_budget=selected,
        reference_budget=selected,
        passing_budgets=passing,
        evidence_digest=content_digest(evidence),
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(left, right, atol=1e-12, rtol=0.0) else 0.0
    return float(np.dot(left, right) / denominator)


def _marginal_first_qubits(probabilities: np.ndarray, n_qubits: int) -> np.ndarray:
    n_total = round(math.log2(probabilities.shape[1]))
    if 2**n_total != probabilities.shape[1] or not 0 < n_qubits <= n_total:
        raise ValueError("target probability width is not a valid qubit law")
    return probabilities.reshape(
        probabilities.shape[0], 2 ** (n_total - n_qubits), 2**n_qubits
    ).sum(axis=1)


def run_estimator_audit(
    manifest: FoundationsManifest,
    exact_target_path: Path,
    *,
    repetitions: int = 8,
) -> EstimatorAudit:
    """Measure the frozen n=8 truth and n=16 repeat-consistency criteria.

    The same Complete-2-local construction, three low/anchor/high Conditions,
    and one exact Ising instance are used at both widths.  At n=8 the exact
    loss and gradient are enumerated; at n=16 the preregistered repeat-only
    diagnostics are reported.  Every sampled candidate uses equal operator and
    model/reference-sample budgets.
    """
    if repetitions < 3:
        raise ValueError("estimator audit requires at least three repetitions")
    import hashlib

    target_sha256 = hashlib.sha256(exact_target_path.read_bytes()).hexdigest()
    with np.load(exact_target_path, allow_pickle=False) as archive:
        required = {
            "coupling_seed",
            "betas",
            "couplings",
            "bitstrings",
            "energies",
            "probabilities",
        }
        if set(archive.files) != required:
            raise ValueError("estimator target archive schema differs")
        all_probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
        archive_betas = tuple(float(value) for value in archive["betas"])
    expected_betas = manifest.beta_grid.training + manifest.beta_grid.held_out
    if not np.allclose(archive_betas, expected_betas, atol=1e-14, rtol=0.0):
        raise ValueError("estimator target archive beta grid differs")
    condition_indices = manifest.representability.beta_indices
    betas = tuple(manifest.beta_grid.training[index] for index in condition_indices)
    target16 = all_probabilities[np.asarray(condition_indices)]
    grid = IsingBetaGrid(
        training=manifest.beta_grid.training,
        heldout=manifest.beta_grid.held_out,
        provenance="arxiv_foundations_v1_estimator_audit",
    )

    fixtures: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for n_qubits in (8, 16):
        ansatz = complete_ansatz(n_qubits)
        parameters = np.asarray(initialize_parameters(ansatz.n_parameters, 811))
        angles = beta_to_broadcast_angles(np.asarray(betas), grid, n_qubits)
        targets = _marginal_first_qubits(target16, 8) if n_qubits == 8 else target16
        fixtures[n_qubits] = (
            np.asarray(ansatz.generators, dtype=np.float64),
            parameters,
            np.asarray(angles, dtype=np.float64),
            targets,
        )

    generators8, parameters8, angles8, target8 = fixtures[8]
    target_fouriers8 = exact_target_fouriers(target8)

    def exact_objective(theta: jnp.ndarray) -> jnp.ndarray:
        return exact_conditional_mmd(
            theta,
            jnp.asarray(generators8),
            jnp.asarray(angles8),
            target_fouriers8,
            manifest.estimator.bandwidths,
        ).loss

    exact_loss, exact_gradient = jax.value_and_grad(exact_objective)(
        jnp.asarray(parameters8)
    )
    exact_loss = float(exact_loss)
    exact_gradient = np.asarray(exact_gradient, dtype=np.float64)

    rows = []
    for budget in manifest.estimator.candidate_budgets:
        width_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for n_qubits in (8, 16):
            generators, parameters, angles, targets = fixtures[n_qubits]
            losses = []
            gradients = []
            for repeat in range(repetitions):
                rng = np.random.default_rng(
                    manifest.data.root_seed + 10_000 * n_qubits + 100 * budget + repeat
                )
                target_samples = np.stack(
                    [
                        rng.choice(
                            2**n_qubits,
                            size=budget,
                            replace=True,
                            p=probability,
                        )
                        for probability in targets
                    ]
                )
                bits = (
                    target_samples[..., None] >> np.arange(n_qubits, dtype=np.int64)
                ) & 1
                key = jax.random.PRNGKey(
                    manifest.data.root_seed + n_qubits + budget + repeat
                )

                generator_values = jnp.asarray(generators)
                angle_values = jnp.asarray(angles)
                target_values = jnp.asarray(bits, dtype=jnp.float64)
                audit_budget = budget
                audit_key = key

                def sampled(
                    theta: jnp.ndarray,
                    generator_values: jnp.ndarray = generator_values,
                    angle_values: jnp.ndarray = angle_values,
                    target_values: jnp.ndarray = target_values,
                    audit_budget: int = audit_budget,
                    audit_key: jax.Array = audit_key,
                ) -> jnp.ndarray:
                    return conditional_mmd(
                        theta,
                        generator_values,
                        angle_values,
                        target_values,
                        manifest.estimator.bandwidths,
                        audit_budget,
                        audit_budget,
                        audit_key,
                    ).loss

                loss, gradient = jax.value_and_grad(sampled)(jnp.asarray(parameters))
                losses.append(float(loss))
                gradients.append(np.asarray(gradient, dtype=np.float64))
            width_results[n_qubits] = (np.asarray(losses), np.asarray(gradients))

        losses8, gradients8 = width_results[8]
        losses16, gradients16 = width_results[16]
        mean_gradient8 = gradients8.mean(axis=0)
        gradient_scale = float(np.sqrt(np.mean(exact_gradient**2)))
        nrmse = float(
            np.sqrt(np.mean((mean_gradient8 - exact_gradient) ** 2))
            / max(gradient_scale, np.finfo(np.float64).eps)
        )
        pairwise = [
            _cosine(gradients16[left], gradients16[right])
            for left in range(repetitions)
            for right in range(left + 1, repetitions)
        ]
        mean16 = float(np.mean(losses16))
        loss_cv = float(
            np.std(losses16, ddof=1) / max(abs(mean16), np.finfo(np.float64).eps)
        )
        rows.append(
            EstimatorAuditRow(
                operator_budget=budget,
                reference_budget=budget,
                n8_absolute_loss_bias=abs(float(np.mean(losses8)) - exact_loss),
                n8_mean_gradient_cosine=float(
                    np.mean(
                        [_cosine(gradient, exact_gradient) for gradient in gradients8]
                    )
                ),
                n8_normalized_gradient_rmse=nrmse,
                n16_repeated_loss_cv=loss_cv,
                n16_median_pairwise_gradient_cosine=float(np.median(pairwise)),
            )
        )
    row_tuple = tuple(rows)
    selection = select_estimator_budget(manifest, row_tuple)
    return EstimatorAudit(
        manifest_digest=manifest.digest,
        repetitions=repetitions,
        target_sha256=target_sha256,
        rows=row_tuple,
        selection=selection,
    )
