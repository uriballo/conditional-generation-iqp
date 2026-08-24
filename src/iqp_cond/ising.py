"""Frozen periodic random-coupling Ising targets and MCMC certification.

The coupling construction reproduces the public PennyLane Ising dataset
generator: a periodic square lattice, legacy NumPy RNG, independent directed
``U[0, 2]`` edge weights, and an explicit symmetrisation step.  Exact
enumeration supplies scientific truth; MCMC samples are used only for the
finite Training and Validation datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import pairwise
from typing import Literal

import networkx as nx
import numpy as np

from .exact import enumerate_bitstrings, total_variation_distance

CANONICAL_COUPLING_SEED = 666
CANONICAL_BETA = 1.0 / 3.0
DEFAULT_LATTICE_SHAPE = (4, 4)


@dataclass(frozen=True)
class BetaGrid:
    """Five Training betas and their four adjacent Held-out midpoints."""

    training: tuple[float, ...]
    heldout: tuple[float, ...]
    anchor: float = CANONICAL_BETA
    provenance: str = "target_only"

    def __post_init__(self) -> None:
        if len(self.training) != 5 or len(self.heldout) != 4:
            raise ValueError(
                "the Ising grid needs five Training and four Held-out betas",
            )
        if tuple(sorted(self.training)) != self.training:
            raise ValueError("Training betas must be strictly ordered")
        if len(set(self.training)) != len(self.training):
            raise ValueError("Training betas must be unique")
        if not any(np.isclose(beta, self.anchor) for beta in self.training):
            raise ValueError("Training betas must include the beta=1/3 anchor")
        expected = tuple(
            (left + right) / 2.0
            for left, right in zip(self.training[:-1], self.training[1:], strict=True)
        )
        if not np.allclose(self.heldout, expected, atol=1e-14, rtol=0.0):
            raise ValueError("Held-out betas must be adjacent Training midpoints")

    @classmethod
    def from_interval(cls, beta_min: float, beta_max: float) -> BetaGrid:
        """Construct a five-point grid that keeps beta=1/3 as its centre point."""
        if not beta_min < CANONICAL_BETA < beta_max:
            raise ValueError("the calibration interval must contain beta=1/3")
        training = (
            float(beta_min),
            float((beta_min + CANONICAL_BETA) / 2.0),
            CANONICAL_BETA,
            float((CANONICAL_BETA + beta_max) / 2.0),
            float(beta_max),
        )
        heldout = tuple(
            (left + right) / 2.0
            for left, right in pairwise(training)
        )
        return cls(training=training, heldout=heldout)


@dataclass(frozen=True)
class ExactIsingTarget:
    """An exactly enumerated zero-field Ising target in little-endian order."""

    coupling_seed: int
    beta: float
    couplings: np.ndarray
    bitstrings: np.ndarray
    energies: np.ndarray
    probabilities: np.ndarray

    @property
    def temperature(self) -> float:
        return 1.0 / self.beta

    @property
    def digest(self) -> str:
        payload = b"".join(
            (
                np.asarray(self.coupling_seed, dtype=np.int64).tobytes(),
                np.asarray(self.beta, dtype=np.float64).tobytes(),
                np.ascontiguousarray(self.couplings, dtype=np.float64).tobytes(),
                np.ascontiguousarray(self.probabilities, dtype=np.float64).tobytes(),
            ),
        )
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class TargetDiagnostics:
    """Exact target-only diagnostics used by beta calibration."""

    coupling_seed: int
    beta: float
    normalized_entropy: float
    max_probability: float
    mean_energy: float
    energy_std: float
    mean_magnetization: float
    mean_absolute_magnetization: float


@dataclass(frozen=True)
class CalibrationThresholds:
    """Frozen eligibility limits for target-only interval calibration."""

    min_adjacent_tvd: float = 0.015
    max_adjacent_tvd: float = 0.40
    min_normalized_entropy: float = 0.18
    max_endpoint_probability: float = 0.40


@dataclass(frozen=True)
class CalibrationResult:
    """Selected grid and auditable diagnostics for every screened interval."""

    grid: BetaGrid
    coupling_seeds: tuple[int, ...]
    selected_interval: tuple[float, float]
    candidate_records: tuple[dict[str, object], ...]
    rule: str = "widest eligible interval; ties by larger minimum adjacent TVD"


@dataclass(frozen=True)
class SamplerThresholds:
    """Frozen, scale-normalised MCMC quality thresholds."""

    min_ess_per_chain: float = 100.0
    max_chain_energy_z: float = 0.30
    max_energy_mean_z: float = 0.10
    max_magnetization_error: float = 0.08
    max_correlation_rmse: float = 0.08
    max_character_rmse: float = 0.08


@dataclass(frozen=True)
class SamplerAudit:
    """Exact-reference certification for one collection of independent chains."""

    coupling_seed: int
    beta: float
    n_chains: int
    samples_per_chain: int
    min_energy_ess: float
    max_chain_energy_z: float
    energy_mean_z: float
    magnetization_error: float
    correlation_rmse: float
    character_rmse: float
    empirical_bitstring_tvd: float
    passed: bool
    failed_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IsingDataset:
    """Role-tagged finite data with a content-addressed identity."""

    role: Literal["training", "validation", "certification"]
    coupling_seed: int
    beta: float
    data_seed: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 2 or samples.shape[1] != 16:
            raise ValueError("Ising datasets must have shape (samples, 16)")
        if not np.all((samples == 0) | (samples == 1)):
            raise ValueError("Ising datasets must be binary")

    @property
    def digest(self) -> str:
        metadata = (
            f"{self.role}|{self.coupling_seed}|{self.beta:.17g}|{self.data_seed}|"
            f"{self.samples.shape}"
        ).encode()
        content = metadata + np.ascontiguousarray(self.samples).tobytes()
        return sha256(content).hexdigest()


def periodic_lattice_couplings(
    seed: int,
    shape: tuple[int, int] = DEFAULT_LATTICE_SHAPE,
) -> np.ndarray:
    """Reproduce PennyLane's periodic random-positive coupling prescription."""
    height, width = shape
    if height < 2 or width < 2:
        raise ValueError("periodic lattice dimensions must both be at least two")
    graph = nx.grid_2d_graph(height, width, periodic=True)
    adjacency = nx.to_numpy_array(graph, nodelist=list(graph.nodes()), dtype=float)
    # ``RandomState`` and ``rand`` are intentional: the public generator uses
    # NumPy's legacy stream, so ``default_rng`` would change seed-666 couplings.
    rng = np.random.RandomState(seed)
    directed = adjacency * rng.rand(*adjacency.shape) * 2.0
    couplings = (directed + directed.T) / 2.0
    np.fill_diagonal(couplings, 0.0)
    return couplings


def ising_energies(bitstrings: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    """Return ``-1/2 s^T J s`` for binary strings mapped by ``s=2x-1``."""
    bitstrings = np.asarray(bitstrings)
    couplings = np.asarray(couplings, dtype=np.float64)
    if bitstrings.ndim != 2 or couplings.shape != (bitstrings.shape[1],) * 2:
        raise ValueError("bitstrings and coupling matrix must align")
    if not np.allclose(couplings, couplings.T):
        raise ValueError("couplings must be symmetric")
    spins = 2.0 * bitstrings - 1.0
    return -0.5 * np.einsum("bi,ij,bj->b", spins, couplings, spins)


def exact_ising_target(
    coupling_seed: int,
    beta: float,
    shape: tuple[int, int] = DEFAULT_LATTICE_SHAPE,
) -> ExactIsingTarget:
    """Enumerate and normalise one periodic random-coupling Ising target."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    n_qubits = shape[0] * shape[1]
    bitstrings = enumerate_bitstrings(n_qubits)
    couplings = periodic_lattice_couplings(coupling_seed, shape)
    energies = ising_energies(bitstrings, couplings)
    log_weights = -float(beta) * energies
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    probabilities = weights / weights.sum()
    return ExactIsingTarget(
        coupling_seed=coupling_seed,
        beta=float(beta),
        couplings=couplings,
        bitstrings=bitstrings,
        energies=energies,
        probabilities=probabilities,
    )


def target_diagnostics(target: ExactIsingTarget) -> TargetDiagnostics:
    """Calculate exact thermodynamic diagnostics without model information."""
    probabilities = target.probabilities
    energies = target.energies
    spins = 2.0 * target.bitstrings - 1.0
    magnetization = spins.mean(axis=1)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
    mean_energy = float(probabilities @ energies)
    energy_std = float(
        np.sqrt(probabilities @ ((energies - mean_energy) ** 2)),
    )
    return TargetDiagnostics(
        coupling_seed=target.coupling_seed,
        beta=target.beta,
        normalized_entropy=float(entropy / np.log(probabilities.size)),
        max_probability=float(probabilities.max()),
        mean_energy=mean_energy,
        energy_std=energy_std,
        mean_magnetization=float(probabilities @ magnetization),
        mean_absolute_magnetization=float(probabilities @ np.abs(magnetization)),
    )


def beta_to_broadcast_angles(
    beta: float | np.ndarray,
    grid: BetaGrid,
    n_qubits: int = 16,
) -> np.ndarray:
    """Map beta affinely to the frozen broadcast ``RY`` range ``[0, pi/4]``."""
    values = np.asarray(beta, dtype=np.float64)
    beta_min, beta_max = grid.training[0], grid.training[-1]
    if np.any(values < beta_min - 1e-14) or np.any(values > beta_max + 1e-14):
        raise ValueError("beta lies outside the frozen calibration interval")
    alpha = (np.pi / 4.0) * (values - beta_min) / (beta_max - beta_min)
    return np.broadcast_to(alpha[..., None], (*alpha.shape, n_qubits)).copy()


def calibrate_beta_grid(
    coupling_seeds: tuple[int, ...],
    candidate_intervals: tuple[tuple[float, float], ...],
    thresholds: CalibrationThresholds | None = None,
) -> CalibrationResult:
    """Select a shared interval using only exact Target properties.

    The widest eligible candidate wins.  Eligibility requires distinguishable
    adjacent target laws on every instance while avoiding an already collapsed
    endpoint.  No argument can carry model, checkpoint, or Held-out metrics.
    """
    if len(set(coupling_seeds)) != len(coupling_seeds):
        raise ValueError("coupling seeds must be unique")
    if CANONICAL_COUPLING_SEED not in coupling_seeds:
        raise ValueError("coupling seeds must include canonical seed 666")
    if not candidate_intervals:
        raise ValueError("at least one candidate interval is required")
    if thresholds is None:
        thresholds = CalibrationThresholds()

    records: list[dict[str, object]] = []
    eligible: list[tuple[float, float, float, BetaGrid]] = []
    for beta_min, beta_max in candidate_intervals:
        grid = BetaGrid.from_interval(beta_min, beta_max)
        instance_adjacent_tvds: list[float] = []
        entropies: list[float] = []
        maxima: list[float] = []
        for seed in coupling_seeds:
            targets = [exact_ising_target(seed, beta) for beta in grid.training]
            diagnostics = [target_diagnostics(target) for target in targets]
            entropies.extend(item.normalized_entropy for item in diagnostics)
            maxima.extend(item.max_probability for item in diagnostics)
            instance_adjacent_tvds.extend(
                total_variation_distance(left.probabilities, right.probabilities)
                for left, right in pairwise(targets)
            )
        min_tvd = float(min(instance_adjacent_tvds))
        max_tvd = float(max(instance_adjacent_tvds))
        min_entropy = float(min(entropies))
        max_probability = float(max(maxima))
        passed = (
            min_tvd >= thresholds.min_adjacent_tvd
            and max_tvd <= thresholds.max_adjacent_tvd
            and min_entropy >= thresholds.min_normalized_entropy
            and max_probability <= thresholds.max_endpoint_probability
        )
        records.append(
            {
                "interval": [beta_min, beta_max],
                "training_betas": list(grid.training),
                "heldout_betas": list(grid.heldout),
                "min_adjacent_tvd": min_tvd,
                "median_adjacent_tvd": float(np.median(instance_adjacent_tvds)),
                "max_adjacent_tvd": max_tvd,
                "min_normalized_entropy": min_entropy,
                "max_probability": max_probability,
                "passed": passed,
            },
        )
        if passed:
            eligible.append((beta_max - beta_min, min_tvd, -beta_min, grid))

    if not eligible:
        raise RuntimeError("no candidate beta interval passed target-only calibration")
    selected = max(eligible, key=lambda item: item[:3])[3]
    return CalibrationResult(
        grid=selected,
        coupling_seeds=coupling_seeds,
        selected_interval=(selected.training[0], selected.training[-1]),
        candidate_records=tuple(records),
    )


def metropolis_chains(
    couplings: np.ndarray,
    beta: float,
    *,
    n_chains: int,
    samples_per_chain: int,
    burn_in: int,
    thinning: int,
    seed: int,
    symmetry_mix: bool = True,
) -> np.ndarray:
    """Run independent PennyLane-style single-spin Metropolis sweep chains.

    The released targets have zero local fields and therefore an exact global
    spin-flip symmetry.  By default each retained state receives an independent
    global-flip proposal, which is always accepted and prevents finite chains
    from remaining trapped in only one of the two ferromagnetic sectors.  This
    extra transition preserves the exact target and is recorded by callers.
    """
    couplings = np.asarray(couplings, dtype=np.float64)
    if couplings.ndim != 2 or couplings.shape[0] != couplings.shape[1]:
        raise ValueError("couplings must be square")
    if beta <= 0.0 or min(n_chains, samples_per_chain, thinning) < 1 or burn_in < 0:
        raise ValueError("invalid MCMC configuration")
    rng = np.random.default_rng(seed)
    n_spins = couplings.shape[0]
    spins = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(n_chains, n_spins))
    output = np.empty((n_chains, samples_per_chain, n_spins), dtype=np.int8)
    chain_indices = np.arange(n_chains)

    def sweep() -> None:
        for _ in range(n_spins):
            flip = rng.integers(0, n_spins, size=n_chains)
            local_field = np.sum(couplings[flip] * spins, axis=1)
            delta_energy = 2.0 * spins[chain_indices, flip] * local_field
            accept_probability = np.minimum(1.0, np.exp(-beta * delta_energy))
            accept = rng.random(n_chains) < accept_probability
            accepted_chains = chain_indices[accept]
            spins[accepted_chains, flip[accept]] *= -1

    for _ in range(burn_in):
        sweep()
    for sample_index in range(samples_per_chain):
        for _ in range(thinning):
            sweep()
        if symmetry_mix:
            global_flip = rng.random(n_chains) < 0.5
            spins[global_flip] *= -1
        output[:, sample_index] = ((spins + 1) // 2).astype(np.int8)
    return output


def _effective_sample_size(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean()
    variance = float(centered @ centered / len(values))
    if variance <= np.finfo(float).eps:
        return 0.0
    max_lag = min(len(values) // 2, 1_000)
    autocorrelations: list[float] = []
    for lag in range(1, max_lag + 1):
        correlation = float(centered[:-lag] @ centered[lag:])
        correlation /= (len(values) - lag) * variance
        autocorrelations.append(correlation)
        if lag % 2 == 0 and sum(autocorrelations[-2:]) <= 0.0:
            autocorrelations = autocorrelations[:-2]
            break
    tau = max(1.0, 1.0 + 2.0 * sum(autocorrelations))
    return float(len(values) / tau)


def audit_sampler(
    chains: np.ndarray,
    target: ExactIsingTarget,
    thresholds: SamplerThresholds | None = None,
) -> SamplerAudit:
    """Certify chain mixing, moments, correlations, and distributional error."""
    chains = np.asarray(chains, dtype=np.int8)
    if chains.ndim != 3 or chains.shape[2] != target.bitstrings.shape[1]:
        raise ValueError("chains must have shape (chains, samples, qubits)")
    if chains.shape[0] < 2:
        raise ValueError("sampler certification requires independent chains")
    if not np.all((chains == 0) | (chains == 1)):
        raise ValueError("chains must be binary")
    if thresholds is None:
        thresholds = SamplerThresholds()

    exact_probabilities = target.probabilities
    exact_spins = 2.0 * target.bitstrings - 1.0
    exact_energy_mean = float(exact_probabilities @ target.energies)
    exact_energy_std = float(
        np.sqrt(exact_probabilities @ ((target.energies - exact_energy_mean) ** 2)),
    )
    exact_magnetization = exact_probabilities @ exact_spins.mean(axis=1)
    exact_correlations = np.einsum(
        "b,bi,bj->ij",
        exact_probabilities,
        exact_spins,
        exact_spins,
    )

    flat = chains.reshape(-1, chains.shape[-1])
    energies_by_chain = np.stack(
        [ising_energies(chain, target.couplings) for chain in chains],
    )
    energy_ess = tuple(_effective_sample_size(values) for values in energies_by_chain)
    scale = max(exact_energy_std, np.finfo(float).eps)
    chain_energy_z = np.abs(energies_by_chain.mean(axis=1) - exact_energy_mean) / scale
    empirical_energy_mean = float(energies_by_chain.mean())
    energy_mean_z = abs(empirical_energy_mean - exact_energy_mean) / scale

    empirical_spins = 2.0 * flat - 1.0
    magnetization_error = abs(float(empirical_spins.mean()) - exact_magnetization)
    empirical_correlations = np.einsum(
        "bi,bj->ij",
        empirical_spins,
        empirical_spins,
    ) / len(empirical_spins)
    off_diagonal = ~np.eye(empirical_correlations.shape[0], dtype=bool)
    correlation_errors = (
        empirical_correlations[off_diagonal] - exact_correlations[off_diagonal]
    )
    correlation_rmse = float(
        np.sqrt(np.mean(correlation_errors**2)),
    )

    one_and_two_masks = [np.eye(flat.shape[1], dtype=np.int8)]
    pair_masks = []
    for left in range(flat.shape[1]):
        for right in range(left + 1, flat.shape[1]):
            mask = np.zeros(flat.shape[1], dtype=np.int8)
            mask[[left, right]] = 1
            pair_masks.append(mask)
    masks = np.concatenate([one_and_two_masks[0], np.asarray(pair_masks)], axis=0)
    empirical_characters = 1.0 - 2.0 * ((flat @ masks.T) % 2)
    exact_characters = 1.0 - 2.0 * ((target.bitstrings @ masks.T) % 2)
    character_rmse = float(
        np.sqrt(
            np.mean(
                (
                    empirical_characters.mean(axis=0)
                    - exact_probabilities @ exact_characters
                )
                ** 2,
            ),
        ),
    )

    powers = 1 << np.arange(flat.shape[1], dtype=np.int64)
    counts = np.bincount(flat @ powers, minlength=2 ** flat.shape[1])
    empirical_probabilities = counts / counts.sum()
    empirical_tvd = total_variation_distance(
        exact_probabilities,
        empirical_probabilities,
    )

    checks = {
        "effective_sample_size": min(energy_ess) >= thresholds.min_ess_per_chain,
        "chain_agreement": float(chain_energy_z.max()) <= thresholds.max_chain_energy_z,
        "energy_moment": energy_mean_z <= thresholds.max_energy_mean_z,
        "magnetization": magnetization_error <= thresholds.max_magnetization_error,
        "correlations": correlation_rmse <= thresholds.max_correlation_rmse,
        "distributional_characters": character_rmse <= thresholds.max_character_rmse,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return SamplerAudit(
        coupling_seed=target.coupling_seed,
        beta=target.beta,
        n_chains=chains.shape[0],
        samples_per_chain=chains.shape[1],
        min_energy_ess=float(min(energy_ess)),
        max_chain_energy_z=float(chain_energy_z.max()),
        energy_mean_z=float(energy_mean_z),
        magnetization_error=float(magnetization_error),
        correlation_rmse=correlation_rmse,
        character_rmse=character_rmse,
        empirical_bitstring_tvd=empirical_tvd,
        passed=not failed,
        failed_checks=failed,
    )


def make_finite_datasets(
    target: ExactIsingTarget,
    *,
    training_seed: int,
    validation_seed: int,
    n_chains: int = 8,
    burn_in: int = 4_000,
    thinning: int = 32,
) -> tuple[IsingDataset, IsingDataset, SamplerAudit, SamplerAudit]:
    """Generate disjoint 5,000/2,000 data from independent chain collections."""
    training_per_chain = int(np.ceil(5_000 / n_chains))
    validation_per_chain = int(np.ceil(2_000 / n_chains))
    training_chains = metropolis_chains(
        target.couplings,
        target.beta,
        n_chains=n_chains,
        samples_per_chain=training_per_chain,
        burn_in=burn_in,
        thinning=thinning,
        seed=training_seed,
    )
    validation_chains = metropolis_chains(
        target.couplings,
        target.beta,
        n_chains=n_chains,
        samples_per_chain=validation_per_chain,
        burn_in=burn_in,
        thinning=thinning,
        seed=validation_seed,
    )
    training = IsingDataset(
        role="training",
        coupling_seed=target.coupling_seed,
        beta=target.beta,
        data_seed=training_seed,
        samples=training_chains.reshape(-1, 16)[:5_000],
    )
    validation = IsingDataset(
        role="validation",
        coupling_seed=target.coupling_seed,
        beta=target.beta,
        data_seed=validation_seed,
        samples=validation_chains.reshape(-1, 16)[:2_000],
    )
    return (
        training,
        validation,
        audit_sampler(training_chains, target),
        audit_sampler(validation_chains, target),
    )
