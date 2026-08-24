"""Exact targets and diagnostics for the physical-data representability pilot."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from .exact import enumerate_bitstrings

MFC_ZENODO_RECORD = "https://doi.org/10.5281/zenodo.6768686"
MFC_DOWNLOAD = "https://zenodo.org/api/records/6768686/files/{energy}_GeV.pt/content"


@dataclass(frozen=True)
class PhysicalTarget:
    """One exactly evaluable target distribution and its physical coordinates."""

    dataset: str
    condition_name: str
    condition_value: float
    n_qubits: int
    probabilities: np.ndarray
    observables: dict[str, np.ndarray]
    provenance: str

    @property
    def digest(self) -> str:
        return sha256(np.asarray(self.probabilities, dtype="<f8").tobytes()).hexdigest()


def periodic_square_edges(side: int = 4) -> tuple[tuple[int, int], ...]:
    """Return each undirected nearest-neighbour edge once on a square torus."""
    if side < 3:
        raise ValueError("side must be at least three")
    edges = []
    for row in range(side):
        for column in range(side):
            site = side * row + column
            edges.append((site, side * row + (column + 1) % side))
            edges.append((site, side * ((row + 1) % side) + column))
    return tuple(edges)


def ferromagnetic_target(beta: float, side: int = 4) -> PhysicalTarget:
    """Periodic zero-field ferromagnetic spin law at inverse temperature beta."""
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    bits = enumerate_bitstrings(side * side)
    spins = 2.0 * bits - 1.0
    edges = periodic_square_edges(side)
    bond_sum = sum(spins[:, left] * spins[:, right] for left, right in edges)
    log_weights = beta * bond_sum
    probabilities = np.exp(log_weights - np.max(log_weights))
    probabilities /= probabilities.sum()
    magnetization = spins.mean(axis=1)
    return PhysicalTarget(
        dataset="ferromagnetic_spin",
        condition_name="inverse_temperature",
        condition_value=float(beta),
        n_qubits=side * side,
        probabilities=probabilities,
        observables={
            "energy_per_bond": -bond_sum / len(edges),
            "magnetization": magnetization,
            "absolute_magnetization": np.abs(magnetization),
        },
        provenance="exact periodic square-lattice Gibbs law; J=1, zero field",
    )


def curie_weiss_target(
    beta: float,
    *,
    n_qubits: int = 16,
    coupling: float = 1.0,
    field: float = 0.0,
) -> PhysicalTarget:
    """Exact finite-size Curie--Weiss law in the computational basis.

    The extensive Hamiltonian is ``-J M^2/(2n) - h M``, where
    ``M=sum_i s_i`` and ``s_i=2x_i-1``.  At zero field the thermodynamic
    critical inverse temperature is ``beta_c=1/J``.
    """
    values = (beta, coupling, field)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Curie--Weiss parameters must be finite")
    if beta <= 0.0 or coupling <= 0.0:
        raise ValueError("beta and coupling must be positive")
    if n_qubits < 1 or n_qubits > 20:
        raise ValueError("n_qubits must lie between 1 and 20")
    bits = enumerate_bitstrings(n_qubits)
    magnetization_sum = (2.0 * bits - 1.0).sum(axis=1)
    energy = -(coupling / (2.0 * n_qubits)) * magnetization_sum**2
    energy -= field * magnetization_sum
    log_weights = -beta * energy
    probabilities = np.exp(log_weights - np.max(log_weights))
    probabilities /= probabilities.sum()
    magnetization = magnetization_sum / n_qubits
    return PhysicalTarget(
        dataset="curie_weiss",
        condition_name="inverse_temperature",
        condition_value=float(beta),
        n_qubits=n_qubits,
        probabilities=probabilities,
        observables={
            "energy_per_spin": energy / n_qubits,
            "magnetization": magnetization,
            "absolute_magnetization": np.abs(magnetization),
            "magnetization_squared": magnetization**2,
        },
        provenance=(
            f"exact finite-size Curie--Weiss Gibbs law; J={coupling:g}, h={field:g}"
        ),
    )


def lattice_gas_target(
    chemical_potential: float,
    *,
    beta: float = 2.0,
    attraction: float = 1.0,
    side: int = 4,
) -> PhysicalTarget:
    """Grand-canonical attractive lattice gas on a periodic square lattice."""
    values = (chemical_potential, beta, attraction)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("lattice-gas parameters must be finite")
    if beta <= 0.0 or attraction <= 0.0:
        raise ValueError("beta and attraction must be positive")
    bits = enumerate_bitstrings(side * side).astype(np.float64)
    edges = periodic_square_edges(side)
    occupied_pairs = sum(bits[:, left] * bits[:, right] for left, right in edges)
    particle_number = bits.sum(axis=1)
    log_weights = beta * (
        attraction * occupied_pairs + chemical_potential * particle_number
    )
    probabilities = np.exp(log_weights - np.max(log_weights))
    probabilities /= probabilities.sum()
    return PhysicalTarget(
        dataset="attractive_lattice_gas",
        condition_name="chemical_potential",
        condition_value=float(chemical_potential),
        n_qubits=side * side,
        probabilities=probabilities,
        observables={
            "density": particle_number / (side * side),
            "occupied_pair_fraction": occupied_pairs / len(edges),
            "grand_energy_per_site": -(
                attraction * occupied_pairs + chemical_potential * particle_number
            )
            / (side * side),
        },
        provenance=(
            "exact periodic square-lattice grand-canonical law; "
            f"beta={beta:g}, attraction={attraction:g}"
        ),
    )


def _torch_float_tensor(path: Path) -> np.ndarray:
    """Read this dataset's simple PyTorch float tensor without importing torch."""
    with ZipFile(path) as archive:
        storage_names = [
            name for name in archive.namelist() if name.startswith("archive/data/")
        ]
        if len(storage_names) != 1:
            raise ValueError("expected exactly one tensor storage")
        raw = archive.read(storage_names[0])
    values = np.frombuffer(raw, dtype="<f4")
    if values.size != 100_000 * 5 * 4:
        raise ValueError("unexpected MFC tensor shape")
    return values.reshape(100_000, 5, 4)


def fetch_mfc_tensor(energy_gev: int, cache: Path) -> tuple[Path, str]:
    """Fetch one immutable Zenodo MFC tensor and return its byte digest."""
    if energy_gev not in {50, 75, 100, 125, 150, 175, 200}:
        raise ValueError("energy must be one of the seven archived conditions")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{energy_gev}_GeV.pt"
    if not destination.exists():
        with urlopen(MFC_DOWNLOAD.format(energy=energy_gev), timeout=120) as response:
            destination.write_bytes(response.read())
    digest = sha256(destination.read_bytes()).hexdigest()
    return destination, digest


def mfc_outgoing_energy_target(energy_gev: int, cache: Path) -> PhysicalTarget:
    """Sixteen-bin outgoing-muon-energy marginal from the archived MFC events."""
    path, source_digest = fetch_mfc_tensor(energy_gev, cache)
    events = _torch_float_tensor(path)
    # The event tensor stores incoming particles first and outgoing particles
    # afterwards. Row 2, column 0 is the outgoing-muon energy used here.
    scaled_energy = np.clip(events[:, 2, 0] / float(energy_gev), 0.0, 1.0)
    counts, _ = np.histogram(scaled_energy, bins=16, range=(0.0, 1.0))
    probabilities = counts.astype(np.float64) / counts.sum()
    bin_centres = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    return PhysicalTarget(
        dataset="mfc_outgoing_muon_energy",
        condition_name="incoming_muon_energy_gev",
        condition_value=float(energy_gev),
        n_qubits=4,
        probabilities=probabilities,
        observables={
            "scaled_outgoing_energy": bin_centres,
            "scaled_outgoing_energy_squared": bin_centres**2,
        },
        provenance=f"{MFC_ZENODO_RECORD}; source_sha256={source_digest}",
    )


def expectation_errors(
    target: PhysicalTarget,
    model_probabilities: np.ndarray,
) -> dict[str, float]:
    """Return model-minus-target errors for the target's physical observables."""
    model = np.asarray(model_probabilities, dtype=np.float64)
    if model.shape != target.probabilities.shape:
        raise ValueError("model and target probability vectors must align")
    return {
        name: float(np.dot(model - target.probabilities, values))
        for name, values in target.observables.items()
    }
