"""Generate deterministic Curie--Weiss supplement tables from the final report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.curie_weiss_protocol import CurieWeissCampaignManifest
from iqp_cond.curie_weiss_statistics import thermodynamic_metrics
from iqp_cond.exact import enumerate_bitstrings
from iqp_cond.gaussian_runner import exact_mmd_squared


def _write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"immutable generated table differs: {path}")
    if not path.exists():
        path.write_text(content)


def _effect(value: dict[str, object]) -> str:
    return f"{value['estimate']:+.3f} [{value['lower']:+.3f}, {value['upper']:+.3f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/curie-weiss-foundations-v1/report/campaign-report.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/generated"))
    arguments = parser.parse_args(argv)
    report = read_immutable_json(arguments.report)
    if report.get("completed_runs") != 270:
        raise RuntimeError("tables require the complete 270-fit report")
    comparisons = {
        (row["n_qubits"], row["comparison"]): row["interval"]
        for row in report["comparisons"]
    }
    manifest = CurieWeissCampaignManifest.from_json(
        Path("configs/curie_weiss_foundations_v1.json")
    )
    summary_lines = []
    names = (
        "training_no_condition_training",
        "training_no_condition_held_out",
        "adjacent_oracle_held_out",
        "adjacent_mixture_held_out",
        "unrestricted_minus_aligned_held_out",
    )
    for n_qubits in (8, 12, 16):
        values = [_effect(comparisons[(n_qubits, name)]) for name in names]
        summary_lines.append(f"{n_qubits} & " + " & ".join(values) + r" \\")
    _write_immutable(
        arguments.output / "curie_weiss_effects.tex",
        "\n".join(summary_lines) + "\n",
    )

    rows = report["response_rows"]
    response_lines = []
    for n_qubits in (8, 12, 16):
        for beta in tuple(0.5 + 0.1 * index for index in range(11)):
            selected = [
                row
                for row in rows
                if row["n_qubits"] == n_qubits and np.isclose(row["beta"], beta)
            ]
            model = {
                name: float(np.mean([row["conditioned"][name] for row in selected]))
                for name in (
                    "profile_tvd",
                    "bitstring_tvd",
                    "absolute_magnetization",
                    "susceptibility",
                    "binder_cumulant",
                    "energy_per_spin",
                    "heat_capacity_per_spin",
                )
            }
            target = exact_curie_weiss_target(n_qubits, beta)
            exact = thermodynamic_metrics(target, target.probabilities).to_dict()
            role = "T" if any(row["role"] == "training" for row in selected) else "H"
            response_lines.append(
                f"{n_qubits} & {beta:.1f} & {role} & "
                f"{model['profile_tvd']:.3f} & "
                f"{model['bitstring_tvd']:.3f} & "
                f"{model['absolute_magnetization']:.3f}/"
                f"{exact['absolute_magnetization']:.3f} & "
                f"{model['susceptibility']:.2f}/{exact['susceptibility']:.2f} & "
                f"{model['binder_cumulant']:.3f}/{exact['binder_cumulant']:.3f} & "
                f"{model['energy_per_spin']:.3f}/{exact['energy_per_spin']:.3f} & "
                f"{model['heat_capacity_per_spin']:.3f}/{exact['heat_capacity_per_spin']:.3f}"
                r" \\"
            )
    _write_immutable(
        arguments.output / "curie_weiss_response_table.tex",
        "\n".join(response_lines) + "\n",
    )

    mmd_lines = []
    for n_qubits in (8, 12, 16):
        bitstrings = enumerate_bitstrings(n_qubits)
        weights = bitstrings.sum(axis=1)
        degeneracy = np.asarray(
            [np.count_nonzero(weights == value) for value in range(n_qubits + 1)]
        )
        for role, betas in (
            ("training", manifest.training_betas),
            ("held out", manifest.held_out_betas),
        ):
            values = {sigma: [] for sigma in (1.0, 2.0, 4.0)}
            for beta in betas:
                target = exact_curie_weiss_target(n_qubits, beta)
                for row in [
                    item
                    for item in rows
                    if item["n_qubits"] == n_qubits and np.isclose(item["beta"], beta)
                ]:
                    profile = np.asarray(row["conditioned_profile"])
                    model = profile[weights] / degeneracy[weights]
                    for sigma in values:
                        values[sigma].append(
                            exact_mmd_squared(
                                model, target.probabilities, sigmas=(sigma,)
                            )
                        )
            mmd_lines.append(
                f"{n_qubits} & {role} & "
                + " & ".join(f"{np.mean(values[sigma]):.5f}" for sigma in values)
                + r" \\"
            )
    _write_immutable(
        arguments.output / "curie_weiss_mmd_bandwidths.tex",
        "\n".join(mmd_lines) + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
