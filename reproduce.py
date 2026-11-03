"""Reproduce every experiment of the paper in a row, then its figures and tables.

Each experiment writes to its own folder under the output root and resumes from
completed stages. The hardware counts come from data/hardware/iqm-emerald;
resubmitting the circuits needs an IQM access token and is not part of this
script. ``--check`` compares a finished full run with the bundles in data/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXPERIMENTS = (
    "gaussian",
    "curie-weiss",
    "curie-weiss-twisted",
    "curie-weiss-diagonal",
    "curie-weiss-classical",
    "scaling",
    "hardware",
    "figures",
)
HARDWARE_NOTE = (
    "Hardware: the IQM Emerald counts are read from data/hardware/iqm-emerald. "
    "Rerunning the circuits on the device requires an IQM access token and is "
    "not part of this script."
)
# Bundle name in data/ and the run folder that produces it, for the figures.
BUNDLES = {
    "gaussian_lattice": "gaussian/replay",
    "gaussian2d-diagonal": "gaussian-diagonal/replay",
    "curie-weiss-followup": "cw/final",
    "curie-weiss-twisted": "cw-twisted-bundle",
    "curie-weiss-diagonal": "cw-diagonal-k4/final",
    "curie-weiss-classical": "cw-classical/hpo",
    "scaling": "scaling/bundle",
}
DIAGONAL_VARIANTS = {
    "gaussian-diagonal": ("affine", "lattice"),
    "gaussian-diagonal-quadratic": ("quadratic", "lattice"),
    "gaussian-diagonal-complete": ("affine", "complete"),
    "gaussian-diagonal-complete-quadratic": ("quadratic", "complete"),
}
# Run output and stored bundle compared by --check.
CHECKS = [
    ("gaussian/replay", "gaussian_lattice"),
    ("gaussian-diagonal/replay", "gaussian2d-diagonal"),
    (
        "gaussian-diagonal-quadratic/replay",
        "gaussian2d-diagonal-variants/lattice-quadratic",
    ),
    (
        "gaussian-diagonal-complete/replay",
        "gaussian2d-diagonal-variants/complete-affine",
    ),
    (
        "gaussian-diagonal-complete-quadratic/replay",
        "gaussian2d-diagonal-variants/complete-quadratic",
    ),
    ("cw/final/figure_data.json", "curie-weiss-followup/figure_data.json"),
    ("cw-twisted-bundle/figure_data.json", "curie-weiss-twisted/figure_data.json"),
    (
        "cw-twisted-bundle/kappa0_control.json",
        "curie-weiss-twisted/kappa0_control.json",
    ),
    ("cw-diagonal-k4/final/figure_data.json", "curie-weiss-diagonal/figure_data.json"),
    ("cw-classical/hpo/small", "curie-weiss-classical/small"),
    ("cw-classical/hpo/large", "curie-weiss-classical/large"),
    ("scaling/bundle/figure_data.json", "scaling/figure_data.json"),
    ("scaling/bundle/physics", "scaling/physics"),
    ("scaling-all-to-all-64/search", "scaling/locality/all-to-all-64/search"),
    *(
        pair
        for seed in (100, 101, 102)
        for pair in (
            (
                f"scaling-all-to-all-64/final/n64-s{seed}.json",
                f"scaling/locality/all-to-all-64/fits/n64-s{seed}.json",
            ),
            (
                f"scaling-all-to-all-64/evaluations/n64-s{seed}.json",
                f"scaling/locality/all-to-all-64/fits/n64-s{seed}-evaluation.json",
            ),
        )
    ),
    *(
        pair
        for n in (16, 30)
        for seed in (100, 101, 102)
        for pair in (
            (
                f"hardware/fits/n{n}-s{seed}.json",
                f"hardware/iqm-emerald/models/n{n}-s{seed}.json",
            ),
            (
                f"hardware/evaluations/n{n}-s{seed}.json",
                f"hardware/iqm-emerald/models/n{n}-s{seed}-observables.json",
            ),
        )
    ),
    (
        "hardware/noiseless/n30-s100-noiseless-profiles.json",
        "hardware/iqm-emerald/models/n30-s100-noiseless-profiles.json",
    ),
]


def gaussian(root, smoke, workers):
    from paper_experiments import (
        gaussian2d_diagonal_report,
        gaussian2d_diagonal_run,
        gaussian2d_lattice_report,
        gaussian2d_lattice_run,
    )

    stage = "smoke" if smoke else "campaign"
    gaussian2d_lattice_run.main(["--output", str(root / "gaussian"), "--stage", stage])
    # The report accepts only the completed 31-fit campaign.
    if not smoke and not (root / "gaussian/replay/SUMMARY.json").exists():
        gaussian2d_lattice_report.main(
            [
                "--campaign",
                str(root / "gaussian"),
                "--output",
                str(root / "gaussian/replay"),
            ]
        )
    for name, (basis, ansatz) in DIAGONAL_VARIANTS.items():
        run = ["--output", str(root / name), "--stage", stage]
        gaussian2d_diagonal_run.main([*run, "--basis", basis, "--ansatz", ansatz])
        if not (root / name / "replay/SUMMARY.json").exists():
            gaussian2d_diagonal_report.main(
                [
                    "--campaign",
                    str(root / name),
                    "--output",
                    str(root / name / "replay"),
                ]
            )


def curie_weiss(root, smoke, workers):
    from paper_experiments import cw_followup_hpo, cw_followup_stages, cw_followup_tpe

    hpo = root / "cw" / "hpo"
    if not (hpo / "random-review-ready.json").exists():
        cw_followup_hpo.run_all(hpo, smoke=smoke, workers=workers)
    if not (hpo / "tpe-review-ready.json").exists():
        cw_followup_tpe.run_all(hpo, smoke=smoke, workers=workers)
    if not (root / "cw" / "final" / "review-ready.json").exists():
        cw_followup_stages.run_all(hpo, phase="reporting", smoke=smoke, workers=workers)


def encoder_campaign(campaign, directory, smoke, workers):
    from paper_experiments import cw_encoder_hpo

    markers = {
        "random": "random-review-ready.json",
        "tpe": "tpe-review-ready.json",
        "confirmation": "confirmation/review-ready.json",
        "final": "final/review-ready.json",
    }
    for phase, marker in markers.items():
        if not (directory / marker).exists():
            cw_encoder_hpo.run(
                directory, phase, smoke=smoke, workers=workers, campaign=campaign
            )


def curie_weiss_twisted(root, smoke, workers):
    from paper_experiments import cw_kappa0_control, cw_twisted_bundle

    encoder_campaign("twisted", root / "cw-twisted", smoke, workers)
    encoder_campaign(
        "twisted_independent", root / "cw-twisted-independent", smoke, workers
    )
    # The bundle accepts only final stages with ten repeats.
    if not smoke:
        cw_twisted_bundle.build(
            [root / "cw-twisted", root / "cw-twisted-independent"],
            root / "cw-twisted-bundle",
        )
    campaign = json.loads((root / "cw-twisted/final/campaign.json").read_text())
    cw_kappa0_control.run(
        root / "cw-kappa0", campaign["winner"], workers=workers, smoke=smoke
    )
    cw_kappa0_control.bundle(
        root / "cw-kappa0", root / "cw-twisted-bundle/kappa0_control.json"
    )


def curie_weiss_diagonal(root, smoke, workers):
    encoder_campaign("diagonal_k4", root / "cw-diagonal-k4", smoke, workers)


def curie_weiss_classical(root, smoke, workers):
    from paper_experiments import cw_classical_campaign

    markers = {
        "random": "random-review-ready.json",
        "tpe": "tpe-review-ready.json",
        "confirmation": "confirmation.json",
        "final": "figure_data.json",
    }
    for capacity in ("small", "large"):
        directory = root / "cw-classical" / "hpo" / capacity
        for phase, marker in markers.items():
            if not (directory / marker).exists():
                cw_classical_campaign.run_all(
                    directory,
                    phase=phase,
                    capacity=capacity,
                    smoke=smoke,
                    workers=workers,
                )


def scaling(root, smoke, workers):
    from paper_experiments import scaling_hpo

    out = root / "scaling"
    flag = ["--smoke"] if smoke else []
    # Smoke runs one trial at 16 and 36 qubits; the smoke confirmation reads
    # protocol candidates that a one-trial search does not produce, and the
    # smoke final trains a fixed protocol trial.
    search = ["--trials", "1", "--sizes", "16", "36"] if smoke else []
    scaling_hpo.main(["run", str(out / "search"), *search, *flag])
    if not smoke:
        scaling_hpo.main(
            ["confirm", str(out / "confirm"), "--search", str(out / "search")]
        )
    sizes = (
        ["16", "36"]
        if smoke
        else [str(n) for n in scaling_hpo.PROTOCOL["final"]["sizes"]]
    )
    scaling_hpo.main(
        [
            "final",
            str(out / "final"),
            "--search",
            str(out / "search"),
            "--confirmation",
            str(out / "confirm"),
            "--sizes",
            *sizes,
            *flag,
        ]
    )
    scaling_hpo.main(
        ["evaluate", str(out / "evaluations"), "--fits", str(out / "final"), *flag]
    )
    scaling_hpo.main(
        [
            "bundle",
            str(out / "bundle"),
            "--fits",
            str(out / "final"),
            "--evaluations",
            str(out / "evaluations"),
            *flag,
        ]
    )

    # 64-qubit all-to-all comparison: its own S1 search at 64 qubits, then the
    # final fits of the best search trial.
    local = root / "scaling-all-to-all-64"
    size = "16" if smoke else "64"
    common = ["--ansatz", "complete", "--sizes", size, *flag]
    trials = ["--trials", "1"] if smoke else []
    scaling_hpo.main(["run", str(local / "search"), "--family", "S1", *trials, *common])
    rows = [json.loads(p.read_text()) for p in (local / "search").glob("S1-*.json")]
    best = min((r for r in rows if r["status"] == "ok"), key=lambda r: r["score"])
    trial = f"S1-{best['index']:03d}"
    print(f"all-to-all 64: best search trial {trial}", flush=True)
    scaling_hpo.main(
        [
            "final",
            str(local / "final"),
            "--search",
            str(local / "search"),
            "--trial",
            trial,
            *common,
        ]
    )
    scaling_hpo.main(
        ["evaluate", str(local / "evaluations"), "--fits", str(local / "final"), *flag]
    )


def hardware(root, smoke, workers):
    """Classical training of the deployed 16- and 30-qubit lattice fits."""
    from paper_experiments import hardware_noiseless, scaling_hpo

    out, search = root / "hardware", root / "scaling"
    flag = ["--smoke"] if smoke else []
    scaling_hpo.main(
        [
            "final",
            str(out / "fits"),
            "--search",
            str(search / "search"),
            "--confirmation",
            str(search / "confirm"),
            "--sizes",
            *(["16"] if smoke else ["16", "30"]),
            *flag,
        ]
    )
    scaling_hpo.main(
        ["evaluate", str(out / "evaluations"), "--fits", str(out / "fits"), *flag]
    )
    # Exact noiseless profiles of the deployed fit; the 30-qubit state needs
    # about 40 GB of memory, so the smoke run uses the 16-qubit fit.
    size = 16 if smoke else 30
    if not list((out / "noiseless").glob(f"n{size}-s*-noiseless-profiles.json")):
        hardware_noiseless.main(
            ["--size", str(size), "--output", str(out / "noiseless")]
        )
    print(HARDWARE_NOTE, flush=True)


def figures(root, smoke, workers):
    from paper_experiments import hardware_report, target_families
    from paper_experiments.cw_classical_report import render as render_classical
    from paper_experiments.figures import render_all

    # Full runs render from their own bundles, linked under the names of data/.
    # The reporting code rejects smoke bundles, so smoke runs render data/.
    data = DATA
    if not smoke:
        data = root / "data"
        data.mkdir(exist_ok=True)
        for name, source in BUNDLES.items():
            link = data / name
            if not link.is_symlink():
                link.symlink_to(root / source, target_is_directory=True)
    output = root / "figures"
    render_all(data, output)
    target_families.render(output)
    hardware_report.main(["--output", str(output / "hardware")])
    for capacity in ("small", "large"):
        source = root / "cw-classical" / "hpo" / capacity / "figure_data.json"
        render_classical(
            source,
            root / "cw-classical-replay" / capacity,
            iqp_data_root=None if smoke else data,
            allow_smoke=smoke,
        )
    print(f"figures and tables: {output}", flush=True)
    print(HARDWARE_NOTE, flush=True)


STAGES = {
    "gaussian": gaussian,
    "curie-weiss": curie_weiss,
    "curie-weiss-twisted": curie_weiss_twisted,
    "curie-weiss-diagonal": curie_weiss_diagonal,
    "curie-weiss-classical": curie_weiss_classical,
    "scaling": scaling,
    "hardware": hardware,
    "figures": figures,
}


def _compare(fresh, stored, where, problems, rtol=1e-5, atol=1e-8):
    """Numeric leaves of ``stored`` against ``fresh``; strings and timings skipped."""
    if isinstance(stored, dict):
        for key, value in stored.items():
            if "seconds" in key:
                continue
            if not isinstance(fresh, dict) or key not in fresh:
                problems.append(f"{where}/{key}: missing")
            else:
                _compare(fresh[key], value, f"{where}/{key}", problems, rtol, atol)
        return
    if isinstance(stored, (bool, int, float)) or (
        isinstance(stored, (list, np.ndarray))
        and np.asarray(stored, dtype=object).dtype != object
    ):
        a, b = np.asarray(fresh), np.asarray(stored)
        if a.shape != b.shape:
            problems.append(f"{where}: shape {a.shape} != {b.shape}")
        elif b.dtype.kind in "biuf" and not np.allclose(
            a, b, rtol=rtol, atol=atol, equal_nan=True
        ):
            diff = np.max(np.abs(a.astype(float) - b.astype(float)))
            problems.append(f"{where}: max |diff| {diff:.3g}")
        return
    if isinstance(stored, list):
        if not isinstance(fresh, list) or len(fresh) != len(stored):
            problems.append(f"{where}: length differs")
            return
        for i, (x, y) in enumerate(zip(fresh, stored, strict=True)):
            _compare(x, y, f"{where}[{i}]", problems, rtol, atol)


def _load(path):
    if path.suffix == ".npz":
        with np.load(path) as arrays:
            return {
                k: arrays[k] for k in arrays.files if arrays[k].dtype.kind in "biuf"
            }
    return json.loads(path.read_text())


def check(root, data=DATA):
    """Compare a finished full run with the stored bundles; return the failures."""
    pairs = []
    for fresh, stored in CHECKS:
        fresh, stored = root / fresh, data / stored
        if stored.is_dir():
            pairs += [
                (fresh / p.relative_to(stored), p)
                for p in sorted(stored.rglob("*"))
                if p.suffix in (".json", ".npz")
            ]
        else:
            pairs.append((fresh, stored))
    failures = 0
    for target, source in pairs:
        problems = []
        if not target.exists():
            problems.append("missing")
        else:
            _compare(_load(target), _load(source), "", problems)
        if problems:
            failures += 1
            print(f"FAIL {target.relative_to(root)}")
            for line in problems[:5]:
                print(f"     {line}")
    print(f"{len(pairs) - failures} of {len(pairs)} files match data/")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reproduce",
        help="output root under results/ (smoke runs use OUTPUT/smoke)",
    )
    parser.add_argument("--smoke", action="store_true", help="reduced pipeline check")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", choices=EXPERIMENTS, action="append")
    parser.add_argument(
        "--check", action="store_true", help="compare a finished full run with data/"
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.output.resolve() / ("smoke" if args.smoke else "")
    # The classical campaign writes only under results/.
    if not root.is_relative_to(ROOT / "results"):
        parser.error("--output must lie under results/")
    if args.check:
        return int(check(root) > 0)
    for name in args.only or EXPERIMENTS:
        print(f"== {name}", flush=True)
        STAGES[name](root, args.smoke, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
