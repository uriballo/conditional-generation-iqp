"""Reproduce every experiment of the paper in a row, then its figures and tables.

By default each experiment trains its final fits from the configurations
selected in data/ and builds every evaluation, bundle, figure and table from
them. ``--full`` also reruns the hyperparameter searches and confirmations that
selected those configurations. Each experiment writes to its own folder under
the output root and resumes from completed stages. The hardware counts come from
data/hardware/iqm-emerald; resubmitting the circuits needs an IQM access token
and is not part of this script. ``--check`` compares a finished run with the
bundles in data/.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    "score-study",
    "hardware",
    "figures",
)
HARDWARE_NOTE = (
    "Hardware: the IQM Emerald counts are read from data/hardware/iqm-emerald. "
    "Rerunning the circuits on the device requires an IQM access token and is "
    "not part of this script."
)
# Hamming-weight campaign of gaussian2d_hpo and its bundle under data/; the
# input campaign comes first because the diagonal bundles compare against it.
GAUSSIAN = {
    "input": "gaussian_lattice",
    "diagonal": "gaussian2d-diagonal",
    "diagonal_quadratic": "gaussian2d-diagonal-variants/lattice-quadratic",
    "diagonal_complete": "gaussian2d-diagonal-variants/complete-affine",
    "diagonal_complete_quadratic": "gaussian2d-diagonal-variants/complete-quadratic",
}
ALL_TO_ALL = "scaling/locality/all-to-all-64-budget512"
# Bundle name in data/ and the run folder that produces it, for the figures.
BUNDLES = {
    "gaussian_lattice": "gaussian/bundles/gaussian_lattice",
    "gaussian2d-diagonal": "gaussian/bundles/gaussian2d-diagonal",
    "gaussian2d-diagonal-variants": "gaussian/bundles/gaussian2d-diagonal-variants",
    "curie-weiss-followup": "cw/final",
    "curie-weiss-twisted": "cw-twisted-bundle",
    "curie-weiss-diagonal": "cw-diagonal-k4/final",
    "curie-weiss-classical": "cw-classical/hpo",
    "scaling": "scaling/bundle",
}
SEEDS = (100, 101, 102)
# Run output and stored bundle compared by --check.
CHECKS = [
    *((f"gaussian/bundles/{name}", name) for name in GAUSSIAN.values()),
    (
        "gaussian/bundles/gaussian2d-diagonal-variants/five_way.json",
        "gaussian2d-diagonal-variants/five_way.json",
    ),
    ("cw/final/figure_data.json", "curie-weiss-followup/figure_data.json"),
    ("cw-twisted-bundle/figure_data.json", "curie-weiss-twisted/figure_data.json"),
    (
        "cw-twisted-bundle/kappa0_control.json",
        "curie-weiss-twisted/kappa0_control.json",
    ),
    ("cw-diagonal-k4/final/figure_data.json", "curie-weiss-diagonal/figure_data.json"),
    *(
        (
            f"cw-classical/hpo/{capacity}/figure_data.json",
            f"curie-weiss-classical/{capacity}/figure_data.json",
        )
        for capacity in ("small", "large")
    ),
    ("cw-classical/hpo/comparison.json", "curie-weiss-classical/comparison.json"),
    ("scaling/bundle/figure_data.json", "scaling/figure_data.json"),
    ("scaling/bundle/physics", "scaling/physics"),
    ("scaling-all-to-all-64-budget512/fits", f"{ALL_TO_ALL}/fits"),
    ("scaling-all-to-all-64-budget512/table.json", f"{ALL_TO_ALL}/table.json"),
    ("score-study/spearman.json", "scaling/score_study/spearman.json"),
    *(
        pair
        for n in (16, 30)
        for seed in SEEDS
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
# Search and confirmation records, produced only by --full.
FULL_CHECKS = [
    *(
        (
            f"cw-classical/hpo/{capacity}/{name}",
            f"curie-weiss-classical/{capacity}/{name}",
        )
        for capacity in ("small", "large")
        for name in ("shortlist.json", "confirmation.json")
    ),
    ("scaling-all-to-all-64-budget512/search", f"{ALL_TO_ALL}/search"),
    ("score-study/checkpoints.json", "scaling/score_study/checkpoints.json"),
]


def _stored(path):
    return json.loads((DATA / path).read_text())


def gaussian(root, smoke, full, workers):
    from paper_experiments import (
        gaussian2d_diagonal_report,
        gaussian2d_hpo,
        gaussian2d_lattice_report,
    )

    bundles = root / "gaussian" / "bundles"
    markers = {
        "random": "random-complete.json",
        "tpe": "tpe-complete.json",
        "confirmation": "confirmation/winner.json",
        "final": "final/complete.json",
    }
    for campaign, name in GAUSSIAN.items():
        directory = root / "gaussian" / campaign
        summary = DATA / name / "SUMMARY.json"
        winner = None if full else json.loads(summary.read_text())["search"]["winner"]
        for phase, marker in markers.items():
            if (full or phase == "final") and not (directory / marker).exists():
                gaussian2d_hpo.run(
                    directory,
                    campaign,
                    phase,
                    smoke=smoke,
                    workers=workers,
                    winner=winner,
                )
        if (bundles / name / "SUMMARY.json").exists():
            continue
        report = ["--campaign", str(directory), "--output", str(bundles / name)]
        if not full:
            report += ["--search-from", str(summary)]
        if campaign == "input":
            gaussian2d_lattice_report.main(report)
        else:
            gaussian2d_diagonal_report.main([*report, "--data", str(bundles)])
    gaussian2d_diagonal_report.main(
        [
            "--five-way",
            "--data",
            str(bundles),
            "--output",
            str(bundles / "gaussian2d-diagonal-variants/five_way.json"),
        ]
    )


def curie_weiss(root, smoke, full, workers):
    from paper_experiments import cw_followup_hpo, cw_followup_stages, cw_followup_tpe

    if (root / "cw" / "final" / "review-ready.json").exists():
        return
    if not full:
        winners = _stored("curie-weiss-followup/provenance/confirmation-winners.json")
        cw_followup_stages.run_final(
            root / "cw" / "final", winners, smoke=smoke, workers=workers
        )
        return
    hpo = root / "cw" / "hpo"
    if not (hpo / "random-review-ready.json").exists():
        cw_followup_hpo.run_all(hpo, smoke=smoke, workers=workers)
    if not (hpo / "tpe-review-ready.json").exists():
        cw_followup_tpe.run_all(hpo, smoke=smoke, workers=workers)
    cw_followup_stages.run_all(hpo, phase="reporting", smoke=smoke, workers=workers)


def encoder_campaign(campaign, directory, winner, smoke, full, workers):
    """cw_encoder_hpo phases; without --full only the final fits of ``winner``."""
    from paper_experiments import cw_encoder_hpo

    markers = {
        "random": "random-review-ready.json",
        "tpe": "tpe-review-ready.json",
        "confirmation": "confirmation/review-ready.json",
        "final": "final/review-ready.json",
    }
    for phase, marker in markers.items():
        if (full or phase == "final") and not (directory / marker).exists():
            cw_encoder_hpo.run(
                directory,
                phase,
                smoke=smoke,
                workers=workers,
                campaign=campaign,
                winner=None if full else winner,
            )


def curie_weiss_twisted(root, smoke, full, workers):
    from paper_experiments import cw_kappa0_control, cw_twisted_bundle

    families = _stored("curie-weiss-twisted/figure_data.json")["families"]
    for campaign, directory, family in (
        ("twisted", "cw-twisted", "shared_twisted_parity"),
        ("twisted_independent", "cw-twisted-independent", "independent_twisted_parity"),
    ):
        winner = {k: families[family][k] for k in ("trial_id", "config")}
        encoder_campaign(campaign, root / directory, winner, smoke, full, workers)
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


def curie_weiss_diagonal(root, smoke, full, workers):
    stored = _stored("curie-weiss-diagonal/figure_data.json")
    winner = {k: stored[k] for k in ("trial_id", "config")}
    encoder_campaign(
        "diagonal_k4", root / "cw-diagonal-k4", winner, smoke, full, workers
    )


def curie_weiss_classical(root, smoke, full, workers):
    from paper_experiments import cw_classical_campaign

    markers = {
        "random": "random-review-ready.json",
        "tpe": "tpe-review-ready.json",
        "confirmation": "confirmation.json",
        "final": "figure_data.json",
    }
    for capacity in ("small", "large"):
        directory = root / "cw-classical" / "hpo" / capacity
        stored = DATA / "curie-weiss-classical" / capacity / "confirmation.json"
        for phase, marker in markers.items():
            if (full or phase == "final") and not (directory / marker).exists():
                cw_classical_campaign.run_all(
                    directory,
                    phase=phase,
                    capacity=capacity,
                    smoke=smoke,
                    workers=workers,
                    confirmation=None if full else stored,
                )


def scaling(root, smoke, full, workers):
    from paper_experiments import scaling_hpo, scaling_locality_table

    out = root / "scaling"
    flag = ["--smoke"] if smoke else []
    if full:
        # Smoke runs one trial at 16 and 36 qubits; the smoke confirmation reads
        # protocol candidates that a one-trial search does not produce, and the
        # smoke final trains a fixed protocol trial.
        search = ["--trials", "1", "--sizes", "16", "36"] if smoke else []
        scaling_hpo.main(["run", str(out / "search"), *search, *flag])
        if not smoke:
            scaling_hpo.main(
                ["confirm", str(out / "confirm"), "--search", str(out / "search")]
            )
        selection = [
            "--search",
            str(out / "search"),
            "--confirmation",
            str(out / "confirm"),
        ]
    else:
        selection = ["--selection", str(DATA / "scaling/physics/fits/n64-s100.json")]
    sizes = (
        ["16", "36"]
        if smoke
        else [str(n) for n in scaling_hpo.PROTOCOL["final"]["sizes"]]
    )
    scaling_hpo.main(
        ["final", str(out / "final"), *selection, "--sizes", *sizes, *flag]
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

    # 64-qubit all-to-all comparison at the training budget of the lattice fits:
    # its own S1 search at 64 qubits, then the final fits of the best trial.
    local = root / "scaling-all-to-all-64-budget512"
    size = "16" if smoke else "64"
    common = ["--ansatz", "complete", "--sizes", size, *flag]
    if full:
        trials = ["--trials", "1"] if smoke else []
        budget = ["--budget", "512", "1024"]
        scaling_hpo.main(
            ["run", str(local / "search"), "--family", "S1", *budget, *trials, *common]
        )
        rows = [json.loads(p.read_text()) for p in (local / "search").glob("S1-*.json")]
        best = min((r for r in rows if r["status"] == "ok"), key=lambda r: r["score"])
        trial = f"S1-{best['index']:03d}"
        print(f"all-to-all 64: best search trial {trial}", flush=True)
        selection = ["--search", str(local / "search"), "--trial", trial]
    else:
        selection = ["--selection", str(DATA / ALL_TO_ALL / "fits/n64-s100.json")]
    scaling_hpo.main(["final", str(local / "final"), *selection, *common])
    scaling_hpo.main(
        ["evaluate", str(local / "evaluations"), "--fits", str(local / "final"), *flag]
    )
    # Receipts and evaluations side by side, as in data/.
    (local / "fits").mkdir(exist_ok=True)
    for seed in SEEDS:
        name = f"n{size}-s{seed}"
        shutil.copyfile(
            local / "final" / f"{name}.json", local / "fits" / f"{name}.json"
        )
        shutil.copyfile(
            local / "evaluations" / f"{name}.json",
            local / "fits" / f"{name}-evaluation.json",
        )
    # The table reads 64-qubit fits.
    if not smoke:
        table = scaling_locality_table.table(local / "fits")
        (local / "table.json").write_text(json.dumps(table, indent=2) + "\n")


def score_study(root, smoke, full, workers):
    """Sixteen-qubit selection-score study. Without --full, and in smoke runs,
    the correlations are computed from the stored checkpoints."""
    from paper_experiments import scaling_score_study

    out = root / "score-study"
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = DATA / "scaling/score_study/checkpoints.json"
    if full and not smoke:
        scaling_score_study.main(["run", str(out / "runs")])
        checkpoints = out / "checkpoints.json"
        trials = scaling_score_study.collect(out / "runs")
        checkpoints.write_text(json.dumps(trials) + "\n")
    result = scaling_score_study.analyze(json.loads(checkpoints.read_text()))
    (out / "spearman.json").write_text(json.dumps(result, indent=2) + "\n")


def hardware(root, smoke, full, workers):
    """Classical training of the deployed 16- and 30-qubit lattice fits."""
    from paper_experiments import hardware_noiseless, scaling_hpo

    out, search = root / "hardware", root / "scaling"
    flag = ["--smoke"] if smoke else []
    if full:
        selection = [
            "--search",
            str(search / "search"),
            "--confirmation",
            str(search / "confirm"),
        ]
    else:
        receipt = DATA / "hardware/iqm-emerald/models/n16-s100.json"
        selection = ["--selection", str(receipt)]
    scaling_hpo.main(
        [
            "final",
            str(out / "fits"),
            *selection,
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


def figures(root, smoke, full, workers):
    from paper_experiments import hardware_report, target_families
    from paper_experiments.cw_classical_report import comparison
    from paper_experiments.cw_classical_report import render as render_classical
    from paper_experiments.figures import render_all

    # Runs render from their own bundles, linked under the names of data/.
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
    comparison(
        data / "curie-weiss-classical",
        data / "curie-weiss-followup",
        root / "cw-classical" / "hpo",
    )
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
    "score-study": score_study,
    "hardware": hardware,
    "figures": figures,
}


def _compare(fresh, stored, where, problems, rtol=1e-5, atol=1e-8):
    """Numeric leaves of ``stored`` against ``fresh``; strings and timings skipped."""
    if isinstance(stored, dict):
        for key, value in stored.items():
            if "second" in key:
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


def check(root, data=DATA, full=False):
    """Compare a finished run with the stored bundles; return the failures."""
    pairs = []
    for fresh, stored in CHECKS + (FULL_CHECKS if full else []):
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="also rerun the hyperparameter searches and confirmations",
    )
    parser.add_argument("--smoke", action="store_true", help="reduced pipeline check")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", choices=EXPERIMENTS, action="append")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare a finished run with data/ (with --full, also the searches)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.output.resolve() / ("smoke" if args.smoke else "")
    # The classical campaign writes only under results/.
    if not root.is_relative_to(ROOT / "results"):
        parser.error("--output must lie under results/")
    if args.check:
        return int(check(root, full=args.full) > 0)
    for name in args.only or EXPERIMENTS:
        print(f"== {name}", flush=True)
        STAGES[name](root, args.smoke, args.full, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
