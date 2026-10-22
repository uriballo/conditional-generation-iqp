"""Entry point for figure replay and the Curie--Weiss searches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("figures", "hpo"), default="figures")
    p.add_argument(
        "--hpo-phase",
        choices=("random", "tpe", "confirmation", "final", "reporting"),
        default="random",
    )
    p.add_argument(
        "--experiment",
        choices=(
            "all",
            "gaussian-lattice",
            "curie-weiss",
            "curie-weiss-classical",
            "scaling",
        ),
        default="all",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--classical-capacity", choices=("small", "large"), default="small")
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument(
        "--data-root",
        type=Path,
        help="compact data directory (default: data/)",
    )
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be positive")
    if args.experiment == "curie-weiss-classical":
        from paper_experiments.cw_classical_config import FAMILIES, model, protocol

        namespace = (
            args.output
            / ("smoke" if args.smoke else args.mode)
            / args.classical_capacity
        )
        if args.dry_run:
            print(
                json.dumps(
                    dict(
                        phase=args.hpo_phase,
                        capacity=args.classical_capacity,
                        protocol=protocol(args.smoke),
                        output=str(namespace),
                        architectures={
                            family: model(family).architecture(
                                16, args.classical_capacity
                            )
                            for family in FAMILIES
                        },
                    ),
                    indent=2,
                )
            )
            return 0
        if args.mode == "figures":
            if args.data_root is None:
                p.error(
                    "classical figures require --data-root pointing to a final campaign"
                )
            from paper_experiments.cw_classical_report import render

            source = args.data_root
            if source.is_dir():
                source = source / "figure_data.json"
            paths = render(source, namespace / "figures", allow_smoke=args.smoke)
            print("\n".join(str(path) for path in paths))
        else:
            from paper_experiments.cw_classical_campaign import run_all

            result = run_all(
                namespace,
                phase=args.hpo_phase,
                capacity=args.classical_capacity,
                smoke=args.smoke,
                workers=args.workers,
            )
            print(json.dumps(result, indent=2))
        return 0
    cw_choices = ("all", "curie-weiss")
    if args.hpo_phase != "random" and not (
        args.mode == "hpo" and args.experiment in cw_choices
    ):
        p.error("--hpo-phase applies only to Curie-Weiss HPO")
    if args.mode == "hpo":
        if args.experiment not in cw_choices:
            p.error("HPO applies only to the current Curie-Weiss campaign")
        from paper_experiments.cw_followup_hpo import protocol

        if args.hpo_phase in ("confirmation", "final", "reporting"):
            from paper_experiments.cw_followup_stages import run_all
        elif args.hpo_phase == "tpe":
            from paper_experiments.cw_followup_tpe import run_all
        else:
            from paper_experiments.cw_followup_hpo import run_all
    if args.dry_run:
        if args.mode == "hpo":
            plan = protocol(args.smoke)
            if args.hpo_phase != "random":
                plan = {"phase": args.hpo_phase, "protocol": plan}
        else:
            plan = dict(
                data=str(args.data_root or Path("data")),
                output=str(args.output / "figures"),
                experiment=args.experiment,
            )
        print(json.dumps(plan, indent=2))
        return 0
    namespace = args.output / ("smoke" if args.smoke else args.mode)
    if args.mode == "figures":
        from paper_experiments.figures import render_all

        paths = render_all(
            args.data_root or Path(__file__).parent / "data",
            namespace,
            experiment=args.experiment,
        )
        print("\n".join(str(path) for path in paths))
        return 0
    if args.mode == "hpo":
        extra = (
            {"phase": args.hpo_phase}
            if args.hpo_phase in ("confirmation", "final", "reporting")
            else {}
        )
        run_all(
            namespace / "hpo" if args.smoke else namespace,
            smoke=args.smoke,
            workers=args.workers,
            **extra,
        )
        message = f"Curie-Weiss {args.hpo_phase} complete"
        if args.hpo_phase in ("random", "tpe"):
            message += ", paused for review"
        print(f"{message}: {namespace}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
