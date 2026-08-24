"""Run and summarize the frozen v2 matched-capacity control."""

from __future__ import annotations

import argparse
from pathlib import Path

from iqp_cond.curie_weiss_capacity_control import (
    load_manifest,
    run_seed,
    summarize,
    summary_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "curie_weiss_m5_foundations_v2.json"
DEFAULT_OUTPUT = ROOT / "results" / "curie-weiss-capacity-control-v2"
DEFAULT_REPORT = (
    ROOT
    / "results"
    / "curie-weiss-m5-foundations-v2"
    / "report"
    / "campaign-report.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ancilla-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "run":
        seeds = (args.seed,) if args.seed is not None else manifest.optimizer_seeds
        for seed in seeds:
            path = run_seed(ROOT, manifest, args.output, seed)
            print(path.relative_to(ROOT))
        return
    summary = summarize(ROOT, manifest, args.output, args.ancilla_report)
    markdown = summary_markdown(summary)
    path = args.output / "summary.md"
    path.write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
