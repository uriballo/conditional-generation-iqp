#!/usr/bin/env python3
"""Build the local coauthor-review package without submission tooling."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "output" / "author-review" / "conditional-iqp-draft"
ARCHIVE = DESTINATION.with_suffix(".tar.gz")

FILES = (
    "paper/main.tex",
    "paper/supplement.tex",
    "paper/references.bib",
    "paper/outline.md",
    "paper/self-review.md",
    "paper/CLAIM_CHANGES.md",
    "paper/ARXIV_METADATA_DRAFT.md",
    "output/pdf/conditional_iqp_generation.pdf",
    "results/curie-weiss-m5-foundations-v3/report/campaign-report.json",
    "results/curie-weiss-m5-foundations-v3/report/claim-registry.json",
    "results/curie-weiss-capacity-control-v2/summary.json",
    "results/curie-weiss-capacity-control-v2/summary.md",
    "CONTEXT.md",
    "pyproject.toml",
    "uv.lock",
)

DIRECTORIES = (
    "paper/generated",
    "paper/gaussian_figures",
    "paper/curie_weiss_ancilla_figures",
    "paper/symmetry_figures",
    "paper/figures",
    "configs",
    "archive",
    "docs",
    "output/arxiv",
    "results/gaussian-foundations-v1/evidence",
    "results/gaussian-foundations-v1/report",
    "results/curie-weiss-ancilla-pilot-v1",
    "results/curie-weiss-m5-foundations-v2/report",
    "results/curie-weiss-capacity-control-v2/results",
    "results/two-block-screen-v1",
    "results/radial-shell-screen-v1",
    "results/quantum-campaign-v1",
    "results/curie-weiss-foundations-v1/datasets",
    "results/ancilla/summary",
    "results/encoding/summary",
    "results/encoding_benchmark/summary",
    "results/interpolation/summary",
    "results/summary",
    "src/iqp_cond",
    "scripts",
    "tests",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> Path:
    missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
    missing.extend(
        relative for relative in DIRECTORIES if not (ROOT / relative).is_dir()
    )
    if missing:
        raise RuntimeError(f"author-review inputs are missing: {missing}")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir(parents=True)
    for relative in FILES:
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for relative in DIRECTORIES:
        shutil.copytree(
            ROOT / relative,
            DESTINATION / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )

    forbidden = [
        path
        for path in DESTINATION.rglob("*")
        if path.is_file()
        and ("upload" in path.name.lower() or "submit" in path.name.lower())
    ]
    if forbidden:
        raise RuntimeError(f"submission tooling entered review package: {forbidden}")

    (DESTINATION / "README.md").write_text(
        "# Conditional IQP author-review package\n\n"
        "This local package is for coauthor review. It contains the manuscript "
        "source and PDF, bibliography, figures, generated claims, frozen "
        "campaign summaries, the historical v2 capacity diagnostic, "
        "exploratory symmetry-screen results, and the sealed Gaussian "
        "execution-source snapshot.\n\n"
        "Start with `paper/CLAIM_CHANGES.md`, then run `uv sync --group dev`, "
        "`uv run pytest -q`, and `uv run ruff check src scripts tests`. Build "
        "the manuscript with `(cd paper && latexmk -pdf main.tex)`.\n\n"
        "No upload or submission script is included. Draft metadata are "
        "advisory; the authors retain sole responsibility for approval and "
        "submission. Review `paper/CLAIM_CHANGES.md` first.\n"
    )
    records = {
        str(path.relative_to(DESTINATION)): _sha256(path)
        for path in sorted(DESTINATION.rglob("*"))
        if path.is_file()
    }
    (DESTINATION / "SHA256SUMS.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    return DESTINATION


def build_archive(directory: Path) -> Path:
    """Create the single-file handoff archive next to the review directory."""
    archive = shutil.make_archive(
        str(ARCHIVE).removesuffix(".tar.gz"),
        "gztar",
        root_dir=directory.parent,
        base_dir=directory.name,
    )
    return Path(archive)


def main() -> int:
    directory = build()
    print(directory)
    print(build_archive(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
