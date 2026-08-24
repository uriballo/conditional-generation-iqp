"""Build and verify the Gaussian plus Curie--Weiss arXiv package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json

FORBIDDEN = (
    "ising model",
    "ising campaign",
    "theorem gate",
    "kill tree",
    "quantum-ready",
    "quantum readiness",
    "failure draft",
)
FIGURES = (
    "fig1_pipeline_targets.pdf",
    "fig2_headline_profiles.pdf",
    "fig3_size_width_surface.pdf",
    "fig4_condition_count.pdf",
    "fig5_ablations.pdf",
)
CURIE_WEISS_FIGURES = (
    "curie_weiss_ancilla_temperature_response.pdf",
    "curie_weiss_m5_fidelity.pdf",
)
GENERATED = (
    "gaussian_claims.tex",
    "factorial_cells.tex",
    "factorial_wins.tex",
    "raw_bitstring_cells.tex",
    "condition_count_cells.tex",
    "condition_count_per_condition.tex",
    "condition_count_intervals.tex",
    "secondary_diagnostics.tex",
    "all_intervals.tex",
    "all_intervals_paginated.tex",
    "curie_weiss_ancilla_claims.tex",
    "curie_weiss_ancilla_effects.tex",
    "curie_weiss_ancilla_response.tex",
    "curie_weiss_ancilla_all_rows.tex",
    "curie_weiss_ancilla_pilot.tex",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, default=Path("results/gaussian-foundations-v1")
    )
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"required arXiv source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _check_forbidden(root: Path) -> None:
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".tex", ".bib", ".json", ".md"}:
            continue
        text = path.read_text(errors="replace").lower()
        for phrase in FORBIDDEN:
            if phrase in text:
                raise RuntimeError(f"forbidden phrase {phrase!r} in {path}")


def _run(command: list[str], cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{output[-8000:]}")
    return output


def _verify_compile(source: Path) -> Path:
    environment = os.environ.copy()
    environment.update(
        {
            "SOURCE_DATE_EPOCH": "1786406400",
            "TZ": "UTC",
            "LC_ALL": "C",
        }
    )
    _run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        source,
        environment,
    )
    log = (source / "main.log").read_text(errors="replace")
    rejected = (
        "undefined citations",
        "undefined references",
        "There were undefined",
        "Overfull \\hbox",
        "Overfull \\vbox",
    )
    matches = [phrase for phrase in rejected if phrase in log]
    if matches:
        raise RuntimeError(f"manuscript log failed quality gate: {matches}")
    pdf = source / "main.pdf"
    info = _run(["pdfinfo", str(pdf)], source, environment)
    pages = next(
        int(line.split(":", 1)[1])
        for line in info.splitlines()
        if line.startswith("Pages:")
    )
    for page in range(1, pages + 1):
        text = _run(
            ["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"],
            source,
            environment,
        )
        if len("".join(text.split())) < 20:
            raise RuntimeError(f"blank or nearly blank PDF page: {page}")
    return pdf


def _clean_build_products(source: Path) -> None:
    for suffix in (
        ".aux",
        ".bbl",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".pdf",
    ):
        path = source / f"main{suffix}"
        if path.exists():
            path.unlink()


def _source_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SOURCE_SHA256.json"
    }


def _write_deterministic_archive(source: Path, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(source.rglob("*")):
                    if not path.is_file():
                        continue
                    arcname = path.relative_to(source).as_posix()
                    info = archive.gettarinfo(str(path), arcname)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    if output.exists():
        if output.read_bytes() != temporary.read_bytes():
            raise RuntimeError("immutable arXiv archive differs from regenerated bytes")
        temporary.unlink()
    else:
        temporary.replace(output)
    return _sha256(output)


def _populate(repository: Path, stage: Path, args: argparse.Namespace) -> None:
    campaign = repository / args.campaign
    paper = repository / args.paper
    report = campaign / "report" / "complete-report.json"
    claim_registry = campaign / "report" / "claim-registry.json"
    report_data = read_immutable_json(report)
    if report_data.get("completed_atomic_models") != 1_020:
        raise RuntimeError("canonical promotion requires all 1,020 models")
    registry_data = json.loads(claim_registry.read_text())
    execution_source = read_immutable_json(
        campaign / "evidence" / "execution-source.json"
    )
    if (
        registry_data.get("completed_atomic_models") != 1_020
        or registry_data.get("manifest_digest") != report_data.get("manifest_digest")
        or registry_data.get("report_sha256") != _sha256(report)
        or report_data.get("execution_source") != execution_source
    ):
        raise RuntimeError("claim registry does not bind the complete report")
    curie_campaign = repository / "results" / "curie-weiss-m5-foundations-v1"
    curie_report = curie_campaign / "report" / "campaign-report.json"
    curie_registry = curie_campaign / "report" / "claim-registry.json"
    curie_report_data = read_immutable_json(curie_report)
    curie_registry_data = read_immutable_json(curie_registry)
    if (
        curie_report_data.get("completed_runs") != 80
        or curie_registry_data.get("report_sha256") != _sha256(curie_report)
        or curie_registry_data.get("manifest_digest")
        != curie_report_data.get("manifest_digest")
    ):
        raise RuntimeError("Curie--Weiss registry does not bind all 80 fits")
    _copy(paper / "main.tex", stage / "main.tex")
    _copy(paper / "supplement.tex", stage / "supplement.tex")
    _copy(paper / "references.bib", stage / "references.bib")
    for name in GENERATED:
        _copy(paper / "generated" / name, stage / "generated" / name)
    for name in FIGURES:
        _copy(paper / "gaussian_figures" / name, stage / "gaussian_figures" / name)
    for name in CURIE_WEISS_FIGURES:
        _copy(
            paper / "curie_weiss_ancilla_figures" / name,
            stage / "curie_weiss_ancilla_figures" / name,
        )
    _copy(
        repository / "configs" / "gaussian_foundations_v1.json",
        stage / "evidence" / "gaussian_foundations_v1.json",
    )
    _copy(report, stage / "evidence" / "complete-report.json")
    _copy(claim_registry, stage / "evidence" / "claim-registry.json")
    _copy(
        campaign / "evidence" / "execution-source.json",
        stage / "evidence" / "execution-source.json",
    )
    _copy(
        repository / "configs" / "curie_weiss_m5_foundations_v1.json",
        stage / "evidence" / "curie_weiss_m5_foundations_v1.json",
    )
    _copy(curie_report, stage / "evidence" / "curie-weiss-report.json")
    _copy(curie_registry, stage / "evidence" / "curie-weiss-claims.json")
    _copy(
        curie_campaign / "store" / "preflight.json",
        stage / "evidence" / "curie-weiss-runtime-preflight.json",
    )
    class_path = subprocess.run(
        ["kpsewhich", "quantumarticle.cls"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _copy(Path(class_path), stage / "quantumarticle.cls")
    _check_forbidden(stage)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path.cwd().resolve()
    output = repository / args.output
    with tempfile.TemporaryDirectory(prefix="conditional-iqp-arxiv-") as directory:
        stage = Path(directory) / "source"
        stage.mkdir()
        _populate(repository, stage, args)
        pdf = _verify_compile(stage)
        verified_pdf = Path(directory) / "verified-main.pdf"
        shutil.copyfile(pdf, verified_pdf)
        _clean_build_products(stage)
        manifest = _source_manifest(stage)
        (stage / "SOURCE_SHA256.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        archive = output / "arxiv" / "conditional-iqp-cw-ancilla-v1-source.tar.gz"
        archive_sha = _write_deterministic_archive(stage, archive)
        with tempfile.TemporaryDirectory(prefix="gaussian-arxiv-check-") as check_dir:
            check = Path(check_dir)
            with tarfile.open(archive, "r:gz") as handle:
                handle.extractall(check, filter="data")
            expected = json.loads((check / "SOURCE_SHA256.json").read_text())
            actual = _source_manifest(check)
            if expected != actual:
                raise RuntimeError("arXiv source archive checksum verification failed")
            _check_forbidden(check)
            extracted_pdf = _verify_compile(check)
            if _sha256(extracted_pdf) != _sha256(verified_pdf):
                raise RuntimeError("clean manuscript builds are not deterministic")
        canonical = output / "pdf" / "conditional_iqp_generation.pdf"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = output / "arxiv" / "conditional-iqp-cw-ancilla-v1-release.json"
        if metadata_path.exists():
            if (
                not canonical.is_file()
                or canonical.read_bytes() != verified_pdf.read_bytes()
            ):
                raise RuntimeError("released canonical Gaussian PDF differs")
        else:
            if canonical.is_file():
                historical = (
                    output
                    / "history"
                    / f"conditional_iqp_generation-{_sha256(canonical)[:16]}.pdf"
                )
                historical.parent.mkdir(parents=True, exist_ok=True)
                if not historical.exists():
                    shutil.copyfile(canonical, historical)
            shutil.copyfile(verified_pdf, canonical)
        metadata = {
            "schema_version": 1,
            "campaign_version": (
                "gaussian_foundations_v1+curie_weiss_m5_foundations_v1"
            ),
            "completed_atomic_models": 1_100,
            "gaussian_models": 1_020,
            "curie_weiss_models": 80,
            "canonical_pdf_sha256": _sha256(canonical),
            "source_archive_sha256": archive_sha,
            "upload_ready": False,
            "upload_blockers": [
                "verified author names",
                "verified affiliations",
                "author contributions",
                "competing interests",
                "arXiv category and license",
                "final human proof approval",
            ],
            "submission_performed": False,
        }
        encoded_metadata = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        if metadata_path.exists():
            if metadata_path.read_text() != encoded_metadata:
                raise RuntimeError("released Gaussian metadata differs")
        else:
            metadata_path.write_text(encoded_metadata)
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
