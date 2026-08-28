"""Gaussian/Curie--Weiss manuscript scope, evidence, and build audits."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageStat

from iqp_cond.artifacts import read_immutable_json, sha256_file

REPOSITORY = Path(__file__).parents[1]
PAPER = REPOSITORY / "paper"
MAIN = PAPER / "main.tex"
SUPPLEMENT = PAPER / "supplement.tex"
BIBLIOGRAPHY = PAPER / "references.bib"
FOUNDATIONS_ARCHIVE = (
    REPOSITORY / "archive" / "arxiv-foundations-v1-internal" / "d6faacd5"
)
FAILURE_ARCHIVE = (
    REPOSITORY / "archive" / "quantum-campaign-v1-failure-path" / "cc94836e"
)
FORBIDDEN = (
    "ising campaign",
    "random-coupling ising",
    "theorem gate",
    "kill tree",
    "quantum-ready",
    "quantum readiness",
    "failure draft",
)


def _without_comments(source: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in source.splitlines())


def _environment(source: str, name: str) -> str:
    begin = f"\\begin{{{name}}}"
    end = f"\\end{{{name}}}"
    start = source.index(begin) + len(begin)
    return source[start : source.index(end, start)]


def test_prior_drafts_are_content_addressed_and_gaussian_manifest_is_separate():
    assert sha256_file(FAILURE_ARCHIVE / "conditional_iqp_generation.pdf") == (
        "cc94836e63bd4ecacbd59c66c119043df43f396e2def4c8bd6413f4db6807724"
    )
    assert sha256_file(FOUNDATIONS_ARCHIVE / "main.tex") == (
        "d6faacd5eb3ae6c6b20340ed5d4b67a5a17eba98d81b46652b829d1215fffbd5"
    )
    assert sha256_file(FOUNDATIONS_ARCHIVE / "supplement.tex") == (
        "64a1cb4b0825a994b77e01cef87f4b034782e4f9d2339ec6b7b2b04ffe139f2f"
    )
    assert (
        sha256_file(
            REPOSITORY
            / "output"
            / "arxiv"
            / "conditional_generation_classically_trainable_iqp.tar.gz"
        )
        == "3f4a5689a98a213887c7299456948c81953d141929ddfdb01536b7fbef9f1234"
    )
    assert (REPOSITORY / "configs" / "gaussian_foundations_v1.json").is_file()


def test_reused_gaussian_summaries_match_the_regeneration_evidence_exactly():
    evidence = read_immutable_json(
        REPOSITORY
        / "results"
        / "gaussian-foundations-v1"
        / "evidence"
        / "legacy-regeneration.json"
    )
    assert evidence["exact_agreement"] is True
    assert len(evidence["verified_summary_sha256"]) == 9
    for relative, expected in evidence["verified_summary_sha256"].items():
        assert sha256_file(REPOSITORY / relative) == expected


def test_gaussian_execution_dependency_closure_is_sealed():
    evidence = read_immutable_json(
        REPOSITORY
        / "results"
        / "gaussian-foundations-v1"
        / "evidence"
        / "execution-source.json"
    )
    assert evidence["manifest_digest"] == (
        "386ce98be93501f078d45b29e7fa498db1600c44d0e22f4483c3c8d5c7ec0d8c"
    )
    assert len(evidence["files"]) == 19
    snapshot = (
        REPOSITORY
        / "results"
        / "gaussian-foundations-v1"
        / "evidence"
        / "execution-source-files"
    )
    for relative, expected in evidence["files"].items():
        assert sha256_file(snapshot / relative) == expected
    assert evidence["preflight_source_tree_digest"]


def test_abstract_states_the_supported_contribution_and_scope():
    source = _without_comments(MAIN.read_text())
    abstract = re.sub(r"\s+", " ", _environment(source, "abstract"))
    words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", abstract)
    assert 180 <= len(words) <= 260
    assert "state-conditioned IQP Born machines" in abstract
    assert "classically tractable input state" in abstract
    assert "amplitude-ratio evaluation" in abstract
    assert "single diagonal unitary fixed across the family" in abstract
    assert "discretized Gaussian families" in abstract
    assert "held-out critical point" in abstract
    assert "fit Gaussian campaign" not in abstract
    assert "fit Curie--Weiss campaign" not in abstract
    assert "total-variation distance" not in abstract
    assert "Symmetry-matched inputs" not in abstract
    assert "quantum advantage" not in abstract.lower()


def test_title_scope_language_and_release_metadata_are_exact():
    source = re.sub(r"\s+", " ", MAIN.read_text())
    assert "\\title{Conditional generation with instantaneous quantum " in source
    sections = re.findall(r"\\section\{([^}]+)\}", MAIN.read_text())
    assert sections == [
        "Introduction",
        "Conditional IQP model",
        "Limits of conditional universality",
        "Conditional IQP training",
        "Experimental setup",
        "Results",
        "Discussion",
        "Conclusions",
    ]
    assert MAIN.read_text().count("fig:pipeline") >= 1
    assert "\\author{Oriol Balló-Gimbernat}" in source
    assert "\\email{oriol.ballo@eurecat.org}" in source
    assert "\\author{Emiliano Tolotti}" in source
    assert "\\author{Emiliano Tolotti}" in source


def test_new_manuscript_and_bibliography_exclude_prior_campaign_material():
    for path in (MAIN, SUPPLEMENT, BIBLIOGRAPHY):
        source = path.read_text().lower()
        for phrase in FORBIDDEN:
            assert phrase not in source, f"{phrase!r} found in {path.name}"


def test_citation_keys_resolve_and_supplement_contains_required_contracts():
    sources = "\n".join((MAIN.read_text(), SUPPLEMENT.read_text()))
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", sources, flags=re.DOTALL)
        for key in group.split(",")
    }
    defined = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", BIBLIOGRAPHY.read_text()))
    unresolved_todos = cited - defined
    assert cited and unresolved_todos <= {"???"}
    assert "\\bibliography{references}" in MAIN.read_text()
    for required in (
        "Analytical details",
        "Gaussian protocol and complete factorial summary",
        "Curie--Weiss protocol and results",
        "Exploratory symmetry screens",
        "Archived encoding comparison",
        "Finite estimator audit",
        "Negative and boundary studies",
        "Campaign integrity and reproduction",
    ):
        assert f"\\section{{{required}}}" in SUPPLEMENT.read_text()


def test_bibliography_is_focused_and_every_source_is_used():
    sources = "\n".join((MAIN.read_text(), SUPPLEMENT.read_text()))
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", sources, flags=re.DOTALL)
        for key in group.split(",")
    }
    defined = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", BIBLIOGRAPHY.read_text()))

    assert 12 <= len(defined) <= 30
    assert cited - {"???"} == defined
    assert {
        "kissConditionalBornMachine2022",
        "zengConditionalQCBM2023",
    } <= cited


def test_manuscript_compiles_and_renders_cleanly_in_temporary_tree(tmp_path):
    latexmk = shutil.which("latexmk")
    pdfinfo = shutil.which("pdfinfo")
    pdftoppm = shutil.which("pdftoppm")
    assert latexmk and pdfinfo and pdftoppm
    paper = tmp_path / "paper"
    paper.mkdir()
    for source in (MAIN, SUPPLEMENT, BIBLIOGRAPHY):
        shutil.copy2(source, paper / source.name)
    for directory in (
        "generated",
        "gaussian_figures",
        "curie_weiss_ancilla_figures",
        "symmetry_figures",
        "figures",
    ):
        source_directory = PAPER / directory
        if source_directory.is_dir():
            shutil.copytree(source_directory, paper / directory)
    completed = subprocess.run(
        [
            latexmk,
            "-gg",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "main.tex",
        ],
        cwd=paper,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-8_000:]
    log = (paper / "main.log").read_text(errors="replace")
    assert "LaTeX Error" not in log
    assert "undefined references" not in log.lower()
    undefined_citations = re.findall(
        r"Citation `([^']+)' on page \d+ undefined", log, re.IGNORECASE
    )
    assert set(undefined_citations) <= {"???"}
    assert "Overfull \\hbox" not in log and "Overfull \\vbox" not in log

    pdf = paper / "main.pdf"
    metadata = subprocess.run(
        [pdfinfo, str(pdf)], check=True, capture_output=True, text=True
    ).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)$", metadata, re.MULTILINE).group(1))
    assert 5 <= pages <= 60
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    subprocess.run(
        [pdftoppm, "-png", "-r", "50", str(pdf), str(rendered / "page")],
        check=True,
        capture_output=True,
    )
    images = sorted(rendered.glob("*.png"))
    assert len(images) == pages
    for image_path in images:
        image = Image.open(image_path).convert("L")
        statistics = ImageStat.Stat(image)
        assert statistics.extrema[0][0] < 245, f"blank page: {image_path.name}"
        assert statistics.mean[0] < 253.0, f"near-blank page: {image_path.name}"
