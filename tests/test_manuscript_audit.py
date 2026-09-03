"""Checks for the canonical Overleaf manuscript package."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[1]
OVERLEAF = REPOSITORY / "Overleaf"
MAIN = OVERLEAF / "main.tex"
BIBLIOGRAPHY = OVERLEAF / "references.bib"


def _without_comments(source: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in source.splitlines())


def test_overleaf_package_contains_only_canonical_sources_and_assets():
    assert MAIN.is_file()
    assert BIBLIOGRAPHY.is_file()
    assert not (REPOSITORY / "paper").exists()
    forbidden_suffixes = {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log"}
    assert not any(path.suffix in forbidden_suffixes for path in OVERLEAF.rglob("*"))
    assert not (OVERLEAF / "main.pdf").exists()


def test_all_referenced_figures_exist():
    source = _without_comments(MAIN.read_text())
    references = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    assert references
    concrete = [name for name in references if "#" not in name]
    missing = [name for name in concrete if not (OVERLEAF / name).is_file()]
    assert not missing


def test_citation_keys_resolve():
    source = _without_comments(MAIN.read_text())
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", source, flags=re.DOTALL)
        for key in group.split(",")
    }
    defined = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", BIBLIOGRAPHY.read_text()))
    assert cited
    assert cited - defined <= {"???"}


def test_overleaf_package_compiles_in_a_clean_directory(tmp_path):
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        pytest.skip("latexmk is not installed")
    work = tmp_path / "Overleaf"
    shutil.copytree(OVERLEAF, work)
    completed = subprocess.run(
        [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "main.tex",
        ],
        cwd=work,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-8_000:]
    log = (work / "main.log").read_text(errors="replace")
    assert "LaTeX Error" not in log
    assert "undefined references" not in log.lower()
    assert (work / "main.pdf").is_file()
