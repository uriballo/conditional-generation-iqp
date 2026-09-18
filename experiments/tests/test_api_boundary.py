"""Experiments may use only the public library API.

The rule keeps library development decoupled from experiment development: if a
study needs behaviour the public API does not offer, the fix belongs in the
library repository.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"\biqp_cond\.(?:_|src\b)|from\s+iqp_cond\s+import\s+_")


def _experiment_sources() -> list[Path]:
    return [path for path in EXPERIMENTS.rglob("*.py") if "tests" not in path.parts]


def test_no_private_library_imports() -> None:
    for path in _experiment_sources():
        source = path.read_text()
        ast.parse(source)
        assert not FORBIDDEN.search(source), (
            f"{path.relative_to(EXPERIMENTS)} reaches into library internals"
        )


def test_no_campaign_tree_imports() -> None:
    for path in _experiment_sources():
        source = path.read_text()
        assert "sys.path" not in source, (
            f"{path.relative_to(EXPERIMENTS)} manipulates sys.path"
        )
