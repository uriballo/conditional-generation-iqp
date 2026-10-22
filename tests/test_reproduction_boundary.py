"""The paper runner must stand alone and use the published library API."""

import ast
from pathlib import Path


def test_paper_code_has_no_retired_research_dependencies():
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "run_experiments.py",
        *sorted((root / "paper_experiments").glob("*.py")),
    ]
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert not module.startswith(
                    ("experiments", "paper.", "iqp_cond.exact")
                ), (
                    path,
                    module,
                )
                assert not any(
                    part.startswith("_")
                    for part in module.split(".")
                    if module.startswith("iqp_cond")
                ), (path, module)


def test_data_holds_only_current_bundles():
    root = Path(__file__).resolve().parents[1] / "data"
    assert {path.name for path in root.iterdir() if path.is_dir()} == {
        "curie-weiss-classical",
        "curie-weiss-diagonal",
        "curie-weiss-followup",
        "curie-weiss-twisted",
        "gaussian2d-diagonal",
        "gaussian2d-diagonal-variants",
        "gaussian_lattice",
        "hardware",
        "scaling",
    }
