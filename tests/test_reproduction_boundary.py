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
                assert not module.startswith(("experiments", "iqp_cond.exact")), (
                    path,
                    module,
                )
                assert not any(
                    part.startswith("_")
                    for part in module.split(".")
                    if module.startswith("iqp_cond")
                ), (path, module)


def test_paper_source_uses_only_retained_figures():
    root = Path(__file__).resolve().parents[1]
    source = (root / "paper" / "main.tex").read_text()
    assert "curie_weiss_ancilla_tvd" not in source
    assert "136+16r" not in source
    for filename in [
        "conditional_iqp_workflow.pdf",
        "gaussian_translation_profiles.pdf",
        "curie_weiss_model_comparison.pdf",
        "curie_weiss_lattice_scaling_mmd.pdf",
    ]:
        assert filename in source
        assert (root / "paper" / "figures" / filename).is_file()
