"""Generate the standalone Ashkin--Teller appendix tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = (
    ("cat_encoder_only", "Cat input"),
    ("cat", "Cat IQP"),
    ("bond_encoder_only", "Bond input"),
    ("bond", "Bond IQP"),
    ("log_composition", "Log composition"),
    ("log_linear", "Log-linear"),
)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _row(condition: list[float], values: list[float]) -> str:
    a, b = condition
    return f"({a:g},{b:g}) & {' & '.join(_fmt(x) for x in values)} " + "\\\\"


def _table(data: dict, size: int) -> str:
    conditions = data["test_conditions"]
    models = data["models"]
    tvd = [
        [models[key]["test"]["tvd"]["condition_means"][i] for key, _ in MODELS]
        for i in range(9)
    ]
    corr = [
        [
            models[key]["test"]["correlation_rmse"]["condition_means"][i]
            for key, _ in MODELS
        ]
        for i in range(9)
    ]
    lines = [
        "\\begin{plainwidetext}",
        "\\noindent\\begin{minipage}{\\textwidth}",
        "\\refstepcounter{table}",
        "\\centering",
        "\\noindent\\parbox{\\textwidth}{\\small Table~\\thetable: "
        f"Ashkin--Teller $n={size}$ test-condition means over ten seeds. "
        "Exact TVD and connected-correlation RMSE are reported for each interior "
        "condition; lower is better. Input controls are deterministic.}",
        f"\\label{{tab:at-conditions-n{size}}}",
        "\\par\\vspace{6pt}\\small",
        "\\begin{tabular}{@{}c" + "r" * 6 + "@{}}",
        "\\toprule",
        "$(a,b)$ & Cat input & Cat IQP & Bond input & Bond IQP & Log composition & "
        "Log-linear \\\\",
        "\\midrule",
        "\\multicolumn{7}{@{}l}{\\emph{Joint TVD}} \\\\",
    ]
    lines += [
        _row(condition, row) for condition, row in zip(conditions, tvd, strict=True)
    ]
    lines += [
        "\\midrule",
        "\\multicolumn{7}{@{}l}{\\emph{Connected-correlation RMSE}} \\\\",
    ]
    lines += [
        _row(condition, row) for condition, row in zip(conditions, corr, strict=True)
    ]
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}",
        "\\end{plainwidetext}",
        "",
    ]
    return "\n".join(lines)


def generate(n8: Path, n12: Path, output: Path) -> None:
    d8 = json.loads(n8.read_text())
    d12 = json.loads(n12.read_text())
    for data in (d8, d12):
        assert len(data["test_conditions"]) == 9
        assert all(
            len(data["models"][key]["test"]["tvd"]["condition_means"]) == 9
            for key, _ in MODELS
        )
        assert all(
            len(data["models"][key]["test"]["correlation_rmse"]["condition_means"]) == 9
            for key, _ in MODELS
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_table(d8, 8) + "\n\\clearpage\n" + _table(d12, 12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n8",
        type=Path,
        default=Path("docs/research/ashkin-teller-confirmation-result-2026-09-07.json"),
    )
    parser.add_argument(
        "--n12",
        type=Path,
        default=Path("docs/research/ashkin-teller-n12-result-2026-09-07.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("Overleaf/ashkin_teller_tables.tex")
    )
    args = parser.parse_args()
    generate(args.n8, args.n12, args.output)


if __name__ == "__main__":
    main()
