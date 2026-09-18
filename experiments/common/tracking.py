"""Run tracking: a local index now, MLflow planned.

Every run writes ``results/<study>/<run_id>/`` (see
:mod:`experiments.common.runner`).  ``append_index`` keeps a single JSON-lines
index at the results root for quick summaries, and :func:`log_mlflow` marks
where the planned MLflow integration goes: it will mirror each run directory
into an MLflow experiment (params from ``config.json``, metrics from
``metrics.json``, the checkpoint as an artifact) once ``mlflow`` is added as an
optional dependency.  Nothing here requires MLflow today.
"""

from __future__ import annotations

import json
from pathlib import Path

from .runner import RESULTS

__all__ = ["INDEX_NAME", "append_index", "load_index", "log_mlflow", "summarize"]

INDEX_NAME = "index.jsonl"


def _index_path(results_root: Path | None = None) -> Path:
    root = RESULTS if results_root is None else results_root
    return root / INDEX_NAME


def append_index(record: dict[str, object], results_root: Path | None = None) -> Path:
    """Append one run record to the local JSON-lines index."""
    path = _index_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {key: value for key, value in record.items() if key != "directory"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return path


def load_index(results_root: Path | None = None) -> list[dict[str, object]]:
    """Return every indexed run record."""
    path = _index_path(results_root)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(
    study: str | None = None,
    results_root: Path | None = None,
) -> list[dict[str, object]]:
    """Return one summary row per (study, encoder, n_visible, ancillas)."""
    grouped: dict[tuple, list[float]] = {}
    for record in load_index(results_root):
        if study is not None and record.get("study") != study:
            continue
        key = (
            record.get("study"),
            record.get("encoder"),
            record.get("n_visible"),
            record.get("n_ancillas"),
        )
        grouped.setdefault(key, []).append(float(record.get("held_out_tvd_mean")))
    rows = []
    for (name, encoder, n_visible, n_ancillas), values in sorted(grouped.items()):
        rows.append(
            {
                "study": name,
                "encoder": encoder,
                "n_visible": n_visible,
                "n_ancillas": n_ancillas,
                "runs": len(values),
                "held_out_tvd_mean": sum(values) / len(values),
            },
        )
    return rows


def log_mlflow(run_directory: Path, tracking_uri: str | None = None) -> None:
    """Mirror a run directory into MLflow (planned, not yet implemented).

    The integration will be a thin adapter over the run directory: params from
    ``config.json``, metrics from ``metrics.json``, and ``checkpoint.npz`` as
    an artifact, with ``tracking_uri`` defaulting to a local file store under
    ``results/mlruns``.
    """
    raise NotImplementedError(
        "MLflow tracking is planned; run directories and the JSON-lines index "
        "are the current interface",
    )
