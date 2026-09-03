#!/usr/bin/env python3
"""Run the bounded n=8 Curie encoder-ablation selection overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iqp_cond.curie_encoder_ablation import (
    AblationIdentity,
    AblationManifest,
    evaluate_run,
    plan_runs,
    train_run,
)
from iqp_cond.protocol import ProtocolValidationError

DEFAULT_CONFIG = Path("configs/curie_encoder_ablation_n8_screen_v1.json")
SCREEN_VERSION = "curie_encoder_ablation_n8_screen_v1"
PARENT_MANIFEST = "configs/curie_encoder_ablation_v1.json"
VISIBLE_QUBITS = (8,)
ANCILLAS = (0, 10)
SEEDS = (100, 101, 102)
ARMS = (
    "product_generic_affine",
    "spin_flip_generic_affine",
    "product_informed_piecewise",
    "spin_flip_informed_piecewise",
)


class Screen:
    """Strict selection overlay on the frozen 240-run parent campaign."""

    def __init__(self, path: Path = DEFAULT_CONFIG):
        self.path = path.resolve()
        try:
            self.raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError("invalid screen manifest") from error
        expected_keys = {
            "schema_version",
            "campaign_version",
            "parent_manifest",
            "parent_manifest_digest",
            "visible_qubits",
            "ancillas",
            "seeds",
            "arms",
            "artifacts",
        }
        if set(self.raw) != expected_keys:
            raise ProtocolValidationError("screen manifest schema differs")
        if (
            self.raw["schema_version"] != 1
            or self.raw["campaign_version"] != SCREEN_VERSION
        ):
            raise ProtocolValidationError("screen manifest identity differs")
        if (
            tuple(self.raw["visible_qubits"]) != VISIBLE_QUBITS
            or tuple(self.raw["ancillas"]) != ANCILLAS
            or tuple(self.raw["seeds"]) != SEEDS
        ):
            raise ProtocolValidationError("screen coordinates differ")
        if tuple(self.raw["arms"]) != ARMS or self.raw["artifacts"] != {
            "root": "results/curie-encoder-ablation-v1",
            "immutable": True,
        }:
            raise ProtocolValidationError("screen arms or artifact protocol differs")
        if self.raw["parent_manifest"] != PARENT_MANIFEST:
            raise ProtocolValidationError("unexpected parent manifest")
        parent_path = (self.repository / self.raw["parent_manifest"]).resolve()
        self.parent = AblationManifest.from_json(parent_path)
        if self.raw["parent_manifest_digest"] != self.parent.digest:
            raise ProtocolValidationError("parent manifest digest differs")

    @property
    def repository(self) -> Path:
        return self.path.parent.parent

    @property
    def output_root(self) -> Path:
        return self.repository / self.raw["artifacts"]["root"]

    @property
    def parent_digest(self) -> str:
        return self.parent.digest

    def runs(self) -> tuple[AblationIdentity, ...]:
        return tuple(
            run
            for run in plan_runs(self.parent)
            if run.visible_qubits in VISIBLE_QUBITS
            and run.ancillas in ANCILLAS
            and run.seed in SEEDS
            and run.arm in ARMS
        )

    def plan(self) -> list[dict[str, Any]]:
        return [
            {"index": index, **run.to_dict(), "run_id": run.run_id}
            for index, run in enumerate(self.runs())
        ]


def write(payload: Any, target: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target is None:
        print(text, end="")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def _select_run(
    runs: tuple[AblationIdentity, ...], index: int | None, run_id: str | None
) -> AblationIdentity:
    if (index is None) == (run_id is None):
        raise ValueError("select exactly one of --index or --run-id")
    if index is not None:
        if not 0 <= index < len(runs):
            raise ValueError("run index is outside the screen")
        return runs[index]
    selected = next((run for run in runs if run.run_id == run_id), None)
    if selected is None:
        raise ValueError("run ID is outside the screen")
    return selected


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("plan", "status", "train", "evaluate"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--index", type=int)
    p.add_argument("--run-id")
    p.add_argument("--json-output", type=Path)
    a = p.parse_args(argv)
    screen = Screen(a.config)
    runs = screen.runs()
    if a.command == "plan":
        payload: Any = {
            "campaign": screen.raw["campaign_version"],
            "parent_manifest_digest": screen.parent_digest,
            "count": len(runs),
            "runs": screen.plan(),
        }
    elif a.command == "status":
        run_ids = {run.run_id for run in runs}
        training_directory = screen.output_root / "training"
        evaluation_directory = screen.output_root / "evaluations"
        trained = (
            {path.stem for path in training_directory.glob("*.json")}
            if training_directory.is_dir()
            else set()
        )
        evaluated = (
            {path.stem for path in evaluation_directory.glob("*.json")}
            if evaluation_directory.is_dir()
            else set()
        )
        payload = {
            "campaign": screen.raw["campaign_version"],
            "planned": len(run_ids),
            "trained": len(run_ids & trained),
            "evaluated": len(run_ids & evaluated),
        }
    else:
        selected = _select_run(runs, a.index, a.run_id)
        operation = train_run if a.command == "train" else evaluate_run
        payload = [
            operation(
                screen.parent,
                selected,
                output_root=screen.output_root,
            )
        ]
    write(payload, a.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
