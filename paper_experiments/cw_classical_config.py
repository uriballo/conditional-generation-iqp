"""Accepted classical comparison, separate from frozen IQP campaigns."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from .cw_followup_config import limits as parent_limits
from .cw_followup_config import protocol as parent_protocol
from .cw_followup_config import stage_betas, stream_record

FAMILIES = ("rbm", "transformer", "vae")
CAPACITIES = ("small", "large")
PROTOCOL_PATH = Path(__file__).with_name("cw_classical_protocol.json")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def protocol(smoke=False):
    value = json.loads(PROTOCOL_PATH.read_text())
    recorded = value.pop("protocol_sha256")
    if digest(value) != recorded:
        raise ValueError("classical protocol hash mismatch")
    parent = parent_protocol()
    if value["parent_protocol_sha256"] != parent["protocol_sha256"]:
        raise ValueError("classical parent protocol differs")
    value["protocol_sha256"] = recorded
    value["smoke"] = bool(smoke)
    return value


def model(family):
    if family not in FAMILIES:
        raise ValueError("unknown classical family")
    return importlib.import_module(f"paper_experiments.cw_classical_{family}")


def limits(smoke=False):
    if smoke:
        return dict(
            n=4,
            n_train=8,
            batch_size=4,
            steps=2,
            checkpoint_interval=1,
            trials=3,
            repeats={"screen": 1, "confirmation": 1, "final": 2},
            starts={"screen": 1, "confirmation": 1, "final": 1},
            sampling_draws=64,
        )
    parent = parent_limits(parent_protocol())
    return dict(
        n=16,
        n_train=parent["n_train"],
        batch_size=parent["batch_size"],
        steps=parent["steps"],
        checkpoint_interval=parent["validation_interval"],
        trials=parent["trials"],
        repeats=parent["repeats"],
        starts=parent["starts"],
        sampling_draws=4096,
    )


def betas(stage, smoke=False, *, selection=False):
    if stage not in ("screen", "confirmation", "final"):
        raise ValueError("unknown fit stage")
    if smoke:
        a = (0.29, 0.89, 1.61)
        if selection or stage == "screen":
            return a
        if stage == "confirmation":
            return tuple(sorted((*a, 0.65, 1.25)))
        return (0.0, 0.4, 1.0, 2.0)
    return stage_betas(parent_protocol(), "screen" if selection else stage)


def candidate(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.0003, 0.1, log=True),
        init_scale=trial.suggest_float("init_scale", 0.1, 3.0, log=True),
        b1=trial.suggest_float("b1", 0.8, 0.95),
        b2=1 - trial.suggest_float("one_minus_beta2", 0.001, 0.05, log=True),
        schedule=trial.suggest_categorical("schedule", ["constant", "cosine"]),
    )


def classical_stream(*keys):
    value = {"keys": ["cw-classical-20261014-v1", *keys]}
    value["sha256"] = digest(value["keys"])
    value["seed"] = int(value["sha256"][:8], 16)
    return value


def target_stream(stage, repeat, beta):
    """Reproduce the IQP target pools, including their original namespace."""
    return stream_record(stage, repeat, str(float(beta)), "target_pool")


def seed_manifest(smoke=False):
    budget = limits(smoke)
    records = {}

    def add(record):
        records[record["sha256"]] = record

    for stage in ("screen", "confirmation", "final"):
        for repeat in range(budget["repeats"][stage]):
            for beta in protocol()["training"]["shared_anchors"]:
                add(target_stream(stage, repeat, beta))
            for start in range(budget["starts"][stage]):
                for purpose in ("initialization", "batches", "conditions", "latent"):
                    add(classical_stream(stage, repeat, start, purpose))
    for family in FAMILIES:
        for capacity in CAPACITIES:
            add(classical_stream("sampler", family, capacity, "random", 0))
            for number in range(budget["trials"], 2 * budget["trials"]):
                add(classical_stream("sampler", family, capacity, "tpe", number))
            for repeat in range(budget["repeats"]["final"]):
                add(classical_stream("final", repeat, family, capacity, "samples"))
    add(classical_stream("report", "paired-bootstrap"))
    occupied = {}
    for record in records.values():
        if record["seed"] in occupied:
            raise ValueError("unintended classical uint32 seed collision")
        occupied[record["seed"]] = record["sha256"]
    return dict(
        streams=list(records.values()),
        backend_seed_bits=32,
        intentional_reuse=(
            "target pools paired with IQP by stage/repeat/anchor; classical "
            "initialization/batch/anchor/latent keys shared across candidates "
            "and families/capacities, with model-dependent interpretation"
        ),
    )
