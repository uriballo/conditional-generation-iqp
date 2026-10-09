"""Configuration helpers for the accepted parity CW follow-up."""

from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

FAMILIES = (
    "shared_parity",
    "shared_product",
    "independent_parity",
    "independent_unconditional",
)
MARKERS = (0.4, 1.0, 2.0)
C_STAR = 0.8231097389772882
_MASTER_SEED = 2026100801
_PAIRS = tuple(f"{i}:{j}" for i in range(9) for j in range(i, 9))
_GRID = (4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 7.99)
_PROTOCOL = Path(__file__).with_name("cw_followup_protocol.json")


def _canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()


def protocol(smoke=False):
    value = json.loads(_PROTOCOL.read_text())
    digest = value.pop("protocol_sha256", None)
    actual = hashlib.sha256(_canonical(value)).hexdigest()
    if digest != actual or actual != (
        "135330d93182535afa2eb8cfc933340e3e86f64ea57f9e28d366cdde9127f02d"
    ):
        raise ValueError("protocol hash mismatch")
    value["protocol_sha256"] = digest
    if smoke:
        value["runtime"] = dict(
            steps=2,
            validation_interval=1,
            n_train=8,
            batch_size=4,
            n_ops=8,
            mc_samples=8,
            repeats={"screen": 1, "confirmation": 1, "final": 1},
            starts={"screen": 1, "confirmation": 1, "final": 1},
            trials=3,
            startup_trials=3,
        )
    return copy.deepcopy(value)


def limits(spec):
    if "runtime" in spec:
        return copy.deepcopy(spec["runtime"])
    return dict(
        steps=2000,
        validation_interval=100,
        n_train=1024,
        batch_size=256,
        n_ops=2048,
        mc_samples=1024,
        trials=60,
        startup_trials=60,
        repeats={"screen": 5, "confirmation": 5, "final": 10},
        starts={"screen": 1, "confirmation": 3, "final": 3},
    )


def candidate(trial, family):
    if family not in FAMILIES:
        raise ValueError("unknown follow-up family")
    r = {
        "learning_rate": trial.suggest_float("learning_rate", 0.0003, 0.1, log=True),
        "init_scale": trial.suggest_float("init_scale", 0.01, 1.0, log=True),
        "b1": trial.suggest_float("b1", 0.8, 0.95),
        "b2": 1 - trial.suggest_float("one_minus_beta2", 0.001, 0.05, log=True),
        "schedule": trial.suggest_categorical("schedule", ["constant", "cosine"]),
    }
    pair = trial.suggest_categorical("kernel_pair", list(_PAIRS))
    i, j = (int(x) for x in pair.split(":"))
    r["expected_weights"] = (2, _GRID[i], _GRID[j])
    if family != "independent_unconditional":
        r["beta_scale"] = trial.suggest_float("beta_scale", 0.25, 1.5, log=True)
    return r


def structured_seed(*keys):
    return int.from_bytes(
        hashlib.sha256(
            _canonical([_MASTER_SEED, "cw-parity-hpo-followup-20261009", *keys])
        ).digest()[:4],
        "big",
    )


def stream_record(*keys):
    payload = [_MASTER_SEED, "cw-parity-hpo-followup-20261009", *keys]
    d = hashlib.sha256(_canonical(payload)).hexdigest()
    return {
        "campaign_id": "cw-parity-hpo-followup-20261009",
        "keys": payload,
        "sha256": d,
        "seed": int(d[:8], 16),
    }


def stage_betas(spec, stage, case=None):
    t = spec["temperature_sets"]
    a_grid = tuple(float(Fraction(x)) for x in t["A_exact"])
    b_grid = tuple(float(Fraction(x)) for x in t["B_exact"])
    if stage not in ("screen", "confirmation", "final"):
        raise ValueError(f"unknown stage: {stage}")
    if isinstance(case, (int, float)):
        return (float(case),)
    if case and case not in ("independent", "shared"):
        try:
            return (float(case),)
        except ValueError:
            raise ValueError(f"unknown case: {case}") from None
    return tuple(
        a_grid
        if stage == "screen"
        else tuple(sorted(set(a_grid + b_grid)))
        if stage == "confirmation"
        else tuple(float(Fraction(x)) for x in t["final_shared_grid_exact"])
    )


def selection_betas(spec, case=None):
    return (
        (float(case),)
        if isinstance(case, (int, float))
        else tuple(float(Fraction(x)) for x in spec["temperature_sets"]["A_exact"])
    )
