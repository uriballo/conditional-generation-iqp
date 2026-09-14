#!/usr/bin/env python3
"""Training-condition-only engineering gate; no held-out target evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import jax
import numpy as np
from ashkin_teller_data_efficiency import TRAIN, arm_initial, draw_data
from ashkin_teller_data_efficiency_model import generators, sampled_losses

from iqp_cond.ashkin_teller_size_baselines import fit_baselines
from iqp_cond.exact_tvd_training import CampaignTrainingConfig, train_campaign


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=("cat", "bond", "classical"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    training, validation = draw_data(4700)
    start = time.monotonic()
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "devices": [str(x) for x in jax.devices()],
        "seed": 4700,
        "production_inclusion": False,
        "held_out_targets_accessed": False,
    }
    if a.encoder == "classical":
        records = []
        for size in (100, 300, 1000):
            t = time.monotonic()
            q, meta = fit_baselines(TRAIN, training[:, :size], validation, TRAIN)
            assert all(np.isfinite(v).all() for v in q.values())
            assert all(
                c["success"] and c["grad_inf"] <= 1e-6 for c in meta["convergence"]
            )
            records.append(
                {"size": size, "seconds": time.monotonic() - t, "metadata": meta}
            )
            print("CLASSICAL SIZE", size, "COMPLETE", flush=True)
        info["records"] = records
    else:
        g = generators(4)
        initial = arm_initial(training[:, :300], g, 4700)
        loss, val, bank = sampled_losses(g, validation, 4700, a.encoder)
        val_times = []

        def check(params):
            v = float(val(params))
            assert np.isfinite(v)
            val_times.append(time.monotonic() - start)
            print(
                "VALIDATION", len(val_times) - 1, "seconds", val_times[-1], flush=True
            )
            return v

        r = train_campaign(
            initial,
            training[:, :300],
            check,
            loss,
            4700,
            CampaignTrainingConfig(
                steps=50, batch_size=100, n_samples=2048, validation_interval=25
            ),
            include_initial_checkpoint=True,
        )
        assert r.train_losses.shape == (50,) and r.batches.shape == (50, 16, 100)
        assert np.isfinite(r.train_losses).all() and np.isfinite(r.final_params).all()
        assert np.max(r.batches) < 300 and np.min(r.batches) >= 0
        np.savez_compressed(
            a.output / (a.encoder + ".npz"),
            initial=initial,
            final=r.final_params,
            train_losses=r.train_losses,
            validation_losses=r.validation_losses,
            batches=r.batches,
            **bank,
        )
        info.update(
            {
                "encoder": a.encoder,
                "train_size": 300,
                "n_ancillas": 4,
                "steps": 50,
                "validation_times": val_times,
                "steady_seconds_per_update": (val_times[2] - val_times[1]) / 25,
                "estimated_500_update_seconds": val_times[1]
                + 19 * (val_times[2] - val_times[1]),
            }
        )
    info["seconds"] = time.monotonic() - start
    info["status"] = "Complete"
    (a.output / (a.encoder + ".json")).write_text(json.dumps(info, indent=2) + "\n")
    print("COMPLETE", a.encoder, info["seconds"], flush=True)


if __name__ == "__main__":
    main()
