"""Protocol checks for the product encoded lattice scaling campaign."""
import iqp_cond as ciqp
import numpy as np

from paper_experiments.ansatz import rectangular_lattice_ansatz
from paper_experiments.campaign import inventory
from paper_experiments.phase_encoder import BroadcastPhaseEncoder
from paper_experiments.runtime import (
    SCALING_SEEDS,
    SCALING_SIZES,
    build_job,
    config_for,
)
from paper_experiments.studies import build_study


def test_scaling_inventory_and_protocol_are_preserved():
    jobs = inventory("scaling")
    assert len(jobs) == len(SCALING_SIZES) * len(SCALING_SEEDS)
    expected = {(n, s) for n in SCALING_SIZES for s in SCALING_SEEDS}
    assert {(job["size"], job["seed"]) for job in jobs} == expected
    config = config_for(build_job("scaling", 100, size=64))
    assert config.encoder == "phase_075"
    assert (config.steps, config.n_ops, config.mc_samples) == (5000, 512, 1024)
    assert config.bodyness_targets == (2.0, 4.0)

def test_scaling_builds_rectangular_untied_lattice_with_product_encoder():
    job = build_job("scaling", 100, size=64)
    config = config_for(job)
    study = build_study(job, config)
    model = ciqp.ConditionalModel(
        study.make_ansatz(config), study.make_encoder(np.asarray([0.4]), config)
    )
    assert isinstance(model.encoder, BroadcastPhaseEncoder)
    assert (
        model.ansatz.generators.shape == rectangular_lattice_ansatz(64).generators.shape
    )
    assert model.ansatz.n_parameters == 64 + 112
