"""Reported Gaussian and lattice-scaling job inventories."""

import json

import pytest

from paper_experiments.campaign import inventory
from run_experiments import main


def test_reported_inventory():
    jobs = inventory()
    assert len(jobs) == 23
    assert len({j["job_id"] for j in jobs}) == 23
    assert {j["experiment"] for j in jobs} == {"gaussian", "scaling"}
    assert len(inventory(smoke=True)) == 2


@pytest.mark.parametrize("experiment", ["curie-weiss", "symmetry", "unknown"])
def test_retired_experiments_are_rejected(experiment):
    with pytest.raises(ValueError, match="unsupported"):
        inventory(experiment)


def test_dry_run_writes_nothing(tmp_path, capsys):
    assert (
        main(["--experiment", "scaling", "--dry-run", "--output", str(tmp_path)]) == 0
    )
    assert not list(tmp_path.iterdir())
    assert len(json.loads(capsys.readouterr().out)) == 18


def test_gaussian_settings_are_preserved():
    jobs = inventory("gaussian")
    assert len(jobs) == 5
    assert {job["seed"] for job in jobs} == set(range(17100, 17105))
    config = jobs[0]["config"]
    for key, value in {
        "n_train": 512,
        "n_validation": 1000,
        "n_test": 1000,
        "steps": 1024,
        "batch_size": 128,
        "n_ops": 1024,
        "mc_samples": 1024,
        "validation_interval": 32,
        "validation_n_ops": 1024,
        "validation_mc_samples": 1024,
        "bodyness_targets": [1.0],
        "restarts": 1,
    }.items():
        assert config[key] == value


@pytest.mark.parametrize("experiment", ["all", "gaussian", "gaussian-lattice"])
def test_current_gaussian_cannot_silently_launch_historical_fits(tmp_path, experiment):
    with pytest.raises(SystemExit) as error:
        main(["--experiment", experiment, "--output", str(tmp_path)])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())
