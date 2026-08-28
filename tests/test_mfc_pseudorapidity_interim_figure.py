import numpy as np

from scripts.plot_mfc_pseudorapidity_interim import summarize


def _run(energy, family, seed):
    target = np.arange(1, 17, dtype=float)
    target /= target.sum()
    return {
        "run": {
            "campaign_version": "mfc_pseudorapidity_confirmatory_v1",
            "bin_count": 16,
            "coding": "binary",
            "held_out_energy_gev": energy,
            "family": family,
            "seed": seed,
        },
        "target_physical_probabilities": target.tolist(),
        "prediction_physical_probabilities": (np.ones(16) / 16).tolist(),
    }


def test_missing_panel_is_absent_and_partial_counts_are_preserved():
    result = summarize(
        [
            _run(75, "conditioned_iqp", 100),
            _run(75, "conditioned_iqp", 101),
            _run(100, "no_condition_iqp", 100),
        ]
    )
    assert 150 not in result
    assert result[75]["models"]["conditioned_iqp"][0] == 2
    assert result[100]["models"]["no_condition_iqp"][0] == 1


def test_all_families_and_seedless_histogram_have_tvd_summaries():
    families = (
        "conditioned_iqp",
        "no_condition_iqp",
        "separate_iqp",
        "categorical_mlp",
        "histogram_interpolation",
    )
    result = summarize(
        [
            _run(125, family, None if family == "histogram_interpolation" else seed)
            for seed, family in enumerate(families, start=100)
        ]
    )
    assert set(result[125]["models"]) == set(families)
    assert set(result[125]["tvd"]) == set(families)
    assert result[125]["models"]["histogram_interpolation"][0] == 1
    assert result[125]["tvd"]["histogram_interpolation"][0] == 1
