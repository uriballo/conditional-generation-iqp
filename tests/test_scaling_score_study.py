import json

import pytest

from paper_experiments.scaling_score_study import DATA_DIR, analyze


def test_shipped_checkpoints_reproduce_reported_correlations():
    trials = json.loads((DATA_DIR / "checkpoints.json").read_text())
    result = analyze(trials)
    shipped = json.loads((DATA_DIR / "spearman.json").read_text())
    for subset, values in shipped.items():
        assert result[subset] == pytest.approx(values)
    assert result["all"]["configurations"] == 120
    assert result["all"]["checkpoints"] == 2520
    assert round(result["all"]["dev_score_vs_tvd"], 2) == 0.97
    assert round(result["all"]["train_mmd_vs_tvd"], 2) == 0.83
