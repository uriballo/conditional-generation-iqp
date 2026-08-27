import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.plot_all_datasets import BETAS, ROOT, figure, mfc_profiles, profiles


def test_curie_weiss_overview_grid_and_shells():
    _, curie = profiles()
    assert np.array_equal(BETAS, [0.5, 0.8, 1.0, 1.2, 1.5])
    assert curie.shape == (5, 17)
    assert np.allclose(curie.sum(axis=1), 1.0)
    assert np.all(curie >= 0)


def test_mfc_overview_is_sixteen_bin_normalized_profile():
    if not (ROOT / ".scratch" / "mfc-zenodo" / "50_GeV.pt").is_file():
        return
    mfc = mfc_profiles()
    assert mfc.shape == (5, 16)
    assert np.allclose(mfc.sum(axis=1), 1.0)
    assert np.all(mfc >= 0)


def test_overview_preserves_shared_probability_scale():
    if not (ROOT / ".scratch" / "mfc-zenodo" / "50_GeV.pt").is_file():
        return
    rendered = figure()
    try:
        assert all(axis.get_ylim() == (0.0, 0.3) for axis in rendered.axes)
    finally:
        rendered.clear()
