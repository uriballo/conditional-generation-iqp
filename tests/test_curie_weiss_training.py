import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.curie_weiss_protocol import CurieWeissFamily
from iqp_cond.curie_weiss_training import parameter_expansion, symmetry_expansion


def test_two_coefficients_expand_to_the_complete_compiled_generator_sequence():
    ansatz = complete_ansatz(8)
    expansion = symmetry_expansion(np.asarray(ansatz.generators))
    assert expansion.shape == (36, 2)
    params = expansion @ np.asarray([0.2, -0.3])
    assert np.all(params[:8] == 0.2)
    assert np.all(params[8:] == -0.3)
    assert np.array_equal(ansatz.generators, complete_ansatz(8).generators)


def test_unrestricted_ablation_has_separate_full_parameter_identity():
    ansatz = complete_ansatz(12)
    tied = parameter_expansion(ansatz, CurieWeissFamily.CONDITIONED)
    full = parameter_expansion(ansatz, CurieWeissFamily.UNRESTRICTED_CONDITIONED)
    assert tied.shape == (78, 2)
    assert full.shape == (78, 78)
