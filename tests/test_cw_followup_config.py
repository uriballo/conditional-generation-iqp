from paper_experiments.cw_followup_config import (
    FAMILIES,
    candidate,
    limits,
    protocol,
    selection_betas,
    stage_betas,
    stream_record,
)


class Trial:
    def __init__(self, values):
        self.values = iter(values)

    def suggest_float(self, *_args, **_kwargs):
        return next(self.values)

    def suggest_categorical(self, name, choices):
        return choices[0] if name == "schedule" else choices[17]


def test_protocol_hash_grids_and_runtime():
    spec = protocol()
    temps = spec["temperature_sets"]
    assert len(temps["A_exact"]) == 17
    assert len(temps["B_exact"]) == 20
    assert stage_betas(spec, "screen")[0] == 0.05
    assert stage_betas(spec, "screen")[-1] == 1.975
    assert len(stage_betas(spec, "confirmation")) == 37
    assert selection_betas(spec, "shared_parity")[0] == 0.05
    assert selection_betas(spec, 1.0) == (1.0,)

    smoke = protocol(True)["runtime"]
    assert smoke["steps"] == 2 and smoke["n_train"] == 8
    assert smoke["repeats"] == {"screen": 1, "confirmation": 1, "final": 1}
    assert limits(spec)["steps"] == 2000
    assert limits(spec)["trials"] == 60
    assert limits(spec)["repeats"] == {"screen": 5, "confirmation": 5, "final": 10}
    assert limits(spec)["starts"] == {"screen": 1, "confirmation": 3, "final": 3}


def test_candidate_has_exact_pair_domain_and_unconditional_omits_scale():
    values = [0.01, 0.1, 0.8, 0.99, 0.75]
    for family in FAMILIES:
        result = candidate(Trial(values.copy()), family)
        assert result["expected_weights"][0] == 2
        assert len(result["expected_weights"]) == 3
        if family == "independent_unconditional":
            assert "beta_scale" not in result
        else:
            assert "beta_scale" in result


def test_stream_namespace_is_distinct_and_reproducible():
    a = stream_record("screen", "shared_parity")
    b = stream_record("screen", "shared_parity")
    c = stream_record("screen", "old-campaign")
    assert a == b
    assert a["campaign_id"] == "cw-parity-hpo-followup-20261009"
    assert a["seed"] != c["seed"]
