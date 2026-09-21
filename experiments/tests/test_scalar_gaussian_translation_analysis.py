"""Focused tests for deterministic scalar benchmark analysis."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.scalar_gaussian_translation import analysis, campaign, protocol


def _distribution(condition: float) -> np.ndarray:
    return protocol.target_probabilities(np.asarray([condition]))[0]


def test_metric_row_reports_all_exact_metrics() -> None:
    target = _distribution(0.5)
    row = analysis.metric_row(target, target)
    assert set(row) == set(analysis.METRIC_KEYS)
    assert row["bitstring_tvd"] == 0
    assert row["hamming_profile_tvd"] == 0
    assert row["absolute_mean_error"] == 0
    assert row["absolute_std_error"] == 0


def test_classification_and_extraction_metadata() -> None:
    anchors = np.asarray([0.0, 0.5, 1.0])
    assert protocol.classify_condition(0.5, "interpolation", anchors) == "training"
    assert (
        protocol.classify_condition(0.25, "interpolation", anchors)
        == "held_out_interpolation"
    )
    assert (
        protocol.classify_condition(0.9, "near", np.asarray([0.0, 0.4, 0.8]))
        == "near_extrapolation"
    )


def test_anchor_interpolation_is_convex_and_holds_endpoints() -> None:
    anchors = np.asarray([0.0, 0.25, 0.5])
    distributions = np.stack([_distribution(c) for c in anchors])
    conditions = np.asarray([0.0, 0.125, 0.75, 0.9])
    rows = analysis.interpolation_rows(anchors, distributions, conditions, "far", "dev")
    assert rows[0]["bitstring_tvd"] == pytest.approx(0.0)
    assert rows[1]["generated_mean"] == pytest.approx(
        0.5
        * (
            analysis._moments(distributions[0])[0]
            + analysis._moments(distributions[1])[0]
        )
    )
    assert rows[-1]["generated_mean"] == pytest.approx(
        analysis._moments(distributions[-1])[0]
    )
    assert all(
        row["hyperparameter_id"] == "linear-convex-anchor-mixture" for row in rows
    )


def test_records_refuse_incomplete_job_set(tmp_path: Path) -> None:
    jobs = campaign.build_jobs("stage1")
    manifest = {
        "phase": "stage1",
        "split": "dev",
        "protocol_sha256": campaign._sha256(campaign.PROTOCOL_PATH),
        "source_file_sha256": campaign._source_snapshot()[0],
        "source_snapshot_sha256": campaign._source_snapshot()[1],
        "jobs": [campaign.asdict(j) for j in jobs],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing job records"):
        analysis.load_records(manifest_path, tmp_path / "records")


def test_record_loader_ignores_sibling_phase_records(tmp_path: Path) -> None:
    jobs = campaign.build_jobs("stage1")
    records = tmp_path / "job_records"
    records.mkdir()
    manifest = {
        "phase": "stage1",
        "split": "dev",
        "protocol_sha256": campaign._sha256(campaign.PROTOCOL_PATH),
        "source_file_sha256": campaign._source_snapshot()[0],
        "source_snapshot_sha256": campaign._source_snapshot()[1],
        "jobs": [campaign.asdict(job) for job in jobs],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (records / "other-phase-record.json").write_text("{}")
    with pytest.raises(ValueError, match="missing job records"):
        analysis.load_records(manifest_path, records)


def test_encoder_only_is_exactly_zero_iqp_baseline() -> None:
    rows = analysis.encoder_only_rows(np.asarray([0.2, 0.6]), "broadcast_affine", "dev")
    assert len(rows) == 2
    assert all(row["model_kind"] == "encoder_only" for row in rows)
    assert all(np.isfinite(row["bitstring_tvd"]) for row in rows)


def test_target_diagnostics_include_truncation_fields() -> None:
    payload = analysis.target_diagnostics(np.asarray([0.0, 0.5, 1.0]))
    assert len(payload["rows"]) == 3
    assert {
        "nominal_mean",
        "exponent_width",
        "actual_mean",
        "actual_std",
        "boundary_shell_mass",
    } <= set(payload["rows"][0])
    assert set(payload["translation_to_std_contraction_ratio"]) == {
        "near_extrapolation_0.8_to_1.0",
        "far_extrapolation_0.5_to_1.0",
    }


def test_stage2_selection_key_tiebreaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final deterministic tie break is compute, then candidate id."""
    # Exercise the frozen ordering independently of expensive checkpoint synthesis.
    ranking = [
        {
            "id": "z",
            "median_held_interpolation_tvd": 0.2,
            "worst_seed_held_mean_tvd": 0.3,
            "compute": 600,
        },
        {
            "id": "a",
            "median_held_interpolation_tvd": 0.2,
            "worst_seed_held_mean_tvd": 0.3,
            "compute": 400,
        },
    ]
    assert (
        sorted(
            ranking,
            key=lambda x: (
                x["median_held_interpolation_tvd"],
                x["worst_seed_held_mean_tvd"],
                x["compute"],
                x["id"],
            ),
        )[0]["id"]
        == "a"
    )


def test_aggregate_has_seed_and_variance_fields() -> None:
    rows = [
        {
            "encoder": "broadcast_affine",
            "protocol": "interpolation",
            "anchor_count": 3,
            "placement": "even",
            "layout_seed": None,
            "budget_mode": "fixed_total",
            "category": "held_out_interpolation",
            "optimizer_seed": seed,
            "bitstring_tvd": float(seed) / 10,
            **{key: 0.0 for key in analysis.METRIC_KEYS if key != "bitstring_tvd"},
        }
        for seed in (1, 2, 3)
    ]
    group = analysis.aggregate_rows(rows)["groups"][0]
    assert group["seeds"] == [1, 2, 3]
    assert group["metrics"]["bitstring_tvd"]["median"] == pytest.approx(0.2)
    assert "within_layout_row_variance" in group
    assert "layout_variance" not in group
    assert "optimizer_seed_variance" not in group


def test_stage1_summary_uses_no_sibling_audit_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    poisoned_audit = tmp_path / "audit.json"
    poisoned_audit.write_text("not json")

    def fake_rows(path, records_dir=None):
        assert path == manifest_path
        assert poisoned_audit.read_text() == "not json"
        return {"phase": "stage1", "split": "dev", "protocol_sha256": "test"}, []

    monkeypatch.setattr(analysis, "_dev_rows", fake_rows)
    with pytest.raises(ValueError, match="requires three dev seeds"):
        analysis.stage1_summary(manifest_path)


def test_frozen_gate_evaluator_uses_condition_matched_encoder_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    category_conditions = {
        "training": (0.0, 1.0),
        "held_out_interpolation": (0.25,),
        "near_extrapolation": (0.9,),
        "far_extrapolation": (0.95,),
    }
    for encoder in ("broadcast_affine", "qubit_affine", "anti_block"):
        for seed in (401, 402, 403):
            for category, conditions in category_conditions.items():
                for condition in conditions:
                    rows.append(
                        {
                            "encoder": encoder,
                            "category": category,
                            "optimizer_seed": seed,
                            "bitstring_tvd": 0.1,
                            "condition": condition,
                            "generated_mean": 1.0 + 5.0 * condition,
                            "target_mean": 1.0 + 5.0 * condition,
                            "absolute_std_error": 0.05,
                            "hamming_profile_tvd": 0.08,
                            "within_shell_residual": 0.02,
                            "protocol": "interpolation",
                            "anchor_count": 2,
                            "placement": "random",
                            "layout_seed": 1101,
                            "budget_mode": "fixed_total",
                        }
                    )

    def baseline(conditions, encoder, split):
        assert split == "audit"
        return [
            {"condition": float(condition), "bitstring_tvd": 0.5}
            for condition in conditions
        ]

    monkeypatch.setattr(analysis, "encoder_only_rows", baseline)
    category_caps = {
        category: {
            "median_seed_mean_tvd_at_most": 0.2,
            "worst_seed_mean_tvd_at_most": 0.2,
            "worst_condition_tvd_at_most": 0.2,
        }
        for category in category_conditions
    }
    gates = {
        "frozen": True,
        "criteria": {
            "categories": category_caps,
            "encoder_only_improvement": {
                "median_seed_mean_tvd_improvement_at_least": 0.2,
                "worst_seed_mean_tvd_improvement_at_least": 0.2,
            },
            "mean_movement": {
                "median_generated_to_target_ratio_at_least": 0.8,
                "worst_seed_median_ratio_at_least": 0.8,
                "correct_sign_fraction_at_least": 1.0,
            },
            "width": {
                "median_absolute_std_error_at_most": 0.1,
                "worst_seed_mean_absolute_std_error_at_most": 0.1,
            },
            "structure": {
                "median_hamming_profile_tvd_at_most": 0.1,
                "median_within_shell_residual_at_most": 0.05,
                "worst_seed_mean_hamming_profile_tvd_at_most": 0.1,
                "worst_seed_mean_within_shell_residual_at_most": 0.05,
            },
            "reliability": {
                "max_optimizer_seed_range_at_most": 0.05,
                "max_random_layout_median_tvd_at_most_by_category": {
                    category: 0.2 for category in category_conditions
                },
            },
        },
    }
    gate_path = tmp_path / "gates.json"
    gate_path.write_text(json.dumps(gates))
    result = analysis.evaluate_gates(rows, gate_path)
    assert result["encoders"]["broadcast_affine"]["pass"] is True
    assert result["encoders"]["qubit_affine"]["pass"] is True
    assert result["encoders"]["anti_block"]["pass"] is None


def _gate_fixture_rows() -> list[dict[str, object]]:
    rows = []
    for encoder in ("broadcast_affine", "qubit_affine", "anti_block"):
        for seed in (401, 402, 403):
            for category, conditions in {
                "training": (0.0, 1.0),
                "held_out_interpolation": (0.25,),
                "near_extrapolation": (0.9,),
                "far_extrapolation": (0.95,),
            }.items():
                for condition in conditions:
                    rows.append(
                        {
                            "encoder": encoder,
                            "category": category,
                            "optimizer_seed": seed,
                            "bitstring_tvd": 0.1,
                            "condition": condition,
                            "generated_mean": 1.0 + 5.0 * condition,
                            "target_mean": 1.0 + 5.0 * condition,
                            "absolute_std_error": 0.05,
                            "hamming_profile_tvd": 0.08,
                            "within_shell_residual": 0.02,
                            "protocol": "interpolation",
                            "anchor_count": 2,
                            "placement": "even",
                            "layout_seed": None,
                            "budget_mode": "fixed_total",
                            "hyperparameter_id": "hp",
                            "model_kind": "conditional",
                        }
                    )
    return rows


def _permissive_gates(path: Path, *, random_layout_cap: float = 0.25) -> None:
    categories = (
        "training",
        "held_out_interpolation",
        "near_extrapolation",
        "far_extrapolation",
    )
    path.write_text(
        json.dumps(
            {
                "frozen": True,
                "audit_seeds": [401, 402, 403],
                "criteria": {
                    "categories": {
                        category: {
                            "median_seed_mean_tvd_at_most": 0.8,
                            "worst_seed_mean_tvd_at_most": 0.8,
                            "worst_condition_tvd_at_most": 0.8,
                        }
                        for category in categories
                    },
                    "encoder_only_improvement": {
                        "median_seed_mean_tvd_improvement_at_least": -1.0,
                        "worst_seed_mean_tvd_improvement_at_least": -1.0,
                    },
                    "mean_movement": {
                        "median_generated_to_target_ratio_at_least": 0.5,
                        "worst_seed_median_ratio_at_least": 0.5,
                        "correct_sign_fraction_at_least": 1.0,
                    },
                    "width": {
                        "median_absolute_std_error_at_most": 0.2,
                        "worst_seed_mean_absolute_std_error_at_most": 0.2,
                    },
                    "structure": {
                        "median_hamming_profile_tvd_at_most": 0.2,
                        "median_within_shell_residual_at_most": 0.2,
                        "worst_seed_mean_hamming_profile_tvd_at_most": 0.2,
                        "worst_seed_mean_within_shell_residual_at_most": 0.2,
                    },
                    "reliability": {
                        "max_optimizer_seed_range_at_most": 0.5,
                        "max_random_layout_median_tvd_at_most_by_category": {
                            "near_extrapolation": random_layout_cap
                        },
                    },
                },
            }
        )
    )


def test_mirrored_rows_cannot_change_qualification(tmp_path: Path, monkeypatch) -> None:
    gates_path = tmp_path / "gates.json"
    _permissive_gates(gates_path)
    monkeypatch.setattr(
        analysis,
        "encoder_only_rows",
        lambda conditions, encoder, split: [
            {"condition": float(condition), "bitstring_tvd": 0.2}
            for condition in conditions
        ],
    )
    rows = _gate_fixture_rows()
    baseline = analysis.evaluate_gates(rows, gates_path)
    mirrors = []
    for row in rows:
        if row["protocol"] == "interpolation":
            poisoned = dict(row)
            poisoned.update(
                protocol="mirror_far", bitstring_tvd=0.99, generated_mean=-1000.0
            )
            mirrors.append(poisoned)
    changed = analysis.evaluate_gates(rows + mirrors, gates_path)
    assert changed == baseline


def test_bad_random_layout_is_not_hidden_by_pooled_median(
    tmp_path: Path, monkeypatch
) -> None:
    gates_path = tmp_path / "gates.json"
    _permissive_gates(gates_path, random_layout_cap=0.25)
    monkeypatch.setattr(
        analysis,
        "encoder_only_rows",
        lambda conditions, encoder, split: [
            {"condition": float(condition), "bitstring_tvd": 0.2}
            for condition in conditions
        ],
    )
    rows = _gate_fixture_rows()
    random_rows = []
    for seed, tvd in (
        (401, 0.3),
        (402, 0.3),
        (403, 0.3),
        (401, 0.1),
        (402, 0.1),
        (403, 0.1),
    ):
        row = next(
            row
            for row in rows
            if row["encoder"] == "broadcast_affine"
            and row["category"] == "near_extrapolation"
            and row["optimizer_seed"] == seed
        ).copy()
        row.update(
            placement="random",
            layout_seed=1101 if tvd == 0.3 else 1102,
            bitstring_tvd=tvd,
        )
        random_rows.append(row)
    result = analysis.evaluate_gates(rows + random_rows, gates_path)
    detail = result["encoders"]["broadcast_affine"]["check_details"][
        "reliability.random_layout.near_extrapolation"
    ]
    assert np.median([row["bitstring_tvd"] for row in random_rows]) <= 0.25
    assert detail["pass"] is False
    assert detail["observed"]["maximum"] == pytest.approx(0.3)


def test_gate_json_contains_native_boolean_values(tmp_path: Path, monkeypatch) -> None:
    gates_path = tmp_path / "gates.json"
    _permissive_gates(gates_path)
    monkeypatch.setattr(
        analysis,
        "encoder_only_rows",
        lambda conditions, encoder, split: [
            {"condition": float(condition), "bitstring_tvd": 0.2}
            for condition in conditions
        ],
    )
    result = analysis.evaluate_gates(_gate_fixture_rows(), gates_path)
    serialized = json.dumps(result)
    restored = json.loads(serialized)
    check = restored["encoders"]["broadcast_affine"]["checks"][
        "training.median_seed_mean"
    ]
    detail = restored["encoders"]["broadcast_affine"]["check_details"][
        "training.median_seed_mean"
    ]
    assert check is True and detail["pass"] is True
    assert '"training.median_seed_mean": "True"' not in serialized


def test_audit_attestation_binds_inputs_and_recomputes_digest(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    manifest_path = tmp_path / "audit.json"
    gates_path = tmp_path / "gates.json"
    rows_path = tmp_path / "rows.csv"
    gate_result_path = tmp_path / "gate-result.json"
    records_dir = tmp_path / "job_records"
    records_dir.mkdir()
    job_id = "job-1"
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"checkpoint")
    (records_dir / f"{job_id}.json").write_text("record")
    gates_path.write_text(json.dumps({"audit_seeds": [401], "frozen": True}))
    gates_hash = analysis._sha256(gates_path)
    manifest = {
        "phase": "audit",
        "split": "audit",
        "job_count": 1,
        "frozen_gates_sha256": gates_hash,
    }
    manifest_path.write_text(json.dumps(manifest))
    analysis._write_csv(rows_path, [{"condition": 0.5, "bitstring_tvd": 0.1}])
    gate_result = {
        "frozen_gates_sha256": gates_hash,
        "encoders": {
            encoder: {"checks": {f"c{i}": True for i in range(28)}, "pass": True}
            for encoder in ("broadcast_affine", "qubit_affine")
        },
    }
    gate_result["encoders"]["anti_block"] = {"checks": {}, "pass": None}
    gate_result_path.write_text(json.dumps(gate_result))
    job = SimpleNamespace(job_id=job_id, optimizer_seed=401)
    monkeypatch.setattr(
        analysis,
        "load_records",
        lambda manifest_arg, records_arg: (
            manifest,
            [{"job": job, "checkpoint": checkpoint}],
        ),
    )
    monkeypatch.setattr(
        analysis.campaign.runner,
        "repo_state",
        lambda: {"paper_revision": "abc123", "paper_dirty": False},
    )
    monkeypatch.setattr(analysis, "evaluate_gates", lambda rows, path: gate_result)
    expected_rows = [{"condition": 0.5, "bitstring_tvd": 0.1}]
    monkeypatch.setattr(
        analysis, "extract_rows", lambda entries, phase, split: expected_rows
    )
    output = tmp_path / "attestation.json"
    first = analysis.attest_audit(
        manifest_path, records_dir, gates_path, rows_path, gate_result_path, output
    )
    payload = {key: value for key, value in first.items() if key != "content_sha256"}
    expected_digest = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )
        .hexdigest()
    )
    assert first["content_sha256"] == expected_digest
    assert first["gate_verification"]["valid"] is True
    assert first["row_verification"]["matches_regenerated_rows"] is True
    assert first["launch_valid"] is False
    assert first["repository_state"]["paper_revision"] == "abc123"
    assert first["job_records"][job_id]["sha256"] == analysis._sha256(
        records_dir / f"{job_id}.json"
    )
    assert first["job_records"][job_id]["checkpoint_sha256"] == analysis._sha256(
        checkpoint
    )
    checkpoint.write_bytes(b"changed checkpoint")
    second = analysis.attest_audit(
        manifest_path, records_dir, gates_path, rows_path, gate_result_path, output
    )
    assert first["content_sha256"] != second["content_sha256"]


def _source_manifest(
    tmp_path: Path, library_revision: str = "lib-old"
) -> tuple[Path, dict, dict]:
    protocol_hash = campaign._sha256(campaign.PROTOCOL_PATH)
    files = {
        str(campaign.PROTOCOL_PATH.relative_to(campaign.ROOT)): protocol_hash,
        "iqp-cond/src/iqp_cond/example.py": "a" * 64,
    }
    state = {
        "paper_revision": "paper-old",
        "paper_dirty": True,
        "library_revision": library_revision,
        "library_version": "0.1",
    }
    gates_path = tmp_path / "frozen-gates.json"
    gates_path.write_text(json.dumps({"frozen": True}))
    manifest = {
        "study": "scalar_gaussian_translation",
        "phase": "audit",
        "split": "audit",
        "jobs": [],
        "job_count": 0,
        "frozen_gates_path": str(gates_path),
        "frozen_gates_sha256": analysis._sha256(gates_path),
        "protocol_sha256": protocol_hash,
        "source_file_sha256": files,
        "source_snapshot_sha256": analysis._snapshot_digest(files, library_revision),
        "library_revision": library_revision,
        "state": state,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    _write_test_provenance_root(path)
    return path, manifest, files


def _write_test_provenance_root(manifest_path: Path) -> Path:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    gates_path = Path(manifest["frozen_gates_path"]).resolve()
    protocol_path = campaign.PROTOCOL_PATH.resolve()
    root_path = manifest_path.parent / "provenance-root.json"
    payload = {
        "schema_version": 1,
        "identity": {
            "study": manifest["study"],
            "phase": manifest["phase"],
            "split": manifest["split"],
        },
        "paths": {
            "audit_manifest": manifest_path.name,
            "frozen_gates": str(gates_path),
            "protocol": str(protocol_path),
        },
        "audit_manifest_sha256": analysis._sha256(manifest_path),
        "frozen_gates_sha256": analysis._sha256(gates_path),
        "protocol_sha256": analysis._sha256(protocol_path),
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "job_count": manifest["job_count"],
    }
    payload["content_sha256"] = analysis._content_digest(payload)
    root_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return root_path


def _pin_test_provenance_root(monkeypatch, root_path: Path) -> None:
    payload = json.loads(Path(root_path).read_text())
    monkeypatch.setattr(
        analysis,
        "AUDIT_PROVENANCE_ROOT_CONTENT_SHA256",
        payload["content_sha256"],
    )


def _mock_current_source(
    monkeypatch, files: dict[str, str], revision: str, dirty=False
):
    state = {
        "paper_revision": "paper-new",
        "paper_dirty": dirty,
        "library_revision": revision,
        "library_version": "0.1",
    }
    monkeypatch.setattr(analysis, "_repo_state", lambda: state)
    monkeypatch.setattr(campaign.runner, "repo_state", lambda: state)
    monkeypatch.setattr(
        campaign,
        "_source_snapshot",
        lambda: (files, analysis._snapshot_digest(files, revision)),
    )
    return state


def test_source_equivalence_bridges_library_revision_only_change(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, manifest, files = _source_manifest(tmp_path)
    current_state = _mock_current_source(monkeypatch, files, "lib-new")
    bridge_path = tmp_path / "bridge.json"
    root_path = tmp_path / "provenance-root.json"
    _pin_test_provenance_root(monkeypatch, root_path)
    bridge = analysis.create_source_equivalence(manifest_path, bridge_path, root_path)
    current_state["paper_revision"] = "paper-C2"
    validated = analysis.validate_source_equivalence(
        manifest_path, bridge_path, root_path
    )
    assert bridge["original"]["library_revision"] == "lib-old"
    assert bridge["current"]["library_revision"] == "lib-new"
    assert (
        bridge["original"]["source_file_sha256"]
        == bridge["current"]["source_file_sha256"]
    )
    assert validated["content_sha256"] == bridge["content_sha256"]
    assert manifest_path.read_text() == json.dumps(manifest)


def test_source_equivalence_rejects_dirty_changed_files_and_bad_digest(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _, files = _source_manifest(tmp_path)
    bridge_path = tmp_path / "bridge.json"
    root_path = tmp_path / "provenance-root.json"
    _pin_test_provenance_root(monkeypatch, root_path)
    _mock_current_source(monkeypatch, files, "lib-new", dirty=True)
    with pytest.raises(RuntimeError, match="clean paper repository"):
        analysis.create_source_equivalence(manifest_path, bridge_path, root_path)

    changed = dict(files)
    changed["iqp-cond/src/iqp_cond/example.py"] = "b" * 64
    _mock_current_source(monkeypatch, changed, "lib-new")
    with pytest.raises(RuntimeError, match="file hashes differ"):
        analysis.create_source_equivalence(manifest_path, bridge_path, root_path)

    _mock_current_source(monkeypatch, files, "lib-new")
    analysis.create_source_equivalence(manifest_path, bridge_path, root_path)
    payload = json.loads(bridge_path.read_text())
    payload["current"]["library_revision"] = "tampered"
    bridge_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content digest"):
        analysis.validate_source_equivalence(manifest_path, bridge_path, root_path)


def test_source_equivalence_rejects_post_creation_provenance_changes(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _, files = _source_manifest(tmp_path)
    state = _mock_current_source(monkeypatch, files, "lib-new")
    bridge_path = tmp_path / "bridge.json"
    root_path = tmp_path / "provenance-root.json"
    _pin_test_provenance_root(monkeypatch, root_path)
    analysis.create_source_equivalence(manifest_path, bridge_path, root_path)

    changed_files = dict(files)
    changed_files["iqp-cond/src/iqp_cond/example.py"] = "b" * 64
    monkeypatch.setattr(
        campaign,
        "_source_snapshot",
        lambda: (
            changed_files,
            analysis._snapshot_digest(changed_files, "lib-new"),
        ),
    )
    with pytest.raises(RuntimeError, match="provenance differs"):
        analysis.validate_source_equivalence(manifest_path, bridge_path, root_path)
    monkeypatch.setattr(
        campaign,
        "_source_snapshot",
        lambda: (files, analysis._snapshot_digest(files, "lib-new")),
    )

    state["library_revision"] = "lib-changed"
    with pytest.raises(RuntimeError, match="provenance differs"):
        analysis.validate_source_equivalence(manifest_path, bridge_path, root_path)

    state["library_revision"] = "lib-new"
    state["paper_dirty"] = True
    with pytest.raises(RuntimeError, match="clean paper repository"):
        analysis.validate_source_equivalence(manifest_path, bridge_path, root_path)

    state["paper_dirty"] = False
    payload = json.loads(bridge_path.read_text())
    payload["protocol_sha256"] = "0" * 64
    payload["content_sha256"] = analysis._content_digest(payload)
    bridge_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="protocol differs"):
        analysis.validate_source_equivalence(manifest_path, bridge_path, root_path)


def test_provenance_root_rejects_coordinated_manifest_rewrite(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, manifest, _ = _source_manifest(tmp_path)
    root_path = tmp_path / "provenance-root.json"
    _pin_test_provenance_root(monkeypatch, root_path)
    bridge_path = tmp_path / "bridge.json"
    files = manifest["source_file_sha256"]
    _mock_current_source(monkeypatch, files, "lib-new")
    analysis.create_source_equivalence(manifest_path, bridge_path, root_path)
    manifest["created_utc"] = "rewritten"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="audit_manifest_sha256 mismatch"):
        analysis.validate_source_equivalence(manifest_path, bridge_path, root_path)


def test_committed_audit_provenance_root_matches_frozen_inputs() -> None:
    project = Path(__file__).resolve().parents[1] / "scalar_gaussian_translation"
    manifest_path = project / "manifests" / "audit.json"
    root_path = project / "review" / "audit_provenance_root.json"
    root = analysis.validate_provenance_root(manifest_path, root_path)
    assert root["file_sha256"] == analysis._sha256(root_path)
    assert root["content_sha256"] == json.loads(root_path.read_text())["content_sha256"]
    assert root["content_sha256"] == analysis.AUDIT_PROVENANCE_ROOT_CONTENT_SHA256


def test_analysis_loader_accepts_valid_bridge_while_campaign_loader_stays_strict(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, manifest, files = _source_manifest(tmp_path)
    job = campaign.CampaignJob(
        phase="audit",
        split="audit",
        encoder="broadcast_affine",
        protocol="interpolation",
        anchors=(0.0, 1.0),
        anchor_count=2,
        placement="even",
        layout_seed=None,
        optimizer_seed=401,
        budget_mode="fixed_total",
        hyperparameter_id="test",
        hyperparameters={},
    )
    manifest["jobs"] = [campaign.asdict(job)]
    manifest["job_count"] = 1
    manifest_path.write_text(json.dumps(manifest))
    root_path = _write_test_provenance_root(manifest_path)
    _pin_test_provenance_root(monkeypatch, root_path)
    _mock_current_source(monkeypatch, files, "lib-new")
    bridge_path = tmp_path / "bridge.json"
    analysis.create_source_equivalence(manifest_path, bridge_path, root_path)
    with pytest.raises(RuntimeError, match="source snapshot changed"):
        campaign._load_manifest(manifest_path)
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.npz"
    checkpoint.write_bytes(b"checkpoint")
    (records_dir / f"{job.job_id}.json").write_text(
        json.dumps(
            {
                "job": campaign.asdict(job),
                "run_record": {"directory": str(run_dir), "run_id": "run"},
            }
        )
    )
    manifest, entries = analysis.load_records(
        manifest_path, records_dir, bridge_path, root_path
    )
    assert manifest["library_revision"] == "lib-old"
    assert len(entries) == 1
    assert entries[0]["job"] == job
    assert entries[0]["checkpoint"] == checkpoint


def test_read_only_rows_pipeline_forwards_source_equivalence(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "audit.json"
    manifest = {"phase": "audit", "split": "audit"}
    bridge_path = tmp_path / "bridge.json"
    calls = []

    root_path = tmp_path / "root.json"

    def fake_load(path, records_dir, source_equivalence_path, provenance_root_path):
        calls.append((path, records_dir, source_equivalence_path, provenance_root_path))
        return manifest, []

    monkeypatch.setattr(analysis, "load_records", fake_load)
    loaded_manifest, rows = analysis._dev_rows(
        manifest_path, tmp_path / "records", bridge_path, root_path
    )
    assert loaded_manifest == manifest
    assert rows == []
    assert calls == [(manifest_path, tmp_path / "records", bridge_path, root_path)]


def _attestation_fixture(
    tmp_path: Path, monkeypatch, *, dirty=False, check_count=28, pass_value=True
):
    from types import SimpleNamespace

    manifest_path = tmp_path / "audit.json"
    gates_path = tmp_path / "gates.json"
    rows_path = tmp_path / "rows.csv"
    gate_result_path = tmp_path / "gate-result.json"
    tmp_path.mkdir(parents=True, exist_ok=True)
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    job_id = "job-1"
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"checkpoint")
    (records_dir / f"{job_id}.json").write_text("record")
    gates_path.write_text(json.dumps({"audit_seeds": [401]}))
    gates_hash = analysis._sha256(gates_path)
    protocol_hash = campaign._sha256(campaign.PROTOCOL_PATH)
    source_files = {
        str(campaign.PROTOCOL_PATH.relative_to(campaign.ROOT)): protocol_hash,
        "iqp-cond/src/iqp_cond/example.py": "a" * 64,
    }
    revision = "lib-current"
    state = {
        "paper_revision": "paper-current",
        "paper_dirty": dirty,
        "library_revision": revision,
        "library_version": "0.1",
    }
    manifest = {
        "study": "scalar_gaussian_translation",
        "phase": "audit",
        "split": "audit",
        "jobs": [{}],
        "job_count": 1,
        "frozen_gates_path": str(gates_path),
        "frozen_gates_sha256": gates_hash,
        "protocol_sha256": protocol_hash,
        "source_file_sha256": source_files,
        "source_snapshot_sha256": analysis._snapshot_digest(source_files, revision),
        "library_revision": revision,
        "state": state,
    }
    manifest_path.write_text(json.dumps(manifest))
    expected_rows = [{"condition": 0.5, "bitstring_tvd": 0.1}]
    analysis._write_csv(rows_path, expected_rows)
    checks = {f"check_{index}": True for index in range(check_count)}
    result = {
        "frozen_gates_sha256": gates_hash,
        "encoders": {
            name: {"checks": checks, "pass": pass_value}
            for name in ("broadcast_affine", "qubit_affine")
        },
    }
    result["encoders"]["anti_block"] = {"checks": {}, "pass": None}
    gate_result_path.write_text(json.dumps(result))
    job = SimpleNamespace(job_id=job_id, optimizer_seed=401)
    monkeypatch.setattr(
        analysis,
        "load_records",
        lambda manifest_arg, records_arg: (
            manifest,
            [{"job": job, "checkpoint": checkpoint}],
        ),
    )
    monkeypatch.setattr(analysis, "_repo_state", lambda: state)
    monkeypatch.setattr(campaign.runner, "repo_state", lambda: state)
    monkeypatch.setattr(
        campaign,
        "_source_snapshot",
        lambda: (source_files, analysis._snapshot_digest(source_files, revision)),
    )
    monkeypatch.setattr(analysis, "evaluate_gates", lambda rows, path: result)
    monkeypatch.setattr(
        analysis,
        "extract_rows",
        lambda entries, phase, split: expected_rows,
    )
    return manifest_path, records_dir, gates_path, rows_path, gate_result_path, result


def test_audit_attestation_rejects_gate_mismatch_count_and_failed_pass(
    tmp_path: Path, monkeypatch
) -> None:
    args = _attestation_fixture(tmp_path / "mismatch", monkeypatch)
    *prefix, supplied, result = args
    supplied.write_text(json.dumps({**result, "extra": True}))
    with pytest.raises(ValueError, match="differs from recomputation"):
        analysis.attest_audit(*prefix, supplied, tmp_path / "out.json")

    count_dir = tmp_path / "count"
    count_dir.mkdir()
    args = _attestation_fixture(count_dir, monkeypatch, check_count=27)
    manifest, records, gates, rows, supplied, _ = args
    with pytest.raises(ValueError, match="exactly 28"):
        analysis.attest_audit(
            manifest, records, gates, rows, supplied, count_dir / "out.json"
        )

    pass_dir = tmp_path / "failed-pass"
    pass_dir.mkdir()
    args = _attestation_fixture(pass_dir, monkeypatch, pass_value=False)
    manifest, records, gates, rows, supplied, _ = args
    with pytest.raises(ValueError, match="exactly 28"):
        analysis.attest_audit(
            manifest, records, gates, rows, supplied, pass_dir / "out.json"
        )


def test_audit_attestation_rejects_rows_not_regenerated_from_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, records, gates, rows, supplied, _ = _attestation_fixture(
        tmp_path, monkeypatch
    )
    analysis._write_csv(rows, [{"condition": 0.5, "bitstring_tvd": 0.12}])
    with pytest.raises(ValueError, match="checkpoint regeneration"):
        analysis.attest_audit(
            manifest, records, gates, rows, supplied, tmp_path / "out.json"
        )


def test_dirty_audit_attestation_is_provisional_but_validates_gates(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, records, gates, rows, gate_result, _ = _attestation_fixture(
        tmp_path, monkeypatch, dirty=True
    )
    attestation = analysis.attest_audit(
        manifest,
        records,
        gates,
        rows,
        gate_result,
        tmp_path / "attestation.json",
    )
    assert attestation["gate_verification"]["valid"] is True
    assert attestation["source_provenance"]["validated"] is True
    assert attestation["launch_valid"] is False


def test_clean_provenance_and_both_gate_results_enable_launch_valid(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, records, gates, rows, gate_result, _ = _attestation_fixture(
        tmp_path, monkeypatch, dirty=False
    )
    root_path = _write_test_provenance_root(manifest)
    _pin_test_provenance_root(monkeypatch, root_path)
    attestation = analysis.attest_audit(
        manifest,
        records,
        gates,
        rows,
        gate_result,
        tmp_path / "attestation.json",
        provenance_root_path=root_path,
    )
    assert attestation["launch_valid"] is True


def test_clean_direct_audit_without_provenance_root_is_not_launch_valid(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, records, gates, rows, gate_result, _ = _attestation_fixture(
        tmp_path, monkeypatch, dirty=False
    )
    attestation = analysis.attest_audit(
        manifest,
        records,
        gates,
        rows,
        gate_result,
        tmp_path / "attestation.json",
    )
    assert attestation["source_provenance"]["validated"] is True
    assert attestation["provenance_root"] is None
    assert attestation["launch_valid"] is False


def test_audit_attestation_accepts_verified_source_equivalence_bridge(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    manifest_path, records_dir, gates, rows, gate_result, _ = _attestation_fixture(
        tmp_path, monkeypatch, dirty=False
    )
    manifest = json.loads(manifest_path.read_text())
    source_files = manifest["source_file_sha256"]
    original_state = {
        "paper_revision": "paper-audit",
        "paper_dirty": True,
        "library_revision": "lib-old",
        "library_version": "0.1",
    }
    manifest.update(
        jobs=[{}],
        library_revision="lib-old",
        source_snapshot_sha256=analysis._snapshot_digest(source_files, "lib-old"),
        state=original_state,
    )
    manifest_path.write_text(json.dumps(manifest))
    bridge_path = tmp_path / "source-equivalence.json"
    root_path = _write_test_provenance_root(manifest_path)
    _pin_test_provenance_root(monkeypatch, root_path)
    analysis.create_source_equivalence(manifest_path, bridge_path, root_path)
    current_state = campaign.runner.repo_state()
    current_state["paper_revision"] = "paper-C2"
    checkpoint = tmp_path / "checkpoint.npz"
    job = SimpleNamespace(job_id="job-1", optimizer_seed=401)
    monkeypatch.setattr(
        analysis,
        "load_records",
        lambda manifest_arg, records_arg, bridge_arg, root_arg: (
            manifest,
            [{"job": job, "checkpoint": checkpoint}],
        ),
    )
    attestation = analysis.attest_audit(
        manifest_path,
        records_dir,
        gates,
        rows,
        gate_result,
        tmp_path / "attestation.json",
        bridge_path,
        root_path,
    )
    bridge_payload = json.loads(bridge_path.read_text())
    assert attestation["source_provenance"] == {
        "validated": True,
        "method": "source_equivalence_bridge",
        "bridge_file_sha256": analysis._sha256(bridge_path),
        "bridge_content_sha256": bridge_payload["content_sha256"],
    }
    assert attestation["launch_valid"] is True
