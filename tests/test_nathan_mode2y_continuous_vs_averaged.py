from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.nathan_local_vector_truth import evaluate_local_vector_truth
from vbb_study.digital_twin.nathan_mode2w_fix_sequential_master import _source_config
from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import (
    MODE2Y_ALLOWED_OUTCOMES,
    MODE2Y_MIN_HERO_GRID_N,
    MODE2Y_SELECTED_Z_M,
    Mode2YStudyConfig,
    build_mode2y_input_fields,
    build_mode2y_pre_axicon_fields,
    build_sector_averaged_alpha,
    run_mode2y_study,
    write_mode2y_outputs,
)
from vbb_study.digital_twin.nathan_vector_hexagon import mode2n_source_target


@pytest.fixture(scope="module")
def source_data():
    cfg = _source_config(grid_n=192, z_planes=21, z_start_m=0.0, z_end_m=0.2)
    return mode2n_source_target(cfg, grid_n=192, z_planes=21)


@pytest.fixture(scope="module")
def inputs(source_data):
    return build_mode2y_input_fields(source_data)


@pytest.fixture(scope="module")
def fast_result():
    config = Mode2YStudyConfig(grid_n=192, z_step_m=0.01, publication_quality=False)
    return run_mode2y_study(config)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("mode2y")
    config = Mode2YStudyConfig(grid_n=192, z_step_m=0.01, publication_quality=False)
    return write_mode2y_outputs(
        root / "outputs",
        document_path=root / "docs/85_nathan_mode2y_continuous_vs_averaged.md",
        config=config,
    )


def test_continuous_and_averaged_fields_are_distinct(inputs):
    assert not np.allclose(inputs.continuous_ex, inputs.averaged_ex)
    assert not np.allclose(inputs.continuous_ey, inputs.averaged_ey)


def test_equal_input_power_normalisation_holds(inputs):
    assert inputs.averaged_power_after_normalisation == pytest.approx(inputs.continuous_power, rel=2e-14)
    assert inputs.metadata["same_amplitude_envelope"] is True
    assert inputs.metadata["same_total_input_power"] is True


def test_sector_averaged_alpha_is_constant_inside_each_sector(source_data, inputs):
    for sector in range(6):
        values = inputs.averaged_alpha_rad[inputs.sector_index == sector]
        assert np.max(np.abs(np.angle(np.exp(2j * (values - values[0]))))) < 1e-13


def test_sector_average_uses_authoritative_sector_labels(source_data):
    alpha, radial, index = build_sector_averaged_alpha(
        source_data["grid"]["PHI"], sector_rotation_rad=float(source_data["config"].sector_rotation_rad)
    )
    assert alpha.shape == source_data["A"].shape
    assert index.shape == alpha.shape
    assert np.array_equal(radial, source_data["radial_mask"])


def test_continuous_passes_local_truth_while_surrogate_fails(source_data, inputs):
    x = source_data["grid"]["x"]
    continuous = evaluate_local_vector_truth(
        "continuous", inputs.continuous_ex, inputs.continuous_ey, x, x, inputs.continuous_alpha_rad
    )
    averaged = evaluate_local_vector_truth(
        "averaged", inputs.averaged_ex, inputs.averaged_ey, x, x, inputs.continuous_alpha_rad
    )
    assert continuous.metrics.passed_full_vector_truth_gate
    assert not averaged.metrics.passed_full_vector_truth_gate


def test_z60_and_requested_landmarks_are_included_exactly():
    values = Mode2YStudyConfig().z_values_m()
    for required in MODE2Y_SELECTED_Z_M:
        assert np.any(np.isclose(values, required, rtol=0.0, atol=1e-12))


def test_publication_hero_config_is_at_least_1024_and_not_384():
    config = Mode2YStudyConfig()
    assert config.grid_n >= MODE2Y_MIN_HERO_GRID_N
    assert config.grid_n != 384
    with pytest.raises(ValueError):
        Mode2YStudyConfig(grid_n=384, publication_quality=True).validate()


def test_ideal_and_realistic_routes_are_both_audited(fast_result):
    assert set(fast_result.routes) == {
        "ideal_continuous",
        "ideal_sector_averaged",
        "realistic_continuous_common_4f",
        "realistic_sector_averaged_common_4f",
    }


def test_continuous_common_4f_route_uses_validated_controls(source_data, inputs):
    config = Mode2YStudyConfig(grid_n=192, z_step_m=0.01, publication_quality=False)
    _, reports = build_mode2y_pre_axicon_fields(source_data, inputs, config)
    report = reports["realistic_continuous_common_4f"]
    assert report["carrier_lpmm"] == pytest.approx(6.25)
    assert report["iris_radius_lpmm"] == pytest.approx(2.5)


def test_propagated_outputs_differ_on_native_arrays(fast_result):
    continuous = fast_result.routes["ideal_continuous"].selected_planes["z60.000mm"]
    averaged = fast_result.routes["ideal_sector_averaged"].selected_planes["z60.000mm"]
    assert continuous.shape == averaged.shape == (192, 192)
    assert np.linalg.norm(continuous - averaged) > 0.0


def test_strict_gate_and_sharpness_metrics_are_separate(fast_result):
    for row in fast_result.summary_rows:
        assert isinstance(row["strict_hexagon_pass"], bool)
        assert np.isfinite(row["edge_gradient_sharpness_mm_inv"])
        assert np.isfinite(row["corner_concentration"])
        assert "strict_fail_reasons" in row


def test_native_metrics_and_display_interpolation_are_separated(fast_result):
    for route in fast_result.routes.values():
        assert route.metadata["native_grid_metrics"] is True
        assert route.metadata["display_interpolation_used_for_metrics"] is False


def test_outcome_is_explicit_and_predeclared(fast_result):
    assert fast_result.outcome in MODE2Y_ALLOWED_OUTCOMES
    assert fast_result.outcome_reason


def test_difference_and_profile_figures_are_generated(generated):
    paths = generated["figure_paths"]
    assert Path(paths["ideal_xy_focus_png"]).is_file()
    assert Path(paths["realistic_focused_png"]).is_file()
    assert Path(paths["ideal_propagation_png"]).is_file()
    assert Path(paths["realistic_profiles_png"]).is_file()
    assert Path(paths["dashboard_png"]).is_file()


def test_summary_csv_json_scope_and_outcome_are_written(generated):
    for key in ("summary_csv", "summary_json", "scope_manifest", "outcome_report", "document_path"):
        assert Path(generated[key]).is_file()


def test_scope_is_sequential_only_and_makes_no_sample_claim(generated):
    scope = json.loads(Path(generated["scope_manifest"]).read_text(encoding="utf-8"))
    report = json.loads(Path(generated["outcome_report"]).read_text(encoding="utf-8"))
    assert scope["split_arm_pbs_architecture_used"] is False
    assert "sequential collinear beam" in scope["accepted_architecture"]
    assert scope["microfabrication_sample_plane_success_claim"] is False
    assert report["no_microfabrication_sample_plane_success_claim"] is True
