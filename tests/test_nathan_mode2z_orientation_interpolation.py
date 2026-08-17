from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.nathan_local_vector_truth import line_orientation_error
from vbb_study.digital_twin.nathan_mode2w_fix_sequential_master import _source_config
from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import build_mode2y_input_fields
from vbb_study.digital_twin.nathan_mode2z_orientation_interpolation import (
    MODE2Z_ALLOWED_OUTCOMES,
    MODE2Z_DEFAULT_ETA_VALUES,
    Mode2ZSweepConfig,
    build_interpolated_alpha,
    build_interpolated_input_field,
    in_sector_orientation_delta,
    mode2z_route_id,
    mode2z_trend_rows,
    write_mode2z_outputs,
)
from vbb_study.digital_twin.nathan_vector_hexagon import mode2n_source_target


@pytest.fixture(scope="module")
def source_data():
    cfg = _source_config(grid_n=192, z_planes=11, z_start_m=0.0, z_end_m=0.2)
    return mode2n_source_target(cfg, grid_n=192, z_planes=11)


@pytest.fixture(scope="module")
def inputs(source_data):
    return build_mode2y_input_fields(source_data)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("mode2z")
    config = Mode2ZSweepConfig(
        grid_n=192,
        eta_values=(0.0, 0.5, 1.0),
        z_step_m=0.02,
        publication_quality=False,
    )
    return write_mode2z_outputs(
        root / "outputs",
        document_path=root / "docs/86_nathan_mode2z_orientation_interpolation.md",
        config=config,
    )


def test_default_eta_grid_has_eleven_ordered_endpoints():
    assert MODE2Z_DEFAULT_ETA_VALUES == pytest.approx(tuple(np.linspace(0.0, 1.0, 11)))
    assert MODE2Z_DEFAULT_ETA_VALUES[0] == 0.0
    assert MODE2Z_DEFAULT_ETA_VALUES[-1] == 1.0


def test_in_sector_delta_never_exceeds_thirty_degrees(source_data, inputs):
    delta = in_sector_orientation_delta(
        source_data["grid"]["PHI"],
        inputs.sector_index,
        sector_rotation_rad=float(source_data["config"].sector_rotation_rad),
    )
    assert np.max(np.abs(delta)) <= np.pi / 6.0 + 1e-12


def test_eta_endpoints_reproduce_mode2y_jones_fields(source_data, inputs):
    ex0, ey0, _ = build_interpolated_input_field(source_data, inputs, 0.0)
    ex1, ey1, _ = build_interpolated_input_field(source_data, inputs, 1.0)
    assert np.allclose(ex0, inputs.averaged_ex, rtol=0.0, atol=2e-13)
    assert np.allclose(ey0, inputs.averaged_ey, rtol=0.0, atol=2e-13)
    assert np.allclose(ex1, inputs.continuous_ex, rtol=0.0, atol=2e-13)
    assert np.allclose(ey1, inputs.continuous_ey, rtol=0.0, atol=2e-13)


def test_midpoint_halves_native_line_error(source_data, inputs):
    alpha0 = build_interpolated_alpha(source_data, inputs, 0.0)
    alpha_half = build_interpolated_alpha(source_data, inputs, 0.5)
    error0 = line_orientation_error(alpha0, inputs.continuous_alpha_rad)
    error_half = line_orientation_error(alpha_half, inputs.continuous_alpha_rad)
    assert np.allclose(error_half, 0.5 * error0, rtol=0.0, atol=2e-13)


def test_every_eta_preserves_input_power(source_data, inputs):
    for eta in (0.0, 0.2, 0.5, 0.8, 1.0):
        ex, ey, _ = build_interpolated_input_field(source_data, inputs, eta)
        power = np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2)
        assert power == pytest.approx(inputs.continuous_power, rel=2e-14)


def test_publication_config_requires_n1024_and_exact_z60():
    config = Mode2ZSweepConfig()
    assert config.grid_n >= 1024
    assert config.grid_n != 384
    assert np.any(np.isclose(config.z_values_m(), 0.06, rtol=0.0, atol=1e-12))
    with pytest.raises(ValueError):
        Mode2ZSweepConfig(grid_n=384, publication_quality=True).validate()


def test_both_routes_and_all_eta_values_are_propagated(generated):
    result = generated["result"]
    assert set(result.routes) == {
        mode2z_route_id(route, eta)
        for route in ("ideal", "realistic")
        for eta in result.config.eta_values
    }


def test_eta_one_passes_ideal_local_vector_truth(generated):
    rows = {float(row["eta"]): row for row in generated["result"].input_rows}
    assert rows[1.0]["full_local_vector_truth_pass"] is True
    assert rows[1.0]["local_angle_rms_rad"] < 1e-10
    assert rows[0.0]["local_angle_rms_rad"] > rows[1.0]["local_angle_rms_rad"]


def test_native_metrics_and_display_interpolation_remain_separate(generated):
    for route in generated["result"].routes.values():
        assert route.metadata["native_grid_metrics"] is True
        assert route.metadata["display_interpolation_used_for_metrics"] is False


def test_trend_audit_and_outcome_are_predeclared(generated):
    result = generated["result"]
    assert len(result.trend_rows) == 8
    assert result.outcome in MODE2Z_ALLOWED_OUTCOMES
    assert result.outcome_reason


def test_monotone_plateaus_and_bounded_correlation_are_not_misclassified():
    rows = []
    for route in ("ideal", "realistic"):
        for eta, corr, edge, width, fwhm in (
            (0.0, 0.975, 7.0, 0.16, 0.10),
            (0.5, 0.985, 8.5, 0.16, 0.07),
            (1.0, 0.995, 10.5, 0.12, 0.07),
        ):
            rows.append({
                "optical_route": route,
                "eta": eta,
                "z60_correlation_to_v0": corr,
                "edge_gradient_sharpness_mm_inv": edge,
                "threshold_transition_width_mm": width,
                "bright_ridge_fwhm_mm": fwhm,
            })
    trends = mode2z_trend_rows(rows)
    assert all(row["endpoint_pass"] for row in trends)
    assert all(row["monotonic_pass"] for row in trends)
    assert all(row["trend_pass"] for row in trends)


def test_summary_contains_energy_shape_and_axial_diagnostics(generated):
    required = {
        "morphology_quality_index",
        "morphology_energy_pareto",
        "peak_intensity",
        "useful_region_power",
        "best_z_mm",
        "propagation_persistence_fraction",
    }
    for row in generated["result"].summary_rows:
        assert required.issubset(row)


def test_publication_figures_and_tables_are_written(generated):
    paths = generated["figure_paths"]
    for key in ("inputs_png", "ideal_xy_png", "realistic_xy_png", "metrics_png", "propagation_png", "tradeoff_png", "gates_png"):
        assert Path(paths[key]).is_file()
    for key in ("summary_csv", "summary_json", "z_metrics_csv", "trend_csv", "trend_json", "input_truth_csv", "input_truth_json", "gate_definition", "scope_manifest", "outcome_report", "document_path"):
        assert Path(generated[key]).is_file()


def test_scope_is_sequential_only_and_makes_no_sample_claim(generated):
    scope = json.loads(Path(generated["scope_manifest"]).read_text(encoding="utf-8"))
    outcome = json.loads(Path(generated["outcome_report"]).read_text(encoding="utf-8"))
    assert scope["split_arm_pbs_architecture_used"] is False
    assert "sequential collinear beam" in scope["accepted_architecture"]
    assert scope["microfabrication_sample_plane_success_claim"] is False
    assert scope["forbidden_n384_hero_data_used"] is False
    assert outcome["no_microfabrication_sample_plane_success_claim"] is True
