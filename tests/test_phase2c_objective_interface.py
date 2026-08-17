from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vbb_study.digital_twin.phase2c_objective_interface import (
    CONVERGENCE_CORRELATION_CHANGE_MAX,
    CONVERGENCE_FEATURE_RELATIVE_MAX,
    CONVERGENCE_LONGITUDINAL_ABSOLUTE_MAX,
    FRESNEL_ENERGY_TOLERANCE,
    FRESNEL_SNELL_TOLERANCE,
    FRESNEL_TRANSVERSALITY_TOLERANCE,
    PHASE2C_ALLOWED_OUTCOMES,
    PHASE2C_CASE_IDS,
    Phase2CConfig,
    _fourier_resample_complex_plane,
)
from vbb_study.digital_twin.phase2c_figures import (
    _bandlimited_local_render_intensity,
    _plotly_magma_colorscale,
    _surface_crop,
)
from vbb_study.equations.vector_debye import DebyeConfig, debye_focus_plane
from vbb_study.equations.vector_fresnel_interface import (
    FresnelInterfaceConfig,
    fresnel_coefficients,
    transmit_vector_field_planar_interface,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "outputs" / "validation" / "phase2c"
FIGURES = ROOT / "outputs" / "figures" / "phase2c"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _control_pupil(kind: str, n: int = 65):
    wavelength = 1029e-9
    focal_length = 4e-3
    na = 0.45
    radius = focal_length * na
    axis = np.linspace(-1.05 * radius, 1.05 * radius, n)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    phi = np.arctan2(Y, X)
    aperture = (np.hypot(X, Y) <= radius).astype(float)
    if kind == "x":
        ex, ey = aperture.astype(complex), np.zeros_like(X, dtype=complex)
    elif kind == "radial":
        ex, ey = aperture * np.cos(phi), aperture * np.sin(phi)
    elif kind == "azimuthal":
        ex, ey = -aperture * np.sin(phi), aperture * np.cos(phi)
    else:
        raise ValueError(kind)
    config = DebyeConfig(
        wavelength,
        1.0,
        na,
        focal_length,
        radius,
        quadrature_order_r=20,
        quadrature_order_phi=60,
        max_output_points=128,
    )
    output = np.linspace(-3e-6, 3e-6, 25)
    return ex, ey, axis, output, config


def _p_plane_wave(theta_rad: float = np.deg2rad(20.0)):
    n = 64
    wavelength = 1029e-9
    mode = 5
    dx = mode * wavelength / (n * np.sin(theta_rad))
    x = (np.arange(n) - n / 2 + 0.5) * dx
    X, _ = np.meshgrid(x, x, indexing="xy")
    phase = np.exp(1j * 2 * np.pi * np.sin(theta_rad) / wavelength * X)
    return np.cos(theta_rad) * phase, np.zeros_like(phase), -np.sin(theta_rad) * phase, dx


def test_phase2c_config_predeclares_publication_and_convergence_contracts() -> None:
    config = Phase2CConfig()
    config.validate()
    validation = Phase2CConfig.validation_preset()
    hero = Phase2CConfig.high_resolution_hero_preset()
    assert config.pupil_grid_n >= 1024
    assert config.objective_fft_pad_factor >= 2
    assert config.h1_objective_fft_pad_factor >= 4
    assert config.h1_output_grid_n >= config.output_grid_n
    assert CONVERGENCE_CORRELATION_CHANGE_MAX == 1e-3
    assert CONVERGENCE_FEATURE_RELATIVE_MAX == 0.01
    assert CONVERGENCE_LONGITUDINAL_ABSOLUTE_MAX == 1e-3
    assert config.mapping_mode == "fixed_physical_optics"
    assert validation.publication_quality is False
    assert hero.publication_quality is True


def test_debye_uniform_x_symmetry_and_y_rotation_controls() -> None:
    debye = _json(VALIDATION / "phase2c_solver_validation.json")["debye"]
    assert debye["uniform_x_mirror_residual"] <= 2e-3
    assert debye["uniform_xy_rotation_correlation"] >= 0.999
    assert debye["radial_rotation_correlation"] >= 0.999


def test_debye_low_na_sequence_converges_to_scalar_and_suppresses_ez() -> None:
    rows = _json(VALIDATION / "phase2c_solver_validation.json")["debye"]["low_na_sequence"]
    assert [float(row["numerical_aperture"]) for row in rows] == [0.45, 0.20, 0.10, 0.05]
    correlations = [float(row["scalar_airy_correlation"]) for row in rows]
    longitudinal = [float(row["longitudinal_power_fraction"]) for row in rows]
    assert all(second >= first for first, second in zip(correlations, correlations[1:]))
    assert all(second < first for first, second in zip(longitudinal, longitudinal[1:]))
    assert correlations[-1] >= 0.999


def test_debye_global_phase_and_linear_scaling() -> None:
    ex, ey, axis, output, config = _control_pupil("x")
    baseline = debye_focus_plane(ex, ey, axis, axis, output, output, 0.0, config)
    phase = debye_focus_plane(ex * np.exp(1j * 0.42), ey, axis, axis, output, output, 0.0, config)
    scale = 1.3 - 0.4j
    scaled = debye_focus_plane(ex * scale, ey, axis, axis, output, output, 0.0, config)
    np.testing.assert_allclose(phase.intensity, baseline.intensity, rtol=2e-12, atol=1e-20)
    np.testing.assert_allclose(scaled.intensity, abs(scale) ** 2 * baseline.intensity, rtol=2e-12, atol=1e-20)


def test_radial_focus_has_strong_ez_and_azimuthal_axis_ez_is_negligible() -> None:
    radial = _control_pupil("radial")
    azimuthal = _control_pupil("azimuthal")
    r = debye_focus_plane(radial[0], radial[1], radial[2], radial[2], radial[3], radial[3], 0.0, radial[4])
    a = debye_focus_plane(
        azimuthal[0], azimuthal[1], azimuthal[2], azimuthal[2], azimuthal[3], azimuthal[3], 0.0, azimuthal[4]
    )
    centre = radial[3].size // 2
    assert r.component_power_fractions["Ez_power_fraction"] > 0.05
    assert abs(a.Ez[centre, centre]) ** 2 / np.max(a.intensity) < 1e-12


def test_fresnel_normal_brewster_snell_and_energy_controls() -> None:
    normal = fresnel_coefficients(1.0, 2.44, 0.0)
    assert complex(normal["t_s"]) == complex(normal["t_p"])
    assert abs(complex(normal["t_s"]) - 2.0 / 3.44) < 1e-14
    brewster = fresnel_coefficients(1.0, 2.44, np.arctan(2.44))
    assert abs(complex(brewster["r_p"])) < 1e-12
    oblique = fresnel_coefficients(1.0, 2.44, np.deg2rad(20.0))
    assert abs(float(oblique["R_s"]) + float(oblique["T_s"]) - 1.0) <= FRESNEL_ENERGY_TOLERANCE
    assert abs(float(oblique["R_p"]) + float(oblique["T_p"]) - 1.0) <= FRESNEL_ENERGY_TOLERANCE
    assert float(oblique["snell_residual"]) <= FRESNEL_SNELL_TOLERANCE


def test_spectral_fresnel_transversality_energy_and_equal_index_identity() -> None:
    ex, ey, ez, dx = _p_plane_wave()
    result = transmit_vector_field_planar_interface(
        ex, ey, ez, dx, dx, FresnelInterfaceConfig(1029e-9, 1.0, 2.44)
    )
    assert abs(result.diagnostics["lossless_R_plus_T"] - 1.0) <= FRESNEL_ENERGY_TOLERANCE
    assert result.diagnostics["transmitted_transversality_residual"] <= FRESNEL_TRANSVERSALITY_TOLERANCE
    assert result.diagnostics["physically_incident_air_bins_marked_tir"] == 0
    identity = transmit_vector_field_planar_interface(
        ex, ey, ez, dx, dx, FresnelInterfaceConfig(1029e-9, 1.0, 1.0)
    )
    np.testing.assert_allclose(identity.Ex, ex, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(identity.Ey, ey, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(identity.Ez, ez, rtol=0.0, atol=1e-13)


def test_fresnel_basis_independence_air_glass_and_tir_controls() -> None:
    fresnel = _json(VALIDATION / "phase2c_solver_validation.json")["fresnel"]
    assert fresnel["normal_incidence_basis_independence_residual"] <= 1e-12
    assert fresnel["glass_to_air_evanescent_bin_count"] > 0
    assert fresnel["checks"]["air_to_glass_has_no_false_tir"] is True
    assert fresnel["checks"]["spectral_transverse_wavevector_preserved"] is True


def test_required_phase2c_machine_outputs_and_document_exist() -> None:
    for name in (
        "phase2c_case_summary.csv",
        "phase2c_objective_benchmark.csv",
        "phase2c_interface_benchmark.csv",
        "phase2c_claim_registry.csv",
        "phase2c_solver_validation.json",
        "phase2c_quadrature_convergence.csv",
        "phase2c_figure_manifest.json",
        "phase2c_outcome_report.json",
    ):
        assert (VALIDATION / name).is_file()
    assert (ROOT / "docs" / "92_phase2c_vectorial_objective_and_interface.md").is_file()


def test_all_five_canonical_cases_are_classified_with_longitudinal_fractions() -> None:
    objective = _rows(VALIDATION / "phase2c_objective_benchmark.csv")
    interface = _rows(VALIDATION / "phase2c_interface_benchmark.csv")
    assert tuple(row["case_id"] for row in objective) == PHASE2C_CASE_IDS
    assert tuple(row["case_id"] for row in interface) == PHASE2C_CASE_IDS
    assert all(float(row["longitudinal_power_fraction"]) > 0.0 for row in objective)
    assert all(row["scalar_longitudinal_field_status"] == "not_modelled" for row in objective)
    assert all(row["scalar_model_longitudinal_status"] == "not_modelled" for row in objective)
    assert all(float(row["vector_transversality_residual"]) <= 1e-12 for row in objective)
    required = {
        "scalar_feature_radius_um",
        "vector_feature_radius_um",
        "scalar_side_lobe_ratio",
        "vector_side_lobe_ratio",
    }
    assert required <= set(objective[0])


@pytest.mark.parametrize("case_id", PHASE2C_CASE_IDS)
def test_canonical_case_execution_and_plane_matching(case_id: str) -> None:
    summary = {row["case_id"]: row for row in _rows(VALIDATION / "phase2c_case_summary.csv")}
    row = summary[case_id]
    assert row["metrics_computed_on_native_arrays"].lower() == "true"
    assert row["display_interpolation_used_for_metrics"].lower() == "false"
    assert row["mapping_mode"] == "fixed_physical_optics"


def test_vortex_winding_and_dark_core_metrics_are_reported() -> None:
    rows = {row["case_id"]: row for row in _rows(VALIDATION / "phase2c_objective_benchmark.csv")}
    for case_id, charge in (("V1", 1.0), ("V3", 3.0)):
        row = rows[case_id]
        assert float(row["requested_topological_charge"]) == charge
        assert float(row["measured_scalar_winding"]) == pytest.approx(charge, abs=0.15)
        assert float(row["measured_vector_transverse_winding"]) == pytest.approx(charge, abs=0.15)
        assert float(row["dark_core_radius_scalar_um"]) > 0.0
        assert float(row["dark_core_radius_vector_um"]) > 0.0


def test_interface_schema_identity_and_power_metrics() -> None:
    rows = _rows(VALIDATION / "phase2c_interface_benchmark.csv")
    required = {
        "post_interface_L2_intensity_error",
        "post_interface_feature_radius_scalar_um",
        "post_interface_feature_radius_vector_um",
        "relative_feature_radius_difference",
        "s_power_fraction",
        "p_power_fraction",
        "power_weighted_mean_incidence_angle_deg",
        "transmitted_transversality_residual",
        "evanescent_power_fraction",
        "interface_model_classification",
    }
    assert required <= set(rows[0])
    assert all(float(row["identity_interface_relative_field_residual"]) <= 1e-12 for row in rows)
    assert all(float(row["transmitted_transversality_residual"]) <= 1e-10 for row in rows)
    assert all(row["phase2a_ledger_factor_applied_to_benchmark_fields"].lower() == "false" for row in rows)
    assert all(row["finite_comparison_crop_used_as_fresnel_input"].lower() == "false" for row in rows)
    assert all(int(row["interface_native_grid_n"]) == 1024 for row in rows)
    assert all(
        float(row["maximum_incidence_angle_deg"])
        <= float(row["discrete_grid_support_limit_incidence_angle_deg"]) + 1e-12
        for row in rows
    )


def test_solver_validation_and_canonical_convergence_pass() -> None:
    validation = _json(VALIDATION / "phase2c_solver_validation.json")
    assert validation["all_passed"] is True
    assert validation["debye"]["all_passed"] is True
    assert validation["fresnel"]["all_passed"] is True
    assert validation["canonical_grid_convergence"]["all_passed"] is True
    assert all(row["passed"] for row in validation["canonical_grid_convergence"]["rows"])


def test_three_level_quadrature_convergence_csv_uses_predeclared_limits() -> None:
    rows = _rows(VALIDATION / "phase2c_quadrature_convergence.csv")
    assert [row["level"] for row in rows] == ["low", "medium", "high"]
    assert [(int(row["quadrature_order_r"]), int(row["quadrature_order_phi"])) for row in rows] == [
        (24, 72),
        (32, 96),
        (48, 144),
    ]
    for row in rows[1:]:
        assert float(row["correlation_change_from_previous"]) <= CONVERGENCE_CORRELATION_CHANGE_MAX
        assert float(row["feature_radius_relative_change_from_previous"]) <= CONVERGENCE_FEATURE_RELATIVE_MAX
        assert float(row["longitudinal_absolute_change_from_previous"]) <= CONVERGENCE_LONGITUDINAL_ABSOLUTE_MAX
        assert row["meets_predeclared_change_limits"].lower() == "true"


def test_outcome_is_predeclared_and_upstream_outputs_are_unchanged() -> None:
    report = _json(VALIDATION / "phase2c_outcome_report.json")
    manifest = _json(VALIDATION / "phase2c_figure_manifest.json")
    assert report["outcome"] in PHASE2C_ALLOWED_OUTCOMES
    assert report["outcome"] == "PHASE2C-B"
    assert report["upstream_outputs_unchanged"] is True
    assert manifest["upstream_hashes_before"] == manifest["upstream_hashes_after"]
    assert report["phase2a_energy_ledger_modified"] is False
    assert report["phase2a_energy_factor_reapplied"] is False


def test_h1_remains_strict_hexagonal_with_stable_ridge_and_vector_sensitive_edge() -> None:
    report = _json(VALIDATION / "phase2c_outcome_report.json")
    assert report["H1_remains_strict_hexagonal_scalar"] is True
    assert report["H1_remains_strict_hexagonal_vector"] is True
    assert report["H1_ridge_width_relative_change"] <= 0.01
    assert report["H1_edge_sharpness_relative_change"] > 0.10
    assert np.isfinite(float(report["H1_transition_width_relative_change"]))


def test_h1_required_shape_and_interface_change_metrics_exist() -> None:
    objective = next(
        row for row in _rows(VALIDATION / "phase2c_objective_benchmark.csv") if row["case_id"] == "H1"
    )
    interface = next(
        row for row in _rows(VALIDATION / "phase2c_interface_benchmark.csv") if row["case_id"] == "H1"
    )
    assert {
        "scalar_C6", "vector_C6", "scalar_C3", "vector_C3",
        "scalar_strict_hexagon", "vector_strict_hexagon",
        "scalar_edge_sharpness", "vector_edge_sharpness",
        "scalar_ridge_width", "vector_ridge_width",
        "scalar_transition_width", "vector_transition_width",
        "local_vector_purity_at_pupil", "longitudinal_power_fraction_at_focus",
    } <= set(objective)
    assert {
        "H1_sector_intensity_balance_relative_change",
        "H1_interface_C6_change",
        "H1_interface_C3_change",
        "H1_interface_edge_sharpness_relative_change",
        "H1_interface_ridge_width_relative_change",
        "H1_interface_transition_width_relative_change",
    } <= set(interface)


def test_figure_manifest_reuses_phase2b_and_keeps_metrics_native() -> None:
    manifest = _json(VALIDATION / "phase2c_figure_manifest.json")
    assert manifest["phase2b_plot_infrastructure_reused"] is True
    assert manifest["all_metrics_native"] is True
    assert manifest["display_interpolation_used_for_metrics"] is False
    assert manifest["figure_count"] == 25
    assert manifest["interactive_asset_count"] == 1
    assert manifest["mapping_mode"] == "fixed_physical_optics"
    performance = manifest["sampling_and_performance"]
    assert performance["cpu_implementation_authoritative"] is True
    assert performance["peak_estimated_memory_bytes"] > 0
    assert performance["native_input_resolution"] == [1024, 1024]
    assert performance["output_point_chunking"]["forbidden_four_dimensional_allocation_used"] is False
    for row in manifest["figures"]:
        assert Path(row["png_path"]).is_file()
        assert Path(row["pdf_path"]).is_file()
        assert row["display_interpolation_used_for_metrics"] is False


def test_h1_3d_outputs_are_pure_fixed_plane_intensity_surfaces() -> None:
    manifest = _json(VALIDATION / "phase2c_figure_manifest.json")
    surfaces = [row for row in manifest["figures"] if row["figure_id"].startswith("H1_3d_")]
    assert len(surfaces) == 5
    for row in surfaces:
        assert "fixed-plane intensity surface" in row["render_method"]
        assert "height and colour are identical" in row["normalisation_policy"]
        assert "linear peak-normalised intensity" in row["normalisation_policy"]
        assert row["display_interpolation"] == "local complex-field Fourier synthesis x8; metrics remain native"
        assert int(row["display_resampling_factor"]) == 8
        assert float(row["render_dx_m"]) == pytest.approx(float(row["native_dx_m"]) / 8.0)
        assert row["native_samples_preserved"] is True
        with Image.open(row["png_path"]) as image:
            assert image.width >= 3000
            assert image.height >= 2400
    interactive = next(row for row in surfaces if row["figure_id"] == "H1_3d_h1_vector_debye")
    assert interactive["interactive_default_view"] == "exact top-down heatmap parity"
    assert interactive["interactive_alternate_view"] == "perspective oblique 3D"
    html_path = Path(interactive["interactive_html_path"])
    assert html_path.is_file()
    html = html_path.read_text(encoding="utf-8")
    assert "Linear parity" in html
    assert "Shape emphasis" in html
    assert "Top-down parity" in html
    assert '"type":"heatmap"' in html
    assert '"zsmooth":false' in html
    assert "rgb(0, 0, 4)" in html
    assert "render 0.0129 um" in html
    assert "<script src=" not in html.lower()


def test_bandlimited_surface_resampling_preserves_native_intensity_samples() -> None:
    n = 32
    dx = 0.2e-6
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    ex = np.exp(-(X**2 + Y**2) / (2.0e-6**2)) * np.exp(1j * 2.0e6 * X)
    ey = 0.35j * ex * np.exp(-1j * 1.2e6 * Y)
    native = np.abs(ex) ** 2 + np.abs(ey) ** 2
    rendered, rendered_x = _bandlimited_local_render_intensity((ex, ey), x, 8, 2.0e-6)
    native_selected = np.flatnonzero(np.abs(x) <= float(np.max(np.abs(rendered_x))) + 1e-15)
    centre = rendered_x.size // 2
    render_selected = centre + (native_selected - n // 2) * 8
    np.testing.assert_allclose(
        rendered[np.ix_(render_selected, render_selected)],
        native[np.ix_(native_selected, native_selected)],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(rendered_x[render_selected], x[native_selected], rtol=0.0, atol=1e-15)


def test_interface_fourier_resampling_preserves_native_complex_samples() -> None:
    n = 32
    dx = 0.4e-6
    native_x = (np.arange(n) - n // 2) * dx
    target_x = (np.arange(-24, 25) * dx / 2.0).astype(float)
    X, Y = np.meshgrid(native_x, native_x, indexing="xy")
    field = np.exp(-(X**2 + Y**2) / (3.0e-6**2)) * np.exp(1j * (1.0e6 * X - 0.6e6 * Y))
    rendered = _fourier_resample_complex_plane(field, native_x, target_x)
    native_indices = np.flatnonzero(np.abs(native_x) <= np.max(np.abs(target_x)) + 1e-15)
    render_indices = np.flatnonzero(np.isclose(np.mod(np.arange(target_x.size) - target_x.size // 2, 2), 0))
    np.testing.assert_allclose(
        rendered[np.ix_(render_indices, render_indices)],
        field[np.ix_(native_indices, native_indices)],
        rtol=2e-12,
        atol=2e-12,
    )


def test_surface_crop_rejects_mismatched_render_axis() -> None:
    with pytest.raises(ValueError, match="does not match coordinate axis length"):
        _surface_crop(np.ones((128, 128)), np.linspace(-1.0, 1.0, 32), 0.5)


def test_plotly_magma_scale_is_dense_and_spans_unit_interval() -> None:
    scale = _plotly_magma_colorscale()
    assert len(scale) == 256
    assert scale[0][0] == 0.0
    assert scale[-1][0] == 1.0
    assert scale[0][1] == "rgb(0, 0, 4)"


def test_claim_registry_uses_only_allowed_statuses_and_records_calibration() -> None:
    allowed = {
        "validated",
        "validated_with_scope",
        "approximation_acceptable",
        "approximation_materially_different",
        "superseded",
        "calibration_required",
        "blocked",
    }
    rows = _rows(VALIDATION / "phase2c_claim_registry.csv")
    assert list(rows[0]) == [
        "claim_id",
        "beam_case",
        "comparison_type",
        "previous_scalar_claim",
        "vector_reference_result",
        "benchmark_classification",
        "status",
        "quantitative_valid",
        "calibration_required",
        "evidence_path",
        "notes",
    ]
    assert len(rows) >= 12
    assert {row["status"] for row in rows} <= allowed
    assert any(row["status"] == "approximation_materially_different" for row in rows)
    assert any(
        row["status"] == "approximation_materially_different" and row["beam_case"] == "H1"
        for row in rows
    )
    assert any(row["calibration_required"].lower() == "true" for row in rows)


def test_phase1_and_phase2a_non_regression_contracts_remain_declared() -> None:
    phase1 = _json(VALIDATION.parent / "phase1_critical_repairs" / "phase1_repair_summary.json")
    repairs = {row["repair_id"]: row for row in phase1["repairs"]}
    assert repairs["P1A"]["safe_default"] == "preserve_vortex"
    assert repairs["P1B"]["ring_formula"] == "wavelength_m * f_lens_m * kr_m_inv / (2*pi)"
    assert repairs["P1B"]["carrier_formula"] == "wavelength_m * f_lens_m * carrier_cpm"
    assert repairs["P1C"]["quantitative_limit_fraction"] == 0.05
    assert repairs["P1D"]["fixed_mode"] == "fixed_physical_optics"
    phase2a = _json(VALIDATION.parent / "phase2a" / "canonical_hardware_manifest.json")
    assert phase2a["mapping_mode"] == "fixed_physical_optics"


def test_upstream_hash_registry_covers_phase1_phase2a_and_phase2b() -> None:
    manifest = _json(VALIDATION / "phase2c_figure_manifest.json")
    paths = tuple(manifest["upstream_hashes_before"])
    assert any("phase1_critical_repairs" in path for path in paths)
    assert any("phase2a" in path for path in paths)
    assert any("phase2b_visual_diagnostics" in path for path in paths)
    assert manifest["upstream_hashes_before"] == manifest["upstream_hashes_after"]
