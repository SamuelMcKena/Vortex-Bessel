from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vbb_study.digital_twin.nathan_vector_hexagon import (
    NathanSourceParityConfig,
    source_parity_grid,
)
from vbb_study.digital_twin.phase2e_report_pipeline import (
    SAS_LINE_CORRELATION_FLOOR,
    SCALAR_GRID_RELATIVE_L2_CEILING,
    generate_phase2e_outputs,
)
from vbb_study.digital_twin.phase2e_report_visualisation import _safe_corr
from vbb_study.digital_twin.phase2e_spectral_propagation import (
    DensePropagationMap,
    map_correlation,
    native_line_parity,
)
from vbb_study.equations.fields import fft2c
from vbb_study.equations.propagation import asm_longitudinal_wavenumber_m_inv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "phase2e_report_visualisation"
MANIFEST = OUT / "00_manifest" / "phase2e_figure_manifest.json"
EXPECTED_CASES = {"G0", "B0", "V1", "V3", "H1_CONTINUOUS", "H1_AVERAGED"}
EXPECTED_HEROES = {
    "hero_vortex_beam_family",
    "hero_vortex_parameter_dependence",
    "hero_vortex_ideal_realistic_degraded",
    "hero_h1_continuous_vs_averaged",
    "hero_scalar_vector_objective_benchmark",
    "hero_energy_loss_efficiency",
}
EXPECTED_SWEEPS = {
    "vortex_charge",
    "radial_wavevector",
    "input_beam_radius",
    "propagation_distance",
    "aperture_radius",
    "effective_objective_na",
    "defocus_aberration",
    "error_input_beam_decentre",
    "error_input_beam_tilt",
    "error_slm_phase",
    "error_fourier_iris_offset",
    "error_pupil_decentre",
    "error_axicon_decentre",
    "error_zernike_defocus",
    "error_zernike_astigmatism",
    "error_zernike_coma",
    "error_zernike_spherical",
}
EXPECTED_SURFACES = {
    "surface_b0",
    "surface_v1",
    "surface_v3",
    "surface_h1_continuous",
    "surface_h1_averaged",
}
EXPECTED_PROPAGATION_ATLASES = {
    "propagation_g0",
    "propagation_b0",
    "propagation_v1",
    "propagation_v3",
    "propagation_h1_continuous",
    "propagation_h1_averaged",
}
EXPECTED_BOUNDARY_AUDIT = "propagation_b0_boundary_audit"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _figures() -> list[dict]:
    return _json(MANIFEST)


def test_01_outcome_and_required_families_are_complete() -> None:
    outcome = _json(OUT / "10_final" / "phase2e_outcome_report.json")
    assert outcome["outcome"] == "PHASE2E-A"
    assert outcome["figure_count"] == 47
    assert set(outcome["case_ids"]) == EXPECTED_CASES
    assert set(outcome["hero_figure_ids"]) == EXPECTED_HEROES
    assert set(outcome["sweep_ids"]) == EXPECTED_SWEEPS
    assert set(outcome["pure_3d_case_ids"]) == {
        "B0", "V1", "V3", "H1_CONTINUOUS", "H1_AVERAGED"
    }
    assert outcome["accepted_physics_changed"] is False
    assert outcome["accepted_arrays_overwritten"] is False
    assert outcome["upstream_outputs_unchanged"] is True
    assert outcome["all_dense_propagation_native_parity_valid"] is True
    assert outcome["all_headline_scalar_propagation_valid"] is True
    assert outcome["all_scalar_dense_propagation_grid_converged"] is True
    assert outcome["minimum_dense_propagation_sas_line_correlation"] >= SAS_LINE_CORRELATION_FLOOR
    assert outcome["scalar_grid_relative_l2_ceiling"] == SCALAR_GRID_RELATIVE_L2_CEILING
    assert outcome["propagation_boundary_diagnosis_valid"] is True


def test_02_figure_manifest_has_unique_complete_png_pdf_pairs() -> None:
    figures = _figures()
    assert len(figures) == 47
    assert len({row["figure_id"] for row in figures}) == 47
    assert {row["figure_id"] for row in figures if row["report_role"] == "hero_figure"} == EXPECTED_HEROES
    for row in figures:
        for key in ("png_path", "pdf_path"):
            path = ROOT / row[key]
            assert path.is_file(), path
            assert path.stat().st_size > 10_000, path


def test_03_pngs_have_report_grade_pixel_dimensions() -> None:
    for row in _figures():
        with Image.open(ROOT / row["png_path"]) as image:
            width, height = image.size
        assert width >= 2400, row["figure_id"]
        assert height >= 1200, row["figure_id"]


def test_04_pure_3d_surfaces_share_physical_scaling_contract() -> None:
    records = {row["figure_id"]: row for row in _figures()}
    assert EXPECTED_SURFACES <= records.keys()
    for figure_id in EXPECTED_SURFACES:
        row = records[figure_id]
        assert row["figure_family"] == "pure_3d_intensity"
        assert row["x_unit"] == row["y_unit"] == "mm"
        assert row["z_unit"] == "normalised intensity"
        assert row["z_limits"] == [0.0, 1.0]
        assert row["x_limits"] == row["y_limits"]
        assert row["matched_axes"] is True
        assert "same beam-centred transverse intensity array" in row["data_basis"]
        assert "no propagation axis" in row["notes"].lower()
    scalar_limits = {tuple(records[key]["x_limits"]) for key in ("surface_b0", "surface_v1", "surface_v3")}
    h1_limits = {tuple(records[key]["x_limits"]) for key in ("surface_h1_continuous", "surface_h1_averaged")}
    assert scalar_limits == {(-0.25, 0.25)}
    assert h1_limits == {(-0.3, 0.3)}


def test_05_transverse_rois_and_full_longitudinal_fields_are_distinct() -> None:
    style = _json(OUT / "00_manifest" / "phase2e_style_contract.json")
    assert style["scalar_focus_halfwidth_m"] <= 0.25e-3
    assert style["h1_focus_halfwidth_m"] <= 0.30e-3
    records = {row["figure_id"]: row for row in _figures()}
    assert "central" in records["sweep_effective_objective_na"]["notes"].lower()
    assert "roi inset" in records["sweep_effective_objective_na"]["notes"].lower()
    realism = records["hero_vortex_ideal_realistic_degraded"]
    assert "beam-centred" in realism["notes"].lower()
    assert "insets" in realism["notes"].lower()
    objective = records["hero_scalar_vector_objective_benchmark"]
    assert "stacked vertically" in objective["notes"].lower()
    for case_id in ("b0", "v1", "v3"):
        route = records[f"{case_id}_ideal_realistic_error_atlas"]
        assert route["x_limits"] == route["y_limits"] == [-0.8, 0.8]
        assert "insets" in route["notes"].lower()
    assert EXPECTED_PROPAGATION_ATLASES <= records.keys()
    propagation_root = OUT / "01b_propagation_maps"
    assert not list(propagation_root.glob("*dense_xz_yz_full_field*"))
    assert not list(propagation_root.glob("*linear_slice_normalised*"))
    assert not [
        path
        for path in propagation_root.iterdir()
        if "log" in path.name.lower() or "db" in path.name.lower()
    ]
    for figure_id in EXPECTED_PROPAGATION_ATLASES:
        propagation = records[figure_id]
        assert propagation["figure_family"] == "dense_propagation_full_field"
        assert "1025x601 physical spectral synthesis" in propagation["display_interpolation"]
        assert propagation["x_limits"][1] - propagation["x_limits"][0] >= 9.9
        assert propagation["y_limits"] == [0.0, 200.0]
        assert propagation["roi_occupancy"] == {}
        assert propagation["linear_log_mode"] == "linear throughout"
        assert "linear I/global Imax" in propagation["normalisation_policy"]
        assert "global_linear_dual_range" in propagation["png_path"]
        combined_contract = " ".join(
            str(propagation[key])
            for key in ("png_path", "pdf_path", "normalisation_policy", "linear_log_mode")
        ).lower()
        assert "log10" not in combined_contract
        assert "db" not in combined_contract
        assert "gamma" not in combined_contract
        assert "no propagation roi" in propagation["notes"].lower()
        if figure_id in {"propagation_g0", "propagation_b0", "propagation_v1", "propagation_v3"}:
            assert "accepted finite-aperture" in propagation["data_basis"]
            assert "accepted finite-aperture realistic fixed-bench route" in propagation["notes"].lower()
    for case_id in ("g0", "b0", "v1", "v3"):
        core = records[f"core_{case_id}"]
        assert "accepted finite-aperture realistic fixed-bench field" in core["data_basis"]
        assert "accepted hard objective-pupil edge" in core["notes"].lower()
    assert EXPECTED_BOUNDARY_AUDIT in records
    assert records[EXPECTED_BOUNDARY_AUDIT]["figure_family"] == "propagation_boundary_truth_audit"


def test_06_display_interpolation_never_supplies_metrics() -> None:
    figures = _figures()
    assert all(row["display_interpolation_used_for_metrics"] is False for row in figures)
    assert all(row["metrics_computed_on_native_arrays"] is True for row in figures if row["metric_bearing"])
    cases = _json(OUT / "09_summary" / "phase2e_case_summary.json")
    assert all(row["metrics_computed_on_native_arrays"] is True for row in cases)
    assert all(row["display_interpolation_used_for_metrics"] is False for row in cases)


def test_07_sweeps_are_valid_and_remain_diagnostic() -> None:
    rows = _json(OUT / "09_summary" / "phase2e_sweep_summary.json")
    assert len(rows) == 85
    assert {row["sweep_id"] for row in rows} == EXPECTED_SWEEPS
    assert all(row["nyquist_pass"] is True for row in rows)
    assert all(row["sas_valid"] is True for row in rows)
    assert all(row["metrics_computed_on_native_sas_array"] is True for row in rows)
    assert all(row["display_interpolation_used_for_metrics"] is False for row in rows)
    base = [row for row in rows if not row["sweep_id"].startswith("error_")]
    errors = [row for row in rows if row["sweep_id"].startswith("error_")]
    assert len(base) == 35
    assert len(errors) == 50
    assert all(row["maturity"] == "diagnostic_screening_only" for row in base)
    assert all(row["maturity"] == "diagnostic_physical_error_sweep" for row in errors)
    assert max(float(row["morphology_relative_l2_to_baseline"]) for row in errors) > 0.7
    assert max(float(row["centroid_shift_m"]) for row in errors) > 0.25e-3
    assert all(row["accepted_result_replaced"] is False for row in rows)


def test_08_all_accepted_endpoints_are_reproduced_including_h1() -> None:
    rows = _json(OUT / "09_summary" / "phase2e_endpoint_reproduction_audit.json")
    assert len(rows) == 66
    assert all(row["reproduced"] is True for row in rows)
    joined = " ".join(str(value) for row in rows for value in row.values()).lower()
    assert "continuous" in joined
    assert "averaged" in joined


def test_09_upstream_hashes_still_match_current_files() -> None:
    expected = _json(OUT / "00_manifest" / "phase2e_upstream_hashes.json")
    assert expected
    for relative_path, digest in expected.items():
        path = ROOT / relative_path
        assert path.is_file(), path
        assert _sha256(path) == digest, relative_path


def test_10_artifact_manifest_hashes_every_declared_output() -> None:
    manifest = _json(OUT / "00_manifest" / "phase2e_artifact_manifest.json")
    rows = manifest["artifacts"]
    assert manifest["artifact_count_excluding_manifest"] == len(rows) == 109
    for row in rows:
        path = OUT / row["relative_path"]
        assert path.is_file(), path
        assert path.stat().st_size == row["size_bytes"]
        assert _sha256(path) == row["sha256"]


def test_11_existing_output_requires_explicit_overwrite() -> None:
    with pytest.raises(FileExistsError, match="pass overwrite=True explicitly"):
        generate_phase2e_outputs(output_root=OUT, overwrite=False)


def test_12_report_states_scope_and_display_boundaries() -> None:
    text = (ROOT / "docs" / "94_phase2e_report_visualisation_and_parameter_sweeps.md").read_text(encoding="utf-8")
    assert "Transverse ROI occupancy is presentation-governed" in text
    assert "No x-z/y-z ROI is applied" in text
    assert "global\n  linear `I/Imax`" in text
    assert "No per-z renormalisation" in text
    assert "globally linear shared-peak" in text
    assert "logarithm, gamma" in text
    assert "1.8 mm hard pupil" in text
    assert "finite-aperture effects" in text
    assert "All metrics are computed before display" in text
    assert "not calibrated experimental predictions" in text
    assert "nonlinear material modification" in text


def test_13_dense_map_case_audit_passes_independent_quality_gates() -> None:
    rows = _json(OUT / "09_summary" / "phase2e_case_summary.json")
    assert {row["case_id"] for row in rows} == EXPECTED_CASES
    for row in rows:
        assert int(row["propagation_transverse_samples"]) == 1025
        assert int(row["propagation_z_samples"]) == 601
        assert float(row["propagation_native_parity_error"]) <= 1.0e-10
        if row["case_id"] in {"G0", "B0", "V1", "V3"}:
            assert row["propagation_sas_line_validation_applicability"] == "scalar_same_source_sas"
            assert float(row["propagation_sas_x_line_correlation"]) >= SAS_LINE_CORRELATION_FLOOR
            assert float(row["propagation_sas_y_line_correlation"]) >= SAS_LINE_CORRELATION_FLOOR
            assert int(row["propagation_source_grid_n"]) == 1024
            assert int(row["propagation_convergence_grid_n"]) == 768
            assert float(row["propagation_xz_grid_convergence_correlation"]) >= 0.98
            assert float(row["propagation_yz_grid_convergence_correlation"]) >= 0.98
            assert float(row["propagation_xz_grid_convergence_relative_l2"]) <= SCALAR_GRID_RELATIVE_L2_CEILING
            assert float(row["propagation_yz_grid_convergence_relative_l2"]) <= SCALAR_GRID_RELATIVE_L2_CEILING
        else:
            assert row["propagation_sas_x_line_correlation"] is None
            assert row["propagation_sas_y_line_correlation"] is None
            assert row["propagation_sas_line_validation_applicability"] == "not_applicable_to_projected_vector_route"
            assert float(row["propagation_highn_cross_grid_sas_x_line_correlation"]) >= 0.80
            assert float(row["propagation_highn_cross_grid_sas_y_line_correlation"]) >= 0.80


def test_16_propagation_boundary_audit_is_physically_diagnostic() -> None:
    audit = _json(OUT / "09_summary" / "phase2e_propagation_boundary_audit.json")
    assert float(audit["hard_pupil_power_fraction"]) < 0.85
    assert 0.105 <= float(audit["geometric_pupil_bessel_zone_m"]) <= 0.120
    assert 0.120 <= float(audit["gaussian_radius_bessel_zone_m"]) <= 0.130
    assert float(audit["bandlimited_to_unbandlimited_on_axis_correlation"]) >= 0.9999
    assert float(audit["bandlimited_to_unbandlimited_max_abs_normalised_difference"]) <= 1.0e-3
    assert float(audit["hard_pupil"]["ripple_rms_normalised"]) >= 5.0 * float(
        audit["ideal_untruncated"]["ripple_rms_normalised"]
    )
    assert float(audit["hard_pupil_to_realistic_on_axis_correlation"]) >= 0.99


def test_17_headline_scalar_propagation_is_clean_linear_and_full_field() -> None:
    rows = _json(OUT / "09_summary" / "phase2e_headline_propagation_summary.json")
    assert {row["case_id"] for row in rows} == {"G0", "B0", "V1", "V3"}
    for row in rows:
        assert row["hard_pupil_active"] is True
        assert row["slm_or_4f_errors_active"] is True
        assert row["accepted_fixed_bench_route"] is True
        assert int(row["source_grid_n"]) == 1024
        assert int(row["transverse_samples"]) == 1025
        assert int(row["z_samples"]) == 601
        assert float(row["x_max_m"]) - float(row["x_min_m"]) >= 9.9e-3
        assert float(row["z_max_m"]) - float(row["z_min_m"]) >= 0.2
        assert float(row["native_line_max_abs_intensity_error"]) <= 1.0e-10
        assert row["display_intensity_mapping"] == "paired global linear I/Imax at 0--1 and 0--0.01 colour ranges"
        assert row["axial_scale_display"] == "shared z-peak / global Imax, linear"
        assert row["per_z_renormalisation"] is False
        assert row["display_spatial_interpolation"] == "none"


def test_14_axis_sampled_direct_synthesis_matches_native_inverse_fft() -> None:
    grid = source_parity_grid(NathanSourceParityConfig(grid_n=64))
    field = np.exp(-np.asarray(grid["R"]) ** 2 / (1.0e-3**2)) * np.exp(
        3j * np.asarray(grid["PHI"])
    )
    kz = asm_longitudinal_wavenumber_m_inv(
        np.asarray(grid["FX"]),
        np.asarray(grid["FY"]),
        wavelength_m=1.029e-6,
        include_evanescent=True,
    )
    parity = native_line_parity(
        (fft2c(field),),
        kz,
        grid,
        1.029e-6,
        z_m=60.0e-3,
    )
    assert parity["native_line_max_abs_intensity_error"] <= 1.0e-12
    assert parity["native_line_intensity_correlation"] >= 1.0 - 1.0e-12


def test_15_correlation_audits_never_mutate_intensity_maps() -> None:
    x = np.linspace(-1.0, 1.0, 64)
    z = np.linspace(0.0, 1.0, 32)
    base = np.exp(-((z[:, None] - 0.45) / 0.25) ** 2) * (
        0.15 + np.cos(4.0 * np.pi * x)[None, :] ** 2
    )
    comparison = np.roll(base, 1, axis=1)
    first = DensePropagationMap(x, x.copy(), z, base.copy(), base.copy(), {})
    second = DensePropagationMap(x, x.copy(), z, comparison.copy(), comparison.copy(), {})
    first_before = first.xz_intensity.copy()
    second_before = second.xz_intensity.copy()
    map_correlation(first, second)
    assert np.array_equal(first.xz_intensity, first_before)
    assert np.array_equal(second.xz_intensity, second_before)

    row_source = base.copy()
    row_before = row_source.copy()
    _safe_corr(row_source[12], comparison[12])
    assert np.array_equal(row_source, row_before)
