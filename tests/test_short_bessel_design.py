from __future__ import annotations

from dataclasses import replace

import numpy as np

from vbb_study.short_bessel import (
    J0_HALF_INTENSITY_ARGUMENT,
    ShortBesselDesignInput,
    design_short_bessel,
    first_zero_diameter_from_kr_m,
    j0_fwhm_diameter_from_kr_m,
    kr_from_j0_fwhm_diameter_m,
    programmable_axicon_phase_rad,
    slm_phase_preview,
)


UM = 1.0e-6


def test_j0_fwhm_conversion_round_trip() -> None:
    diameter = 5.0 * UM
    kr = kr_from_j0_fwhm_diameter_m(diameter)
    recovered = j0_fwhm_diameter_from_kr_m(kr)

    assert np.isclose(recovered, diameter, rtol=0.0, atol=1e-15)
    assert np.isclose(J0_HALF_INTENSITY_ARGUMENT, 1.126364239377483, rtol=0.0, atol=1e-12)


def test_5um_by_500um_target_matches_expected_inverse_design_numbers() -> None:
    result = design_short_bessel(ShortBesselDesignInput())

    assert np.isclose(result.target_kr_sample_m_inv, 450_545.6957509932, rtol=2e-10)
    assert np.isclose(result.target_first_zero_diameter_um, 10.675168269834575, rtol=2e-10)
    assert np.isclose(result.required_sample_beam_radius_um, 25.4434707689678, rtol=2e-10)
    assert np.isclose(result.required_pre_beam_radius_mm, 3.152430991848659, rtol=2e-10)
    assert np.isclose(result.predicted_length_with_current_beam_um, 317.2155084713136, rtol=2e-10)
    assert np.isclose(result.required_beam_radius_scale_factor, 1.5762154959243295, rtol=2e-10)

    assert result.required_objective_na < result.objective_na_available
    assert result.na_feasible is True
    assert result.aperture_feasible is True
    assert result.target_phase_sampling_feasible is True
    assert result.hard_feasible is True


def test_repo_default_mapping_leaves_useful_slm_aperture_and_sampling_margin() -> None:
    result = design_short_bessel()

    assert np.isclose(result.slm_safe_radius_mm, 3.888, rtol=0.0, atol=1e-12)
    assert result.slm_aperture_margin_mm > 0.70
    assert result.target_pixels_per_phase_wrap > 200.0
    assert result.target_phase_wrap_period_um > 1700.0
    assert result.required_objective_na < 0.08


def test_measured_physical_beam_equal_to_target_requires_no_residual_axicon() -> None:
    cfg = replace(ShortBesselDesignInput(), measured_physical_fwhm_um=5.0)
    result = design_short_bessel(cfg)

    assert result.residual_slm_kr_sample_m_inv is not None
    assert abs(result.residual_slm_kr_sample_m_inv) < 1e-9
    assert abs(result.residual_slm_kr_pre_m_inv) < 1e-9
    assert result.residual_phase_sign == "no_radial_correction_needed"
    assert np.isinf(result.residual_phase_wrap_period_um)
    assert np.isinf(result.residual_pixels_per_phase_wrap)


def test_tighter_physical_beam_requires_opposite_sign_slm_axicon() -> None:
    # A 3 um physical-only core has larger k_r than the requested 5 um core, so
    # the programmable radial phase must subtract/cancel part of the physical
    # axicon's radial wavevector.
    cfg = replace(ShortBesselDesignInput(), measured_physical_fwhm_um=3.0)
    result = design_short_bessel(cfg)

    assert result.physical_kr_sample_m_inv > result.target_kr_sample_m_inv
    assert result.residual_slm_kr_sample_m_inv < 0.0
    assert result.residual_slm_kr_pre_m_inv < 0.0
    assert result.residual_phase_sign == "subtract_radial_wavevector"
    assert result.residual_phase_sampling_feasible is True


def test_broader_physical_beam_requires_same_sign_added_slm_axicon() -> None:
    cfg = replace(ShortBesselDesignInput(), measured_physical_fwhm_um=8.0)
    result = design_short_bessel(cfg)

    assert result.physical_kr_sample_m_inv < result.target_kr_sample_m_inv
    assert result.residual_slm_kr_sample_m_inv > 0.0
    assert result.residual_slm_kr_pre_m_inv > 0.0
    assert result.residual_phase_sign == "add_radial_wavevector"


def test_phase_preview_uses_residual_when_physical_measurement_is_supplied() -> None:
    cfg = replace(ShortBesselDesignInput(), measured_physical_fwhm_um=3.0)
    result = design_short_bessel(cfg)
    preview = slm_phase_preview(result, use_full_resolution=False)

    assert preview["mode"] == "physical_measured_residual"
    assert np.isclose(preview["signed_kr_pre_m_inv"], result.residual_slm_kr_pre_m_inv)
    assert preview["gray_uint8"].dtype == np.uint8
    assert preview["phase_wrapped_rad"].shape == preview["gray_uint8"].shape


def test_signed_phase_addition_is_linear_in_radial_wavevector() -> None:
    r = np.linspace(0.0, 4.0e-3, 1000)
    physical_kr = 4200.0
    slm_kr = -900.0

    combined = programmable_axicon_phase_rad(r, physical_kr) + programmable_axicon_phase_rad(r, slm_kr)
    expected = programmable_axicon_phase_rad(r, physical_kr + slm_kr)

    assert np.allclose(combined, expected, rtol=0.0, atol=1e-14)


def test_first_zero_definition_remains_distinct_from_fwhm_definition() -> None:
    result = design_short_bessel()
    first_zero = first_zero_diameter_from_kr_m(result.target_kr_sample_m_inv) / UM

    assert np.isclose(first_zero, result.target_first_zero_diameter_um)
    assert first_zero > result.target_fwhm_um
