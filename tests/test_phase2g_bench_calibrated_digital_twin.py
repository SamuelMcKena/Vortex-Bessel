from __future__ import annotations

import math

import numpy as np
import pytest

from vbb_study.calibration.bench_binding import bind_calibration_to_manifest
from vbb_study.calibration.camera_comparison import (
    CameraCalibration,
    compare_simulation_to_camera,
    fit_gaussian_2d,
)
from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.calibration.shack_hartmann import (
    correction_phase_on_slm,
    reconstruct_opd_from_slopes,
)
from vbb_study.calibration.slm_phase import (
    SLMPhaseCalibration,
    calibrated_phase_to_grey,
    physical_carrier_phase,
    slm_device_grid,
)
from vbb_study.calibration.validation import validate_calibration_bundle
from vbb_study.digital_twin.bench_calibrated_route import (
    BenchCalibratedInputs,
    build_bench_calibrated_route,
)
from vbb_study.digital_twin.calibrated_material_response import (
    fit_binary_material_response,
    modification_probability,
)
from vbb_study.digital_twin.objective_sample_route import (
    ObjectiveSampleConfig,
    focus_pupil_field_into_sample,
)
from vbb_study.digital_twin.ultrafast_exposure import (
    GaussianPulse,
    gaussian_peak_intensity_from_fluence_W_cm2,
    gaussian_spectral_grid,
    pulse_spacing_m,
)
from vbb_study.vector_arm_config import SLMPanelConfig


TWOPI = 2.0 * np.pi


def _linear_lut(panel_id: str = "synthetic") -> SLMPhaseCalibration:
    grey = np.arange(256, dtype=float)
    phase = TWOPI * grey / 255.0
    return SLMPhaseCalibration(
        panel_id=panel_id,
        wavelength_m=1029e-9,
        grey_levels=grey,
        phase_rad=phase,
        calibration_date="synthetic",
    )


def test_physical_carrier_is_exactly_20_pixels() -> None:
    panel = SLMPanelConfig()
    assert panel.carrier_lp_per_mm == pytest.approx(6.25)
    assert panel.carrier_period_px == pytest.approx(20.0)
    grid = slm_device_grid(panel)
    x = np.asarray(grid["x"], dtype=float)
    phase = physical_carrier_phase(x, panel=panel)
    assert phase[20] - phase[0] == pytest.approx(TWOPI, rel=0.0, abs=1e-12)


def test_measured_lut_round_trip_and_full_stroke_gate() -> None:
    desired = np.linspace(0.0, TWOPI, 1000, endpoint=False)
    result = calibrated_phase_to_grey(desired, _linear_lut())
    assert result.grey_u8.dtype == np.uint8
    assert float(np.sqrt(np.mean(result.phase_error_rad**2))) < 0.01

    grey = np.arange(256, dtype=float)
    bad = SLMPhaseCalibration(
        panel_id="bad",
        wavelength_m=1029e-9,
        grey_levels=grey,
        phase_rad=1.8 * math.pi * grey / 255.0,
    )
    with pytest.raises(ValueError, match=r"below 2\*pi"):
        calibrated_phase_to_grey(desired, bad)


def test_shack_hartmann_reconstructs_quadratic_opd() -> None:
    x = np.linspace(-2e-3, 2e-3, 31)
    y = np.linspace(-1.5e-3, 1.5e-3, 25)
    X, Y = np.meshgrid(x, y, indexing="xy")
    a = 1.7e-3
    b = -0.8e-3
    c = 0.6e-3
    W = a * X**2 + b * Y**2 + c * X * Y
    sx = 2.0 * a * X + c * Y
    sy = 2.0 * b * Y + c * X
    rec = reconstruct_opd_from_slopes(sx, sy, x, y)
    expected = W - float(np.mean(W))
    assert np.sqrt(np.mean((rec.opd_m - expected) ** 2)) < 1e-12
    assert rec.residual_rms_m < 1e-13


def test_shack_hartmann_correction_cancels_opd_phase() -> None:
    x = np.linspace(-1e-3, 1e-3, 21)
    y = np.linspace(-1e-3, 1e-3, 21)
    X, Y = np.meshgrid(x, y, indexing="xy")
    W = 2e-4 * (X**2 - 0.5 * Y**2)
    sx = 4e-4 * X
    sy = -2e-4 * Y
    rec = reconstruct_opd_from_slopes(sx, sy, x, y)
    correction = correction_phase_on_slm(rec, x, y, X, Y, wavelength_m=1029e-9)
    residual = TWOPI * rec.opd_m / 1029e-9 + correction
    assert np.max(np.abs(residual)) < 1e-12


def test_camera_gaussian_fit_and_identity_comparison() -> None:
    q = 5e-6
    cal = CameraCalibration(object_plane_scale_m_per_pixel=q)
    ny, nx = 181, 201
    x = (np.arange(nx) - (nx - 1) / 2) * q
    y = (np.arange(ny) - (ny - 1) / 2) * q
    X, Y = np.meshgrid(x, y, indexing="xy")
    x0, y0 = 70e-6, -45e-6
    wx, wy = 220e-6, 140e-6
    theta = 0.31
    c, s = math.cos(theta), math.sin(theta)
    xr = c * (X - x0) + s * (Y - y0)
    yr = -s * (X - x0) + c * (Y - y0)
    image = 2.3 * np.exp(-2.0 * (xr**2 / wx**2 + yr**2 / wy**2)) + 0.07
    fit = fit_gaussian_2d(image, cal)
    assert fit.success
    assert fit.x0_m == pytest.approx(x0, abs=3e-6)
    assert fit.y0_m == pytest.approx(y0, abs=3e-6)
    assert sorted([fit.wx_m, fit.wy_m]) == pytest.approx(sorted([wx, wy]), rel=0.03)

    comparison = compare_simulation_to_camera(image - 0.07, cal, image - 0.07, x, y)
    assert comparison.metrics["energy_normalised_correlation"] > 0.999999
    assert comparison.metrics["energy_normalised_l2"] < 1e-6


def test_calibration_schema_preserves_old_bundle_and_20px_contract() -> None:
    current = canonical_calibration_template()
    assert current["slm"]["carrier_period_px"] == 20
    assert validate_calibration_bundle(CalibrationBundle(current)).valid_schema
    old = {k: v for k, v in current.items() if k not in {"wavefront_sensor", "temporal"}}
    old["schema_version"] = "1.0"
    assert validate_calibration_bundle(CalibrationBundle(old)).valid_schema


def test_bench_binding_keeps_6250_cpm_carrier() -> None:
    data = canonical_calibration_template()
    bundle = CalibrationBundle(data)
    bound = bind_calibration_to_manifest(bundle)
    carrier = next(row for row in bound.manifest["parameters"] if row["parameter"] == "carrier_frequency_cpm")
    assert carrier["value"] == pytest.approx(6250.0)


def test_bench_calibrated_nominal_route_smoke() -> None:
    bundle = CalibrationBundle(canonical_calibration_template())
    route = build_bench_calibrated_route(
        "V1",
        grid_n=128,
        calibrated=BenchCalibratedInputs(calibration_bundle=bundle),
        window_m=10e-3,
    )
    assert route["post_axicon"].shape == (128, 128)
    assert route["metadata"]["carrier_period_px"] == pytest.approx(20.0)
    assert route["metadata"]["axicon_tilt_status"].startswith("none")


def test_objective_sample_vector_route_energy_and_longitudinal_field() -> None:
    n = 64
    radius = 1.8e-3
    x = np.linspace(-2.0e-3, 2.0e-3, n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    pupil = np.exp(-(X**2 + Y**2) / (1.2e-3**2)).astype(complex)
    grid = {"x": x, "dx": float(x[1] - x[0])}
    result = focus_pupil_field_into_sample(
        pupil,
        grid,
        config=ObjectiveSampleConfig(
            wavelength_m=1029e-9,
            numerical_aperture=0.45,
            objective_focal_length_m=4e-3,
            objective_pupil_radius_m=radius,
            incident_medium_index=1.0,
            sample_refractive_index=1.45,
            sample_depth_m=5e-6,
            fft_pad_factor=1,
        ),
    )
    assert result.focal_plane_air.intensity.shape == (n, n)
    assert result.interface_transmission.diagnostics["lossless_R_plus_T"] == pytest.approx(1.0, abs=1e-9)
    assert result.focal_plane_air.component_power_fractions["Ez_power_fraction"] > 0.0
    assert np.all(np.isfinite(result.field_in_sample.intensity))


def test_absorbing_sample_depth_is_explicitly_blocked() -> None:
    n = 64
    x = np.linspace(-2e-3, 2e-3, n)
    pupil = np.ones((n, n), dtype=complex)
    with pytest.raises(ValueError, match="absorbing-media depth propagation"):
        focus_pupil_field_into_sample(
            pupil,
            {"x": x, "dx": float(x[1] - x[0])},
            config=ObjectiveSampleConfig(
                wavelength_m=1029e-9,
                numerical_aperture=0.45,
                objective_focal_length_m=4e-3,
                objective_pupil_radius_m=1.8e-3,
                sample_refractive_index=1.45 + 1e-5j,
                sample_depth_m=1e-6,
                fft_pad_factor=1,
            ),
        )


def test_ultrafast_gaussian_relations() -> None:
    pulse = GaussianPulse(
        central_wavelength_m=1029e-9,
        intensity_fwhm_s=260e-15,
        pulse_energy_J=10e-6,
        repetition_rate_Hz=100e3,
    )
    spectrum = gaussian_spectral_grid(pulse, n_omega=41)
    spectral_intensity = spectrum.field_amplitude_weight**2
    assert np.trapezoid(spectral_intensity, spectrum.omega_rad_s) == pytest.approx(1.0, rel=1e-6)
    expected_tbw = 2.0 * math.log(2.0) / math.pi
    assert pulse.transform_limited_frequency_fwhm_Hz * pulse.intensity_fwhm_s == pytest.approx(expected_tbw)
    I = gaussian_peak_intensity_from_fluence_W_cm2(1.0, pulse.intensity_fwhm_s)
    assert float(I) == pytest.approx(math.sqrt(4.0 * math.log(2.0) / math.pi) / 260e-15)
    assert pulse_spacing_m(scan_speed_m_s=1e-3, repetition_rate_Hz=100e3) == pytest.approx(10e-9)


def test_material_response_is_empirical_and_requires_mixed_data() -> None:
    with pytest.raises(ValueError, match="both modified and unmodified"):
        fit_binary_material_response(
            np.linspace(0.5, 5.0, 20),
            np.full(20, 10.0),
            np.ones(20),
            material_name="synthetic",
        )

    F = np.geomspace(0.4, 8.0, 80)
    N = np.geomspace(1.0, 100.0, 80)
    score = -3.0 + 1.8 * np.log(F) + 0.45 * np.log(N)
    y = (score > 0.0).astype(float)
    fit = fit_binary_material_response(F, N, y, material_name="synthetic")
    assert fit.converged
    low = float(modification_probability(fit, 0.5, 5.0))
    high = float(modification_probability(fit, 6.0, 50.0))
    assert high > low
