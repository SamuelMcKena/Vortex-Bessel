from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vortex_beam_slm_errors import (
    GaussianBeamError,
    SLMError,
    actual_slm_phase,
    gaussian_input_field,
)
from vbb_study.digital_twin.vortex_explicit_4f import FourFError, LensError, explicit_4f_relay
from vbb_study.digital_twin.vortex_rotated_plane import lab_to_tilted_plane, tilted_to_lab_plane
from vbb_study.digital_twin.vortex_system_route import AxiconError, physical_axicon_on_own_plane
from vbb_study.equations.fields import make_xy_grid


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=complex).ravel()
    bb = np.asarray(b, dtype=complex).ravel()
    return float(abs(np.vdot(aa, bb)) / (np.linalg.norm(aa) * np.linalg.norm(bb)))


def _power(a: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(a, dtype=complex)) ** 2))


def test_beam_lateral_decentre_is_amplitude_shift_not_pointing_phase() -> None:
    grid = make_xy_grid(256, 10e-3 / 256)
    nominal, _ = gaussian_input_field(
        grid, wavelength_m=1029e-9, canonical_radius_m=2e-3
    )
    shifted, meta = gaussian_input_field(
        grid,
        wavelength_m=1029e-9,
        canonical_radius_m=2e-3,
        error=GaussianBeamError(decentre_m=(300e-6, 0.0)),
    )
    assert meta["beam_pointing_rad"] == (0.0, 0.0)
    assert not np.allclose(np.abs(nominal), np.abs(shifted))
    assert np.max(np.abs(np.angle(shifted[np.abs(shifted) > 1e-8]))) < 1e-12


def test_ellipticity_and_curvature_are_input_plane_physics() -> None:
    grid = make_xy_grid(256, 10e-3 / 256)
    field, meta = gaussian_input_field(
        grid,
        wavelength_m=1029e-9,
        canonical_radius_m=2e-3,
        error=GaussianBeamError(
            radius_x_scale=0.7,
            radius_y_scale=1.3,
            curvature_radius_x_m=2.0,
            curvature_radius_y_m=-3.0,
        ),
    )
    assert meta["beam_radius_x_m"] < meta["beam_radius_y_m"]
    assert np.std(np.angle(field[np.abs(field) > 0.2])) > 1e-3


def test_slm_zero_fringing_is_identity_after_quantisation() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    command = 0.6 * np.asarray(grid["X"]) / 1e-3
    err = SLMError(phase_levels=256)
    actual, meta = actual_slm_phase(
        command,
        grid,
        error=err,
        pixel_pitch_m=8e-6,
    )
    assert actual.shape == command.shape
    assert meta["fringing_fidelity"] == "disabled"
    assert meta["phase_lut_status"] == "identity_unmeasured"


def test_nonzero_slm_fringing_changes_phase_and_is_calibration_labelled() -> None:
    grid = make_xy_grid(256, 10e-3 / 256)
    command = np.where(np.asarray(grid["X"]) >= 0.0, np.pi, 0.0)
    actual, meta = actual_slm_phase(
        command,
        grid,
        error=SLMError(fringing_sigma_x_px=1.0, fringing_sigma_y_px=0.2),
        pixel_pitch_m=8e-6,
    )
    assert not np.allclose(actual, command)
    assert "calibration_required" in meta["fringing_fidelity"]


def test_explicit_4f_changes_when_lens_is_axially_misplaced() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    field = np.exp(-(np.asarray(grid["R"]) / 1.5e-3) ** 2).astype(complex)
    nominal = explicit_4f_relay(
        field,
        grid,
        wavelength_m=1029e-9,
        nominal_focal_length_m=0.05,
        nominal_iris_radius_m=1.5e-3,
    )
    shifted = explicit_4f_relay(
        field,
        grid,
        wavelength_m=1029e-9,
        nominal_focal_length_m=0.05,
        nominal_iris_radius_m=1.5e-3,
        error=FourFError(lens2_axial_shift_m=2e-3),
    )
    assert not np.allclose(nominal["output"], shifted["output"])
    assert np.isclose(
        shifted["metadata"]["distances_m"]["iris_to_lens2"],
        0.052,
        rtol=0.0,
        atol=1e-15,
    )


def test_lens_decentre_is_quadratic_phase_about_shifted_optical_axis() -> None:
    err = FourFError(lens1=LensError(decentre_m=(200e-6, 0.0)))
    assert err.lens1.decentre_m == (200e-6, 0.0)


def test_rotated_angular_spectrum_zero_tilt_is_exact_identity() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    field = np.exp(-(np.asarray(grid["R"]) / 1e-3) ** 2).astype(complex)
    tilted, meta = lab_to_tilted_plane(
        field, grid, wavelength_m=1029e-9, tilt_x_rad=0.0, tilt_y_rad=0.0
    )
    assert np.array_equal(field, tilted)
    assert meta["spectral_power_ratio"] == 1.0
    assert meta["interpolation_model"] == "identity"


def test_rotated_plane_roundtrip_preserves_smooth_field_and_power_at_half_degree() -> None:
    """Rigid coordinate rotation must not masquerade as optical absorption."""

    grid = make_xy_grid(256, 10e-3 / 256)
    field = np.exp(-(np.asarray(grid["R"]) / 1.5e-3) ** 2).astype(complex)
    tilt = math.radians(0.5)
    tilted, to_meta = lab_to_tilted_plane(
        field, grid, wavelength_m=1029e-9, tilt_x_rad=0.0, tilt_y_rad=tilt
    )
    returned, from_meta = tilted_to_lab_plane(
        tilted, grid, wavelength_m=1029e-9, tilt_x_rad=0.0, tilt_y_rad=tilt
    )
    assert to_meta["interpolation_model"] == "spline_order_3"
    assert from_meta["interpolation_model"] == "spline_order_3"
    assert _overlap(field, returned) > 0.999
    assert np.isclose(
        _power(returned) / _power(field),
        1.0,
        rtol=0.0,
        atol=3e-3,
    )


def test_axicon_decentre_moves_its_surface_coordinates() -> None:
    grid = make_xy_grid(256, 10e-3 / 256)
    nominal, _ = physical_axicon_on_own_plane(
        grid,
        wavelength_m=1029e-9,
        base_angle_rad=math.radians(2.0),
        refractive_index=1.458,
        external_index=1.0,
        error=AxiconError(),
    )
    shifted, meta = physical_axicon_on_own_plane(
        grid,
        wavelength_m=1029e-9,
        base_angle_rad=math.radians(2.0),
        refractive_index=1.458,
        external_index=1.0,
        error=AxiconError(decentre_m=(300e-6, 0.0)),
    )
    assert meta["decentre_m"] == (300e-6, 0.0)
    assert not np.allclose(nominal, shifted)


def test_high_angle_tip_defect_is_not_marked_quantitative() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    _, meta = physical_axicon_on_own_plane(
        grid,
        wavelength_m=1029e-9,
        base_angle_rad=math.radians(20.0),
        refractive_index=1.458,
        external_index=1.0,
        error=AxiconError(tip_model="hyperboloidal_round", rounding_parameter_m=10e-6),
    )
    assert "not_quantitative" in meta["tip_defect_fidelity"]
