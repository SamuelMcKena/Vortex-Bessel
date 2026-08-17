from __future__ import annotations

import math

import numpy as np
import pytest

from vbb_study.digital_twin.vector_refractive_axicon import (
    fresnel_transmit_vector_3d,
    sample_vector_field_on_tilted_entrance,
)
from vbb_study.digital_twin.vector_refractive_axicon_eikonal import (
    build_tilted_vector_refractive_axicon_field,
    estimate_common_vector_eikonal,
    lab_reference_eikonal_direction_consistency,
)
from vbb_study.digital_twin.vortex_axicon_oblique_reference import refract_direction
from vbb_study.digital_twin.vortex_error_reference_models import snell_axicon_geometry
from vbb_study.digital_twin.vortex_refractive_axicon import RefractiveAxiconGeometry
from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix
from vbb_study.equations.fields import make_xy_grid
from vbb_study.vector_field import VectorField, propagate_vector_asm


WAVELENGTH = 1.029e-6
N_AX = 1.458
N_EXT = 1.0
GAMMA = math.radians(2.0)


def _geometry(base_angle_rad: float = GAMMA) -> RefractiveAxiconGeometry:
    return RefractiveAxiconGeometry(
        base_angle_rad=float(base_angle_rad),
        clear_radius_m=1.2e-3,
        centre_thickness_m=2.0e-3,
        refractive_index=N_AX,
        external_index=N_EXT,
    )


def _field(n: int = 128, window_m: float = 3.0e-3, *, circular: bool = False) -> VectorField:
    grid = make_xy_grid(n, window_m / n)
    R = np.asarray(grid["R"], dtype=float)
    amp = np.exp(-(R / 0.70e-3) ** 2)
    if circular:
        ex = amp / np.sqrt(2.0)
        ey = 1j * amp / np.sqrt(2.0)
    else:
        ex = amp.astype(np.complex128)
        ey = np.zeros_like(ex)
    return VectorField(
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex),
        grid=grid,
        wavelength_m=WAVELENGTH,
        medium_index=N_EXT,
    )


def _plane_wave_field(n: int = 128, window_m: float = 3.0e-3) -> VectorField:
    grid = make_xy_grid(n, window_m / n)
    ex = np.full((n, n), 1.0 / np.sqrt(2.0), dtype=np.complex128)
    ey = 1j * ex
    return VectorField(
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex),
        grid=grid,
        wavelength_m=WAVELENGTH,
        medium_index=N_EXT,
    )


def _ray_anisotropy(result) -> float:
    valid = np.asarray(result.geometry_bundle.valid, dtype=bool)
    rays = np.asarray(result.outgoing_direction_lab, dtype=float)[valid]
    mean = np.mean(rays, axis=0)
    radius = np.linalg.norm(rays[:, :2] - mean[None, :2], axis=1)
    return float((np.max(radius) - np.min(radius)) / np.mean(radius))


def test_vector_fresnel_oblique_interface_closes_R_plus_T() -> None:
    theta_i = math.radians(27.0)
    d1 = np.asarray([math.sin(theta_i), 0.0, math.cos(theta_i)])
    normal = np.asarray([0.0, 0.0, 1.0])
    d2 = refract_direction(d1, normal, n1=1.0, n2=1.46)
    e = np.asarray([0.3 + 0.1j, 0.8 - 0.2j, 0.0], dtype=np.complex128)
    _et, transmission, reflection = fresnel_transmit_vector_3d(
        e, d1, d2, normal, n1=1.0, n2=1.46
    )
    assert abs(float(transmission + reflection) - 1.0) < 2e-12
    assert 0.0 < float(transmission) < 1.0


def test_vector_fresnel_broadcasts_one_ray_over_spatial_vector_field() -> None:
    d1 = np.asarray([0.0, 0.0, 1.0])
    normal = np.asarray([0.0, 0.0, 1.0])
    d2 = refract_direction(d1, normal, n1=1.0, n2=1.46)
    e = np.zeros((17, 19, 3), dtype=np.complex128)
    e[..., 0] = 1.0
    et, transmission, reflection = fresnel_transmit_vector_3d(
        e, d1, d2, normal, n1=1.0, n2=1.46
    )
    assert et.shape == e.shape
    assert transmission.shape == e.shape[:-1]
    np.testing.assert_allclose(transmission + reflection, 1.0, rtol=0.0, atol=2e-12)


def test_zero_tilt_surface_sampling_is_exact_coordinate_identity() -> None:
    source = _field()
    sampled, carrier, poynting, meta = sample_vector_field_on_tilted_entrance(
        source, tilt_x_rad=0.0, tilt_y_rad=0.0
    )
    projected = propagate_vector_asm(source, 0.0)
    np.testing.assert_allclose(sampled.ex, projected.ex, rtol=0.0, atol=5e-13)
    np.testing.assert_allclose(sampled.ey, projected.ey, rtol=0.0, atol=5e-13)
    np.testing.assert_allclose(sampled.ez, projected.ez, rtol=0.0, atol=5e-13)
    np.testing.assert_allclose(carrier, [0.0, 0.0], rtol=0.0, atol=1e-8)
    bright = np.abs(sampled.ex) > 0.1 * np.max(np.abs(sampled.ex))
    mean_s = np.mean(poynting[bright], axis=0)
    mean_s /= np.linalg.norm(mean_s)
    np.testing.assert_allclose(mean_s, [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
    assert meta["source_transversality_residual"] < 1e-12
    assert meta["spectral_transversality_power_ratio"] < 1e-24


def test_common_eikonal_uses_phase_normal_not_total_poynting() -> None:
    source = _field(circular=True)
    envelope, carrier, poynting, _ = sample_vector_field_on_tilted_entrance(
        source,
        tilt_x_rad=0.0,
        tilt_y_rad=math.radians(5.0),
    )
    e = np.stack([envelope.ex, envelope.ey, envelope.ez], axis=-1)
    estimate = estimate_common_vector_eikonal(
        e,
        poynting,
        envelope.grid,
        carrier_local_cpm=(float(carrier[0]), float(carrier[1])),
        rotation_local_to_lab=rotation_matrix(0.0, math.radians(5.0)),
        wavelength_m=WAVELENGTH,
        medium_index=N_EXT,
    )
    assert estimate.metadata["phase_direction_policy"] == "Snell_uses_grad_Phi_not_total_structured_Poynting"
    assert estimate.metadata["p95_component_wavevector_disagreement_fraction"] < 0.01
    assert estimate.metadata["p95_reconstructed_gradient_error_fraction"] < 0.01
    assert np.all(estimate.direction_local[..., 2][estimate.valid_mask] > 0.0)


def test_zero_tilt_vector_two_surface_solver_matches_exact_plane_wave_snell_cone_and_flux() -> None:
    result = build_tilted_vector_refractive_axicon_field(
        _plane_wave_field(),
        geometry=_geometry(),
        tilt_x_rad=0.0,
        tilt_y_rad=0.0,
        reference_gap_m=0.20e-3,
        output_n=256,
        output_window_m=3.0e-3,
    )
    valid = np.asarray(result.geometry_bundle.valid, dtype=bool)
    rays = np.asarray(result.outgoing_direction_lab, dtype=float)
    measured = float(np.median(np.hypot(rays[..., 0], rays[..., 1])[valid]))
    expected = snell_axicon_geometry(
        base_angle_rad=GAMMA,
        refractive_index=N_AX,
        external_index=N_EXT,
    ).exact_radial_direction_sine
    assert abs(measured - expected) < 2e-11
    assert abs(float(result.metadata["ray_flux_closure_ratio"]) - 1.0) < 2e-12
    assert abs(float(result.metadata["final_flux_closure_ratio"]) - 1.0) < 2e-12
    assert float(result.metadata["interface1_max_abs_R_plus_T_minus_1"]) < 2e-12
    assert float(result.metadata["interface2_max_abs_R_plus_T_minus_1"]) < 2e-12
    assert float(result.metadata["final_transversality_residual"]) < 1e-10
    assert float(result.metadata["coverage_fraction"]) > 0.25
    assert abs(float(result.metadata["interpolation_flux_power_correction_ratio"]) - 1.0) < 0.10


def test_tilted_vector_eikonal_phase_gradient_matches_traced_wavevector() -> None:
    result = build_tilted_vector_refractive_axicon_field(
        _field(circular=True),
        geometry=_geometry(),
        tilt_x_rad=0.0,
        tilt_y_rad=math.radians(5.0),
        reference_gap_m=0.20e-3,
        output_n=256,
        output_window_m=3.0e-3,
    )
    check = lab_reference_eikonal_direction_consistency(
        result,
        wavelength_m=WAVELENGTH,
        external_index=N_EXT,
        trim_pixels=5,
    )
    assert check["median_relative_direction_error"] < 0.01
    assert check["p95_relative_direction_error"] < 0.05


def test_x_and_y_vector_tilts_are_rotationally_equivalent_for_circular_input() -> None:
    angle = math.radians(5.0)
    source = _field(circular=True)
    rx = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=_geometry(),
        tilt_x_rad=angle,
        tilt_y_rad=0.0,
        reference_gap_m=0.20e-3,
        output_n=256,
        output_window_m=3.0e-3,
    )
    ry = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=_geometry(),
        tilt_x_rad=0.0,
        tilt_y_rad=angle,
        reference_gap_m=0.20e-3,
        output_n=256,
        output_window_m=3.0e-3,
    )
    assert abs(_ray_anisotropy(rx) - _ray_anisotropy(ry)) < 3e-4
    assert abs(
        float(rx.metadata["all_transmitted_flux_au"])
        / float(ry.metadata["all_transmitted_flux_au"])
        - 1.0
    ) < 5e-4


def test_tilt_sign_is_mirror_symmetric_in_scalar_ray_metrics() -> None:
    angle = math.radians(5.0)
    source = _field(circular=True)
    plus = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=_geometry(),
        tilt_x_rad=0.0,
        tilt_y_rad=angle,
        reference_gap_m=0.20e-3,
        output_n=256,
        output_window_m=3.0e-3,
    )
    minus = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=_geometry(),
        tilt_x_rad=0.0,
        tilt_y_rad=-angle,
        reference_gap_m=0.20e-3,
        output_n=256,
        output_window_m=3.0e-3,
    )
    assert abs(_ray_anisotropy(plus) - _ray_anisotropy(minus)) < 3e-4
    assert abs(
        float(plus.metadata["all_transmitted_flux_au"])
        / float(minus.metadata["all_transmitted_flux_au"])
        - 1.0
    ) < 5e-4


def test_high_cone_angle_is_blocked_when_fft_sampling_cannot_represent_wavevectors() -> None:
    with pytest.raises(ValueError, match="sampling cannot represent"):
        build_tilted_vector_refractive_axicon_field(
            _field(),
            geometry=_geometry(base_angle_rad=math.radians(20.0)),
            tilt_x_rad=0.0,
            tilt_y_rad=math.radians(2.0),
            reference_gap_m=0.20e-3,
            output_n=128,
            output_window_m=3.0e-3,
        )
