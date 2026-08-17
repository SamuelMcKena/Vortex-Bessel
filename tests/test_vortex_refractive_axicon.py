from __future__ import annotations

import math

import numpy as np
import pytest

from vbb_study.digital_twin.vortex_error_reference_models import snell_axicon_geometry
from vbb_study.digital_twin.vortex_refractive_axicon import (
    RefractiveAxiconGeometry,
    eikonal_direction_consistency,
    trace_refractive_axicon_bundle,
)


WAVELENGTH = 1.029e-6
N_AX = 1.458
N_EXT = 1.0
GAMMA = math.radians(2.0)


def _grid(n: int = 101, halfwidth_m: float = 1.5e-3) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-halfwidth_m, halfwidth_m, n)
    return np.meshgrid(axis, axis, indexing="xy")


def _geometry() -> RefractiveAxiconGeometry:
    return RefractiveAxiconGeometry(
        base_angle_rad=GAMMA,
        clear_radius_m=2.0e-3,
        centre_thickness_m=2.0e-3,
        refractive_index=N_AX,
        external_index=N_EXT,
    )


def _cone_anisotropy(bundle) -> float:
    outgoing = np.asarray(bundle.outgoing_lab)[bundle.valid]
    mean = np.mean(outgoing, axis=0)
    radius = np.linalg.norm(outgoing[:, :2] - mean[None, :2], axis=1)
    return float((np.max(radius) - np.min(radius)) / np.mean(radius))


def test_geometry_rejects_nonpositive_edge_thickness() -> None:
    bad = RefractiveAxiconGeometry(
        base_angle_rad=math.radians(20.0),
        clear_radius_m=5.0e-3,
        centre_thickness_m=1.0e-3,
        refractive_index=N_AX,
    )
    with pytest.raises(ValueError, match="centre thickness"):
        bad.validate()


def test_zero_tilt_two_surface_bundle_matches_exact_snell_cone() -> None:
    X, Y = _grid()
    bundle = trace_refractive_axicon_bundle(
        X,
        Y,
        geometry=_geometry(),
        apex_exclusion_radius_m=60e-6,
    )
    outgoing = np.asarray(bundle.outgoing_lab)
    transverse = np.sqrt(outgoing[..., 0] ** 2 + outgoing[..., 1] ** 2)
    measured = float(np.median(transverse[bundle.valid]))
    reference = snell_axicon_geometry(
        base_angle_rad=GAMMA,
        refractive_index=N_AX,
        external_index=N_EXT,
    )
    assert abs(measured - reference.exact_radial_direction_sine) < 2e-10
    assert abs(bundle.reference_normal_lab[0]) < 1e-10
    assert abs(bundle.reference_normal_lab[1]) < 1e-10
    assert bundle.reference_normal_lab[2] > 0.999999999


def test_surface_opl_gradient_reproduces_outgoing_wavevector() -> None:
    X, Y = _grid(n=121, halfwidth_m=1.4e-3)
    bundle = trace_refractive_axicon_bundle(
        X,
        Y,
        geometry=_geometry(),
        tilt_y_rad=math.radians(5.0),
        apex_exclusion_radius_m=80e-6,
    )
    consistency = eikonal_direction_consistency(
        bundle,
        wavelength_m=WAVELENGTH,
        external_index=N_EXT,
        trim_pixels=5,
    )
    assert consistency["median_relative_direction_error"] < 0.01
    assert consistency["p95_relative_direction_error"] < 0.05


def test_x_y_tilt_are_rotationally_equivalent_for_axisymmetric_geometry() -> None:
    X, Y = _grid(n=81, halfwidth_m=1.3e-3)
    angle = math.radians(5.0)
    bx = trace_refractive_axicon_bundle(
        X,
        Y,
        geometry=_geometry(),
        tilt_x_rad=angle,
        apex_exclusion_radius_m=80e-6,
    )
    by = trace_refractive_axicon_bundle(
        X,
        Y,
        geometry=_geometry(),
        tilt_y_rad=angle,
        apex_exclusion_radius_m=80e-6,
    )
    np.testing.assert_allclose(
        _cone_anisotropy(bx),
        _cone_anisotropy(by),
        rtol=2e-3,
        atol=2e-5,
    )


def test_vector_fresnel_transmission_is_physical() -> None:
    X, Y = _grid(n=61, halfwidth_m=1.0e-3)
    bundle = trace_refractive_axicon_bundle(
        X,
        Y,
        geometry=_geometry(),
        tilt_y_rad=math.radians(5.0),
        polarization_lab=np.asarray([1.0, 0.0, 0.0]),
        apex_exclusion_radius_m=80e-6,
    )
    transmission = np.asarray(bundle.fresnel_power_transmission)
    assert np.all(np.isfinite(transmission[bundle.valid]))
    assert float(np.min(transmission[bundle.valid])) > 0.0
    assert float(np.max(transmission[bundle.valid])) <= 1.0 + 1e-12
    assert bundle.output_polarization_lab is not None
