"""Stage 8C.1 foundation repair tests."""

import math

import numpy as np
import pytest

from vbb_study.digital_twin.energy_accounting import (
    LaserSource,
    OpticalComponent,
    compute_energy_ledger,
    scale_intensity_to_fluence_j_cm2,
)
from vbb_study.digital_twin.field_coupling import (
    OpticalFieldPlane,
    extract_plane_from_surfacefield,
    extract_stack_from_surfacefield,
    plane_from_arrays,
    stack_from_arrays,
)
from vbb_study.digital_twin.field_fluence import scale_stack_to_fluence


def test_energy_ledger_uses_true_source_metadata_when_provided():
    source = LaserSource(
        wavelength_nm=800.0,
        pulse_duration_fs=120.0,
        repetition_rate_Hz=10_000.0,
        pulse_energy_before_optics_uJ=50.0,
        average_power_limit_W=2.0,
        beam_radius_mm=1.25,
        polarisation_state="circular",
    )
    ledger = compute_energy_ledger(
        50.0,
        10_000.0,
        [OpticalComponent("sample", "interface", 0.5)],
        source=source,
    )
    assert ledger.source is source
    assert ledger.source.wavelength_nm == 800.0
    assert ledger.source.pulse_duration_fs == 120.0
    assert ledger.source.beam_radius_mm == 1.25
    assert ledger.source.polarisation_state == "circular"


def test_energy_ledger_without_source_does_not_fabricate_metadata():
    ledger = compute_energy_ledger(10.0, 1_000.0, [])
    assert ledger.source is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"intensity": np.array([[1.0, np.nan]])},
        {"intensity": np.array([[1.0, np.inf]])},
        {"dx_um": np.nan},
        {"dy_um": np.inf},
        {"pulse_energy_uJ": np.nan},
        {"pulse_energy_uJ": np.inf},
    ],
)
def test_scale_intensity_rejects_non_finite_inputs(kwargs):
    params = {
        "intensity": np.ones((2, 2)),
        "dx_um": 1.0,
        "dy_um": 1.0,
        "pulse_energy_uJ": 1.0,
    }
    params.update(kwargs)
    with pytest.raises(ValueError):
        scale_intensity_to_fluence_j_cm2(**params)


def test_optical_field_plane_has_centered_xy_coordinates():
    plane = plane_from_arrays(np.ones((3, 4)), dx_um=2.0, dy_um=3.0, z_um=5.0)
    assert isinstance(plane, OpticalFieldPlane)
    assert plane.x_um.shape == (4,)
    assert plane.y_um.shape == (3,)
    assert np.allclose(plane.x_um, [-3.0, -1.0, 1.0, 3.0])
    assert np.allclose(plane.y_um, [-3.0, 0.0, 3.0])
    assert np.all(np.diff(plane.x_um) > 0)
    assert np.all(np.diff(plane.y_um) > 0)


def test_extract_plane_preserves_real_grid_coordinates():
    class Field:
        pass

    f = Field()
    f.intensity = np.ones((3, 4))
    f.grid = {
        "dx": 0.5e-6,
        "x": np.array([-2.0, -1.0, 0.0, 1.0]) * 1e-6,
        "y": np.array([-3.0, 0.0, 3.0]) * 1e-6,
    }
    plane = extract_plane_from_surfacefield(f)
    assert np.allclose(plane.x_um, [-2.0, -1.0, 0.0, 1.0])
    assert np.allclose(plane.y_um, [-3.0, 0.0, 3.0])


def test_extract_stack_uses_crop_grid_y_when_present():
    nz, ny, nx = 2, 3, 4
    x_m = np.array([-2.0, -1.0, 0.0, 1.0]) * 1e-6
    y_m = np.array([-3.0, 0.0, 3.0]) * 1e-6
    vol = {
        "intensity_stack": np.ones((nz, ny, nx)),
        "z": np.array([0.0, 10e-6]),
        "crop_grid": {"x": x_m, "y": y_m},
    }
    stack = extract_stack_from_surfacefield(vol)
    assert np.allclose(stack.x_um, x_m * 1e6)
    assert np.allclose(stack.y_um, y_m * 1e6)
    assert "assumed_y_equals_x" not in stack.metadata


def test_extract_stack_y_equals_x_fallback_records_metadata():
    n = 4
    x_m = (np.arange(n) - (n - 1) / 2.0) * 0.25e-6
    vol = {
        "intensity_stack": np.ones((2, n, n)),
        "z": np.array([0.0, 10e-6]),
        "crop_grid": {"x": x_m},
    }
    stack = extract_stack_from_surfacefield(vol)
    assert stack.metadata["assumed_y_equals_x"] is True
    assert np.allclose(stack.y_um, stack.x_um)


@pytest.mark.parametrize(
    "n, expected",
    [
        (4, np.array([-1.5, -0.5, 0.5, 1.5])),
        (5, np.array([-2.0, -1.0, 0.0, 1.0, 2.0])),
    ],
)
def test_extract_stack_reconstructs_coordinates_centered_even_and_odd(n, expected):
    vol = {
        "intensity_stack": np.ones((2, n, n)),
        "z": np.array([0.0, 10e-6]),
        "crop_grid": {"dx": 0.5e-6},
    }
    stack = extract_stack_from_surfacefield(vol)
    assert np.allclose(stack.x_um, expected * 0.5)
    assert np.allclose(stack.y_um, expected * 0.5)


def test_stack_fluence_exposes_raw_captured_power_and_conserved_plane_energy():
    x = np.linspace(-1.0, 1.0, 4)
    y = np.linspace(-1.0, 1.0, 4)
    z = np.array([0.0, 1.0, 2.0])
    intensity = np.ones((3, 4, 4))
    intensity[1] *= 2.0
    intensity[2] *= 4.0
    stack = stack_from_arrays(intensity, x, y, z)
    res = scale_stack_to_fluence(stack, 2.0)
    assert hasattr(res, "raw_transverse_integral_by_z")
    assert hasattr(res, "raw_captured_power_fraction_by_z")
    assert np.allclose(res.transverse_energy_by_z_uJ, 2.0)
    assert np.allclose(res.raw_captured_power_fraction_by_z, [0.25, 0.5, 1.0])
    assert math.isclose(res.propagation_energy_drift_fraction, 0.75)

