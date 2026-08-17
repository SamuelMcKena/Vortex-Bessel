"""
Stage 8C field-coupling tests.

Covers canonical plane/stack constructors, validators, the [z, y, x] convention,
and the SurfaceField / volume extractors (including loud failures).
"""

import numpy as np
import pytest

from vbb_study.digital_twin.field_coupling import (
    MissingOpticalFieldError,
    InvalidOpticalFieldError,
    UnsupportedSurfaceFieldError,
    OpticalFieldPlane,
    OpticalFieldStack,
    validate_intensity_plane,
    validate_intensity_stack,
    plane_from_arrays,
    stack_from_arrays,
    extract_plane_from_surfacefield,
    extract_stack_from_surfacefield,
    SOURCE_UNIT_TEST_FIXTURE,
    SOURCE_SYNTHETIC_PLACEHOLDER,
)


# ---------------------------------------------------------------------------
# validate_intensity_plane / stack
# ---------------------------------------------------------------------------


def test_validate_intensity_plane_ok():
    arr = validate_intensity_plane(np.ones((4, 5)))
    assert arr.shape == (4, 5)
    assert arr.dtype == float


def test_validate_intensity_plane_rejects_negative():
    I = np.ones((4, 4))
    I[1, 1] = -0.1
    with pytest.raises(InvalidOpticalFieldError, match="negative"):
        validate_intensity_plane(I)


def test_validate_intensity_plane_rejects_nan():
    I = np.ones((4, 4))
    I[0, 0] = np.nan
    with pytest.raises(InvalidOpticalFieldError, match="non-finite"):
        validate_intensity_plane(I)


def test_validate_intensity_plane_rejects_inf():
    I = np.ones((4, 4))
    I[0, 0] = np.inf
    with pytest.raises(InvalidOpticalFieldError, match="non-finite"):
        validate_intensity_plane(I)


def test_validate_intensity_plane_rejects_3d():
    with pytest.raises(InvalidOpticalFieldError, match="2D"):
        validate_intensity_plane(np.ones((2, 3, 4)))


def test_validate_intensity_plane_rejects_empty():
    with pytest.raises(InvalidOpticalFieldError, match="empty"):
        validate_intensity_plane(np.array([]).reshape(0, 0))


def test_validate_intensity_stack_ok():
    arr = validate_intensity_stack(np.ones((3, 4, 5)))
    assert arr.shape == (3, 4, 5)


def test_validate_intensity_stack_rejects_2d():
    with pytest.raises(InvalidOpticalFieldError, match="3D"):
        validate_intensity_stack(np.ones((4, 5)))


def test_validate_intensity_stack_rejects_negative():
    I = np.ones((2, 3, 3))
    I[0, 0, 0] = -1.0
    with pytest.raises(InvalidOpticalFieldError, match="negative"):
        validate_intensity_stack(I)


# ---------------------------------------------------------------------------
# plane_from_arrays
# ---------------------------------------------------------------------------


def test_plane_from_arrays_valid():
    p = plane_from_arrays(np.ones((8, 8)), dx_um=0.5, dy_um=0.5, z_um=10.0,
                          source_status=SOURCE_UNIT_TEST_FIXTURE)
    assert isinstance(p, OpticalFieldPlane)
    assert p.dx_um == 0.5
    assert p.dy_um == 0.5
    assert p.z_um == 10.0
    assert p.intensity.shape == (8, 8)


def test_plane_from_arrays_rejects_negative_intensity():
    I = np.ones((4, 4))
    I[0, 0] = -1.0
    with pytest.raises(InvalidOpticalFieldError):
        plane_from_arrays(I, dx_um=1.0, dy_um=1.0)


def test_plane_from_arrays_rejects_nan_intensity():
    I = np.ones((4, 4))
    I[2, 2] = np.nan
    with pytest.raises(InvalidOpticalFieldError):
        plane_from_arrays(I, dx_um=1.0, dy_um=1.0)


def test_plane_from_arrays_rejects_zero_dx():
    with pytest.raises(InvalidOpticalFieldError):
        plane_from_arrays(np.ones((4, 4)), dx_um=0.0, dy_um=1.0)


def test_plane_from_arrays_rejects_negative_dy():
    with pytest.raises(InvalidOpticalFieldError):
        plane_from_arrays(np.ones((4, 4)), dx_um=1.0, dy_um=-1.0)


def test_plane_z_um_can_be_none():
    p = plane_from_arrays(np.ones((4, 4)), dx_um=1.0, dy_um=1.0, z_um=None)
    assert p.z_um is None


# ---------------------------------------------------------------------------
# stack_from_arrays — convention and consistency
# ---------------------------------------------------------------------------


def test_stack_from_arrays_valid():
    nz, ny, nx = 3, 4, 5
    I = np.ones((nz, ny, nx))
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-1.5, 1.5, ny)
    z = np.linspace(0, 10, nz)
    s = stack_from_arrays(I, x, y, z, source_status=SOURCE_UNIT_TEST_FIXTURE)
    assert isinstance(s, OpticalFieldStack)
    assert s.intensity_zyx.shape == (nz, ny, nx)


def test_stack_convention_is_zyx():
    """intensity_zyx.shape must be (len z, len y, len x)."""
    nz, ny, nx = 6, 4, 5
    I = np.ones((nz, ny, nx))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    s = stack_from_arrays(I, x, y, z)
    assert s.intensity_zyx.shape[0] == z.size
    assert s.intensity_zyx.shape[1] == y.size
    assert s.intensity_zyx.shape[2] == x.size


def test_stack_from_arrays_rejects_shape_mismatch():
    I = np.ones((3, 4, 5))
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 4)
    z = np.linspace(0, 1, 99)  # wrong length
    with pytest.raises(InvalidOpticalFieldError, match="shape does not match"):
        stack_from_arrays(I, x, y, z)


def test_stack_from_arrays_rejects_non_monotonic_coords():
    I = np.ones((3, 4, 5))
    x = np.array([0.0, 1.0, 0.5, 2.0, 3.0])  # not monotonic
    y = np.linspace(0, 1, 4)
    z = np.linspace(0, 1, 3)
    with pytest.raises(InvalidOpticalFieldError, match="monotonic"):
        stack_from_arrays(I, x, y, z)


def test_stack_dx_dy_from_coords():
    I = np.ones((2, 4, 5))
    x = np.linspace(0, 8, 5)   # spacing 2.0
    y = np.linspace(0, 3, 4)   # spacing 1.0
    z = np.linspace(0, 1, 2)
    s = stack_from_arrays(I, x, y, z)
    assert abs(s.dx_um - 2.0) < 1e-9
    assert abs(s.dy_um - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Extractors — fakes and loud failures
# ---------------------------------------------------------------------------


class _FakeSurfaceField:
    """Mimics vbb_studies.SurfaceField: Ex complex + grid dict in metres."""

    def __init__(self, n=16, dx_m=0.25e-6, z_surface_m=80e-6):
        x = (np.arange(n) - n / 2) * dx_m
        X, Y = np.meshgrid(x, x, indexing="xy")
        R = np.hypot(X, Y)
        self.Ex = np.exp(-(R**2) / (2 * (2e-6) ** 2)).astype(complex)
        self.Ey = None
        self.Ez = None
        self.grid = {"N": n, "dx": dx_m, "x": x, "X": X, "Y": Y, "R": R}
        self.z_surface_m = z_surface_m
        self.medium_before = 1.0
        self.metadata = {"note": "fake"}


def test_extract_plane_from_fake_surfacefield():
    sf = _FakeSurfaceField()
    plane = extract_plane_from_surfacefield(sf)
    assert isinstance(plane, OpticalFieldPlane)
    assert plane.intensity.shape == (16, 16)
    # grid dx 0.25 µm
    assert abs(plane.dx_um - 0.25) < 1e-9
    # z_surface 80 µm
    assert abs(plane.z_um - 80.0) < 1e-6
    assert plane.metadata["intensity_components"] == "|Ex|^2"


def test_extract_plane_preferred_z_overrides():
    sf = _FakeSurfaceField()
    plane = extract_plane_from_surfacefield(sf, preferred_z_um=12.0)
    assert plane.z_um == 12.0


def test_extract_plane_none_raises_missing():
    with pytest.raises(MissingOpticalFieldError):
        extract_plane_from_surfacefield(None)


def test_extract_plane_unsupported_object_raises():
    class _Bad:
        pass
    with pytest.raises(UnsupportedSurfaceFieldError):
        extract_plane_from_surfacefield(_Bad())


def test_extract_plane_object_without_sampling_raises():
    class _NoSampling:
        intensity = np.ones((8, 8))  # 2D intensity but no dx/grid
    with pytest.raises(UnsupportedSurfaceFieldError, match="sampling"):
        extract_plane_from_surfacefield(_NoSampling())


def test_extract_stack_from_volume_dict():
    nz, n = 5, 12
    x_m = (np.arange(n) - n / 2) * 0.3e-6
    stack = np.ones((nz, n, n), dtype=np.float32)
    vol = {
        "intensity_stack": stack,
        "z": np.linspace(-20e-6, 20e-6, nz),
        "crop_grid": {"x": x_m, "dx": 0.3e-6, "N": n},
        "propagation_method": "bl_asm",
        "peak_index": 2,
    }
    s = extract_stack_from_surfacefield(vol)
    assert isinstance(s, OpticalFieldStack)
    assert s.intensity_zyx.shape == (nz, n, n)
    assert abs(s.dx_um - 0.3) < 1e-9
    # z converted m -> µm: range -20..20 µm
    assert abs(s.z_um[0] - (-20.0)) < 1e-6


def test_extract_stack_from_result_dict_nesting_volume():
    nz, n = 3, 8
    x_m = (np.arange(n) - n / 2) * 0.5e-6
    vol = {
        "intensity_stack": np.ones((nz, n, n), dtype=np.float32),
        "z": np.linspace(0, 10e-6, nz),
        "crop_grid": {"x": x_m, "dx": 0.5e-6, "N": n},
    }
    result = {"volume": vol, "design": {}}
    s = extract_stack_from_surfacefield(result)
    assert s.intensity_zyx.shape == (nz, n, n)


def test_extract_stack_none_raises_missing():
    with pytest.raises(MissingOpticalFieldError):
        extract_stack_from_surfacefield(None)


def test_extract_stack_on_plane_object_raises():
    """A single-plane SurfaceField is not a stack."""
    sf = _FakeSurfaceField()
    with pytest.raises(UnsupportedSurfaceFieldError):
        extract_stack_from_surfacefield(sf)


# ---------------------------------------------------------------------------
# Source-status governance
# ---------------------------------------------------------------------------


def test_unit_test_fixture_source_status_allowed():
    p = plane_from_arrays(np.ones((4, 4)), 1.0, 1.0, source_status=SOURCE_UNIT_TEST_FIXTURE)
    assert p.source_status == SOURCE_UNIT_TEST_FIXTURE
    assert p.is_governed_source is True


def test_synthetic_placeholder_is_non_governed():
    p = plane_from_arrays(np.ones((4, 4)), 1.0, 1.0, source_status=SOURCE_SYNTHETIC_PLACEHOLDER)
    assert p.is_governed_source is False


# ---------------------------------------------------------------------------
# Real engine integration (skips if engine import fails)
# ---------------------------------------------------------------------------


def test_real_surfacefield_roundtrip():
    bt = pytest.importorskip("bessel_twin_core")
    res = bt.run_case(preset="fast", path="ideal", case_id="stage8c_coupling")
    plane = extract_plane_from_surfacefield(res["surface_field"])
    assert plane.intensity.ndim == 2
    assert plane.dx_um > 0
    assert plane.source_status == "real_optical_field"
    stack = extract_stack_from_surfacefield(res["volume"])
    assert stack.intensity_zyx.ndim == 3
    assert stack.intensity_zyx.shape[0] == stack.z_um.size
