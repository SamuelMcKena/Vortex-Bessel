from __future__ import annotations

import numpy as np
import pytest

from vbb_study.digital_twin.vortex_rotated_plane import (
    lab_to_tilted_plane,
    tilted_to_lab_plane,
)
from vbb_study.equations.fields import make_xy_grid


EPS = np.finfo(float).tiny


def _power(field: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(field, dtype=np.complex128)) ** 2))


def _field_overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.complex128).ravel()
    bb = np.asarray(b, dtype=np.complex128).ravel()
    return float(
        abs(np.vdot(aa, bb))
        / max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), EPS)
    )


def _intensity_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.abs(np.asarray(a, dtype=np.complex128)) ** 2
    bb = np.abs(np.asarray(b, dtype=np.complex128)) ** 2
    aa = aa.ravel() - float(np.mean(aa))
    bb = bb.ravel() - float(np.mean(bb))
    return float(
        np.dot(aa, bb)
        / max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), EPS)
    )


@pytest.mark.parametrize(
    ("axis", "tilt_deg"),
    (("x", 0.25), ("x", 0.5), ("y", 0.25), ("y", 0.5)),
)
def test_carrier_aware_rotated_plane_preserves_off_axis_spectrum(
    axis: str,
    tilt_deg: float,
) -> None:
    """Regression for the former y-tilt loss of an off-axis +1 order."""

    grid = make_xy_grid(256, 10.0e-3 / 256)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    R = np.hypot(X, Y)
    wavelength_m = 1.029e-6
    carrier_cpm = 6250.0
    field = np.exp(-(R / 2.0e-3) ** 2) * np.exp(
        1j * 2.0 * np.pi * carrier_cpm * X
    )

    angle = np.deg2rad(float(tilt_deg))
    tilt_x = angle if axis == "x" else 0.0
    tilt_y = angle if axis == "y" else 0.0

    tilted, forward = lab_to_tilted_plane(
        field,
        grid,
        wavelength_m=wavelength_m,
        tilt_x_rad=tilt_x,
        tilt_y_rad=tilt_y,
    )
    returned, inverse = tilted_to_lab_plane(
        tilted,
        grid,
        wavelength_m=wavelength_m,
        tilt_x_rad=tilt_x,
        tilt_y_rad=tilt_y,
    )

    assert forward["spectral_center_model"] == "intensity_weighted_spectral_centroid"
    assert inverse["spectral_center_model"] == "intensity_weighted_spectral_centroid"
    assert float(forward["spectral_power_ratio"]) > 0.995
    assert float(inverse["spectral_power_ratio"]) > 0.995
    assert abs(_power(returned) / _power(field) - 1.0) < 5.0e-4
    assert _field_overlap(field, returned) > 0.9999
    assert _intensity_correlation(field, returned) > 0.9999
