from __future__ import annotations

import numpy as np
import pytest

from vbb_study.calibration.shack_hartmann import (
    correction_phase_on_slm,
    reconstruct_opd_from_slopes,
)


TWOPI = 2.0 * np.pi


def test_circular_shack_hartmann_pupil_reconstructs_and_maps_to_slm() -> None:
    x = np.linspace(-2.0e-3, 2.0e-3, 35)
    y = np.linspace(-2.0e-3, 2.0e-3, 35)
    X, Y = np.meshgrid(x, y, indexing="xy")
    radius = 1.65e-3
    valid = X**2 + Y**2 <= radius**2

    # Quadratic + cross term: Southwell trapezoidal edge equations are exact for
    # these linear slopes, so the masked-pupil reconstruction has a clean
    # analytic identity test apart from the unobservable piston.
    a = 1.2e-3
    b = -0.7e-3
    c = 0.4e-3
    W = a * X**2 + b * Y**2 + c * X * Y
    sx = 2.0 * a * X + c * Y
    sy = 2.0 * b * Y + c * X

    rec = reconstruct_opd_from_slopes(sx, sy, x, y, valid_mask=valid)
    expected = W - float(np.mean(W[valid]))
    assert np.sqrt(np.mean((rec.opd_m[valid] - expected[valid]) ** 2)) < 2e-12
    assert np.all(np.isnan(rec.opd_m[~valid]))
    assert rec.metadata["masked_pupil_supported"] is True

    wavelength = 1029e-9
    correction = correction_phase_on_slm(
        rec,
        x,
        y,
        X,
        Y,
        wavelength_m=wavelength,
        outside_value_rad=0.0,
    )
    residual = TWOPI * rec.opd_m[valid] / wavelength + correction[valid]
    assert np.max(np.abs(residual)) < 2e-10

    # Corners lie outside the convex hull of the illuminated circular lenslet
    # pupil and must not receive invented extrapolated correction.
    assert correction[0, 0] == pytest.approx(0.0)
    assert correction[0, -1] == pytest.approx(0.0)
    assert correction[-1, 0] == pytest.approx(0.0)
    assert correction[-1, -1] == pytest.approx(0.0)


def test_masked_shack_hartmann_registration_preserves_no_extrapolation_policy() -> None:
    x = np.linspace(-1.5e-3, 1.5e-3, 25)
    y = np.linspace(-1.5e-3, 1.5e-3, 25)
    X, Y = np.meshgrid(x, y, indexing="xy")
    valid = X**2 + Y**2 <= (1.2e-3) ** 2
    W = 8e-4 * X**2 + 5e-4 * Y**2
    sx = 1.6e-3 * X
    sy = 1.0e-3 * Y
    rec = reconstruct_opd_from_slopes(sx, sy, x, y, valid_mask=valid)

    # Ask for a target grid shifted well away from the measured pupil. The
    # registered query points are outside the measured convex hull, so the
    # correction must stay at the explicit outside value instead of extrapolating.
    correction = correction_phase_on_slm(
        rec,
        x,
        y,
        X,
        Y,
        wavelength_m=1029e-9,
        offset_x_m=10e-3,
        outside_value_rad=0.123,
    )
    assert np.allclose(correction, 0.123, rtol=0.0, atol=1e-15)
