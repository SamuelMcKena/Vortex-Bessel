from __future__ import annotations

import numpy as np
import pytest

from vbb_study.calibration.camera_comparison import CameraCalibration
from vbb_study.calibration.vector_observables import (
    LinearAnalyzerCalibration,
    analyzer_intensity,
    compare_vector_analyzer_frames,
    linear_stokes_from_frames,
    petal_observable,
)


def _vector_ring(ell: int, n: int = 301, q: float = 5e-6):
    x = (np.arange(n, dtype=float) - (n - 1) / 2.0) * q
    X, Y = np.meshgrid(x, x, indexing="xy")
    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    radius = 420e-6
    width = 70e-6
    amp = np.exp(-0.5 * ((R - radius) / width) ** 2)
    ex = amp * np.cos(abs(int(ell)) * phi)
    ey = amp * np.sin(abs(int(ell)) * phi)
    return x, X, Y, ex.astype(complex), ey.astype(complex), radius


@pytest.mark.parametrize(("ell", "expected_petals"), [(1, 2), (3, 6)])
def test_vector_analyzer_petal_count_is_two_abs_ell(ell: int, expected_petals: int) -> None:
    _, X, Y, ex, ey, radius = _vector_ring(ell)
    image = analyzer_intensity(ex, ey, LinearAnalyzerCalibration(0, 0))
    petals = petal_observable(image, X, Y, ring_radius_m=radius)
    assert petals.petal_count == expected_petals
    assert petals.modulation_fraction > 0.6


def test_linear_stokes_reconstruction_recovers_known_linear_state() -> None:
    _, _, _, ex, ey, _ = _vector_ring(1)
    frames = {
        angle: analyzer_intensity(ex, ey, LinearAnalyzerCalibration(angle, angle))
        for angle in (0, 45, 90, 135)
    }
    stokes = linear_stokes_from_frames(frames)
    direct_s0 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    direct_s1 = np.abs(ex) ** 2 - np.abs(ey) ** 2
    direct_s2 = 2.0 * np.real(ex * np.conj(ey))
    assert np.max(np.abs(stokes["S0"] - direct_s0)) < 1e-12
    assert np.max(np.abs(stokes["S1"] - direct_s1)) < 1e-12
    assert np.max(np.abs(stokes["S2"] - direct_s2)) < 1e-12


def test_vector_camera_identity_comparison_recovers_spots_and_psi() -> None:
    ell = 3
    x, _, _, ex, ey, _ = _vector_ring(ell, n=241, q=6e-6)
    frames = {
        angle: analyzer_intensity(ex, ey, LinearAnalyzerCalibration(angle, angle))
        for angle in (0, 45, 90, 135)
    }
    result = compare_vector_analyzer_frames(
        frames,
        camera_calibration=CameraCalibration(object_plane_scale_m_per_pixel=6e-6),
        simulated_ex=ex,
        simulated_ey=ey,
        simulated_x_m=x,
        simulated_y_m=x,
        expected_ell=ell,
    )
    assert result.polarization_angle_rms_rad < 1e-6
    assert result.metadata["expected_analyzer_petal_count"] == 6
    assert result.metadata["measured_expected_petal_match_fraction"] == pytest.approx(1.0)
    for angle in (0, 45, 90, 135):
        assert result.frame_comparisons[angle].metrics["energy_normalised_correlation"] > 0.999999
        assert result.measured_petals[angle].petal_count == 6
        assert result.simulated_petals[angle].petal_count == 6


def test_analyzer_extinction_ratio_leaks_orthogonal_component() -> None:
    n = 32
    ex = np.zeros((n, n), dtype=complex)
    ey = np.ones((n, n), dtype=complex)
    ideal = analyzer_intensity(ex, ey, LinearAnalyzerCalibration(0, 0, extinction_ratio=np.inf))
    finite = analyzer_intensity(ex, ey, LinearAnalyzerCalibration(0, 0, extinction_ratio=1000.0))
    assert np.max(ideal) == pytest.approx(0.0)
    assert np.mean(finite) == pytest.approx(1e-3)
