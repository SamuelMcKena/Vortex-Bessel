import numpy as np

from vbb_study.digital_twin.physical_observable_fit import (
    axisymmetric_radial_morphology,
    centroid_trajectory,
    longitudinal_brightness_envelope,
    longitudinal_envelope_rmse,
)


def _gaussian(n=96, sigma=11.0, shift=(0.0, 0.0)):
    yy, xx = np.indices((n, n), dtype=float)
    cy = (n-1)/2 + float(shift[0])
    cx = (n-1)/2 + float(shift[1])
    return np.exp(-((yy-cy)**2 + (xx-cx)**2)/(2*sigma*sigma))


def test_axisymmetric_morphology_marginalizes_translation():
    a = np.stack([_gaussian(shift=(0, 0)), _gaussian(shift=(2, -3))])
    b = np.stack([_gaussian(shift=(4, 5)), _gaussian(shift=(-2, 1))])
    assert axisymmetric_radial_morphology(a, b) < 0.02


def test_centroid_trajectory_detects_lateral_motion():
    stationary = np.stack([_gaussian() for _ in range(5)])
    moving = np.stack([_gaussian(shift=(0, i*2.0)) for i in range(5)])
    assert centroid_trajectory(stationary, moving) > 0.05


def test_longitudinal_envelope_preserves_relative_plane_brightness():
    base = _gaussian()
    stack = np.stack([0.2*base, 0.5*base, 1.0*base, 0.4*base])
    env = longitudinal_brightness_envelope(stack)
    assert np.allclose(env, [0.2, 0.5, 1.0, 0.4], atol=1e-12)


def test_longitudinal_envelope_ignores_one_global_gain_only():
    base = _gaussian()
    a = np.stack([0.2*base, 0.6*base, 1.0*base])
    b = 7.0*a
    c = np.stack([0.2*base, 0.9*base, 1.0*base])
    assert longitudinal_envelope_rmse(a, b) < 1e-12
    assert longitudinal_envelope_rmse(a, c) > 0.05
