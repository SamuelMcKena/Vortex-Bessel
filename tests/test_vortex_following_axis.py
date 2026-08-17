from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.vortex_following_propagation import (
    phase_winding_charge_map,
    transverse_morphology_axis,
)
from vbb_study.digital_twin.vortex_morphology_tracking import (
    track_bessel_feature_axis,
)
from vbb_study.equations.fields import make_xy_grid


def test_transverse_vortex_axis_follows_phase_singularity_not_energy_centroid() -> None:
    grid = make_xy_grid(256, 8e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    x0 = 240e-6
    y0 = -120e-6
    # Deliberately asymmetric amplitude: its intensity centroid is not the
    # topological vortex centre.
    R2 = (X - x0) ** 2 + (Y - y0) ** 2
    vortex = (X - x0) + 1j * (Y - y0)
    envelope = np.exp(-R2 / (0.42e-3**2)) * (1.0 + 0.7 * X / 1e-3)
    field = envelope * vortex
    axis = transverse_morphology_axis(
        field,
        grid,
        vortex_charge=1,
        seed_x_m=x0,
        seed_y_m=y0,
        search_radius_m=0.35e-3,
    )
    assert axis.detected_topological_charge == 1
    assert abs(axis.x_m - x0) < 1.5 * float(grid["dx"])
    assert abs(axis.y_m - y0) < 1.5 * float(grid["dx"])


def test_charge_three_split_core_is_recovered_by_charge_weighted_axis() -> None:
    grid = make_xy_grid(256, 8e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    centres = [(-20e-6, 0.0), (10e-6, 17e-6), (10e-6, -17e-6)]
    field = np.ones_like(X, dtype=np.complex128)
    for cx, cy in centres:
        field *= (X - cx) + 1j * (Y - cy)
    field *= np.exp(-(X * X + Y * Y) / (0.45e-3**2))
    axis = transverse_morphology_axis(
        field,
        grid,
        vortex_charge=3,
        seed_x_m=0.0,
        seed_y_m=0.0,
        search_radius_m=0.2e-3,
    )
    assert axis.detected_topological_charge == 3
    assert axis.selected_singularity_count >= 1
    assert abs(axis.x_m) < 2.0 * float(grid["dx"])
    assert abs(axis.y_m) < 2.0 * float(grid["dx"])


def test_phase_winding_map_has_expected_total_charge_for_single_vortex() -> None:
    grid = make_xy_grid(128, 10e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = (X - 35e-6) + 1j * (Y + 25e-6)
    charge = phase_winding_charge_map(field)
    assert int(np.sum(charge)) == 1


def test_longitudinal_vortex_tracker_finds_dark_channel_between_bright_lobes() -> None:
    coordinate = np.linspace(-300e-6, 300e-6, 601)
    z = np.linspace(0.0, 1.0, 51)
    true_axis = -90e-6 + 150e-6 * z
    rows = []
    for axis in true_axis:
        left = np.exp(-((coordinate - (axis - 28e-6)) / 11e-6) ** 2)
        right = 0.72 * np.exp(-((coordinate - (axis + 28e-6)) / 11e-6) ** 2)
        broad_asymmetry = 0.12 * np.exp(-((coordinate - (axis + 120e-6)) / 70e-6) ** 2)
        rows.append(left + right + broad_asymmetry)
    intensity = np.asarray(rows)
    track = track_bessel_feature_axis(
        intensity,
        coordinate,
        vortex_charge=1,
        seed_coordinate_m=float(true_axis[0]),
        search_halfwidth_m=120e-6,
    )
    assert float(np.max(np.abs(track.coordinate_m - true_axis))) < 4e-6
    assert track.detected_fraction > 0.95
    assert track.maximum_detected_step_m < 5e-6


def test_longitudinal_b0_tracker_follows_central_peak() -> None:
    coordinate = np.linspace(-250e-6, 250e-6, 501)
    z = np.linspace(0.0, 1.0, 41)
    true_axis = 70e-6 * np.sin(0.8 * z)
    intensity = np.asarray(
        [
            np.exp(-((coordinate - axis) / 14e-6) ** 2)
            + 0.4 * np.exp(-((coordinate - axis - 75e-6) / 18e-6) ** 2)
            for axis in true_axis
        ]
    )
    track = track_bessel_feature_axis(
        intensity,
        coordinate,
        vortex_charge=0,
        seed_coordinate_m=float(true_axis[0]),
        search_halfwidth_m=100e-6,
    )
    assert float(np.max(np.abs(track.coordinate_m - true_axis))) < 3e-6
    assert track.detected_fraction > 0.95
