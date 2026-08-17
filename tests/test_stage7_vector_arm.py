from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from vbb_study.design import default_config
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl
from vbb_study.slm_model import apply_slm
from vbb_study.vector_arm_chain import (
    default_vector_arm_grid,
    fit_global_phase_error,
    gaussian_envelope,
    headless_angle_delta,
    local_polarization_angle,
    run_vector_arm,
)
from vbb_study.vector_arm_config import SLMPanelConfig, VectorArmConfig
from vbb_study.vector_arm_metrics import (
    assert_degenerate_rejection,
    assert_three_sided_acceptance,
    h6_z_curve,
)
from vbb_study.vector_arm_sweeps import delta_rotation_sweep
from vbb_study.vector_axicon import assert_locked_kr_fingerprint, run_vector_axicon_to_surface
from vbb_study.vector_field import VectorField, propagate_vector_asm, spectral_transversality_residual
from vbb_study.vector_fourier import apply_fourier_iris, carrier_collinearity_report


ROOT = Path(__file__).resolve().parents[1]
TWOPI = 2.0 * np.pi


def _sample_complex_ring(
    values: np.ndarray,
    grid: dict[str, object],
    radius_m: float,
    *,
    samples: int = 6144,
) -> np.ndarray:
    theta = (np.arange(int(samples), dtype=float) + 0.5) * TWOPI / float(samples)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    dx = float(grid["dx"])
    dy = float(grid.get("dy", dx))
    cols = (float(radius_m) * np.cos(theta) - float(x[0])) / dx
    rows = (float(radius_m) * np.sin(theta) - float(y[0])) / dy
    real = map_coordinates(np.real(values), [rows, cols], order=1, mode="nearest")
    imag = map_coordinates(np.imag(values), [rows, cols], order=1, mode="nearest")
    return real + 1j * imag


def _phase_winding(profile: np.ndarray) -> float:
    increments = np.angle(np.roll(profile, -1) * np.conj(profile))
    return float(np.sum(increments) / TWOPI)


def _comb_leakage(profile: np.ndarray, target_mod: int) -> float:
    samples = int(profile.size)
    modes = np.arange(-samples // 2, samples // 2)
    power = np.abs(np.fft.fftshift(np.fft.fft(profile))) ** 2
    in_comb = ((modes - int(target_mod)) % 3) == 0
    return float(np.sum(power[~in_comb]) / np.sum(power[in_comb]))


def _analytic_circular_profiles(cfg: VectorArmConfig, *, samples: int = 6144) -> tuple[np.ndarray, np.ndarray]:
    theta = (np.arange(int(samples), dtype=float) + 0.5) * TWOPI / float(samples)
    cell = TWOPI / float(cfg.n_pairs)
    local = np.mod(theta - float(cfg.sector_rotation_rad), cell)
    phi0 = np.where(local < float(cfg.sector_duty) * cell, 0.5 * np.pi, 0.0)
    return np.exp(-1j * (theta + phi0)), np.exp(1j * (theta + phi0))


def _peak_annulus_radius(intensity: np.ndarray, grid: dict[str, object]) -> float:
    radius = np.asarray(grid["R"], dtype=float)
    bins = np.linspace(0.0, float(np.max(radius)), 80)
    labels = np.digitize(radius.ravel(), bins)
    flat = np.asarray(intensity, dtype=float).ravel()
    means: list[float] = []
    centers: list[float] = []
    for index in range(2, len(bins)):
        mask = labels == index
        if np.any(mask):
            means.append(float(np.mean(flat[mask])))
            centers.append(float(0.5 * (bins[index - 1] + bins[index])))
    return centers[int(np.argmax(means))]


def test_vector_arm_config_defaults_and_validation() -> None:
    cfg = VectorArmConfig()
    assert cfg.wavelength_m == pytest.approx(1029e-9)
    assert cfg.pulse_duration_s == pytest.approx(260e-15)
    assert cfg.waist_m == pytest.approx(2e-3)
    assert cfg.n_pairs == 3
    assert cfg.sector_duty == pytest.approx(0.5)
    assert cfg.slm1.active_width_m == pytest.approx(1920 * 8e-6)
    assert cfg.slm1.active_height_m == pytest.approx(1080 * 8e-6)
    assert cfg.slm1.carrier_lp_per_m == pytest.approx(6.94e3)
    assert cfg.slm2.carrier_sign == -1
    assert replace(cfg, ideal_components=True).effective_slm1.fill_factor == pytest.approx(1.0)

    with pytest.raises(ValueError):
        SLMPanelConfig(phase_levels=1)
    with pytest.raises(ValueError):
        SLMPanelConfig(fill_factor=1.1)
    with pytest.raises(ValueError):
        VectorArmConfig(sector_duty=1.0)


def test_vector_asm_energy_evanescent_decay_and_scalar_cross_check() -> None:
    wavelength = 1029e-9
    grid = make_xy_grid(64, 50e-6)
    ex = np.exp(-(np.asarray(grid["R"], dtype=float) ** 2) / (0.10**2))
    field = VectorField(ex=ex, ey=np.zeros_like(ex), grid=grid, wavelength_m=wavelength)

    at_entry = propagate_vector_asm(field, 0.0)
    propagated = propagate_vector_asm(field, 1e-3)
    assert abs(propagated.power / at_entry.power - 1.0) < 1e-12
    assert spectral_transversality_residual(propagated) < 1e-12

    scalar = angular_spectrum_propagate_bl(
        ex,
        grid,
        wavelength,
        1e-3,
        bandlimit=False,
        include_evanescent=True,
    )
    rel_rms = np.sqrt(np.mean((np.abs(propagated.ex) ** 2 - np.abs(scalar) ** 2) ** 2)) / np.sqrt(
        np.mean(np.abs(scalar) ** 4)
    )
    assert float(rel_rms) < 1e-10

    ev_grid = make_xy_grid(64, wavelength / 4.0)
    checker = np.where(np.indices((64, 64)).sum(axis=0) % 2, 1.0, -1.0)
    evanescent = VectorField(ex=checker, ey=np.zeros_like(checker), grid=ev_grid, wavelength_m=wavelength)
    ev_entry = propagate_vector_asm(evanescent, 0.0)
    ev_forward = propagate_vector_asm(evanescent, 5e-6)
    assert ev_forward.power <= ev_entry.power
    assert ev_forward.power < 1e-12 * ev_entry.power


def test_slm_and_fourier_iris_ledgers_are_exact() -> None:
    grid = make_xy_grid(64, 20e-6)
    field = np.exp(-(np.asarray(grid["R"], dtype=float) ** 2) / (0.45e-3**2))
    phase = 0.3 * np.asarray(grid["X"], dtype=float) / float(grid["dx"])
    panel = SLMPanelConfig(n_x=512, n_y=512, fill_factor=0.93, carrier_lp_per_mm=1.0)
    out = apply_slm(field, phase, grid, panel, quantise_phase=True, apply_carrier=True)
    ledger = out.ledger
    accounted = ledger.modulated_power + ledger.unmodulated_power + ledger.interference_power
    assert abs(ledger.total_power - accounted) / ledger.input_power < 1e-12
    assert ledger.relative_error < 1e-12

    dfx = 1.0 / (64 * 20e-6)
    fx = 8.0 * dfx
    tilted = np.exp(-(np.asarray(grid["R"], dtype=float) ** 2) / (0.8e-3**2)) * np.exp(
        1j * TWOPI * fx * np.asarray(grid["X"], dtype=float)
    )
    iris = apply_fourier_iris(
        tilted,
        grid,
        signal_fx_cpm=fx,
        iris_radius_frac=0.5,
        wavelength_m=1029e-9,
        tilt_tolerance_rad=1e-6,
    )
    assert iris.ledger.relative_error < 1e-12
    assert np.hypot(*iris.residual_tilt_rad) < 1e-6


def test_ideal_chain_flip_winding_orientation_mod3_and_delta_law() -> None:
    cfg = VectorArmConfig(ideal_components=True)
    grid = default_vector_arm_grid(cfg, 128)

    run = run_vector_arm(cfg, grid=grid, return_debug=True)
    _, err = fit_global_phase_error(run.field, run.target)
    assert err < 1e-10

    naive = run_vector_arm(cfg, grid=grid, naive_psi2=True, return_debug=True)
    _, naive_err = fit_global_phase_error(naive.field, naive.target)
    assert naive_err > 0.1

    plus, minus = run.field.circular_components()
    radius = 0.5 * cfg.waist_m
    assert abs(_phase_winding(_sample_complex_ring(plus, grid, radius)) + 1.0) < 1e-9
    assert abs(_phase_winding(_sample_complex_ring(minus, grid, radius)) - 1.0) < 1e-9

    radial = run_vector_arm(cfg, grid=grid, all_radial=True)
    stokes = radial.stokes()
    x_axis = (np.abs(np.asarray(grid["Y"])) <= float(grid["dx"])) & (np.asarray(grid["X"]) > 0.25 * cfg.waist_m)
    y_axis = (np.abs(np.asarray(grid["X"])) <= float(grid["dx"])) & (np.asarray(grid["Y"]) > 0.25 * cfg.waist_m)
    assert float(np.mean(stokes["S1"][x_axis])) > 0.0
    assert float(np.mean(stokes["S1"][y_axis])) < 0.0
    assert float(np.max(np.abs(stokes["S3"])) / np.max(stokes["S0"])) < 1e-10
    psi = local_polarization_angle(radial)
    theta = np.arctan2(np.asarray(grid["Y"]), np.asarray(grid["X"]))
    annulus = (np.asarray(grid["R"]) > 0.45 * cfg.waist_m) & (np.asarray(grid["R"]) < 0.55 * cfg.waist_m)
    angle_err = headless_angle_delta(psi[annulus], theta[annulus])
    assert float(np.sqrt(np.mean(angle_err**2))) < 1e-9

    analytic_plus, analytic_minus = _analytic_circular_profiles(cfg)
    assert _comb_leakage(analytic_plus, -1) < 1e-8
    assert _comb_leakage(analytic_minus, +1) < 1e-8
    transverse = np.sqrt(np.abs(run.field.ex) ** 2 + np.abs(run.field.ey) ** 2)
    assert float(np.max(np.abs(run.field.ez)) / np.max(transverse)) < 1e-6

    rows = delta_rotation_sweep(cfg, deltas=(0.2, 0.7, 1.4), grid=grid)
    assert max(abs(float(row["error_rad"])) for row in rows) < 1e-3


def test_carrier_collinearity_is_two_sided() -> None:
    base = VectorArmConfig()
    slm1 = replace(base.slm1, carrier_lp_per_mm=1.5625, fill_factor=1.0)
    slm2 = replace(base.slm2, carrier_lp_per_mm=1.5625, carrier_sign=-1, fill_factor=1.0)
    cfg = replace(base, slm1=slm1, slm2=slm2, quantise=False, apply_fill_factor=False, apply_carrier=True)
    grid = default_vector_arm_grid(cfg, 128)

    run = run_vector_arm(cfg, grid=grid, return_debug=True)
    plus, minus = run.field.circular_components()
    report = carrier_collinearity_report(plus, minus, grid)
    assert report.separation_pixels < 0.5
    assert run.ledgers["slm1"]["relative_error"] < 1e-12
    assert run.ledgers["slm2"]["relative_error"] < 1e-12

    wrong = replace(cfg, slm2=replace(slm2, carrier_sign=+1))
    wrong_field = run_vector_arm(wrong, grid=grid)
    wrong_plus, wrong_minus = wrong_field.circular_components()
    wrong_report = carrier_collinearity_report(wrong_plus, wrong_minus, grid)
    assert wrong_report.separation_pixels > 10.0


def test_vector_axicon_h6_controls_degenerates_and_proposed_fingerprints() -> None:
    cfg = VectorArmConfig(ideal_components=True)
    grid = default_vector_arm_grid(cfg, 64)
    twin = default_config("fast")
    twin = replace(twin, grid=replace(twin.grid, axial_points=3))
    z_values = [-1e-6, 0.0, 1e-6]

    segmented = run_vector_arm(cfg, grid=grid)
    radial = run_vector_arm(cfg, grid=grid, all_radial=True)
    amp = gaussian_envelope(grid, cfg)
    scalar_x = VectorField(ex=amp, ey=np.zeros_like(amp), grid=grid, wavelength_m=cfg.wavelength_m)

    segmented_surface = run_vector_axicon_to_surface(segmented, twin, z_values_m=z_values)
    radial_surface = run_vector_axicon_to_surface(radial, twin, z_values_m=z_values)
    scalar_surface = run_vector_axicon_to_surface(scalar_x, twin, z_values_m=z_values)
    assert segmented_surface.intensity_stack.shape == (3, 64, 64)
    assert segmented_surface.surface.medium_before == pytest.approx(1.0)
    assert_locked_kr_fingerprint(segmented_surface.parameters.k_r_surface_m_inv)

    radius = _peak_annulus_radius(segmented_surface.intensity_stack[1], segmented_surface.focal_plane.grid)
    segmented_curve = h6_z_curve(
        segmented_surface.intensity_stack,
        segmented_surface.z_values_m,
        segmented_surface.focal_plane.grid,
        radius,
        angular_samples=1024,
    )
    radial_curve = h6_z_curve(
        radial_surface.intensity_stack,
        radial_surface.z_values_m,
        segmented_surface.focal_plane.grid,
        radius,
        angular_samples=1024,
    )
    scalar_curve = h6_z_curve(
        scalar_surface.intensity_stack,
        scalar_surface.z_values_m,
        segmented_surface.focal_plane.grid,
        radius,
        angular_samples=1024,
    )
    threshold = 0.6 * float(segmented_curve["best_h6"])
    assert_three_sided_acceptance(segmented_curve, radial_curve, scalar_curve, threshold)
    assert_degenerate_rejection(threshold)

    proposed_path = ROOT / "stage7_proposed_fingerprints.json"
    proposed = json.loads(proposed_path.read_text(encoding="utf-8"))
    assert proposed["status"] == "proposed_not_canonical_lock"
    for value in proposed["fingerprints"].values():
        if isinstance(value, str):
            float.fromhex(value)
