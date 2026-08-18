import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE = (Path(__file__).resolve().parents[1] / "notebooks" / "experimental" /
          "axicon_aberration_correction" / "q20_phase_physics.py")
spec = importlib.util.spec_from_file_location("q20_phase_physics", MODULE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def angle_err(a, b):
    return abs(np.angle(np.exp(1j * (a - b))))


def test_cone_geometry_and_annulus_spacing():
    wl = 1030e-9
    kr = 4.9e5
    g = mod.cone_geometry(wl, kr)
    assert 0 < g.alpha_rad < 0.2
    z = np.array([-2.0, -1.0, 0.0]) * 1e-3
    mapping = mod.annulus_mapping_from_z(
        z, wavelength_m=wl, k_perp_m_inv=kr)
    assert np.allclose(np.diff(mapping.rho_m), 1e-3 * g.tan_alpha)
    assert not mapping.absolute_radius_calibrated


def test_annulus_absolute_mapping():
    mapping = mod.annulus_mapping_from_z(
        np.array([-2.0, -1.0, 0.0]) * 1e-3,
        wavelength_m=1030e-9,
        k_perp_m_inv=4.9e5,
        z_at_relative_zero_from_axicon_m=25e-3,
    )
    assert mapping.absolute_radius_calibrated
    assert mapping.radius_reference == "absolute"
    assert np.all(mapping.rho_m > 0)


def test_gauge_removes_unobservable_row_piston():
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    base = 0.7 * np.cos(2 * theta) - 0.25 * np.sin(3 * theta)
    rows = np.stack([base + 0.2, base + 2.4, base - 2.8])
    aligned = mod.gauge_annular_phase_rows(rows)
    for i in range(1, 3):
        diff = np.angle(np.exp(1j * (aligned[i] - aligned[0])))
        assert np.max(np.abs(diff)) < 1e-10


def test_phasor_interpolation_respects_wrap_seam():
    deg = np.pi / 180
    phase = np.array([[179 * deg] * 16, [-179 * deg] * 16])
    out = mod._bilinear_phasor_interpolation(
        np.exp(1j * phase),
        np.array([1.0, 2.0]),
        np.array([[1.5]]),
        np.array([[0.7]]),
    )
    assert angle_err(np.angle(out[0, 0]), np.pi) < 2 * deg


def test_transverse_residual_excludes_target_vortex_and_blocks_hardware():
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    z = np.linspace(-17, 0, 18) * 1e-3
    rows = np.stack([
        0.35 * np.cos(2 * theta + 0.2 * i) + 0.1 * np.sin(3 * theta)
        for i in range(len(z))
    ])
    out = mod.assemble_transverse_residual_phase(
        rows,
        z,
        wavelength_m=1030e-9,
        k_perp_m_inv=4.9e5,
        grid_size=128,
    )
    assert out["contains_programmed_vortex_phase"] is False
    assert out["radial_piston_recovered"] is False
    assert out["hardware_ready"] is False
    assert out["radius_reference"] == "relative"
    phase = out["residual_phase_rad"]
    assert np.isfinite(phase).any()
    assert np.isnan(phase).any()


def test_central_band_sections_keep_full_signed_width():
    stack = np.zeros((5, 11, 13))
    stack[:, 5, :] = 1
    stack[:, :, 6] += 2
    xz, yz = mod.central_band_sections(stack, half_width_px=0)
    assert xz.shape == (5, 13)
    assert yz.shape == (5, 11)
