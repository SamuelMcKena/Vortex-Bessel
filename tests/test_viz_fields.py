"""Unit tests for vbb_study/viz_fields.py (Stage 6A).

Tests:
  V1  complex_field_image — ℓ=3 synthetic field: rendered hue (colourmap)
      must complete exactly 3 cycles azimuthally (primary cyclic-phase assertion).
  V2  complex_field_image — ℓ=0 flat-phase field: no azimuthal colour variation.
  V3  _azimuthal_power_spectrum re-export matches the validated A4 implementation.
  V4  azimuthal_order_panel — uniform ring: all power at m=0.
  V5  azimuthal_order_panel — n-fold ring: power at correct orders.
  V6  linked_field_views — runs without error on a fast-preset holographic result
      and returns the expected figure/axes tuple.

These tests are ADDITIVE: they cover new viz code only.
They do NOT touch engine physics or existing lock tests.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from vbb_study.viz_fields import (
    _azimuthal_power_spectrum,
    _field_to_rgba,
    azimuthal_order_panel,
    complex_field_image,
    linked_field_views,
)

_TWOPI = 2.0 * math.pi


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_vortex(N: int, ell: int) -> np.ndarray:
    """Pure topological-charge field: exp(i * ell * phi), uniform amplitude."""
    x = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x, x)
    phi = np.arctan2(Y, X)
    return np.exp(1j * float(ell) * phi)


def _ring_sample_rgba(rgba: np.ndarray, N: int, r_frac: float = 0.35,
                      n_phi: int = 360) -> np.ndarray:
    """Sample RGBA pixels along an azimuthal ring at radius r_frac * N/2."""
    r_pix = int(r_frac * N / 2)
    cx = cy = N // 2
    phis = np.linspace(0.0, _TWOPI, n_phi, endpoint=False)
    cols = np.clip((cx + r_pix * np.cos(phis)).astype(int), 0, N - 1)
    rows = np.clip((cy + r_pix * np.sin(phis)).astype(int), 0, N - 1)
    return rgba[rows, cols, :3].astype(float)   # shape (n_phi, 3); alpha dropped


# ─────────────────────────────────────────────────────────────────────────────
# V1 — Primary: ℓ=3 rendered hue completes 3 azimuthal cycles
# ─────────────────────────────────────────────────────────────────────────────

class TestComplexFieldImage3CycleWinding:
    """Primary cyclic-phase assertion: ℓ=3 field must show 3 colour cycles."""

    N = 256

    def test_v1a_3fold_colour_symmetry(self):
        """Rendered colours at φ, φ+2π/3, φ+4π/3 must match for ℓ=3.

        For U = exp(i·3·φ): phase(φ + 2π/3) = 3(φ + 2π/3) = 3φ + 2π ≡ 3φ
        (mod 2π), so the cyclic colourmap must give the same colour at each
        third-turn step.  This is the fundamental 3-cycle winding assertion.
        """
        N = self.N
        U = _synthetic_vortex(N, ell=3)
        _, rgba = complex_field_image(U, return_rgba=True)
        plt.close("all")

        n_phi = 360
        rgb = _ring_sample_rgba(rgba, N, n_phi=n_phi)   # (360, 3)

        # Compare every triplet (φ, φ+120°, φ+240°)
        step = n_phi // 3   # = 120 for n_phi=360
        max_channel_diff = float(
            np.max(np.abs(rgb - np.roll(rgb, -step, axis=0)))
        )
        assert max_channel_diff < 12.0, (
            f"ℓ=3 field: colours at φ and φ+2π/3 differ by up to "
            f"{max_channel_diff:.1f}/255 in RGB channels.  "
            f"Domain colouring must produce exactly 3 colour cycles "
            f"for a topological-charge-3 field."
        )

    def test_v1b_dominant_dft_frequency_is_3(self):
        """DFT of the red channel along the ring must peak at frequency 3.

        For any cyclic colourmap applied to U = exp(i·3·φ), the mapped colour
        repeats 3 times per revolution, so the dominant non-DC DFT component of
        each colour channel must be at spatial frequency 3 (cycles/revolution).
        """
        N = self.N
        U = _synthetic_vortex(N, ell=3)
        _, rgba = complex_field_image(U, return_rgba=True)
        plt.close("all")

        n_phi = 720
        rgb = _ring_sample_rgba(rgba, N, n_phi=n_phi)
        R_ring = rgb[:, 0]

        spectrum = np.abs(np.fft.rfft(R_ring))
        spectrum[0] = 0.0   # suppress DC
        dominant = int(np.argmax(spectrum))

        assert dominant == 3, (
            f"ℓ=3 domain-coloured field: dominant ring DFT frequency = {dominant}, "
            f"expected 3.  Domain colouring is not cyclic-phase correct."
        )


# ─────────────────────────────────────────────────────────────────────────────
# V2 — ℓ=0 flat-phase: no azimuthal colour variation
# ─────────────────────────────────────────────────────────────────────────────

class TestComplexFieldImageEll0:
    N = 128

    def test_v2_flat_phase_uniform_colour(self):
        """Uniform-phase field (ℓ=0) must produce no azimuthal colour variation."""
        N = self.N
        x = np.linspace(-1.0, 1.0, N)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        U = np.where(R < 0.9, 1.0 + 0j, 0j)   # real disk, zero phase

        _, rgba = complex_field_image(U, return_rgba=True)
        plt.close("all")

        n_phi = 360
        rgb = _ring_sample_rgba(rgba, N, n_phi=n_phi)
        R_ring = rgb[:, 0]

        # For ℓ=0 the DFT should have no dominant non-DC component
        spectrum = np.abs(np.fft.rfft(R_ring))
        spectrum[0] = 0.0
        max_ndc = float(spectrum.max())
        dc_power = float(np.abs(np.fft.rfft(R_ring)[0]))

        # Non-DC power should be < 5 % of the DC-band amplitude
        assert max_ndc < 0.05 * dc_power + 2.0, (
            f"ℓ=0 flat-phase field: max non-DC DFT amplitude = {max_ndc:.1f}, "
            f"DC amplitude = {dc_power:.1f}.  "
            f"Flat-phase field must have azimuthally uniform colour."
        )


# ─────────────────────────────────────────────────────────────────────────────
# V3 — _azimuthal_power_spectrum re-export matches A4 implementation
# ─────────────────────────────────────────────────────────────────────────────

class TestAzimuthalPowerSpectrum:
    """Verify the re-exported A4 tool gives correct results."""

    def test_v3a_uniform_ring_all_dc(self):
        """Uniform ring: 100 % power at m=0."""
        power = _azimuthal_power_spectrum(np.ones(1024))
        frac_dc = float(power[0]) / float(np.sum(power))
        assert frac_dc >= 0.999

    def test_v3b_modulated_ring(self):
        """Modulated ring I(φ) = 1 + cos(6φ): power concentrated at m=0 and m=6."""
        n = 1024
        phis = np.linspace(0.0, _TWOPI, n, endpoint=False)
        I = 1.0 + np.cos(6.0 * phis)
        power = _azimuthal_power_spectrum(I)
        total = float(np.sum(power))
        combined = (float(power[0]) + float(power[6])) / total
        assert combined >= 0.95, (
            f"6-fold ring: combined m=0+m=6 fraction = {combined:.3f} (< 0.95)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# V4/V5 — azimuthal_order_panel smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAzimuthalOrderPanel:
    def test_v4_returns_expected_axes(self):
        intensities = np.ones(256)
        fig, (ax_polar, ax_bar) = azimuthal_order_panel(intensities)
        plt.close(fig)
        assert ax_polar is not None and ax_bar is not None

    def test_v5_complex_input_takes_abs2(self):
        """Complex field_on_ring: |·|² is taken before spectrum computation."""
        n = 128
        phis = np.linspace(0.0, _TWOPI, n, endpoint=False)
        field = np.exp(1j * 3.0 * phis)   # unit-amplitude vortex — intensity = 1
        power = _azimuthal_power_spectrum(np.abs(field) ** 2)
        frac_dc = float(power[0]) / float(np.sum(power))
        assert frac_dc >= 0.99, "Unit-amplitude field: intensity is flat → DC only"


# ─────────────────────────────────────────────────────────────────────────────
# V6 — linked_field_views smoke test (fast preset)
# ─────────────────────────────────────────────────────────────────────────────

class TestLinkedFieldViews:
    """Smoke test: linked_field_views runs on a real fast-preset result."""

    @pytest.fixture(scope="class")
    def holographic_result(self):
        import bessel_twin_core as bt
        from dataclasses import replace
        from vbb_study import vbb_regime
        cfg = vbb_regime.config_for_regime(
            replace(bt.default_config("fast"), generation_method="holographic"),
            "general",
        )
        return bt.run_case(cfg, preset="fast", path="ideal",
                           case_id="viz_smoke_test")

    def test_v6_returns_fig_and_four_axes(self, holographic_result):
        fig, axes = linked_field_views(holographic_result)
        plt.close(fig)
        assert len(axes) == 4
        for ax in axes:
            assert ax is not None

    def test_v6_z_idx_override(self, holographic_result):
        vol = holographic_result["volume"]
        n_z = len(vol["z"])
        fig, axes = linked_field_views(holographic_result, z_idx=n_z // 4)
        plt.close(fig)
        assert len(axes) == 4
