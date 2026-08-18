from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_consistent_multiplane import (
    RetrievalGrid,
    ideal_relative_phase,
    multiplane_phase_retrieval,
    propagate_stack,
    signed_longitudinal_sections,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


def _corr(a, b):
    av = np.asarray(a, float).ravel()
    bv = np.asarray(b, float).ravel()
    return float(np.corrcoef(av, bv)[0, 1])


def test_propagate_stack_uses_relative_distance_to_reference():
    rg = RetrievalGrid(n=48, dx_m=8e-6, wavelength_m=1030e-9)
    grid = rg.grid()
    field = np.exp(-(grid["R"] / 90e-6) ** 2) * np.exp(
        1j * (2.0 * grid["PHI"] + 0.7 * grid["X"] / 90e-6)
    )
    z = np.asarray([-2.0e-3, 0.0, 3.0e-3])
    stack = propagate_stack(field, z, 1, rg)

    assert np.allclose(stack[1], field, rtol=1e-12, atol=1e-12)
    expected_back = angular_spectrum_propagate_bl(
        field, grid, rg.wavelength_m, -2.0e-3,
        bandlimit=True, include_evanescent=False,
    )
    assert np.allclose(stack[0], expected_back, rtol=1e-11, atol=1e-11)


def test_residual_phase_removes_complete_nominal_vortex_and_radial_phase():
    n = 96
    yy, xx = np.indices((n, n))
    x = (xx - (n - 1) / 2) / (n / 2)
    y = (yy - (n - 1) / 2) / (n / 2)
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    amp = np.exp(-(r / 0.7) ** 2)

    nominal_phase = 20.0 * th - 7.0 * r + 0.8 * r**2
    true_residual = 0.35 * x - 0.22 * y + 0.18 * r**2 * np.cos(2 * th)
    nominal = amp * np.exp(1j * nominal_phase)
    retrieved = nominal * np.exp(1j * true_residual)
    support = r < 0.75

    residual, confidence = ideal_relative_phase(
        retrieved, nominal, support=support, smooth_sigma_px=0.0, confidence_floor=0.0
    )
    valid = support & np.isfinite(residual) & (confidence > 0)
    # Piston is deliberately removed by the implementation, so compare after
    # removing the circular mean from the known residual too.
    known = true_residual[valid]
    known -= np.angle(np.sum(np.exp(1j * known)))
    phase_error = np.angle(np.exp(1j * (residual[valid] - known)))
    assert float(np.sqrt(np.mean(phase_error**2))) < 1e-8


def test_one_global_field_reproduces_synthetic_multiplane_stack():
    rg = RetrievalGrid(n=48, dx_m=9e-6, wavelength_m=1030e-9)
    grid = make_xy_grid(rg.n, rg.dx_m)
    amp = np.exp(-(grid["R"] / 95e-6) ** 2)
    phase = 3 * grid["PHI"] - 2.0e5 * grid["R"] + 0.3 * grid["X"] / 95e-6
    field_ref = amp * np.exp(1j * phase)
    z = np.asarray([-1.2e-3, -0.6e-3, 0.0, 0.7e-3, 1.4e-3])
    truth = propagate_stack(field_ref, z, 2, rg)

    amplitudes = []
    for field in truth:
        a = np.abs(field)
        a /= np.linalg.norm(a)
        amplitudes.append(a)
    amplitudes = np.stack(amplitudes)

    result = multiplane_phase_retrieval(
        amplitudes,
        z,
        rg,
        reference_index=2,
        initial_phase=np.angle(field_ref),
        iterations=8,
        relaxation=0.6,
    )
    predicted = result.predicted_intensity
    target = np.stack([np.abs(u) ** 2 / np.max(np.abs(u) ** 2) for u in truth])
    assert min(_corr(predicted[i], target[i]) for i in range(len(z))) > 0.995


def test_longitudinal_sections_preserve_real_transverse_drift():
    stack = np.zeros((5, 31, 31), float)
    for iz in range(5):
        stack[iz, 15, 10 + 2 * iz] = 1.0
    xz, yz = signed_longitudinal_sections(stack, band_px=0)
    assert list(np.argmax(xz, axis=1)) == [10, 12, 14, 16, 18]
    # The y-z section is sampled at fixed x=center and therefore only sees the
    # plane whose drifting spot crosses that coordinate.
    assert int(np.argmax(np.max(yz, axis=1))) == 2
