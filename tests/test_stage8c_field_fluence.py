"""
Stage 8C field-fluence tests.

Covers transverse integrals, energy-conserving plane/stack scaling, propagation
drift reporting, peak intensity via the Stage 8B conversion, and summary content.
"""

import math
import numpy as np
import pytest

from vbb_study.digital_twin.field_coupling import (
    plane_from_arrays,
    stack_from_arrays,
    SOURCE_UNIT_TEST_FIXTURE,
)
from vbb_study.digital_twin.field_fluence import (
    FluencePlaneResult,
    FluenceStackResult,
    transverse_integral_um2,
    integrated_energy_uJ_from_fluence,
    scale_plane_to_fluence,
    scale_stack_to_fluence,
    peak_intensity_from_fluence_result,
    field_fluence_summary,
    MODEL_STATUS,
)
from vbb_study.digital_twin.energy_accounting import peak_intensity_w_cm2


def _uniform_plane(n=10, dx=1.0, value=1.0):
    return plane_from_arrays(
        np.full((n, n), value, dtype=float), dx_um=dx, dy_um=dx,
        z_um=0.0, source_status=SOURCE_UNIT_TEST_FIXTURE,
    )


# ---------------------------------------------------------------------------
# transverse_integral_um2
# ---------------------------------------------------------------------------


def test_transverse_integral_basic():
    I = np.ones((10, 10))
    # sum(I)=100, dx=dy=2 → 100 * 2 * 2 = 400
    assert math.isclose(transverse_integral_um2(I, 2.0, 2.0), 400.0, rel_tol=1e-12)


def test_transverse_integral_rejects_bad_sampling():
    with pytest.raises(Exception):
        transverse_integral_um2(np.ones((4, 4)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# integrated_energy_uJ_from_fluence
# ---------------------------------------------------------------------------


def test_integrated_energy_recovers_pulse_energy():
    plane = _uniform_plane(n=12, dx=0.5)
    res = scale_plane_to_fluence(plane, pulse_energy_uJ=5.0)
    recovered = integrated_energy_uJ_from_fluence(res.fluence_j_cm2, res.dx_um, res.dy_um)
    assert math.isclose(recovered, 5.0, rel_tol=1e-9)


def test_integrated_energy_units():
    # F = 1 J/cm² over a 1 µm × 1 µm pixel: dA = 1e-8 cm²; E = 1e-8 J = 1e-2 µJ
    F = np.array([[1.0]])
    E_uJ = integrated_energy_uJ_from_fluence(F, 1.0, 1.0)
    assert math.isclose(E_uJ, 1e-2, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# scale_plane_to_fluence
# ---------------------------------------------------------------------------


def test_plane_scaling_conserves_energy():
    plane = _uniform_plane(n=16, dx=0.25)
    res = scale_plane_to_fluence(plane, pulse_energy_uJ=3.3)
    assert isinstance(res, FluencePlaneResult)
    assert math.isclose(res.integrated_energy_uJ, 3.3, rel_tol=1e-9)


def test_plane_peak_fluence_scales_linearly():
    plane = _uniform_plane(n=8, dx=0.5)
    r1 = scale_plane_to_fluence(plane, 1.0)
    r2 = scale_plane_to_fluence(plane, 2.0)
    assert math.isclose(r2.peak_fluence_j_cm2, 2.0 * r1.peak_fluence_j_cm2, rel_tol=1e-9)


def test_plane_scaling_model_status():
    res = scale_plane_to_fluence(_uniform_plane(), 1.0)
    assert res.model_status == MODEL_STATUS == "fluence_prediction"
    assert res.final_export_allowed is False


def test_plane_scaling_invalid_pulse_energy_raises():
    with pytest.raises(ValueError):
        scale_plane_to_fluence(_uniform_plane(), -1.0)


def test_plane_scaling_nonuniform_peak():
    I = np.zeros((5, 5))
    I[2, 2] = 10.0
    plane = plane_from_arrays(I, 1.0, 1.0, source_status=SOURCE_UNIT_TEST_FIXTURE)
    res = scale_plane_to_fluence(plane, 1.0)
    # All energy in one pixel: F_peak = E[J]/dA[cm²] = 1e-6 / 1e-8 = 100 J/cm²
    assert math.isclose(res.peak_fluence_j_cm2, 100.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# scale_stack_to_fluence
# ---------------------------------------------------------------------------


def _make_stack(nz=4, n=8, dx=0.5):
    x = (np.arange(n) - n / 2) * dx
    y = x.copy()
    z = np.linspace(0.0, 30.0, nz)
    I = np.ones((nz, n, n), dtype=float)
    # Make planes differ in raw intensity to exercise drift reporting.
    for i in range(nz):
        I[i] *= (i + 1)
    return stack_from_arrays(I, x, y, z, source_status=SOURCE_UNIT_TEST_FIXTURE)


def test_stack_scaling_conserves_energy_per_plane():
    stack = _make_stack()
    res = scale_stack_to_fluence(stack, pulse_energy_uJ=2.0)
    assert isinstance(res, FluenceStackResult)
    # Every plane must integrate to the pulse energy.
    assert np.allclose(res.transverse_energy_by_z_uJ, 2.0, rtol=1e-9)


def test_stack_drift_from_raw_integrals():
    stack = _make_stack(nz=4)  # raw integrals scale as 1,2,3,4
    res = scale_stack_to_fluence(stack, 1.0)
    # drift = (max-min)/max = (4-1)/4 = 0.75
    assert math.isclose(res.propagation_energy_drift_fraction, 0.75, rel_tol=1e-9)


def test_stack_peak_z_reported():
    stack = _make_stack(nz=5)
    res = scale_stack_to_fluence(stack, 1.0)
    assert res.peak_z_um in set(stack.z_um.tolist())


def test_stack_unsupported_normalisation_raises():
    with pytest.raises(ValueError, match="normalisation"):
        scale_stack_to_fluence(_make_stack(), 1.0, normalisation="bogus")


def test_stack_zero_plane_raises():
    x = np.linspace(-2, 2, 6)
    y = x.copy()
    z = np.linspace(0, 10, 3)
    I = np.ones((3, 6, 6))
    I[1] = 0.0  # zero transverse integral on a plane
    stack = stack_from_arrays(I, x, y, z, source_status=SOURCE_UNIT_TEST_FIXTURE)
    with pytest.raises(Exception):
        scale_stack_to_fluence(stack, 1.0)


def test_stack_invalid_pulse_energy_raises():
    with pytest.raises(ValueError):
        scale_stack_to_fluence(_make_stack(), float("nan"))


# ---------------------------------------------------------------------------
# peak_intensity_from_fluence_result — reuses Stage 8B conversion
# ---------------------------------------------------------------------------


def test_peak_intensity_uses_stage8b_conversion_plane():
    res = scale_plane_to_fluence(_uniform_plane(), 1.0)
    expected = peak_intensity_w_cm2(res.peak_fluence_j_cm2, 100.0).peak_intensity_w_cm2
    got = peak_intensity_from_fluence_result(res, 100.0)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_peak_intensity_stack_uses_global_peak():
    stack = _make_stack()
    res = scale_stack_to_fluence(stack, 1.0)
    peak_F = float(np.max(res.peak_fluence_by_z_j_cm2))
    expected = peak_intensity_w_cm2(peak_F, 200.0).peak_intensity_w_cm2
    got = peak_intensity_from_fluence_result(res, 200.0)
    assert math.isclose(got, expected, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# field_fluence_summary
# ---------------------------------------------------------------------------


def test_summary_plane_contains_status_and_caveat():
    s = field_fluence_summary(_uniform_plane(), 1.0, 260.0)
    assert s["kind"] == "plane"
    assert s["model_status"] == "fluence_prediction"
    assert s["final_export_allowed"] is False
    assert "caveat" in s and "not absorbed energy" in s["caveat"].lower()
    assert s["energy_conservation_residual_uJ"] < 1e-9


def test_summary_stack_contains_status_and_drift():
    s = field_fluence_summary(_make_stack(), 1.0, 260.0)
    assert s["kind"] == "stack"
    assert s["model_status"] == "fluence_prediction"
    assert "propagation_energy_drift_fraction" in s
    assert "caveat" in s
    assert s["max_transverse_energy_residual_uJ"] < 1e-9


def test_summary_rejects_bad_type():
    with pytest.raises(TypeError):
        field_fluence_summary(object(), 1.0, 100.0)
