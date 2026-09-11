from __future__ import annotations

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.short_bessel import design_short_bessel
from vbb_study.short_bessel_realizability import (
    combined_slm_phase_preview,
    propagate_vortex_case,
    vortex_conical_field,
)
from vbb_study.short_bessel_propagation import ShortBesselPropagationConfig
from vbb_study.viz_fields import phase_winding


UM = 1.0e-6


def test_equivalent_vortex_input_has_requested_winding() -> None:
    design = design_short_bessel()
    grid = make_xy_grid(257, 0.35 * UM)
    radius_m = 30.0 * UM

    for ell in (0, 1, 3):
        field = vortex_conical_field(
            design,
            ell=ell,
            sample_beam_radius_m=radius_m,
            grid=grid,
        )
        # Use a contour outside the singular core but comfortably inside the beam.
        winding = phase_winding(field, grid, 10.0 * UM, n_phi=720)
        assert abs(winding - ell) < 0.03


def test_v1_v3_propagation_preserves_topological_charge() -> None:
    design = design_short_bessel()
    cfg = ShortBesselPropagationConfig(
        grid_n=257,
        dx_um=0.35,
        z_min_um=0.0,
        z_max_um=300.0,
        z_points=41,
    )
    for ell in (1, 3):
        case = propagate_vortex_case(
            design,
            ell=ell,
            sample_beam_radius_m=30.0 * UM,
            config=cfg,
            snapshot_z_um=(),
        )
        assert abs(case.metrics.measured_phase_winding - ell) < 0.1
        assert case.metrics.measured_main_ring_diameter_um is not None
        assert case.metrics.measured_main_ring_diameter_um > 0.0
        assert case.metrics.predicted_main_ring_diameter_um is not None


def test_b0_propagation_stays_near_requested_core_scale() -> None:
    design = design_short_bessel()
    cfg = ShortBesselPropagationConfig(
        grid_n=257,
        dx_um=0.35,
        z_min_um=0.0,
        z_max_um=350.0,
        z_points=51,
    )
    case = propagate_vortex_case(
        design,
        ell=0,
        sample_beam_radius_m=30.0 * UM,
        config=cfg,
        snapshot_z_um=(),
    )
    assert case.metrics.central_fwhm_um is not None
    assert abs(case.metrics.central_fwhm_um - 5.0) < 0.35


def test_slm_radial_plus_vortex_preview_is_8bit_and_charge_dependent() -> None:
    design = design_short_bessel()
    p0 = combined_slm_phase_preview(design, ell=0, nx=320, ny=180)
    p3 = combined_slm_phase_preview(design, ell=3, nx=320, ny=180)

    assert p0["gray_uint8"].dtype == np.uint8
    assert p3["gray_uint8"].dtype == np.uint8
    assert p0["gray_uint8"].shape == (180, 320)
    assert not np.array_equal(p0["gray_uint8"], p3["gray_uint8"])
