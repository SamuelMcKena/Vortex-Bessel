from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.phase2e_production_repair import (
    build_nominal_source,
    physical_on_axis_trace,
    production_sampling_gate,
    selected_plane_reference_rows,
)


def test_n3072_passes_production_sampling_gate() -> None:
    gate = production_sampling_gate(3072)
    assert gate["samples_per_radial_period"] > 12.0
    assert gate["preferred_phase_increment_gate"] is True
    assert gate["production_sampling_pass"] is True


def test_n512_fails_production_sampling_gate() -> None:
    gate = production_sampling_gate(512)
    assert gate["production_sampling_pass"] is False


def test_nominal_route_has_no_historical_hard_pupil() -> None:
    source, grid, metadata = build_nominal_source("B0", grid_n=256, aperture_model="none")
    assert source.shape == (256, 256)
    assert grid["N"] == 256
    assert metadata["aperture_model"] == "none"
    assert metadata["historical_objective_pupil_application_count"] == 0
    assert metadata["objective_transform_application_count"] == 0
    assert metadata["aperture_retained_power_fraction"] == 1.0


def test_hard_aperture_is_explicit_sensitivity_case() -> None:
    nominal, _, nominal_meta = build_nominal_source("B0", grid_n=256, aperture_model="none")
    hard, _, hard_meta = build_nominal_source("B0", grid_n=256, aperture_model="hard")
    assert hard_meta["aperture_role"].endswith("sensitivity_case")
    assert hard_meta["historical_objective_pupil_application_count"] == 1
    assert hard_meta["aperture_retained_power_fraction"] < 1.0
    assert not np.allclose(nominal, hard)
    assert nominal_meta["aperture_role"] == "no_additional_real_space_aperture"


def test_physical_axis_is_evaluated_at_zero_not_native_half_pixel() -> None:
    source, grid, metadata = build_nominal_source("B0", grid_n=128, aperture_model="none")
    assert not np.any(np.isclose(np.asarray(grid["x"]), 0.0))
    values = physical_on_axis_trace(
        source,
        grid,
        float(metadata["wavelength_m"]),
        (20e-3, 40e-3),
    )
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def test_selected_plane_rows_use_fixed_physical_window() -> None:
    rows = selected_plane_reference_rows(
        "B0",
        n_values=(128, 192),
        z_values_m=(20e-3,),
        window_m=10e-3,
    )
    assert len(rows) == 2
    assert {row["window_m"] for row in rows} == {10e-3}
    assert {row["z_m"] for row in rows} == {20e-3}
    assert all(np.isfinite(row["physical_on_axis_intensity_raw"]) for row in rows)
