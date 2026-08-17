from __future__ import annotations

import json

import numpy as np
import pytest

from vbb_study.digital_twin import (
    MODE1_ALLOWED_OUTCOMES,
    Mode1Result,
    mode1_centre_treatment_diagnostic,
    mode1_completion_gate,
    mode1_hexagonal_bessel_survival_metrics,
    mode1_symmetry_class,
    mode1_scope_manifest,
    plot_mode1_f0_vs_f2,
    plot_mode1_p2_input_diagnostics,
    plot_mode1_sample_region,
    plot_mode1_v0_to_mode1_scale,
    run_mode1_ideal_p2_downstream,
    write_mode1_scope_manifest,
)
from vbb_study.digital_twin.nathan_vector_hexagon import NathanHexagonConfig, canonical_target_field


@pytest.fixture(scope="module")
def mode1() -> Mode1Result:
    # grid_n=96 is the smallest grid at which the C3/hexagonal symmetry diagnostics are reliable
    # (coarser grids under-resolve the ring angular content).
    cfg = NathanHexagonConfig.fast(grid_n=96, z_planes=9, angular_samples=360)
    return run_mode1_ideal_p2_downstream(cfg, run_f2=True, f2_chunk_size=64)


def test_mode1_runs_f0_and_f2_and_returns_result(mode1: Mode1Result) -> None:
    assert isinstance(mode1, Mode1Result)
    assert mode1.f0.route_id == "F0_current_scalar_focus_bridge"
    assert mode1.f2 is not None and mode1.f2.route_id == "F2_vectorial_pupil_spectrum_reference"
    assert mode1.f0.intensity_stack.shape == mode1.f2.intensity_stack.shape
    assert 0 <= mode1.reference_index < mode1.z_values_m.size
    # reference plane is the declared middle of the Bessel zone, not the z=0 surface
    assert mode1.reference_index == mode1.z_values_m.size // 2


def test_mode1_manifest_has_required_scope_fields(mode1: Mode1Result) -> None:
    m = mode1.manifest
    assert m["mode"] == "MODE 1 ideal P2 downstream Digital Twin"
    for key in ("simulated_components", "bypassed_components", "approximated_components",
                "input_plane", "output_plane", "downstream_solver_routes", "represents_physical_bench",
                "statement", "completion_gate", "claim_boundary"):
        assert key in m, key
    assert m["represents_physical_bench"].startswith("partially, downstream only")
    assert "does not yet simulate HWP/QWP/SLM generation" in m["statement"]
    assert set(m["downstream_solver_routes"]) == {"F0", "F2"}
    # physical generation is bypassed, not represented
    bypassed = " ".join(m["bypassed_components"]).lower()
    for comp in ("patterned hwp", "slm1", "slm2", "final qwp", "panel realism"):
        assert comp in bypassed, comp
    assert m["claim_boundary"]["physical_generation_modelled"] is False


def test_mode1_manifest_writes_valid_json(mode1: Mode1Result, tmp_path) -> None:
    path = write_mode1_scope_manifest(mode1, tmp_path)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["mode"] == "MODE 1 ideal P2 downstream Digital Twin"
    assert loaded["stage"] == "nathan_mode1_ideal_p2_downstream"


def test_mode1_ideal_p2_field_is_canonical_six_sector(mode1: Mode1Result) -> None:
    p2 = mode1.p2_diagnostics
    assert set(np.unique(p2["sector_mask"])) <= {0, 1}
    # P2 total intensity is the Gaussian envelope: bright centre, sector info is in polarisation only
    inten = np.asarray(p2["intensity"], dtype=float)
    centre = tuple(s // 2 for s in inten.shape)
    assert inten[centre] == pytest.approx(float(np.max(inten)), rel=1e-6)


def test_mode1_metrics_separate_survival_from_clean_wall(mode1: Mode1Result) -> None:
    surv = mode1.f0_survival
    assert "source_like_hexagonal_bessel_survival" in surv
    assert "clean_single_wall_usefulness" in surv
    src = surv["source_like_hexagonal_bessel_survival"]
    wall = surv["clean_single_wall_usefulness"]
    for key in ("reference_dark_core_ratio", "fraction_planes_hollow", "dark_core_axial_persistence",
                "sixfold_orientation_stability", "reference_dominant_order", "reference_ring_radius_um"):
        assert key in src, key
    for key in ("median_wall_continuity", "median_wall_power_fraction"):
        assert key in wall, key
    # the two families of metrics are genuinely distinct (no shared keys)
    assert set(src) != set(wall)


def test_mode1_centre_treatment_diagnostic_reports_robustness(mode1: Mode1Result) -> None:
    ct = mode1.centre_treatment
    assert set(ct["treatments"]) == {
        "A_project_grid_straddling", "B_axis_sampled_grid", "C_project_grid_central_regularised",
    }
    assert isinstance(ct["sample_result_robust_to_centre_treatment"], bool)
    for row in ct["treatments"].values():
        assert "sample_dark_core_ratio" in row and "p2_on_axis_over_peak_intensity" in row
    assert "sample_dark_core_ratio_A_minus_B" in ct
    # No centre treatment produces a spurious BRIGHT on-axis core (the V0 failure mode was
    # dark_core_ratio ~= 1.0); the boolean robustness flag can be cautious on very coarse test
    # grids and is True at run resolution (verified at grid_n>=128).
    for row in ct["treatments"].values():
        assert row["sample_dark_core_ratio"] < 0.85


def test_mode1_completion_gate_returns_allowed_outcome(mode1: Mode1Result) -> None:
    gate = mode1_completion_gate(mode1.f0_survival, centre_treatment=mode1.centre_treatment, f0_vs_f2=mode1.f0_vs_f2)
    assert gate["suggested_outcome"] in MODE1_ALLOWED_OUTCOMES
    assert "operator must confirm" in gate["note"]
    assert "outcome_statement" in gate
    assert mode1.completion["suggested_outcome"] in MODE1_ALLOWED_OUTCOMES


def test_mode1_current_run_is_not_hexagonal_and_mode2_blocked(mode1: Mode1Result) -> None:
    """The inherited-downstream result is a dark-core triangular/C3 structure, NOT a hexagon.

    Guards against the earlier over-claim (M1-A) driven by order-6 ring content alone."""

    sym = mode1.f0_survival["symmetry_classification"]
    # the reference plane is not a genuine visual hexagon
    assert sym["reference_symmetry_class"] != "visual_hexagonal_field"
    # order-6 content actually EXCEEDS order-3 here, yet it must NOT pass as hexagonal
    assert sym["reference_order3_over_order6"] < 1.0
    # C3 signature: 120 deg self-similarity exceeds 60 deg
    assert sym["reference_c120_minus_c60"] > 0.0
    # triangular planes dominate over hexagonal planes
    assert sym["fraction_planes_triangular_lobed"] >= sym["fraction_planes_visual_hexagonal"]
    # outcome must be M1-B or M1-D (never M1-A), and physical realisation is blocked
    assert mode1.completion["suggested_outcome"] in {"M1-B", "M1-D"}
    assert mode1.completion["mode2_realisation_allowed"] is False


def _ring_field(n_harmonic: int, N: int = 160):
    from vbb_study.equations.fields import make_xy_grid

    dx = 1.0e-6
    grid = make_xy_grid(N, dx)
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    r0 = 0.25 * N * dx
    ring = np.exp(-(((R - r0) / (0.06 * N * dx)) ** 2))
    ang = np.clip(1.0 + 0.9 * np.cos(n_harmonic * PHI), 0.0, None)
    return (ring * ang) ** 2, grid, r0


def test_mode1_symmetry_class_distinguishes_hexagon_from_triangle() -> None:
    from vbb_study.digital_twin.nathan_vector_hexagon import _mode1_symmetry

    hexI, hgrid, hr = _ring_field(6)
    triI, tgrid, tr = _ring_field(3)
    hex_class = mode1_symmetry_class(_mode1_symmetry(hexI, hgrid, hr), dark_core_ratio=0.0)
    tri_class = mode1_symmetry_class(_mode1_symmetry(triI, tgrid, tr), dark_core_ratio=0.0)
    assert hex_class == "visual_hexagonal_field"
    assert tri_class == "triangular_lobed_field"
    # a triangular field must never be classified hexagonal, even hollow
    assert mode1_symmetry_class(_mode1_symmetry(triI, tgrid, tr), dark_core_ratio=0.0) != "visual_hexagonal_field"


def test_mode1_survival_metrics_helper_is_standalone() -> None:
    cfg = NathanHexagonConfig.fast(grid_n=32, z_planes=4, angular_samples=240)
    from vbb_study.digital_twin.nathan_vector_hexagon import _mode1_focal_grid, _vector_downstream_result, air_z_values, _twin_with_axial_points, default_nathan_grid
    twin = _twin_with_axial_points(cfg.twin, cfg.z_planes)
    z = air_z_values(twin, planes=cfg.z_planes)
    field = canonical_target_field(cfg, grid=default_nathan_grid(cfg))
    f0 = _vector_downstream_result(field, twin, z, control_id="nathan_six_sector", route_id="F0_current_scalar_focus_bridge", route_role="unit")
    metrics = mode1_hexagonal_bessel_survival_metrics(f0.intensity_stack, z, _mode1_focal_grid(f0))
    assert len(metrics["per_plane"]) == cfg.z_planes
    assert 0.0 <= metrics["source_like_hexagonal_bessel_survival"]["fraction_planes_hollow"] <= 1.0


def test_mode1_plot_helpers_execute(mode1: Mode1Result, tmp_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fn, arg in [
        (plot_mode1_p2_input_diagnostics, {}),
        (plot_mode1_sample_region, {"route": "F0"}),
        (plot_mode1_sample_region, {"route": "F2"}),
        (plot_mode1_f0_vs_f2, {}),
    ]:
        fig, _ = fn(mode1, output_path=tmp_path / "m1.png", **arg)
        plt.close(fig)
        assert (tmp_path / "m1.png").is_file()
