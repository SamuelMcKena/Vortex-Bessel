from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from vbb_study.digital_twin import (
    MODE1B_ALLOWED_OUTCOMES,
    Mode1BCandidate,
    Mode1BFeasibility,
    Mode1BTemplateScore,
    audit_mode1b_ring_count_planes,
    build_mode1b_target_template,
    effective_ring_count_for_plane,
    mode1b_candidate_passes_hexagon_gate,
    mode1b_candidate_row,
    mode1b_completion_gate,
    mode1b_feasibility,
    one_over_e_field_radius_from_intensity,
    run_mode1_ideal_p2_downstream,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    NathanHexagonConfig,
    _mode1_focal_grid,
    _mode1_symmetry,
    _v0_plane_diagnostics,
    compare_to_v0_template,
    mode1_symmetry_class,
)
from vbb_study.equations.fields import make_xy_grid


@pytest.fixture(scope="module")
def v0_target():
    return build_mode1b_target_template(grid_n=384, z_planes=21)


def _ring_field(n_harmonic: int, N: int = 160, *, bright_core: bool = False):
    dx = 1.0e-6
    grid = make_xy_grid(N, dx)
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    r0 = 0.25 * N * dx
    ring = np.exp(-(((R - r0) / (0.06 * N * dx)) ** 2))
    ang = np.clip(1.0 + 0.9 * np.cos(n_harmonic * PHI), 0.0, None)
    image = (ring * ang) ** 2
    if bright_core:
        image = image + 3.0 * np.exp(-(R / (0.05 * N * dx)) ** 2)
    return image, grid, r0


def _plausible_feasibility() -> Mode1BFeasibility:
    return mode1b_feasibility(
        wavelength_m=1030e-9,
        n_axicon=1.458,
        n_medium=1.0,
        base_angle_deg=8.0,
        beam_radius_m=120e-6,
        objective_na=0.45,
        slm_pixel_pitch_m=8e-6,
    )


def _good_score() -> Mode1BTemplateScore:
    return Mode1BTemplateScore(
        angular_profile_correlation=0.90,
        xy_correlation=0.80,
        x_profile_correlation=0.80,
        y_profile_correlation=0.80,
        best_rotation_deg=0.0,
        candidate_ring_radius_m=40e-6,
        target_ring_radius_m=40e-6,
        scale_factor=1.0,
    )


def test_one_over_e_field_radius_from_intensity_recovers_gaussian_radius() -> None:
    grid = make_xy_grid(256, 2.0e-6)
    R = np.asarray(grid["R"], dtype=float)
    w = 80.0e-6
    intensity = np.exp(-2.0 * R**2 / w**2)
    measured = one_over_e_field_radius_from_intensity(intensity, grid)
    assert measured == pytest.approx(w, rel=0.03)


def test_v0_target_template_builds_as_visual_hexagonal_field(v0_target) -> None:
    assert v0_target.classification == "visual_hexagonal_field"
    assert v0_target.metadata["v0_effective_ring_count"] > 20.0
    assert v0_target.dark_core_ratio < 0.01


def test_v0_ring_count_remains_about_31() -> None:
    from vbb_study.digital_twin import NathanSourceParityConfig

    cfg = NathanSourceParityConfig()
    rings = effective_ring_count_for_plane(radius_1e_field_m=cfg.beam_radius_m, k_r_m_inv=cfg.k_r_m_inv)
    assert rings == pytest.approx(31.0, rel=0.03)


def test_ring_count_audit_includes_required_planes() -> None:
    audit = audit_mode1b_ring_count_planes()
    plane_ids = {row["plane_id"] for row in audit.rows}
    assert {
        "v0_source_plane",
        "old_mode1b_mixed_sample_radius_pre_axicon_kr",
        "p2_handoff_plane",
        "axicon_or_pupil_input_plane",
        "sample_plane_design_radius",
    } <= plane_ids
    assert audit.corrected_ring_count_at_p2 is not None
    assert audit.corrected_ring_count_at_axicon_or_pupil is not None


def test_large_old_new_ring_count_ratio_blocks_final_m1b_b_claim() -> None:
    audit = audit_mode1b_ring_count_planes()
    assert audit.old_new_ratio is not None
    assert audit.old_new_ratio > 5.0
    assert "cannot be treated as final" in audit.conclusion


def test_current_inherited_mode1_fails_mode1b_hexagon_gate(v0_target) -> None:
    cfg = NathanHexagonConfig.fast(grid_n=96, z_planes=7, angular_samples=240)
    mode1 = run_mode1_ideal_p2_downstream(cfg, run_f2=False, run_centre_treatment=False)
    plane = np.asarray(mode1.f0.intensity_stack[mode1.reference_index], dtype=float)
    grid = _mode1_focal_grid(mode1.f0)
    diag = _v0_plane_diagnostics(plane, grid)
    symmetry = _mode1_symmetry(plane, grid, diag["ring_radius_m"])
    cls = mode1_symmetry_class(symmetry, diag["central_core_darkness"])
    score = compare_to_v0_template(plane, grid, v0_target, candidate_ring_radius_m=diag["ring_radius_m"])
    passed, reasons = mode1b_candidate_passes_hexagon_gate(
        symmetry_class=cls,
        symmetry=symmetry,
        template_score=score,
        dark_core_ratio=diag["central_core_darkness"],
        feasibility=_plausible_feasibility(),
    )
    assert cls == "triangular_lobed_field" or not passed
    assert not passed
    assert reasons


def test_synthetic_triangular_annulus_fails_gate() -> None:
    image, grid, ring = _ring_field(3)
    symmetry = _mode1_symmetry(image, grid, ring)
    cls = mode1_symmetry_class(symmetry, dark_core_ratio=0.0)
    passed, reasons = mode1b_candidate_passes_hexagon_gate(
        symmetry_class=cls,
        symmetry=symmetry,
        template_score=_good_score(),
        dark_core_ratio=0.0,
        feasibility=_plausible_feasibility(),
    )
    assert cls == "triangular_lobed_field"
    assert not passed
    assert any("visual_hexagonal_field" in reason or "order-3" in reason for reason in reasons)


def test_synthetic_hexagonal_annulus_passes_gate() -> None:
    image, grid, ring = _ring_field(6)
    symmetry = _mode1_symmetry(image, grid, ring)
    cls = mode1_symmetry_class(symmetry, dark_core_ratio=0.0)
    passed, reasons = mode1b_candidate_passes_hexagon_gate(
        symmetry_class=cls,
        symmetry=symmetry,
        template_score=_good_score(),
        dark_core_ratio=0.0,
        feasibility=_plausible_feasibility(),
    )
    assert cls == "visual_hexagonal_field"
    assert passed
    assert reasons == ()


def test_high_order6_content_alone_cannot_pass() -> None:
    image, grid, ring = _ring_field(6, bright_core=True)
    symmetry = _mode1_symmetry(image, grid, ring)
    cls = mode1_symmetry_class(symmetry, dark_core_ratio=1.0)
    passed, reasons = mode1b_candidate_passes_hexagon_gate(
        symmetry_class=cls,
        symmetry=symmetry,
        template_score=_good_score(),
        dark_core_ratio=1.0,
        feasibility=_plausible_feasibility(),
    )
    assert not passed
    assert any("central core" in reason or "visual_hexagonal_field" in reason for reason in reasons)


def test_feasibility_detects_slm_infeasible_phase_periods() -> None:
    f = mode1b_feasibility(
        wavelength_m=1030e-9,
        n_axicon=1.458,
        n_medium=1.0,
        base_angle_deg=20.0,
        beam_radius_m=80e-6,
        objective_na=0.45,
        slm_pixel_pitch_m=8e-6,
    )
    assert not f.slm_encoded_axicon_feasible
    assert "phase period too fine" in " ".join(f.notes)


def test_feasibility_marks_high_base_angle_exploratory() -> None:
    f = mode1b_feasibility(
        wavelength_m=1030e-9,
        n_axicon=1.458,
        n_medium=1.0,
        base_angle_deg=20.0,
        beam_radius_m=80e-6,
        objective_na=0.45,
        slm_pixel_pitch_m=8e-6,
    )
    assert f.thin_axicon_paraxial_warning
    assert f.feasibility_class == "exploratory_high_angle_redesign"


def test_candidate_schema_is_stable() -> None:
    names = {field.name for field in fields(Mode1BCandidate)}
    required = {
        "candidate_id",
        "beam_radius_m",
        "base_angle_deg",
        "sector_rotation_rad",
        "z_reference_m",
        "grid_n",
        "symmetry_class",
        "template_score",
        "feasibility",
        "dark_core_ratio",
        "ring_radius_m",
        "pass_hexagon_gate",
        "fail_reasons",
        "output_paths",
        "model_family",
        "radius_plane_id",
        "radius_1e_field_m",
        "k_r_m_inv",
        "ring_count",
    }
    assert required <= names


def test_candidate_rows_include_radius_plane_and_model_family() -> None:
    candidate = _candidate_with(_plausible_feasibility(), model_family="plane_corrected_free_space_continuation")
    row = mode1b_candidate_row(candidate)
    assert row["radius_plane_id"] == "p2_handoff_plane"
    assert row["model_family"] == "plane_corrected_free_space_continuation"


def _candidate_with(
    feasibility: Mode1BFeasibility,
    *,
    model_family: str = "plane_corrected_free_space_continuation",
) -> Mode1BCandidate:
    return Mode1BCandidate(
        candidate_id="schema_candidate",
        beam_radius_m=80e-6,
        base_angle_deg=feasibility.axicon_base_angle_deg,
        sector_rotation_rad=0.0,
        z_reference_m=1e-3,
        grid_n=64,
        symmetry_class="visual_hexagonal_field",
        template_score=_good_score(),
        feasibility=feasibility,
        dark_core_ratio=0.0,
        ring_radius_m=40e-6,
        pass_hexagon_gate=True,
        fail_reasons=(),
        model_family=model_family,
        radius_plane_id="p2_handoff_plane",
        radius_1e_field_m=80e-6,
        k_r_m_inv=feasibility.k_r_m_inv,
        ring_count=feasibility.ring_count,
    )


def test_mode2_gate_blocked_unless_outcome_is_realistic() -> None:
    exploratory = _candidate_with(
        mode1b_feasibility(
            wavelength_m=1030e-9,
            n_axicon=1.458,
            n_medium=1.0,
            base_angle_deg=20.0,
            beam_radius_m=80e-6,
            objective_na=0.45,
            slm_pixel_pitch_m=8e-6,
        )
    )
    exploratory_gate = mode1b_completion_gate([exploratory])
    assert exploratory_gate["suggested_outcome"] == "M1B-A-exploratory"
    assert exploratory_gate["mode2_realisation_allowed"] is False

    realistic = _candidate_with(_plausible_feasibility(), model_family="actual_inherited_downstream")
    realistic_gate = mode1b_completion_gate([realistic])
    assert realistic_gate["suggested_outcome"] == "M1B-A-realistic"
    assert realistic_gate["suggested_outcome"] in MODE1B_ALLOWED_OUTCOMES
    assert realistic_gate["mode2_realisation_allowed"] is True
