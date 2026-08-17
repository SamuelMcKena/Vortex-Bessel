from __future__ import annotations

from dataclasses import replace

import pytest

from vbb_study.digital_twin import (
    Mode1ECandidateResult,
    NathanHexagonConfig,
    audit_mode1c_aperture_ring_limit,
    audit_mode1c_kr_mapping,
    build_mode1e_source_template,
    make_mode1e_redesigned_config,
    mode1e_na_required,
    mode1e_outcome_report,
    mode1e_required_pre_kr,
    mode1e_scope_manifest,
    mode1e_surface_kr_from_mapping,
    run_mode1e_candidate,
    run_mode1e_current_inherited_control,
)


@pytest.fixture(scope="module")
def base_config() -> NathanHexagonConfig:
    return NathanHexagonConfig.fast()


@pytest.fixture(scope="module")
def mapping(base_config):
    return audit_mode1c_kr_mapping(base_config)


@pytest.fixture(scope="module")
def aperture(base_config):
    return audit_mode1c_aperture_ring_limit(base_config)


@pytest.fixture(scope="module")
def template_n12():
    return build_mode1e_source_template(12.0, grid_n=384, z_planes=9)


def _na_for(ring_count: float, radius_m: float, mapping) -> float:
    k_pre = mode1e_required_pre_kr(ring_count, radius_m)
    k_surface = mode1e_surface_kr_from_mapping(k_pre, mapping.k_scale_surface_over_pre)
    return mode1e_na_required(k_surface, mapping.wavelength_m)


def test_n12_at_slm_safe_radius_is_within_objective_na(mapping, aperture) -> None:
    na = _na_for(12.0, aperture.p2_radius_max_with_safety_m, mapping)

    assert na < 0.45
    assert na == pytest.approx(0.3935, abs=2e-3)


def test_n31_at_slm_safe_radius_requires_na_above_0p9_and_is_infeasible(base_config, mapping, aperture) -> None:
    na = _na_for(31.0, aperture.p2_radius_max_with_safety_m, mapping)
    cfg2, cand = make_mode1e_redesigned_config(
        base_config,
        p2_radius_m=aperture.p2_radius_max_with_safety_m,
        target_ring_count=31.0,
    )

    assert na > 0.9
    assert cfg2 is None
    assert cand.config_status == "infeasible_by_budget"
    assert cand.run_status == "infeasible_not_run"
    assert not cand.within_objective_na


def test_redesigned_config_resolves_pre_kr_close_to_requested(base_config, aperture) -> None:
    cfg2, cand = make_mode1e_redesigned_config(
        base_config,
        p2_radius_m=aperture.p2_radius_max_with_safety_m,
        target_ring_count=12.0,
    )

    assert cfg2 is not None
    assert cand.config_status == "redesigned_config_resolved"
    from vbb_study.vector_axicon import resolve_vector_axicon_parameters

    params = resolve_vector_axicon_parameters(cfg2.twin)
    assert params.k_r_pre_m_inv == pytest.approx(cand.k_r_pre_m_inv, rel=1e-9)


def test_redesigned_config_resolves_surface_kr_or_reports_fingerprint_mismatch(base_config, aperture) -> None:
    cfg2, cand = make_mode1e_redesigned_config(
        base_config,
        p2_radius_m=aperture.p2_radius_max_with_safety_m,
        target_ring_count=12.0,
    )

    assert cand.config_status in {"redesigned_config_resolved", "surface_kr_fingerprint_mismatch"}
    if cand.config_status == "redesigned_config_resolved":
        from vbb_study.vector_axicon import resolve_vector_axicon_parameters

        params = resolve_vector_axicon_parameters(cfg2.twin)
        assert params.k_r_surface_m_inv == pytest.approx(cand.k_r_surface_m_inv, rel=1e-9)
    else:
        assert cand.run_status == "blocked_fingerprint_mismatch"
        assert cand.failure_reason


def test_source_template_n12_builds_as_accepted_visual_hexagon(template_n12) -> None:
    assert template_n12.symmetry_class == "visual_hexagonal_field"
    assert template_n12.accepted_hexagon is True
    assert template_n12.ring_radius_m > 0.0
    assert template_n12.angular_profile.size > 0
    assert template_n12.as_mode1b_template().classification == "visual_hexagonal_field"


def test_redesigned_actual_downstream_run_does_not_trip_locked_fingerprint(base_config, aperture, template_n12) -> None:
    result = run_mode1e_candidate(
        base_config,
        {"N12": template_n12},
        target_ring_count=12.0,
        p2_radius_m=aperture.p2_radius_max_with_safety_m,
        grid_n=96,
        z_planes=5,
    )

    assert result.run_status == "actual_downstream_f0"
    assert result.is_actual_downstream_run
    assert result.kr_pre_match_rel_error < 1e-9
    assert result.kr_surface_match_rel_error < 1e-9
    assert result.candidate.symmetry_class in {
        "visual_hexagonal_field",
        "triangular_lobed_field",
        "dark_core_structured_field",
    }
    assert "N12" in result.template_scores


def test_current_inherited_control_remains_non_hexagonal(base_config, template_n12) -> None:
    control = run_mode1e_current_inherited_control(
        base_config,
        {"N12": template_n12},
        grid_n=128,
        z_planes=9,
    )

    assert control.tier == "current_inherited_control"
    assert control.is_actual_downstream_run
    assert control.candidate.symmetry_class != "visual_hexagonal_field"
    assert control.template_gate_pass is False


def _fake_result(base_config, aperture, **candidate_overrides) -> Mode1ECandidateResult:
    _, cand = make_mode1e_redesigned_config(
        base_config,
        p2_radius_m=aperture.p2_radius_max_with_safety_m,
        target_ring_count=12.0,
    )
    return Mode1ECandidateResult(candidate=replace(cand, **candidate_overrides))


def test_proxy_only_candidates_cannot_produce_m1e_a(base_config, mapping, aperture) -> None:
    proxy = _fake_result(
        base_config,
        aperture,
        run_status="proxy_only_run",
        symmetry_class="visual_hexagonal_field",
        template_gate_pass=True,
    )
    report = mode1e_outcome_report(mapping=mapping, aperture=aperture, results=[proxy])

    assert report["suggested_outcome"] != "M1E-A"
    assert report["mode2a_2b_realisation_allowed"] is False
    assert report["n_non_actual_gate_passes_excluded"] == 1
    assert report["n_confirmed_with_f2"] == 0


def test_actual_f0_pass_without_f2_confirmation_is_not_m1e_a(base_config, mapping, aperture) -> None:
    unconfirmed = _fake_result(
        base_config,
        aperture,
        run_status="actual_downstream_f0",
        symmetry_class="visual_hexagonal_field",
        template_gate_pass=True,
    )
    report = mode1e_outcome_report(mapping=mapping, aperture=aperture, results=[unconfirmed])

    assert report["suggested_outcome"] == "M1E-D"
    assert report["mode2a_2b_realisation_allowed"] is False


def test_mode2a_2b_remains_blocked_unless_m1e_a(base_config, mapping, aperture) -> None:
    failing = _fake_result(
        base_config,
        aperture,
        run_status="actual_downstream_f0",
        symmetry_class="triangular_lobed_field",
        template_gate_pass=False,
        failure_reason=("reference-plane class is triangular_lobed_field, not visual_hexagonal_field",),
    )
    blocked_report = mode1e_outcome_report(mapping=mapping, aperture=aperture, results=[failing])
    blocked_manifest = mode1e_scope_manifest(blocked_report)

    assert blocked_report["suggested_outcome"] == "M1E-B"
    assert blocked_report["mode2a_2b_gate"] == "blocked"
    assert blocked_manifest["mode2a_2b_realisation_allowed"] is False
    assert blocked_manifest["physical_route_approval"] is False

    confirmed = replace(
        _fake_result(
            base_config,
            aperture,
            run_status="actual_downstream_f0_f2",
            symmetry_class="visual_hexagonal_field",
            template_gate_pass=True,
        ),
        f0_vs_f2={"reference_full_field_correlation": 0.9},
    )
    open_report = mode1e_outcome_report(mapping=mapping, aperture=aperture, results=[confirmed])
    open_manifest = mode1e_scope_manifest(open_report)

    assert open_report["suggested_outcome"] == "M1E-A"
    assert open_report["mode2a_2b_gate"] == "open_only_for_confirmed_redesigned_configuration"
    assert open_manifest["mode2a_2b_realisation_allowed"] is True
    assert open_manifest["physical_route_approval"] is False
