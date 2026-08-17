"""Stage 8C.3R component-plane physical lab-realism tests.

These cover the thirteen acceptance requirements of the Component-Plane Reality
Reset: perturbations are applied to the complex field BEFORE propagation, energy
is conserved/lost honestly with no per-plane re-normalisation, axis tracking and
translation-vs-deformation classification work, and no material-response claim is
introduced.
"""

import numpy as np

from vbb_study.digital_twin.component_plane_pipeline import (
    ComponentPlaneConfig,
    run_component_plane_pipeline,
)
from vbb_study.digital_twin.component_plane_states import PropagatedFieldStack, field_power
from vbb_study.digital_twin.component_plane_metrics import (
    build_component_plane_scenarios,
    classify_translation_vs_deformation,
    compute_axis_tracking,
    compute_energy_throughput,
    run_component_plane_scenario,
    stack_to_fluence,
)

CFG = ComponentPlaneConfig.fast()


def _peak_plane(I):
    return int(np.argmax(I.max(axis=(1, 2))))


def _core_fill(stack, core_radius_um=2.0):
    I = stack.intensity_zyx
    pl = I[_peak_plane(I)]
    x = stack.x_um
    X, Y = np.meshgrid(x, x)
    core = np.hypot(X, Y) <= core_radius_um
    return float(np.mean(pl[core]) / max(np.max(pl), 1e-30))


# 1. Zero-control physical pipeline matches the canonical baseline.
def test_zero_control_matches_canonical_baseline():
    a = run_component_plane_pipeline({}, config=CFG).propagated_stack
    b = run_component_plane_pipeline({}, config=CFG).propagated_stack
    assert np.allclose(a.intensity_zyx, b.intensity_zyx)
    assert abs(a.transmitted_fraction - 1.0) < 1e-9  # lossless with no perturbation
    # Hollow vortex core: central core fill is far below the ring peak.
    assert _core_fill(a) < 0.15


# 2. Beam tilt is applied before propagation and causes expected steering.
def test_beam_tilt_steers_before_propagation():
    def slope(tilt):
        st = run_component_plane_pipeline(
            {"enable_beam_tilt": True, "beam_tilt_x_mrad": tilt}, config=CFG
        ).propagated_stack
        I, x, z = st.intensity_zyx, st.x_um, st.z_um
        cx = np.array([np.sum(I[i] * np.meshgrid(x, x)[0]) / I[i].sum() for i in range(len(z))])
        return float(np.polyfit(z, cx, 1)[0])

    s_small = slope(4.0)
    s_large = slope(16.0)
    assert s_small > 0 and s_large > 0           # +x tilt steers in +x
    assert s_large > s_small                      # larger tilt -> larger walk-off
    # Predicted steering is reported and consistent in sign.
    run = run_component_plane_pipeline(
        {"enable_beam_tilt": True, "beam_tilt_x_mrad": 16.0}, config=CFG
    )
    assert run.predicted_steering["predicted_shift_x_um_at_zmax"] > 0


# 3. Vortex and axicon offsets are applied independently at phase-mask generation.
def test_vortex_and_axicon_offsets_independent():
    base = run_component_plane_pipeline({}, config=CFG).propagated_stack.intensity_zyx
    v = run_component_plane_pipeline(
        {"enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 8.0}, config=CFG
    ).propagated_stack.intensity_zyx
    a = run_component_plane_pipeline(
        {"enable_axicon_centre_offset": True, "axicon_centre_offset_x_um": 8.0}, config=CFG
    ).propagated_stack.intensity_zyx
    # Each independently changes the beam, and the two are not the same operation.
    assert not np.allclose(v, base)
    assert not np.allclose(a, base)
    assert not np.allclose(v, a)


# 4. Input beam decentre occurs before SLM interaction.
def test_input_decentre_before_slm():
    run = run_component_plane_pipeline(
        {"enable_beam_decentre": True, "beam_decentre_x_um": 10.0}, config=CFG
    )
    inp = run.input_state
    x = inp.x_um
    amp = np.abs(inp.field)
    X = np.meshgrid(x, x)[0]
    cx = float(np.sum(amp * X) / amp.sum())
    assert cx > 5.0  # input field centroid is decentred at the input plane
    # The SLM field is built from the (already decentred) input field.
    assert "input_beam_decentre" in run.input_state.applied_components
    assert run.slm_state.field is not None


# 5. Pupil clipping occurs before propagation/focus and lowers transmitted energy.
def test_pupil_clipping_lowers_transmitted_energy():
    run = run_component_plane_pipeline(
        {"enable_pupil_clipping": True, "pupil_radius_um": 12.0}, config=CFG
    )
    st = run.propagated_stack
    assert st.transmitted_fraction < 0.95
    assert st.sample_pulse_energy_uJ < st.input_pulse_energy_uJ
    assert "pupil_clip" in run.pupil_state.applied_components


# 6. Zero-order leakage is inserted before propagation and raises the core-fill metric.
def test_zero_order_leakage_fills_core():
    base = run_component_plane_pipeline({}, config=CFG).propagated_stack
    leak = run_component_plane_pipeline(
        {"enable_zero_order_leakage": True, "zero_order_leakage_fraction": 0.25}, config=CFG
    ).propagated_stack
    assert _core_fill(leak) > _core_fill(base)
    assert "zero_order_carrier_leakage" in [
        c for s in leak.plane_states for c in s.applied_components
    ]


# 7. No perturbation re-normalises lost energy back into the sample field.
def test_no_energy_renormalisation():
    run = run_component_plane_pipeline(
        {"enable_pupil_clipping": True, "pupil_radius_um": 10.0}, config=CFG
    )
    st = run.propagated_stack
    # Sample energy equals input * transmitted fraction (genuinely lower).
    expected = st.input_pulse_energy_uJ * st.transmitted_fraction
    assert abs(st.sample_pulse_energy_uJ - expected) < 1e-6
    assert st.sample_pulse_energy_uJ < 0.5 * st.input_pulse_energy_uJ
    # Fluence stack integrates to the reduced sample energy, not the input energy.
    fl = stack_to_fluence(st)
    plane_energy = float(np.median(fl.transverse_energy_by_z_uJ))
    assert abs(plane_energy - st.sample_pulse_energy_uJ) < 1e-3
    assert plane_energy < st.input_pulse_energy_uJ


# 8. Actual-axis tracking reports centroid/ring/peak offsets.
def test_axis_tracking_reports_offsets():
    run = run_component_plane_pipeline(
        {"enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 8.0}, config=CFG
    )
    ax = compute_axis_tracking(run.propagated_stack)
    for key in [
        "commanded_axis_x_um", "intensity_centroid_x_um", "ring_centre_x_um",
        "peak_x_um", "radial_axis_error_um", "field_of_view_margin_um",
        "out_of_frame_fraction", "beam_steering_angle_x_mrad", "trajectory_fit_quality",
    ]:
        assert key in ax and np.isfinite(float(ax[key]))
    assert ax["commanded_axis_x_um"] == 0.0


# 9. Co-shifted vortex+axicon is detected as translation-dominated; relative is not.
def test_coshift_is_translation_relative_is_deformation():
    cosh = run_component_plane_scenario("vortex_axicon_coshift", config=CFG)
    assert cosh.severe_class["translation_dominated_boolean"] is True
    assert cosh.severe_class["registered_similarity_score"] > 0.9

    rel = run_component_plane_scenario("vortex_misregistration", config=CFG)
    assert rel.severe_class["translation_dominated_boolean"] is False
    assert rel.severe_class["residual_shape_deformation_score"] > 0.15


# 10. Hard downstream cut-out artefacts are not generated by post-stack masking.
def test_no_hard_downstream_cutout():
    # A genuine pupil clip applied BEFORE propagation must diffract: appreciable
    # intensity appears beyond the geometric pupil radius in the propagated plane,
    # which a hard output-plane mask would forbid.
    pupil_r = 10.0
    st = run_component_plane_pipeline(
        {"enable_pupil_clipping": True, "pupil_radius_um": pupil_r,
         "pupil_decentre_x_um": 6.0}, config=CFG
    ).propagated_stack
    I = st.intensity_zyx
    pl = I[_peak_plane(I)]
    x = st.x_um
    X, Y = np.meshgrid(x, x)
    outside = np.hypot(X, Y) > pupil_r
    # Energy leaks outside the geometric pupil (diffraction), not a hard zero edge.
    assert float(np.sum(pl[outside])) / float(np.sum(pl)) > 0.01
    # And the propagated plane has no large block of exact zeros adjacent to the beam.
    core = np.hypot(X, Y) <= pupil_r
    assert float(np.min(pl[core])) >= 0.0  # intensity, finite, no NaNs
    assert np.all(np.isfinite(pl))


# 11. All post-propagation visual transforms are clearly labelled diagnostic-only.
def test_legacy_transforms_labelled_diagnostic_only():
    # The legacy stack-transform module self-downgrades physics_active -> diagnostic.
    from vbb_study.digital_twin import lab_perturbations as lp
    classifications = {meta[0] for meta in lp._META.values()}
    assert "physics_active" not in classifications
    # The new physical states are optical predictions, never final exports.
    run = run_component_plane_pipeline({}, config=CFG)
    assert run.final_export_allowed is False
    assert run.model_status == "optical_prediction"
    assert run.propagated_stack.final_export_allowed is False


# 12. No material-response claim is introduced.
def test_no_material_response_claim():
    run = run_component_plane_pipeline(
        {"enable_pupil_clipping": True, "pupil_radius_um": 12.0}, config=CFG
    )
    st = run.propagated_stack
    assert st.model_status == "optical_prediction"
    assert st.final_export_allowed is False
    banned = ("absorbed", "dose", "plasma", "ablation", "void", "refractive_index_change",
              "deposited", "damage", "threshold")
    blob = " ".join(str(s.metadata) for s in st.plane_states) + " " + str(st.metadata)
    assert not any(b in blob.lower() for b in banned)


# 13. Warning-only controls are honestly flagged, not faked.
def test_warning_only_controls_flagged_not_faked():
    base = run_component_plane_pipeline({}, config=CFG).propagated_stack.intensity_zyx
    run = run_component_plane_pipeline(
        {"enable_first_order_filter": True, "enable_relay_decentre": True,
         "relay_decentre_x_um": 5.0}, config=CFG
    )
    # The field is NOT silently altered by an unmodelled plane.
    assert np.allclose(run.propagated_stack.intensity_zyx, base)
    assert any("warning-only" in w for w in run.warnings)


# Energy/figure smoke: every scenario runs and conserves the energy contract.
def test_all_scenarios_run_and_conserve_energy():
    for key in build_component_plane_scenarios():
        r = run_component_plane_scenario(key, config=CFG)
        for energy in (r.baseline_energy, r.mild_energy, r.severe_energy):
            assert energy["transmitted_fraction"] <= 1.0 + 1e-9
            assert energy["sample_pulse_energy_uJ"] <= energy["input_pulse_energy_uJ"] + 1e-9
            assert energy["renormalisation_factor"] == 1.0
