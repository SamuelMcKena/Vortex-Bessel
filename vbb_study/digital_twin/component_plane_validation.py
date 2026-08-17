"""Stage 8C.3R.1 free-space reference-plane validation.

This module validates the Stage 8C.3R component-plane upstream model at a
**free-space reference plane** (n = 1.0), before any material/interface model
exists.  It provides:

  * ``canonical_free_space_reference`` - an independent ideal-field construction
    + locked angular-spectrum propagation, used as the canonical baseline;
  * ``zero_control_equivalence`` - proves the zero-control pipeline reproduces
    that canonical baseline (and the analytical Bessel-Gauss expectation);
  * ``compute_energy_audit`` - raw field power per component, transmitted
    fractions, encircled/core/annular/side-lobe energy redistribution,
    ``energy_accounting_valid`` and ``peak_rise_supported_by_energy_redistribution``;
  * ``validate_beam_tilt`` - measured vs analytical free-space steering slope;
  * ``fov_convergence_check`` - standard vs expanded grid/FOV reliability state.

No sample material model is active.  All outputs are optical/fluence diagnostics
only; ``final_export_allowed=False``.  No material modification / absorbed energy
/ plasma / index change is computed or claimed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np
from scipy import special as sp

from vbb_study.equations.fields import make_xy_grid, gaussian_amplitude
from vbb_study.equations.propagation import angular_spectrum_propagate_bl

from vbb_study.digital_twin.component_plane_pipeline import (
    ComponentPlaneConfig,
    ComponentPlaneRun,
    run_component_plane_pipeline,
)
from vbb_study.digital_twin.component_plane_states import field_power
from vbb_study.digital_twin.component_plane_metrics import (
    compute_axis_tracking,
    compute_energy_throughput,
    stack_to_fluence,
    _centroid,
    _peak_plane_index,
    _peak_xy,
    _cosine_similarity,
)
from vbb_study.digital_twin.annular_axis_tracking import (
    estimate_annular_axis,
    track_axis_trajectory,
)

_UM = 1e-6

# Documented Stage 8C.3R.1 tolerances (see docs/34).
TOL_EQUIV_SIMILARITY = 1.0 - 1e-9
TOL_EQUIV_POWER_REL = 1e-9
TOL_RING_RADIUS_REL = 0.12
TOL_CORE_NULL_RATIO = 0.15
TOL_TILT_SLOPE_REL = 0.15
TOL_FOV_PEAK_REL = 0.05


# ---------------------------------------------------------------------------
# Canonical free-space reference (independent of the pipeline closure)
# ---------------------------------------------------------------------------


def canonical_free_space_reference(config: ComponentPlaneConfig | None = None) -> dict[str, Any]:
    """Build the ideal zero-control field from the equation helpers and propagate
    it with the locked band-limited angular-spectrum propagator (n = 1.0).

    This is an independent code path from ``run_component_plane_pipeline`` and is
    used as the canonical baseline for zero-control equivalence.
    """
    config = config or ComponentPlaneConfig()
    grid = make_xy_grid(int(config.grid_N), config.dx_m)
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    x_um = np.asarray(grid["x"], dtype=float) * 1e6

    w0_m = float(config.beam_waist_um) * _UM
    kr = float(config.kr_rad_per_um) / _UM
    amp = gaussian_amplitude(R, w0_m)
    phi = float(config.ell) * PHI - kr * R
    E0 = amp * np.exp(1j * phi)

    z_um = np.linspace(float(config.z_min_um), float(config.z_max_um), int(config.n_z))
    intensity = np.empty((z_um.size, x_um.size, x_um.size), dtype=float)
    for i, zi in enumerate(z_um):
        U = angular_spectrum_propagate_bl(
            E0, grid, config.wavelength_m, float(zi) * _UM,
            n_medium=float(config.n_medium), bandlimit=bool(config.bandlimit),
        )
        intensity[i] = np.abs(U) ** 2

    ell = abs(int(config.ell))
    ring_analytical_um = (
        0.0 if ell == 0 else float(sp.jnp_zeros(ell, 1)[0] / kr) * 1e6
    )
    return {
        "entrance_field": E0,
        "intensity_zyx": intensity,
        "x_um": x_um,
        "z_um": z_um,
        "entrance_power": field_power(E0, config.dx_um, config.dx_um),
        "ring_radius_analytical_um": ring_analytical_um,
        "n_medium": float(config.n_medium),
        "propagation_medium": "free_space_n_1.0",
    }


# ---------------------------------------------------------------------------
# Zero-control equivalence
# ---------------------------------------------------------------------------


def zero_control_equivalence(config: ComponentPlaneConfig | None = None) -> dict[str, Any]:
    """Compare the zero-control component-plane path with the canonical reference."""
    config = config or ComponentPlaneConfig()
    run = run_component_plane_pipeline({}, config=config)
    ref = canonical_free_space_reference(config)

    st = run.propagated_stack
    Ipipe = np.asarray(st.intensity_zyx, dtype=float)
    Iref = ref["intensity_zyx"]
    x = np.asarray(st.x_um, dtype=float)

    # Reference-plane complex field (free-space reference entrance).
    Epipe = np.asarray(run.reference_plane_state.field)
    Eref = ref["entrance_field"]
    cfield_sim = _complex_cosine_similarity(Epipe, Eref)

    sel = _peak_plane_index(Iref)
    intensity_sim = _cosine_similarity(Ipipe[sel], Iref[sel])

    Fpipe = np.asarray(stack_to_fluence(st).fluence_zyx_j_cm2, dtype=float)
    # Canonical fluence: scale the reference stack to the same reference energy.
    fluence_sim = _cosine_similarity(Fpipe[sel], _normalise_like(Iref[sel], Fpipe[sel]))

    peak_pipe = float(np.max(Fpipe[sel]))
    peak_ref_intensity = float(np.max(Iref[sel]))
    peak_pipe_int = float(np.max(Ipipe[sel]))
    peak_fluence_rel_diff = abs(peak_pipe_int - peak_ref_intensity) / max(peak_ref_intensity, 1e-30)

    px_pipe = _peak_xy(Ipipe[sel], x, x)
    px_ref = _peak_xy(Iref[sel], x, x)
    peak_pos_diff = float(np.hypot(px_pipe[0] - px_ref[0], px_pipe[1] - px_ref[1]))

    cen_pipe = _centroid(Ipipe[sel], x, x)
    cen_ref = _centroid(Iref[sel], x, x)
    axis_pos_diff = float(np.hypot(cen_pipe[0] - cen_ref[0], cen_pipe[1] - cen_ref[1]))

    power_pipe = field_power(Epipe, st.dx_um, st.dy_um)
    power_rel_diff = abs(power_pipe - ref["entrance_power"]) / max(ref["entrance_power"], 1e-30)

    # Analytical Bessel-Gauss physical checks.
    ax = compute_axis_tracking(st, plane_index=sel, core_radius_um=2.0)
    ring_meas = float(ax["ring_radius_um"])
    ring_anal = float(ref["ring_radius_analytical_um"])
    ring_rel_diff = (abs(ring_meas - ring_anal) / ring_anal) if ring_anal > 0 else 0.0
    core_null_ratio = float(ax["core_fill_fraction"])

    # Equivalence is gated on the robust field/power metrics. The (x,y) peak
    # argmax is azimuthally degenerate on a rotationally symmetric ring, so a tiny
    # FP difference between the two propagator call orders can flip it to the
    # opposite side of the ring; it is reported for information only.
    equivalent = bool(
        cfield_sim >= TOL_EQUIV_SIMILARITY
        and intensity_sim >= TOL_EQUIV_SIMILARITY
        and fluence_sim >= TOL_EQUIV_SIMILARITY
        and power_rel_diff <= TOL_EQUIV_POWER_REL
        and axis_pos_diff <= float(st.dx_um) + 1e-9
    )
    physically_valid = bool(
        ring_rel_diff <= TOL_RING_RADIUS_REL and core_null_ratio <= TOL_CORE_NULL_RATIO
    )
    return {
        "complex_field_similarity": cfield_sim,
        "intensity_similarity": intensity_sim,
        "fluence_similarity": fluence_sim,
        "peak_fluence_rel_diff": float(peak_fluence_rel_diff),
        "peak_position_diff_um": peak_pos_diff,
        "peak_position_note": "argmax is azimuthally degenerate on a symmetric ring; informational only",
        "axis_position_diff_um": axis_pos_diff,
        "raw_field_power_rel_diff": float(power_rel_diff),
        "ring_radius_measured_um": ring_meas,
        "ring_radius_analytical_um": ring_anal,
        "ring_radius_rel_diff": float(ring_rel_diff),
        "core_null_ratio": core_null_ratio,
        "selected_plane_index": int(sel),
        "equivalent_within_tolerance": equivalent,
        "physically_valid_bessel_gauss": physically_valid,
        "tolerances": {
            "similarity_min": TOL_EQUIV_SIMILARITY,
            "power_rel_max": TOL_EQUIV_POWER_REL,
            "ring_radius_rel_max": TOL_RING_RADIUS_REL,
            "core_null_ratio_max": TOL_CORE_NULL_RATIO,
        },
        "propagation_medium": "free_space_n_1.0",
    }


# ---------------------------------------------------------------------------
# Energy / normalisation audit
# ---------------------------------------------------------------------------


def _radial_energy_fractions(plane: np.ndarray, x: np.ndarray, y: np.ndarray,
                             cx: float, cy: float, ring_radius_um: float,
                             peak_x: float, peak_y: float) -> dict[str, float]:
    total = float(np.sum(plane))
    if total <= 0:
        return {k: 0.0 for k in ("core", "annulus", "side_lobe", "encircled_near_peak")}
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - cx, Y - cy)
    rr = max(ring_radius_um, float(np.mean(np.abs(np.diff(x)))))
    core = r <= 0.5 * rr
    annulus = (r > 0.5 * rr) & (r <= 1.5 * rr)
    side = r > 1.5 * rr
    rp = np.hypot(X - peak_x, Y - peak_y)
    encircled = rp <= 0.5 * rr
    return {
        "core": float(np.sum(plane[core]) / total),
        "annulus": float(np.sum(plane[annulus]) / total),
        "side_lobe": float(np.sum(plane[side]) / total),
        "encircled_near_peak": float(np.sum(plane[encircled]) / total),
    }


def compute_energy_audit(
    run: ComponentPlaneRun,
    *,
    baseline_run: ComponentPlaneRun | None = None,
    plane_index: int | None = None,
    fov_reliable: bool = True,
) -> dict[str, Any]:
    """Full Stage 8C.3R.1 energy/normalisation audit for one run."""
    st = run.propagated_stack
    base = compute_energy_throughput(run)
    I = np.asarray(st.intensity_zyx, dtype=float)
    x = np.asarray(st.x_um, dtype=float)
    if plane_index is None:
        plane_index = _peak_plane_index(I)
    plane = I[int(plane_index)]

    # Raw field power before/after every component (from each captured field).
    ledger = []
    states = list(st.plane_states)
    prev_power = None
    for s in states:
        p_after = None if s.field is None else field_power(s.field, s.dx_um, s.dy_um)
        ledger.append({
            "plane": s.plane_name,
            "energy_before_uJ": float(s.pulse_energy_before_uJ),
            "energy_after_uJ": float(s.pulse_energy_after_uJ),
            "transmitted_fraction": float(s.transmitted_fraction),
            "raw_field_power_before": float(prev_power) if prev_power is not None else float("nan"),
            "raw_field_power_after": float(p_after) if p_after is not None else float("nan"),
            "applied": list(s.applied_components),
        })
        if p_after is not None:
            prev_power = p_after

    ax = compute_axis_tracking(st, plane_index=int(plane_index), core_radius_um=2.0)
    cen = (ax["intensity_centroid_x_um"], ax["intensity_centroid_y_um"])
    fractions = _radial_energy_fractions(
        plane, x, x, cen[0], cen[1], float(ax["ring_radius_um"]),
        float(ax["peak_x_um"]), float(ax["peak_y_um"]),
    )

    # No component may add power (transmitted fraction > 1 within fp tolerance).
    no_gain = all(row["transmitted_fraction"] <= 1.0 + 1e-9 for row in ledger)
    renorm_ok = abs(base["renormalisation_factor"] - 1.0) < 1e-12
    total_ok = st.transmitted_fraction <= 1.0 + 1e-9
    energy_accounting_valid = bool(no_gain and renorm_ok and total_ok)

    peak_fluence = float(base["peak_fluence_J_cm2"])
    peak_rise_status = "no_peak_rise"
    peak_rise_supported = False
    if baseline_run is not None:
        base_energy = compute_energy_throughput(baseline_run)
        base_peak = float(base_energy["peak_fluence_J_cm2"])
        energy_fell = st.sample_pulse_energy_uJ < baseline_run.propagated_stack.sample_pulse_energy_uJ - 1e-9
        base_ax = compute_axis_tracking(baseline_run.propagated_stack,
                                        plane_index=int(plane_index), core_radius_um=2.0)
        base_plane = baseline_run.propagated_stack.intensity_zyx[int(plane_index)]
        base_frac = _radial_energy_fractions(
            base_plane, x, x, base_ax["intensity_centroid_x_um"], base_ax["intensity_centroid_y_um"],
            float(base_ax["ring_radius_um"]), float(base_ax["peak_x_um"]), float(base_ax["peak_y_um"]),
        )
        concentrated = fractions["encircled_near_peak"] >= base_frac["encircled_near_peak"] - 1e-6
        peak_rose = peak_fluence > base_peak + 1e-9
        if peak_rose:
            peak_rise_supported = bool(
                energy_fell and renorm_ok and energy_accounting_valid
                and fov_reliable and concentrated
            )
            peak_rise_status = "peak_rise_supported" if peak_rise_supported else "peak_rise_unvalidated"

    audit = dict(base)
    audit.update({
        "per_plane_ledger": ledger,
        "selected_plane_index": int(plane_index),
        "encircled_energy_fraction_near_peak": fractions["encircled_near_peak"],
        "core_energy_fraction": fractions["core"],
        "annular_energy_fraction": fractions["annulus"],
        "side_lobe_energy_fraction": fractions["side_lobe"],
        "energy_accounting_valid": energy_accounting_valid,
        "peak_rise_supported_by_energy_redistribution": peak_rise_supported,
        "peak_rise_status": peak_rise_status,
        "fov_reliable_input": bool(fov_reliable),
    })
    return audit


# ---------------------------------------------------------------------------
# Beam-tilt analytical validation
# ---------------------------------------------------------------------------


def validate_beam_tilt(
    tilt_x_mrad: float,
    tilt_y_mrad: float = 0.0,
    *,
    config: ComponentPlaneConfig | None = None,
) -> dict[str, Any]:
    """Validate the measured steering slope against the free-space relation
    ``dx/dz = kx/kz`` with ``kx = k0 sin(theta_x)``."""
    config = config or ComponentPlaneConfig()
    run = run_component_plane_pipeline(
        {"enable_beam_tilt": True, "beam_tilt_x_mrad": tilt_x_mrad, "beam_tilt_y_mrad": tilt_y_mrad},
        config=config,
    )
    st = run.propagated_stack
    I = np.asarray(st.intensity_zyx, dtype=float)
    x = np.asarray(st.x_um, dtype=float)
    y = np.asarray(st.y_um, dtype=float)
    z = np.asarray(st.z_um, dtype=float)

    # Expected slopes from the phase ramp.
    k = config.k_medium_rad_per_m
    kx = k * np.sin(float(tilt_x_mrad) * 1e-3)
    ky = k * np.sin(float(tilt_y_mrad) * 1e-3)
    kz = np.sqrt(max(k**2 - kx**2 - ky**2, 1e-12))
    exp_sx = float(kx / kz)
    exp_sy = float(ky / kz)

    # Measure from the robust fitted ring/core axis trajectory (NOT the
    # azimuthally-degenerate brightest pixel), over the valid z range only.
    traj = track_axis_trajectory(I, x, y, z, estimator_mode="auto")
    mx = float(traj["measured_slope_x"]); my = float(traj["measured_slope_y"])
    fit_q = float(traj["trajectory_fit_quality"])
    zc_lo, zc_hi = traj["valid_z_fit_range_um"]

    dx_um = float(st.dx_um)
    z_span = float(zc_hi - zc_lo) if np.isfinite(zc_hi) else 0.0
    px_disp = abs(float(np.hypot(mx, my)) * z_span / dx_um)
    abs_err = float(np.hypot(mx - exp_sx, my - exp_sy))
    exp_mag = max(float(np.hypot(exp_sx, exp_sy)), 1e-12)
    rel_err = abs_err / exp_mag
    zc = np.array([zc_lo, zc_hi])
    return {
        "commanded_tilt_x_mrad": float(tilt_x_mrad),
        "commanded_tilt_y_mrad": float(tilt_y_mrad),
        "expected_slope_x": exp_sx,
        "expected_slope_y": exp_sy,
        "measured_slope_x": float(mx),
        "measured_slope_y": float(my),
        "absolute_slope_error": abs_err,
        "relative_slope_error": float(rel_err),
        "absolute_error": abs_err,
        "relative_error": float(rel_err),
        "trajectory_fit_quality": fit_q,
        "fit_quality": fit_q,
        "valid_z_fit_range_um": (float(zc.min()), float(zc.max())),
        "grid_pixels_of_displacement": float(px_disp),
        "grid_resolved_displacement": float(px_disp),
        "agrees_within_tolerance": bool(rel_err <= TOL_TILT_SLOPE_REL),
        "propagation_medium": "free_space_n_1.0",
    }


# ---------------------------------------------------------------------------
# Crop / FOV convergence
# ---------------------------------------------------------------------------


def _fov_metrics(run: ComponentPlaneRun) -> dict[str, float]:
    st = run.propagated_stack
    fl = stack_to_fluence(st)
    I = np.asarray(st.intensity_zyx, float)
    x = np.asarray(st.x_um, float)
    y = np.asarray(st.y_um, float)
    sel = _peak_plane_index(I)
    ax = compute_axis_tracking(st, plane_index=sel, core_radius_um=2.0)
    est = estimate_annular_axis(I[sel], x, y)
    traj = track_axis_trajectory(I, x, y, np.asarray(st.z_um, float))
    return {
        "peak_fluence": float(np.max(fl.peak_fluence_by_z_j_cm2)),
        # robust annular axis metrics (PRIMARY)
        "ring_centre_x_um": float(est["ring_centre_x_um"]),
        "ring_centre_y_um": float(est["ring_centre_y_um"]),
        "core_centre_x_um": float(est["core_centre_x_um"]),
        "core_centre_y_um": float(est["core_centre_y_um"]),
        "axis_intercept_x_um": float(traj["axis_intercept_at_z0_x_um"]),
        "axis_intercept_y_um": float(traj["axis_intercept_at_z0_y_um"]),
        "axis_error_um": float(est["beam_axis_error_um"]),
        "steering_x_mrad": float(traj["beam_steering_angle_x_mrad"]),
        # raw brightest pixel kept for DIAGNOSTIC ONLY
        "raw_peak_x_um": float(ax["peak_x_um"]),
        "raw_peak_y_um": float(ax["peak_y_um"]),
        "captured_power_drift": float(fl.propagation_energy_drift_fraction),
        "field_of_view_margin_um": float(ax["field_of_view_margin_um"]),
        "out_of_frame_fraction": float(ax["out_of_frame_fraction"]),
    }


def fov_convergence_check(
    controls: Mapping[str, Any] | None = None,
    *,
    config: ComponentPlaneConfig | None = None,
    expand_factor: float = 1.5,
) -> dict[str, Any]:
    """Compare standard vs expanded grid/FOV and assign a reliability label.

    For annular fields the reliability is driven by the fitted ring/core centre
    and axis-trajectory convergence plus peak-fluence / captured-power
    convergence and FOV margin -- NOT by the azimuthally-degenerate raw peak
    position (which is reported diagnostically only).
    """
    config = config or ComponentPlaneConfig()
    controls = dict(controls or {})
    expanded = replace(config, grid_N=int(round(config.grid_N * expand_factor)))

    std = _fov_metrics(run_component_plane_pipeline(controls, config=config))
    exp = _fov_metrics(run_component_plane_pipeline(controls, config=expanded))

    peak_rel = abs(std["peak_fluence"] - exp["peak_fluence"]) / max(exp["peak_fluence"], 1e-30)
    ring_diff = float(np.hypot(std["ring_centre_x_um"] - exp["ring_centre_x_um"],
                               std["ring_centre_y_um"] - exp["ring_centre_y_um"]))
    core_diff = float(np.hypot(std["core_centre_x_um"] - exp["core_centre_x_um"],
                               std["core_centre_y_um"] - exp["core_centre_y_um"]))
    axis_traj_diff = float(np.hypot(std["axis_intercept_x_um"] - exp["axis_intercept_x_um"],
                                    std["axis_intercept_y_um"] - exp["axis_intercept_y_um"]))
    axis_err_diff = abs(std["axis_error_um"] - exp["axis_error_um"])
    raw_peak_diff = float(np.hypot(std["raw_peak_x_um"] - exp["raw_peak_x_um"],
                                   std["raw_peak_y_um"] - exp["raw_peak_y_um"]))
    slope_diff = abs(std["steering_x_mrad"] - exp["steering_x_mrad"])
    drift_diff = abs(std["captured_power_drift"] - exp["captured_power_drift"])

    dx_um = float(config.dx_um)
    # Reliability is driven by ring/core/trajectory + peak fluence + power + FOV,
    # never by the raw annular peak position.
    if (std["out_of_frame_fraction"] > 0.02 or std["field_of_view_margin_um"] < 0.0
            or peak_rel > 0.25 or ring_diff > 5.0 * dx_um):
        reliability = "invalid_out_of_frame"
    elif (peak_rel > TOL_FOV_PEAK_REL or std["out_of_frame_fraction"] > 0.005
          or ring_diff > 1.5 * dx_um or axis_traj_diff > 2.0 * dx_um):
        reliability = "caution_crop_limited"
    else:
        reliability = "numerically_reliable"

    return {
        "standard": std,
        "expanded": exp,
        "standard_grid_N": int(config.grid_N),
        "expanded_grid_N": int(expanded.grid_N),
        "peak_fluence_rel_diff": float(peak_rel),
        "ring_centre_difference_um": ring_diff,
        "core_centre_difference_um": core_diff,
        "axis_trajectory_difference_um": axis_traj_diff,
        "axis_error_difference_um": axis_err_diff,
        "raw_peak_position_difference_um": raw_peak_diff,  # diagnostic only
        "raw_peak_status": "not_a_primary_axis_metric_for_annular_fields",
        "axis_trajectory_slope_diff_mrad": float(slope_diff),
        "captured_power_drift_diff": float(drift_diff),
        "field_of_view_margin_um": std["field_of_view_margin_um"],
        "out_of_frame_fraction": std["out_of_frame_fraction"],
        "metric_convergence_status": reliability,
        "propagation_medium": "free_space_n_1.0",
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _complex_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a).ravel()
    bv = np.asarray(b).ravel()
    na = float(np.sqrt(np.sum(np.abs(av) ** 2)))
    nb = float(np.sqrt(np.sum(np.abs(bv) ** 2)))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.clip(np.abs(np.vdot(av, bv)) / (na * nb), 0.0, 1.0))


def _normalise_like(src: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Scale ``src`` so its sum matches ``like`` (for fluence-similarity comparison)."""
    s = float(np.sum(src))
    if s <= 0:
        return src
    return src * (float(np.sum(like)) / s)
