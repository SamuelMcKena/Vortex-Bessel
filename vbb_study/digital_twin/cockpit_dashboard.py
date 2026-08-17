"""Integrated Stage 8C.1 cockpit dashboard and field diagnostics.

The helpers here choose safe display planes, expose central-ROI and target-depth
fluence metrics, and render the beam-to-write cockpit. They do not implement
material response and do not change optical propagation physics.
"""

from __future__ import annotations

import textwrap
from math import ceil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from vbb_study.digital_twin.energy_accounting import EnergyLedger
from vbb_study.digital_twin.exposure_bookkeeping import line_exposure_summary
from vbb_study.digital_twin.field_coupling import OpticalFieldStack
from vbb_study.digital_twin.field_figures import CaveatsRequiredError
from vbb_study.digital_twin.field_fluence import (
    FluenceStackResult,
    peak_intensity_from_fluence_result,
)

STAGE = "stage8c2_integrated_cockpit_dashboard"
MODEL_STATUS = "diagnostic_preview"
FIGURE_STATUS = "diagnostic_allowed"
FINAL_EXPORT_ALLOWED = False

# ---------------------------------------------------------------------------
# Visual design tokens (Stage 8C.2 polish)
# ---------------------------------------------------------------------------

STATUS_DARK = {
    "pass": "#1b5e20",
    "caution": "#e65100",
    "fail": "#b71c1c",
    "missing": "#455a64",
    "diagnostic_only": "#0d47a1",
    "disabled_future": "#4a148c",
    "info": "#37474f",
}
STATUS_LIGHT = {
    "pass": "#e8f5e9",
    "caution": "#fff3e0",
    "fail": "#ffebee",
    "missing": "#eceff1",
    "diagnostic_only": "#e3f2fd",
    "disabled_future": "#f3e5f5",
    "info": "#eceff1",
}
STATUS_SEVERITY = {
    "pass": 0,
    "diagnostic_only": 0,
    "disabled_future": 0,
    "info": 0,
    "missing": 1,
    "caution": 2,
    "fail": 3,
}
STATUS_WORD = {
    "pass": "PASS",
    "caution": "WARN",
    "fail": "FAIL",
    "missing": "MISS",
    "diagnostic_only": "DIAG",
    "disabled_future": "OFF",
    "info": "INFO",
}
BADGE_WORD = {"pass": "PASS", "caution": "CAUTION", "fail": "FAIL"}

_SHORT_STAGE_LABELS = {
    "laser_source": "Laser",
    "pre_slm_beam_conditioning": "Pre-SLM",
    "telescope_or_beam_expander": "Telesc.",
    "slm1_phase": "SLM1",
    "slm2_phase_or_axicon": "SLM2/Ax",
    "first_order_filter": "1st-ord",
    "relay_optics": "Relay",
    "objective_and_pupil": "Obj/Pup",
    "sample_interface": "Sample",
    "in_sample_propagation": "Propag.",
    "field_to_fluence": "Fluence",
    "exposure_bookkeeping": "Expose",
    "future_material_response_disabled": "Material",
}

# Colormaps used consistently across the dashboard.
_CMAP_INTENSITY = "inferno"
_CMAP_FLUENCE = "viridis"


def choose_display_plane(
    z_um: np.ndarray,
    *,
    selected_z_mode: str = "target_depth",
    target_depth_um: float = 0.0,
    custom_z_um: float | None = None,
    optical_peak_z_um: float | None = None,
    global_peak_near_boundary: bool = False,
) -> dict[str, Any]:
    """Choose the cockpit display plane without promoting crop-edge artifacts."""
    z = np.asarray(z_um, dtype=float)
    if z.ndim != 1 or z.size == 0:
        raise ValueError("z_um must be a non-empty 1D coordinate array.")

    mode = str(selected_z_mode or "target_depth")
    if mode == "optical_peak":
        if global_peak_near_boundary:
            idx = _nearest_index(z, target_depth_um)
            reason = "target_depth_safe: global optical peak is near crop boundary"
        else:
            target = target_depth_um if optical_peak_z_um is None else float(optical_peak_z_um)
            idx = _nearest_index(z, target)
            reason = "optical_peak"
    elif mode == "sample_surface":
        idx = _nearest_index(z, 0.0)
        reason = "sample_surface"
    elif mode == "custom":
        if custom_z_um is None:
            raise ValueError("custom_z_um is required when selected_z_mode='custom'.")
        idx = _nearest_index(z, float(custom_z_um))
        reason = "custom"
    else:
        idx = _nearest_index(z, float(target_depth_um))
        reason = "target_depth"

    return {
        "selected_plane_index": int(idx),
        "selected_plane_z_um": float(z[idx]),
        "selected_plane_reason": reason,
        "requested_mode": mode,
    }


def compute_peak_location_diagnostics(
    stack: OpticalFieldStack,
    fluence_result: FluenceStackResult,
    *,
    target_depth_um: float = 0.0,
    central_roi_half_width_um: float = 10.0,
    selected_z_mode: str = "target_depth",
    custom_z_um: float | None = None,
    boundary_margin_px: int = 1,
    pulse_duration_fs: float | None = None,
) -> dict[str, Any]:
    """Compute global, central-ROI, target-depth, and surface fluence diagnostics."""
    if not isinstance(stack, OpticalFieldStack):
        raise TypeError(f"stack must be OpticalFieldStack; got {type(stack).__name__}.")
    if not isinstance(fluence_result, FluenceStackResult):
        raise TypeError(
            f"fluence_result must be FluenceStackResult; got {type(fluence_result).__name__}."
        )

    F = np.asarray(fluence_result.fluence_zyx_j_cm2, dtype=float)
    if F.shape != stack.intensity_zyx.shape:
        raise ValueError("fluence stack shape does not match optical stack shape.")
    if not np.all(np.isfinite(F)):
        raise ValueError("fluence stack contains non-finite values.")

    nz, ny, nx = F.shape
    z = np.asarray(stack.z_um, dtype=float)
    x = np.asarray(stack.x_um, dtype=float)
    y = np.asarray(stack.y_um, dtype=float)

    global_flat = int(np.argmax(F))
    gz_i, gy_i, gx_i = (int(v) for v in np.unravel_index(global_flat, F.shape))
    global_peak_value = float(F[gz_i, gy_i, gx_i])
    distance_px = int(min(gy_i, ny - 1 - gy_i, gx_i, nx - 1 - gx_i))
    near_boundary = bool(distance_px <= int(boundary_margin_px))

    roi_x_idx = np.flatnonzero(np.abs(x) <= float(central_roi_half_width_um))
    roi_y_idx = np.flatnonzero(np.abs(y) <= float(central_roi_half_width_um))
    if roi_x_idx.size == 0:
        roi_x_idx = np.array([_nearest_index(x, 0.0)])
    if roi_y_idx.size == 0:
        roi_y_idx = np.array([_nearest_index(y, 0.0)])
    roi = F[:, roi_y_idx[:, None], roi_x_idx]
    roi_flat = int(np.argmax(roi))
    rz_i, ry_local, rx_local = (int(v) for v in np.unravel_index(roi_flat, roi.shape))
    ry_i = int(roi_y_idx[ry_local])
    rx_i = int(roi_x_idx[rx_local])

    central_roi_peak_by_z = np.max(roi, axis=(1, 2))
    target_i = _nearest_index(z, float(target_depth_um))
    surface_i = _nearest_index(z, 0.0)

    selected = choose_display_plane(
        z,
        selected_z_mode=selected_z_mode,
        target_depth_um=float(target_depth_um),
        custom_z_um=custom_z_um,
        optical_peak_z_um=float(z[gz_i]),
        global_peak_near_boundary=near_boundary,
    )
    selected_i = int(selected["selected_plane_index"])

    drift = float(fluence_result.propagation_energy_drift_fraction)
    warning_messages: list[str] = []
    if near_boundary:
        warning_messages.append("global optical peak is near the crop boundary")
    if np.isfinite(drift) and drift > 0.20:
        warning_messages.append(f"raw captured-power drift is high ({drift:.1%})")
    warning_level = "caution" if warning_messages else "pass"

    peak_intensity = None
    if pulse_duration_fs is not None:
        peak_intensity = peak_intensity_from_fluence_result(fluence_result, pulse_duration_fs)

    return {
        "global_peak_value": global_peak_value,
        "global_peak_x_um": float(x[gx_i]),
        "global_peak_y_um": float(y[gy_i]),
        "global_peak_z_um": float(z[gz_i]),
        "global_peak_index_zyx": (gz_i, gy_i, gx_i),
        "global_peak_distance_to_boundary_px": distance_px,
        "global_peak_near_boundary": near_boundary,
        "central_roi_peak_value": float(F[rz_i, ry_i, rx_i]),
        "central_roi_peak_x_um": float(x[rx_i]),
        "central_roi_peak_y_um": float(y[ry_i]),
        "central_roi_peak_z_um": float(z[rz_i]),
        "central_roi_peak_index_zyx": (rz_i, ry_i, rx_i),
        "central_roi_peak_by_z_j_cm2": central_roi_peak_by_z,
        "target_depth_peak_value": float(np.max(F[target_i])),
        "target_depth_z_um": float(z[target_i]),
        "target_depth_index": int(target_i),
        "sample_surface_peak_value": float(np.max(F[surface_i])),
        "sample_surface_z_um": float(z[surface_i]),
        "sample_surface_index": int(surface_i),
        "selected_plane_z_um": selected["selected_plane_z_um"],
        "selected_plane_index": selected_i,
        "selected_plane_reason": selected["selected_plane_reason"],
        "selected_plane_peak_value": float(np.max(F[selected_i])),
        "captured_power_drift_fraction": drift,
        "raw_captured_power_fraction_by_z": fluence_result.raw_captured_power_fraction_by_z,
        "peak_intensity_w_cm2": peak_intensity,
        "warning_level": warning_level,
        "warning_messages": warning_messages,
    }


def build_warning_flags(
    *,
    energy_ledger: EnergyLedger | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    first_order_geometry_valid: bool | None = None,
    pupil_clipping_fraction: float | None = None,
    power_limit_exceeded: bool | None = None,
    pulse_overlap_fraction: float | None = None,
    captured_power_drift_fraction: float | None = None,
    perturbation_result: Any | None = None,
) -> dict[str, dict[str, str]]:
    """Build cockpit warning flags for hardware-feasibility panels."""
    flags = {
        "captured_power_drift": {"status": "pass", "message": "raw captured-power drift acceptable"},
        "crop_edge_peak": {"status": "pass", "message": "global peak is not near crop edge"},
        "first_order_geometry": {"status": "pass", "message": "first-order geometry valid"},
        "pupil_clipping": {"status": "pass", "message": "pupil clipping not detected"},
        "power_limit": {"status": "pass", "message": "average power within configured limit"},
        "pulse_overlap": {"status": "pass", "message": "pulse spacing supports continuous scan bookkeeping"},
    }
    if perturbation_result is not None:
        uncoupled = getattr(perturbation_result, "uncoupled_enabled_controls", [])
        active = getattr(perturbation_result, "active_controls", [])
        if uncoupled:
            flags["active_lab_realism"] = {
                "status": "caution",
                "message": f"{len(uncoupled)} enabled lab-realism control(s) are report-only/future",
            }
        elif active:
            flags["active_lab_realism"] = {
                "status": "pass",
                "message": f"{len(active)} active perturbation control(s) coupled downstream",
            }

    diag = dict(diagnostics or {})
    drift = captured_power_drift_fraction
    if drift is None:
        drift = diag.get("captured_power_drift_fraction")
    if drift is not None and np.isfinite(float(drift)) and float(drift) > 0.20:
        flags["captured_power_drift"] = {
            "status": "caution",
            "message": f"raw captured-power drift {float(drift):.1%}",
        }

    if bool(diag.get("global_peak_near_boundary", False)):
        flags["crop_edge_peak"] = {
            "status": "caution",
            "message": "global peak is near crop boundary; do not use as headline plane",
        }

    if first_order_geometry_valid is False:
        flags["first_order_geometry"] = {
            "status": "caution",
            "message": "first-order geometry is not currently valid",
        }

    if pupil_clipping_fraction is not None and np.isfinite(float(pupil_clipping_fraction)):
        if float(pupil_clipping_fraction) > 0:
            flags["pupil_clipping"] = {
                "status": "caution",
                "message": f"pupil clipping fraction {float(pupil_clipping_fraction):.3g}",
            }

    if power_limit_exceeded is None and energy_ledger is not None:
        power_limit_exceeded = any("exceeds limit" in str(w).lower() for w in energy_ledger.ledger_warnings)
    if power_limit_exceeded:
        flags["power_limit"] = {"status": "fail", "message": "average power exceeds configured limit"}

    if exposure_summary is not None:
        pulse_overlap_fraction = exposure_summary.get("overlap_fraction", pulse_overlap_fraction)
        if any("gap" in str(w).lower() or "dotted" in str(w).lower() for w in exposure_summary.get("warnings", [])):
            flags["pulse_overlap"] = {
                "status": "caution",
                "message": "pulse spacing warning from exposure bookkeeping",
            }
    if pulse_overlap_fraction is not None and np.isfinite(float(pulse_overlap_fraction)):
        if float(pulse_overlap_fraction) < 0:
            flags["pulse_overlap"] = {"status": "caution", "message": "pulse spacing exceeds effective diameter"}

    return flags


def build_cockpit_summary(
    *,
    controls: Mapping[str, Any],
    energy_ledger: EnergyLedger | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    lab_report: Any | None = None,
) -> dict[str, Any]:
    """Collect cockpit inputs and computed outputs into one summary mapping."""
    if exposure_summary is None and energy_ledger is not None:
        exposure_summary = line_exposure_summary(
            energy_ledger.energy_at_sample_uJ,
            controls.get("repetition_rate_Hz", 25_000.0),
            controls.get("scan_speed_mm_s", 1.0),
            controls.get("line_length_um", 500.0),
            controls.get("effective_diameter_um", 3.0),
        )
    return {
        "controls": dict(controls),
        "energy_ledger": energy_ledger,
        "exposure_summary": dict(exposure_summary or {}),
        "diagnostics": dict(diagnostics or {}),
        "lab_report": lab_report,
        "warning_flags": build_warning_flags(
            energy_ledger=energy_ledger,
            exposure_summary=exposure_summary,
            diagnostics=diagnostics,
        ),
        "model_status": MODEL_STATUS,
        "final_export_allowed": FINAL_EXPORT_ALLOWED,
    }


def plot_integrated_cockpit_dashboard(
    stack: OpticalFieldStack,
    fluence_result: FluenceStackResult,
    *,
    controls: Mapping[str, Any] | None = None,
    energy_ledger: EnergyLedger | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    lab_report: Any | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    show_caveats: bool = True,
    dpi: int = 180,
    display_scaling: str = "percentile",
    display_percentile_clip: tuple[float, float] = (0.5, 99.5),
    title: str = "Stage 8C.2 Integrated Beam-to-Write Cockpit",
    perturbation_result: Any | None = None,
    degradation_metrics: Mapping[str, Any] | None = None,
) -> "matplotlib.figure.Figure":
    """Render the integrated optical/energy/exposure cockpit dashboard.

    Stage 8C.2 polish: GridSpec hierarchy with a header band + PASS/CAUTION/FAIL
    status badge, a stage-by-stage beam-path strip, high-visibility warning
    cards, readable summary/energy/exposure cards, annotated XY/ROI panels, a
    dominant annotated XZ propagation panel, peak-vs-z and raw captured-power
    drift plots, and an interpretation/claim-boundary card.
    """
    controls = dict(controls or {})
    if output_path is not None and not show_caveats:
        raise CaveatsRequiredError(
            "Refusing to save Stage 8C.1 dashboard with show_caveats=False."
        )
    if diagnostics is None:
        diagnostics = compute_peak_location_diagnostics(
            stack,
            fluence_result,
            target_depth_um=float(controls.get("focus_depth_um", 0.0)),
            central_roi_half_width_um=float(controls.get("central_roi_half_width_um", 10.0)),
            selected_z_mode=str(controls.get("selected_z_mode", controls.get("selected_z_um", "target_depth"))),
            custom_z_um=controls.get("custom_z_um"),
            pulse_duration_fs=controls.get("pulse_duration_fs"),
        )
    diagnostics = dict(diagnostics)
    if exposure_summary is None and energy_ledger is not None:
        exposure_summary = line_exposure_summary(
            energy_ledger.energy_at_sample_uJ,
            float(controls.get("repetition_rate_Hz", 25_000.0)),
            float(controls.get("scan_speed_mm_s", 1.0)),
            float(controls.get("line_length_um", 500.0)),
            float(controls.get("effective_diameter_um", 3.0)),
        )
    exposure_summary = dict(exposure_summary or {})
    warning_flags = build_warning_flags(
        energy_ledger=energy_ledger,
        exposure_summary=exposure_summary,
        diagnostics=diagnostics,
        perturbation_result=perturbation_result,
    )

    overall_status = compute_overall_status(
        warning_flags, diagnostics=diagnostics, lab_report=lab_report
    )
    strip = build_beam_path_strip(lab_report)

    selected_idx = int(diagnostics["selected_plane_index"])
    z = np.asarray(stack.z_um, dtype=float)
    x = np.asarray(stack.x_um, dtype=float)
    y = np.asarray(stack.y_um, dtype=float)
    extent_xy = _extent_from_coords(x, y)
    F = fluence_result.fluence_zyx_j_cm2

    surface_z = 0.0
    target_z = float(controls.get("focus_depth_um", diagnostics.get("target_depth_z_um", 0.0)))
    selected_z = float(diagnostics["selected_plane_z_um"])
    global_peak_z = float(diagnostics["global_peak_z_um"])
    roi_peak_z = float(diagnostics.get("central_roi_peak_z_um", selected_z))

    fig = plt.figure(figsize=(20.0, 23.5))
    gs = fig.add_gridspec(
        8, 12,
        height_ratios=[0.62, 0.64, 0.78, 1.55, 1.95, 1.5, 1.5, 1.5],
        hspace=0.55, wspace=0.65,
        left=0.045, right=0.972, top=0.986, bottom=0.022,
    )

    # --- A. header band + status badge ---
    _header_band(fig.add_subplot(gs[0, :]), controls, energy_ledger, diagnostics,
                 overall_status, title)

    # --- B. stage-by-stage beam-path strip ---
    _beam_path_strip_panel(fig.add_subplot(gs[1, :]), strip)

    # --- C. high-visibility warning cards ---
    _warning_cards_panel(fig.add_subplot(gs[2, :]), warning_flags)

    # --- D/E. experiment / energy ledger / exposure ---
    _card(fig.add_subplot(gs[3, 0:4]), "Experiment request",
          _experiment_card_lines(controls, diagnostics), status="diagnostic_only")
    _ledger_bars(fig.add_subplot(gs[3, 4:8]), energy_ledger)
    _card(fig.add_subplot(gs[3, 8:12]), "Exposure bookkeeping",
          _exposure_card_lines(controls, exposure_summary),
          status=_exposure_status(exposure_summary))

    # --- F. optical / fluence panels ---
    ax_int = fig.add_subplot(gs[4, 0:4])
    im = ax_int.imshow(
        _display_map(stack.intensity_zyx[selected_idx], display_scaling, display_percentile_clip),
        origin="lower", extent=extent_xy, cmap=_CMAP_INTENSITY, aspect="equal",
    )
    ax_int.set_title(f"XY optical intensity @ selected z = {selected_z:.3g} um",
                     fontsize=12, fontweight="bold")
    ax_int.set_xlabel("x (um)"); ax_int.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax_int, fraction=0.046, pad=0.04, label="intensity (a.u.)")
    _annotate(ax_int, [
        f"route: {controls.get('generation_method', 'holographic')}  ell={controls.get('ell', 'na')}",
        f"plane: {diagnostics.get('selected_plane_reason', 'na')}",
    ])

    ax_flu = fig.add_subplot(gs[4, 4:8])
    im = ax_flu.imshow(
        _display_map(F[selected_idx], display_scaling, display_percentile_clip),
        origin="lower", extent=extent_xy, cmap=_CMAP_FLUENCE, aspect="equal",
    )
    ax_flu.set_title("XY optical fluence @ selected z", fontsize=12, fontweight="bold")
    ax_flu.set_xlabel("x (um)"); ax_flu.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax_flu, fraction=0.046, pad=0.04, label="J/cm^2")
    _annotate(ax_flu, [
        f"selected peak: {_fmt(diagnostics.get('selected_plane_peak_value'))} J/cm^2",
        f"E@sample: {_energy_at_sample(energy_ledger)} uJ",
    ])

    ax_roi = fig.add_subplot(gs[4, 8:12])
    roi_half = float(controls.get("central_roi_half_width_um", 10.0))
    xi = np.flatnonzero(np.abs(x) <= roi_half)
    yi = np.flatnonzero(np.abs(y) <= roi_half)
    if xi.size == 0:
        xi = np.array([_nearest_index(x, 0.0)])
    if yi.size == 0:
        yi = np.array([_nearest_index(y, 0.0)])
    roi_f = F[selected_idx][np.ix_(yi, xi)]
    im = ax_roi.imshow(
        _display_map(roi_f, display_scaling, display_percentile_clip),
        origin="lower", extent=_extent_from_coords(x[xi], y[yi]), cmap=_CMAP_FLUENCE, aspect="equal",
    )
    ax_roi.set_title(f"Central ROI fluence zoom (+/-{roi_half:.3g} um)", fontsize=12, fontweight="bold")
    ax_roi.set_xlabel("x (um)"); ax_roi.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax_roi, fraction=0.046, pad=0.04, label="J/cm^2")
    _annotate(ax_roi, [
        f"ROI peak: {_fmt(diagnostics.get('central_roi_peak_value'))} J/cm^2",
        f"at z = {roi_peak_z:.3g} um",
    ])

    # --- G. dominant annotated XZ propagation panel ---
    ax_xz = fig.add_subplot(gs[5:7, 0:8])
    yc = _nearest_index(y, 0.0)
    xz = F[:, yc, :].T
    im = ax_xz.imshow(
        _display_map(xz, display_scaling, display_percentile_clip),
        origin="lower",
        extent=(float(np.min(z)), float(np.max(z)), float(np.min(x)), float(np.max(x))),
        cmap=_CMAP_FLUENCE, aspect="auto",
    )
    ax_xz.set_title("XZ optical fluence along propagation (y = 0 slice)",
                    fontsize=14, fontweight="bold")
    ax_xz.set_xlabel("z (um)   [propagation axis; sample surface at z = 0]", fontsize=12)
    ax_xz.set_ylabel("x (um)", fontsize=12)
    for value, color, label, style in [
        (surface_z, "#29b6f6", "surface z=0", "-"),
        (target_z, "#ffb300", "target depth", "--"),
        (selected_z, "#00e676", "selected z", "-."),
        (global_peak_z, "#ff1744", "global peak z", ":"),
    ]:
        ax_xz.axvline(value, color=color, lw=2.2, ls=style, label=label)
    ax_xz.legend(fontsize=10, loc="upper right", framealpha=0.85)
    fig.colorbar(im, ax=ax_xz, fraction=0.03, pad=0.02, label="J/cm^2")
    if diagnostics.get("global_peak_near_boundary"):
        ax_xz.text(
            0.5, 0.045,
            "CAUTION: global peak near crop boundary - not a trustworthy headline plane",
            transform=ax_xz.transAxes, ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round", facecolor="#b71c1c", alpha=0.92, edgecolor="none"),
        )

    # --- peak-vs-z and raw captured-power drift ---
    ax_peak = fig.add_subplot(gs[5, 8:12])
    ax_peak.plot(z, fluence_result.peak_fluence_by_z_j_cm2, color="#1565c0", lw=2.2, label="global peak")
    if "central_roi_peak_by_z_j_cm2" in diagnostics:
        ax_peak.plot(z, np.asarray(diagnostics["central_roi_peak_by_z_j_cm2"], float),
                     color="#6a1b9a", ls="--", lw=2.0, label="central ROI peak")
    ax_peak.axvline(selected_z, color="#00c853", lw=1.8, label="selected")
    ax_peak.axvline(target_z, color="#ffb300", lw=1.8, ls="--", label="target")
    ax_peak.set_title("Peak fluence vs z", fontsize=12, fontweight="bold")
    ax_peak.set_xlabel("z (um)"); ax_peak.set_ylabel("J/cm^2")
    ax_peak.legend(fontsize=8, loc="best")
    ax_peak.grid(alpha=0.25)

    ax_drift = fig.add_subplot(gs[6, 8:12])
    ax_drift.plot(z, fluence_result.raw_captured_power_fraction_by_z, color="#8e24aa", lw=2.2)
    ax_drift.fill_between(z, fluence_result.raw_captured_power_fraction_by_z, color="#8e24aa", alpha=0.12)
    ax_drift.set_title("Raw captured-power fraction vs z", fontsize=12, fontweight="bold")
    ax_drift.set_xlabel("z (um)"); ax_drift.set_ylabel("captured fraction")
    ax_drift.grid(alpha=0.25)
    drift = float(diagnostics.get("captured_power_drift_fraction", float("nan")))
    drift_status = "caution" if (np.isfinite(drift) and drift > 0.20) else "pass"
    ax_drift.text(
        0.5, 0.93, (f"drift = {drift:.1%}" if np.isfinite(drift) else "drift = n/a"),
        transform=ax_drift.transAxes, ha="center", va="top", fontsize=11, fontweight="bold",
        color=STATUS_DARK[drift_status],
        bbox=dict(boxstyle="round", facecolor=STATUS_LIGHT[drift_status],
                  edgecolor=STATUS_DARK[drift_status]),
    )

    # --- interpretation / claim boundary + future-disabled ---
    _card(fig.add_subplot(gs[7, 0:8]), "Interpretation & claim boundary",
          _claim_boundary_lines(diagnostics, perturbation_result, degradation_metrics),
          status=overall_status, body_fs=10.5)
    _card(fig.add_subplot(gs[7, 8:12]), "Future physics - disabled",
          _future_disabled_text().split("\n"), status="disabled_future", body_fs=9.5)

    meta = {
        "stage": STAGE,
        "figure_status": FIGURE_STATUS,
        "model_status": MODEL_STATUS,
        "final_export_allowed": False,
        "selected_plane_z_um": diagnostics["selected_plane_z_um"],
        "warning_level": diagnostics.get("warning_level", "pass"),
        "overall_status": overall_status,
        "stage8c3_active_controls": len(getattr(perturbation_result, "active_controls", []) or []),
        "stage8c3_uncoupled_controls": len(getattr(perturbation_result, "uncoupled_enabled_controls", []) or []),
    }
    fig.stage8c1_metadata = meta  # type: ignore[attr-defined]
    fig.stage8c2_metadata = meta  # type: ignore[attr-defined]

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches="tight",
            metadata={
                "Title": title,
                "stage": STAGE,
                "figure_status": FIGURE_STATUS,
                "model_status": MODEL_STATUS,
                "final_export_allowed": "False",
                "overall_status": overall_status,
                "Description": "Stage 8C.2 optical/fluence cockpit dashboard; diagnostic only; no material response.",
            },
        )
    return fig


def make_interpretation_text(
    *,
    controls: Mapping[str, Any] | None = None,
    energy_ledger: EnergyLedger | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    lab_report: Any | None = None,
) -> str:
    """Return plain-language cockpit interpretation with the Stage 8C.1 boundary."""
    controls = dict(controls or {})
    exposure_summary = dict(exposure_summary or {})
    diagnostics = dict(diagnostics or {})

    input_energy = controls.get("pulse_energy_before_optics_uJ", "not available")
    sample_energy = "not available"
    avg_power = "not available"
    limit_state = "not available"
    if energy_ledger is not None:
        sample_energy = f"{energy_ledger.energy_at_sample_uJ:.4g}"
        avg_power = f"{energy_ledger.average_power_at_sample_W:.4g}"
        limit_state = "above" if any("exceeds limit" in str(w).lower() for w in energy_ledger.ledger_warnings) else "below"

    first_order = "valid" if float(controls.get("selected_first_order_fraction", 0.0)) > 0 else "not valid"
    route = "valid" if controls.get("generation_method", "holographic") in {"holographic", "physical"} else "not valid"
    pupil = "not available from current engine"
    pulse_spacing = exposure_summary.get("pulse_spacing_um", "not available")
    pulses_per_spot = exposure_summary.get("pulses_per_spot", "not available")
    selected_f = diagnostics.get("selected_plane_peak_value", diagnostics.get("global_peak_value", "not available"))
    central_f = diagnostics.get("central_roi_peak_value", "not available")
    target_f = diagnostics.get("target_depth_peak_value", "not available")
    peak_i = diagnostics.get("peak_intensity_w_cm2", "not available")
    edge = "is" if diagnostics.get("global_peak_near_boundary") else "is not"
    drift = diagnostics.get("captured_power_drift_fraction", "not available")
    if isinstance(drift, (float, int)) and np.isfinite(float(drift)):
        drift_text = f"{100.0 * float(drift):.2f}%"
    else:
        drift_text = str(drift)
    lab_state = _lab_status(lab_report)

    return "\n".join(
        [
            f"Energy at sample is {sample_energy} uJ from {input_energy} uJ before optics.",
            f"Average power is {limit_state} the configured limit; sample average power is {avg_power} W.",
            f"The optical route is {route}.",
            f"The first-order geometry is {first_order}.",
            f"Pupil fill/clipping status is {pupil}.",
            f"Pulse spacing is {_fmt(pulse_spacing)} um, giving {_fmt(pulses_per_spot)} pulses per spot.",
            f"For the selected plane, peak optical fluence is {_fmt(selected_f)} J/cm^2.",
            f"Central ROI peak is {_fmt(central_f)} J/cm^2.",
            f"Target-depth peak is {_fmt(target_f)} J/cm^2.",
            f"Peak intensity estimate is {_fmt(peak_i)} W/cm^2, assuming no nonlinear reshaping.",
            f"The global field peak {edge} near the crop boundary.",
            f"Captured-power drift is {drift_text}.",
            f"Lab realism status is {lab_state}.",
            "This output is optical fluence only; no material modification is predicted.",
        ]
    )


def _nearest_index(coords: np.ndarray, value: float) -> int:
    arr = np.asarray(coords, dtype=float)
    return int(np.argmin(np.abs(arr - float(value))))


def _extent_from_coords(x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    dx = float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    dy = float(np.mean(np.abs(np.diff(y)))) if y.size > 1 else 1.0
    return (
        float(np.min(x) - 0.5 * dx),
        float(np.max(x) + 0.5 * dx),
        float(np.min(y) - 0.5 * dy),
        float(np.max(y) + 0.5 * dy),
    )


def _display_map(arr: np.ndarray, scaling: str, percentile_clip: tuple[float, float]) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if scaling == "log":
        floor = np.nanmax(data) * 1e-12 if np.nanmax(data) > 0 else 1e-12
        return np.log10(np.maximum(data, floor))
    if scaling == "percentile":
        lo, hi = np.nanpercentile(data, percentile_clip)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return np.clip(data, lo, hi)
    return data


def _plot_energy_ledger(ax: Any, ledger: EnergyLedger | None) -> None:
    if ledger is None:
        _text_panel(ax, "not available from current engine", "Energy Ledger")
        return
    labels = ["input"] + [row.component_name[:12] for row in ledger.rows]
    first_energy = ledger.rows[0].energy_in_uJ if ledger.rows else ledger.energy_at_sample_uJ
    values = [first_energy] + [row.energy_out_uJ for row in ledger.rows]
    ax.bar(np.arange(len(values)), values, color="#2374ab")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("pulse energy (uJ)")
    ax.set_title("Energy Ledger")
    ax.axhline(ledger.energy_at_sample_uJ, color="#d62728", ls=":", lw=1, label="sample")
    ax.legend(fontsize=7)


def _text_panel(ax: Any, text: str, title: str) -> None:
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=8.5, wrap=True)


def _experiment_summary_text(controls: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"Laser: {controls.get('wavelength_nm', 'na')} nm, {controls.get('pulse_duration_fs', 'na')} fs, {controls.get('repetition_rate_Hz', 'na')} Hz",
            f"Pulse energy before optics: {controls.get('pulse_energy_before_optics_uJ', 'na')} uJ",
            f"Route: {controls.get('generation_method', 'holographic')} VBB, ell={controls.get('ell', 'na')}",
            f"Sample: {controls.get('material_name', 'na')}, n={controls.get('refractive_index', 'na')}",
            f"Focus target: {controls.get('focus_depth_um', 'na')} um",
            f"Selected plane: {diagnostics.get('selected_plane_z_um', 'na')} um",
            f"Reason: {diagnostics.get('selected_plane_reason', 'na')}",
            "Claim: optical fluence only",
        ]
    )


def _exposure_text(controls: Mapping[str, Any], exposure: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"writing_mode: {controls.get('writing_mode', 'na')}",
            f"scan axis: {controls.get('scan_axis', 'na')}",
            f"scan speed: {controls.get('scan_speed_mm_s', 'na')} mm/s",
            f"pulse spacing: {_fmt(exposure.get('pulse_spacing_um', 'na'))} um",
            f"pulses/spot: {_fmt(exposure.get('pulses_per_spot', 'na'))}",
            f"line time: {_fmt(exposure.get('line_duration_s', 'na'))} s",
            f"total pulses: {exposure.get('total_pulses_on_line', 'na')}",
            "Status: exposure bookkeeping only",
        ]
    )


def _warning_flags_text(flags: Mapping[str, Mapping[str, str]]) -> str:
    lines = []
    for name, info in flags.items():
        mark = {"pass": "OK", "caution": "WARN", "fail": "FAIL"}.get(info.get("status"), "INFO")
        lines.append(f"{mark:4s} {name}: {info.get('message', '')}")
    return "\n".join(lines)


def _diagnostic_warning_text(diagnostics: Mapping[str, Any], show_caveats: bool) -> str:
    lines = [
        f"Status: {diagnostics.get('warning_level', 'pass')}",
        f"global_peak_near_boundary: {diagnostics.get('global_peak_near_boundary')}",
        f"global_peak_distance_to_boundary_px: {diagnostics.get('global_peak_distance_to_boundary_px')}",
        f"selected_plane_reason: {diagnostics.get('selected_plane_reason')}",
        f"captured_power_drift: {_fmt(diagnostics.get('captured_power_drift_fraction'))}",
    ]
    for msg in diagnostics.get("warning_messages", []) or []:
        lines.append(f"warning: {msg}")
    if show_caveats:
        lines.extend(
            [
                "",
                "Plain meaning:",
                "Optical fluence is shown for inspection.",
                "It is not absorbed energy, material change, ablation,",
                "a written feature, or a calibrated prediction.",
            ]
        )
    return "\n".join(lines)


def _lab_report_text(lab_report: Any | None) -> str:
    if lab_report is None:
        return "not available from current engine"
    rows = lab_report.to_rows() if hasattr(lab_report, "to_rows") else []
    lines = []
    for row in rows[:10]:
        lines.append(f"{row['status_level'][:12]:12s} {row['stage_name']}")
    if len(rows) > 10:
        lines.append(f"... +{len(rows) - 10} stages")
    return "\n".join(lines)


def _future_disabled_text() -> str:
    return "\n".join(
        [
            "Material response: disabled until calibration",
            "Dose accumulation: Stage 8E",
            "Threshold maps: future proxy/calibration stage",
            "Microscope proxy: later",
            "Waveguide prediction: later, requires dn calibration",
            "Surface ablation: later, separate model",
            "Nonlinear propagation: later, requires material constants",
            "Thermal accumulation: later, requires thermal model/constants",
        ]
    )


def _lab_status(lab_report: Any | None) -> str:
    if lab_report is None or not hasattr(lab_report, "stages"):
        return "not available"
    levels = [stage.status_level for stage in lab_report.stages]
    for level in ("fail", "caution", "missing", "diagnostic_only", "disabled_future", "pass"):
        if level in levels:
            return level
    return "not available"


def _fmt(value: Any) -> str:
    if isinstance(value, (float, int)):
        if np.isfinite(float(value)):
            return f"{float(value):.4g}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (float, int)) and np.isfinite(float(value)):
        return f"{100.0 * float(value):.1f}%"
    return str(value)


def _humanize(name: str) -> str:
    return str(name).replace("_", " ")


def _energy_at_sample(energy_ledger: EnergyLedger | None) -> str:
    if energy_ledger is None:
        return "na"
    return f"{energy_ledger.energy_at_sample_uJ:.4g}"


# ---------------------------------------------------------------------------
# Stage 8C.2 public helpers
# ---------------------------------------------------------------------------


def compute_overall_status(
    warning_flags: Mapping[str, Mapping[str, str]] | None = None,
    *,
    diagnostics: Mapping[str, Any] | None = None,
    lab_report: Any | None = None,
) -> str:
    """Collapse warning flags, diagnostics, and lab stages into pass/caution/fail.

    "missing" / "diagnostic_only" / "disabled_future" are treated as
    informational for the headline badge (they are surfaced in the cards), while
    "caution" -> CAUTION and "fail" -> FAIL.
    """
    severity = 0
    for info in (warning_flags or {}).values():
        severity = max(severity, STATUS_SEVERITY.get(info.get("status", "pass"), 0))
    if diagnostics is not None:
        severity = max(severity, STATUS_SEVERITY.get(diagnostics.get("warning_level", "pass"), 0))
    if lab_report is not None and hasattr(lab_report, "stages"):
        for stage in lab_report.stages:
            severity = max(severity, STATUS_SEVERITY.get(stage.status_level, 0))
    if severity >= 3:
        return "fail"
    if severity >= 2:
        return "caution"
    return "pass"


def build_beam_path_strip(lab_report: Any | None) -> list[dict[str, Any]]:
    """Return compact per-stage chips for the beam-path strip."""
    if lab_report is None or not hasattr(lab_report, "stages"):
        return []
    chips: list[dict[str, Any]] = []
    for stage in lab_report.stages:
        status = stage.status_level if stage.status_level in STATUS_DARK else "info"
        chips.append(
            {
                "stage_name": stage.stage_name,
                "short_label": _SHORT_STAGE_LABELS.get(stage.stage_name, stage.stage_name[:8]),
                "status_level": status,
                "enabled": bool(stage.enabled),
                "note": (stage.warnings[0] if stage.warnings else ""),
            }
        )
    return chips


# ---------------------------------------------------------------------------
# Stage 8C.2 drawing helpers
# ---------------------------------------------------------------------------


def _blank_panel(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _header_band(
    ax: Any,
    controls: Mapping[str, Any],
    energy_ledger: EnergyLedger | None,
    diagnostics: Mapping[str, Any],
    overall_status: str,
    title: str,
) -> None:
    _blank_panel(ax)
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfd8dc")
        spine.set_linewidth(1.5)
    ax.text(0.012, 0.74, title, fontsize=20, fontweight="bold", color="#102027", va="center")
    ax.text(
        0.012, 0.30,
        "Optical / fluence diagnostic only  -  no material response  -  final_export_allowed=False",
        fontsize=11.5, color="#37474f", va="center", style="italic",
    )
    lines = [
        f"route : {controls.get('generation_method', 'holographic')}   ell={controls.get('ell', 'na')}",
        f"lambda: {controls.get('wavelength_nm', 'na')} nm   tau: {controls.get('pulse_duration_fs', 'na')} fs   "
        f"f: {controls.get('repetition_rate_Hz', 'na')} Hz",
        f"E in  : {controls.get('pulse_energy_before_optics_uJ', 'na')} uJ   "
        f"E@sample: {_energy_at_sample(energy_ledger)} uJ",
        f"sel z : {_fmt(diagnostics.get('selected_plane_z_um'))} um   target: {controls.get('focus_depth_um', 'na')} um",
    ]
    ax.text(0.45, 0.52, "\n".join(lines), fontsize=10.5, color="#263238",
            va="center", ha="left", family="monospace", linespacing=1.7)
    dark = STATUS_DARK[overall_status]
    light = STATUS_LIGHT[overall_status]
    ax.add_patch(FancyBboxPatch(
        (0.845, 0.14), 0.14, 0.72,
        boxstyle="round,pad=0.005,rounding_size=0.06",
        facecolor=light, edgecolor=dark, lw=3.0,
    ))
    ax.text(0.915, 0.66, "DISPLAY TRUST", ha="center", va="center", fontsize=9.5, color=dark, fontweight="bold")
    ax.text(0.915, 0.40, BADGE_WORD.get(overall_status, overall_status.upper()),
            ha="center", va="center", fontsize=23, color=dark, fontweight="bold")


def _beam_path_strip_panel(ax: Any, strip: list[dict[str, Any]]) -> None:
    _blank_panel(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, 0.97, "Beam path - stage status", fontsize=12, fontweight="bold", va="top")
    if not strip:
        ax.text(0.5, 0.45, "lab realism report not available", ha="center", va="center",
                fontsize=11, color="#b71c1c")
        return
    n = len(strip)
    gap = 0.006
    w = (1.0 - gap * (n - 1)) / n
    y0, h = 0.12, 0.62
    for i, chip in enumerate(strip):
        st = chip["status_level"]
        dark = STATUS_DARK.get(st, "#37474f")
        light = STATUS_LIGHT.get(st, "#eceff1")
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.002,rounding_size=0.03",
            facecolor=light, edgecolor=dark, lw=1.6,
        ))
        ax.add_patch(Rectangle((x, y0 + h - 0.17), w, 0.17, color=dark))
        ax.text(x + w / 2, y0 + h - 0.085, STATUS_WORD.get(st, "INFO"),
                ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
        ax.text(x + w / 2, y0 + 0.24, chip["short_label"], ha="center", va="center",
                fontsize=7.4, color="#212121", fontweight="bold")
        if not chip["enabled"]:
            ax.text(x + w / 2, y0 + 0.07, "(off)", ha="center", va="center",
                    fontsize=6.0, color="#607d8b")
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap, y0 + h / 2), xytext=(x + w, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#b0bec5", lw=1.2))


def _warning_cards_panel(ax: Any, warning_flags: Mapping[str, Mapping[str, str]] | None) -> None:
    _blank_panel(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, 0.985, "Warnings & feasibility flags", fontsize=12, fontweight="bold", va="top")
    items = list((warning_flags or {}).items())
    if not items:
        ax.text(0.5, 0.4, "no warning flags", ha="center", va="center", fontsize=10)
        return
    ncol = 3
    nrow = ceil(len(items) / ncol)
    gx, gy = 0.014, 0.05
    top = 0.85
    cw = (1.0 - gx * (ncol - 1)) / ncol
    ch = (top - gy * (nrow - 1)) / nrow
    for idx, (name, info) in enumerate(items):
        r, c = divmod(idx, ncol)
        st = info.get("status", "pass")
        dark = STATUS_DARK.get(st, "#37474f")
        light = STATUS_LIGHT.get(st, "#eceff1")
        x = c * (cw + gx)
        card_top = top - r * (ch + gy)
        y = card_top - ch
        ax.add_patch(FancyBboxPatch(
            (x, y), cw, ch, boxstyle="round,pad=0.004,rounding_size=0.05",
            facecolor=light, edgecolor=dark, lw=1.6,
        ))
        ax.add_patch(Rectangle((x, y + ch - 0.13), cw, 0.13, color=dark))
        ax.text(x + 0.012, y + ch - 0.065, f"{STATUS_WORD.get(st, 'INFO')}  {_humanize(name)}",
                ha="left", va="center", fontsize=8.8, color="white", fontweight="bold")
        msg = textwrap.fill(str(info.get("message", "")), width=44)
        ax.text(x + 0.012, y + ch - 0.17, msg, ha="left", va="top", fontsize=7.8, color="#212121")


def _card(
    ax: Any,
    title: str,
    body_lines: Any,
    *,
    status: str = "diagnostic_only",
    body_fs: float = 10.0,
    title_fs: float = 12.5,
) -> None:
    dark = STATUS_DARK.get(status, "#37474f")
    light = STATUS_LIGHT.get(status, "#eceff1")
    _blank_panel(ax)
    for spine in ax.spines.values():
        spine.set_edgecolor(dark)
        spine.set_linewidth(2.0)
    ax.set_facecolor(light)
    ax.add_patch(Rectangle((0, 0.88), 1, 0.12, color=dark, zorder=2))
    ax.text(0.025, 0.94, title, color="white", fontsize=title_fs, fontweight="bold",
            va="center", ha="left", zorder=3)
    body = "\n".join(str(line) for line in body_lines) if isinstance(body_lines, (list, tuple)) else str(body_lines)
    ax.text(0.025, 0.83, body, color="#212121", fontsize=body_fs, va="top", ha="left",
            family="monospace", zorder=3, linespacing=1.45)


def _ledger_bars(ax: Any, ledger: EnergyLedger | None) -> None:
    if ledger is None:
        _blank_panel(ax)
        ax.text(0.5, 0.5, "energy ledger not available", ha="center", va="center", fontsize=10)
        return
    labels = ["input"] + [_SHORT_STAGE_LABELS.get(r.component_name, r.component_name[:8]) for r in ledger.rows]
    first = ledger.rows[0].energy_in_uJ if ledger.rows else ledger.energy_at_sample_uJ
    values = [first] + [r.energy_out_uJ for r in ledger.rows]
    colors = ["#455a64"] + ["#1565c0"] * len(ledger.rows)
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("pulse energy (uJ)", fontsize=10)
    ax.set_title("Energy ledger", fontsize=12, fontweight="bold")
    ax.axhline(ledger.energy_at_sample_uJ, color="#d32f2f", ls=":", lw=1.6)
    ax.annotate(
        f"E@sample = {ledger.energy_at_sample_uJ:.3g} uJ  ({ledger.total_throughput_fraction:.0%})",
        xy=(0.5, 0.93), xycoords="axes fraction", ha="center", fontsize=9,
        color="#d32f2f", fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)


def _annotate(ax: Any, lines: list[str]) -> None:
    ax.text(
        0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
        fontsize=8.2, color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000", alpha=0.55, edgecolor="none"),
    )


def _experiment_card_lines(controls: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> list[str]:
    return [
        f"laser     : {controls.get('wavelength_nm', 'na')} nm / {controls.get('pulse_duration_fs', 'na')} fs",
        f"rep rate  : {controls.get('repetition_rate_Hz', 'na')} Hz",
        f"E in      : {controls.get('pulse_energy_before_optics_uJ', 'na')} uJ",
        f"route     : {controls.get('generation_method', 'holographic')}  ell={controls.get('ell', 'na')}",
        f"sample    : {controls.get('material_name', 'na')}  n={controls.get('refractive_index', 'na')}",
        f"target z  : {controls.get('focus_depth_um', 'na')} um",
        f"selected z: {_fmt(diagnostics.get('selected_plane_z_um'))} um",
        f"   reason : {diagnostics.get('selected_plane_reason', 'na')}",
        f"scan      : {controls.get('writing_mode', 'na')} @ {controls.get('scan_speed_mm_s', 'na')} mm/s",
        "claim     : optical fluence only",
    ]


def _exposure_card_lines(controls: Mapping[str, Any], exposure: Mapping[str, Any]) -> list[str]:
    return [
        f"mode       : {controls.get('writing_mode', 'na')}  axis={controls.get('scan_axis', 'na')}",
        f"scan speed : {controls.get('scan_speed_mm_s', 'na')} mm/s",
        f"pulse pitch: {_fmt(exposure.get('pulse_spacing_um'))} um",
        f"pulses/spot: {_fmt(exposure.get('pulses_per_spot'))}",
        f"line time  : {_fmt(exposure.get('line_duration_s'))} s",
        f"total pulse: {exposure.get('total_pulses_on_line', 'na')}",
        f"continuity : {_continuity(exposure)}",
        "status     : exposure bookkeeping only",
    ]


def _continuity(exposure: Mapping[str, Any]) -> str:
    warnings = exposure.get("warnings", []) or []
    if not warnings:
        return "ok"
    return textwrap.shorten("; ".join(str(w) for w in warnings), width=42, placeholder="...")


def _exposure_status(exposure: Mapping[str, Any] | None) -> str:
    if not exposure:
        return "missing"
    return "caution" if (exposure.get("warnings") or []) else "pass"


def _claim_boundary_lines(
    diagnostics: Mapping[str, Any],
    perturbation_result: Any | None = None,
    degradation_metrics: Mapping[str, Any] | None = None,
) -> list[str]:
    lines = [
        f"display trust : {str(diagnostics.get('warning_level', 'pass')).upper()}",
        f"selected plane: {_fmt(diagnostics.get('selected_plane_z_um'))} um "
        f"({diagnostics.get('selected_plane_reason', 'na')})",
        f"global peak near boundary: {diagnostics.get('global_peak_near_boundary')}",
        f"captured-power drift: {_fmt_pct(diagnostics.get('captured_power_drift_fraction'))}",
        "Active perturbation sensitivity available:",
        "  outputs/figures/digital_twin/stage8c3_misalignment_sensitivity_sweep_preview.png",
    ]
    if perturbation_result is not None:
        active = getattr(perturbation_result, "active_controls", []) or []
        uncoupled = getattr(perturbation_result, "uncoupled_enabled_controls", []) or []
        lines += [
            "",
            "Active lab realism perturbations:",
            f"ACTIVE: {len(active)} coupled control(s)",
            f"REPORT_ONLY/FUTURE: {len(uncoupled)} enabled uncoupled control(s)",
        ]
        for row in list(active)[:3]:
            lines.append(f"  ACTIVE {row.control}: {row.classification}")
        for row in list(uncoupled)[:3]:
            label = "FUTURE" if row.classification == "future_not_implemented" else "REPORT_ONLY"
            lines.append(f"  {label} {row.control}: {row.classification}")
    if degradation_metrics:
        lines += [
            "",
            "Compact degradation metrics:",
            f"centroid shift: ({_fmt(degradation_metrics.get('centroid_x_um'))}, {_fmt(degradation_metrics.get('centroid_y_um'))}) um",
            f"symmetry score: {_fmt(degradation_metrics.get('symmetry_score'))}",
            f"azimuthal uniformity: {_fmt(degradation_metrics.get('azimuthal_uniformity_score'))}",
            f"core fill: {_fmt(degradation_metrics.get('core_fill_fraction'))}",
            f"peak fluence change: {_fmt_pct(degradation_metrics.get('peak_fluence_change_fraction'))}",
            f"pupil clipping: {_fmt_pct(degradation_metrics.get('pupil_clipped_power_fraction'))}",
            f"captured-power drift: {_fmt_pct(degradation_metrics.get('captured_power_drift_fraction'))}",
        ]
    for msg in diagnostics.get("warning_messages", []) or []:
        lines.append(f"warning: {msg}")
    lines += [
        "",
        "This panel shows OPTICAL FLUENCE only.",
        "Not absorbed energy. Not a deposited-energy volume.",
        "Not material modification. Not a written feature.",
        "Not plasma, nonlinear, or thermal accumulation.",
        "Dose, thresholds, and material response are future stages.",
    ]
    return lines
