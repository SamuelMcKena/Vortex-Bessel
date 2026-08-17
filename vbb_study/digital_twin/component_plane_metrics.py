"""Stage 8C.3R component-plane diagnostics: energy ledger, axis tracking,
translation-vs-deformation classification, scenarios, and the preview figure.

Everything here consumes the physically propagated
:class:`PropagatedFieldStack` (perturbed before propagation), so the metrics
describe genuine optical behaviour, not a post-stack image transform.

Model status: optical / fluence diagnostic only.  ``final_export_allowed=False``.
No material response / dose / plasma / index change is computed or claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.digital_twin.field_coupling import stack_from_arrays
from vbb_study.digital_twin.field_fluence import FluenceStackResult, scale_stack_to_fluence
from vbb_study.digital_twin.component_plane_pipeline import (
    ComponentPlaneConfig,
    ComponentPlaneRun,
    run_component_plane_pipeline,
)
from vbb_study.digital_twin.component_plane_states import PropagatedFieldStack
from vbb_study.digital_twin.annular_axis_tracking import (
    RAW_PEAK_LABEL,
    estimate_annular_axis,
    track_axis_trajectory,
)

FINAL_EXPORT_ALLOWED = False
FIGURE_STATUS = "diagnostic_allowed"
MODEL_STATUS = "optical_prediction"


# ---------------------------------------------------------------------------
# Fluence conversion (honest: scale to the energy that actually survived)
# ---------------------------------------------------------------------------


def stack_to_fluence(stack: PropagatedFieldStack) -> FluenceStackResult:
    """Scale a propagated stack to fluence using the *transmitted* sample energy.

    Passive clipping lowers ``sample_pulse_energy_uJ``; we never re-normalise back
    to the pre-clip input energy.
    """
    ofs = stack_from_arrays(
        stack.intensity_zyx,
        stack.x_um,
        stack.y_um,
        stack.z_um,
        field_label="stage8c3r_component_plane",
        source_status="unit_test_fixture",
    )
    return scale_stack_to_fluence(ofs, float(stack.sample_pulse_energy_uJ))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _nearest_index(coords: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(coords, dtype=float) - float(value))))


def _peak_plane_index(intensity: np.ndarray) -> int:
    return int(np.argmax(np.max(intensity, axis=(1, 2))))


def _centroid(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    s = float(np.sum(plane))
    if s <= 0:
        return 0.0, 0.0
    X, Y = np.meshgrid(x, y)
    return float(np.sum(plane * X) / s), float(np.sum(plane * Y) / s)


def _ring_centre(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Centroid of the bright ring (intensity > 50% peak)."""
    pk = float(np.max(plane))
    if pk <= 0:
        return 0.0, 0.0
    mask = plane >= 0.5 * pk
    if not np.any(mask):
        return _centroid(plane, x, y)
    X, Y = np.meshgrid(x, y)
    w = plane * mask
    s = float(np.sum(w))
    return float(np.sum(w * X) / s), float(np.sum(w * Y) / s)


def _peak_xy(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    iy, ix = np.unravel_index(int(np.argmax(plane)), plane.shape)
    return float(x[ix]), float(y[iy])


def _core_fill_fraction(plane: np.ndarray, x: np.ndarray, y: np.ndarray, core_radius_um: float) -> float:
    """Mean intensity inside the core ROI relative to the plane peak."""
    pk = float(np.max(plane))
    if pk <= 0:
        return 0.0
    X, Y = np.meshgrid(x, y)
    core = np.hypot(X, Y) <= float(core_radius_um)
    if not np.any(core):
        return 0.0
    return float(np.mean(plane[core]) / pk)


def _out_of_frame_fraction(plane: np.ndarray, border_px: int = 2) -> float:
    """Fraction of plane energy in the outer ``border_px`` frame (FOV spill)."""
    total = float(np.sum(plane))
    if total <= 0:
        return 0.0
    mask = np.zeros(plane.shape, dtype=bool)
    mask[:border_px, :] = mask[-border_px:, :] = True
    mask[:, :border_px] = mask[:, -border_px:] = True
    return float(np.sum(plane[mask]) / total)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.clip(np.sum(a * b) / (na * nb), 0.0, 1.0))


def _shift_to_align(plane: np.ndarray, dx_px: int, dy_px: int) -> np.ndarray:
    return np.roll(np.roll(plane, -dy_px, axis=0), -dx_px, axis=1)


def _ring_fit_quality(plane: np.ndarray, x: np.ndarray, y: np.ndarray,
                      cx: float, cy: float) -> tuple[float, float]:
    """Return (ring_fit_quality, ring_radius_um).

    Fit quality = 1 - normalised radial spread of the bright-ring pixels about the
    ring centre; 1.0 means a perfectly circular ring.
    """
    pk = float(np.max(plane))
    if pk <= 0:
        return 0.0, 0.0
    mask = plane >= 0.5 * pk
    if not np.any(mask):
        return 0.0, 0.0
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X[mask] - cx, Y[mask] - cy)
    w = plane[mask]
    mean_r = float(np.average(r, weights=w))
    if mean_r <= 1e-9:
        return 1.0, 0.0
    std_r = float(np.sqrt(np.average((r - mean_r) ** 2, weights=w)))
    return float(np.clip(1.0 - std_r / mean_r, 0.0, 1.0)), mean_r


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------


def compute_axis_tracking(
    stack: PropagatedFieldStack,
    *,
    plane_index: int | None = None,
    core_radius_um: float = 6.0,
) -> dict[str, Any]:
    """Commanded-axis vs actual-beam-axis metrics, incl. a z-trajectory fit.

    Stage 8C.3R.2 deliberately makes the fitted annular ring/core centre the
    primary axis estimate.  ``peak_x_um``/``peak_y_um`` remain as backward-
    compatible aliases for the raw brightest-pixel diagnostic, but they do not
    drive steering, FOV convergence, or axis-error metrics.
    """
    I = np.asarray(stack.intensity_zyx, dtype=float)
    x = np.asarray(stack.x_um, dtype=float)
    y = np.asarray(stack.y_um, dtype=float)
    z = np.asarray(stack.z_um, dtype=float)
    if plane_index is None:
        plane_index = _peak_plane_index(I)
    plane_index = int(np.clip(plane_index, 0, I.shape[0] - 1))
    plane = I[plane_index]

    cen_x, cen_y = _centroid(plane, x, y)
    est = estimate_annular_axis(plane, x, y)
    traj = track_axis_trajectory(I, x, y, z, estimator_mode="auto")
    ring_x, ring_y = float(est["ring_centre_x_um"]), float(est["ring_centre_y_um"])
    peak_x, peak_y = float(est["brightest_pixel_x_um"]), float(est["brightest_pixel_y_um"])
    fit_quality = float(traj["trajectory_fit_quality"])
    core_fill = _core_fill_fraction(plane, x, y, core_radius_um)
    axis_x = float(est["beam_axis_x_um"])
    axis_y = float(est["beam_axis_y_um"])
    axis_error = float(np.hypot(axis_x, axis_y)) if np.isfinite(axis_x) else float("nan")

    return {
        "commanded_axis_x_um": 0.0,
        "commanded_axis_y_um": 0.0,
        "intensity_centroid_x_um": cen_x,
        "intensity_centroid_y_um": cen_y,
        "ring_centre_x_um": ring_x,
        "ring_centre_y_um": ring_y,
        "ring_fit_quality": float(est["ring_fit_quality"]),
        "ring_fit_method": est["ring_fit_method"],
        "ring_fit_reliable": bool(est["ring_fit_reliable"]),
        "ring_radius_um": float(est["ring_radius_um"]),
        "ring_circularity": float(est["ring_circularity"]),
        "azimuthal_uniformity": float(est["azimuthal_uniformity"]),
        "core_centre_x_um": float(est["core_centre_x_um"]),
        "core_centre_y_um": float(est["core_centre_y_um"]),
        "core_fit_quality": float(est["core_fit_quality"]),
        "core_fit_reliable": bool(est["core_fit_reliable"]),
        "roi_intensity_centroid_x_um": float(est["roi_intensity_centroid_x_um"]),
        "roi_intensity_centroid_y_um": float(est["roi_intensity_centroid_y_um"]),
        "phase_singularity_x_um": float(est["phase_singularity_x_um"]),
        "phase_singularity_y_um": float(est["phase_singularity_y_um"]),
        "phase_singularity_reliable": bool(est["phase_singularity_reliable"]),
        "beam_axis_x_um": axis_x,
        "beam_axis_y_um": axis_y,
        "beam_axis_method": est["beam_axis_method"],
        "beam_axis_fit_quality": float(est["beam_axis_fit_quality"]),
        "beam_axis_reliability": est["beam_axis_reliability"],
        "beam_axis_error_um": axis_error,
        "brightest_pixel_x_um": peak_x,
        "brightest_pixel_y_um": peak_y,
        "brightest_pixel_status": RAW_PEAK_LABEL,
        "peak_x_um": peak_x,
        "peak_y_um": peak_y,
        "peak_status": RAW_PEAK_LABEL,
        "radial_axis_error_um": axis_error,
        "field_of_view_margin_um": float(est["field_of_view_margin_um"]),
        "out_of_frame_fraction": float(est["out_of_frame_fraction"]),
        "axis_intercept_at_surface_x_um": float(traj["axis_intercept_at_z0_x_um"]),
        "axis_intercept_at_surface_y_um": float(traj["axis_intercept_at_z0_y_um"]),
        # Stage 8C.3R.1 reference-plane names (z0 = first reference plane in the stack)
        "axis_intercept_at_z0_x_um": float(traj["axis_intercept_at_z0_x_um"]),
        "axis_intercept_at_z0_y_um": float(traj["axis_intercept_at_z0_y_um"]),
        "reference_plane_axis_error_um": float(traj["reference_plane_axis_error_um"]),
        "valid_z_fit_range_um": tuple(traj["valid_z_fit_range_um"]),
        "valid_plane_fraction": float(traj["valid_plane_fraction"]),
        "target_plane_axis_error_um": axis_error,
        "beam_steering_angle_x_mrad": float(traj["beam_steering_angle_x_mrad"]),
        "beam_steering_angle_y_mrad": float(traj["beam_steering_angle_y_mrad"]),
        "trajectory_fit_quality": fit_quality,
        "centre_trajectory_x_um": list(traj["axis_x_by_z_um"]),
        "centre_trajectory_y_um": list(traj["axis_y_by_z_um"]),
        "axis_x_by_z_um": list(traj["axis_x_by_z_um"]),
        "axis_y_by_z_um": list(traj["axis_y_by_z_um"]),
        "per_plane_axis_method": list(traj["per_plane_method"]),
        "per_plane_axis_reliability": list(traj["per_plane_reliability"]),
        "plane_index": plane_index,
        "core_fill_fraction": core_fill,
        "central_darkness_contrast": float(np.clip(1.0 - core_fill, 0.0, 1.0)),
    }


def compute_energy_throughput(run: ComponentPlaneRun) -> dict[str, Any]:
    """Honest energy ledger for one component-plane run."""
    st = run.propagated_stack
    fluence = stack_to_fluence(st)
    peak_fluence = float(np.max(fluence.peak_fluence_by_z_j_cm2))
    sample_energy = float(st.sample_pulse_energy_uJ)
    rows = []
    for s in st.plane_states:
        rows.append(
            {
                "plane": s.plane_name,
                "energy_before_uJ": float(s.pulse_energy_before_uJ),
                "energy_after_uJ": float(s.pulse_energy_after_uJ),
                "transmitted_fraction": float(s.transmitted_fraction),
                "applied": list(s.applied_components),
            }
        )
    return {
        "input_pulse_energy_uJ": float(st.input_pulse_energy_uJ),
        "sample_pulse_energy_uJ": sample_energy,
        # Stage 8C.3R.1 free-space reference-plane name (n=1.0); alias of sample energy.
        "reference_plane_pulse_energy_uJ": sample_energy,
        "transmitted_fraction": float(st.transmitted_fraction),
        "throughput_loss_fraction": float(1.0 - st.transmitted_fraction),
        "peak_fluence_J_cm2": peak_fluence,
        "peak_to_sample_energy_ratio": float(peak_fluence / max(sample_energy, 1e-12)),
        "peak_to_reference_energy_ratio": float(peak_fluence / max(sample_energy, 1e-12)),
        "renormalisation_factor": 1.0,  # no per-plane renormalisation is applied
        "renormalisation_note": "no per-plane re-normalisation to pre-clip energy (display-only scaling excluded)",
        "per_plane_ledger": rows,
        "propagation_energy_drift_fraction": float(fluence.propagation_energy_drift_fraction),
    }


def classify_translation_vs_deformation(
    baseline: PropagatedFieldStack,
    perturbed: PropagatedFieldStack,
    *,
    plane_index: int | None = None,
    core_radius_um: float = 6.0,
) -> dict[str, Any]:
    """Separate translation, shape deformation, clipping and core contamination."""
    Ib = np.asarray(baseline.intensity_zyx, dtype=float)
    Ip = np.asarray(perturbed.intensity_zyx, dtype=float)
    x = np.asarray(baseline.x_um, dtype=float)
    y = np.asarray(baseline.y_um, dtype=float)
    if plane_index is None:
        plane_index = _peak_plane_index(Ib)
    plane_index = int(np.clip(plane_index, 0, Ib.shape[0] - 1))
    a = Ib[plane_index]
    b = Ip[plane_index]

    bx, by = _centroid(b, x, y)
    ax, ay = _centroid(a, x, y)
    centroid_shift = float(np.hypot(bx - ax, by - ay))
    dx_um = float(np.mean(np.abs(np.diff(x))))
    dx_px = int(round((bx - ax) / dx_um))
    dy_px = int(round((by - ay) / dx_um))

    unreg = _cosine_similarity(a, b)
    b_aligned = _shift_to_align(b, dx_px, dy_px)
    reg = _cosine_similarity(a, b_aligned)
    residual = float(np.clip(1.0 - reg, 0.0, 1.0))

    translation_dominated = bool(
        (reg - unreg) > 0.05 and reg > 0.90 and centroid_shift > 0.5 * dx_um
    )

    core_fill = _core_fill_fraction(b, x, y, core_radius_um)
    base_core = _core_fill_fraction(a, x, y, core_radius_um)
    return {
        "unregistered_similarity_score": unreg,
        "registered_similarity_score": reg,
        "centroid_shift_um": centroid_shift,
        "translation_dominated_boolean": translation_dominated,
        "residual_shape_deformation_score": residual,
        "throughput_loss_fraction": float(1.0 - perturbed.transmitted_fraction),
        "core_contamination_fraction": float(max(0.0, core_fill - base_core)),
        "core_fill_fraction": float(core_fill),
        "out_of_frame_fraction": _out_of_frame_fraction(b),
        "plane_index": plane_index,
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentPlaneScenario:
    key: str
    title: str
    physical_placement: str
    mild_controls: Mapping[str, Any]
    severe_controls: Mapping[str, Any]
    mild_label: str
    severe_label: str
    expected_class: str  # translation | deformation | clipping | core_contamination
    note: str = ""


def build_component_plane_scenarios() -> dict[str, ComponentPlaneScenario]:
    """The required physically-placed Stage 8C.3R scenarios."""
    return {
        "vortex_misregistration": ComponentPlaneScenario(
            key="vortex_misregistration",
            title="Scenario 1 - relative vortex/axicon misregistration (vortex moved)",
            physical_placement="SLM phase-mask generation (independent vortex centre)",
            mild_controls={"enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 3.0},
            severe_controls={"enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 8.0},
            mild_label="vortex x=3 um; axicon fixed",
            severe_label="vortex x=8 um; axicon fixed",
            expected_class="deformation",
            note="relative phase-mask error; residual deformation after recentering",
        ),
        "axicon_misregistration": ComponentPlaneScenario(
            key="axicon_misregistration",
            title="Scenario 1b - relative misregistration (axicon moved, vortex fixed)",
            physical_placement="SLM phase-mask generation (independent axicon centre)",
            mild_controls={"enable_axicon_centre_offset": True, "axicon_centre_offset_x_um": 3.0},
            severe_controls={"enable_axicon_centre_offset": True, "axicon_centre_offset_x_um": 8.0},
            mild_label="axicon x=3 um; vortex fixed",
            severe_label="axicon x=8 um; vortex fixed",
            expected_class="deformation",
            note="inverse relative phase-mask error",
        ),
        "vortex_axicon_coshift": ComponentPlaneScenario(
            key="vortex_axicon_coshift",
            title="Translation diagnostic - co-shifted vortex+axicon",
            physical_placement="SLM phase-mask generation (common centre shift)",
            mild_controls={
                "enable_beam_decentre": True, "beam_decentre_x_um": 3.0,
                "enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 3.0,
                "enable_axicon_centre_offset": True, "axicon_centre_offset_x_um": 3.0,
            },
            severe_controls={
                "enable_beam_decentre": True, "beam_decentre_x_um": 8.0,
                "enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 8.0,
                "enable_axicon_centre_offset": True, "axicon_centre_offset_x_um": 8.0,
            },
            mild_label="beam+vortex+axicon x=3 um",
            severe_label="beam+vortex+axicon x=8 um",
            expected_class="translation",
            note="beam and mask move together = clean translation, not deformation",
        ),
        "input_decentre_slm_area": ComponentPlaneScenario(
            key="input_decentre_slm_area",
            title="Scenario 2 - input beam decentre + finite SLM active area",
            physical_placement="input complex field + SLM amplitude/active-area",
            mild_controls={
                "enable_beam_decentre": True, "beam_decentre_x_um": 6.0,
                "enable_slm_active_area": True, "slm_active_width_um": 44.0, "slm_active_height_um": 44.0,
            },
            severe_controls={
                "enable_beam_decentre": True, "beam_decentre_x_um": 16.0,
                "enable_slm_active_area": True, "slm_active_width_um": 30.0, "slm_active_height_um": 30.0,
            },
            mild_label="decentre 6 um; SLM area 44 um",
            severe_label="decentre 16 um; SLM area 30 um",
            expected_class="clipping",
            note="beam moves against a fixed device aperture; throughput falls",
        ),
        "beam_tilt_pupil": ComponentPlaneScenario(
            key="beam_tilt_pupil",
            title="Scenario 3 - beam tilt + finite pupil",
            physical_placement="input phase ramp + objective pupil aperture",
            mild_controls={
                "enable_beam_tilt": True, "beam_tilt_x_mrad": 4.0,
                "enable_pupil_clipping": True, "pupil_radius_um": 26.0,
            },
            severe_controls={
                "enable_beam_tilt": True, "beam_tilt_x_mrad": 16.0,
                "enable_pupil_clipping": True, "pupil_radius_um": 20.0,
            },
            mild_label="tilt 4 mrad; pupil 26 um",
            severe_label="tilt 16 mrad; pupil 20 um",
            expected_class="translation",
            note="tilt steers the axis; pupil sets throughput",
        ),
        "pupil_decentre_clip": ComponentPlaneScenario(
            key="pupil_decentre_clip",
            title="Scenario 4 - objective pupil decentre / clipping",
            physical_placement="objective pupil plane (aperture before focus)",
            mild_controls={"enable_pupil_clipping": True, "pupil_radius_um": 18.0, "pupil_decentre_x_um": 5.0},
            severe_controls={"enable_pupil_clipping": True, "pupil_radius_um": 12.0, "pupil_decentre_x_um": 12.0},
            mild_label="pupil 18 um; decentre 5 um",
            severe_label="pupil 12 um; decentre 12 um",
            expected_class="clipping",
            note="lower throughput + diffraction from a hard pupil edge (no output cut-out)",
        ),
        "zernike_aberrations": ComponentPlaneScenario(
            key="zernike_aberrations",
            title="Scenario 5 - low-order aberrations (coma/astig/defocus)",
            physical_placement="objective pupil plane phase (waves)",
            mild_controls={
                "enable_zernike_aberrations": True,
                "zernike_coma_x_waves": 0.15, "zernike_astig_0_waves": 0.15, "zernike_defocus_waves": 0.15,
            },
            severe_controls={
                "enable_zernike_aberrations": True,
                "zernike_coma_x_waves": 0.8, "zernike_astig_0_waves": 0.6, "zernike_defocus_waves": 0.7,
            },
            mild_label="coma/astig/defocus = 0.15 waves",
            severe_label="coma 0.8, astig 0.6, defocus 0.7 waves",
            expected_class="deformation",
            note="pupil phase aberration -> lopsided ring + axial shift",
        ),
        "zernike_defocus": ComponentPlaneScenario(
            key="zernike_defocus",
            title="Scenario 5a - individual defocus",
            physical_placement="objective pupil plane phase (waves)",
            mild_controls={"enable_zernike_aberrations": True, "zernike_defocus_waves": 0.2},
            severe_controls={"enable_zernike_aberrations": True, "zernike_defocus_waves": 1.0},
            mild_label="defocus 0.2 waves", severe_label="defocus 1.0 waves",
            expected_class="deformation", note="pure defocus -> axial peak shift",
        ),
        "zernike_astigmatism": ComponentPlaneScenario(
            key="zernike_astigmatism",
            title="Scenario 5b - individual astigmatism (0 deg)",
            physical_placement="objective pupil plane phase (waves)",
            mild_controls={"enable_zernike_aberrations": True, "zernike_astig_0_waves": 0.2},
            severe_controls={"enable_zernike_aberrations": True, "zernike_astig_0_waves": 0.8},
            mild_label="astig0 0.2 waves", severe_label="astig0 0.8 waves",
            expected_class="deformation", note="pure astigmatism -> elliptical ring",
        ),
        "zernike_coma": ComponentPlaneScenario(
            key="zernike_coma",
            title="Scenario 5c - individual coma (x)",
            physical_placement="objective pupil plane phase (waves)",
            mild_controls={"enable_zernike_aberrations": True, "zernike_coma_x_waves": 0.2},
            severe_controls={"enable_zernike_aberrations": True, "zernike_coma_x_waves": 0.8},
            mild_label="coma_x 0.2 waves", severe_label="coma_x 0.8 waves",
            expected_class="deformation", note="pure coma -> lopsided ring",
        ),
        "zernike_spherical": ComponentPlaneScenario(
            key="zernike_spherical",
            title="Scenario 5d - individual spherical",
            physical_placement="objective pupil plane phase (waves)",
            mild_controls={"enable_zernike_aberrations": True, "zernike_spherical_waves": 0.2},
            severe_controls={"enable_zernike_aberrations": True, "zernike_spherical_waves": 0.8},
            mild_label="spherical 0.2 waves", severe_label="spherical 0.8 waves",
            expected_class="deformation", note="pure spherical -> radial redistribution + axial shift",
        ),
        "zero_order_leakage": ComponentPlaneScenario(
            key="zero_order_leakage",
            title="Scenario 6 - zero-order leakage (core contamination)",
            physical_placement="SLM order content (coherent unmodulated carrier)",
            mild_controls={"enable_zero_order_leakage": True, "zero_order_leakage_fraction": 0.05},
            severe_controls={"enable_zero_order_leakage": True, "zero_order_leakage_fraction": 0.25},
            mild_label="zero-order fraction 0.05",
            severe_label="zero-order fraction 0.25",
            expected_class="core_contamination",
            note="carrier fills the dark vortex core",
        ),
        "combined_stress": ComponentPlaneScenario(
            key="combined_stress",
            title="Scenario 7 - combined diagnostic stress test",
            physical_placement="multiple upstream planes (diagnostic stress only)",
            mild_controls={
                "enable_beam_decentre": True, "beam_decentre_x_um": 5.0,
                "enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 3.0,
                "enable_pupil_clipping": True, "pupil_radius_um": 24.0,
                "enable_zernike_aberrations": True, "zernike_coma_x_waves": 0.2,
                "enable_zero_order_leakage": True, "zero_order_leakage_fraction": 0.05,
            },
            severe_controls={
                "enable_beam_decentre": True, "beam_decentre_x_um": 12.0,
                "enable_slm_active_area": True, "slm_active_width_um": 34.0, "slm_active_height_um": 34.0,
                "enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 8.0,
                "enable_beam_tilt": True, "beam_tilt_x_mrad": 10.0,
                "enable_pupil_clipping": True, "pupil_radius_um": 16.0, "pupil_decentre_x_um": 6.0,
                "enable_zernike_aberrations": True, "zernike_coma_x_waves": 0.6,
                "enable_zero_order_leakage": True, "zero_order_leakage_fraction": 0.12,
            },
            mild_label="mild combined stress",
            severe_label="severe combined stress",
            expected_class="deformation",
            note="diagnostic stress test only; not a predicted normal lab condition",
        ),
    }


@dataclass(frozen=True)
class ComponentPlaneScenarioResult:
    scenario: ComponentPlaneScenario
    config: ComponentPlaneConfig
    baseline: ComponentPlaneRun
    mild: ComponentPlaneRun
    severe: ComponentPlaneRun
    selected_plane_index: int
    baseline_axis: Mapping[str, Any]
    mild_axis: Mapping[str, Any]
    severe_axis: Mapping[str, Any]
    baseline_energy: Mapping[str, Any]
    mild_energy: Mapping[str, Any]
    severe_energy: Mapping[str, Any]
    mild_class: Mapping[str, Any]
    severe_class: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_export_allowed: bool = False


def run_component_plane_scenario(
    scenario: str | ComponentPlaneScenario,
    *,
    config: ComponentPlaneConfig | None = None,
) -> ComponentPlaneScenarioResult:
    """Run baseline/mild/severe for a scenario and compute all diagnostics."""
    if isinstance(scenario, str):
        scenario = build_component_plane_scenarios()[scenario]
    config = config or ComponentPlaneConfig()

    baseline = run_component_plane_pipeline({}, config=config)
    mild = run_component_plane_pipeline(scenario.mild_controls, config=config)
    severe = run_component_plane_pipeline(scenario.severe_controls, config=config)

    sel = _peak_plane_index(baseline.propagated_stack.intensity_zyx)
    # Core ROI must sit INSIDE the dark vortex core (ring radius ~3.6 um here)
    # so that genuine core contamination is measured, not the bright ring tail.
    core_r = 2.0
    return ComponentPlaneScenarioResult(
        scenario=scenario,
        config=config,
        baseline=baseline,
        mild=mild,
        severe=severe,
        selected_plane_index=sel,
        baseline_axis=compute_axis_tracking(baseline.propagated_stack, plane_index=sel, core_radius_um=core_r),
        mild_axis=compute_axis_tracking(mild.propagated_stack, plane_index=sel, core_radius_um=core_r),
        severe_axis=compute_axis_tracking(severe.propagated_stack, plane_index=sel, core_radius_um=core_r),
        baseline_energy=compute_energy_throughput(baseline),
        mild_energy=compute_energy_throughput(mild),
        severe_energy=compute_energy_throughput(severe),
        mild_class=classify_translation_vs_deformation(
            baseline.propagated_stack, mild.propagated_stack, plane_index=sel, core_radius_um=core_r
        ),
        severe_class=classify_translation_vs_deformation(
            baseline.propagated_stack, severe.propagated_stack, plane_index=sel, core_radius_um=core_r
        ),
        metadata={
            "stage": "stage8c3r_component_plane",
            "scenario": scenario.key,
            "expected_class": scenario.expected_class,
            "warnings": list(severe.warnings),
        },
    )


# ---------------------------------------------------------------------------
# Stage 8C.3R.2 individual diagnostic response curves (multi-point sweeps)
# ---------------------------------------------------------------------------

DIAGNOSTIC_SWEEP_LABEL = (
    "Diagnostic sensitivity sweep. Not an experimentally measured laboratory tolerance."
)


@dataclass(frozen=True)
class ResponseCurveFamily:
    key: str
    title: str
    param_label: str
    param_values: tuple[float, ...]
    control_fn: Any                      # value -> controls dict
    expected_class: str = "deformation"


@dataclass(frozen=True)
class ResponseCurveResult:
    family: ResponseCurveFamily
    param_values: tuple[float, ...]
    rows: tuple[Mapping[str, Any], ...]
    label: str = DIAGNOSTIC_SWEEP_LABEL
    final_export_allowed: bool = False


def build_response_curve_families() -> dict[str, ResponseCurveFamily]:
    """The ten physically-represented free-space perturbation families."""
    z = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    return {
        "vortex_offset": ResponseCurveFamily(
            "vortex_offset", "Vortex-centre offset (axicon fixed)", "vortex offset (um)",
            (0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
            lambda v: {"enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": v}),
        "axicon_offset": ResponseCurveFamily(
            "axicon_offset", "Axicon-centre offset (vortex fixed)", "axicon offset (um)",
            (0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
            lambda v: {"enable_axicon_centre_offset": True, "axicon_centre_offset_x_um": v}),
        "beam_decentre": ResponseCurveFamily(
            "beam_decentre", "Input decentre (fixed SLM area 40 um)", "beam decentre (um)",
            (0.0, 3.0, 6.0, 9.0, 12.0, 16.0),
            lambda v: {"enable_beam_decentre": True, "beam_decentre_x_um": v,
                       "enable_slm_active_area": True, "slm_active_width_um": 40.0, "slm_active_height_um": 40.0},
            expected_class="clipping"),
        "beam_tilt": ResponseCurveFamily(
            "beam_tilt", "Input beam tilt", "tilt (mrad)",
            (0.0, 4.0, 8.0, 12.0, 18.0, 24.0),
            lambda v: {"enable_beam_tilt": True, "beam_tilt_x_mrad": v},
            expected_class="translation"),
        "pupil_decentre": ResponseCurveFamily(
            "pupil_decentre", "Pupil decentre (radius 18 um)", "pupil decentre (um)",
            (0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
            lambda v: {"enable_pupil_clipping": True, "pupil_radius_um": 18.0, "pupil_decentre_x_um": v},
            expected_class="clipping"),
        "defocus": ResponseCurveFamily(
            "defocus", "Defocus", "defocus (waves)", z,
            lambda v: {"enable_zernike_aberrations": True, "zernike_defocus_waves": v}),
        "astigmatism": ResponseCurveFamily(
            "astigmatism", "Astigmatism (0 deg)", "astig (waves)", z,
            lambda v: {"enable_zernike_aberrations": True, "zernike_astig_0_waves": v}),
        "coma": ResponseCurveFamily(
            "coma", "Coma (x)", "coma (waves)", z,
            lambda v: {"enable_zernike_aberrations": True, "zernike_coma_x_waves": v}),
        "spherical": ResponseCurveFamily(
            "spherical", "Spherical", "spherical (waves)", z,
            lambda v: {"enable_zernike_aberrations": True, "zernike_spherical_waves": v}),
        "zero_order": ResponseCurveFamily(
            "zero_order", "Zero-order leakage fraction", "leakage fraction",
            (0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
            lambda v: {"enable_zero_order_leakage": True, "zero_order_leakage_fraction": v},
            expected_class="core_contamination"),
    }


def _single_run_axis_reliability(axis: Mapping[str, Any], ring_fit_quality: float) -> str:
    if float(axis["out_of_frame_fraction"]) > 0.02 or float(axis["field_of_view_margin_um"]) < 0.0:
        return "invalid_out_of_frame"
    if float(axis["out_of_frame_fraction"]) > 0.005 or ring_fit_quality < 0.5:
        return "caution_crop_limited"
    return "numerically_reliable"


def run_response_curve(
    family: str | ResponseCurveFamily,
    *,
    config: ComponentPlaneConfig | None = None,
) -> ResponseCurveResult:
    """Run a multi-point parameter sweep and return per-point diagnostic metrics."""
    if isinstance(family, str):
        family = build_response_curve_families()[family]
    config = config or ComponentPlaneConfig()

    baseline = run_component_plane_pipeline({}, config=config)
    base_st = baseline.propagated_stack
    sel = _peak_plane_index(base_st.intensity_zyx)
    x = np.asarray(base_st.x_um, float)
    y = np.asarray(base_st.y_um, float)

    rows: list[dict[str, Any]] = []
    for v in family.param_values:
        run = run_component_plane_pipeline(family.control_fn(v), config=config)
        st = run.propagated_stack
        en = compute_energy_throughput(run)
        ax = compute_axis_tracking(st, plane_index=sel, core_radius_um=2.0)
        est = estimate_annular_axis(st.intensity_zyx[sel], x, y)
        cls = classify_translation_vs_deformation(base_st, st, plane_index=sel, core_radius_um=2.0)
        rows.append({
            "param_value": float(v),
            "diagnostic_label": DIAGNOSTIC_SWEEP_LABEL,
            "transmitted_fraction": float(en["transmitted_fraction"]),
            "reference_plane_pulse_energy_uJ": float(en["reference_plane_pulse_energy_uJ"]),
            "axis_error_um": float(est["beam_axis_error_um"]),
            "axis_method": est["beam_axis_method"],
            "axis_reliability": est["beam_axis_reliability"],
            "azimuthal_uniformity": float(est["azimuthal_uniformity"]),
            "ring_fit_quality": float(est["ring_fit_quality"]),
            "ring_circularity": float(est["ring_circularity"]),
            "residual_shape_deformation": float(cls["residual_shape_deformation_score"]),
            "core_contamination_fraction": float(cls["core_contamination_fraction"]),
            "core_fill_fraction": float(est["core_fill_fraction"]),
            "central_darkness_contrast": float(est["central_darkness_contrast"]),
            "peak_to_reference_energy_ratio": float(en["peak_to_reference_energy_ratio"]),
            "numerical_reliability": _single_run_axis_reliability(ax, float(est["ring_fit_quality"])),
        })
    return ResponseCurveResult(family=family, param_values=tuple(family.param_values), rows=tuple(rows))


# ---------------------------------------------------------------------------
# Preview figure
# ---------------------------------------------------------------------------


def _wrap_phase(field: np.ndarray) -> np.ndarray:
    return np.angle(field)


def _extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _axis_overlay(ax: Any, axis_metrics: Mapping[str, Any],
                  ext: tuple[float, float, float, float]) -> None:
    """Edge ticks (never over the beam): white = commanded axis (x=0, y=0),
    cyan = measured ring centre.  The gap between the white and cyan ticks on
    each border reads off the lateral axis offset directly."""
    xmin, xmax, ymin, ymax = ext
    xspan, yspan = xmax - xmin, ymax - ymin
    tlen_y = 0.05 * yspan   # bottom-edge tick height
    tlen_x = 0.05 * xspan   # left-edge tick width
    rx = float(axis_metrics["ring_centre_x_um"])
    ry = float(axis_metrics["ring_centre_y_um"])
    # bottom border: x positions (commanded white, measured cyan)
    ax.plot([0.0, 0.0], [ymin, ymin + tlen_y], color="white", lw=2.2, zorder=6, clip_on=False)
    ax.plot([rx, rx], [ymin, ymin + tlen_y], color="cyan", lw=2.2, zorder=7, clip_on=False)
    # left border: y positions
    ax.plot([xmin, xmin + tlen_x], [0.0, 0.0], color="white", lw=2.2, zorder=6, clip_on=False)
    ax.plot([xmin, xmin + tlen_x], [ry, ry], color="cyan", lw=2.2, zorder=7, clip_on=False)


def _card(ax: Any, title: str, lines: list[str], face: str, edge: str) -> None:
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=face, edgecolor=edge, lw=1.4, clip_on=False))
    ax.text(0.04, 0.95, title, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="top", ha="left", color=edge)
    ax.text(0.04, 0.82, "\n".join(lines), transform=ax.transAxes, fontsize=8.2,
            va="top", ha="left", family="monospace", color="#1a1a1a")


def plot_component_plane_reality_preview(
    scenario: str | ComponentPlaneScenario = "pupil_decentre_clip",
    *,
    config: ComponentPlaneConfig | None = None,
    result: ComponentPlaneScenarioResult | None = None,
    output_path: str | Path | None = None,
    dpi: int = 170,
    title: str = "Stage 8C.3R Component-Plane Physical Lab-Realism",
) -> "matplotlib.figure.Figure":
    """Render the physically coherent component-plane reality preview.

    Layout: component-plane thumbnails (input field, SLM phase, sample-entrance
    intensity) + baseline/mild/severe XY & XZ fluence with commanded-vs-actual
    axis overlays + energy / axis / classification / metric-delta cards.
    """
    if result is None:
        result = run_component_plane_scenario(scenario, config=config)
    sc = result.scenario
    sel = int(result.selected_plane_index)

    base_st = result.baseline.propagated_stack
    mild_st = result.mild.propagated_stack
    sev_st = result.severe.propagated_stack
    x = np.asarray(base_st.x_um, dtype=float)
    y = np.asarray(base_st.y_um, dtype=float)
    z = np.asarray(base_st.z_um, dtype=float)
    yc = _nearest_index(y, 0.0)

    F0 = np.asarray(stack_to_fluence(base_st).fluence_zyx_j_cm2, dtype=float)
    Fm = np.asarray(stack_to_fluence(mild_st).fluence_zyx_j_cm2, dtype=float)
    Fs = np.asarray(stack_to_fluence(sev_st).fluence_zyx_j_cm2, dtype=float)

    xy = [F0[sel], Fm[sel], Fs[sel]]
    xz = [F0[:, yc, :].T, Fm[:, yc, :].T, Fs[:, yc, :].T]
    xy_vmax = max(float(np.max(a)) for a in xy)
    xz_vmax = max(float(np.max(a)) for a in xz)
    xy_diff = xy[2] - xy[0]
    xz_diff = xz[2] - xz[0]
    xy_dabs = max(float(np.max(np.abs(xy_diff))), 1e-12)
    xz_dabs = max(float(np.max(np.abs(xz_diff))), 1e-12)

    fig = plt.figure(figsize=(18.4, 17.2), facecolor="white")
    gs = fig.add_gridspec(4, 4, height_ratios=[0.95, 1.0, 1.0, 1.15],
                          left=0.075, right=0.95, top=0.88, bottom=0.05,
                          hspace=0.42, wspace=0.26)
    fig.suptitle(f"{title}\n{sc.title}", x=0.045, y=0.975, ha="left", va="top",
                 fontsize=18, fontweight="bold")
    for bx, txt, ec, fc in [
        (0.075, "DIAGNOSTIC ONLY", "#0d47a1", "#e3f2fd"),
        (0.205, "NO MATERIAL RESPONSE", "#4a148c", "#f3e5f5"),
        (0.375, "COMPONENT-PLANE PHYSICS", "#1b5e20", "#e8f5e9"),
        (0.575, "PERTURBED BEFORE PROPAGATION", "#bf360c", "#fff3e0"),
    ]:
        fig.text(bx, 0.915, txt, ha="left", va="center", fontsize=9.5, fontweight="bold",
                 color=ec, bbox=dict(boxstyle="round,pad=0.32", facecolor=fc, edgecolor=ec, lw=1.2))
    fig.text(0.075, 0.895, f"Physical placement: {sc.physical_placement}    "
             f"Mild: {sc.mild_label}    Severe: {sc.severe_label}",
             fontsize=10, ha="left", va="center", color="#263238")

    # --- Row 0: component-plane thumbnails (from the severe run) -----------
    sev = result.severe
    ext = _extent(x, y)
    in_amp = np.abs(sev.input_state.field)
    slm_phase = _wrap_phase(sev.slm_state.field)
    samp_int = np.abs(sev.sample_entrance_state.field) ** 2
    thumbs = [
        (in_amp, "input |E|  (input complex field)", "magma"),
        (slm_phase, "SLM phase arg(E)  (phase-mask plane)", "twilight"),
        (samp_int, "free-space reference |E|^2  (n=1.0)", "viridis"),
        (F0[sel], "baseline reference-plane XY fluence", "viridis"),
    ]
    for col, (arr, lab, cmap) in enumerate(thumbs):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal")
        ax.set_title(lab, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    col_titles = ["Aligned baseline", "Mild perturbation", "Severe perturbation", "Severe - baseline"]
    axis_metrics = [result.baseline_axis, result.mild_axis, result.severe_axis]

    # --- Row 1: XY fluence + axis overlays --------------------------------
    for col in range(3):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(xy[col], origin="lower", extent=ext, cmap="viridis",
                       vmin=0.0, vmax=xy_vmax, aspect="equal")
        _axis_overlay(ax, axis_metrics[col], ext)
        ax.set_title(col_titles[col], fontsize=11, fontweight="bold")
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        if col == 0:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="XY fluence (J/cm^2)")
    axd = fig.add_subplot(gs[1, 3])
    imd = axd.imshow(xy_diff, origin="lower", extent=ext, cmap="coolwarm",
                     vmin=-xy_dabs, vmax=xy_dabs, aspect="equal")
    axd.set_title(col_titles[3], fontsize=11, fontweight="bold")
    axd.set_xlabel("x (um)"); axd.set_ylabel("y (um)")
    fig.colorbar(imd, ax=axd, fraction=0.046, pad=0.02, label="delta J/cm^2")

    # --- Row 2: XZ fluence + per-panel peak plane + fitted axis trajectory --
    ext_xz = (float(z.min()), float(z.max()), float(x.min()), float(x.max()))
    F_list = [F0, Fm, Fs]
    peak_z = []
    for col in range(3):
        Fc = F_list[col]
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(xz[col], origin="lower", extent=ext_xz, cmap="viridis",
                       vmin=0.0, vmax=xz_vmax, aspect="auto")
        # commanded optical axis (x = 0) reference
        ax.axhline(0.0, color="0.85", ls=":", lw=0.8, alpha=0.55)
        # this panel's own peak-intensity plane
        pk_i = int(np.argmax(Fc.max(axis=(1, 2))))
        peak_z.append(float(z[pk_i]))
        ax.axvline(z[pk_i], color="white", ls="--", lw=1.5, alpha=0.95,
                   label=f"peak-intensity plane (z={z[pk_i]:.0f} um)")
        # fitted beam-axis trajectory through propagation
        traj = np.asarray(axis_metrics[col]["centre_trajectory_x_um"], dtype=float)
        ax.plot(z, traj, ls="--", color="cyan", lw=1.4, alpha=0.9,
                label="fitted beam-axis trajectory")
        ax.set_xlabel("z (um)"); ax.set_ylabel("x (um)")
        ax.legend(loc="upper right", fontsize=6.8, framealpha=0.72, handlelength=1.4)
        if col == 0:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="XZ fluence (J/cm^2)")
    axd = fig.add_subplot(gs[2, 3])
    imd = axd.imshow(xz_diff, origin="lower", extent=ext_xz, cmap="coolwarm",
                     vmin=-xz_dabs, vmax=xz_dabs, aspect="auto")
    axd.axvline(peak_z[2], color="black", ls="--", lw=1.2, alpha=0.7,
                label=f"severe peak plane (z={peak_z[2]:.0f} um)")
    axd.set_xlabel("z (um)"); axd.set_ylabel("x (um)")
    axd.legend(loc="upper right", fontsize=6.8, framealpha=0.72, handlelength=1.4)
    fig.colorbar(imd, ax=axd, fraction=0.046, pad=0.02, label="delta J/cm^2")

    # --- Row 3: cards ------------------------------------------------------
    be, me, se = result.baseline_energy, result.mild_energy, result.severe_energy
    energy_lines = [
        f"input pulse energy : {be['input_pulse_energy_uJ']:.2f} uJ",
        "                     base / mild / severe",
        f"transmitted frac   : {be['transmitted_fraction']:.3f} / {me['transmitted_fraction']:.3f} / {se['transmitted_fraction']:.3f}",
        f"ref-plane energy uJ: {be['sample_pulse_energy_uJ']:.2f} / {me['sample_pulse_energy_uJ']:.2f} / {se['sample_pulse_energy_uJ']:.2f}",
        f"peak fluence       : {be['peak_fluence_J_cm2']:.2f} / {me['peak_fluence_J_cm2']:.2f} / {se['peak_fluence_J_cm2']:.2f}",
        f"peak/energy ratio  : {be['peak_to_sample_energy_ratio']:.3f} / {me['peak_to_sample_energy_ratio']:.3f} / {se['peak_to_sample_energy_ratio']:.3f}",
        f"renorm factor      : {se['renormalisation_factor']:.1f} (no pre-clip renorm)",
        f"prop energy drift  : {se['propagation_energy_drift_fraction']:.3f}",
    ]
    _card(fig.add_subplot(gs[3, 0]), "Energy throughput ledger", energy_lines, "#eceff1", "#37474f")

    sa = result.severe_axis
    axis_lines = [
        "commanded axis     : (0.000, 0.000) um (white edge ticks)",
        f"ring centre (sev)  : ({sa['ring_centre_x_um']:.3f}, {sa['ring_centre_y_um']:.3f}) um (cyan edge ticks)",
        f"intensity centroid : ({sa['intensity_centroid_x_um']:.3f}, {sa['intensity_centroid_y_um']:.3f}) um",
        f"peak point         : ({sa['peak_x_um']:.3f}, {sa['peak_y_um']:.3f}) um",
        f"radial axis error  : {sa['radial_axis_error_um']:.3f} um",
        f"steering x/y       : {sa['beam_steering_angle_x_mrad']:.3f} / {sa['beam_steering_angle_y_mrad']:.3f} mrad",
        f"trajectory fit R2  : {sa['trajectory_fit_quality']:.3f}",
        f"FOV margin         : {sa['field_of_view_margin_um']:.2f} um",
        f"out-of-frame frac  : {sa['out_of_frame_fraction']:.4f}",
    ]
    _card(fig.add_subplot(gs[3, 1]), "Commanded vs actual beam axis", axis_lines, "#e3f2fd", "#0d47a1")

    scl = result.severe_class
    verdicts = []
    if scl["translation_dominated_boolean"]:
        verdicts.append("TRANSLATION-DOMINATED")
    if scl["residual_shape_deformation_score"] > 0.15 and not scl["translation_dominated_boolean"]:
        verdicts.append("SHAPE DEFORMATION")
    if scl["throughput_loss_fraction"] > 0.05:
        verdicts.append("CLIPPING / THROUGHPUT LOSS")
    if scl["core_contamination_fraction"] > 0.02:
        verdicts.append("CORE CONTAMINATION")
    if scl["out_of_frame_fraction"] > 0.01:
        verdicts.append("FOV SPILL")
    class_lines = [
        f"expected           : {sc.expected_class}",
        f"unreg similarity   : {scl['unregistered_similarity_score']:.3f}",
        f"reg similarity     : {scl['registered_similarity_score']:.3f}",
        f"centroid shift     : {scl['centroid_shift_um']:.3f} um",
        f"residual deform    : {scl['residual_shape_deformation_score']:.3f}",
        f"throughput loss    : {scl['throughput_loss_fraction']:.3f}",
        f"core contamination : {scl['core_contamination_fraction']:.3f}",
        f"out-of-frame frac  : {scl['out_of_frame_fraction']:.4f}",
        "verdict: " + (", ".join(verdicts) if verdicts else "near-baseline"),
    ]
    _card(fig.add_subplot(gs[3, 2]), "Translation / deformation / clipping", class_lines, "#f3e5f5", "#4a148c")

    summary_lines = [
        f"scenario   : {sc.key}",
        f"placement  : {sc.physical_placement}",
        f"note       : {sc.note}",
        "",
        "claim: optical fluence diagnostic only;",
        "       no material response.",
        "final_export_allowed = False",
    ]
    if result.severe.warnings:
        summary_lines.append("")
        summary_lines.append("warning-only controls (not faked):")
        for w in list(result.severe.warnings)[:3]:
            summary_lines.append("  - " + w.split(":")[0])
    _card(fig.add_subplot(gs[3, 3]), "What changed and why",
          summary_lines, "#e8f5e9", "#1b5e20")

    fig.stage8c3r_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r_component_plane",
        "scenario": sc.key,
        "figure_status": FIGURE_STATUS,
        "model_status": MODEL_STATUS,
        "final_export_allowed": False,
        "severe_transmitted_fraction": float(se["transmitted_fraction"]),
        "severe_radial_axis_error_um": float(sa["radial_axis_error_um"]),
        "severe_residual_deformation": float(scl["residual_shape_deformation_score"]),
        "severe_translation_dominated": bool(scl["translation_dominated_boolean"]),
    }
    fig.stage8c3r_result = result  # type: ignore[attr-defined]

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
            "Title": title, "stage": "stage8c3r_component_plane", "scenario": sc.key,
            "final_export_allowed": "False",
            "Description": "Stage 8C.3R component-plane physical lab-realism; optical fluence diagnostic only.",
        })
    return fig
