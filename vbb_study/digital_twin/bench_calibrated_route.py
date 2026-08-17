"""Bench-calibrated dual-SLM -> 4F -> physical-axicon route.

This route is layered on top of the already-audited propagation/error modules.
It differs from the canonical sensitivity route in exactly the places where
laboratory data must enter: measured hardware values, measured SLM LUTs,
commanded Shack-Hartmann correction maps, static panel maps, and real refractive
axicon geometry for rigid tilt.

No missing measurement is auto-filled.  Non-tilted axicon propagation may use
the validated exact-Snell thin-cone morphology model.  Any non-zero rigid axicon
tilt is *forced* through the explicit two-surface refractive/eikonal reference;
the rejected rotated thin-phase surrogate is not available here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from vbb_study.calibration.bench_binding import BenchBindingReport, bind_calibration_to_manifest
from vbb_study.calibration.schema import CalibrationBundle
from vbb_study.calibration.slm_phase import SLMPhaseCalibration, calibrated_phase_to_grey
from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest
from vbb_study.digital_twin.phase2a_contracts import PHASE2A_CANONICAL_SLM_MODEL, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import (
    SLMError,
    actual_slm_phase,
    apply_fringing_surrogate,
    gaussian_input_field,
    transformed_pattern_coordinates,
)
from vbb_study.digital_twin.vortex_explicit_4f import explicit_4f_relay
from vbb_study.digital_twin.vortex_refractive_axicon import (
    RefractiveAxiconGeometry,
    trace_refractive_axicon_bundle,
)
from vbb_study.digital_twin.vortex_refractive_axicon_wave import build_refractive_axicon_reference_field
from vbb_study.digital_twin.vortex_system_route import (
    SystemErrorConfig,
    physical_axicon_on_own_plane,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.slm_model import apply_slm, pixelate


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class BenchCalibratedInputs:
    calibration_bundle: CalibrationBundle
    slm1_phase_calibration: SLMPhaseCalibration | None = None
    slm2_phase_calibration: SLMPhaseCalibration | None = None
    slm1_correction_phase_rad: np.ndarray | None = None
    slm2_correction_phase_rad: np.ndarray | None = None
    slm1_static_phase_map_rad: np.ndarray | None = None
    slm2_static_phase_map_rad: np.ndarray | None = None
    lens1_opd_map_m: np.ndarray | None = None
    lens2_opd_map_m: np.ndarray | None = None
    axicon_surface_height_error_m: np.ndarray | None = None
    refractive_axicon_geometry: RefractiveAxiconGeometry | None = None
    input_polarization_lab: np.ndarray | None = None



def _ell(case_id: str) -> int:
    try:
        return {"B0": 0, "V1": 1, "V3": 3}[case_id]
    except KeyError as exc:
        raise ValueError(f"unsupported calibrated scalar case {case_id!r}") from exc


def _shape_checked_optional(values: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != route grid {shape}")
    return arr


def _realised_panel_phase(
    command_rad: np.ndarray,
    grid: Mapping[str, Any],
    *,
    error: SLMError,
    pixel_pitch_m: float,
    calibration: SLMPhaseCalibration | None,
    static_phase_map_rad: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return physical panel phase from either measured LUT or sensitivity model."""

    if calibration is None:
        return actual_slm_phase(
            command_rad,
            grid,
            error=error,
            pixel_pitch_m=pixel_pitch_m,
            lut_phase_rad=None,
            static_phase_map_rad=static_phase_map_rad,
        )

    mask = calibrated_phase_to_grey(command_rad, calibration)
    actual = np.asarray(mask.realised_phase_rad, dtype=float)
    if error.phase_stroke_scale != 1.0 or error.phase_bias_rad != 0.0:
        actual = actual * float(error.phase_stroke_scale) + float(error.phase_bias_rad)
    actual = apply_fringing_surrogate(
        actual,
        grid,
        pixel_pitch_m=float(pixel_pitch_m),
        sigma_x_px=float(error.fringing_sigma_x_px),
        sigma_y_px=float(error.fringing_sigma_y_px),
    )
    if static_phase_map_rad is not None:
        actual = actual + np.asarray(static_phase_map_rad, dtype=float)
    metadata = {
        "phase_levels": int(error.phase_levels),
        "phase_stroke_scale": float(error.phase_stroke_scale),
        "phase_bias_rad": float(error.phase_bias_rad),
        "fringing_sigma_x_px": float(error.fringing_sigma_x_px),
        "fringing_sigma_y_px": float(error.fringing_sigma_y_px),
        "phase_lut_status": "measured_inverse_LUT_hardware_mask",
        "phase_lut_panel_id": calibration.panel_id,
        "phase_lut_stroke_rad": float(calibration.phase_stroke_rad),
        "phase_lut_mask_rms_error_rad": float(mask.metadata["phase_error_rms_rad"]),
        "static_phase_map_status": "measured_or_user_supplied" if static_phase_map_rad is not None else "none",
        "fringing_fidelity": (
            "calibration_required_direction_dependent_phase_convolution_surrogate"
            if (error.fringing_sigma_x_px or error.fringing_sigma_y_px)
            else "disabled"
        ),
    }
    return np.asarray(actual, dtype=float), metadata


def build_bench_calibrated_route(
    case_id: str,
    *,
    grid_n: int,
    calibrated: BenchCalibratedInputs,
    error_config: SystemErrorConfig = SystemErrorConfig(),
    window_m: float = 10.0e-3,
) -> dict[str, Any]:
    """Build the physically bound source-scale route for one lab state."""

    binding: BenchBindingReport = bind_calibration_to_manifest(calibrated.calibration_bundle)
    manifest = binding.manifest
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    pixel_pitch = float(hardware_value(manifest, "slm_pixel_pitch_m"))
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    f4f = float(hardware_value(manifest, "fourf_focal_length_m"))
    iris_radius = float(hardware_value(manifest, "fourier_iris_radius_m"))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))
    gamma0 = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))

    if not np.isclose(carrier, 6250.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("physical bench route must use the confirmed 20-pixel / 6.25 lp/mm carrier")

    grid = make_xy_grid(int(grid_n), float(window_m) / int(grid_n))
    shape = np.shape(grid["X"])
    beam, beam_meta = gaussian_input_field(
        grid,
        wavelength_m=wavelength,
        canonical_radius_m=beam_radius,
        error=error_config.beam,
    )
    panel = _panel_from_manifest(manifest)

    corr1 = _shape_checked_optional(calibrated.slm1_correction_phase_rad, shape, "slm1_correction_phase_rad")
    corr2 = _shape_checked_optional(calibrated.slm2_correction_phase_rad, shape, "slm2_correction_phase_rad")
    static1 = _shape_checked_optional(calibrated.slm1_static_phase_map_rad, shape, "slm1_static_phase_map_rad")
    static2 = _shape_checked_optional(calibrated.slm2_static_phase_map_rad, shape, "slm2_static_phase_map_rad")

    x1, y1 = transformed_pattern_coordinates(grid, error_config.slm1)
    desired1 = float(_ell(case_id)) * np.arctan2(y1, x1)
    if corr1 is not None:
        desired1 = desired1 + corr1
    command1 = pixelate(desired1, grid, panel)
    actual1, slm1_meta = _realised_panel_phase(
        command1,
        grid,
        error=error_config.slm1,
        pixel_pitch_m=pixel_pitch,
        calibration=calibrated.slm1_phase_calibration,
        static_phase_map_rad=static1,
    )
    slm1 = apply_slm(
        beam,
        actual1,
        grid,
        panel,
        phase_is_prepared=True,
        quantise_phase=False,
        apply_fill_factor=True,
        apply_carrier=False,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )

    x2, _ = transformed_pattern_coordinates(grid, error_config.slm2)
    desired2 = TWOPI * carrier * x2
    if corr2 is not None:
        desired2 = desired2 + corr2
    command2 = pixelate(desired2, grid, panel)
    actual2, slm2_meta = _realised_panel_phase(
        command2,
        grid,
        error=error_config.slm2,
        pixel_pitch_m=pixel_pitch,
        calibration=calibrated.slm2_phase_calibration,
        static_phase_map_rad=static2,
    )
    slm2 = apply_slm(
        slm1.total,
        actual2,
        grid,
        panel,
        phase_is_prepared=True,
        quantise_phase=False,
        apply_fill_factor=True,
        apply_carrier=False,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )

    relay = explicit_4f_relay(
        slm2.total,
        grid,
        wavelength_m=wavelength,
        nominal_focal_length_m=f4f,
        nominal_iris_radius_m=iris_radius,
        nominal_carrier_cpm=carrier,
        error=error_config.fourf,
        lens1_opd_map_m=calibrated.lens1_opd_map_m,
        lens2_opd_map_m=calibrated.lens2_opd_map_m,
    )
    X = np.asarray(grid["X"], dtype=float)
    selected_order = np.asarray(relay["output"], dtype=np.complex128) * np.exp(+1j * TWOPI * carrier * X)

    tx, ty = map(float, error_config.axicon.tilt_rad)
    if tx != 0.0 or ty != 0.0:
        geometry = calibrated.refractive_axicon_geometry
        if geometry is None:
            raise ValueError(
                "non-zero rigid axicon tilt requires measured RefractiveAxiconGeometry; "
                "the rejected thin tilted-phase surrogate is unavailable"
            )
        if int(grid_n) > 512:
            raise ValueError(
                "explicit refractive-axicon eikonal raster reference is capped at N<=512; "
                "run convergence references rather than labelling an N1536 thin surrogate as physical"
            )
        if calibrated.axicon_surface_height_error_m is not None:
            raise ValueError(
                "surface-height maps are not yet coupled into the finite-surface tilted axicon solver; "
                "do not combine two incompatible surface models"
            )
        bundle = trace_refractive_axicon_bundle(
            np.asarray(grid["X"], dtype=float),
            np.asarray(grid["Y"], dtype=float),
            geometry=geometry,
            tilt_x_rad=tx,
            tilt_y_rad=ty,
            polarization_lab=calibrated.input_polarization_lab,
        )
        reference = build_refractive_axicon_reference_field(
            selected_order,
            grid,
            bundle=bundle,
            wavelength_m=wavelength,
            output_n=int(grid_n),
            output_window_m=float(window_m),
            use_fresnel_power=calibrated.input_polarization_lab is not None,
        )
        post_axicon = np.asarray(reference.field, dtype=np.complex128)
        post_grid = dict(reference.grid)
        axicon_meta: Mapping[str, Any] = {
            "model": "explicit_two_surface_refractive_eikonal_wave_reference",
            "geometry": {
                "base_angle_rad": float(geometry.base_angle_rad),
                "clear_radius_m": float(geometry.clear_radius_m),
                "centre_thickness_m": float(geometry.centre_thickness_m),
                "edge_thickness_m": float(geometry.edge_thickness_m),
                "refractive_index": float(geometry.refractive_index),
                "external_index": float(geometry.external_index),
            },
            "tilt_rad": [tx, ty],
            "ray_metadata": dict(bundle.metadata),
            "wave_metadata": dict(reference.metadata),
        }
        tilt_status = "explicit_refractive_two_surface_reference"
    else:
        axicon_error = error_config.axicon
        if calibrated.refractive_axicon_geometry is not None and axicon_error.clear_radius_m is None:
            from dataclasses import replace
            axicon_error = replace(axicon_error, clear_radius_m=float(calibrated.refractive_axicon_geometry.clear_radius_m))
        axicon_t, axicon_meta = physical_axicon_on_own_plane(
            grid,
            wavelength_m=wavelength,
            base_angle_rad=gamma0,
            refractive_index=n_ax,
            external_index=n_ext,
            error=axicon_error,
            surface_height_error_m=calibrated.axicon_surface_height_error_m,
        )
        post_axicon = selected_order * axicon_t
        post_grid = grid
        tilt_status = "none_exact_Snell_thin_cone_morphology"

    return {
        "grid": grid,
        "post_axicon_grid": post_grid,
        "input_beam": np.asarray(beam, dtype=np.complex128),
        "post_slm1": np.asarray(slm1.total, dtype=np.complex128),
        "post_slm2": np.asarray(slm2.total, dtype=np.complex128),
        "fourier_plane_before_iris": relay["fourier_plane_before_iris"],
        "fourier_iris_mask": relay["iris_mask"],
        "post_4f_selected_order": selected_order,
        "post_axicon": np.asarray(post_axicon, dtype=np.complex128),
        "metadata": {
            "route_id": "phase2g_bench_calibrated_dual_slm_4f_axicon",
            "case_id": case_id,
            "vortex_charge": int(_ell(case_id)),
            "wavelength_m": wavelength,
            "carrier_frequency_cpm": carrier,
            "carrier_period_px": 20.0,
            "binding": {
                "calibration_id": calibrated.calibration_bundle.calibration_id,
                "data_classification": calibrated.calibration_bundle.data_classification,
                "replaced_parameters": list(binding.replaced_parameters),
                "unresolved_parameters": list(binding.unresolved_parameters),
                "absolute_bench_ready": bool(binding.absolute_bench_ready),
            },
            "beam": beam_meta,
            "slm1": {**slm1_meta, **dict(slm1.metadata)},
            "slm2": {**slm2_meta, **dict(slm2.metadata)},
            "fourf": relay["metadata"],
            "axicon": dict(axicon_meta),
            "axicon_tilt_status": tilt_status,
            "claim_policy": (
                "absolute laboratory claims require a laboratory_measurement calibration bundle, "
                "measured SLM LUTs/static corrections as applicable, calibrated observation scale, "
                "and explicit refractive axicon geometry for rigid tilt"
            ),
        },
    }


__all__ = ["BenchCalibratedInputs", "build_bench_calibrated_route"]
