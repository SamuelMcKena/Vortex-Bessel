"""Bench-calibrated serial dual-SLM vector route.

This module carries the existing six-sector radial/azimuthal vector generator
through the same calibration philosophy as the scalar/vortex Phase 2G route.
It does not replace the established vector study; it binds its physical hardware
sequence to measured SLM LUTs, measured polarization optics and the physical 4F.

Physical sequence retained from ``vector_arm_chain``::

    input Jones field
      -> SLM1 (director component only)
      -> 180-degree relay inversion
      -> HWP
      -> SLM2 (same director component)
      -> QWP
      -> explicit physical 4F / selected common diffraction order
      -> refractive axicon (exact normal-incidence Snell cone)

The two SLM blaze signs are opposite. After the relay inversion/HWP component
swap, both synthesized polarization channels carry the same SLM2-frame selected
order. The physical bench blaze is fixed at 20 pixels = 6.25 lp/mm.

Rigid axicon tilt is deliberately blocked here. The scalar two-surface
refractive reference cannot be used as if it were a full vector surface solver;
polarization transport through a tilted refractive axicon requires its own
validated vector surface-remapping implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import numpy as np

from vbb_study.calibration.bench_binding import bind_calibration_to_manifest
from vbb_study.calibration.schema import CalibrationBundle, source_at, value_at
from vbb_study.calibration.slm_phase import SLMPhaseCalibration, calibrated_phase_to_grey
from vbb_study.calibration.validation import calibration_readiness_for_claim
from vbb_study.digital_twin.nathan_vector_hexagon import (
    NathanHexagonConfig,
    canonical_target_field,
    compare_vector_fields,
    default_nathan_grid,
)
from vbb_study.digital_twin.objective_pupil_mapping import (
    ObjectivePupilMappingConfig,
    map_post_axicon_to_objective_pupil,
)
from vbb_study.digital_twin.objective_sample_route import (
    ObjectiveSampleConfig,
    ObjectiveSampleResult,
    focus_vector_pupil_into_sample,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import apply_fringing_surrogate
from vbb_study.digital_twin.vortex_error_reference_models import exact_refractive_axicon_kr_m_inv
from vbb_study.digital_twin.vortex_explicit_4f import FourFError, explicit_4f_relay
from vbb_study.digital_twin.phase2a_contracts import PHASE2A_CANONICAL_SLM_MODEL, hardware_value
from vbb_study.slm_model import apply_slm, pixelate, slm_active_aperture
from vbb_study.vbb_polarized_train import retarder_jones
from vbb_study.vector_arm_chain import gaussian_envelope, synthesise_psi1, synthesise_psi2
from vbb_study.vector_axicon import fresnel_sp_amplitudes
from vbb_study.vector_field import VectorField


TWOPI = 2.0 * np.pi
EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class BenchCalibratedVectorInputs:
    calibration_bundle: CalibrationBundle
    slm1_phase_calibration: SLMPhaseCalibration
    slm2_phase_calibration: SLMPhaseCalibration
    slm1_correction_phase_rad: np.ndarray | None = None
    slm2_correction_phase_rad: np.ndarray | None = None
    slm1_static_phase_map_rad: np.ndarray | None = None
    slm2_static_phase_map_rad: np.ndarray | None = None
    slm1_fringing_sigma_px: tuple[float, float] = (0.0, 0.0)
    slm2_fringing_sigma_px: tuple[float, float] = (0.0, 0.0)
    lens1_opd_map_m: np.ndarray | None = None
    lens2_opd_map_m: np.ndarray | None = None
    fourf_error: FourFError = FourFError()
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0)
    axicon_tilt_rad: tuple[float, float] = (0.0, 0.0)


def _required_value(bundle: CalibrationBundle, path: str) -> Any:
    value = value_at(bundle, path)
    if value in (None, ""):
        raise ValueError(f"calibrated vector route requires {path}")
    return value


def _shape_optional(values: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != route shape {shape}")
    return arr


def _director_components(
    ex: np.ndarray,
    ey: np.ndarray,
    axis_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(float(axis_rad))
    s = math.sin(float(axis_rad))
    along = c * np.asarray(ex, dtype=np.complex128) + s * np.asarray(ey, dtype=np.complex128)
    orth = -s * np.asarray(ex, dtype=np.complex128) + c * np.asarray(ey, dtype=np.complex128)
    return along, orth


def _from_director_components(
    along: np.ndarray,
    orth: np.ndarray,
    axis_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(float(axis_rad))
    s = math.sin(float(axis_rad))
    ex = c * along - s * orth
    ey = s * along + c * orth
    return np.asarray(ex, dtype=np.complex128), np.asarray(ey, dtype=np.complex128)


def _calibrated_slm_on_director(
    ex: np.ndarray,
    ey: np.ndarray,
    desired_structured_phase_rad: np.ndarray,
    grid: Mapping[str, Any],
    *,
    panel,
    director_axis_rad: float,
    calibration: SLMPhaseCalibration,
    correction_phase_rad: np.ndarray | None,
    static_phase_map_rad: np.ndarray | None,
    fringing_sigma_px: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one measured-LUT SLM to its physical director component."""

    X = np.asarray(grid["X"], dtype=float)
    phase = np.asarray(desired_structured_phase_rad, dtype=float)
    if correction_phase_rad is not None:
        phase = phase + np.asarray(correction_phase_rad, dtype=float)
    phase = phase + TWOPI * float(panel.carrier_lp_per_m) * X
    command = pixelate(phase, grid, panel)
    mask = calibrated_phase_to_grey(command, calibration)
    realised = np.asarray(mask.realised_phase_rad, dtype=float)
    if static_phase_map_rad is not None:
        realised = realised + np.asarray(static_phase_map_rad, dtype=float)
    sx, sy = map(float, fringing_sigma_px)
    realised = apply_fringing_surrogate(
        realised,
        grid,
        pixel_pitch_m=float(panel.pitch_m),
        sigma_x_px=sx,
        sigma_y_px=sy,
    )

    along, orth = _director_components(ex, ey, director_axis_rad)
    applied = apply_slm(
        along,
        realised,
        grid,
        panel,
        phase_is_prepared=True,
        quantise_phase=False,
        apply_fill_factor=True,
        apply_carrier=False,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )
    orth_out = np.where(slm_active_aperture(grid, panel), orth, 0.0j)
    ex_out, ey_out = _from_director_components(applied.total, orth_out, director_axis_rad)
    return ex_out, ey_out, {
        "panel_id": calibration.panel_id,
        "carrier_cpm": float(panel.carrier_lp_per_m),
        "carrier_period_px": float(panel.carrier_period_px),
        "lut_phase_stroke_rad": float(calibration.phase_stroke_rad),
        "lut_mask_rms_phase_error_rad": float(mask.metadata["phase_error_rms_rad"]),
        "fringing_sigma_px": [sx, sy],
        "static_phase_map": "supplied" if static_phase_map_rad is not None else "none",
        "correction_phase_map": "supplied" if correction_phase_rad is not None else "none",
        "slm_ledger": applied.ledger.as_dict(),
        "slm_metadata": dict(applied.metadata),
    }


def _input_jones_field(
    grid: Mapping[str, Any],
    vector_cfg,
    bundle: CalibrationBundle,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    angle_deg = float(_required_value(bundle, "polarization.input_linear_angle_deg"))
    relative_phase = float(_required_value(bundle, "polarization.input_relative_phase_rad"))
    dolp = float(_required_value(bundle, "polarization.input_degree_linear_polarization"))
    if dolp < 0.999:
        raise ValueError(
            "calibrated coherent vector route requires an effectively fully polarized input; "
            "partial polarization requires a coherency/Mueller propagation layer"
        )
    amp = gaussian_envelope(grid, vector_cfg)
    angle = math.radians(angle_deg)
    ex = amp * math.cos(angle)
    ey = amp * math.sin(angle) * np.exp(1j * relative_phase)
    return np.asarray(ex, dtype=np.complex128), np.asarray(ey, dtype=np.complex128), {
        "input_linear_angle_deg": angle_deg,
        "input_relative_phase_rad": relative_phase,
        "input_degree_linear_polarization": dolp,
    }


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _apply_exact_snell_vector_axicon(
    field: VectorField,
    *,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float,
    clear_radius_m: float | None,
    decentre_m: tuple[float, float],
) -> tuple[VectorField, dict[str, Any]]:
    """Normal-incidence exact-Snell thin-cone vector axicon.

    The radial field component is local p and the azimuthal component is local s.
    The shared conical phase uses the exact output direction from Snell's law,
    while the two-interface field amplitudes use the existing Fresnel reference.
    """

    grid = field.grid
    X = np.asarray(grid["X"], dtype=float) - float(decentre_m[0])
    Y = np.asarray(grid["Y"], dtype=float) - float(decentre_m[1])
    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    c = np.cos(phi)
    s = np.sin(phi)
    er = c * field.ex + s * field.ey
    et = -s * field.ex + c * field.ey

    kr = exact_refractive_axicon_kr_m_inv(
        wavelength_m=float(field.wavelength_m),
        base_angle_rad=float(base_angle_rad),
        refractive_index=float(refractive_index),
        external_index=float(external_index),
    )
    t_entry, t_p, t_s = fresnel_sp_amplitudes(
        float(refractive_index),
        float(external_index),
        float(base_angle_rad),
    )
    phase = np.exp(-1j * float(kr) * R)
    er_out = phase * t_entry * t_p * er
    et_out = phase * t_entry * t_s * et
    ex = c * er_out - s * et_out
    ey = s * er_out + c * et_out
    if clear_radius_m is not None:
        aperture = R <= float(clear_radius_m)
        ex = np.where(aperture, ex, 0.0j)
        ey = np.where(aperture, ey, 0.0j)
    out = VectorField(
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex, dtype=np.complex128),
        grid=grid,
        wavelength_m=field.wavelength_m,
        medium_index=field.medium_index,
        metadata={**dict(field.metadata), "axicon_model": "exact_Snell_normal_incidence_vector_thin_cone"},
    )
    return out, {
        "model": "exact_Snell_normal_incidence_vector_thin_cone",
        "base_angle_rad": float(base_angle_rad),
        "refractive_index": float(refractive_index),
        "external_index": float(external_index),
        "exact_kr_m_inv": float(kr),
        "clear_radius_m": None if clear_radius_m is None else float(clear_radius_m),
        "decentre_m": list(map(float, decentre_m)),
        "fresnel_field_amplitudes": {
            "t_entry": _complex_pair(t_entry),
            "t_p": _complex_pair(t_p),
            "t_s": _complex_pair(t_s),
        },
        "fresnel_amplitude_magnitudes": {
            "t_entry": float(abs(t_entry)),
            "t_p": float(abs(t_p)),
            "t_s": float(abs(t_s)),
        },
        "rigid_tilt_supported": False,
    }


def build_calibrated_segmented_vector_route(
    config: NathanHexagonConfig,
    *,
    calibrated: BenchCalibratedVectorInputs,
    grid_n: int | None = None,
) -> dict[str, Any]:
    """Build the calibrated six-sector dual-SLM vector route through the axicon."""

    bundle = calibrated.calibration_bundle
    binding = bind_calibration_to_manifest(bundle)
    vector_readiness = calibration_readiness_for_claim(bundle, "segmented_vector_hexagon")
    manifest = binding.manifest
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    f4f = float(hardware_value(manifest, "fourf_focal_length_m"))
    iris_radius = float(hardware_value(manifest, "fourier_iris_radius_m"))
    base_angle = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))
    clear_radius_raw = value_at(bundle, "axicon.clear_radius_m")
    clear_radius = None if clear_radius_raw in (None, "") else float(clear_radius_raw)

    tx, ty = map(float, calibrated.axicon_tilt_rad)
    if tx != 0.0 or ty != 0.0:
        raise ValueError(
            "calibrated vector axicon rigid tilt is blocked until a full vector two-surface "
            "refractive field-remapping solver is independently validated"
        )

    n = int(config.grid_n if grid_n is None else grid_n)
    vcfg = replace(config.vector, wavelength_m=wavelength, waist_m=beam_radius, ideal_components=False)
    if not np.isclose(vcfg.slm1.carrier_period_px, 20.0, rtol=0.0, atol=1e-12) or not np.isclose(
        vcfg.slm2.carrier_period_px, 20.0, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("calibrated vector route requires the physical 20-pixel carrier on both panels")
    if vcfg.slm1.carrier_sign != +1 or vcfg.slm2.carrier_sign != -1:
        raise RuntimeError("segmented vector route requires opposite SLM carrier signs (+1,-1)")

    grid = default_nathan_grid(replace(config, vector=vcfg, grid_n=n))
    shape = np.shape(grid["X"])
    corr1 = _shape_optional(calibrated.slm1_correction_phase_rad, shape, "slm1_correction_phase_rad")
    corr2 = _shape_optional(calibrated.slm2_correction_phase_rad, shape, "slm2_correction_phase_rad")
    static1 = _shape_optional(calibrated.slm1_static_phase_map_rad, shape, "slm1_static_phase_map_rad")
    static2 = _shape_optional(calibrated.slm2_static_phase_map_rad, shape, "slm2_static_phase_map_rad")

    for calibration, label in (
        (calibrated.slm1_phase_calibration, "SLM1"),
        (calibrated.slm2_phase_calibration, "SLM2"),
    ):
        if abs(float(calibration.wavelength_m) - wavelength) > 1.0e-9:
            raise ValueError(f"{label} phase LUT wavelength differs from route wavelength by >1 nm")

    director_deg = float(_required_value(bundle, "polarization.slm_director_axis_deg"))
    director = math.radians(director_deg)
    hwp_delta = float(_required_value(bundle, "polarization.segmented_vector_hwp_retardance_rad"))
    hwp_axis = math.radians(float(_required_value(bundle, "polarization.segmented_vector_hwp_fast_axis_deg")))
    qwp_delta = float(_required_value(bundle, "polarization.segmented_vector_qwp_retardance_rad"))
    qwp_axis = math.radians(float(_required_value(bundle, "polarization.segmented_vector_qwp_fast_axis_deg")))

    ex0, ey0, input_meta = _input_jones_field(grid, vcfg, bundle)
    psi1 = synthesise_psi1(vcfg, grid)
    ex1, ey1, slm1_meta = _calibrated_slm_on_director(
        ex0,
        ey0,
        psi1,
        grid,
        panel=vcfg.slm1,
        director_axis_rad=director,
        calibration=calibrated.slm1_phase_calibration,
        correction_phase_rad=corr1,
        static_phase_map_rad=static1,
        fringing_sigma_px=calibrated.slm1_fringing_sigma_px,
    )

    ex1 = np.flip(ex1, axis=(0, 1))
    ey1 = np.flip(ey1, axis=(0, 1))
    ex_hwp, ey_hwp = retarder_jones(ex1, ey1, hwp_delta, hwp_axis)

    psi2 = synthesise_psi2(vcfg, grid)
    ex2, ey2, slm2_meta = _calibrated_slm_on_director(
        ex_hwp,
        ey_hwp,
        psi2,
        grid,
        panel=vcfg.slm2,
        director_axis_rad=director,
        calibration=calibrated.slm2_phase_calibration,
        correction_phase_rad=corr2,
        static_phase_map_rad=static2,
        fringing_sigma_px=calibrated.slm2_fringing_sigma_px,
    )
    ex_qwp, ey_qwp = retarder_jones(ex2, ey2, qwp_delta, qwp_axis)
    pre4f = VectorField(
        ex=ex_qwp,
        ey=ey_qwp,
        ez=np.zeros_like(ex_qwp, dtype=np.complex128),
        grid=grid,
        wavelength_m=wavelength,
        medium_index=1.0,
        metadata={"stage": "after_calibrated_QWP_before_physical_4F"},
    )

    selected_carrier = float(vcfg.slm2.carrier_lp_per_m)
    relay_x = explicit_4f_relay(
        pre4f.ex,
        grid,
        wavelength_m=wavelength,
        nominal_focal_length_m=f4f,
        nominal_iris_radius_m=iris_radius,
        nominal_carrier_cpm=selected_carrier,
        error=calibrated.fourf_error,
        lens1_opd_map_m=calibrated.lens1_opd_map_m,
        lens2_opd_map_m=calibrated.lens2_opd_map_m,
    )
    relay_y = explicit_4f_relay(
        pre4f.ey,
        grid,
        wavelength_m=wavelength,
        nominal_focal_length_m=f4f,
        nominal_iris_radius_m=iris_radius,
        nominal_carrier_cpm=selected_carrier,
        error=calibrated.fourf_error,
        lens1_opd_map_m=calibrated.lens1_opd_map_m,
        lens2_opd_map_m=calibrated.lens2_opd_map_m,
    )
    X = np.asarray(grid["X"], dtype=float)
    demod = np.exp(+1j * TWOPI * selected_carrier * X)
    post4f = VectorField(
        ex=np.asarray(relay_x["output"], dtype=np.complex128) * demod,
        ey=np.asarray(relay_y["output"], dtype=np.complex128) * demod,
        ez=np.zeros_like(X, dtype=np.complex128),
        grid=grid,
        wavelength_m=wavelength,
        medium_index=1.0,
        metadata={"stage": "physical_4F_selected_order_demodulated"},
    )

    post_axicon, axicon_meta = _apply_exact_snell_vector_axicon(
        post4f,
        base_angle_rad=base_angle,
        refractive_index=n_ax,
        external_index=n_ext,
        clear_radius_m=clear_radius,
        decentre_m=calibrated.axicon_decentre_m,
    )
    target = canonical_target_field(replace(config, vector=vcfg, grid_n=n), grid=grid)

    pre4f_demod = VectorField(
        ex=pre4f.ex * np.exp(-1j * TWOPI * selected_carrier * X),
        ey=pre4f.ey * np.exp(-1j * TWOPI * selected_carrier * X),
        ez=pre4f.ez,
        grid=grid,
        wavelength_m=wavelength,
        medium_index=1.0,
    )
    encoder_comparison = compare_vector_fields(pre4f_demod, target)

    return {
        "grid": grid,
        "input_field": VectorField(ex0, ey0, grid=grid, wavelength_m=wavelength),
        "post_slm1_relay": VectorField(ex1, ey1, grid=grid, wavelength_m=wavelength),
        "post_hwp": VectorField(ex_hwp, ey_hwp, grid=grid, wavelength_m=wavelength),
        "post_slm2": VectorField(ex2, ey2, grid=grid, wavelength_m=wavelength),
        "pre_4f_vector": pre4f,
        "post_4f_selected_order": post4f,
        "post_axicon": post_axicon,
        "target": target,
        "metadata": {
            "route_id": "phase2g_calibrated_segmented_vector_dual_slm",
            "calibration_id": bundle.calibration_id,
            "data_classification": bundle.data_classification,
            "core_manifest_absolute_ready": bool(binding.absolute_bench_ready),
            "segmented_vector_readiness": {
                "ready": bool(vector_readiness.ready),
                "status": vector_readiness.status,
                "missing_measurements": list(vector_readiness.missing_measurements),
                "non_calibrated_measurements": list(vector_readiness.non_calibrated_measurements),
            },
            "absolute_segmented_vector_comparison_ready": bool(vector_readiness.ready and not bundle.is_synthetic),
            "unresolved_manifest_parameters": list(binding.unresolved_parameters),
            "carrier_period_px": 20.0,
            "selected_common_carrier_cpm": selected_carrier,
            "input": input_meta,
            "slm_director_axis_deg": director_deg,
            "slm1": slm1_meta,
            "slm2": slm2_meta,
            "hwp": {
                "retardance_rad": hwp_delta,
                "fast_axis_rad": hwp_axis,
                "sources": [
                    source_at(bundle, "polarization.segmented_vector_hwp_retardance_rad"),
                    source_at(bundle, "polarization.segmented_vector_hwp_fast_axis_deg"),
                ],
            },
            "qwp": {
                "retardance_rad": qwp_delta,
                "fast_axis_rad": qwp_axis,
                "sources": [
                    source_at(bundle, "polarization.segmented_vector_qwp_retardance_rad"),
                    source_at(bundle, "polarization.segmented_vector_qwp_fast_axis_deg"),
                ],
            },
            "fourf_x": dict(relay_x["metadata"]),
            "fourf_y": dict(relay_y["metadata"]),
            "axicon": axicon_meta,
            "encoder_target_comparison": {
                "complex_overlap": float(encoder_comparison.complex_overlap),
                "normalized_rms_error": float(encoder_comparison.normalized_rms_error),
                "stokes_rms_error": float(encoder_comparison.stokes_rms_error),
                "angle_rms_rad": float(encoder_comparison.angle_rms_rad),
                "power_ratio": float(encoder_comparison.power_ratio),
            },
            "claim_policy": (
                "absolute segmented-vector bench prediction requires measured SLM LUTs, input polarization, "
                "HWP/QWP state, 4F geometry, objective mapping, camera calibration and axicon geometry; "
                "non-zero refractive axicon tilt remains blocked for vector claims"
            ),
        },
    }


def calibrated_vector_route_to_sample(
    route: Mapping[str, Any],
    *,
    mapping_config: ObjectivePupilMappingConfig,
    objective_config: ObjectiveSampleConfig,
) -> dict[str, Any]:
    """Map a calibrated post-axicon vector field through objective and sample."""

    field = route["post_axicon"]
    if not isinstance(field, VectorField):
        raise TypeError("route['post_axicon'] must be a VectorField")
    mx = map_post_axicon_to_objective_pupil(
        field.ex,
        field.grid,
        wavelength_m=field.wavelength_m,
        config=mapping_config,
    )
    my = map_post_axicon_to_objective_pupil(
        field.ey,
        field.grid,
        wavelength_m=field.wavelength_m,
        config=mapping_config,
    )
    if not np.allclose(mx.grid["x"], my.grid["x"], rtol=0.0, atol=1e-15):
        raise RuntimeError("vector component objective mappings produced inconsistent grids")
    focused: ObjectiveSampleResult = focus_vector_pupil_into_sample(
        mx.field,
        my.field,
        mx.grid,
        config=objective_config,
    )
    return {
        "objective_pupil_ex": mx.field,
        "objective_pupil_ey": my.field,
        "objective_pupil_grid": mx.grid,
        "objective_mapping_ex": mx.metadata,
        "objective_mapping_ey": my.metadata,
        "sample_result": focused,
        "metadata": {
            "route": "calibrated_segmented_vector_post_axicon_to_vector_Debye_sample",
            "spatially_varying_vector_pupil_preserved": True,
        },
    }


__all__ = [
    "BenchCalibratedVectorInputs",
    "build_calibrated_segmented_vector_route",
    "calibrated_vector_route_to_sample",
]
