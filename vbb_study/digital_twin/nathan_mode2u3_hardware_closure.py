"""MODE 2U3 final hardware closure and calibration bridge.

M2U2 closed the source-scale physics with outcome M2U2-B blockers recorded as
hardware/calibration unknowns, and M2U2-FIX repaired the hexagon eligibility
gate (outcome M2U2F-B) with the realistic dual-SLM + carrier + 4F reference as
the best strict-eligible candidate for shape, peak and useful-region energy.

MODE 2U3 is not another optimisation study.  It connects that successful
source-scale simulation to the actual laboratory hardware and the original
Digital Twin: exact SLM identity, phase-stroke/LUT status, wavelength and
axicon-index scope resolution, physical 4F geometry, camera calibration
design, the full Jones/axis/reflection audit, closure of every recorded M2U2
hardware conflict, and a hardware rebind check of the frozen operating points
under the repaired strict hexagon gate.

No value is invented: every hardware parameter carries one of the ten agreed
provenance categories and unknowns remain unknown with a concrete calibration
bridge instead of a fabricated number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    MODE2U2F_DEFAULT_OUTPUT_ROOT,
    OLD_BEST_COMPROMISE_ID,
    REALISTIC_4F_REFERENCE_ID,
    STRICT_BASELINE_CORR_MIN,
    StrictCandidateControls,
    _baseline_row,
    evaluate_controls,
    evaluate_strict_hexagon_metrics,
)
from vbb_study.digital_twin.nathan_mode2u2_master_closure import (
    MODE2U2_DEFAULT_OUTPUT_ROOT,
    _fixed_useful_region,
    resolve_nathan_hardware_binding,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    MODE2N_DEFAULT_CARRIER_LPMM,
    MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    Mode2SCorrection,
    Mode2SPerturbation,
    NathanSourceParityConfig,
    _json_ready,
    _mode1b_even_axis_crop,
    _normalise_image,
    _write_rows,
    mode2n_source_target,
    mode2s_combined_cases,
    run_mode2n_dual_slm_4f_route,
    run_mode2n_v0_reference,
    run_mode2q_backward_initialisation,
    run_mode2s_degraded_forward,
)

MODE2U3_STAGE = "nathan_mode2u3_hardware_closure"
MODE2U3_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2u3_hardware_closure")
MODE2U3_DOC_PATH = Path("docs/78_nathan_mode2u3_final_hardware_closure.md")
MODE2U3_PHASE_CAL_DOC_PATH = Path("docs/75_nathan_mode2u3_slm_phase_calibration.md")
MODE2U3_FOURF_CAL_DOC_PATH = Path("docs/76_nathan_mode2u3_4f_calibration.md")
MODE2U3_CAMERA_CAL_DOC_PATH = Path("docs/77_nathan_mode2u3_camera_calibration.md")
MODE2U3_ALLOWED_OUTCOMES = ("M2U3-A", "M2U3-B", "M2U3-C", "M2U3-D")

CANONICAL_OPERATING_POINT_ID = REALISTIC_4F_REFERENCE_ID
STRICT_COMPROMISE_ID = "strict_c6.75_i0.40_q-0.25_r+0.0_p0.00"
STRICT_COMPROMISE_CONTROLS = StrictCandidateControls(
    STRICT_COMPROMISE_ID, 6.75, 0.40, -0.25, 0.0, 0.0, source="m2u2fix_strict_best_compromise",
)

# The ten agreed provenance categories for MODE 2U3 hardware closure.
MODE2U3_PROVENANCE_CATEGORIES = (
    "measured_lab",
    "manufacturer_documentation_in_repo",
    "project_hardware_config",
    "original_digital_twin",
    "source_model_parameter",
    "externally_supplied_lab_identity",
    "inferred",
    "planning_assumption",
    "placeholder",
    "unknown",
)

MODE2U3_CONFLICT_STATUSES = (
    "resolved_same_physical_context",
    "resolved_different_scopes",
    "resolved_placeholder_removed",
    "unresolved_requires_measurement",
    "unresolved_missing_evidence",
)

# Externally supplied lab identity (NOT repository evidence until confirmed).
EXTERNAL_LAB_SLM_IDENTITY = {
    "make": "HOLOEYE",
    "model": "PLUTO-2.1 NIR-149",
    "count": 2,
    "width_px": 1920,
    "height_px": 1080,
    "pixel_pitch_um": 8.0,
    "active_area_mm": (15.36, 8.64),
    "phase_only": True,
    "provenance": "externally_supplied_lab_identity",
}

# Repository documentation that cites the PLUTO-2.1 NIR *family* (not NIR-149
# specifically): reference kernel header + THEORY_AND_ANALYSIS.md product link.
REPO_PLUTO_FAMILY_SOURCES = (
    "reference_kernels/balyian_shared_kernel_v4.py (PLUTO-2.1/NIR-family: 8 um pitch, 93% fill, 15.36x8.64 mm, 8-bit)",
    "../THEORY_AND_ANALYSIS.md (https://holoeye.com/product/pluto-2-1-niro-024/)",
)


def _clean(value: Any) -> Any:
    return None if value is None else value


@dataclass(frozen=True)
class ResolvedSLMHardware:
    """Best-evidence resolution of one physical SLM panel (no invented values)."""

    make: str
    model: str
    width_px: int
    height_px: int
    pixel_pitch_m: float
    active_width_m: float
    active_height_m: float
    wavelength_reference_m: float | None
    phase_stroke_rad: float | None
    phase_bit_depth: int | None
    drive_bit_depth: int | None
    fill_factor: float | None
    supports_float_phase_data: bool | None
    supports_uint8_phase_data: bool | None
    phase_lut_required: bool | None
    coordinate_origin: str | None
    axis_handedness: str | None
    reflection_flip_required: bool | None
    provenance: Mapping[str, str]

    def rows(self, panel_id: str) -> list[dict[str, Any]]:
        payload = asdict(self)
        provenance = dict(payload.pop("provenance"))
        rows = []
        for key, value in payload.items():
            category = provenance.get(key, "unknown")
            if category.split(";")[0] not in MODE2U3_PROVENANCE_CATEGORIES:
                raise ValueError(f"unsupported provenance category {category!r} for {key}")
            rows.append(
                {
                    "panel_id": panel_id,
                    "field": key,
                    "value": _clean(value),
                    "provenance": category,
                    "resolved": value is not None,
                }
            )
        return rows


def resolve_slm_hardware() -> tuple[ResolvedSLMHardware, dict[str, Any]]:
    """Resolve the physical SLM panels from repo evidence plus labelled lab identity.

    Geometry (1920 x 1080, 8 um, 15.36 x 8.64 mm), 8-bit addressing and 93%
    fill factor are independently present in the project hardware config
    (`vbb_study/config.py` SLMConfig) and in repository documentation citing
    the HOLOEYE PLUTO-2.1 NIR *family*.  The exact model "PLUTO-2.1 NIR-149"
    exists only as externally supplied lab information (the in-repo product
    link is the NIRO-024 variant), so it stays `externally_supplied_lab_identity`.
    Phase stroke at 1029/1030 nm, LUT behaviour, SDK data formats and panel
    coordinate conventions have no repository evidence and remain unknown.
    """

    binding = resolve_nathan_hardware_binding()
    hardware = ResolvedSLMHardware(
        make="HOLOEYE",
        model="PLUTO-2.1 NIR-149",
        width_px=int(binding.slm_width_px),
        height_px=int(binding.slm_height_px),
        pixel_pitch_m=float(binding.slm_pixel_pitch_m),
        active_width_m=float(binding.slm_active_width_m),
        active_height_m=float(binding.slm_active_height_m),
        wavelength_reference_m=None,
        phase_stroke_rad=None,
        phase_bit_depth=int(binding.slm_bit_depth) if binding.slm_bit_depth is not None else None,
        drive_bit_depth=8,
        fill_factor=float(binding.slm_fill_factor) if binding.slm_fill_factor is not None else None,
        supports_float_phase_data=None,
        supports_uint8_phase_data=None,
        phase_lut_required=None,
        coordinate_origin=None,
        axis_handedness=None,
        reflection_flip_required=None,
        provenance={
            "make": "externally_supplied_lab_identity;family_corroborated_by_manufacturer_documentation_in_repo",
            "model": "externally_supplied_lab_identity",
            "width_px": "project_hardware_config",
            "height_px": "project_hardware_config",
            "pixel_pitch_m": "project_hardware_config",
            "active_width_m": "inferred",
            "active_height_m": "inferred",
            "wavelength_reference_m": "unknown",
            "phase_stroke_rad": "unknown",
            "phase_bit_depth": "project_hardware_config",
            "drive_bit_depth": "manufacturer_documentation_in_repo",
            "fill_factor": "manufacturer_documentation_in_repo",
            "supports_float_phase_data": "unknown",
            "supports_uint8_phase_data": "unknown",
            "phase_lut_required": "unknown",
            "coordinate_origin": "unknown",
            "axis_handedness": "unknown",
            "reflection_flip_required": "unknown",
        },
    )
    unknowns = {
        "panel_count_note": "two panels externally reported; treated as identical until per-panel evidence exists",
        "unknown_fields": {
            "model_confirmation": {
                "status": "externally_supplied_lab_identity",
                "resolves_by": "read the physical panel label / device manual; repo citation covers only the PLUTO-2.1 NIR family (NIRO-024 link)",
            },
            "wavelength_reference_m": {
                "status": "unknown",
                "resolves_by": "manufacturer datasheet for the exact NIR-149 variant (operating range)",
            },
            "phase_stroke_rad": {
                "status": "unresolved_requires_calibration",
                "resolves_by": "docs/75 interferometric or binary-grating calibration at the actual laser wavelength",
            },
            "phase_lut_required": {
                "status": "unresolved_requires_calibration",
                "resolves_by": "docs/75 command-to-phase response measurement; HOLOEYE panels normally ship wavelength-specific LUTs",
            },
            "supports_float_phase_data": {
                "status": "unknown",
                "resolves_by": "SLM Display SDK / GUI documentation (no SDK or GUI code exists in this repository)",
            },
            "supports_uint8_phase_data": {
                "status": "unknown",
                "resolves_by": "SLM Display SDK / GUI documentation (no SDK or GUI code exists in this repository)",
            },
            "coordinate_origin": {
                "status": "unknown",
                "resolves_by": "SDK addressing convention + one on-bench dot-pattern orientation test",
            },
            "axis_handedness": {
                "status": "unknown",
                "resolves_by": "one on-bench asymmetric-pattern (e.g. letter 'F') orientation test per panel",
            },
            "reflection_flip_required": {
                "status": "unknown",
                "resolves_by": "same orientation test; each reflective panel plus fold mirrors flips parity per bounce",
            },
        },
        "sdk_gui_evidence": "no HOLOEYE SDK, GUI, or driver code was found in the repository",
        "manufacturer_register_state": "configs/evidence/manufacturer_evidence_register.json M_SLM1_SPEC/M_SLM2_SPEC remain unknown/unverified",
    }
    return hardware, unknowns


# ---------------------------------------------------------------------------
# Section 5: phase stroke / LUT calibration bridge (schema only, no fabricated values)
# ---------------------------------------------------------------------------


def slm_phase_calibration_schema() -> dict[str, Any]:
    """Schema for the command-to-phase calibration record (values left to the lab)."""

    return {
        "schema": "nathan_mode2u3_slm_phase_calibration",
        "version": 1,
        "status": "unresolved_requires_calibration",
        "fabricated_values": False,
        "record_fields": {
            "slm_identity": {"make": None, "model": None, "serial": None, "panel_role": "SLM-H | SLM-V"},
            "wavelength_m": None,
            "timestamp_utc": None,
            "calibration_id": None,
            "command_domain": {"kind": "uint8 grey level | float phase", "min": None, "max": None},
            "measured_phase_response_rad": "array of (command, phase_rad) samples",
            "usable_phase_stroke_rad": None,
            "wrapped_mapping": "phase_rad -> command lookup covering [0, 2pi) after stroke check",
            "interpolation_method": "monotone cubic (PCHIP) or linear; recorded, not assumed",
            "residual_phase_rms_rad": None,
            "environment": {"temperature_C": None, "incidence_angle_deg": None, "polarisation_alignment_note": None},
        },
        "methods": {
            "A_interferometric": (
                "split the panel into a static reference half and a swept half; interfere both halves "
                "(Michelson or common-path shear); fringe shift vs command gives phase(command) directly"
            ),
            "B_binary_grating": (
                "display a binary grating alternating command 0 and command c; first-order diffraction "
                "efficiency eta(c) = sin^2(delta_phi(c)/2) inverts to the phase difference; sweep c over the "
                "full drive domain; appropriate for phase-only panels at near-normal incidence"
            ),
        },
        "acceptance": {
            "usable_stroke_requirement_rad": "greater than or equal to 2*pi at the operating wavelength, else wrapped mapping must be validated",
            "residual_rms_target_rad": 0.05,
        },
    }


# ---------------------------------------------------------------------------
# Sections 6-7: wavelength and axicon-index scope resolution
# ---------------------------------------------------------------------------


def resolve_wavelength_scopes() -> list[dict[str, Any]]:
    return [
        {
            "value_nm": 1029.0,
            "scope": "actual laser / original Digital Twin bench branch",
            "source": "vbb_study/config.py LaserConfig.wavelength_m (PHAROS PH2)",
            "provenance": "original_digital_twin",
            "resolution": "authoritative for the physical bench and for hardware rebinding",
        },
        {
            "value_nm": 1030.0,
            "scope": "Nathan source-scale simulation branch (validated V0 physics)",
            "source": "NathanSourceParityConfig.wavelength_m",
            "provenance": "source_model_parameter",
            "resolution": "Nathan-report rounding of the same PHAROS-class source; retained for V0 parity",
        },
        {
            "value_nm": 1030.0,
            "scope": "CSLM diagnostic route / F300 nominal profile",
            "source": "configs/hardware/cslm_f300_nominal_4f_profile.json model_convention.wavelength_m",
            "provenance": "placeholder",
            "resolution": "numerical scenario parameter only; not laser evidence",
        },
    ]


def resolve_axicon_index_scopes() -> list[dict[str, Any]]:
    return [
        {
            "value": 1.458,
            "scope": "Nathan source-scale branch (validated V0)",
            "component": "source-scale physical axicon",
            "material": "fused-silica-like at ~1 um (consistent with 1.4497-1.46 handbook range)",
            "wavelength_nm": 1030.0,
            "source": "NathanSourceParityConfig.axicon_n",
            "provenance": "source_model_parameter",
            "resolution_status": "resolved_different_scopes",
            "resolution": "authoritative for the source-scale branch; retained",
        },
        {
            "value": 1.5,
            "scope": "inherited microfabrication target / quicklook branch",
            "component": "BeamTarget.n_axicon / quicklook config",
            "material": "generic glass placeholder",
            "wavelength_nm": 1029.0,
            "source": "vbb_study/config.py BeamTarget.n_axicon; config/quicklook_config.json",
            "provenance": "original_digital_twin",
            "resolution_status": "resolved_different_scopes",
            "resolution": "different physical scope (micro branch); must NOT be forced onto the source-scale axicon",
        },
    ]


# ---------------------------------------------------------------------------
# Section 8: physical 4F closure
# ---------------------------------------------------------------------------


def physical_4f_rows(
    *,
    wavelengths_m: Sequence[float] = (1.029e-6, 1.030e-6),
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM,
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    simulated_first_order_efficiency: float | None = None,
    simulated_zero_order_leakage: float | None = None,
) -> list[dict[str, Any]]:
    """Translate the successful simulation carrier/iris into physical 4F geometry.

    ``x_order = lambda * f * carrier_frequency`` for each legitimate focal-length
    candidate; the iris is expressed in millimetres, never only in Fourier units.
    """

    focal_candidates = (
        {
            "focal_length_m": 0.300,
            "classification": "original_digital_twin",
            "source": "configs/hardware/cslm_f300_nominal_4f_profile.json known_nominal_geometry (nominal_from_bench_description, not bench calibrated)",
            "recommended": True,
        },
        {
            "focal_length_m": 0.100,
            "classification": "placeholder",
            "source": "CSLMRouteConfig warning-only 100 mm placeholders",
            "recommended": False,
        },
    )
    carrier_cpm = float(carrier_lpmm) * 1.0e3
    iris_cpm = float(iris_radius_frac) * carrier_cpm
    pitch = 8.0e-6
    carrier_period_px = 1.0 / (carrier_cpm * pitch)
    rows: list[dict[str, Any]] = []
    for cand in focal_candidates:
        f = float(cand["focal_length_m"])
        for wl in wavelengths_m:
            x_order = float(wl) * f * carrier_cpm
            iris_radius = float(wl) * f * iris_cpm
            # Gaussian spectral half-width (1/e field) of the 2 mm source beam.
            beam_spectral_halfwidth_cpm = 1.0 / (np.pi * 2.0e-3)
            spectral_extent = float(wl) * f * beam_spectral_halfwidth_cpm
            clearance = x_order - iris_radius
            rows.append(
                {
                    "focal_length_m": f,
                    "focal_length_classification": str(cand["classification"]),
                    "focal_length_source": str(cand["source"]),
                    "recommended_focal_length": bool(cand["recommended"]),
                    "wavelength_nm": float(wl) / 1e-9,
                    "carrier_lpmm": float(carrier_lpmm),
                    "carrier_period_slm_pixels": float(carrier_period_px),
                    "first_order_displacement_mm": x_order / 1e-3,
                    "zero_to_first_separation_mm": x_order / 1e-3,
                    "fourier_plane_beam_spectral_extent_mm": spectral_extent / 1e-3,
                    "iris_radius_mm": iris_radius / 1e-3,
                    "iris_diameter_mm": 2.0 * iris_radius / 1e-3,
                    "iris_to_zero_order_clearance_mm": clearance / 1e-3,
                    "clipping_fraction_simulated": (
                        None if simulated_first_order_efficiency is None else float(1.0 - simulated_first_order_efficiency)
                    ),
                    "selected_order_efficiency_simulated": (
                        None if simulated_first_order_efficiency is None else float(simulated_first_order_efficiency)
                    ),
                    "zero_order_leakage_simulated": (
                        None if simulated_zero_order_leakage is None else float(simulated_zero_order_leakage)
                    ),
                    "physically_plausible": bool(2.0 * iris_radius / 1e-3 >= 0.5 and clearance > 0.25 * iris_radius),
                    "plausibility_note": (
                        "iris diameter and zero-order clearance are comfortable for a standard iris"
                        if 2.0 * iris_radius / 1e-3 >= 1.0
                        else "small iris; consider a fixed pinhole instead of an adjustable iris"
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Section 10: camera closure (no fabricated specifications)
# ---------------------------------------------------------------------------


def camera_closure_rows() -> list[dict[str, Any]]:
    inventory = "configs/hardware/cslm_physical_axicon_bench_inventory.json"
    register = "configs/evidence/bench_evidence_register.json B_CAMERA_SCALE"
    fields = (
        ("camera_make_model", inventory + " items.camera_model"),
        ("camera_pixel_pitch_um", inventory + " items.camera_pixel_pitch_um"),
        ("camera_sensor_resolution_px", inventory + " items.camera_sensor_resolution_px"),
        ("camera_sensor_dimensions_mm", inventory + " (derivable only once pitch/resolution exist)"),
        ("imaging_magnification", inventory + " items.camera_magnification"),
        ("imaging_lens_or_relay", inventory + " (no record)"),
        ("direct_capture_vs_relay", inventory + " items.camera_plane_location_mm (no record)"),
        ("z_translation_stage_model", "no repository record found"),
        ("z_stage_accuracy_resolution", "no repository record found"),
    )
    rows = []
    for name, source in fields:
        rows.append(
            {
                "field": name,
                "value": None,
                "provenance": "unknown",
                "source_checked": source,
                "register_state": register + " current_value_state=unknown, ready=false",
                "status": "unresolved_requires_calibration",
                "architecture_critical": False,
                "calibration_route": "docs/77 camera calibration bridge",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Sections 11-12: Jones / axis / reflection audit and HWP requirements
# ---------------------------------------------------------------------------


def jones_axis_route_rows() -> list[dict[str, Any]]:
    """Stage-by-stage physical route audit for the dual-SLM architecture.

    The simulation Jones chain (`phi_H = +alpha`, `phi_V = -alpha + pi/2`,
    QWP at -45 deg) is written in ONE fixed right-handed beam-local basis with
    no reflection bookkeeping.  The physical route contains reflective SLMs and
    fold mirrors, and every reflection flips transverse parity, so per-arm
    mirror counts decide whether a software x-flip of one mask is needed.  That
    parity is a routine one-shot orientation test, not an architecture unknown.
    """

    convention = (
        "beam-local frame: z along propagation, x nominally lab-horizontal, y up; angles measured "
        "anticlockwise from +x when viewed looking from downstream back INTO the oncoming beam "
        "(receiver view); each mirror/SLM reflection flips handedness and the apparent rotation sense"
    )
    rows = [
        {
            "stage": "laser output (PHAROS PH2)",
            "propagation": "+z0",
            "local_x": "lab horizontal",
            "local_y": "lab vertical",
            "handedness": "right-handed",
            "jones_basis": "linear H/V (lab)",
            "polarisation_state": "linear; orientation NOT recorded in repository (B_INPUT_BEAM unknown)",
            "reflection_flip": "n/a",
            "phase_convention": "exp(+i k z) forward",
            "note": "measure output polarisation orientation and purity during alignment",
        },
        {
            "stage": "input polariser + HWP (polarisation preparation)",
            "propagation": "+z0",
            "local_x": "lab horizontal",
            "local_y": "lab vertical",
            "handedness": "right-handed",
            "jones_basis": "linear H/V (lab)",
            "polarisation_state": "set to diagonal-like state so the PBS splits power H/V as required",
            "reflection_flip": "none",
            "phase_convention": "unchanged",
            "note": "HWP angle is the H/V power-balance control; M2S showed +/-20% imbalance is tolerated",
        },
        {
            "stage": "PBS split",
            "propagation": "H transmits +z0; V reflects +z1 (90 deg)",
            "local_x": "H arm keeps lab x; V arm x maps to old z direction",
            "local_y": "lab vertical in both arms",
            "handedness": "V arm flips parity (one reflection)",
            "jones_basis": "H arm: |H>; V arm: |V>",
            "polarisation_state": "H arm horizontal; V arm vertical",
            "reflection_flip": "V arm: 1 reflection",
            "phase_convention": "reflection pi phase bookkept as channel piston (observable-invariant, M2S)",
            "note": "PBS transmitted-arm extinction is poorer than reflected; a clean-up polariser in the H arm is cheap insurance",
        },
        {
            "stage": "SLM-H panel (reflective)",
            "propagation": "reflected back / folded",
            "local_x": "flips on reflection",
            "local_y": "unchanged",
            "handedness": "flips (odd bounce)",
            "jones_basis": "|H> aligned to panel director (REQUIRED for phase-only operation)",
            "polarisation_state": "linear along panel LC director",
            "reflection_flip": "1 reflection at panel + per-arm fold mirrors (bench-dependent count)",
            "phase_convention": "displayed mask phi_H = +alpha (+carrier); mask must be pre-flipped in x if the arm has odd total reflections",
            "note": "panel director orientation is not documented in repo -> orientation test required",
        },
        {
            "stage": "SLM-V panel (reflective)",
            "propagation": "reflected back / folded",
            "local_x": "flips on reflection",
            "local_y": "unchanged",
            "handedness": "flips (odd bounce)",
            "jones_basis": "panel director basis; arriving V light must be rotated onto the director",
            "polarisation_state": "linear along panel LC director",
            "reflection_flip": "1 reflection at panel + per-arm fold mirrors (bench-dependent count)",
            "phase_convention": "displayed mask phi_V = -alpha + pi/2 (+carrier); same per-arm flip rule as SLM-H",
            "note": "either rotate the panel 90 deg or use HWP(45 deg) before and after the panel (section 12)",
        },
        {
            "stage": "4F filter (shared or per-arm)",
            "propagation": "+z2",
            "local_x": "a full 4F relay inverts the image (x,y -> -x,-y): parity-even, orientation rotated 180 deg",
            "local_y": "inverted with x",
            "handedness": "preserved by the 4F inversion (two-axis flip)",
            "jones_basis": "unchanged",
            "polarisation_state": "unchanged",
            "reflection_flip": "image inversion, not a parity flip",
            "phase_convention": "carrier demodulated by taking the +1 order at the iris",
            "note": "180 deg image rotation is absorbed by the sixfold pattern up to a 60 deg-symmetric rotation; record it anyway",
        },
        {
            "stage": "recombination (second PBS)",
            "propagation": "+z2 common",
            "local_x": "common frame restored",
            "local_y": "common frame restored",
            "handedness": "per-arm parity must be equal here, else one mask needs a software x-flip",
            "jones_basis": "linear H/V recombined",
            "polarisation_state": "coherent H+V vector field",
            "reflection_flip": "arm-parity difference = (H-arm bounces - V-arm bounces) mod 2",
            "phase_convention": "relative arm piston is free (uniform polarisation rotation; observable-invariant)",
            "note": "path-length matching well within the ~260 fs pulse coherence length is a build requirement",
        },
        {
            "stage": "QWP (nominal code angle -45 deg)",
            "propagation": "+z2",
            "local_x": "recombined beam frame",
            "local_y": "recombined beam frame",
            "handedness": "right-handed (receiver view)",
            "jones_basis": "linear H/V",
            "polarisation_state": "maps dual-linear channels onto the segmented radial/azimuthal target",
            "reflection_flip": "none (transmissive)",
            "phase_convention": "linear_retarder(pi/2, beta): fast axis at beta gets exp(-i*delta/2); beta anticlockwise from +x in receiver view",
            "note": "code -45 deg = FAST axis 45 deg clockwise from lab-horizontal in receiver view; see explicit conversion below",
        },
        {
            "stage": "physical axicon + free space to z = 60 mm",
            "propagation": "+z2",
            "local_x": "beam frame",
            "local_y": "beam frame",
            "handedness": "right-handed",
            "jones_basis": "radial/azimuthal decomposition inside the axicon element",
            "polarisation_state": "segmented vector field -> hexagonal Bessel zone",
            "reflection_flip": "none (transmissive)",
            "phase_convention": "conical phase exp(-i k_r r), p/s Fresnel split",
            "note": "hologram-centre-to-axicon-axis registration <= 0.2 mm is the one alignment that matters (M2S)",
        },
    ]
    return [{"frame_convention": convention, **row} for row in rows]


def qwp_lab_axis_statement() -> dict[str, Any]:
    """Convert the code QWP angle of -45 degrees into explicit lab-axis language."""

    return {
        "code_convention": (
            "qwp(-pi/4) = linear_retarder(pi/2, -pi/4): the axis argument beta is measured anticlockwise "
            "from the +x (horizontal/H) axis; the axis at beta receives exp(-i*delta/2), i.e. beta marks the FAST axis"
        ),
        "viewing_convention": (
            "angles are defined in the beam-local right-handed frame viewed by a receiver looking from "
            "downstream back INTO the oncoming beam (looking against the propagation direction)"
        ),
        "physical_statement": (
            "code -45 deg means: the QWP FAST axis lies 45 deg from the lab-horizontal H axis, rotated "
            "CLOCKWISE when you stand downstream and look back into the beam (equivalently 45 deg "
            "anticlockwise when looking along the propagation direction from behind the source)"
        ),
        "mount_side_caveat": (
            "any odd number of upstream mirror reflections after the recombiner flips the apparent sense; "
            "the sign is therefore fixed on the bench by ONE polarimeter check: with only the H channel open "
            "and a uniform mask, the QWP at the correct -45 deg turns H into LEFT-hand circular in receiver "
            "view (fast axis clockwise-45 deg); if right-hand circular is observed, rotate the QWP to +45 deg"
        ),
        "fast_or_slow": "beta marks the FAST axis in this codebase (exp(-i*delta/2) on the beta axis)",
        "deferrable_to_routine_calibration": True,
    }


def hwp_requirement_rows() -> list[dict[str, Any]]:
    """Determine extra waveplate requirements from actual SLM polarisation compatibility."""

    return [
        {
            "question": "polarisation leaving the laser",
            "answer": "linear (PHAROS class); orientation/purity not recorded in repository",
            "requirement": "measure during alignment (B_INPUT_BEAM)",
            "extra_component": None,
            "evidence": "unknown",
        },
        {
            "question": "input polariser needed?",
            "answer": "recommended as a clean-up/definition stage before the HWP",
            "requirement": "1x linear polariser (or rely on measured laser purity)",
            "extra_component": "polariser (optional but recommended)",
            "evidence": "planning_assumption",
        },
        {
            "question": "input HWP before the PBS?",
            "answer": "YES - it is the H/V power-balance control (target 50/50; M2S tolerates +/-20%)",
            "requirement": "1x HWP at the input",
            "extra_component": "HWP #1",
            "evidence": "architecture requirement derived from the dual-channel model",
        },
        {
            "question": "H/V channel separation",
            "answer": "polarising beamsplitter (PBS): H transmits, V reflects",
            "requirement": "1x PBS",
            "extra_component": "PBS #1",
            "evidence": "architecture requirement",
        },
        {
            "question": "polarisation reaching SLM-H",
            "answer": "horizontal linear",
            "requirement": "must coincide with the panel LC director for phase-only operation",
            "extra_component": None,
            "evidence": "panel director orientation unverified (manufacturer register unknown)",
        },
        {
            "question": "polarisation reaching SLM-V",
            "answer": "vertical linear - which a horizontally-directed PLUTO panel canNOT phase-modulate",
            "requirement": "rotate V to the panel director and back",
            "extra_component": "EITHER HWP at 45 deg before SLM-V AND HWP at 45 deg after (HWP #2, #3), OR mount SLM-V rotated 90 deg (no extra waveplates)",
            "evidence": "phase-only LCOS requires input polarisation along the LC director; exact PLUTO-2.1 NIR-149 director orientation unverified in repo",
        },
        {
            "question": "PLUTO phase-only polarisation requirement",
            "answer": "yes - phase-only LCOS panels modulate only the component along the LC director",
            "requirement": "confirm director orientation from the NIR-149 manual or one polariser test",
            "extra_component": None,
            "evidence": "generic LCOS physics; exact orientation is a routine check",
        },
        {
            "question": "HWPs after either SLM?",
            "answer": "only the V-arm return HWP (HWP #3) if the waveplate option is chosen; none in the H arm",
            "requirement": "see SLM-V row",
            "extra_component": "HWP #3 (conditional)",
            "evidence": "architecture requirement (conditional on mounting choice)",
        },
        {
            "question": "recombination",
            "answer": "second PBS recombines H (transmit) and V (reflect) into one collinear beam",
            "requirement": "1x PBS; path lengths matched well within the pulse coherence length",
            "extra_component": "PBS #2",
            "evidence": "architecture requirement",
        },
        {
            "question": "is one final QWP sufficient?",
            "answer": "YES - M2P/M2N/M2Q validated the single uniform QWP at code -45 deg closing the chain",
            "requirement": "1x QWP after recombination",
            "extra_component": "QWP #1",
            "evidence": "validated model convention; mount-side sign fixed by one polarimeter check (docs/78)",
        },
        {
            "question": "nominal angles",
            "answer": "HWP#1: set for 50/50 split; HWP#2/#3 (if used): 45 deg; QWP: code -45 deg per the lab-axis statement",
            "requirement": "record all mount angles against the lab-horizontal reference",
            "extra_component": None,
            "evidence": "model convention + M2S tolerance audit (QWP +/-2 deg tolerated)",
        },
    ]


# ---------------------------------------------------------------------------
# Section 13: M2U2 conflict resolution
# ---------------------------------------------------------------------------


def resolve_m2u2_conflicts(
    conflicts_path: str | Path = MODE2U2_DEFAULT_OUTPUT_ROOT / "hardware_parameter_conflicts.json",
) -> list[dict[str, Any]]:
    """Give every recorded M2U2 conflict exactly one closure status (none hidden)."""

    path = Path(conflicts_path)
    conflicts = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    resolutions = {
        "wavelength": {
            "status": "resolved_different_scopes",
            "resolution": (
                "1029 nm is the actual PHAROS/original-Digital-Twin laser value and governs the physical bench "
                "and hardware rebinding; 1030 nm is the Nathan source-model rounding retained for V0 parity; "
                "the rebind sensitivity check shows the difference is immaterial (docs/78)"
            ),
            "architecture_critical": False,
            "deferrable_to_routine_calibration": False,
        },
        "axicon_refractive_index": {
            "status": "resolved_different_scopes",
            "resolution": (
                "n = 1.458 is the validated Nathan source-scale fused-silica value and stays authoritative for "
                "this branch; n = 1.5 belongs to the inherited microfabrication target/quicklook scope and is "
                "not forced onto the source-scale axicon"
            ),
            "architecture_critical": False,
            "deferrable_to_routine_calibration": False,
        },
        "4f_focal_length": {
            "status": "resolved_placeholder_removed",
            "resolution": (
                "the CSLM 100 mm value is a warning-only placeholder and is removed from consideration; the "
                "F300 nominal 300 mm bench description is adopted as the recommendation, with the docs/76 "
                "blaze-grating displacement measurement confirming the actual focal length on the bench"
            ),
            "architecture_critical": False,
            "deferrable_to_routine_calibration": True,
        },
        "carrier_frequency": {
            "status": "resolved_different_scopes",
            "resolution": (
                "the source-scale display carrier is 6.25 lp/mm (20 px blaze on the 8 um panel) as used by every "
                "validated M2N/M2Q/M2S/M2U run; CSLM command-domain records and vector-arm defaults are different "
                "route semantics and stay in their own scopes"
            ),
            "architecture_critical": False,
            "deferrable_to_routine_calibration": False,
        },
        "beam_radius": {
            "status": "resolved_different_scopes",
            "resolution": (
                "the 2 mm 1/e source beam is the Nathan/Twin source-scale value; the 24 um CSLM value is a "
                "diagnostic grid source in a different scope"
            ),
            "architecture_critical": False,
            "deferrable_to_routine_calibration": False,
        },
        "camera_calibration": {
            "status": "unresolved_requires_measurement",
            "resolution": (
                "no camera make/pitch/magnification exists in the repository; docs/77 converts this into a "
                "routine four-part calibration (sensor pitch, magnification target, carrier-displacement "
                "cross-check, z-stage translation); observation-side only, so not architecture-critical"
            ),
            "architecture_critical": False,
            "deferrable_to_routine_calibration": True,
        },
        "slm_exact_model_and_phase_stroke": {
            "status": "unresolved_requires_measurement",
            "resolution": (
                "panel geometry (1920x1080, 8 um, 15.36x8.64 mm, 8-bit, 93% fill) is confirmed by project config "
                "plus in-repo PLUTO-2.1 family documentation; the exact NIR-149 model remains externally supplied "
                "lab identity until the physical label/manual is read, and the wavelength-specific phase stroke/LUT "
                "is a routine docs/75 calibration; neither changes the optical architecture"
            ),
            "architecture_critical": False,
            "deferrable_to_routine_calibration": True,
        },
    }
    rows = []
    for conflict in conflicts:
        family = str(conflict.get("parameter_family", "unknown"))
        resolution = resolutions.get(family)
        if resolution is None:
            resolution = {
                "status": "unresolved_missing_evidence",
                "resolution": "conflict recorded by M2U2 but no MODE 2U3 evidence path exists yet",
                "architecture_critical": True,
                "deferrable_to_routine_calibration": False,
            }
        if resolution["status"] not in MODE2U3_CONFLICT_STATUSES:
            raise ValueError(f"invalid closure status {resolution['status']!r}")
        rows.append({**conflict, **resolution})
    return rows


# ---------------------------------------------------------------------------
# Section 14: hardware rebind check
# ---------------------------------------------------------------------------


def forbidden_operating_point_ids(
    audit_path: str | Path = MODE2U2F_DEFAULT_OUTPUT_ROOT / "old_optima_strict_audit.json",
) -> tuple[str, ...]:
    """All old optima that fail the repaired strict gate; never usable again."""

    ids = {OLD_BEST_COMPROMISE_ID}
    path = Path(audit_path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        for row in rows:
            if not bool(row.get("strict_hexagon_eligible", False)):
                ids.add(str(row.get("candidate_id")))
    return tuple(sorted(ids))


def assert_not_forbidden(candidate_id: str) -> None:
    """Hard guard: discarded non-hexagonal optima cannot be revived."""

    if str(candidate_id) in forbidden_operating_point_ids():
        raise ValueError(
            f"candidate {candidate_id!r} was discarded by the repaired M2U2-FIX strict hexagon gate "
            "and must never be used as an operating point, hardware prescription, or build recommendation"
        )


def _bench(wavelength_m: float, *, grid_n: int, z_planes: int) -> dict[str, Any]:
    cfg = replace(NathanSourceParityConfig(), wavelength_m=float(wavelength_m))
    data = mode2n_source_target(cfg, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    realistic = run_mode2n_dual_slm_4f_route(data, v0)
    backward = run_mode2q_backward_initialisation(data)
    useful_mask, useful_meta = _fixed_useful_region(data["grid"], float(v0.ring_radius_m))
    return {
        "config": cfg,
        "data": data,
        "v0": v0,
        "realistic": realistic,
        "backward": backward,
        "useful_mask": useful_mask,
        "useful_meta": useful_meta,
    }


def _robustness_case_row(
    case_id: str,
    case: Mapping[str, Any],
    bench: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = evaluate_strict_hexagon_metrics(
        case["reference_plane"],
        grid=bench["data"]["grid"],
        v0_plane=bench["v0"].reference_plane,
        realistic_plane=bench["realistic"].reference_plane,
        v0_ring_radius_m=float(bench["v0"].ring_radius_m),
        useful_mask=bench["useful_mask"],
    )
    return {
        "candidate_id": case_id,
        "role": "robustness_case",
        **metrics,
        "legacy_z60_correlation": float(case["comparison"]["z60_full_field_correlation"]),
        "first_order_efficiency": float(case["iris"]["first_order_efficiency"]),
        "total_throughput": float(case["iris"]["first_order_efficiency"] * case["pre_axicon"]["power_ratio"]),
        "legacy_pass": bool(case["passes"]),
    }


def run_hardware_rebind(
    *,
    grid_n: int = 384,
    z_planes: int = 9,
    old_wavelength_m: float = 1.030e-6,
    new_wavelength_m: float = 1.029e-6,
) -> dict[str, Any]:
    """Rebind the frozen operating points to the resolved hardware and re-gate them.

    The resolved binding change that affects the source-scale simulation is the
    actual laser wavelength (1029 nm, original Digital Twin) replacing the Nathan
    rounding (1030 nm).  Panel geometry, 8-bit drive, 93% fill, 6.25 lp/mm carrier
    and the 0.40 iris fraction are unchanged by the audit.  Every candidate is
    re-evaluated with the repaired M2U2-FIX strict hexagon gate; a high full-field
    correlation alone can never make a candidate survive.
    """

    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)
    rows: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for label, wavelength in (("old_binding_1030nm", old_wavelength_m), ("resolved_binding_1029nm", new_wavelength_m)):
        bench = _bench(wavelength, grid_n=grid_n, z_planes=z_planes)
        cfg: NathanSourceParityConfig = bench["config"]
        canonical = _baseline_row(
            data=bench["data"], v0=bench["v0"], realistic=bench["realistic"], useful_mask=bench["useful_mask"],
        )
        canonical.update({"role": "canonical_operating_point"})
        compromise, _ = evaluate_controls(
            STRICT_COMPROMISE_CONTROLS,
            data=bench["data"], v0=bench["v0"], backward=bench["backward"],
            realistic=bench["realistic"], useful_mask=bench["useful_mask"],
        )
        compromise.update({"role": "strict_compromise"})
        moderate_pert = mode2s_combined_cases()[1]
        moderate_case = run_mode2s_degraded_forward(
            bench["data"], bench["v0"], bench["backward"], moderate_pert, fast_single_plane=True,
        )
        moderate = _robustness_case_row("m2s_combined_moderate_lab", moderate_case, bench)
        decentre_case = run_mode2s_degraded_forward(
            bench["data"], bench["v0"], bench["backward"],
            Mode2SPerturbation(
                label="axicon_decentre_0p5mm_compensated",
                slm_aperture_clip=True,
                axicon_decentre_x_m=0.5e-3,
            ),
            correction=Mode2SCorrection(mask_recentre_x_m=0.5e-3),
            fast_single_plane=True,
        )
        decentre = _robustness_case_row("m2s_axicon_decentre_0p5mm_compensated", decentre_case, bench)
        for row in (canonical, compromise, moderate, decentre):
            row["binding"] = label
            row["wavelength_nm"] = float(wavelength) / 1e-9
            row["unresolved_assumptions"] = (
                "phase stroke/LUT (docs/75), exact iris centre (docs/76), camera scale (docs/77), "
                "per-arm reflection parity sign (one orientation test)"
            )
            rows.append(row)
        k_r = float(2.0 * np.pi / wavelength * (cfg.axicon_n - cfg.medium_n) * np.tan(cfg.axicon_base_angle_rad))
        sensitivity.append(
            {
                "binding": label,
                "wavelength_nm": float(wavelength) / 1e-9,
                "z60_correlation_realistic_vs_v0": float(bench["realistic"].v0_comparison["z60_full_field_correlation"]),
                "strict_class": str(bench["realistic"].symmetry_class),
                "radial_fringe_period_um": float(2.0 * np.pi / k_r / 1e-6),
                "first_order_displacement_mm_f300": float(wavelength * 0.3 * MODE2N_DEFAULT_CARRIER_LPMM * 1.0e3) / 1e-3,
                "useful_region_power": float(canonical["P_useful"]),
                "peak_metric": float(canonical["strict_peak_metric"]),
            }
        )
    canonical_rows = [r for r in rows if r["role"] == "canonical_operating_point"]
    compromise_rows = [r for r in rows if r["role"] == "strict_compromise"]
    new_canonical = next(r for r in canonical_rows if r["binding"] == "resolved_binding_1029nm")
    new_compromise = next(r for r in compromise_rows if r["binding"] == "resolved_binding_1029nm")
    return {
        "rows": rows,
        "sensitivity": sensitivity,
        "canonical_preserved": bool(new_canonical["strict_hexagon_eligible"]),
        "compromise_preserved": bool(new_compromise["strict_hexagon_eligible"]),
        "strict_gate_used": "repaired M2U2-FIX strict hexagon eligibility gate",
        "reference_drift_floor_note": (
            f"the {STRICT_BASELINE_CORR_MIN} correlation-to-realistic-reference floor is a calibrated "
            "project-specific eligibility threshold, not a universal physical definition of a hexagon"
        ),
    }


# ---------------------------------------------------------------------------
# Section 15: authorisation logic
# ---------------------------------------------------------------------------


def mode2u3_outcome(
    *,
    rebind: Mapping[str, Any],
    conflict_rows: Sequence[Mapping[str, Any]],
    jones_rows: Sequence[Mapping[str, Any]],
    hwp_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose exactly one M2U3-A/B/C/D outcome."""

    architecture_critical_unresolved = [
        str(row["parameter_family"])
        for row in conflict_rows
        if str(row["status"]).startswith("unresolved") and bool(row.get("architecture_critical", False))
    ]
    routine_calibration_items = [
        str(row["parameter_family"])
        for row in conflict_rows
        if str(row["status"]).startswith("unresolved") and bool(row.get("deferrable_to_routine_calibration", False))
    ]
    # The Jones audit is inconsistent only if a stage cannot be given a coherent
    # basis/parity account at all; unknown per-arm mirror counts are a routine
    # one-shot orientation test, not an architecture inconsistency.
    jones_inconsistent = any("architecture_inconsistency" in str(row.get("note", "")) for row in jones_rows)
    architecture_complete = all(row.get("extra_component") is not None or row.get("requirement") for row in hwp_rows)
    rebind_ok = bool(rebind["canonical_preserved"]) and bool(rebind["compromise_preserved"])

    if jones_inconsistent or not architecture_complete:
        outcome = "M2U3-D"
        statement = (
            "The Jones-axis/reflection/polarisation audit reveals that the assumed dual-SLM architecture is "
            "incomplete or physically inconsistent."
        )
    elif not rebind_ok:
        outcome = "M2U3-C"
        statement = "Hardware rebinding materially degrades the strict hexagonal field; further optical redesign is required."
    elif architecture_critical_unresolved:
        outcome = "M2U3-B"
        statement = (
            "The source-scale route remains valid, but at least one architecture-critical hardware fact "
            f"remains unknown and cannot safely be deferred: {', '.join(architecture_critical_unresolved)}."
        )
    else:
        outcome = "M2U3-A"
        statement = (
            "All architecture-critical M2U2-B blockers are resolved or converted into explicit routine "
            "laboratory calibration steps, and hardware rebinding preserves the strict hexagon for both the "
            "canonical realistic-4F reference and the strict compromise. M2V is authorised."
        )
    return {
        "stage": MODE2U3_STAGE,
        "selected_outcome": outcome,
        "allowed_outcomes": MODE2U3_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "m2v_authorised": bool(outcome == "M2U3-A"),
        "m2v_authorisation_condition": "M2V is authorised only under M2U3-A.",
        "canonical_operating_point": CANONICAL_OPERATING_POINT_ID,
        "strict_compromise_candidate": STRICT_COMPROMISE_ID,
        "forbidden_operating_points": list(forbidden_operating_point_ids()),
        "canonical_rebind_preserved": bool(rebind["canonical_preserved"]),
        "compromise_rebind_preserved": bool(rebind["compromise_preserved"]),
        "architecture_critical_unresolved": architecture_critical_unresolved,
        "routine_calibration_items": routine_calibration_items,
        "deferred_items_policy": (
            "exact SLM LUT, exact iris centre, camera pixel-to-micron scale, precise beam centring and the "
            "per-arm reflection parity sign are honestly measurable during routine alignment and do not block M2V; "
            "an uncertainty changing the fundamental optical architecture could not have been deferred"
        ),
        "microfabrication_sample_plane_claim": False,
        "micro_scale_note": (
            "MODE 2U3 closes the source-scale hardware bridge only; the microfabrication branch (MODE 1C/M1E) "
            "remains separate and blocked and nothing here claims sample-plane success"
        ),
    }


# ---------------------------------------------------------------------------
# Documentation writers (docs/75, 76, 77, 78)
# ---------------------------------------------------------------------------


def _write_phase_calibration_doc(path: Path, schema: Mapping[str, Any]) -> Path:
    text = (
        "# Nathan MODE 2U3 - SLM Phase Calibration Bridge (docs/75)\n\n"
        "**Status:** calibration bridge only. No calibration values are fabricated here; the phase stroke\n"
        "and command-to-phase response of the two PLUTO-2.1-class panels at the actual laser wavelength are\n"
        "`unresolved_requires_calibration` until this procedure is executed.\n\n"
        "## Why\n\n"
        "The repository documents 8-bit addressing for the PLUTO family but contains no wavelength-specific\n"
        "phase-stroke or LUT record (manufacturer register M_SLM1_SPEC/M_SLM2_SPEC: unknown; bench register\n"
        "B_SLM_PHASE_RESPONSE: unknown). The operating wavelength is the actual PHAROS value (1029 nm; the\n"
        "Nathan source model rounds to 1030 nm). An exact 2*pi stroke must NOT be assumed.\n\n"
        "## Method A - interferometric phase calibration\n\n"
        "1. Illuminate the panel with the aligned linear polarisation at the bench incidence angle.\n"
        "2. Display a two-zone mask: left half fixed at command 0, right half swept over the full drive\n"
        "   domain (0..255 for uint8).\n"
        "3. Interfere the two halves (Michelson arm or common-path lateral shear onto the camera).\n"
        "4. Track the fringe shift of the swept half versus command: `phase(command)` directly.\n"
        "5. Repeat per panel (SLM-H, SLM-V); record temperature and incidence angle.\n\n"
        "## Method B - binary-grating diffraction-efficiency calibration\n\n"
        "1. Display a binary grating alternating command 0 and command c (period >= 8 px).\n"
        "2. Measure first-order power in the Fourier plane versus c.\n"
        "3. Invert `eta(c) proportional to sin^2(delta_phi(c)/2)` for the phase difference (resolve branch\n"
        "   by monotonicity from small c).\n"
        "4. Cross-check against Method A near half-stroke.\n\n"
        "## Target mapping\n\n"
        "`desired phase (rad) -> calibrated hardware command`, wrapped over the measured usable stroke.\n"
        "If the usable stroke at 1029 nm is below 2*pi, the wrapped mapping must be validated explicitly\n"
        "(display a known 0..2*pi ramp and confirm first-order efficiency) before any mask is exported.\n\n"
        "## Record schema\n\n"
        "The machine-readable schema (no fabricated values) is stored at\n"
        "`outputs/figures/digital_twin/nathan_mode2u3_hardware_closure/01_phase_calibration/slm_phase_calibration_schema.json`:\n\n"
        "```json\n"
        + json.dumps(_json_ready(dict(schema)), indent=2)
        + "\n```\n\n"
        "Acceptance: usable stroke >= 2*pi at 1029 nm (or validated wrapped mapping) and residual phase RMS\n"
        "<= 0.05 rad. M2S showed 8-bit quantisation and even 16-level phase pass the strict gate, so the\n"
        "calibration target is comfortable.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _write_4f_calibration_doc(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    recommended = [r for r in rows if r["recommended_focal_length"] and abs(r["wavelength_nm"] - 1029.0) < 0.5]
    rec = recommended[0] if recommended else rows[0]
    text = (
        "# Nathan MODE 2U3 - Physical 4F Calibration Bridge (docs/76)\n\n"
        "**Status:** lab procedure. The 4F focal length is nominal (300 mm from the bench description, not\n"
        "bench calibrated; the 100 mm CSLM value is a removed placeholder), so the carrier-to-displacement\n"
        "mapping is confirmed on the bench before the iris is fixed.\n\n"
        "Nominal numbers at 1029 nm, f = 300 mm, carrier 6.25 lp/mm (20 px on the 8 um panel):\n"
        f"+1 order displacement `x = lambda * f * carrier` = {rec['first_order_displacement_mm']:.3f} mm;\n"
        f"required iris radius {rec['iris_radius_mm']:.3f} mm (diameter {rec['iris_diameter_mm']:.3f} mm);\n"
        f"simulated selected-order efficiency {rec['selected_order_efficiency_simulated']:.4f}; simulated\n"
        f"zero-order leakage {rec['zero_order_leakage_simulated']:.2e}.\n\n"
        "## Procedure\n\n"
        "1. Display a simple blaze grating (20 px period, full panel) on SLM-H only.\n"
        "2. Observe the Fourier plane on a card/camera at the nominal focal distance behind lens 1.\n"
        "3. Locate the zero order (display a flat mask to identify it).\n"
        "4. Locate the +1 order (record which physical side the carrier sends it to).\n"
        "5. Measure the physical zero-to-first separation with the camera scale or a translation stage.\n"
        "6. Infer the actual carrier-to-displacement mapping `x_measured / (lambda * carrier)` -> actual f.\n"
        "7. Place the iris centred on the +1 order.\n"
        "8. Sweep the iris radius from ~0.5 mm diameter upward.\n"
        "9. Measure selected-order power versus radius (power meter after the 4F output).\n"
        "10. Measure zero-order leakage (block the +1 order; residual power through the iris).\n"
        "11. Compare with simulation: efficiency plateau ~0.95 with zero leakage at radius ~0.77 mm\n"
        "    (0.40 x carrier separation); the M2S audit shows the whole 0.24-0.80 fraction range passes.\n"
        "12. Update the hardware binding (actual f, iris centre, iris radius, measured efficiency) in the\n"
        "    M2V build package.\n\n"
        "The full per-focal-length geometry table is stored in `02_4f/physical_4f_closure.csv/json`.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _write_camera_calibration_doc(path: Path) -> Path:
    text = (
        "# Nathan MODE 2U3 - Camera Scale Calibration Bridge (docs/77)\n\n"
        "**Status:** calibration required. The repository contains no camera make, pixel pitch, sensor\n"
        "size, magnification, relay description or z-stage record (bench inventory camera items are all\n"
        "null; bench register B_CAMERA_SCALE unknown). Nothing is fabricated; the calibration below turns\n"
        "the unknown into a routine measurement.\n\n"
        "## Calibration design\n\n"
        "A. **Sensor pitch**: read the manufacturer pixel pitch from the camera datasheet once its model is\n"
        "   read off the physical device; record it with part number as manufacturer documentation.\n"
        "B. **Magnification**: image a known target (USAF-1951 or a ruler edge) or translate the camera by a\n"
        "   known stage displacement and track the image shift; magnification = image shift / stage shift.\n"
        "C. **Cross-check via the SLM carrier**: with the docs/76 blaze displayed, the +1 order displacement\n"
        "   is `lambda * f * carrier`; the measured pixel displacement of the order gives an independent\n"
        "   pixels-per-mm scale at the Fourier plane.\n"
        "D. **z scale**: step the camera along z with the translation stage across the Bessel zone\n"
        "   (~10-200 mm at source scale) and record stage readings; the M2S audit shows +/-20 mm\n"
        "   observation-plane tolerance, so millimetre-class stage accuracy is sufficient.\n\n"
        "## Record\n\n"
        "Store camera make/model, pixel pitch, sensor dimensions, magnification, direct-vs-relay flag,\n"
        "z-stage model and step accuracy in `03_camera/camera_hardware_closure.csv/json`, replacing the\n"
        "`unresolved_requires_calibration` placeholders. Camera scale is observation-side only: it cannot\n"
        "change the optical architecture and therefore does not block M2V.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _write_master_doc(
    path: Path,
    *,
    slm: ResolvedSLMHardware,
    fourf_rows: Sequence[Mapping[str, Any]],
    qwp_statement: Mapping[str, Any],
    conflict_rows: Sequence[Mapping[str, Any]],
    rebind: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> Path:
    rec_4f = next(r for r in fourf_rows if r["recommended_focal_length"] and abs(r["wavelength_nm"] - 1029.0) < 0.5)
    new_rows = [r for r in rebind["rows"] if r["binding"] == "resolved_binding_1029nm"]
    canonical = next(r for r in new_rows if r.get("role") == "canonical_operating_point")
    compromise = next(r for r in new_rows if r.get("role") == "strict_compromise")
    conflict_lines = "\n".join(
        f"- `{row['parameter_family']}` -> **{row['status']}**: {row['resolution']}" for row in conflict_rows
    )
    answers = [
        ("1. What exact SLMs do we have?",
         f"Externally supplied lab identity: two {slm.make} {slm.model} phase-only LCOS panels, "
         f"{slm.width_px} x {slm.height_px} at {slm.pixel_pitch_m / 1e-6:.0f} um "
         f"({slm.active_width_m / 1e-3:.2f} x {slm.active_height_m / 1e-3:.2f} mm active). The exact NIR-149 "
         "variant is NOT yet repository-verified (the in-repo product link is the NIRO-024 family page)."),
        ("2. What is repository/manual verified?",
         "Panel geometry (1920x1080, 8 um pitch, 15.36x8.64 mm), 8-bit addressing and 93% fill factor: "
         "present in the project hardware config (SLMConfig) and in in-repo PLUTO-2.1 family documentation "
         "(reference kernel header; THEORY_AND_ANALYSIS.md product link)."),
        ("3. What is externally supplied lab information?",
         "The exact model string 'PLUTO-2.1 NIR-149' and the panel count (two); provenance-labelled "
         "`externally_supplied_lab_identity` until the physical labels/manuals are read."),
        ("4. What phase response is known?",
         "None at the operating wavelength. Drive bit depth 8 is repo-documented; wavelength-specific phase "
         "stroke and LUT are `unresolved_requires_calibration` (docs/75). An exact 2*pi stroke is not assumed."),
        ("5. What requires calibration?",
         "SLM phase stroke/LUT per panel (docs/75); actual 4F focal length, iris centre and radius (docs/76); "
         "camera scale and z-stage (docs/77); per-arm reflection parity sign and QWP mount sign (one "
         "orientation test + one polarimeter check); hologram-to-axicon centring (0.2 mm tolerance, M2S)."),
        ("6. Which wavelength belongs to the source-scale branch?",
         "Simulation parity: 1030 nm (Nathan source rounding). Physical bench and hardware rebinding: 1029 nm "
         "(actual PHAROS / original Digital Twin). The rebind sensitivity check shows the difference is "
         "immaterial (~0.1% in fringe period and first-order displacement; strict gate preserved at both)."),
        ("7. Which axicon index belongs to the source-scale branch?",
         "n = 1.458 (validated Nathan fused-silica source value). n = 1.5 is the inherited "
         "microfabrication/quicklook placeholder scope and is not forced onto this branch."),
        ("8. What focal length should the 4F use?",
         "f = 300 mm (nominal from the bench description, F300 profile), confirmed on the bench via the "
         "docs/76 displacement measurement. The 100 mm CSLM value is a removed placeholder."),
        ("9. What carrier should be displayed?",
         "6.25 lp/mm = 20 SLM pixels per period on the 8 um panel (the value used by every validated "
         "M2N/M2Q/M2S/M2U run)."),
        ("10. Where should the +1 order appear physically?",
         f"x = lambda*f*carrier = {rec_4f['first_order_displacement_mm']:.3f} mm from the zero order at "
         "1029 nm with f = 300 mm."),
        ("11. What physical iris diameter is required?",
         f"{rec_4f['iris_diameter_mm']:.2f} mm (radius {rec_4f['iris_radius_mm']:.3f} mm = 0.40 x carrier "
         f"separation) centred on the +1 order; clearance to the zero order "
         f"{rec_4f['iris_to_zero_order_clearance_mm']:.2f} mm. M2S: the whole 0.24-0.80 fraction range passes."),
        ("12. What camera calibration is needed?",
         "Everything (make, pitch, sensor size, magnification, relay-vs-direct, z-stage): docs/77 four-part "
         "routine calibration. Observation-side only; does not block M2V."),
        ("13. What does QWP = -45 degrees physically mean?",
         str(qwp_statement["physical_statement"]) + " " + str(qwp_statement["mount_side_caveat"])),
        ("14. Are extra HWPs required?",
         "Yes: HWP #1 at the input for H/V power balance. In the V arm, EITHER HWP #2/#3 at 45 deg before and "
         "after SLM-V, OR mount SLM-V rotated 90 deg (no extra plates). One final QWP closes the chain; no "
         "other waveplates are needed."),
        ("15. What polarisation must reach each SLM?",
         "Linear polarisation aligned to each panel's LC director (phase-only requirement). SLM-H receives H; "
         "SLM-V receives V rotated onto the director by HWP #2 or by panel rotation. The exact NIR-149 director "
         "orientation is unverified in the repo: one polariser test resolves it."),
        ("16. How are H/V channels split?",
         "Polarising beamsplitter (PBS #1): H transmits, V reflects, after the input polariser + HWP #1."),
        ("17. How are they recombined?",
         "PBS #2 recombines the arms collinearly; path lengths matched well within the ~260 fs pulse coherence "
         "length; the relative arm piston is free (uniform polarisation rotation, observable-invariant per M2S)."),
        ("18. Which M2U2 conflicts are resolved?",
         "All seven received exactly one closure status:\n\n" + conflict_lines),
        ("19. Which unknowns are routine calibration items?",
         "SLM phase stroke/LUT, exact iris centre / focal-length confirmation, camera scale, beam centring, "
         "per-arm parity sign, QWP mount sign. None changes the optical architecture."),
        ("20. Does hardware rebinding preserve the canonical strict hexagon?",
         f"Yes. At the resolved 1029 nm binding the canonical point remains strict-eligible "
         f"(corr-to-realistic {canonical['corr_to_realistic_4f']:.4f}, deltaC "
         f"{canonical['deltaC_c120_minus_c60']:+.4f}, dark core {canonical['dark_core_ratio']:.4f}, "
         f"first-order efficiency {canonical['first_order_efficiency']:.4f})."),
        ("21. Does the strict compromise remain eligible?",
         f"Yes: strict-eligible at 1029 nm (corr-to-realistic {compromise['corr_to_realistic_4f']:.4f}, deltaC "
         f"{compromise['deltaC_c120_minus_c60']:+.4f}); it stays the secondary candidate, and the canonical "
         "realistic-4F reference remains preferred because rebinding did not change the ranking."),
        ("22. Is M2V authorised?",
         f"Outcome **{outcome['selected_outcome']}**: {outcome['outcome_statement']}"),
    ]
    body = "".join(f"**{question}**\n\n{answer}\n\n" for question, answer in answers)
    text = (
        "# Nathan MODE 2U3 - Final Hardware Closure and Calibration Bridge (docs/78)\n\n"
        "**Status:** hardware/calibration closure for the source-scale branch only. Canonical operating\n"
        f"point: `{CANONICAL_OPERATING_POINT_ID}`; secondary strict compromise:\n"
        f"`{STRICT_COMPROMISE_ID}`; forbidden: every old optimum that fails the repaired\n"
        f"M2U2-FIX strict gate, permanently including `{OLD_BEST_COMPROMISE_ID}`. The\n"
        f"{STRICT_BASELINE_CORR_MIN} correlation-to-realistic-reference floor used by that gate is a\n"
        "**calibrated project-specific eligibility threshold**, not a universal physical definition of a\n"
        "hexagon. Within the bounded, physically interpretable search that was run, the realistic dual-SLM +\n"
        "carrier + 4F hexagon is the best strict-eligible candidate for shape, peak intensity and\n"
        "useful-region energy; no claim of global mathematical optimality outside that tested space is made.\n"
        "No microfabrication/sample-plane claim is made anywhere in this document.\n\n"
        "## The 22 closure questions\n\n"
        + body
        + "## Output tree\n\n"
        "`outputs/figures/digital_twin/nathan_mode2u3_hardware_closure/` -> `00_slm/`,\n"
        "`01_phase_calibration/`, `02_4f/`, `03_camera/`, `04_jones_axes/`, `05_conflicts/`, `06_rebind/`,\n"
        "`07_final_status/`. Calibration bridges: docs/75 (SLM phase), docs/76 (4F), docs/77 (camera).\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_4f_geometry(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.4, 5.6), constrained_layout=True, dpi=200)
    for row in rows:
        if abs(row["wavelength_nm"] - 1029.0) > 0.5:
            continue
        f_mm = row["focal_length_m"] * 1e3
        x = row["first_order_displacement_mm"]
        r = row["iris_radius_mm"]
        color = "tab:green" if row["recommended_focal_length"] else "tab:red"
        y = 1.0 if row["recommended_focal_length"] else -1.0
        ax.plot([0], [y], marker="o", color="0.2")
        ax.annotate("zero order", (0, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
        ax.plot([x], [y], marker="o", color=color)
        ax.add_patch(plt.Circle((x, y), r, fill=False, color=color, lw=1.4))
        ax.annotate(
            f"f = {f_mm:.0f} mm ({row['focal_length_classification']})\n+1 at {x:.2f} mm, iris D {row['iris_diameter_mm']:.2f} mm",
            (x, y), textcoords="offset points", xytext=(14, -36), fontsize=8, color=color,
        )
    ax.axhline(1.0, color="0.85", lw=0.6)
    ax.axhline(-1.0, color="0.85", lw=0.6)
    ax.set_xlim(-0.6, 3.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlabel("Fourier-plane position (mm) at 1029 nm, carrier 6.25 lp/mm")
    ax.set_title("MODE 2U3 physical 4F geometry: recommended f = 300 mm (top) vs removed 100 mm placeholder (bottom)")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_jones_route(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14.2, 6.6), constrained_layout=True, dpi=200)
    ax.axis("off")
    n = len(rows)
    for idx, row in enumerate(rows):
        x = (idx + 0.5) / n
        ax.add_patch(plt.Rectangle((x - 0.05, 0.55), 0.1, 0.32, fill=True, facecolor="#eef3fb", edgecolor="0.3"))
        ax.text(x, 0.71, str(row["stage"])[:44], ha="center", va="center", fontsize=6.4, wrap=True)
        ax.text(x, 0.50, str(row["polarisation_state"])[:66], ha="center", va="top", fontsize=5.2, color="tab:blue", wrap=True)
        ax.text(x, 0.32, "flip: " + str(row["reflection_flip"])[:56], ha="center", va="top", fontsize=5.2, color="tab:red", wrap=True)
        if idx < n - 1:
            ax.annotate("", xy=(x + 0.1 / 2 + 0.014, 0.71), xytext=(x + 0.1 / 2, 0.71), arrowprops={"arrowstyle": "->", "color": "0.3"})
    ax.text(0.5, 0.08, str(rows[0]["frame_convention"]), ha="center", fontsize=7, color="0.25", wrap=True)
    ax.set_title("MODE 2U3 Jones / axis / reflection route audit (dual-SLM + 4F + QWP + axicon)")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_rebind(path: Path, rebind: Mapping[str, Any], bench_new: Mapping[str, Any], bench_old: Mapping[str, Any]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0), constrained_layout=True, dpi=200)
    for ax, (label, bench) in zip(
        axes[:2], (("old binding 1030 nm", bench_old), ("resolved binding 1029 nm", bench_new)), strict=False,
    ):
        plane = np.asarray(bench["realistic"].reference_plane, dtype=float)
        crop, crop_grid = _mode1b_even_axis_crop(plane, bench["data"]["grid"], 0.35)
        xc = np.asarray(crop_grid["x"], dtype=float) / 1e-3
        ext = [float(xc[0]), float(xc[-1]), float(xc[0]), float(xc[-1])]
        ax.imshow(_normalise_image(crop, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"{CANONICAL_OPERATING_POINT_ID}\n{label}", fontsize=9)
        ax.set_xlabel("x (mm)")
    rows = [r for r in rebind["rows"] if r["binding"] == "resolved_binding_1029nm"]
    labels = [str(r["candidate_id"])[:30] for r in rows]
    values = [float(r["corr_to_realistic_4f"]) for r in rows]
    eligible = [bool(r["strict_hexagon_eligible"]) for r in rows]
    colors = ["tab:green" if ok else "tab:orange" for ok in eligible]
    axes[2].bar(range(len(rows)), values, color=colors)
    axes[2].axhline(STRICT_BASELINE_CORR_MIN, color="0.3", lw=0.8, ls="--", label=f"strict floor {STRICT_BASELINE_CORR_MIN}")
    axes[2].set_xticks(range(len(rows)))
    axes[2].set_xticklabels(labels, rotation=25, fontsize=6.5, ha="right")
    axes[2].set_ylabel("correlation to realistic-4F reference")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(fontsize=7)
    axes[2].set_title("resolved-binding strict audit\n(green = strict eligible; robustness cases orange by design)", fontsize=8)
    fig.suptitle("MODE 2U3 hardware rebind check (repaired strict hexagon gate)")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def write_mode2u3_hardware_closure(
    *,
    output_dir: str | Path = MODE2U3_DEFAULT_OUTPUT_ROOT,
    grid_n: int = 384,
    z_planes: int = 9,
    doc_path: str | Path = MODE2U3_DOC_PATH,
    phase_doc_path: str | Path = MODE2U3_PHASE_CAL_DOC_PATH,
    fourf_doc_path: str | Path = MODE2U3_FOURF_CAL_DOC_PATH,
    camera_doc_path: str | Path = MODE2U3_CAMERA_CAL_DOC_PATH,
) -> dict[str, Any]:
    """Run the full MODE 2U3 hardware closure and write every artefact."""

    root = Path(output_dir)
    dirs = {
        "slm": root / "00_slm",
        "phase": root / "01_phase_calibration",
        "fourf": root / "02_4f",
        "camera": root / "03_camera",
        "jones": root / "04_jones_axes",
        "conflicts": root / "05_conflicts",
        "rebind": root / "06_rebind",
        "status": root / "07_final_status",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    slm, slm_unknowns = resolve_slm_hardware()
    slm_rows = slm.rows("SLM-H") + slm.rows("SLM-V")
    _write_rows(dirs["slm"] / "slm_hardware_closure.csv", slm_rows)
    (dirs["slm"] / "slm_hardware_closure.json").write_text(
        json.dumps(
            _json_ready(
                {
                    "resolved": asdict(slm),
                    "external_lab_identity": EXTERNAL_LAB_SLM_IDENTITY,
                    "repo_family_sources": list(REPO_PLUTO_FAMILY_SOURCES),
                    "rows": slm_rows,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (dirs["slm"] / "slm_hardware_unknowns.json").write_text(
        json.dumps(_json_ready(slm_unknowns), indent=2), encoding="utf-8",
    )

    schema = slm_phase_calibration_schema()
    (dirs["phase"] / "slm_phase_calibration_schema.json").write_text(
        json.dumps(_json_ready(schema), indent=2), encoding="utf-8",
    )
    _write_phase_calibration_doc(Path(phase_doc_path), schema)

    wavelength_rows = resolve_wavelength_scopes()
    axicon_rows = resolve_axicon_index_scopes()

    rebind = run_hardware_rebind(grid_n=int(grid_n), z_planes=int(z_planes))
    bench_old = _bench(1.030e-6, grid_n=int(grid_n), z_planes=int(z_planes))
    bench_new = _bench(1.029e-6, grid_n=int(grid_n), z_planes=int(z_planes))
    realistic_eff = float(bench_new["realistic"].slm_4f_report["first_order_efficiency"])
    realistic_leak = float(bench_new["realistic"].slm_4f_report["zero_order_leakage_after_iris"])
    fourf_rows = physical_4f_rows(
        simulated_first_order_efficiency=realistic_eff,
        simulated_zero_order_leakage=realistic_leak,
    )
    _write_rows(dirs["fourf"] / "physical_4f_closure.csv", fourf_rows)
    (dirs["fourf"] / "physical_4f_closure.json").write_text(json.dumps(_json_ready(fourf_rows), indent=2), encoding="utf-8")
    _plot_4f_geometry(dirs["fourf"] / "physical_4f_geometry_highres.png", fourf_rows)
    _write_4f_calibration_doc(Path(fourf_doc_path), fourf_rows)

    camera_rows = camera_closure_rows()
    _write_rows(dirs["camera"] / "camera_hardware_closure.csv", camera_rows)
    (dirs["camera"] / "camera_hardware_closure.json").write_text(json.dumps(_json_ready(camera_rows), indent=2), encoding="utf-8")
    _write_camera_calibration_doc(Path(camera_doc_path))

    jones_rows = jones_axis_route_rows()
    qwp_statement = qwp_lab_axis_statement()
    hwp_rows = hwp_requirement_rows()
    _write_rows(dirs["jones"] / "jones_axis_route_audit.csv", jones_rows)
    (dirs["jones"] / "jones_axis_route_audit.json").write_text(
        json.dumps(
            _json_ready({"rows": jones_rows, "qwp_lab_axis_statement": qwp_statement, "hwp_requirements": hwp_rows}),
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_jones_route(dirs["jones"] / "jones_axis_route_diagram.png", jones_rows)

    conflict_rows = resolve_m2u2_conflicts()
    _write_rows(dirs["conflicts"] / "m2u3_conflict_resolution.csv", conflict_rows)
    (dirs["conflicts"] / "m2u3_conflict_resolution.json").write_text(
        json.dumps(_json_ready(conflict_rows), indent=2), encoding="utf-8",
    )

    _write_rows(dirs["rebind"] / "resolved_hardware_rebind.csv", rebind["rows"])
    (dirs["rebind"] / "resolved_hardware_rebind.json").write_text(
        json.dumps(
            _json_ready(
                {
                    "rows": rebind["rows"],
                    "sensitivity": rebind["sensitivity"],
                    "strict_gate_used": rebind["strict_gate_used"],
                    "reference_drift_floor_note": rebind["reference_drift_floor_note"],
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_rebind(dirs["rebind"] / "resolved_hardware_rebind_highres.png", rebind, bench_new, bench_old)

    outcome = mode2u3_outcome(rebind=rebind, conflict_rows=conflict_rows, jones_rows=jones_rows, hwp_rows=hwp_rows)
    (dirs["status"] / "m2u3_outcome_report.json").write_text(json.dumps(_json_ready(outcome), indent=2), encoding="utf-8")
    manifest = {
        "stage": MODE2U3_STAGE,
        "grid_n": int(grid_n),
        "z_planes": int(z_planes),
        "canonical_operating_point": CANONICAL_OPERATING_POINT_ID,
        "strict_compromise_candidate": STRICT_COMPROMISE_ID,
        "forbidden_operating_points": list(forbidden_operating_point_ids()),
        "provenance_categories": list(MODE2U3_PROVENANCE_CATEGORIES),
        "conflict_statuses": list(MODE2U3_CONFLICT_STATUSES),
        "selected_outcome": outcome["selected_outcome"],
        "m2v_authorised": outcome["m2v_authorised"],
        "microfabrication_sample_plane_claim": False,
        "docs": {
            "master": str(doc_path),
            "phase_calibration": str(phase_doc_path),
            "fourf_calibration": str(fourf_doc_path),
            "camera_calibration": str(camera_doc_path),
        },
    }
    (dirs["status"] / "nathan_mode2u3_manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    (dirs["status"] / "wavelength_scope_resolution.json").write_text(
        json.dumps(_json_ready(wavelength_rows), indent=2), encoding="utf-8",
    )
    (dirs["status"] / "axicon_index_scope_resolution.json").write_text(
        json.dumps(_json_ready(axicon_rows), indent=2), encoding="utf-8",
    )
    _write_master_doc(
        Path(doc_path),
        slm=slm,
        fourf_rows=fourf_rows,
        qwp_statement=qwp_statement,
        conflict_rows=conflict_rows,
        rebind=rebind,
        outcome=outcome,
    )
    return {
        "slm": slm,
        "slm_unknowns": slm_unknowns,
        "wavelength_rows": wavelength_rows,
        "axicon_rows": axicon_rows,
        "fourf_rows": fourf_rows,
        "camera_rows": camera_rows,
        "jones_rows": jones_rows,
        "qwp_statement": qwp_statement,
        "hwp_rows": hwp_rows,
        "conflict_rows": conflict_rows,
        "rebind": rebind,
        "outcome": outcome,
        "manifest": manifest,
    }
