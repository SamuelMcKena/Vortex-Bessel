"""Integrated physical-system error route for scalar Bessel/vortex studies.

Research route:
    Gaussian beam -> SLM1 -> SLM2/carrier -> explicit propagated 4F + fixed iris
    -> selected-order carrier removal -> physical axicon -> free space.

The route supports the established B0/V1/V3 cases and arbitrary integer vortex
charges written as ``V<ell>`` (for example ``V20``).  This lets the same physical
beam/SLM/4F/axicon model be used by the q=20 inverse-correction benchmarks rather
than maintaining a separate idealized propagation branch.

The route is intentionally separate from accepted Phase 2A/2B/2C contracts.
Every error is introduced at its physical plane.  Calibration-only quantities
(LUTs, static SLM maps, lens OPD maps, axicon surface maps) are optional arrays;
no fabricated measurement is silently inserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest
from vbb_study.digital_twin.phase2a_contracts import (
    PHASE2A_CANONICAL_SLM_MODEL,
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import (
    GaussianBeamError,
    SLMError,
    actual_slm_phase,
    gaussian_input_field,
    transformed_pattern_coordinates,
)
from vbb_study.digital_twin.vortex_error_reference_models import exact_refractive_axicon_kr_m_inv
from vbb_study.digital_twin.vortex_explicit_4f import FourFError, explicit_4f_relay
from vbb_study.digital_twin.vortex_rotated_plane import lab_to_tilted_plane, tilted_to_lab_plane
from vbb_study.equations.fields import make_xy_grid
from vbb_study.slm_model import apply_slm, pixelate


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi
DEFAULT_WINDOW_M = 10.0e-3


@dataclass(frozen=True)
class AxiconError:
    base_angle_scale: float = 1.0
    refractive_index_scale: float = 1.0
    decentre_m: tuple[float, float] = (0.0, 0.0)
    tilt_rad: tuple[float, float] = (0.0, 0.0)
    clear_radius_m: float | None = None
    tip_model: str = "sharp"
    rounding_parameter_m: float = 0.0
    flat_tip_radius_m: float = 0.0

    def validate(self) -> None:
        if self.base_angle_scale <= 0.0 or self.refractive_index_scale <= 0.0:
            raise ValueError("axicon angle/index scales must be positive")
        if self.clear_radius_m is not None and self.clear_radius_m <= 0.0:
            raise ValueError("axicon clear radius must be positive")
        if self.tip_model not in {"sharp", "hyperboloidal_round", "flat_blunt"}:
            raise ValueError("unsupported axicon tip model")
        if self.rounding_parameter_m < 0.0 or self.flat_tip_radius_m < 0.0:
            raise ValueError("axicon tip dimensions cannot be negative")
        if math.hypot(*map(float, self.tilt_rad)) >= math.radians(20.0):
            raise ValueError("scalar rotated-plane axicon study is not authorised at >=20 deg rigid tilt")


@dataclass(frozen=True)
class SystemErrorConfig:
    beam: GaussianBeamError = GaussianBeamError()
    slm1: SLMError = SLMError()
    slm2: SLMError = SLMError()
    fourf: FourFError = FourFError()
    axicon: AxiconError = AxiconError()


def _ell(case_id: str) -> int:
    """Return the integer vortex charge encoded by a scalar case identifier."""
    if case_id == "B0":
        return 0
    if case_id.startswith("V") and len(case_id) > 1:
        try:
            return int(case_id[1:])
        except ValueError as exc:
            raise ValueError(f"invalid scalar vortex case {case_id!r}") from exc
    raise ValueError(f"unsupported scalar vortex case {case_id!r}")


def axicon_sag_m(
    r_m: np.ndarray,
    base_angle_rad: float,
    *,
    tip_model: str,
    rounding_parameter_m: float,
    flat_tip_radius_m: float,
) -> np.ndarray:
    r = np.maximum(np.asarray(r_m, dtype=float), 0.0)
    slope = math.tan(float(base_angle_rad))
    sharp = r * slope
    if tip_model == "sharp":
        return sharp
    if tip_model == "hyperboloidal_round":
        a = float(rounding_parameter_m)
        if a <= 0.0:
            return sharp
        return np.sqrt(a * a + (r * slope) ** 2) - a
    if tip_model == "flat_blunt":
        rf = float(flat_tip_radius_m)
        return np.maximum(r - rf, 0.0) * slope
    raise ValueError(tip_model)


def physical_axicon_on_own_plane(
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float,
    error: AxiconError,
    surface_height_error_m: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    error.validate()
    X = np.asarray(grid["X"], dtype=float) - float(error.decentre_m[0])
    Y = np.asarray(grid["Y"], dtype=float) - float(error.decentre_m[1])
    R = np.hypot(X, Y)
    gamma = float(base_angle_rad) * float(error.base_angle_scale)
    n_ax = float(refractive_index) * float(error.refractive_index_scale)
    n_ext = float(external_index)

    kr_exact = exact_refractive_axicon_kr_m_inv(
        wavelength_m=float(wavelength_m),
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
    )
    ideal_phase = -kr_exact * R
    sharp_sag = R * math.tan(gamma)
    defect_sag = axicon_sag_m(
        R,
        gamma,
        tip_model=error.tip_model,
        rounding_parameter_m=error.rounding_parameter_m,
        flat_tip_radius_m=error.flat_tip_radius_m,
    )
    k0 = TWOPI / float(wavelength_m)
    defect_phase = -k0 * (n_ax - n_ext) * (defect_sag - sharp_sag)
    total_phase = ideal_phase + defect_phase

    if surface_height_error_m is not None:
        height = np.asarray(surface_height_error_m, dtype=float)
        if height.shape != R.shape:
            raise ValueError("axicon surface map shape does not match grid")
        total_phase += -k0 * (n_ax - n_ext) * height
        surface_status = "measured_or_user_supplied"
    else:
        surface_status = "none"

    transmission = np.exp(1j * total_phase)
    if error.clear_radius_m is not None:
        transmission *= (R <= float(error.clear_radius_m))
        aperture_status = "explicit_physical_clear_aperture"
    else:
        aperture_status = "not_bound_no_clipping"

    tip_quantitative = abs(math.degrees(gamma)) <= 5.0
    return np.asarray(transmission, dtype=np.complex128), {
        "base_angle_rad": gamma,
        "refractive_index": n_ax,
        "external_index": n_ext,
        "exact_kr_m_inv": float(kr_exact),
        "decentre_m": tuple(map(float, error.decentre_m)),
        "clear_radius_m": None if error.clear_radius_m is None else float(error.clear_radius_m),
        "clear_aperture_status": aperture_status,
        "tip_model": error.tip_model,
        "rounding_parameter_m": float(error.rounding_parameter_m),
        "flat_tip_radius_m": float(error.flat_tip_radius_m),
        "surface_height_map_status": surface_status,
        "tip_defect_fidelity": (
            "candidate_shallow_angle_thin_OPD_defect_on_exact_cone"
            if tip_quantitative
            else "not_quantitative_at_high_base_angle_without_full_surface_refraction"
        ),
    }


def build_system_route(
    case_id: str,
    *,
    grid_n: int,
    config: SystemErrorConfig = SystemErrorConfig(),
    window_m: float = DEFAULT_WINDOW_M,
    slm1_phase_lut_rad: np.ndarray | None = None,
    slm2_phase_lut_rad: np.ndarray | None = None,
    slm1_static_phase_map_rad: np.ndarray | None = None,
    slm2_static_phase_map_rad: np.ndarray | None = None,
    lens1_opd_map_m: np.ndarray | None = None,
    lens2_opd_map_m: np.ndarray | None = None,
    axicon_surface_height_error_m: np.ndarray | None = None,
) -> dict[str, Any]:
    config.beam.validate()
    config.slm1.validate()
    config.slm2.validate()
    config.axicon.validate()
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    pixel_pitch = float(hardware_value(manifest, "slm_pixel_pitch_m"))
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    f4f = float(hardware_value(manifest, "fourf_focal_length_m"))
    iris_radius = float(hardware_value(manifest, "fourier_iris_radius_m"))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))
    gamma0 = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))

    grid = make_xy_grid(int(grid_n), float(window_m) / int(grid_n))
    beam, beam_meta = gaussian_input_field(
        grid,
        wavelength_m=wavelength,
        canonical_radius_m=beam_radius,
        error=config.beam,
    )
    panel = _panel_from_manifest(manifest)

    x1, y1 = transformed_pattern_coordinates(grid, config.slm1)
    command1 = pixelate(float(_ell(case_id)) * np.arctan2(y1, x1), grid, panel)
    actual1, slm1_meta = actual_slm_phase(
        command1,
        grid,
        error=config.slm1,
        pixel_pitch_m=pixel_pitch,
        lut_phase_rad=slm1_phase_lut_rad,
        static_phase_map_rad=slm1_static_phase_map_rad,
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

    x2, _ = transformed_pattern_coordinates(grid, config.slm2)
    command2 = pixelate(TWOPI * carrier * x2, grid, panel)
    actual2, slm2_meta = actual_slm_phase(
        command2,
        grid,
        error=config.slm2,
        pixel_pitch_m=pixel_pitch,
        lut_phase_rad=slm2_phase_lut_rad,
        static_phase_map_rad=slm2_static_phase_map_rad,
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
        error=config.fourf,
        lens1_opd_map_m=lens1_opd_map_m,
        lens2_opd_map_m=lens2_opd_map_m,
    )
    X = np.asarray(grid["X"], dtype=float)
    # Two Fourier transforms in a unity-magnification 4F relay produce an image
    # inversion.  Thus an input +G carrier appears as -G at the image plane.  In
    # the selected-order beam frame the nominal carrier is removed with +G.
    selected_order = np.asarray(relay["output"], dtype=np.complex128) * np.exp(
        +1j * TWOPI * carrier * X
    )

    tx, ty = map(float, config.axicon.tilt_rad)
    if tx != 0.0 or ty != 0.0:
        field_on_axicon, to_tilt_meta = lab_to_tilted_plane(
            selected_order,
            grid,
            wavelength_m=wavelength,
            tilt_x_rad=tx,
            tilt_y_rad=ty,
        )
    else:
        field_on_axicon = selected_order
        to_tilt_meta = {"fidelity": "identity_parallel_plane", "spectral_clipped_fraction": 0.0}

    axicon_t, axicon_meta = physical_axicon_on_own_plane(
        grid,
        wavelength_m=wavelength,
        base_angle_rad=gamma0,
        refractive_index=n_ax,
        external_index=n_ext,
        error=config.axicon,
        surface_height_error_m=axicon_surface_height_error_m,
    )
    post_axicon_local = field_on_axicon * axicon_t

    if tx != 0.0 or ty != 0.0:
        post_axicon, from_tilt_meta = tilted_to_lab_plane(
            post_axicon_local,
            grid,
            wavelength_m=wavelength,
            tilt_x_rad=tx,
            tilt_y_rad=ty,
        )
        tilt_status = "scalar_rotated_angular_spectrum"
    else:
        post_axicon = post_axicon_local
        from_tilt_meta = {"fidelity": "identity_parallel_plane", "spectral_clipped_fraction": 0.0}
        tilt_status = "none"

    return {
        "grid": grid,
        "input_beam": beam,
        "post_slm1": np.asarray(slm1.total, dtype=np.complex128),
        "post_slm2": np.asarray(slm2.total, dtype=np.complex128),
        "fourier_plane_before_iris": relay["fourier_plane_before_iris"],
        "fourier_iris_mask": relay["iris_mask"],
        "post_4f_selected_order": selected_order,
        "field_on_axicon_plane": np.asarray(field_on_axicon, dtype=np.complex128),
        "post_axicon_local": np.asarray(post_axicon_local, dtype=np.complex128),
        "post_axicon": np.asarray(post_axicon, dtype=np.complex128),
        "metadata": {
            "route_id": "vortex_explicit_system_error_route_v1",
            "case_id": case_id,
            "vortex_charge": int(_ell(case_id)),
            "grid_n": int(grid_n),
            "window_m": float(window_m),
            "dx_m": float(grid["dx"]),
            "wavelength_m": wavelength,
            "beam": beam_meta,
            "slm1": {**slm1_meta, **dict(slm1.metadata)},
            "slm2": {**slm2_meta, **dict(slm2.metadata)},
            "fourf": relay["metadata"],
            "selected_order_carrier_removal": "plus_G_after_4F_image_inversion",
            "axicon": axicon_meta,
            "axicon_rigid_tilt_rad": (tx, ty),
            "axicon_tilt_status": tilt_status,
            "lab_to_tilted": to_tilt_meta,
            "tilted_to_lab": from_tilt_meta,
            "calibration_policy": {
                "SLM_LUT_and_static_maps": "measured arrays required for absolute hardware claims",
                "SLM_fringing": "kernel must be fitted to actual panel diffraction data",
                "lens_OPD": "measured or manufacturer/bench-derived map required",
                "axicon_clear_aperture": "physical measurement required",
                "axicon_surface_error": "profilometry/interferometry required",
                "high_angle_refractive_tilt": "full vector surface refraction still required for absolute claims",
            },
        },
    }
