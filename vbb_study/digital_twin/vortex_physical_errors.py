"""Physically grounded source/misalignment models for vortex-Bessel studies.

This module is intentionally separate from the accepted Phase 2A/2B/2C
contracts.  It rebuilds the source-scale optical route so perturbations are
introduced at the physical plane where they occur rather than as arbitrary
post-hoc image changes.

Fidelity policy
---------------
* Input beam tilt is applied at the input plane using transverse direction
  cosines before the SLM/4F chain.
* Input beam radius/decentre are applied to the Gaussian before the SLMs.
* Axicon decentre is represented by translating the physical sag profile.
* Axicon tip defects are represented by explicit sag functions (sharp,
  hyperboloidal round tip, flat/blunt truncation).
* Axicon tilt uses a rotated thin-element optical-path model.  This captures
  projected/elliptical conical geometry and path-length change and is suitable
  for small-angle sensitivity studies, but it is not advertised as a full
  vector Snell/Fresnel solution for large tilts.
* Generic Zernike sweeps belong to a separate wavefront-error study and must not
  be labelled as a physical substitute for a specific misaligned optic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.phase2a_canonical import (
    _fourier_first_order,
    _panel_from_manifest,
    _variant_settings,
)
from vbb_study.digital_twin.phase2a_contracts import (
    PHASE2A_CANONICAL_SLM_MODEL,
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.slm_model import apply_slm, slm_active_aperture


EPS = np.finfo(float).tiny
DEFAULT_WINDOW_M = 10.0e-3


@dataclass(frozen=True)
class PhysicalPerturbation:
    """Physical perturbations referenced to explicit optical planes."""

    beam_radius_scale: float = 1.0
    input_beam_decentre_m: tuple[float, float] = (0.0, 0.0)
    input_beam_angle_rad: tuple[float, float] = (0.0, 0.0)
    hologram_decentre_m: tuple[float, float] = (0.0, 0.0)
    fourier_iris_offset_fraction: float = 0.0
    axicon_base_angle_scale: float = 1.0
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0)
    axicon_tilt_rad: tuple[float, float] = (0.0, 0.0)
    axicon_tip_model: str = "sharp"
    axicon_rounding_parameter_m: float = 0.0
    axicon_flat_tip_radius_m: float = 0.0

    def validate(self) -> None:
        if self.beam_radius_scale <= 0.0:
            raise ValueError("beam_radius_scale must be positive")
        if self.axicon_base_angle_scale <= 0.0:
            raise ValueError("axicon_base_angle_scale must be positive")
        if self.axicon_tip_model not in {"sharp", "hyperboloidal_round", "flat_blunt"}:
            raise ValueError("axicon_tip_model must be sharp, hyperboloidal_round, or flat_blunt")
        if self.axicon_rounding_parameter_m < 0.0 or self.axicon_flat_tip_radius_m < 0.0:
            raise ValueError("tip dimensions cannot be negative")
        sx = math.sin(float(self.input_beam_angle_rad[0]))
        sy = math.sin(float(self.input_beam_angle_rad[1]))
        if sx * sx + sy * sy >= 1.0:
            raise ValueError("input beam direction cosines are non-propagating")
        tx, ty = map(float, self.axicon_tilt_rad)
        # The implemented axicon-tilt model is a small-angle thin-element OPD
        # model.  Refuse large angles instead of silently over-claiming fidelity.
        if math.hypot(tx, ty) > math.radians(5.0):
            raise ValueError("axicon tilt exceeds the small-angle thin-element model domain")


def _ell(case_id: str) -> int:
    try:
        return {"B0": 0, "V1": 1, "V3": 3}[case_id]
    except KeyError as exc:
        raise ValueError(f"unsupported physical-error case {case_id!r}") from exc


def incident_plane_wave_phase(
    grid: Mapping[str, Any],
    wavelength_m: float,
    angle_x_rad: float,
    angle_y_rad: float,
) -> np.ndarray:
    """Plane-wave phase for a beam entering the input plane at a finite angle.

    The transverse wavevector is kx=k*sin(angle_x), ky=k*sin(angle_y).
    For small angles this reduces to the familiar linear phase k(theta_x x +
    theta_y y), while retaining direction-cosine behaviour at larger angles.
    """

    sx = math.sin(float(angle_x_rad))
    sy = math.sin(float(angle_y_rad))
    if sx * sx + sy * sy >= 1.0:
        raise ValueError("requested input direction has no real longitudinal component")
    k0 = 2.0 * math.pi / float(wavelength_m)
    x = np.asarray(grid["X"], dtype=float)
    y = np.asarray(grid["Y"], dtype=float)
    return np.exp(1j * k0 * (sx * x + sy * y))


def axicon_sag_m(
    radius_m: np.ndarray,
    base_angle_rad: float,
    *,
    tip_model: str = "sharp",
    rounding_parameter_m: float = 0.0,
    flat_tip_radius_m: float = 0.0,
) -> np.ndarray:
    """Return the variable conical sag responsible for axicon phase.

    ``sharp`` gives h=r*tan(gamma).

    ``hyperboloidal_round`` uses
        h = sqrt(a^2 + (r tan(gamma))^2) - a,
    which approaches the ideal cone at large radius and has zero slope at the
    apex.  It is the convenient shallow-cone form of the hyperboloidal round-tip
    models used for real axicons; an additive constant is physically irrelevant.

    ``flat_blunt`` models a truncated flat central region of radius r_flat with
    a conical flank outside it.  It is a sensitivity model for a genuinely
    flattened/blunted apex, not a claim about a particular manufactured optic.
    """

    r = np.maximum(np.asarray(radius_m, dtype=float), 0.0)
    slope = math.tan(float(base_angle_rad))
    if tip_model == "sharp":
        return r * slope
    if tip_model == "hyperboloidal_round":
        a = float(rounding_parameter_m)
        if a <= 0.0:
            return r * slope
        return np.sqrt(a * a + (r * slope) ** 2) - a
    if tip_model == "flat_blunt":
        rf = float(flat_tip_radius_m)
        return np.maximum(r - rf, 0.0) * slope
    raise ValueError(f"unknown axicon tip model {tip_model!r}")


def _rotation_matrix(tilt_x_rad: float, tilt_y_rad: float) -> np.ndarray:
    cx, sx = math.cos(tilt_x_rad), math.sin(tilt_x_rad)
    cy, sy = math.cos(tilt_y_rad), math.sin(tilt_y_rad)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return ry @ rx


def tilted_axicon_local_radius_and_path_scale(
    grid: Mapping[str, Any],
    *,
    decentre_m: tuple[float, float],
    tilt_rad: tuple[float, float],
) -> tuple[np.ndarray, float]:
    """Map lab rays onto the tilted axicon reference plane.

    Rays are taken parallel to the lab z axis at the axicon plane.  The optic is
    rigidly rotated, the ray/tilted-plane intersection is solved analytically,
    and the local radial coordinate is evaluated in the optic frame.  The
    returned path scale 1/|a_z| accounts for the increased geometrical path in a
    thin-element OPD approximation.
    """

    tx, ty = map(float, tilt_rad)
    rotation = _rotation_matrix(tx, ty)
    e1 = rotation[:, 0]
    e2 = rotation[:, 1]
    axis = rotation[:, 2]
    az = float(axis[2])
    if abs(az) < 1e-9:
        raise ValueError("tilted axicon plane is nearly parallel to incident rays")

    x = np.asarray(grid["X"], dtype=float) - float(decentre_m[0])
    y = np.asarray(grid["Y"], dtype=float) - float(decentre_m[1])
    z_intersection = -(axis[0] * x + axis[1] * y) / az
    u = e1[0] * x + e1[1] * y + e1[2] * z_intersection
    v = e2[0] * x + e2[1] * y + e2[2] * z_intersection
    return np.hypot(u, v), 1.0 / abs(az)


def physical_axicon_transmission(
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    refractive_index: float,
    external_index: float,
    base_angle_rad: float,
    decentre_m: tuple[float, float] = (0.0, 0.0),
    tilt_rad: tuple[float, float] = (0.0, 0.0),
    tip_model: str = "sharp",
    rounding_parameter_m: float = 0.0,
    flat_tip_radius_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the rotated thin-element axicon phase transmission."""

    radius_local, path_scale = tilted_axicon_local_radius_and_path_scale(
        grid, decentre_m=decentre_m, tilt_rad=tilt_rad
    )
    sag = axicon_sag_m(
        radius_local,
        base_angle_rad,
        tip_model=tip_model,
        rounding_parameter_m=rounding_parameter_m,
        flat_tip_radius_m=flat_tip_radius_m,
    )
    k0 = 2.0 * math.pi / float(wavelength_m)
    opd = (float(refractive_index) - float(external_index)) * sag * path_scale
    phase = -k0 * opd
    return np.exp(1j * phase), {
        "axicon_tilt_model": "rotated_thin_element_opd_small_angle",
        "full_vector_snell_fresnel": False,
        "local_path_scale": float(path_scale),
        "tip_model": tip_model,
        "rounding_parameter_m": float(rounding_parameter_m),
        "flat_tip_radius_m": float(flat_tip_radius_m),
    }


def build_physical_route_checkpoints(
    case_id: str,
    *,
    grid_n: int,
    perturbation: PhysicalPerturbation = PhysicalPerturbation(),
    window_m: float = DEFAULT_WINDOW_M,
) -> dict[str, Any]:
    """Rebuild SLM/4F/axicon route with errors inserted at physical planes."""

    perturbation.validate()
    manifest = canonical_hardware_manifest()
    settings = _variant_settings("realistic_fixed_bench_route")
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    k0 = 2.0 * math.pi / wavelength
    canonical_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    beam_radius = canonical_radius * float(perturbation.beam_radius_scale)
    grid = make_xy_grid(int(grid_n), float(window_m) / int(grid_n))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)

    bx, by = perturbation.input_beam_decentre_m
    amplitude = np.exp(-((X - float(bx)) ** 2 + (Y - float(by)) ** 2) / beam_radius**2)
    ax_in, ay_in = perturbation.input_beam_angle_rad
    incident_phase = incident_plane_wave_phase(grid, wavelength, ax_in, ay_in)
    raw_input = np.asarray(amplitude * incident_phase, dtype=np.complex128)

    panel = _panel_from_manifest(manifest)
    panel_aperture = slm_active_aperture(grid, panel)
    hx, hy = perturbation.hologram_decentre_m
    theta = np.arctan2(Y - float(hy), X - float(hx))
    phase1 = float(_ell(case_id)) * theta
    phase2 = np.zeros_like(phase1)

    slm1 = apply_slm(
        raw_input,
        phase1,
        grid,
        panel,
        quantise_phase=True,
        apply_fill_factor=True,
        apply_carrier=False,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )
    slm2 = apply_slm(
        slm1.total,
        phase2,
        grid,
        panel,
        quantise_phase=True,
        apply_fill_factor=True,
        apply_carrier=True,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )
    post_slm = np.asarray(slm2.total, dtype=np.complex128)

    post_filter, first_order_fraction = _fourier_first_order(
        post_slm,
        grid,
        float(hardware_value(manifest, "carrier_frequency_cpm")),
        float(hardware_value(manifest, "fourier_iris_radius_cpm")),
        float(perturbation.fourier_iris_offset_fraction),
    )

    gamma0 = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    gamma = gamma0 * float(perturbation.axicon_base_angle_scale)
    axicon, axicon_meta = physical_axicon_transmission(
        grid,
        wavelength_m=wavelength,
        refractive_index=float(hardware_value(manifest, "axicon_refractive_index")),
        external_index=float(hardware_value(manifest, "axicon_external_medium_index")),
        base_angle_rad=gamma,
        decentre_m=perturbation.axicon_decentre_m,
        tilt_rad=perturbation.axicon_tilt_rad,
        tip_model=perturbation.axicon_tip_model,
        rounding_parameter_m=perturbation.axicon_rounding_parameter_m,
        flat_tip_radius_m=perturbation.axicon_flat_tip_radius_m,
    )
    post_axicon = np.asarray(post_filter * axicon, dtype=np.complex128)

    sx = math.sin(float(ax_in))
    sy = math.sin(float(ay_in))
    sz = math.sqrt(max(0.0, 1.0 - sx * sx - sy * sy))
    return {
        "grid": grid,
        "raw_input": raw_input,
        "post_slm": post_slm,
        "post_filter": np.asarray(post_filter, dtype=np.complex128),
        "post_axicon": post_axicon,
        "metadata": {
            "case_id": case_id,
            "vortex_charge": int(_ell(case_id)),
            "route_id": "phase2e_physical_error_route",
            "window_m": float(window_m),
            "grid_n": int(grid_n),
            "dx_m": float(grid["dx"]),
            "wavelength_m": wavelength,
            "beam_radius_m": float(beam_radius),
            "input_beam_decentre_m": tuple(map(float, perturbation.input_beam_decentre_m)),
            "input_beam_angle_rad": tuple(map(float, perturbation.input_beam_angle_rad)),
            "input_direction_cosines": (float(sx), float(sy), float(sz)),
            "input_angle_applied_plane": "before_SLM1",
            "hologram_decentre_m": tuple(map(float, perturbation.hologram_decentre_m)),
            "fourier_iris_offset_fraction": float(perturbation.fourier_iris_offset_fraction),
            "first_order_efficiency": float(first_order_fraction),
            "objective_transform_application_count": 0,
            "additional_objective_pupil_application_count": 0,
            "axicon_base_angle_rad": float(gamma),
            "axicon_decentre_m": tuple(map(float, perturbation.axicon_decentre_m)),
            "axicon_tilt_rad": tuple(map(float, perturbation.axicon_tilt_rad)),
            "axicon_tip_model": perturbation.axicon_tip_model,
            **axicon_meta,
            "physical_model_notes": (
                "Input beam errors are upstream physical-field errors. Axicon decentre and tip shape "
                "modify the conical sag directly. Axicon tilt is a rotated thin-element OPD model; "
                "large-angle/full-interface claims require vector Snell/Fresnel ray-wave modelling."
            ),
            "calibration_required": [
                "actual input beam angle/decentre",
                "actual axicon tip profile or rounding parameter",
                "actual axicon clear aperture",
                "actual axicon rigid-body tilt/decentre",
            ],
        },
    }


def build_physical_source(
    case_id: str,
    *,
    grid_n: int,
    perturbation: PhysicalPerturbation = PhysicalPerturbation(),
    window_m: float = DEFAULT_WINDOW_M,
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    checkpoints = build_physical_route_checkpoints(
        case_id, grid_n=grid_n, perturbation=perturbation, window_m=window_m
    )
    return checkpoints["post_axicon"], checkpoints["grid"], dict(checkpoints["metadata"])
