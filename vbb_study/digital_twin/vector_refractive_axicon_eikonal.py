"""Authoritative common-eikonal vector two-surface refractive axicon solver.

Why this module exists
----------------------
The first Phase-2H prototype used the *total structured-field Poynting vector* as
its ray direction.  That is correct for one local plane wave, but it is not a
safe Snell-law normal for an arbitrary structured vector superposition because
spin/interference energy currents can make the total Poynting vector differ from
the local phase normal.  The independent Fermat/eikonal gradient gate exposed
that distinction.

This implementation therefore follows the vector geometrical-optics ansatz

    E(r) = a(r) exp(i Phi(r)),       k = grad Phi,

and derives the incident ray direction from a common vector-field phase gradient.
The local canonical phase-gradient estimate is

    q_j = Im[ sum_a E_a^* d_j E_a ] / sum_a |E_a|^2,

with the analytically tracked tilted-plane carrier added exactly.  A sparse
Southwell least-squares integration reconstructs one common scalar eikonal from
(q_x,q_y).  Two explicit validity gates prevent misuse:

1. individual energetic Cartesian components must agree with the common local
   phase gradient to within the declared fraction of |k|;
2. the reconstructed scalar eikonal must reproduce the estimated slopes.

Snell/Fresnel transport then acts on this wavevector.  The electromagnetic
Poynting vector calculated from E and H is retained independently as an energy-
flux diagnostic, never substituted for the phase normal.

The remainder of the model is the Phase-2H phase-safe architecture: real
plano-conical intersections, 3-D local s/p Fresnel transport, finite OPL,
ray-tube Jacobian, fixed laboratory z-plane, separate vector-envelope and
unwrapped-phase remapping, Maxwell spectral projection, flux closure and a hard
Nyquist gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from vbb_study.calibration.shack_hartmann import reconstruct_opd_from_slopes
from vbb_study.digital_twin.vector_refractive_axicon import (
    EPS,
    TWOPI,
    VectorRefractiveAxiconRayBundle,
    VectorRefractiveAxiconResult,
    _cone_intersection_distance,
    _dominant_envelope_phase,
    _mapping_jacobian_signed,
    _output_grid,
    _phase_safe_remap,
    _refract_masked,
    _safe_unit,
    fresnel_transmit_vector_3d,
    sample_vector_field_on_tilted_entrance,
    spectral_normal_flux_au,
)
from vbb_study.digital_twin.vortex_refractive_axicon import RefractiveAxiconGeometry
from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix
from vbb_study.equations.fields import fft2c
from vbb_study.vector_field import VectorField, propagate_vector_asm, spectral_transversality_residual


@dataclass(frozen=True)
class CommonEikonalEstimate:
    phase_rad: np.ndarray
    direction_local: np.ndarray
    valid_mask: np.ndarray
    qx_rad_per_m: np.ndarray
    qy_rad_per_m: np.ndarray
    common_phase_gradient_error_fraction: np.ndarray
    component_phase_gradient_disagreement_fraction: np.ndarray
    poynting_direction_disagreement_rad: np.ndarray
    metadata: Mapping[str, Any]


def _complex_derivatives(field: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    d_dy, d_dx = np.gradient(np.asarray(field, dtype=np.complex128), float(dy_m), float(dx_m))
    return np.asarray(d_dx, dtype=np.complex128), np.asarray(d_dy, dtype=np.complex128)


def estimate_common_vector_eikonal(
    electric_envelope_lab: np.ndarray,
    poynting_lab: np.ndarray,
    grid: Mapping[str, Any],
    *,
    carrier_local_cpm: tuple[float, float],
    rotation_local_to_lab: np.ndarray,
    wavelength_m: float,
    medium_index: float,
    valid_mask: np.ndarray | None = None,
    intensity_floor_fraction: float = 1e-10,
    maximum_component_disagreement_fraction: float = 0.02,
    maximum_reconstruction_error_fraction: float = 0.01,
) -> CommonEikonalEstimate:
    """Estimate and integrate one local vector-wave eikonal on the tilted plane.

    The vector connection ``Im(E* grad E)/|E|^2`` is invariant under a constant
    unitary change of Cartesian basis, so the fixed laboratory components can be
    differentiated directly with respect to the axicon-local surface x/y axes.
    The analytic carrier is added in radians/metre and is never sampled as a
    complex exponential.
    """

    E = np.asarray(electric_envelope_lab, dtype=np.complex128)
    if E.ndim != 3 or E.shape[-1] != 3:
        raise ValueError("electric_envelope_lab must have shape (ny,nx,3)")
    P_lab = np.asarray(poynting_lab, dtype=float)
    if P_lab.shape != E.shape:
        raise ValueError("poynting_lab must match electric field shape")

    dx = float(grid["dx"])
    dy = float(grid.get("dy", dx))
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    intensity_components = np.abs(E) ** 2
    intensity = np.sum(intensity_components, axis=-1)
    floor = float(intensity_floor_fraction) * max(float(np.max(intensity)), EPS)
    valid = intensity > floor
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if np.count_nonzero(valid) < 16:
        raise ValueError("too few illuminated samples for common-eikonal estimation")

    derivative_x = np.empty_like(E)
    derivative_y = np.empty_like(E)
    for component in range(3):
        derivative_x[..., component], derivative_y[..., component] = _complex_derivatives(
            E[..., component], dx, dy
        )

    qx_env_num = np.sum(np.imag(np.conj(E) * derivative_x), axis=-1)
    qy_env_num = np.sum(np.imag(np.conj(E) * derivative_y), axis=-1)
    qx_env = qx_env_num / np.maximum(intensity, EPS)
    qy_env = qy_env_num / np.maximum(intensity, EPS)
    carrier_kx = TWOPI * float(carrier_local_cpm[0])
    carrier_ky = TWOPI * float(carrier_local_cpm[1])
    qx = qx_env + carrier_kx
    qy = qy_env + carrier_ky

    k_medium = TWOPI * float(medium_index) / float(wavelength_m)
    transverse_sq = qx * qx + qy * qy
    propagating = transverse_sq < (0.999999 * k_medium) ** 2
    valid &= propagating & np.isfinite(qx) & np.isfinite(qy)
    if np.count_nonzero(valid) < 16:
        raise ValueError("common eikonal is non-propagating over the illuminated pupil")
    qz = np.sqrt(np.maximum(k_medium * k_medium - transverse_sq, 0.0))
    direction_local = np.stack([qx, qy, qz], axis=-1) / k_medium

    # Componentwise canonical momentum disagreement.  Components near a local
    # zero have negligible weight and cannot destabilise the diagnostic.
    disagreement_sq_num = np.zeros_like(intensity, dtype=float)
    active_component_weight = np.zeros_like(intensity, dtype=float)
    local_component_floor = 1e-8 * np.maximum(intensity, floor)
    for component in range(3):
        weight = intensity_components[..., component]
        active = weight > local_component_floor
        qx_c = np.divide(
            np.imag(np.conj(E[..., component]) * derivative_x[..., component]),
            weight,
            out=np.zeros_like(weight, dtype=float),
            where=active,
        ) + carrier_kx
        qy_c = np.divide(
            np.imag(np.conj(E[..., component]) * derivative_y[..., component]),
            weight,
            out=np.zeros_like(weight, dtype=float),
            where=active,
        ) + carrier_ky
        delta_sq = (qx_c - qx) ** 2 + (qy_c - qy) ** 2
        disagreement_sq_num += np.where(active, weight * delta_sq, 0.0)
        active_component_weight += np.where(active, weight, 0.0)
    component_disagreement = np.sqrt(
        disagreement_sq_num / np.maximum(active_component_weight, EPS)
    ) / k_medium

    # Integrate q/k0 as an optical-path slope.  This gives the least-squares
    # common scalar phase rather than privileging Ex or Ey at polarization zeros.
    k0 = TWOPI / float(wavelength_m)
    reconstruction = reconstruct_opd_from_slopes(
        qx / k0,
        qy / k0,
        x,
        y,
        valid_mask=valid,
        atol=1e-11,
        btol=1e-11,
    )
    phase = k0 * np.asarray(reconstruction.opd_m, dtype=float)
    phase = np.where(np.isfinite(phase), phase, 0.0)
    dphase_dy, dphase_dx = np.gradient(phase, dy, dx)
    reconstruction_error = np.hypot(dphase_dx - qx, dphase_dy - qy) / k_medium

    # Independent Poynting-vs-wave-normal diagnostic.  A difference is allowed
    # for a structured field; it is precisely why Poynting is not fed to Snell.
    rotation = np.asarray(rotation_local_to_lab, dtype=float)
    p_local = P_lab @ rotation
    p_direction, p_nonzero = _safe_unit(p_local, np.asarray([0.0, 0.0, 1.0]))
    dot = np.clip(np.sum(p_direction * direction_local, axis=-1), -1.0, 1.0)
    poynting_angle = np.where(p_nonzero, np.arccos(dot), np.nan)

    component_values = component_disagreement[valid]
    reconstruction_values = reconstruction_error[valid]
    p95_component = float(np.percentile(component_values, 95.0))
    p95_reconstruction = float(np.percentile(reconstruction_values, 95.0))
    if p95_component > float(maximum_component_disagreement_fraction):
        raise ValueError(
            "vector field does not admit one common local eikonal at the declared tolerance: "
            f"p95 component-wavevector disagreement={p95_component:.6g}"
        )
    if p95_reconstruction > float(maximum_reconstruction_error_fraction):
        raise ValueError(
            "vector field common phase is not sufficiently integrable for geometrical-optics refraction: "
            f"p95 reconstructed-gradient error={p95_reconstruction:.6g}"
        )

    return CommonEikonalEstimate(
        phase_rad=np.asarray(phase, dtype=float),
        direction_local=np.asarray(direction_local, dtype=float),
        valid_mask=np.asarray(valid, dtype=bool),
        qx_rad_per_m=np.asarray(qx, dtype=float),
        qy_rad_per_m=np.asarray(qy, dtype=float),
        common_phase_gradient_error_fraction=np.asarray(reconstruction_error, dtype=float),
        component_phase_gradient_disagreement_fraction=np.asarray(component_disagreement, dtype=float),
        poynting_direction_disagreement_rad=np.asarray(poynting_angle, dtype=float),
        metadata={
            "method": "vector_canonical_phase_gradient_plus_Southwell_common_eikonal",
            "phase_direction_policy": "Snell_uses_grad_Phi_not_total_structured_Poynting",
            "p95_component_wavevector_disagreement_fraction": p95_component,
            "p95_reconstructed_gradient_error_fraction": p95_reconstruction,
            "median_poynting_vs_wavevector_angle_rad": float(np.nanmedian(poynting_angle[valid])),
            "p95_poynting_vs_wavevector_angle_rad": float(np.nanpercentile(poynting_angle[valid], 95.0)),
            "southwell_residual_rms_m": float(reconstruction.residual_rms_m),
            "southwell_iterations": int(reconstruction.metadata["lsqr_iterations"]),
            "carrier_local_cpm": [float(carrier_local_cpm[0]), float(carrier_local_cpm[1])],
        },
    )


def build_tilted_vector_refractive_axicon_field(
    upstream_field: VectorField,
    *,
    geometry: RefractiveAxiconGeometry,
    tilt_x_rad: float,
    tilt_y_rad: float,
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0),
    reference_gap_m: float = 0.25e-3,
    output_n: int | None = None,
    output_window_m: float | None = None,
    output_center_lab_m: tuple[float, float] = (0.0, 0.0),
    apex_exclusion_radius_m: float = 0.0,
    minimum_mapping_jacobian: float = 1e-5,
    maximum_nyquist_fraction: float = 0.90,
    maximum_interpolation_flux_correction_fraction: float = 0.10,
    maximum_component_eikonal_disagreement_fraction: float = 0.02,
    maximum_eikonal_reconstruction_error_fraction: float = 0.01,
    maximum_local_plane_wave_flux_disagreement_fraction: float = 0.03,
) -> VectorRefractiveAxiconResult:
    """Construct the transmitted vector field on a fixed laboratory z-plane."""

    geometry.validate()
    if not np.isclose(upstream_field.medium_index, geometry.external_index, rtol=0.0, atol=1e-12):
        raise ValueError("upstream field medium must equal axicon external medium")

    envelope, carrier_local, poynting_lab, surface_meta = sample_vector_field_on_tilted_entrance(
        upstream_field,
        tilt_x_rad=float(tilt_x_rad),
        tilt_y_rad=float(tilt_y_rad),
    )
    grid = envelope.grid
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    dx = float(grid["dx"])
    dy = float(grid.get("dy", dx))
    rotation = rotation_matrix(float(tilt_x_rad), float(tilt_y_rad))
    dec_x, dec_y = map(float, axicon_decentre_m)
    X_rel = X - dec_x
    Y_rel = Y - dec_y
    e_env_lab = np.stack([envelope.ex, envelope.ey, envelope.ez], axis=-1)

    common = estimate_common_vector_eikonal(
        e_env_lab,
        poynting_lab,
        grid,
        carrier_local_cpm=(float(carrier_local[0]), float(carrier_local[1])),
        rotation_local_to_lab=rotation,
        wavelength_m=upstream_field.wavelength_m,
        medium_index=geometry.external_index,
        maximum_component_disagreement_fraction=float(maximum_component_eikonal_disagreement_fraction),
        maximum_reconstruction_error_fraction=float(maximum_eikonal_reconstruction_error_fraction),
    )
    incident_local = np.asarray(common.direction_local, dtype=float)
    e_env_local = e_env_lab @ rotation
    e_transverse = e_env_local - np.sum(e_env_local * incident_local, axis=-1, keepdims=True) * incident_local
    transverse_intensity = np.sum(np.abs(e_transverse) ** 2, axis=-1)
    valid = np.asarray(common.valid_mask, dtype=bool) & (incident_local[..., 2] > 0.0)

    # Compare exact structured-field entrance normal flux to the local-plane-wave
    # flux used by Fresnel theory.  This is a GO validity diagnostic, not a ray
    # direction definition.
    poynting_local = np.asarray(poynting_lab, dtype=float) @ rotation
    exact_normal_flux = np.maximum(poynting_local[..., 2], 0.0)
    local_plane_wave_flux = float(geometry.external_index) * incident_local[..., 2] * transverse_intensity
    flux_scale = np.sum(exact_normal_flux[valid]) / max(np.sum(local_plane_wave_flux[valid]), EPS)
    scaled_local_flux = flux_scale * local_plane_wave_flux
    integrated_flux_disagreement = abs(
        float(np.sum(scaled_local_flux[valid])) / max(float(np.sum(exact_normal_flux[valid])), EPS) - 1.0
    )
    # The scale makes the integrated quantities equal by construction; what
    # matters physically is spatial disagreement after removing the common Z0
    # convention factor.
    local_flux_error = np.abs(scaled_local_flux - exact_normal_flux) / np.maximum(exact_normal_flux, 1e-8 * np.max(exact_normal_flux))
    p95_flux_error = float(np.percentile(local_flux_error[valid], 95.0))
    if p95_flux_error > float(maximum_local_plane_wave_flux_disagreement_fraction):
        raise ValueError(
            "local plane-wave Fresnel flux approximation is not valid for this entrance vector field: "
            f"p95 normal-flux disagreement={p95_flux_error:.6g}"
        )

    flat_normal = np.asarray([0.0, 0.0, 1.0])
    internal_local, valid1 = _refract_masked(
        incident_local,
        flat_normal,
        n1=geometry.external_index,
        n2=geometry.refractive_index,
    )
    e_internal, T1, R1 = fresnel_transmit_vector_3d(
        np.where(valid[..., None], e_transverse, 0.0j),
        incident_local,
        internal_local,
        flat_normal,
        n1=geometry.external_index,
        n2=geometry.refractive_index,
    )
    valid &= valid1

    internal_distance, hit = _cone_intersection_distance(
        X_rel,
        Y_rel,
        internal_local,
        base_angle_rad=geometry.base_angle_rad,
        centre_thickness_m=geometry.centre_thickness_m,
    )
    s_inside = np.where(hit, internal_distance, 0.0)
    exit_x = X_rel + s_inside * internal_local[..., 0]
    exit_y = Y_rel + s_inside * internal_local[..., 1]
    exit_z = s_inside * internal_local[..., 2]
    exit_radius = np.hypot(exit_x, exit_y)
    phi = np.arctan2(exit_y, exit_x)
    sg = math.sin(geometry.base_angle_rad)
    cg = math.cos(geometry.base_angle_rad)
    cone_normal = np.stack([sg * np.cos(phi), sg * np.sin(phi), np.full_like(phi, cg)], axis=-1)
    outgoing_local, valid2 = _refract_masked(
        internal_local,
        cone_normal,
        n1=geometry.refractive_index,
        n2=geometry.external_index,
    )
    e_exit, T2, R2 = fresnel_transmit_vector_3d(
        np.where(hit[..., None], e_internal, 0.0j),
        internal_local,
        outgoing_local,
        cone_normal,
        n1=geometry.refractive_index,
        n2=geometry.external_index,
    )
    valid &= (
        hit & valid2
        & (exit_radius <= geometry.clear_radius_m)
        & (exit_radius >= float(apex_exclusion_radius_m))
    )

    exit_local = np.stack([exit_x, exit_y, exit_z], axis=-1)
    offset_lab = rotation @ np.asarray([dec_x, dec_y, 0.0])
    exit_lab = exit_local @ rotation.T + offset_lab
    outgoing_lab = outgoing_local @ rotation.T
    valid &= outgoing_lab[..., 2] > 0.0
    if np.count_nonzero(valid) < 64:
        raise ValueError("too few transmitted rays remain after physical axicon gates")

    gap = max(float(reference_gap_m), 8.0 * max(dx, dy))
    z_ref = float(np.max(exit_lab[..., 2][valid])) + gap
    s_out = (z_ref - exit_lab[..., 2]) / np.where(outgoing_lab[..., 2] > 0.0, outgoing_lab[..., 2], np.nan)
    valid &= np.isfinite(s_out) & (s_out > 0.0)
    x_ref = exit_lab[..., 0] + s_out * outgoing_lab[..., 0]
    y_ref = exit_lab[..., 1] + s_out * outgoing_lab[..., 1]
    jac_signed, _ = _mapping_jacobian_signed(x_ref, y_ref, dx_m=dx, dy_m=dy)
    jac_abs = np.abs(jac_signed)
    valid &= np.isfinite(jac_abs) & (jac_abs > float(minimum_mapping_jacobian))

    interior = valid.copy()
    if min(interior.shape) > 8:
        interior[:3, :] = False
        interior[-3:, :] = False
        interior[:, :3] = False
        interior[:, -3:] = False
    signs = jac_signed[interior]
    positive_fraction = float(np.mean(signs > 0.0)) if signs.size else 0.0
    negative_fraction = float(np.mean(signs < 0.0)) if signs.size else 0.0
    if positive_fraction > 0.02 and negative_fraction > 0.02:
        raise ValueError("ray map folds before the fixed laboratory reference plane")

    input_flux_density = local_plane_wave_flux
    cumulative_T = T1 * T2
    norm_exit = np.sqrt(np.sum(np.abs(e_exit) ** 2, axis=-1))
    unit_exit_local = np.divide(
        e_exit,
        norm_exit[..., None],
        out=np.zeros_like(e_exit),
        where=norm_exit[..., None] > 100.0 * EPS,
    )
    unit_exit_lab = unit_exit_local @ rotation.T
    desired_intensity = input_flux_density * cumulative_T / np.maximum(
        float(geometry.external_index) * outgoing_lab[..., 2] * jac_abs,
        EPS,
    )
    vector_envelope_rays = unit_exit_lab * np.sqrt(np.maximum(desired_intensity, 0.0))[..., None]
    vector_envelope_rays = np.where(valid[..., None], vector_envelope_rays, 0.0j)

    opl = geometry.refractive_index * s_inside + geometry.external_index * s_out
    total_phase = np.asarray(common.phase_rad, dtype=float) + (TWOPI / upstream_field.wavelength_m) * opl
    phase_piston = float(np.median(total_phase[valid]))
    total_phase = total_phase - phase_piston

    fresnel1_error = float(np.max(np.abs(T1[valid] + R1[valid] - 1.0)))
    fresnel2_error = float(np.max(np.abs(T2[valid] + R2[valid] - 1.0)))
    all_flux = float(np.sum(input_flux_density[valid] * cumulative_T[valid]) * dx * dy)
    reconstructed_ray_flux = float(
        np.sum(
            float(geometry.external_index)
            * np.sum(np.abs(vector_envelope_rays[valid]) ** 2, axis=-1)
            * outgoing_lab[..., 2][valid]
            * jac_abs[valid]
        ) * dx * dy
    )

    n_out = int(upstream_field.ex.shape[0] if output_n is None else output_n)
    if n_out < 64:
        raise ValueError("output_n must be at least 64")
    input_window = max(float(np.ptp(X) + dx), float(np.ptp(Y) + dy))
    mapped_span = max(float(np.ptp(x_ref[valid])), float(np.ptp(y_ref[valid])))
    window = float(output_window_m) if output_window_m is not None else max(input_window, 1.05 * mapped_span)
    if window <= 0.0:
        raise ValueError("output_window_m must be positive")
    out_grid = _output_grid(n_out, window, output_center_lab_m)
    x_out = np.asarray(out_grid["x"], dtype=float)
    y_out = np.asarray(out_grid["y"], dtype=float)
    in_window = (
        valid
        & (x_ref >= x_out[0]) & (x_ref <= x_out[-1])
        & (y_ref >= y_out[0]) & (y_ref <= y_out[-1])
    )
    window_flux = float(np.sum(input_flux_density[in_window] * cumulative_T[in_window]) * dx * dy)
    if window_flux <= EPS:
        raise ValueError("fixed laboratory output window misses the transmitted field")

    f_total_out = geometry.external_index / upstream_field.wavelength_m
    required_fx = f_total_out * float(np.max(np.abs(outgoing_lab[..., 0][in_window])))
    required_fy = f_total_out * float(np.max(np.abs(outgoing_lab[..., 1][in_window])))
    nyquist = 0.5 / float(out_grid["dx"])
    required_nyquist_fraction = max(required_fx, required_fy) / max(nyquist, EPS)
    if required_nyquist_fraction > float(maximum_nyquist_fraction):
        raise ValueError(
            "output sampling cannot represent the refracted vector wavevectors: "
            f"required/nyquist={required_nyquist_fraction:.3f} > {maximum_nyquist_fraction:.3f}; "
            "increase output_n or reduce output_window_m"
        )

    remapped, coverage, remap_meta = _phase_safe_remap(
        vector_envelope_rays,
        total_phase,
        in_window,
        X,
        Y,
        x_ref,
        y_ref,
        out_grid,
    )
    raw = VectorField(
        ex=remapped[..., 0],
        ey=remapped[..., 1],
        ez=remapped[..., 2],
        grid=out_grid,
        wavelength_m=upstream_field.wavelength_m,
        medium_index=geometry.external_index,
        metadata={"stage": "vector_refractive_axicon_common_eikonal_pre_projection"},
    )
    pre_projection_residual = float(spectral_transversality_residual(raw))
    projected = propagate_vector_asm(raw, 0.0)
    raster_flux = spectral_normal_flux_au(projected)
    correction_power_ratio = window_flux / max(raster_flux, EPS)
    if abs(correction_power_ratio - 1.0) > float(maximum_interpolation_flux_correction_fraction):
        raise ValueError(
            "irregular-to-regular vector remapping loses too much normal flux: "
            f"power correction ratio={correction_power_ratio:.6f}"
        )
    global_scale = math.sqrt(correction_power_ratio)
    final = VectorField(
        ex=projected.ex * global_scale,
        ey=projected.ey * global_scale,
        ez=projected.ez * global_scale,
        grid=out_grid,
        wavelength_m=projected.wavelength_m,
        medium_index=projected.medium_index,
        metadata={
            "stage": "vector_refractive_axicon_common_eikonal_fixed_lab_plane",
            "reference_z_lab_m": z_ref,
            "reference_grid_center_lab_m": list(map(float, output_center_lab_m)),
        },
    )
    final_flux = spectral_normal_flux_au(final)
    final_transversality = float(spectral_transversality_residual(final))
    spectrum = np.abs(fft2c(final.ex)) ** 2 + np.abs(fft2c(final.ey)) ** 2 + np.abs(fft2c(final.ez)) ** 2
    edge = max(2, int(0.04 * n_out))
    edge_mask = np.zeros_like(spectrum, dtype=bool)
    edge_mask[:edge, :] = True
    edge_mask[-edge:, :] = True
    edge_mask[:, :edge] = True
    edge_mask[:, -edge:] = True
    spectral_edge_fraction = float(np.sum(spectrum[edge_mask]) / max(float(np.sum(spectrum)), EPS))

    bundle = VectorRefractiveAxiconRayBundle(
        entrance_x_m=X,
        entrance_y_m=Y,
        valid=np.asarray(valid, dtype=bool),
        incident_local=np.asarray(incident_local),
        internal_local=np.asarray(internal_local),
        exit_point_local_m=np.asarray(exit_local),
        exit_normal_local=np.asarray(cone_normal),
        outgoing_local=np.asarray(outgoing_local),
        exit_point_lab_m=np.asarray(exit_lab),
        outgoing_lab=np.asarray(outgoing_lab),
        reference_x_lab_m=np.asarray(x_ref),
        reference_y_lab_m=np.asarray(y_ref),
        reference_z_lab_m=float(z_ref),
        reference_distance_m=np.asarray(s_out),
        internal_distance_m=np.asarray(s_inside),
        optical_path_m=np.asarray(opl),
        mapping_jacobian_signed=np.asarray(jac_signed),
        input_normal_flux_density_au=np.asarray(input_flux_density),
        fresnel_power_surface1=np.asarray(T1),
        fresnel_power_surface2=np.asarray(T2),
        fresnel_reflectance_surface1=np.asarray(R1),
        fresnel_reflectance_surface2=np.asarray(R2),
        entrance_reference_phase_rad=np.asarray(common.phase_rad),
        metadata={"common_eikonal": dict(common.metadata)},
    )
    return VectorRefractiveAxiconResult(
        field=final,
        entrance_surface_envelope_lab=envelope,
        geometry_bundle=bundle,
        coverage_mask=np.asarray(coverage, dtype=bool),
        outgoing_direction_lab=np.asarray(outgoing_lab),
        metadata={
            "outcome": "VECTOR-TWO-SURFACE-REFRACTIVE-AXICON-COMMON-EIKONAL",
            "model_class": "common_eikonal_two_surface_Snell_Fresnel_phase_safe_vector_boundary",
            "supersedes": "provisional_total_Poynting_directed_Phase2H_builder",
            "ray_direction_definition": "local_wavevector_grad_Phi",
            "poynting_role": "independent_energy_flux_diagnostic_only",
            "surface_order": "flat_entrance_then_conical_exit",
            "tilt_x_rad": float(tilt_x_rad),
            "tilt_y_rad": float(tilt_y_rad),
            "axicon_decentre_m": [dec_x, dec_y],
            "reference_z_lab_m": float(z_ref),
            "output_n": n_out,
            "output_window_m": window,
            "output_dx_m": float(out_grid["dx"]),
            "valid_ray_fraction": float(np.mean(valid)),
            "coverage_fraction": float(np.mean(coverage)),
            "mapping_positive_fraction": positive_fraction,
            "mapping_negative_fraction": negative_fraction,
            "common_eikonal": dict(common.metadata),
            "p95_local_plane_wave_normal_flux_error_fraction": p95_flux_error,
            "integrated_local_plane_wave_flux_disagreement_fraction": integrated_flux_disagreement,
            "interface1_max_abs_R_plus_T_minus_1": fresnel1_error,
            "interface2_max_abs_R_plus_T_minus_1": fresnel2_error,
            "all_transmitted_flux_au": all_flux,
            "window_transmitted_flux_au": window_flux,
            "window_capture_fraction": window_flux / max(all_flux, EPS),
            "ray_flux_closure_ratio": reconstructed_ray_flux / max(all_flux, EPS),
            "raster_flux_before_global_closure_au": raster_flux,
            "interpolation_flux_power_correction_ratio": correction_power_ratio,
            "interpolation_global_amplitude_correction": global_scale,
            "final_spectral_normal_flux_au": final_flux,
            "final_flux_closure_ratio": final_flux / max(window_flux, EPS),
            "pre_projection_transversality_residual": pre_projection_residual,
            "final_transversality_residual": final_transversality,
            "required_nyquist_fraction": required_nyquist_fraction,
            "spectral_edge_power_fraction": spectral_edge_fraction,
            "phase_piston_removed_rad": phase_piston,
            "entrance_surface_sampling": surface_meta,
            "phase_remap_policy": "common_eikonal_plus_OPL_unwrapped_separate_from_vector_envelope_until_final_lab_grid",
            **remap_meta,
            "physics_limit": (
                "single-common-eikonal macroscopic vector geometrical optics; multimode/nonintegrable vector fields, "
                "coating stacks, multiple internal reflections, full-volume Maxwell scattering and nonlinear material response are separate"
            ),
            "report_figures_authorised": False,
        },
    )


def lab_reference_eikonal_direction_consistency(
    result: VectorRefractiveAxiconResult,
    *,
    wavelength_m: float,
    external_index: float,
    trim_pixels: int = 4,
) -> dict[str, float]:
    """Verify Fermat: gradient of propagated common phase equals traced k_t."""

    b = result.geometry_bundle
    X = np.asarray(b.entrance_x_m, dtype=float)
    Y = np.asarray(b.entrance_y_m, dtype=float)
    dx = float(np.median(np.diff(X[X.shape[0] // 2])))
    dy = float(np.median(np.diff(Y[:, Y.shape[1] // 2])))
    phase = np.asarray(b.entrance_reference_phase_rad) + (TWOPI / float(wavelength_m)) * np.asarray(b.optical_path_m)
    dphase_dy, dphase_dx = np.gradient(phase, dy, dx)
    det, terms = _mapping_jacobian_signed(
        b.reference_x_lab_m,
        b.reference_y_lab_m,
        dx_m=dx,
        dy_m=dy,
    )
    gx = (terms["dY_dy"] * dphase_dx - terms["dX_dy"] * dphase_dy) / np.where(np.abs(det) > 1e-15, det, np.nan)
    gy = (-terms["dY_dx"] * dphase_dx + terms["dX_dx"] * dphase_dy) / np.where(np.abs(det) > 1e-15, det, np.nan)
    k_ext = TWOPI * float(external_index) / float(wavelength_m)
    expected_x = k_ext * np.asarray(b.outgoing_lab[..., 0])
    expected_y = k_ext * np.asarray(b.outgoing_lab[..., 1])
    scale = np.maximum(np.hypot(expected_x, expected_y), 0.01 * k_ext)
    relative = np.hypot(gx - expected_x, gy - expected_y) / scale
    valid = np.asarray(b.valid, dtype=bool) & np.isfinite(relative) & (np.abs(det) > 1e-8)
    trim = int(trim_pixels)
    if trim > 0:
        interior = np.zeros_like(valid)
        interior[trim:-trim, trim:-trim] = True
        valid &= interior
    values = relative[valid]
    if values.size < 32:
        raise ValueError("too few valid rays for Fermat/eikonal consistency")
    return {
        "median_relative_direction_error": float(np.median(values)),
        "p95_relative_direction_error": float(np.percentile(values, 95.0)),
        "max_relative_direction_error": float(np.max(values)),
    }


__all__ = [
    "CommonEikonalEstimate",
    "build_tilted_vector_refractive_axicon_field",
    "estimate_common_vector_eikonal",
    "lab_reference_eikonal_direction_consistency",
]
