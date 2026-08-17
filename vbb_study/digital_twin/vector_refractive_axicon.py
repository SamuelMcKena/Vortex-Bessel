"""First-principles vector boundary field for a rigidly tilted refractive axicon.

A macroscopic plano-conical axicon is treated as two physical dielectric
interfaces.  The input vector wave is sampled on the tilted flat entrance plane
with a carrier-tracked rotated angular spectrum.  The *local electromagnetic
Poynting direction* supplies the geometrical/eikonal ray direction, and the
complex vector envelope is transported through both surfaces by vector Snell law
and local s/p Fresnel amplitudes.

The numerical design is intentionally phase-safe.  A potentially very large
carrier on the tilted entrance plane is never rasterised as a sampled complex
carrier.  The slowly sampled vector envelope and the unwrapped carrier+OPL
eikonal phase are remapped separately and combined only on the final fixed
laboratory plane after a Nyquist gate.  This prevents a tilted-plane alias from
masquerading as vector axicon physics.

The model is a high-frequency vector geometrical-optics/eikonal boundary-field
construction for a macroscopic optic, followed downstream by vector wave
propagation.  It is not a full-volume FDTD/FEM solution of the glass.

Primary formulation/validation literature:
- Yun, Crabtree & Chipman, Applied Optics 50, 2855-2865 (2011).
- Kim, Wang & Zhang, JOSA A 35, 526-535 (2018).
- Zhao Bin & Li Zhu, Applied Optics 37, 2563-2568 (1998).
- Thaning, Jaroszewicz & Friberg, Applied Optics 42, 9-17 (2003).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import map_coordinates

from vbb_study.digital_twin.vortex_refractive_axicon import RefractiveAxiconGeometry
from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix
from vbb_study.digital_twin.vortex_rotated_plane_baseband import rotate_baseband_angular_spectrum
from vbb_study.equations.fields import fft2c, ifft2c, make_xy_grid
from vbb_study.vector_field import VectorField, propagate_vector_asm, spectral_transversality_residual

EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class VectorRefractiveAxiconRayBundle:
    entrance_x_m: np.ndarray
    entrance_y_m: np.ndarray
    valid: np.ndarray
    incident_local: np.ndarray
    internal_local: np.ndarray
    exit_point_local_m: np.ndarray
    exit_normal_local: np.ndarray
    outgoing_local: np.ndarray
    exit_point_lab_m: np.ndarray
    outgoing_lab: np.ndarray
    reference_x_lab_m: np.ndarray
    reference_y_lab_m: np.ndarray
    reference_z_lab_m: float
    reference_distance_m: np.ndarray
    internal_distance_m: np.ndarray
    optical_path_m: np.ndarray
    mapping_jacobian_signed: np.ndarray
    input_normal_flux_density_au: np.ndarray
    fresnel_power_surface1: np.ndarray
    fresnel_power_surface2: np.ndarray
    fresnel_reflectance_surface1: np.ndarray
    fresnel_reflectance_surface2: np.ndarray
    entrance_reference_phase_rad: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VectorRefractiveAxiconResult:
    field: VectorField
    entrance_surface_envelope_lab: VectorField
    geometry_bundle: VectorRefractiveAxiconRayBundle
    coverage_mask: np.ndarray
    outgoing_direction_lab: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def reference_x_lab_m(self) -> np.ndarray:
        return self.geometry_bundle.reference_x_lab_m

    @property
    def reference_y_lab_m(self) -> np.ndarray:
        return self.geometry_bundle.reference_y_lab_m

    @property
    def reference_z_lab_m(self) -> float:
        return float(self.geometry_bundle.reference_z_lab_m)

    @property
    def optical_path_m(self) -> np.ndarray:
        return self.geometry_bundle.optical_path_m


def _strict_unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError("vector arrays must have final dimension 3")
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError("zero-length vector")
    return arr / norm


def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    valid = np.squeeze(norm > np.finfo(float).eps, axis=-1)
    fb = np.broadcast_to(np.asarray(fallback, dtype=float), arr.shape)
    safe = np.where(valid[..., None], arr, fb)
    return _strict_unit(safe), valid


def _broadcast_vectors(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    values = [np.asarray(value) for value in arrays]
    for value in values:
        if value.shape[-1] != 3:
            raise ValueError("all vector inputs must have final dimension 3")
    lead = np.broadcast_shapes(*(value.shape[:-1] for value in values))
    target = lead + (3,)
    return tuple(np.broadcast_to(value, target) for value in values)


def _vector_spectral_centroid_cpm(field: VectorField) -> tuple[float, float]:
    weight = (
        np.abs(fft2c(field.ex)) ** 2
        + np.abs(fft2c(field.ey)) ** 2
        + np.abs(fft2c(field.ez)) ** 2
    )
    total = float(np.sum(weight))
    if total <= EPS:
        return 0.0, 0.0
    FX = np.asarray(field.grid["FX"], dtype=float)
    FY = np.asarray(field.grid["FY"], dtype=float)
    return float(np.sum(weight * FX) / total), float(np.sum(weight * FY) / total)


def _surface_maxwell_projection_and_poynting(
    components_env: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    medium_index: float,
    carrier_local_cpm: tuple[float, float],
    rotation_local_to_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Project the carrier-tracked spectrum transverse and reconstruct Poynting.

    ``H * Z0 = n (khat x E)`` is used, so the returned Poynting vector omits only
    a common positive ``1/(2 Z0)`` factor.  That factor cancels from directions
    and flux ratios.
    """

    spectrum = np.stack([fft2c(np.asarray(v, dtype=np.complex128)) for v in components_env], axis=-1)
    FX = np.asarray(grid["FX"], dtype=float)
    FY = np.asarray(grid["FY"], dtype=float)
    fc_x, fc_y = map(float, carrier_local_cpm)
    f_total = float(medium_index) / float(wavelength_m)
    fx = fc_x + FX
    fy = fc_y + FY
    transverse_sq = fx * fx + fy * fy
    propagating = transverse_sq < f_total * f_total
    fz = np.sqrt(np.maximum(f_total * f_total - transverse_sq, 0.0))
    khat_local = np.stack([fx, fy, fz], axis=-1) / max(f_total, EPS)
    khat_lab = khat_local @ np.asarray(rotation_local_to_lab, dtype=float).T

    dot = np.sum(spectrum * khat_lab, axis=-1, keepdims=True)
    projected = np.where(propagating[..., None], spectrum - dot * khat_lab, 0.0j)
    h_spec = float(medium_index) * np.cross(khat_lab, projected)
    e_env = np.stack([ifft2c(projected[..., i]) for i in range(3)], axis=-1)
    h_env = np.stack([ifft2c(h_spec[..., i]) for i in range(3)], axis=-1)
    poynting = np.real(np.cross(e_env, np.conj(h_env)))

    trans_num = float(np.sum(np.abs(np.sum(projected * khat_lab, axis=-1)) ** 2))
    trans_den = max(float(np.sum(np.abs(projected) ** 2)), EPS)
    retained = float(np.sum(np.abs(projected) ** 2) / max(float(np.sum(np.abs(spectrum) ** 2)), EPS))
    return e_env, poynting, {
        "spectral_transversality_power_ratio": trans_num / trans_den,
        "transverse_projection_spectral_energy_retained": retained,
        "propagating_spectral_fraction": float(np.mean(propagating)),
    }


def sample_vector_field_on_tilted_entrance(
    field: VectorField,
    *,
    tilt_x_rad: float,
    tilt_y_rad: float,
    spectral_center_cpm: tuple[float, float] | None = None,
) -> tuple[VectorField, np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample the vector wave on the physical tilted flat entrance plane.

    The returned field is a *baseband envelope* in fixed laboratory Cartesian
    components.  The destination carrier is returned separately and is not
    sampled as ``exp(i 2pi f x)`` on this grid.
    """

    source = propagate_vector_asm(field, 0.0)
    fsx, fsy = (
        _vector_spectral_centroid_cpm(source)
        if spectral_center_cpm is None
        else tuple(map(float, spectral_center_cpm))
    )
    f_total = float(field.medium_index) / float(field.wavelength_m)
    if fsx * fsx + fsy * fsy >= f_total * f_total:
        raise ValueError("source vector spectral centre is non-propagating")

    X = np.asarray(source.grid["X"], dtype=float)
    Y = np.asarray(source.grid["Y"], dtype=float)
    demod = np.exp(-1j * TWOPI * (fsx * X + fsy * Y))
    source_env = (source.ex * demod, source.ey * demod, source.ez * demod)
    effective_wavelength = float(field.wavelength_m) / float(field.medium_index)

    mapped: list[np.ndarray] = []
    metas: list[Mapping[str, Any]] = []
    for component in source_env:
        env, meta = rotate_baseband_angular_spectrum(
            component,
            source.grid,
            wavelength_m=effective_wavelength,
            tilt_x_rad=float(tilt_x_rad),
            tilt_y_rad=float(tilt_y_rad),
            source_spectral_center_cpm=(fsx, fsy),
        )
        mapped.append(np.asarray(env, dtype=np.complex128))
        metas.append(dict(meta))

    carrier = tuple(map(float, metas[0]["destination_spectral_center_cpm"]))
    for meta in metas[1:]:
        if not np.allclose(meta["destination_spectral_center_cpm"], carrier, rtol=0.0, atol=1e-9):
            raise RuntimeError("vector components acquired inconsistent entrance-plane carriers")

    rotation = rotation_matrix(float(tilt_x_rad), float(tilt_y_rad))
    e_env, poynting, maxwell_meta = _surface_maxwell_projection_and_poynting(
        (mapped[0], mapped[1], mapped[2]),
        source.grid,
        wavelength_m=field.wavelength_m,
        medium_index=field.medium_index,
        carrier_local_cpm=carrier,
        rotation_local_to_lab=rotation,
    )
    envelope = VectorField(
        ex=e_env[..., 0], ey=e_env[..., 1], ez=e_env[..., 2],
        grid=source.grid, wavelength_m=field.wavelength_m, medium_index=field.medium_index,
        metadata={
            **dict(field.metadata),
            "stage": "tilted_axicon_entrance_baseband_vector_envelope",
            "field_components_basis": "fixed_lab_xyz",
            "surface_coordinates": "axicon_local_xy",
            "analytic_carrier_local_cpm": list(carrier),
            "carrier_representation": "analytic_unsampled",
        },
    )
    return envelope, np.asarray(carrier, dtype=float), poynting, {
        "source_transversality_residual": float(spectral_transversality_residual(source)),
        "source_spectral_center_cpm": [float(fsx), float(fsy)],
        "destination_spectral_center_local_cpm": list(carrier),
        "component_rotated_plane_metadata": metas,
        **maxwell_meta,
    }


def _refract_masked(direction1: np.ndarray, normal12: np.ndarray, *, n1: float, n2: float) -> tuple[np.ndarray, np.ndarray]:
    d1, normal = _broadcast_vectors(np.asarray(direction1, dtype=float), np.asarray(normal12, dtype=float))
    d1 = _strict_unit(d1)
    normal = _strict_unit(normal)
    cos_i = np.sum(d1 * normal, axis=-1, keepdims=True)
    tangent1 = d1 - cos_i * normal
    tangent2 = (float(n1) / float(n2)) * tangent1
    radicand = 1.0 - np.sum(tangent2 * tangent2, axis=-1, keepdims=True)
    valid = np.squeeze((cos_i > 0.0) & (radicand >= -1e-12), axis=-1)
    d2 = tangent2 + np.sqrt(np.maximum(radicand, 0.0)) * normal
    d2, nonzero = _safe_unit(d2, normal)
    return d2, valid & nonzero


def _fallback_s_axis(direction: np.ndarray) -> np.ndarray:
    d = _strict_unit(direction)
    x = np.broadcast_to(np.asarray([1.0, 0.0, 0.0]), d.shape)
    y = np.broadcast_to(np.asarray([0.0, 1.0, 0.0]), d.shape)
    trial = np.where((np.abs(d[..., 0]) < 0.9)[..., None], x, y)
    return _strict_unit(np.cross(d, trial))


def fresnel_transmit_vector_3d(
    electric1: np.ndarray,
    direction1: np.ndarray,
    direction2: np.ndarray,
    normal12: np.ndarray,
    *,
    n1: float,
    n2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply local s/p Fresnel transmission to an arbitrary complex 3-D E field."""

    n1f, n2f = float(n1), float(n2)
    if n1f <= 0.0 or n2f <= 0.0:
        raise ValueError("refractive indices must be positive")
    e, d1, d2, normal = _broadcast_vectors(
        np.asarray(electric1, dtype=np.complex128),
        np.asarray(direction1, dtype=float),
        np.asarray(direction2, dtype=float),
        np.asarray(normal12, dtype=float),
    )
    d1, d2, normal = _strict_unit(d1), _strict_unit(d2), _strict_unit(normal)
    e = e - np.sum(e * d1, axis=-1, keepdims=True) * d1
    cos_i = np.sum(d1 * normal, axis=-1)
    cos_t = np.sum(d2 * normal, axis=-1)
    if np.any(cos_i <= 0.0) or np.any(cos_t <= 0.0):
        raise ValueError("Fresnel transmission requires forward-going rays")

    s_raw = np.cross(normal, d1)
    near_normal = np.linalg.norm(s_raw, axis=-1) < 1e-12
    s_axis = np.where(near_normal[..., None], _fallback_s_axis(d1), s_raw)
    s_axis = _strict_unit(s_axis)
    p1 = _strict_unit(np.cross(d1, s_axis))
    p2 = _strict_unit(np.cross(d2, s_axis))

    ts = 2.0 * n1f * cos_i / (n1f * cos_i + n2f * cos_t)
    tp = 2.0 * n1f * cos_i / (n2f * cos_i + n1f * cos_t)
    rs = (n1f * cos_i - n2f * cos_t) / (n1f * cos_i + n2f * cos_t)
    rp = (n2f * cos_i - n1f * cos_t) / (n2f * cos_i + n1f * cos_t)
    es = np.sum(e * s_axis, axis=-1)
    ep = np.sum(e * p1, axis=-1)
    et = ts[..., None] * es[..., None] * s_axis + tp[..., None] * ep[..., None] * p2
    if np.any(near_normal):
        t0 = 2.0 * n1f / (n1f + n2f)
        e2 = e - np.sum(e * d2, axis=-1, keepdims=True) * d2
        et = np.where(near_normal[..., None], t0 * e2, et)

    ein2 = np.sum(np.abs(e) ** 2, axis=-1)
    eout2 = np.sum(np.abs(et) ** 2, axis=-1)
    T = n2f * cos_t * eout2 / np.maximum(n1f * cos_i * ein2, EPS)
    R = (np.abs(rs * es) ** 2 + np.abs(rp * ep) ** 2) / np.maximum(ein2, EPS)
    if np.any(near_normal):
        r0 = (n1f - n2f) / (n1f + n2f)
        R = np.where(near_normal, abs(r0) ** 2, R)
    dark = ein2 <= 100.0 * EPS
    return et, np.where(dark, 0.0, T), np.where(dark, 0.0, R)


def _cone_intersection_distance(
    x0: np.ndarray,
    y0: np.ndarray,
    direction: np.ndarray,
    *,
    base_angle_rad: float,
    centre_thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=float)
    sx, sy, sz = d[..., 0], d[..., 1], d[..., 2]
    x, y = np.asarray(x0, dtype=float), np.asarray(y0, dtype=float)
    m = math.tan(float(base_angle_rad))
    tc = float(centre_thickness_m)
    a = sz * sz - m * m * (sx * sx + sy * sy)
    b = -2.0 * (tc * sz + m * m * (x * sx + y * sy))
    c = tc * tc - m * m * (x * x + y * y)
    disc = b * b - 4.0 * a * c
    linear = np.abs(a) < 1e-14
    root_disc = np.sqrt(np.maximum(disc, 0.0))
    safe_a = np.where(linear, 1.0, a)
    r1 = (-b - root_disc) / (2.0 * safe_a)
    r2 = (-b + root_disc) / (2.0 * safe_a)
    linear_root = -c / np.where(np.abs(b) > 1e-14, b, np.nan)
    r1 = np.where(linear, linear_root, r1)
    r2 = np.where(linear, np.nan, r2)
    root = np.where((r1 > 0) & (r2 > 0), np.minimum(r1, r2), np.where(r1 > 0, r1, np.where(r2 > 0, r2, np.nan)))
    return root, np.isfinite(root) & (root > 0.0) & (disc >= -1e-12)


def _mapping_jacobian_signed(xr: np.ndarray, yr: np.ndarray, *, dx_m: float, dy_m: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dX_dy, dX_dx = np.gradient(np.asarray(xr, dtype=float), float(dy_m), float(dx_m))
    dY_dy, dY_dx = np.gradient(np.asarray(yr, dtype=float), float(dy_m), float(dx_m))
    det = dX_dx * dY_dy - dX_dy * dY_dx
    return det, {"dX_dx": dX_dx, "dX_dy": dX_dy, "dY_dx": dY_dx, "dY_dy": dY_dy}


def _output_grid(n: int, window_m: float, centre_m: tuple[float, float]) -> dict[str, Any]:
    grid = dict(make_xy_grid(int(n), float(window_m) / int(n)))
    base = np.asarray(grid["x"], dtype=float)
    cx, cy = map(float, centre_m)
    grid["x"] = base + cx
    grid["y"] = base + cy
    grid["X"] = np.asarray(grid["X"], dtype=float) + cx
    grid["Y"] = np.asarray(grid["Y"], dtype=float) + cy
    grid["R"] = np.hypot(grid["X"], grid["Y"])
    grid["PHI"] = np.arctan2(grid["Y"], grid["X"])
    return grid


def _sample_regular(values: np.ndarray, iy: np.ndarray, ix: np.ndarray, *, order: int) -> np.ndarray:
    arr = np.asarray(values)
    coords = np.vstack([iy.ravel(), ix.ravel()])
    if np.iscomplexobj(arr):
        re = map_coordinates(arr.real, coords, order=order, mode="constant", cval=0.0, prefilter=order > 1)
        im = map_coordinates(arr.imag, coords, order=order, mode="constant", cval=0.0, prefilter=order > 1)
        return (re + 1j * im).reshape(ix.shape)
    out = map_coordinates(arr.astype(float), coords, order=order, mode="constant", cval=np.nan, prefilter=order > 1)
    return out.reshape(ix.shape)


def _inverse_map_coordinates(
    valid: np.ndarray,
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    output_grid: Mapping[str, Any],
    *,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    X, Y = np.asarray(x_in, dtype=float), np.asarray(y_in, dtype=float)
    XR, YR = np.asarray(x_ref, dtype=float), np.asarray(y_ref, dtype=float)
    valid_ray = np.asarray(valid, dtype=bool)
    x_axis = np.asarray(X[X.shape[0] // 2], dtype=float)
    y_axis = np.asarray(Y[:, Y.shape[1] // 2], dtype=float)
    dx = float(np.median(np.diff(x_axis)))
    dy = float(np.median(np.diff(y_axis)))
    ids = np.flatnonzero(valid_ray.ravel())
    if ids.size < 64:
        raise ValueError("too few valid rays for inverse field mapping")

    ids_fit = ids[:: max(1, ids.size // 4096)]
    A = np.column_stack([X.ravel()[ids_fit], Y.ravel()[ids_fit], np.ones(ids_fit.size)])
    cx, *_ = np.linalg.lstsq(A, XR.ravel()[ids_fit], rcond=None)
    cy, *_ = np.linalg.lstsq(A, YR.ravel()[ids_fit], rcond=None)
    affine = np.asarray([[cx[0], cx[1]], [cy[0], cy[1]]])
    if abs(float(np.linalg.det(affine))) < 1e-12:
        raise ValueError("ray-map affine inverse seed is singular")
    inv_affine = np.linalg.inv(affine)
    offset = np.asarray([cx[2], cy[2]])
    XT, YT = np.asarray(output_grid["X"], dtype=float), np.asarray(output_grid["Y"], dtype=float)
    seed = np.stack([XT - offset[0], YT - offset[1]], axis=-1) @ inv_affine.T
    xq, yq = seed[..., 0], seed[..., 1]

    _det, terms = _mapping_jacobian_signed(XR, YR, dx_m=dx, dy_m=dy)
    active = np.ones_like(XT, dtype=bool)
    residual = np.full_like(XT, np.inf)
    for _ in range(int(iterations)):
        ix = (xq - x_axis[0]) / dx
        iy = (yq - y_axis[0]) / dy
        xr, yr = _sample_regular(XR, iy, ix, order=1), _sample_regular(YR, iy, ix, order=1)
        dXdx = _sample_regular(terms["dX_dx"], iy, ix, order=1)
        dXdy = _sample_regular(terms["dX_dy"], iy, ix, order=1)
        dYdx = _sample_regular(terms["dY_dx"], iy, ix, order=1)
        dYdy = _sample_regular(terms["dY_dy"], iy, ix, order=1)
        rx, ry = xr - XT, yr - YT
        det = dXdx * dYdy - dXdy * dYdx
        ok = np.isfinite(rx) & np.isfinite(ry) & np.isfinite(det) & (np.abs(det) > 1e-10)
        active &= ok
        safe = np.where(ok, det, 1.0)
        xq = np.where(ok, xq - (dYdy * rx - dXdy * ry) / safe, xq)
        yq = np.where(ok, yq - (-dYdx * rx + dXdx * ry) / safe, yq)
        residual = np.hypot(rx, ry)

    ix = (xq - x_axis[0]) / dx
    iy = (yq - y_axis[0]) / dy
    valid_sample = _sample_regular(valid_ray.astype(float), iy, ix, order=1)
    tolerance = max(0.20 * float(output_grid["dx"]), 1e-10)
    coverage = (
        active & np.isfinite(valid_sample) & (valid_sample > 0.95)
        & np.isfinite(residual) & (residual <= tolerance)
        & (ix >= 0.0) & (ix <= X.shape[1] - 1.0)
        & (iy >= 0.0) & (iy <= X.shape[0] - 1.0)
    )
    rv = residual[coverage]
    return iy, ix, coverage, {
        "inverse_mapping_coverage_fraction": float(np.mean(coverage)),
        "inverse_mapping_median_residual_m": float(np.median(rv)) if rv.size else float("nan"),
        "inverse_mapping_p95_residual_m": float(np.percentile(rv, 95.0)) if rv.size else float("nan"),
        "inverse_mapping_max_residual_m": float(np.max(rv)) if rv.size else float("nan"),
        "inverse_mapping_tolerance_m": float(tolerance),
    }


def _phase_safe_remap(
    vector_envelope: np.ndarray,
    unwrapped_phase_rad: np.ndarray,
    valid: np.ndarray,
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    output_grid: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    iy, ix, coverage, meta = _inverse_map_coordinates(valid, x_in, y_in, x_ref, y_ref, output_grid)
    components = []
    for i in range(3):
        env = _sample_regular(np.asarray(vector_envelope)[..., i], iy, ix, order=3)
        components.append(np.where(coverage, env, 0.0j))
    envelope_regular = np.stack(components, axis=-1)
    phase_regular = _sample_regular(np.asarray(unwrapped_phase_rad, dtype=float), iy, ix, order=1)
    phase_regular = np.where(coverage, phase_regular, 0.0)
    return envelope_regular * np.exp(1j * phase_regular)[..., None], coverage, meta


def spectral_normal_flux_au(field: VectorField) -> float:
    ax, ay, az = fft2c(field.ex), fft2c(field.ey), fft2c(field.ez)
    FX, FY = np.asarray(field.grid["FX"], dtype=float), np.asarray(field.grid["FY"], dtype=float)
    f_total = float(field.medium_index) / float(field.wavelength_m)
    propagating = FX * FX + FY * FY < f_total * f_total
    fz = np.sqrt(np.maximum(f_total * f_total - FX * FX - FY * FY, 0.0))
    cos_theta = np.where(propagating, fz / max(f_total, EPS), 0.0)
    energy = np.abs(ax) ** 2 + np.abs(ay) ** 2 + np.abs(az) ** 2
    ny, nx = field.ex.shape
    dx, dy = float(field.grid["dx"]), float(field.grid.get("dy", field.grid["dx"]))
    return float(field.medium_index) * dx * dy / float(nx * ny) * float(np.sum(cos_theta * energy))


def _dominant_envelope_phase(electric_env: np.ndarray) -> tuple[np.ndarray, int]:
    energies = [float(np.sum(np.abs(electric_env[..., i]) ** 2)) for i in range(3)]
    index = int(np.argmax(energies))
    phase = np.angle(electric_env[..., index])
    return np.unwrap(np.unwrap(phase, axis=1), axis=0), index


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
    maximum_local_nontransverse_power_fraction: float = 0.02,
) -> VectorRefractiveAxiconResult:
    """Trace, polarisation-transport and remap the tilted refractive axicon field."""

    geometry.validate()
    if not np.isclose(upstream_field.medium_index, geometry.external_index, rtol=0.0, atol=1e-12):
        raise ValueError("upstream field medium must equal axicon external medium")

    envelope, carrier_local, poynting_lab, surface_meta = sample_vector_field_on_tilted_entrance(
        upstream_field, tilt_x_rad=float(tilt_x_rad), tilt_y_rad=float(tilt_y_rad)
    )
    grid = envelope.grid
    X, Y = np.asarray(grid["X"], dtype=float), np.asarray(grid["Y"], dtype=float)
    dx, dy = float(grid["dx"]), float(grid.get("dy", grid["dx"]))
    rotation = rotation_matrix(float(tilt_x_rad), float(tilt_y_rad))
    dec_x, dec_y = map(float, axicon_decentre_m)
    X_rel, Y_rel = X - dec_x, Y - dec_y

    e_env_lab = np.stack([envelope.ex, envelope.ey, envelope.ez], axis=-1)
    e_env_local = e_env_lab @ rotation
    poynting_local = np.asarray(poynting_lab, dtype=float) @ rotation
    incident_local, nonzero_flux = _safe_unit(poynting_local, np.asarray([0.0, 0.0, 1.0]))
    entrance_forward = incident_local[..., 2] > 0.0
    surface_intensity = np.sum(np.abs(e_env_local) ** 2, axis=-1)
    bright = surface_intensity > 1e-12 * max(float(np.max(surface_intensity)), EPS)
    e_transverse = e_env_local - np.sum(e_env_local * incident_local, axis=-1, keepdims=True) * incident_local
    transverse_intensity = np.sum(np.abs(e_transverse) ** 2, axis=-1)
    nontransverse_fraction = np.where(
        surface_intensity > EPS,
        np.maximum(surface_intensity - transverse_intensity, 0.0) / np.maximum(surface_intensity, EPS),
        0.0,
    )
    p95_nontransverse = float(np.percentile(nontransverse_fraction[bright], 95.0)) if np.any(bright) else 1.0
    if p95_nontransverse > float(maximum_local_nontransverse_power_fraction):
        raise ValueError(
            "local plane-wave/eikonal approximation is not valid for this entrance field: "
            f"p95 non-transverse power fraction={p95_nontransverse:.4g}"
        )

    flat_normal = np.asarray([0.0, 0.0, 1.0])
    internal_local, snell1_valid = _refract_masked(
        incident_local, flat_normal, n1=geometry.external_index, n2=geometry.refractive_index
    )
    safe_e = np.where((nonzero_flux & entrance_forward & bright)[..., None], e_transverse, 0.0j)
    e_internal_env, T1, R1 = fresnel_transmit_vector_3d(
        safe_e, incident_local, internal_local, flat_normal,
        n1=geometry.external_index, n2=geometry.refractive_index
    )

    internal_distance, hit_cone = _cone_intersection_distance(
        X_rel, Y_rel, internal_local,
        base_angle_rad=geometry.base_angle_rad,
        centre_thickness_m=geometry.centre_thickness_m,
    )
    s_inside = np.where(hit_cone, internal_distance, 0.0)
    exit_x = X_rel + s_inside * internal_local[..., 0]
    exit_y = Y_rel + s_inside * internal_local[..., 1]
    exit_z = s_inside * internal_local[..., 2]
    exit_radius = np.hypot(exit_x, exit_y)
    phi = np.arctan2(exit_y, exit_x)
    sg, cg = math.sin(geometry.base_angle_rad), math.cos(geometry.base_angle_rad)
    cone_normal = np.stack([sg * np.cos(phi), sg * np.sin(phi), np.full_like(phi, cg)], axis=-1)
    outgoing_local, snell2_valid = _refract_masked(
        internal_local, cone_normal, n1=geometry.refractive_index, n2=geometry.external_index
    )
    e_exit_env_local, T2, R2 = fresnel_transmit_vector_3d(
        np.where(hit_cone[..., None], e_internal_env, 0.0j),
        internal_local, outgoing_local, cone_normal,
        n1=geometry.refractive_index, n2=geometry.external_index,
    )

    valid = (
        nonzero_flux & entrance_forward & bright & snell1_valid & hit_cone & snell2_valid
        & (exit_radius <= geometry.clear_radius_m)
        & (exit_radius >= float(apex_exclusion_radius_m))
    )
    exit_local = np.stack([exit_x, exit_y, exit_z], axis=-1)
    tangent_offset_lab = rotation @ np.asarray([dec_x, dec_y, 0.0])
    exit_lab = exit_local @ rotation.T + tangent_offset_lab
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
    jac_signed, _terms = _mapping_jacobian_signed(x_ref, y_ref, dx_m=dx, dy_m=dy)
    jac_abs = np.abs(jac_signed)
    valid &= np.isfinite(jac_abs) & (jac_abs > float(minimum_mapping_jacobian))

    interior = valid.copy()
    if min(interior.shape) > 8:
        interior[:3, :] = interior[-3:, :] = False
        interior[:, :3] = interior[:, -3:] = False
    signs = jac_signed[interior]
    positive_fraction = float(np.mean(signs > 0.0)) if signs.size else 0.0
    negative_fraction = float(np.mean(signs < 0.0)) if signs.size else 0.0
    if positive_fraction > 0.02 and negative_fraction > 0.02:
        raise ValueError("ray map folds before the fixed laboratory reference plane")

    input_flux_density = float(geometry.external_index) * incident_local[..., 2] * transverse_intensity
    cumulative_T = T1 * T2
    norm_exit = np.sqrt(np.sum(np.abs(e_exit_env_local) ** 2, axis=-1))
    unit_exit_local = np.divide(
        e_exit_env_local, norm_exit[..., None], out=np.zeros_like(e_exit_env_local),
        where=norm_exit[..., None] > 100.0 * EPS,
    )
    unit_exit_lab = unit_exit_local @ rotation.T
    desired_intensity = input_flux_density * cumulative_T / np.maximum(
        float(geometry.external_index) * outgoing_lab[..., 2] * jac_abs, EPS
    )
    vector_envelope_rays = unit_exit_lab * np.sqrt(np.maximum(desired_intensity, 0.0))[..., None]
    vector_envelope_rays = np.where(valid[..., None], vector_envelope_rays, 0.0j)

    opl = geometry.refractive_index * s_inside + geometry.external_index * s_out
    opl_ref = float(np.median(opl[valid]))
    carrier_phase = TWOPI * (float(carrier_local[0]) * X + float(carrier_local[1]) * Y)
    unwrapped_eikonal_phase = carrier_phase + (TWOPI / upstream_field.wavelength_m) * (opl - opl_ref)
    envelope_phase, dominant_component = _dominant_envelope_phase(e_env_lab)
    entrance_reference_phase = envelope_phase + carrier_phase

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
    x_out, y_out = np.asarray(out_grid["x"], dtype=float), np.asarray(out_grid["y"], dtype=float)
    in_window = (
        valid & (x_ref >= x_out[0]) & (x_ref <= x_out[-1])
        & (y_ref >= y_out[0]) & (y_ref <= y_out[-1])
    )
    window_flux = float(np.sum(input_flux_density[in_window] * cumulative_T[in_window]) * dx * dy)
    if window_flux <= EPS:
        raise ValueError("fixed laboratory output window misses the transmitted field")

    f_total_out = geometry.external_index / upstream_field.wavelength_m
    req_fx = f_total_out * float(np.max(np.abs(outgoing_lab[..., 0][in_window])))
    req_fy = f_total_out * float(np.max(np.abs(outgoing_lab[..., 1][in_window])))
    nyquist = 0.5 / float(out_grid["dx"])
    required_nyquist_fraction = max(req_fx, req_fy) / max(nyquist, EPS)
    if required_nyquist_fraction > float(maximum_nyquist_fraction):
        raise ValueError(
            "output sampling cannot represent the refracted vector wavevectors: "
            f"required/nyquist={required_nyquist_fraction:.3f} > {maximum_nyquist_fraction:.3f}; "
            "increase output_n or reduce output_window_m"
        )

    remapped, coverage, remap_meta = _phase_safe_remap(
        vector_envelope_rays,
        unwrapped_eikonal_phase,
        in_window,
        X, Y, x_ref, y_ref, out_grid,
    )
    raw = VectorField(
        ex=remapped[..., 0], ey=remapped[..., 1], ez=remapped[..., 2],
        grid=out_grid, wavelength_m=upstream_field.wavelength_m,
        medium_index=geometry.external_index,
        metadata={"stage": "vector_refractive_axicon_lab_plane_pre_projection"},
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
    scale = math.sqrt(correction_power_ratio)
    final = VectorField(
        ex=projected.ex * scale, ey=projected.ey * scale, ez=projected.ez * scale,
        grid=out_grid, wavelength_m=projected.wavelength_m, medium_index=projected.medium_index,
        metadata={
            "stage": "vector_refractive_axicon_fixed_lab_reference_plane",
            "reference_z_lab_m": z_ref,
            "reference_grid_center_lab_m": list(map(float, output_center_lab_m)),
        },
    )
    final_flux = spectral_normal_flux_au(final)
    final_transversality = float(spectral_transversality_residual(final))
    spectrum = np.abs(fft2c(final.ex)) ** 2 + np.abs(fft2c(final.ey)) ** 2 + np.abs(fft2c(final.ez)) ** 2
    edge = max(2, int(0.04 * n_out))
    edge_mask = np.zeros_like(spectrum, dtype=bool)
    edge_mask[:edge, :] = edge_mask[-edge:, :] = True
    edge_mask[:, :edge] = edge_mask[:, -edge:] = True
    spectral_edge_fraction = float(np.sum(spectrum[edge_mask]) / max(float(np.sum(spectrum)), EPS))

    bundle = VectorRefractiveAxiconRayBundle(
        entrance_x_m=X, entrance_y_m=Y, valid=np.asarray(valid, dtype=bool),
        incident_local=np.asarray(incident_local), internal_local=np.asarray(internal_local),
        exit_point_local_m=np.asarray(exit_local), exit_normal_local=np.asarray(cone_normal),
        outgoing_local=np.asarray(outgoing_local), exit_point_lab_m=np.asarray(exit_lab),
        outgoing_lab=np.asarray(outgoing_lab), reference_x_lab_m=np.asarray(x_ref),
        reference_y_lab_m=np.asarray(y_ref), reference_z_lab_m=float(z_ref),
        reference_distance_m=np.asarray(s_out), internal_distance_m=np.asarray(s_inside),
        optical_path_m=np.asarray(opl), mapping_jacobian_signed=np.asarray(jac_signed),
        input_normal_flux_density_au=np.asarray(input_flux_density),
        fresnel_power_surface1=np.asarray(T1), fresnel_power_surface2=np.asarray(T2),
        fresnel_reflectance_surface1=np.asarray(R1), fresnel_reflectance_surface2=np.asarray(R2),
        entrance_reference_phase_rad=np.asarray(entrance_reference_phase),
        metadata={"dominant_envelope_phase_component_index": dominant_component},
    )
    return VectorRefractiveAxiconResult(
        field=final,
        entrance_surface_envelope_lab=envelope,
        geometry_bundle=bundle,
        coverage_mask=np.asarray(coverage, dtype=bool),
        outgoing_direction_lab=np.asarray(outgoing_lab),
        metadata={
            "outcome": "VECTOR-TWO-SURFACE-REFRACTIVE-AXICON-LAB-PLANE",
            "model_class": "local_Poynting_vector_eikonal_two_surface_Snell_Fresnel_phase_safe_remap",
            "surface_order": "flat_entrance_then_conical_exit",
            "tilt_x_rad": float(tilt_x_rad), "tilt_y_rad": float(tilt_y_rad),
            "axicon_decentre_m": [dec_x, dec_y],
            "reference_z_lab_m": float(z_ref),
            "output_n": n_out, "output_window_m": window, "output_dx_m": float(out_grid["dx"]),
            "valid_ray_fraction": float(np.mean(valid)),
            "output_window_ray_fraction": float(np.count_nonzero(in_window) / max(np.count_nonzero(valid), 1)),
            "coverage_fraction": float(np.mean(coverage)),
            "mapping_positive_fraction": positive_fraction,
            "mapping_negative_fraction": negative_fraction,
            "p95_local_nontransverse_power_fraction": p95_nontransverse,
            "interface1_max_abs_R_plus_T_minus_1": fresnel1_error,
            "interface2_max_abs_R_plus_T_minus_1": fresnel2_error,
            "all_transmitted_flux_au": all_flux,
            "window_transmitted_flux_au": window_flux,
            "window_capture_fraction": window_flux / max(all_flux, EPS),
            "ray_flux_closure_ratio": reconstructed_ray_flux / max(all_flux, EPS),
            "raster_flux_before_global_closure_au": raster_flux,
            "interpolation_flux_power_correction_ratio": correction_power_ratio,
            "interpolation_global_amplitude_correction": scale,
            "final_spectral_normal_flux_au": final_flux,
            "final_flux_closure_ratio": final_flux / max(window_flux, EPS),
            "pre_projection_transversality_residual": pre_projection_residual,
            "final_transversality_residual": final_transversality,
            "required_nyquist_fraction": required_nyquist_fraction,
            "spectral_edge_power_fraction": spectral_edge_fraction,
            "opl_reference_m": opl_ref,
            "entrance_surface_sampling": surface_meta,
            "phase_remap_policy": "baseband_vector_envelope_plus_unwrapped_carrier_and_OPL_combined_only_on_final_lab_grid",
            **remap_meta,
            "physics_limit": (
                "macroscopic vector eikonal/Fresnel boundary field; coating stacks, multiple internal reflections, "
                "measured surface figure, full-volume Maxwell scattering and nonlinear material response are separate"
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
    """Check the remapped eikonal gradient against independently traced rays."""

    b = result.geometry_bundle
    X, Y = np.asarray(b.entrance_x_m), np.asarray(b.entrance_y_m)
    dx = float(np.median(np.diff(X[X.shape[0] // 2])))
    dy = float(np.median(np.diff(Y[:, Y.shape[1] // 2])))
    phase = np.asarray(b.entrance_reference_phase_rad) + (TWOPI / float(wavelength_m)) * np.asarray(b.optical_path_m)
    dphase_dy, dphase_dx = np.gradient(phase, dy, dx)
    det, terms = _mapping_jacobian_signed(b.reference_x_lab_m, b.reference_y_lab_m, dx_m=dx, dy_m=dy)
    gx = (terms["dY_dy"] * dphase_dx - terms["dX_dy"] * dphase_dy) / np.where(np.abs(det) > 1e-15, det, np.nan)
    gy = (-terms["dY_dx"] * dphase_dx + terms["dX_dx"] * dphase_dy) / np.where(np.abs(det) > 1e-15, det, np.nan)
    k_ext = TWOPI * float(external_index) / float(wavelength_m)
    expected_x = k_ext * np.asarray(b.outgoing_lab[..., 0])
    expected_y = k_ext * np.asarray(b.outgoing_lab[..., 1])
    scale = np.maximum(np.hypot(expected_x, expected_y), 0.01 * k_ext)
    error = np.hypot(gx - expected_x, gy - expected_y) / scale
    valid = np.asarray(b.valid) & np.isfinite(error) & (np.abs(det) > 1e-8)
    trim = int(trim_pixels)
    if trim > 0:
        interior = np.zeros_like(valid)
        interior[trim:-trim, trim:-trim] = True
        valid &= interior
    values = error[valid]
    if values.size < 32:
        raise ValueError("too few valid rays for eikonal direction consistency")
    return {
        "median_relative_direction_error": float(np.median(values)),
        "p95_relative_direction_error": float(np.percentile(values, 95.0)),
        "max_relative_direction_error": float(np.max(values)),
    }


__all__ = [
    "VectorRefractiveAxiconRayBundle",
    "VectorRefractiveAxiconResult",
    "build_tilted_vector_refractive_axicon_field",
    "fresnel_transmit_vector_3d",
    "lab_reference_eikonal_direction_consistency",
    "sample_vector_field_on_tilted_entrance",
    "spectral_normal_flux_au",
]
