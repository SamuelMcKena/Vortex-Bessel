"""Vector physical-axicon element and air-side surface handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.config import EPS, TwinConfig
from vbb_study.design import compute_design_from_config
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import focus_to_focal_plane
from vbb_study.vector_field import VectorField, propagate_vector_asm
from vbb_study.vbb_studies import SurfaceField

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class VectorAxiconParameters:
    """Resolved physical-axicon parameters from the existing rig config."""

    n_axicon: float
    n_medium: float
    base_angle_rad: float
    aperture_radius_m: float | None
    k_r_pre_m_inv: float
    k_r_surface_m_inv: float
    source: str


@dataclass(frozen=True)
class VectorAxiconResult:
    """Vector axicon and surface-plane propagation result."""

    after_axicon: VectorField
    focal_plane: VectorField
    surface: SurfaceField
    z_values_m: np.ndarray
    intensity_stack: np.ndarray
    parameters: VectorAxiconParameters
    metadata: Mapping[str, Any]


def _grid_r_phi(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "R" in grid and "PHI" in grid:
        return np.asarray(grid["R"], dtype=float), np.asarray(grid["PHI"], dtype=float), np.asarray(grid["X"], dtype=float)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return np.hypot(X, Y), np.arctan2(Y, X), X


def resolve_vector_axicon_parameters(config: TwinConfig) -> VectorAxiconParameters:
    """Resolve axicon and ObjectiveMap-derived wavevector parameters."""

    design = compute_design_from_config(config)
    physical = config.physical_axicon
    n_axicon = float(design.n_axicon if physical.n_axicon is None else physical.n_axicon)
    n_medium = float(getattr(physical, "axicon_medium_n", design.hologram_medium_n))
    if physical.axicon_base_angle_deg is None:
        beta = float(design.gamma_slm_rad)
        source = "BeamDesign.gamma_slm_rad"
    else:
        beta = float(np.deg2rad(physical.axicon_base_angle_deg))
        source = "PhysicalAxiconConfig.axicon_base_angle_deg"
    k_pre = float(config.laser.k0 * (n_axicon - n_medium) * np.tan(beta))
    # Existing BeamDesign already records the focused-side wavevector governed
    # by the ObjectiveMap demagnification.
    k_surface = float(design.kr_sample_m_inv)
    return VectorAxiconParameters(
        n_axicon=n_axicon,
        n_medium=n_medium,
        base_angle_rad=beta,
        aperture_radius_m=physical.axicon_aperture_radius_m,
        k_r_pre_m_inv=k_pre,
        k_r_surface_m_inv=k_surface,
        source=source,
    )


def fresnel_sp_amplitudes(n_axicon: float, n_medium: float, beta_rad: float) -> tuple[complex, complex, complex]:
    """Return ``(t_entry, t_p, t_s)`` for the conical exit face."""

    n_ax = float(n_axicon)
    n_med = float(n_medium)
    beta = float(beta_rad)
    cos_i = np.cos(beta)
    t_entry = 2.0 * n_med / (n_med + n_ax)
    sin_t = (n_ax / n_med) * np.sin(beta) + 0j
    cos_t = np.sqrt(1.0 - sin_t * sin_t)
    # Keep both branch corrections: the first protects ordinary refraction
    # sign, the second protects total-internal-reflection-adjacent cases.
    if np.real(cos_t) < 0.0:
        cos_t = -cos_t
    if np.imag(cos_t) < 0.0:
        cos_t = -cos_t
    t_s = 2.0 * n_ax * cos_i / (n_ax * cos_i + n_med * cos_t)
    t_p = 2.0 * n_ax * cos_i / (n_med * cos_i + n_ax * cos_t)
    return complex(t_entry), complex(t_p), complex(t_s)


def apply_vector_axicon(field: VectorField, config: TwinConfig) -> tuple[VectorField, VectorAxiconParameters, dict[str, Any]]:
    """Apply the thin physical axicon phase and Fresnel s/p split."""

    params = resolve_vector_axicon_parameters(config)
    grid = field.grid
    R, phi, _ = _grid_r_phi(grid)
    er_x = np.cos(phi)
    er_y = np.sin(phi)
    ep_x = -np.sin(phi)
    ep_y = np.cos(phi)
    er = field.ex * er_x + field.ey * er_y
    ephi = field.ex * ep_x + field.ey * ep_y

    t_entry, t_p, t_s = fresnel_sp_amplitudes(params.n_axicon, params.n_medium, params.base_angle_rad)
    phase = np.exp(-1j * abs(float(params.k_r_pre_m_inv)) * R)
    if params.aperture_radius_m is None:
        aperture = np.ones_like(R, dtype=float)
    else:
        aperture = (R <= float(params.aperture_radius_m)).astype(float)

    er_out = t_entry * t_p * er * phase * aperture
    ephi_out = t_entry * t_s * ephi * phase * aperture
    ex = er_out * er_x + ephi_out * ep_x
    ey = er_out * er_y + ephi_out * ep_y
    ez = field.ez * phase * aperture * t_entry * 0.5 * (t_p + t_s)
    out = VectorField(
        ex=ex,
        ey=ey,
        ez=ez,
        grid=grid,
        wavelength_m=field.wavelength_m,
        medium_index=params.n_medium,
        metadata={**dict(field.metadata), "stage": "after_vector_physical_axicon"},
    )
    meta = {
        "t_entry_abs": float(abs(t_entry)),
        "t_p_abs": float(abs(t_p)),
        "t_s_abs": float(abs(t_s)),
        "k_r_pre_m_inv": float(params.k_r_pre_m_inv),
        "k_r_surface_m_inv": float(params.k_r_surface_m_inv),
        "base_angle_deg": float(np.rad2deg(params.base_angle_rad)),
        "n_axicon": float(params.n_axicon),
        "n_medium": float(params.n_medium),
        "aperture_radius_m": None if params.aperture_radius_m is None else float(params.aperture_radius_m),
    }
    return out, params, meta


def focus_vector_to_surface(field: VectorField, config: TwinConfig) -> VectorField:
    """Focus transverse vector components with the existing scalar lens helper."""

    ex_f, focal_grid = focus_to_focal_plane(field.ex, dict(field.grid), config.laser, config.objective)
    ey_f, _ = focus_to_focal_plane(field.ey, dict(field.grid), config.laser, config.objective)
    transverse = VectorField(
        ex=ex_f,
        ey=ey_f,
        ez=np.zeros_like(ex_f, dtype=complex),
        grid=focal_grid,
        wavelength_m=config.laser.wavelength_m,
        medium_index=1.0,
        metadata={**dict(field.metadata), "stage": "focused_surface_plane"},
    )
    return propagate_vector_asm(transverse, 0.0).replace(metadata=transverse.metadata)


def default_surface_z_values(config: TwinConfig) -> np.ndarray:
    """Return the air-side z-stack around the focused surface plane."""

    design = compute_design_from_config(config)
    half_span = float(design.target_bessel_length_m)
    return np.linspace(-half_span, half_span, int(config.grid.axial_points))


def run_vector_axicon_to_surface(
    field: VectorField,
    config: TwinConfig,
    *,
    z_values_m: Sequence[float] | None = None,
) -> VectorAxiconResult:
    """Apply vector axicon, focus to the air-side surface, and build a z-stack."""

    after_axicon, params, ax_meta = apply_vector_axicon(field, config)
    focal = focus_vector_to_surface(after_axicon, config)
    z_values = np.asarray(default_surface_z_values(config) if z_values_m is None else z_values_m, dtype=float)
    stack = []
    for z in z_values:
        plane = propagate_vector_asm(focal, float(z))
        stack.append(plane.intensity.astype(np.float32))
    intensity_stack = np.asarray(stack, dtype=np.float32)
    surface = SurfaceField(
        Ex=focal.ex,
        Ey=focal.ey,
        Ez=focal.ez,
        grid=focal.grid,
        z_surface_m=0.0,
        medium_before=1.0,
        metadata={
            "stage": "stage7_vector_arm_surface_handoff",
            "in_medium_propagation": "out_of_scope",
            "axicon": ax_meta,
            "lock_status": "caller_must_record",
        },
    )
    return VectorAxiconResult(
        after_axicon=after_axicon,
        focal_plane=focal,
        surface=surface,
        z_values_m=z_values,
        intensity_stack=intensity_stack,
        parameters=params,
        metadata=ax_meta,
    )


def _array_hash(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _array_hash(value),
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return {"kind": "float", "hex": float(value).hex()}
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return repr(value)


def export_surface_field_handoff(
    surface: SurfaceField,
    output_path: str | Path,
    *,
    config: Any,
    git_sha: str,
    lock_status: str,
) -> dict[str, Path]:
    """Write a ``SurfaceField`` NPZ plus JSON sidecar hashes."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ex = np.asarray(surface.Ex, dtype=np.complex128)
    ey = np.zeros_like(ex) if surface.Ey is None else np.asarray(surface.Ey, dtype=np.complex128)
    ez = np.zeros_like(ex) if surface.Ez is None else np.asarray(surface.Ez, dtype=np.complex128)
    grid = dict(surface.grid)
    np.savez_compressed(
        path,
        Ex=ex,
        Ey=ey,
        Ez=ez,
        x=np.asarray(grid["x"], dtype=float),
        dx=np.asarray(float(grid["dx"])),
        z_surface_m=np.asarray(float(surface.z_surface_m)),
        medium_before=np.asarray(float(surface.medium_before)),
    )
    sidecar = {
        "artifact": str(path),
        "stage": "stage7_vector_arm_surface_handoff",
        "git_sha": str(git_sha),
        "lock_status": str(lock_status),
        "arrays": {
            "Ex": _jsonable(ex),
            "Ey": _jsonable(ey),
            "Ez": _jsonable(ez),
        },
        "grid": {
            "dx_m": float(grid["dx"]).hex(),
            "n": int(ex.shape[0]),
        },
        "surface": {
            "z_surface_m": float(surface.z_surface_m).hex(),
            "medium_before": float(surface.medium_before).hex(),
        },
        "config": _jsonable(config),
        "metadata": _jsonable(surface.metadata),
    }
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    return {"npz": path, "sidecar": sidecar_path}


def assert_locked_kr_fingerprint(k_r_m_inv: float, *, expected_m_inv: float = 1.603e6, rel_tol: float = 5e-4) -> None:
    """Tripwire for the existing physical-route focused radial wavevector."""

    rel = abs(float(k_r_m_inv) - float(expected_m_inv)) / max(abs(float(expected_m_inv)), EPS)
    assert rel <= float(rel_tol)


__all__ = [
    "VectorAxiconParameters",
    "VectorAxiconResult",
    "apply_vector_axicon",
    "assert_locked_kr_fingerprint",
    "default_surface_z_values",
    "export_surface_field_handoff",
    "focus_vector_to_surface",
    "fresnel_sp_amplitudes",
    "resolve_vector_axicon_parameters",
    "run_vector_axicon_to_surface",
]
