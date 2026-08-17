"""Vortex-Bessel visual atlas built on physically placed perturbations.

The report atlas must distinguish two different questions:

1. What happens when a *physical component or input condition* is wrong?
2. What happens when a generic wavefront contains a named aberration?

Physical misalignments are therefore routed through ``vortex_physical_errors``
so input angle/radius/decentre occur before the SLM/4F chain and axicon errors
modify the physical axicon sag.  Zernike sweeps remain useful, but are explicitly
labelled as wavefront-error sensitivity at the axicon plane rather than as a
surrogate for a particular misaligned optic.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.vortex_physical_errors import (
    DEFAULT_WINDOW_M,
    PhysicalPerturbation,
    build_physical_source,
)
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny
DEFAULT_SCREENING_N = 1536
DEFAULT_REPORT_N = 3072
DEFAULT_Z_M = (20e-3, 40e-3, 60e-3, 80e-3, 100e-3)
DEFAULT_OUTPUT_ROOT = Path("outputs/validation/vortex_visual_atlas")
DEFAULT_FIGURE_ROOT = Path("outputs/figures/vortex_visual_atlas")


@dataclass(frozen=True)
class ZernikeSpec:
    name: str
    radial_order: int
    azimuthal_order: int
    orientation: str


ZERNIKE_REGISTRY: dict[str, ZernikeSpec] = {
    "defocus": ZernikeSpec("defocus", 2, 0, "rotationally_symmetric"),
    "astigmatism_x": ZernikeSpec("astigmatism_x", 2, 2, "cos_2theta"),
    "astigmatism_y": ZernikeSpec("astigmatism_y", 2, -2, "sin_2theta"),
    "coma_x": ZernikeSpec("coma_x", 3, 1, "cos_theta"),
    "coma_y": ZernikeSpec("coma_y", 3, -1, "sin_theta"),
    "spherical": ZernikeSpec("spherical", 4, 0, "rotationally_symmetric"),
}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in values:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def zernike_mode(name: str, grid: Mapping[str, Any], radius_m: float) -> np.ndarray:
    """Return an RMS-normalised real Zernike-like wavefront mode."""

    if name not in ZERNIKE_REGISTRY:
        raise ValueError(f"unknown Zernike mode {name!r}")
    x = np.asarray(grid["X"], dtype=float)
    y = np.asarray(grid["Y"], dtype=float)
    rho = np.hypot(x, y) / float(radius_m)
    theta = np.arctan2(y, x)
    if name == "defocus":
        raw = 2.0 * rho**2 - 1.0
    elif name == "astigmatism_x":
        raw = rho**2 * np.cos(2.0 * theta)
    elif name == "astigmatism_y":
        raw = rho**2 * np.sin(2.0 * theta)
    elif name == "coma_x":
        raw = (3.0 * rho**3 - 2.0 * rho) * np.cos(theta)
    elif name == "coma_y":
        raw = (3.0 * rho**3 - 2.0 * rho) * np.sin(theta)
    elif name == "spherical":
        raw = 6.0 * rho**4 - 6.0 * rho**2 + 1.0
    else:  # pragma: no cover
        raise AssertionError(name)
    mask = rho <= 1.0
    rms = float(np.sqrt(np.mean(np.square(raw[mask]))))
    return np.where(mask, raw / max(rms, EPS), 0.0)


def apply_zernike_waves(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    name: str,
    waves_rms: float,
    radius_m: float,
) -> np.ndarray:
    """Apply a generic wavefront aberration at the axicon plane.

    This is a controlled wavefront-error study.  It must not be labelled as a
    unique physical model of axicon tilt/decentre or input-beam misalignment.
    """

    mode = zernike_mode(name, grid, radius_m)
    return np.asarray(field, dtype=np.complex128) * np.exp(
        1j * 2.0 * np.pi * float(waves_rms) * mode
    )


def build_atlas_source(
    case_id: str,
    *,
    grid_n: int,
    beam_radius_scale: float = 1.0,
    input_beam_decentre_m: tuple[float, float] = (0.0, 0.0),
    input_beam_angle_rad: tuple[float, float] = (0.0, 0.0),
    hologram_decentre_m: tuple[float, float] = (0.0, 0.0),
    fourier_iris_offset_fraction: float = 0.0,
    axicon_angle_scale: float = 1.0,
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0),
    axicon_tilt_rad: tuple[float, float] = (0.0, 0.0),
    axicon_tip_model: str = "sharp",
    axicon_rounding_parameter_m: float = 0.0,
    axicon_flat_tip_radius_m: float = 0.0,
    zernike_name: str | None = None,
    zernike_waves_rms: float = 0.0,
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    """Build one atlas source with each error inserted at its physical plane."""

    perturbation = PhysicalPerturbation(
        beam_radius_scale=float(beam_radius_scale),
        input_beam_decentre_m=tuple(map(float, input_beam_decentre_m)),
        input_beam_angle_rad=tuple(map(float, input_beam_angle_rad)),
        hologram_decentre_m=tuple(map(float, hologram_decentre_m)),
        fourier_iris_offset_fraction=float(fourier_iris_offset_fraction),
        axicon_base_angle_scale=float(axicon_angle_scale),
        axicon_decentre_m=tuple(map(float, axicon_decentre_m)),
        axicon_tilt_rad=tuple(map(float, axicon_tilt_rad)),
        axicon_tip_model=str(axicon_tip_model),
        axicon_rounding_parameter_m=float(axicon_rounding_parameter_m),
        axicon_flat_tip_radius_m=float(axicon_flat_tip_radius_m),
    )
    source, grid, meta = build_physical_source(
        case_id,
        grid_n=int(grid_n),
        perturbation=perturbation,
        window_m=DEFAULT_WINDOW_M,
    )

    if zernike_name is not None and abs(float(zernike_waves_rms)) > 0.0:
        # Same transverse plane as the axicon transmission; multiplicative thin
        # phases commute, so applying immediately after the axicon is equivalent
        # to applying the generic wavefront error immediately before it.
        radius = 2.0e-3
        source = apply_zernike_waves(
            source,
            grid,
            name=zernike_name,
            waves_rms=float(zernike_waves_rms),
            radius_m=radius,
        )
        meta = {
            **meta,
            "generic_wavefront_aberration": zernike_name,
            "generic_wavefront_waves_rms": float(zernike_waves_rms),
            "generic_wavefront_application_plane": "axicon_plane",
        }
    else:
        meta = {
            **meta,
            "generic_wavefront_aberration": "none",
            "generic_wavefront_waves_rms": 0.0,
        }
    return np.asarray(source, dtype=np.complex128), grid, dict(meta)


def propagate_selected_planes(
    source: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float] = DEFAULT_Z_M,
) -> list[np.ndarray]:
    planes: list[np.ndarray] = []
    for z_m in z_values_m:
        propagated = angular_spectrum_propagate_bl(
            source,
            dict(grid),
            float(wavelength_m),
            float(z_m),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        planes.append(np.asarray(np.abs(propagated) ** 2, dtype=np.float32))
    return planes


def transverse_metrics(intensity: np.ndarray, grid: Mapping[str, Any]) -> dict[str, float]:
    I = np.asarray(intensity, dtype=float)
    total = float(np.sum(I))
    x = np.asarray(grid["X"], dtype=float)
    y = np.asarray(grid["Y"], dtype=float)
    cx = float(np.sum(I * x) / max(total, EPS))
    cy = float(np.sum(I * y) / max(total, EPS))
    peak = float(np.max(I))
    centre = int(I.shape[0]) // 2
    return {
        "power_au": total * float(grid["dx"]) ** 2,
        "peak_intensity_au": peak,
        "centroid_x_m": cx,
        "centroid_y_m": cy,
        "native_centre_intensity_ratio": float(I[centre, centre] / max(peak, EPS)),
    }


def parameter_registry() -> dict[str, Sequence[Any]]:
    """Physical input/design parameters, not generic aberrations."""

    return {
        "beam_radius_scale": (0.6, 0.8, 1.0, 1.2, 1.4),
        "axicon_angle_scale": (0.75, 0.875, 1.0, 1.125, 1.25),
        "input_beam_angle_x_rad": (-1.0e-3, -0.5e-3, 0.0, 0.5e-3, 1.0e-3),
        "input_beam_decentre_x_m": (-400e-6, -200e-6, 0.0, 200e-6, 400e-6),
    }


def manufacturing_defect_registry() -> dict[str, Sequence[Any]]:
    return {
        "axicon_rounding_parameter_m": (0.0, 2e-6, 5e-6, 10e-6, 20e-6),
        "axicon_flat_tip_radius_m": (0.0, 10e-6, 25e-6, 50e-6, 100e-6),
    }


def aberration_registry() -> dict[str, Sequence[float]]:
    # Wider diagnostic range than the original plot so morphology is visibly
    # interpretable.  These remain generic wavefront errors, not misalignment models.
    return {name: (-0.50, -0.25, 0.0, 0.25, 0.50) for name in ZERNIKE_REGISTRY}


def alignment_registry() -> dict[str, Sequence[float]]:
    return {
        "axicon_decentre_x_m": (-400e-6, -200e-6, 0.0, 200e-6, 400e-6),
        "axicon_tilt_y_rad": tuple(np.deg2rad([-0.5, -0.25, 0.0, 0.25, 0.5])),
        "hologram_decentre_x_m": (-200e-6, -100e-6, 0.0, 100e-6, 200e-6),
        "fourier_iris_offset_fraction": (-0.6, -0.3, 0.0, 0.3, 0.6),
    }


def _source_kwargs(family: str, parameter: str, value: Any) -> dict[str, Any]:
    if family == "parameter":
        if parameter == "input_beam_angle_x_rad":
            return {"input_beam_angle_rad": (float(value), 0.0)}
        if parameter == "input_beam_decentre_x_m":
            return {"input_beam_decentre_m": (float(value), 0.0)}
        return {parameter: value}
    if family == "manufacturing":
        if parameter == "axicon_rounding_parameter_m":
            return {
                "axicon_tip_model": "sharp" if float(value) == 0.0 else "hyperboloidal_round",
                "axicon_rounding_parameter_m": float(value),
            }
        if parameter == "axicon_flat_tip_radius_m":
            return {
                "axicon_tip_model": "sharp" if float(value) == 0.0 else "flat_blunt",
                "axicon_flat_tip_radius_m": float(value),
            }
    if family == "aberration":
        return {"zernike_name": parameter, "zernike_waves_rms": float(value)}
    if family == "alignment":
        if parameter == "axicon_decentre_x_m":
            return {"axicon_decentre_m": (float(value), 0.0)}
        if parameter == "axicon_tilt_y_rad":
            return {"axicon_tilt_rad": (0.0, float(value))}
        if parameter == "hologram_decentre_x_m":
            return {"hologram_decentre_m": (float(value), 0.0)}
        if parameter == "fourier_iris_offset_fraction":
            return {"fourier_iris_offset_fraction": float(value)}
    raise ValueError((family, parameter, value))


def run_atlas_screening(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cases: Sequence[str] = ("B0", "V1", "V3"),
    grid_n: int = DEFAULT_SCREENING_N,
    z_m: float = 60e-3,
) -> dict[str, Any]:
    """Generate numerical sweep tables using the physical-error backend."""

    rows: list[dict[str, Any]] = []
    registries = (
        ("parameter", parameter_registry()),
        ("manufacturing", manufacturing_defect_registry()),
        ("aberration", aberration_registry()),
        ("alignment", alignment_registry()),
    )
    for case_id in cases:
        for family, registry in registries:
            for parameter, values in registry.items():
                for value in values:
                    kwargs = _source_kwargs(family, parameter, value)
                    source, grid, meta = build_atlas_source(case_id, grid_n=grid_n, **kwargs)
                    plane = propagate_selected_planes(
                        source, grid, float(meta["wavelength_m"]), (z_m,)
                    )[0]
                    rows.append(
                        {
                            "family": family,
                            "parameter": parameter,
                            "value": value,
                            "case_id": case_id,
                            "grid_n": grid_n,
                            "z_m": z_m,
                            "axicon_tilt_model": meta.get("axicon_tilt_model", ""),
                            "full_vector_snell_fresnel": meta.get("full_vector_snell_fresnel", False),
                            **transverse_metrics(plane, grid),
                        }
                    )
    _write_csv(output_root / "atlas_screening_metrics.csv", rows)
    manifest = {
        "outcome": "VORTEX-PHYSICAL-VISUAL-ATLAS-SCREENING",
        "report_figures_authorised": False,
        "screening_grid_n": int(grid_n),
        "report_grid_n": DEFAULT_REPORT_N,
        "route": "physical-plane SLM/4F/axicon source-scale route",
        "cases": list(cases),
        "parameter_registry": {k: list(v) for k, v in parameter_registry().items()},
        "manufacturing_defect_registry": {
            k: list(v) for k, v in manufacturing_defect_registry().items()
        },
        "aberration_registry": {k: list(v) for k, v in aberration_registry().items()},
        "alignment_registry": {k: list(v) for k, v in alignment_registry().items()},
        "zernike_convention": (
            "generic unit-RMS wavefront modes at axicon plane; not used as substitutes for physical misalignment"
        ),
        "axicon_tilt_fidelity": (
            "rotated thin-element OPD small-angle model; full vector Snell/Fresnel large-angle model not yet implemented"
        ),
        "calibration_required": (
            "actual axicon tip profile/rounding, clear aperture, beam angle/decentre, and rigid-body alignment"
        ),
    }
    _write_json(output_root / "atlas_screening_manifest.json", manifest)
    return manifest
