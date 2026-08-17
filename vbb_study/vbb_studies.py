"""Separated beam, surface, and sample study entry points."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence

import numpy as np

from vbb_study.config import BeamDesign, EPS, PathKind, TwinConfig, um
from vbb_study.design import compute_design_from_config
from vbb_study.equations.propagation import focus_to_focal_plane, make_bl_asm_propagator
from vbb_study.facade import core as _bt

StudyKind = Literal["beam_to_surface", "through_sample", "full_source_to_sample"]


@dataclass
class SurfaceField:
    """Air-side hand-off field at the sample surface, before sample entry."""

    Ex: np.ndarray
    Ey: np.ndarray | None
    Ez: np.ndarray | None
    grid: object
    z_surface_m: float
    medium_before: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullJourneyResult:
    """Continuous air-to-sample journey with the surface as z=0."""

    volume: dict[str, Any]
    air_result: dict[str, Any]
    surface: SurfaceField
    sample_result: Any
    continuity: dict[str, Any]
    metrics: dict[str, Any]


def beam_air_config(config: TwinConfig) -> TwinConfig:
    """Return a config copy for the air-only beam study."""

    air_material = replace(config.material, name="air beam path", refractive_index=1.0)
    air_energy = replace(config.energy, sample_surface_transmission=1.0)
    return replace(
        config,
        material=air_material,
        energy=air_energy,
        apply_interface=False,
        correct_interface=False,
        study_kind="beam_to_surface",
    )


def _beam_design(config: TwinConfig) -> tuple[TwinConfig, BeamDesign]:
    air_config = beam_air_config(config)
    return air_config, compute_design_from_config(air_config)


def resolve_surface_z_m(
    config: TwinConfig,
    design: BeamDesign,
    *,
    zone_start_m: float | None = None,
    zone_center_m: float | None = None,
) -> float:
    """Resolve the deliberate air-side surface placement."""

    custom = getattr(config, "surface_z_m", None)
    if custom is not None and np.isfinite(float(custom)):
        return float(custom)
    placement = str(getattr(config, "surface_placement", "zone_center")).lower().strip()
    design_center = 0.5 * float(design.target_bessel_length_m)
    if placement == "zone_start":
        return float(zone_start_m) if zone_start_m is not None and np.isfinite(zone_start_m) else 0.15 * float(design.target_bessel_length_m)
    if placement == "zone_center":
        return float(zone_center_m) if zone_center_m is not None and np.isfinite(zone_center_m) else design_center
    if placement == "custom":
        raise ValueError("surface_placement='custom' requires config.surface_z_m.")
    raise ValueError(f"Unsupported surface placement: {placement!r}")


def _trace_zone_from_field(
    U0: np.ndarray,
    grid: dict[str, Any],
    config: TwinConfig,
    z_values_m: np.ndarray,
) -> dict[str, Any]:
    volume = _bt().propagate_volume(
        U0,
        grid,
        config.laser.wavelength_m,
        z_values_m,
        n_medium=1.0,
        crop_pixels=min(config.grid.crop_pixels, int(grid.get("N", config.grid.crop_pixels))),
        bandlimit=True,
        method=config.propagation.method,
        propagation_config=config.propagation,
    )
    zone = _bt().bessel_zone_metrics(volume["z"], volume["peak"], level=0.5)
    peak_index = int(volume.get("peak_index", int(np.argmax(volume["peak"]))))
    peak_z_m = float(volume["z"][peak_index])
    start_m = float(zone["zone_start_um"]) * um if np.isfinite(zone["zone_start_um"]) else np.nan
    end_m = float(zone["zone_end_um"]) * um if np.isfinite(zone["zone_end_um"]) else np.nan
    center_m = 0.5 * (start_m + end_m) if np.isfinite(start_m) and np.isfinite(end_m) else peak_z_m
    return {
        "volume": volume,
        "zone": zone,
        "zone_start_m": start_m,
        "zone_end_m": end_m,
        "zone_center_m": float(center_m),
        "peak_z_m": peak_z_m,
    }


def _air_scan_plan(
    config: TwinConfig,
    design: BeamDesign,
    U0: np.ndarray,
    grid: dict[str, Any],
    z_values_m: Sequence[float] | None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if z_values_m is not None:
        z_values = np.asarray(z_values_m, dtype=float)
        if z_values.ndim != 1 or z_values.size == 0:
            raise ValueError("z_values_m must be a non-empty 1D sequence.")
        surface_z = resolve_surface_z_m(config, design)
        return z_values, surface_z, {
            "mode": "explicit",
            "surface_z_m": surface_z,
            "z_start_m": float(z_values[0]),
            "z_end_m": float(z_values[-1]),
        }

    target = float(design.target_bessel_length_m)
    coarse_half_span = max(
        float(config.grid.axial_range_m),
        float(getattr(config, "air_scan_coarse_span_factor", 4.0)) * target,
    )
    coarse_points = max(17, int(config.grid.coarse_scan_points))
    coarse_z = np.linspace(-coarse_half_span, coarse_half_span, coarse_points)
    coarse = _trace_zone_from_field(U0, grid, config, coarse_z)
    measured_zone_m = float(coarse["zone"]["bessel_zone_um"]) * um
    center_m = float(coarse["zone_center_m"])
    half_span = max(
        float(config.grid.axial_range_m),
        float(getattr(config, "air_scan_half_span_factor", 1.25)) * target,
        1.25 * measured_zone_m if np.isfinite(measured_zone_m) and measured_zone_m > 0.0 else 0.0,
    )
    points = max(2, int(config.grid.axial_points))
    z_values = np.linspace(center_m - half_span, center_m + half_span, points)
    surface_z = resolve_surface_z_m(
        config,
        design,
        zone_start_m=float(coarse["zone_start_m"]),
        zone_center_m=center_m,
    )
    return z_values, surface_z, {
        "mode": "coarse_fwhm_centered",
        "coarse_z_start_m": float(coarse_z[0]),
        "coarse_z_end_m": float(coarse_z[-1]),
        "coarse_zone_um": float(coarse["zone"]["bessel_zone_um"]),
        "coarse_zone_start_um": float(coarse["zone"]["zone_start_um"]),
        "coarse_zone_end_um": float(coarse["zone"]["zone_end_um"]),
        "coarse_peak_z_um": float(coarse["peak_z_m"] / um),
        "zone_center_m": center_m,
        "surface_z_m": surface_z,
        "z_start_m": float(z_values[0]),
        "z_end_m": float(z_values[-1]),
        "half_span_factor": float(getattr(config, "air_scan_half_span_factor", 1.25)),
    }


def _coerce_z_values(config: TwinConfig, design: BeamDesign, z_values_m: Sequence[float] | None) -> np.ndarray:
    if z_values_m is None:
        return _default_air_z_values(config, design)
    z_values = np.asarray(z_values_m, dtype=float)
    if z_values.ndim != 1 or z_values.size == 0:
        raise ValueError("z_values_m must be a non-empty 1D sequence.")
    return z_values


def _propagate_component_to_surface(
    component: np.ndarray | None,
    grid: dict[str, Any],
    config: TwinConfig,
    z_surface_m: float,
) -> np.ndarray | None:
    if component is None:
        return None
    z_surface = float(z_surface_m)
    if abs(z_surface) <= EPS:
        return np.asarray(component, dtype=complex).copy()
    prop = make_bl_asm_propagator(
        np.asarray(component, dtype=complex),
        grid,
        config.laser.wavelength_m,
        n_medium=1.0,
        bandlimit=True,
    )
    return prop(z_surface)


def _surface_field(
    Ex0: np.ndarray,
    grid: dict[str, Any],
    config: TwinConfig,
    *,
    Ey0: np.ndarray | None = None,
    Ez0: np.ndarray | None = None,
    z_surface_m: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> SurfaceField:
    surface_z = resolve_surface_z_m(config, compute_design_from_config(config)) if z_surface_m is None else float(z_surface_m)
    return SurfaceField(
        Ex=np.asarray(_propagate_component_to_surface(Ex0, grid, config, surface_z), dtype=complex),
        Ey=None if Ey0 is None else np.asarray(_propagate_component_to_surface(Ey0, grid, config, surface_z), dtype=complex),
        Ez=None if Ez0 is None else np.asarray(_propagate_component_to_surface(Ez0, grid, config, surface_z), dtype=complex),
        grid=grid,
        z_surface_m=surface_z,
        medium_before=1.0,
        metadata=dict(metadata or {}),
    )


def _result_from_axicon_stage(
    config: TwinConfig,
    path: PathKind,
    z_values_m: Sequence[float] | None,
) -> dict[str, Any]:
    from . import vbb_axicon

    air_config, design = _beam_design(config)
    stage = vbb_axicon.axicon_stage(getattr(config, "generation_method", "holographic"))
    axicon_result = stage.generate({"design": design}, air_config)
    grid = axicon_result.grid
    U0 = np.asarray(axicon_result.Ex, dtype=complex)
    z_values, surface_z, scan_plan = _air_scan_plan(air_config, design, U0, grid, z_values_m)
    volume = _bt().propagate_volume(
        U0,
        grid,
        air_config.laser.wavelength_m,
        z_values,
        n_medium=1.0,
        crop_pixels=min(air_config.grid.crop_pixels, air_config.grid.ideal_N),
        bandlimit=True,
        method=air_config.propagation.method,
        propagation_config=air_config.propagation,
    )
    surface = _surface_field(
        U0,
        grid,
        air_config,
        Ey0=axicon_result.Ey,
        Ez0=axicon_result.Ez,
        z_surface_m=surface_z,
        metadata={
            "study_kind": "beam_to_surface",
            "generation_method": str(getattr(config, "generation_method", "holographic")),
            "path": path,
            "medium_before": 1.0,
            "design_medium_n": 1.0,
            "axicon_metadata": dict(axicon_result.metadata),
            "air_scan_plan": dict(scan_plan),
        },
    )
    return {
        "path": path,
        "study_kind": "beam_to_surface",
        "design": design,
        "focal_grid": grid,
        "volume": volume,
        "first_order_selected_fraction": float(axicon_result.metadata.get("efficiency", 1.0)),
        "z_zone_centre_m": float(scan_plan.get("zone_center_m", surface_z)),
        "air_scan_plan": scan_plan,
        "interface_zernike_fit": {},
        "generation_method": str(getattr(config, "generation_method", "holographic")),
        "beam_medium_n": 1.0,
        "surface_field": surface,
        "axicon_result": axicon_result,
        "axicon_metadata": dict(axicon_result.metadata),
        "metrics_config": air_config,
    }


def _result_from_realistic_holographic(
    config: TwinConfig,
    z_values_m: Sequence[float] | None,
) -> dict[str, Any]:
    air_config, design = _beam_design(config)
    slm_field = _bt().build_realistic_slm_field(air_config, design)
    grid = slm_field["grid"]
    U = np.asarray(slm_field["U"], dtype=complex)

    if air_config.include_first_order_isolation and air_config.include_blaze:
        filter_geometry = _bt().first_order_filter_geometry(grid, air_config.slm, design)
        order = _bt().isolate_first_order(
            U,
            grid,
            air_config.slm,
            filter_radius_lpmm=filter_geometry["effective_filter_radius_lpmm"],
        )
        order.update(filter_geometry)
        U = np.asarray(order["U_selected"], dtype=complex)
    else:
        order = {
            "U_selected": U,
            "selected_fraction": 1.0,
            "carrier_lpmm": 0.0,
            "filter_radius_lpmm": 0.0,
            "order_mask": np.ones_like(U, dtype=bool),
        }

    pupil = (grid["R"] <= air_config.objective.pupil_radius_m).astype(float)
    U = U * pupil
    U_focus, focal_grid = focus_to_focal_plane(U, grid, air_config.laser, air_config.objective)
    z_values, surface_z, scan_plan = _air_scan_plan(air_config, design, U_focus, focal_grid, z_values_m)
    volume = _bt().propagate_volume(
        U_focus,
        focal_grid,
        air_config.laser.wavelength_m,
        z_values,
        n_medium=1.0,
        crop_pixels=air_config.grid.crop_pixels,
        bandlimit=True,
        method=air_config.propagation.method,
        propagation_config=air_config.propagation,
    )
    surface = _surface_field(
        U_focus,
        focal_grid,
        air_config,
        z_surface_m=surface_z,
        metadata={
            "study_kind": "beam_to_surface",
            "generation_method": "holographic",
            "path": "realistic",
            "medium_before": 1.0,
            "design_medium_n": 1.0,
            "interface_aberration": "absent",
            "air_scan_plan": dict(scan_plan),
        },
    )
    return {
        "path": "realistic",
        "study_kind": "beam_to_surface",
        "design": design,
        "slm_field": slm_field,
        "order": order,
        "pupil_grid": grid,
        "pupil_phase_interface": np.zeros_like(grid["R"]),
        "interface_zernike_fit": {},
        "U_focus": U_focus,
        "focal_grid": focal_grid,
        "z_zone_centre_m": float(scan_plan.get("zone_center_m", surface_z)),
        "volume": volume,
        "air_scan_plan": scan_plan,
        "first_order_selected_fraction": float(order["selected_fraction"]),
        "generation_method": "holographic",
        "beam_medium_n": 1.0,
        "surface_field": surface,
        "metrics_config": air_config,
    }


def build_beam_to_surface_result(
    config: TwinConfig,
    *,
    path: PathKind = "ideal",
    z_values_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build the full air-only beam study result used by ``bt.run_case``."""

    method = str(getattr(config, "generation_method", "holographic")).lower().strip()
    if path == "realistic" and method == "holographic":
        return _result_from_realistic_holographic(config, z_values_m)
    return _result_from_axicon_stage(config, path, z_values_m)


def run_beam_to_surface(config: TwinConfig) -> tuple[dict[str, Any], SurfaceField]:
    """Air-only source-to-surface beam study."""

    result = build_beam_to_surface_result(config, path="ideal", z_values_m=None)
    return result["volume"], result["surface_field"]


def run_through_sample(
    surface: SurfaceField,
    config: TwinConfig,
    *,
    correct_interface: bool = False,
    write_depth_m: float | None = None,
) -> Any:
    """Sample-side study fed by the explicit air-side ``SurfaceField``."""

    if float(surface.medium_before) != 1.0:
        raise ValueError("SurfaceField.medium_before must be air (n=1) for the sample hand-off.")
    from . import vbb_sample_study

    return vbb_sample_study.run_through_sample(
        surface,
        config,
        correct_interface=correct_interface,
        write_depth_m=write_depth_m,
    )


def _surface_power(surface: SurfaceField) -> float:
    grid = surface.grid
    if not isinstance(grid, dict):
        raise TypeError("SurfaceField.grid must be a dict-like grid.")
    fields = [np.asarray(surface.Ex, dtype=complex)]
    if surface.Ey is not None:
        fields.append(np.asarray(surface.Ey, dtype=complex))
    if surface.Ez is not None:
        fields.append(np.asarray(surface.Ez, dtype=complex))
    return float(sum(np.sum(np.abs(field_i) ** 2) for field_i in fields) * float(grid["dx"]) ** 2)


def _volume_subset(volume: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    z = np.asarray(volume["z"], dtype=float)
    keep = np.asarray(mask, dtype=bool)
    if keep.size != z.size:
        raise ValueError("Volume subset mask length must match z axis.")
    out = dict(volume)
    out["z"] = z[keep]
    for key in ("xz", "peak", "onaxis", "total_power", "output_dx_m", "retained_power_fraction"):
        if key in volume:
            arr = np.asarray(volume[key])
            axis = 1 if key == "xz" else 0
            out[key] = np.take(arr, np.flatnonzero(keep), axis=axis)
    if "intensity_stack" in volume:
        out["intensity_stack"] = np.asarray(volume["intensity_stack"])[keep]
    return out


def _stitched_volume(
    air_volume: dict[str, Any],
    sample_volume: dict[str, Any],
    surface: SurfaceField,
    n_medium: float,
) -> dict[str, Any]:
    air_rel_z = np.asarray(air_volume["z"], dtype=float) - float(surface.z_surface_m)
    air_keep = air_rel_z < 0.0
    if not np.any(air_keep):
        air_keep[int(np.argmin(np.abs(air_rel_z)))] = True
    air = _volume_subset({**air_volume, "z": air_rel_z}, air_keep)
    sample_z = np.asarray(sample_volume["z"], dtype=float)
    sample_keep = sample_z >= 0.0
    sample = _volume_subset(sample_volume, sample_keep)

    z = np.concatenate([np.asarray(air["z"], dtype=float), np.asarray(sample["z"], dtype=float)])
    xz = np.concatenate([np.asarray(air["xz"], dtype=float), np.asarray(sample["xz"], dtype=float)], axis=1)
    stack = np.concatenate(
        [np.asarray(air["intensity_stack"], dtype=np.float32), np.asarray(sample["intensity_stack"], dtype=np.float32)],
        axis=0,
    )
    peak = np.concatenate([np.asarray(air["peak"], dtype=float), np.asarray(sample["peak"], dtype=float)])
    onaxis = np.concatenate([np.asarray(air["onaxis"], dtype=float), np.asarray(sample["onaxis"], dtype=float)])
    total_power = np.concatenate([np.asarray(air["total_power"], dtype=float), np.asarray(sample["total_power"], dtype=float)])
    peak_index = int(np.nanargmax(peak))
    medium_n = np.concatenate([np.ones(len(air["z"]), dtype=float), np.full(len(sample["z"]), float(n_medium), dtype=float)])
    return {
        "z": z,
        "xz": xz.astype(np.float32),
        "intensity_stack": stack.astype(np.float32),
        "peak": peak,
        "onaxis": onaxis,
        "total_power": total_power,
        "medium_n": medium_n,
        "surface_index": int(np.searchsorted(z, 0.0)),
        "surface_z_m": 0.0,
        "target_write_depth_m": float(sample_volume.get("target_write_depth_m", np.nan)),
        "planes": {
            "start": stack[0].astype(np.float32),
            "middle": stack[len(stack) // 2].astype(np.float32),
            "end": stack[-1].astype(np.float32),
            "peak": stack[peak_index].astype(np.float32),
        },
        "peak_index": peak_index,
        "crop_grid": sample_volume["crop_grid"],
        "propagation_method": sample_volume.get("propagation_method", air_volume.get("propagation_method", "unknown")),
    }


def run_full_source_to_sample(
    config: TwinConfig,
    *,
    correct_interface: bool = False,
    write_depth_m: float | None = None,
    path: PathKind = "ideal",
    z_values_air_m: Sequence[float] | None = None,
    z_values_sample_m: Sequence[float] | None = None,
) -> FullJourneyResult:
    """Stitch air beam-to-surface and in-medium through-sample propagation."""

    air_result = build_beam_to_surface_result(config, path=path, z_values_m=z_values_air_m)
    surface = air_result["surface_field"]
    sample = run_through_sample(
        surface,
        config,
        correct_interface=correct_interface,
        write_depth_m=write_depth_m,
    )
    if z_values_sample_m is not None:
        from . import vbb_sample_study

        sample = vbb_sample_study.run_through_sample(
            surface,
            config,
            correct_interface=correct_interface,
            write_depth_m=write_depth_m,
            z_values_m=z_values_sample_m,
        )
    air_surface_power = _surface_power(surface)
    refracted_surface_power = float(sample.energy_report["surface_power_air"])
    rel = abs(refracted_surface_power - air_surface_power) / max(air_surface_power, EPS)
    if rel > 1e-9:
        raise RuntimeError(f"Surface refraction changed power by {rel:.3e}; expected continuity before surface loss.")
    sample.volume_result["target_write_depth_m"] = float(sample.correction_report["target_write_depth_m"])
    volume = _stitched_volume(
        air_result["volume"],
        sample.volume_result,
        surface,
        float(config.material.refractive_index),
    )
    z = np.asarray(volume["z"], dtype=float)
    dz = np.abs(np.diff(z))
    metrics = {
        "study_kind": "full_source_to_sample",
        "correction": sample.correction,
        "z_surface_m": 0.0,
        "target_write_depth_um": float(sample.metrics["target_write_depth_um"]),
        "sample_zone_center_um": float(sample.metrics["zone_center_um"]),
        "sample_zone_center_error_um": float(sample.metrics["zone_center_error_um"]),
        "sample_bessel_zone_um": float(sample.metrics["bessel_zone_um"]),
        "continuity_relative_power_error": float(rel),
        "surface_power_air": air_surface_power,
        "refracted_surface_power_before_loss": refracted_surface_power,
        "surface_transmission": float(sample.energy_report["surface_transmission"]),
        "full_axial_dz_um": float(np.nanmedian(dz) / um) if dz.size else np.nan,
        "air_axial_dz_um": float(np.nanmedian(np.abs(np.diff(air_result["volume"]["z"]))) / um)
        if len(air_result["volume"]["z"]) > 1
        else np.nan,
        "sample_axial_dz_um": float(np.nanmedian(np.abs(np.diff(sample.z_axis_m))) / um)
        if len(sample.z_axis_m) > 1
        else np.nan,
        "air_subresult_is_beam_result": True,
        "sample_subresult_is_through_sample_result": True,
    }
    continuity = {
        "air_surface_power": air_surface_power,
        "refracted_surface_power_before_loss": refracted_surface_power,
        "relative_power_error": float(rel),
        "pass": bool(rel <= 1e-9),
        "surface_loss_applied_in_sample": float(sample.energy_report["surface_transmission"]),
    }
    return FullJourneyResult(
        volume=volume,
        air_result=air_result,
        surface=surface,
        sample_result=sample,
        continuity=continuity,
        metrics=metrics,
    )


__all__ = [
    "StudyKind",
    "SurfaceField",
    "FullJourneyResult",
    "beam_air_config",
    "build_beam_to_surface_result",
    "run_beam_to_surface",
    "run_full_source_to_sample",
    "run_through_sample",
]
