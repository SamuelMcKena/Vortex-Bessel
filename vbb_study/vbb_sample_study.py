"""Through-sample propagation fed by the air-side ``SurfaceField`` hand-off."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.config import BeamDesign, EPS as BT_EPS, LaserConfig, TWOPI as BT_TWOPI, TwinConfig, uJ as BT_UJ, um as BT_UM
from vbb_study.design import compute_design_from_config
from vbb_study.equations.fields import fft2c, ifft2c, quantize_phase
from vbb_study.equations.interface import fit_interface_zernike_terms, interface_aberration_pupil
from vbb_study.equations.propagation import make_bl_asm_propagator
from vbb_study.facade import core as _bt


@dataclass
class RefractedField:
    """Surface field after the planar air-to-medium hand-off."""

    Ex: np.ndarray
    Ey: np.ndarray | None
    Ez: np.ndarray | None
    grid: object
    z_surface_m: float
    n_medium: float
    wavelength_m: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """In-medium through-sample result measured from the sample surface."""

    volume: np.ndarray
    z_axis_m: np.ndarray
    metrics: dict[str, Any]
    correction: str
    energy_report: dict[str, Any]
    volume_result: dict[str, Any] = field(default_factory=dict)
    surface_field: RefractedField | None = None
    correction_report: dict[str, Any] = field(default_factory=dict)


def _as_grid(surface: Any) -> dict[str, Any]:
    grid = surface.grid
    if not isinstance(grid, Mapping):
        raise TypeError("SurfaceField.grid must be a mapping compatible with bessel_twin_core grids.")
    required = {"N", "dx", "x", "R", "FX", "FY"}
    missing = sorted(required.difference(grid.keys()))
    if missing:
        raise ValueError(f"SurfaceField.grid is missing required entries: {missing}")
    return dict(grid)


def _component_list(Ex: np.ndarray, Ey: np.ndarray | None = None, Ez: np.ndarray | None = None) -> list[np.ndarray]:
    fields = [np.asarray(Ex, dtype=complex)]
    for field_i in (Ey, Ez):
        if field_i is not None:
            fields.append(np.asarray(field_i, dtype=complex))
    return fields


def _field_power(fields: Sequence[np.ndarray], grid: Mapping[str, Any]) -> float:
    dx2 = float(grid["dx"]) ** 2
    return float(sum(np.sum(np.abs(np.asarray(field_i, dtype=complex)) ** 2) for field_i in fields) * dx2)


def _apply_scalar(fields: Sequence[np.ndarray], scale: float) -> list[np.ndarray]:
    amp = float(scale)
    return [np.asarray(field_i, dtype=complex) * amp for field_i in fields]


def _grid_like_surface_pupil(grid: Mapping[str, Any], config: TwinConfig) -> dict[str, Any]:
    """Map surface angular spectrum samples back onto the objective pupil radius."""

    kt = BT_TWOPI * np.hypot(np.asarray(grid["FX"], dtype=float), np.asarray(grid["FY"], dtype=float))
    kt_edge = max(config.laser.k0 * float(config.objective.NA), BT_EPS)
    rho = kt / kt_edge
    pupil_R = rho * float(config.objective.pupil_radius_m)
    scale = float(config.objective.pupil_radius_m) / kt_edge
    return {
        "N": int(grid["N"]),
        "dx": float(grid["dx"]),
        "x": np.asarray(grid["x"], dtype=float),
        "X": BT_TWOPI * np.asarray(grid["FX"], dtype=float) * scale,
        "Y": BT_TWOPI * np.asarray(grid["FY"], dtype=float) * scale,
        "R": pupil_R,
        "PHI": np.arctan2(np.asarray(grid["FY"], dtype=float), np.asarray(grid["FX"], dtype=float)),
        "FX": np.asarray(grid["FX"], dtype=float),
        "FY": np.asarray(grid["FY"], dtype=float),
    }


def _angular_spectrum_phase(field: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return ifft2c(fft2c(np.asarray(field, dtype=complex)) * np.exp(1j * np.asarray(phase, dtype=float)))


def _apply_angular_phase(fields: Sequence[np.ndarray], phase: np.ndarray) -> list[np.ndarray]:
    return [_angular_spectrum_phase(field_i, phase) for field_i in fields]


def _signed_wrapped_phase(phase: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(phase, dtype=float)))


def _interface_correction_mask(interface_phase: np.ndarray, config: TwinConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the phase-only conjugate mask actually applied upstream."""

    ideal = -np.asarray(interface_phase, dtype=float)
    if bool(getattr(config, "include_quantization", True)):
        applied = quantize_phase(ideal, int(config.slm.phase_bits))
        model = f"{int(config.slm.phase_bits)}bit_phase_mask"
    else:
        applied = ideal
        model = "continuous_phase_mask"
    residual = _signed_wrapped_phase(interface_phase + applied)
    return applied, {
        "model": model,
        "quantized": bool(getattr(config, "include_quantization", True)),
        "phase_bits": int(config.slm.phase_bits),
        "residual_phase_rms_rad": float(np.sqrt(np.mean(residual**2))),
        "residual_phase_peak_rad": float(np.max(np.abs(residual))) if residual.size else 0.0,
    }


def _axial_target_phase(
    grid: Mapping[str, Any],
    config: TwinConfig,
    *,
    n_medium: float,
    target_depth_m: float,
) -> np.ndarray:
    """Phase-only defocus term that places the clean reference at target depth."""

    k = BT_TWOPI * float(n_medium) / float(config.laser.wavelength_m)
    kx = BT_TWOPI * np.asarray(grid["FX"], dtype=float)
    ky = BT_TWOPI * np.asarray(grid["FY"], dtype=float)
    kz = np.sqrt(np.maximum(k**2 - kx**2 - ky**2, 0.0))
    phase = -kz * float(target_depth_m)
    phase = phase - float(np.mean(phase[np.isfinite(phase)]))
    return phase


def _combine_component_volumes(volumes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not volumes:
        raise ValueError("At least one propagated component volume is required.")
    first = dict(volumes[0])
    if len(volumes) == 1:
        return first

    stack = np.zeros_like(np.asarray(first["intensity_stack"], dtype=np.float32), dtype=np.float64)
    xz = np.zeros_like(np.asarray(first["xz"], dtype=np.float32), dtype=np.float64)
    total_power = np.zeros_like(np.asarray(first["total_power"], dtype=float), dtype=float)
    for vol in volumes:
        stack += np.asarray(vol["intensity_stack"], dtype=float)
        xz += np.asarray(vol["xz"], dtype=float)
        total_power += np.asarray(vol["total_power"], dtype=float)

    peak = stack.reshape(stack.shape[0], -1).max(axis=1)
    center = stack.shape[1] // 2
    onaxis = stack[:, center, center]
    peak_index = int(np.nanargmax(peak))
    first.update(
        {
            "intensity_stack": stack.astype(np.float32),
            "xz": xz.astype(np.float32),
            "peak": peak,
            "onaxis": onaxis,
            "total_power": total_power,
            "peak_index": peak_index,
            "planes": {
                "start": stack[0].astype(np.float32),
                "middle": stack[len(stack) // 2].astype(np.float32),
                "end": stack[-1].astype(np.float32),
                "peak": stack[peak_index].astype(np.float32),
            },
        }
    )
    return first


def _propagate_components(
    fields: Sequence[np.ndarray],
    grid: Mapping[str, Any],
    config: TwinConfig,
    z_axis_m: np.ndarray,
    *,
    n_medium: float,
) -> dict[str, Any]:
    volumes = [
        _bt().propagate_volume(
            field_i,
            dict(grid),
            config.laser.wavelength_m,
            z_axis_m,
            n_medium=float(n_medium),
            crop_pixels=min(int(config.grid.crop_pixels), int(grid["N"])),
            bandlimit=True,
            method=config.propagation.method,
            propagation_config=config.propagation,
        )
        for field_i in fields
    ]
    return _combine_component_volumes(volumes)


def _default_sample_z_values(config: TwinConfig) -> np.ndarray:
    target = float(config.target.target_bessel_length_m)
    depth = float(config.material.write_depth_m)
    z_max = max(depth + 1.25 * target, 2.0 * target, float(config.grid.axial_range_m))
    points = max(2, int(config.grid.axial_points))
    return np.linspace(0.0, z_max, points)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    return float(np.linalg.norm((aa - bb).ravel()) / (np.linalg.norm(bb.ravel()) + BT_EPS))


def _field_at_z(field: np.ndarray, grid: Mapping[str, Any], config: TwinConfig, z_m: float, *, n_medium: float) -> np.ndarray:
    prop = make_bl_asm_propagator(
        np.asarray(field, dtype=complex),
        dict(grid),
        config.laser.wavelength_m,
        n_medium=float(n_medium),
        bandlimit=True,
    )
    return np.asarray(prop(float(z_m)), dtype=complex)


def _sample_metrics(
    volume: dict[str, Any],
    design: BeamDesign,
    config: TwinConfig,
    *,
    n_medium: float,
    correction: str,
    target_depth_m: float,
    energy_report: Mapping[str, Any],
    correction_report: Mapping[str, Any],
) -> dict[str, Any]:
    plane = np.asarray(volume["planes"]["peak"], dtype=float)
    cgrid = volume["crop_grid"]
    radial = _bt().extract_radial_metrics(plane, cgrid, design.ell, design.kr_sample_m_inv)
    zone = _bt().bessel_zone_metrics(volume["z"], volume["peak"], level=0.5)
    strict_region = _bt().bessel_region_metrics(volume, design)
    pulse_energy = float(config.energy.pulse_energy_at_surface_air_J) * float(config.energy.sample_surface_transmission)
    fluence = _bt().fluence_metrics(plane, cgrid, radial, config, pulse_energy_J=pulse_energy)
    mod = _bt().modification_proxy_metrics(volume, config, pulse_energy_J=pulse_energy)
    peak_idx = int(volume.get("peak_index", int(np.nanargmax(volume["peak"]))))
    zone_center_um = 0.5 * (float(zone["zone_start_um"]) + float(zone["zone_end_um"]))
    target_depth_um = float(target_depth_m / BT_UM)
    correction_kind = (
        "ideal_numerical_correction"
        if correction == "ideal_numerical_correction"
        else "uncorrected_interface"
    )
    metrics = {
        "study_kind": "through_sample",
        "focused_plane": "sample_in_medium",
        "z_axis_origin": "sample_surface",
        "medium_before": 1.0,
        "medium_n": float(n_medium),
        "beam_medium_n": float(n_medium),
        "correction": correction_kind,
        "interface_correction_label": correction_kind,
        "interface_correction_implementation": "diagnostic_only",
        "interface_aberration": (
            "ideal_numerical_correction"
            if correction_kind == "ideal_numerical_correction"
            else "uncorrected_interface"
        ),
        "ell": int(design.ell),
        "kr_sample_m_inv": float(design.kr_sample_m_inv),
        "ring_radius_um": float(radial["ring_radius_m"] / BT_UM),
        "ring_diameter_um": float(radial["ring_diameter_m"] / BT_UM),
        "core_radius_um": float(radial["core_radius_m"] / BT_UM),
        "core_radius_definition": str(radial["core_radius_definition"]),
        "core_hwhm_radius_um": float(radial["core_hwhm_radius_m"] / BT_UM) if np.isfinite(radial["core_hwhm_radius_m"]) else np.nan,
        "core_first_zero_radius_um": float(radial["core_first_zero_radius_m"] / BT_UM),
        "feature_radius_um": float(radial["feature_radius_m"] / BT_UM),
        "feature_diameter_um": float(radial["feature_diameter_m"] / BT_UM),
        "ring_width_um": float(radial["ring_width_m"] / BT_UM),
        "peak_in_plane": float(np.max(volume["peak"])),
        "peak_onaxis": float(np.max(volume["onaxis"])),
        "peak_z_um": float(volume["z"][peak_idx] / BT_UM),
        "target_write_depth_um": target_depth_um,
        "zone_center_um": float(zone_center_um),
        "zone_center_error_um": float(zone_center_um - target_depth_um),
        "clean_zone_lands_at_target": bool(abs(zone_center_um - target_depth_um) <= max(float(zone["bessel_zone_um"]), 1.0)),
        "canonical_zone_um": float(zone["bessel_zone_um"]),
        "canonical_zone_start_um": float(zone["zone_start_um"]),
        "canonical_zone_end_um": float(zone["zone_end_um"]),
        "strict_bessel_region_um": float(strict_region["bessel_region_um"]),
        "pulse_energy_at_surface_air_uJ": float(config.energy.pulse_energy_at_surface_air_J / BT_UJ),
        "pulse_energy_in_medium_uJ": float(pulse_energy / BT_UJ),
        **zone,
        **strict_region,
        **fluence,
        **mod,
        "surface_power_air": float(energy_report["surface_power_air"]),
        "surface_transmission": float(energy_report["surface_transmission"]),
        "medium_input_power": float(energy_report["medium_input_power"]),
        "phase_only_power_relative_error": float(energy_report["phase_only_power_relative_error"]),
        "interface_defocus_before_waves": float(correction_report["zernike_before"]["defocus_waves"]),
        "interface_spherical_before_waves": float(correction_report["zernike_before"]["spherical_waves"]),
        "interface_residual_rms_before_rad": float(correction_report["zernike_before"]["residual_rms_rad"]),
        "interface_defocus_after_waves": float(correction_report["zernike_after"]["defocus_waves"]),
        "interface_spherical_after_waves": float(correction_report["zernike_after"]["spherical_waves"]),
        "interface_residual_rms_after_rad": float(correction_report["zernike_after"]["residual_rms_rad"]),
        "interface_correction_mask_model": str(correction_report["correction_mask"]["model"]),
        "interface_correction_mask_residual_rms_rad": float(correction_report["correction_mask"]["residual_phase_rms_rad"]),
        "interface_reference_uses_corrected_array": bool(correction_report["reference_uses_corrected_array"]),
        "corrected_rel_l2_to_no_interface": float(correction_report["corrected_rel_l2_to_no_interface"]),
        "uncorrected_rel_l2_to_no_interface": float(correction_report["uncorrected_rel_l2_to_no_interface"]),
    }
    return metrics


def refract_surface_field(surface: Any, n_medium: float = 2.44, wavelength_m: float | None = None) -> RefractedField:
    """Refract the air-side surface field into a planar medium.

    The angular-spectrum hand-off preserves transverse spatial frequency.  The
    subsequent medium propagation recomputes ``kz`` with ``k = n_medium*k0``;
    only evanescent components for the target medium are removed.
    """

    if float(surface.medium_before) != 1.0:
        raise ValueError("SurfaceField.medium_before must be air (n=1) before refraction.")
    grid = _as_grid(surface)
    wavelength = float(LaserConfig().wavelength_m if wavelength_m is None else wavelength_m)
    k_medium = BT_TWOPI * float(n_medium) / wavelength
    kt = BT_TWOPI * np.hypot(np.asarray(grid["FX"], dtype=float), np.asarray(grid["FY"], dtype=float))
    propagating = kt <= k_medium
    prop_fraction = float(np.count_nonzero(propagating) / max(propagating.size, 1))

    fields = _component_list(surface.Ex, surface.Ey, surface.Ez)
    filtered = [ifft2c(fft2c(field_i) * propagating) for field_i in fields]
    metadata = dict(getattr(surface, "metadata", {}) or {})
    metadata.update(
        {
            "refracted_surface": True,
            "transverse_k_preserved": True,
            "medium_before": 1.0,
            "medium_after": float(n_medium),
            "propagating_spectral_fraction": prop_fraction,
        }
    )
    return RefractedField(
        Ex=np.asarray(filtered[0], dtype=complex),
        Ey=None if surface.Ey is None else np.asarray(filtered[1], dtype=complex),
        Ez=None if surface.Ez is None else np.asarray(filtered[-1], dtype=complex),
        grid=grid,
        z_surface_m=float(surface.z_surface_m),
        n_medium=float(n_medium),
        wavelength_m=wavelength,
        metadata=metadata,
    )


def run_through_sample(
    surface: Any,
    config: TwinConfig,
    *,
    correct_interface: bool = False,
    write_depth_m: float | None = None,
    z_values_m: Sequence[float] | None = None,
) -> SampleResult:
    """Refract the surface field and propagate inside the sample medium.

    ``correct_interface=True`` applies an ideal phase-only numerical
    cancellation of the flat-interface pupil aberration.  It is labelled
    ``ideal_numerical_correction`` and ``diagnostic_only`` unless a separate
    hardware route explicitly implements that correction.
    """

    n_medium = float(config.material.refractive_index)
    if abs(n_medium - 1.0) <= 1e-12:
        raise ValueError("run_through_sample expects a sample medium with refractive_index != 1.")
    refracted = refract_surface_field(surface, n_medium=n_medium, wavelength_m=config.laser.wavelength_m)
    grid = _as_grid(refracted)
    input_fields = _component_list(refracted.Ex, refracted.Ey, refracted.Ez)
    surface_power = _field_power(input_fields, grid)
    surface_T = float(config.energy.sample_surface_transmission)
    if surface_T < 0.0:
        raise ValueError("sample_surface_transmission must be non-negative.")
    medium_fields = _apply_scalar(input_fields, np.sqrt(surface_T))
    medium_power = _field_power(medium_fields, grid)
    target_depth = float(config.material.write_depth_m if write_depth_m is None else write_depth_m)
    if target_depth < 0.0:
        raise ValueError("write_depth_m must be non-negative and measured from the sample surface.")

    pupil_grid = _grid_like_surface_pupil(grid, config)
    interface_phase = interface_aberration_pupil(
        pupil_grid,
        config.laser,
        config.objective,
        config.material,
        depth_m=target_depth,
        n1=1.0,
    )
    correction_phase, correction_mask_report = _interface_correction_mask(interface_phase, config)
    axial_target_phase = _axial_target_phase(grid, config, n_medium=n_medium, target_depth_m=target_depth)
    total_phase = axial_target_phase + interface_phase + (correction_phase if bool(correct_interface) else 0.0)
    residual_phase = _signed_wrapped_phase(interface_phase + correction_phase)

    ideal_fields = _apply_angular_phase(medium_fields, axial_target_phase)
    uncorrected_fields = _apply_angular_phase(medium_fields, axial_target_phase + interface_phase)
    corrected_fields = _apply_angular_phase(medium_fields, axial_target_phase + interface_phase + correction_phase)
    selected_fields = corrected_fields if bool(correct_interface) else uncorrected_fields

    powers = {
        "no_interface_input_power": _field_power(ideal_fields, grid),
        "uncorrected_input_power": _field_power(uncorrected_fields, grid),
        "corrected_input_power": _field_power(corrected_fields, grid),
    }
    ideal_power = max(float(powers["no_interface_input_power"]), BT_EPS)
    phase_rel_error = max(
        abs(float(powers["uncorrected_input_power"]) - ideal_power),
        abs(float(powers["corrected_input_power"]) - ideal_power),
    ) / ideal_power
    if phase_rel_error > 1e-6:
        raise RuntimeError(f"Phase-only interface correction changed total power by {phase_rel_error:.3e}.")

    zern_before = fit_interface_zernike_terms(pupil_grid, interface_phase, config.objective.pupil_radius_m)
    zern_after = fit_interface_zernike_terms(pupil_grid, residual_phase, config.objective.pupil_radius_m)

    check_z = target_depth
    ideal_at_depth = _field_at_z(ideal_fields[0], grid, config, check_z, n_medium=n_medium)
    uncorrected_at_depth = _field_at_z(uncorrected_fields[0], grid, config, check_z, n_medium=n_medium)
    corrected_at_depth = _field_at_z(corrected_fields[0], grid, config, check_z, n_medium=n_medium)
    corrected_rel_l2 = _rel_l2(corrected_at_depth, ideal_at_depth)
    uncorrected_rel_l2 = _rel_l2(uncorrected_at_depth, ideal_at_depth)
    if bool(correct_interface) and corrected_rel_l2 > 3e-2:
        raise RuntimeError(f"Corrected field does not recover the independent no-interface reference: rel-L2={corrected_rel_l2:.3e}.")
    if bool(correct_interface) and abs(float(zern_after["spherical_waves"])) > 1e-3:
        raise RuntimeError(
            f"Residual spherical Zernike is not cancelled: {float(zern_after['spherical_waves']):.3e} waves."
        )

    z_axis = _default_sample_z_values(config) if z_values_m is None else np.asarray(z_values_m, dtype=float)
    if z_axis.ndim != 1 or z_axis.size == 0:
        raise ValueError("z_values_m must be a non-empty 1D sequence.")
    if np.nanmin(z_axis) < -BT_EPS:
        raise ValueError("Through-sample z_axis_m is measured from the surface and must be non-negative.")

    volume = _propagate_components(selected_fields, grid, config, z_axis, n_medium=n_medium)
    design = compute_design_from_config(config)
    correction_label = "ideal_numerical_correction" if bool(correct_interface) else "uncorrected_interface"
    energy_report = {
        "surface_power_air": surface_power,
        "surface_transmission": surface_T,
        "medium_input_power": medium_power,
        **powers,
        "phase_only_power_relative_error": float(phase_rel_error),
        "volume_power_min": float(np.nanmin(volume["total_power"])),
        "volume_power_max": float(np.nanmax(volume["total_power"])),
        "volume_power_drift_fraction": float(volume["propagation_power_drift_fraction"]),
        "quantitative_metrics_valid": bool(volume["quantitative_metrics_valid"]),
        "quantitative_metrics_invalid_reason": str(volume["quantitative_metrics_invalid_reason"]),
        "power_check_pass": bool(phase_rel_error <= 1e-6),
    }
    correction_report = {
        "depth_checked_m": check_z,
        "target_write_depth_m": target_depth,
        "axial_target_phase_rms_rad": float(np.sqrt(np.mean(_signed_wrapped_phase(axial_target_phase) ** 2))),
        "phase_only_correction": True,
        "interface_correction_label": correction_label,
        "interface_correction_implementation": "diagnostic_only",
        "correction_mask": correction_mask_report,
        "no_interface_reference": "independent_homogeneous_medium_propagation_from_refracted_surface_field",
        "reference_uses_corrected_array": False,
        "zernike_before": zern_before,
        "zernike_after": zern_after,
        "corrected_rel_l2_to_no_interface": corrected_rel_l2,
        "uncorrected_rel_l2_to_no_interface": uncorrected_rel_l2,
        "corrected_matches_no_interface_pass": bool(corrected_rel_l2 <= 3e-2),
        "residual_spherical_pass": bool(abs(float(zern_after["spherical_waves"])) <= 1e-3),
        "applied_phase_rms_rad": float(np.sqrt(np.mean(np.asarray(total_phase, dtype=float) ** 2))),
    }
    metrics = _sample_metrics(
        volume,
        design,
        config,
        n_medium=n_medium,
        correction=correction_label,
        target_depth_m=target_depth,
        energy_report=energy_report,
        correction_report=correction_report,
    )
    return SampleResult(
        volume=np.asarray(volume["xz"], dtype=float),
        z_axis_m=np.asarray(volume["z"], dtype=float),
        metrics=metrics,
        correction=correction_label,
        energy_report=energy_report,
        volume_result=volume,
        surface_field=refracted,
        correction_report=correction_report,
    )


def plot_sample_result_comparison(
    uncorrected: SampleResult,
    corrected: SampleResult,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save a compact through-sample corrected-vs-uncorrected diagnostic figure."""

    import matplotlib.pyplot as plt

    from . import vbb_metrics, vbb_style

    vbb_style.apply_style()
    results = [uncorrected, corrected]
    labels = ["uncorrected interface", "ideal numerical correction"]
    xy_max = max(float(np.nanmax(r.volume_result["planes"]["peak"])) for r in results) + BT_EPS
    xz_max = max(float(np.nanmax(r.volume_result["xz"])) for r in results) + BT_EPS
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), constrained_layout=True)
    xy_artist = None
    xz_artist = None
    for col, (result, label) in enumerate(zip(results, labels, strict=True)):
        volume = result.volume_result
        grid = volume["crop_grid"]
        x = np.asarray(grid["x"], dtype=float) / BT_UM
        z = np.asarray(volume["z"], dtype=float) / BT_UM
        xy_artist = axes[0, col].imshow(
            vbb_style.display_scale(np.asarray(volume["planes"]["peak"], dtype=float) / xy_max, gamma=0.45, normalise=False),
            origin="lower",
            extent=[float(x[0]), float(x[-1]), float(x[0]), float(x[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, col].set_title(f"{label} XY at peak depth")
        axes[0, col].set_xlabel("x [um, in-medium sample]")
        axes[0, col].set_ylabel("y [um, in-medium sample]")

        xz_artist = axes[1, col].imshow(
            vbb_style.display_scale(np.asarray(volume["xz"], dtype=float) / xz_max, gamma=0.45, normalise=False),
            origin="lower",
            aspect="auto",
            extent=[float(z[0]), float(z[-1]), float(x[0]), float(x[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, col].axvspan(
            float(result.metrics["zone_start_um"]),
            float(result.metrics["zone_end_um"]),
            color="white",
            alpha=0.12,
            lw=0,
        )
        per_z = vbb_metrics.per_z_metrics_from_volume(
            volume,
            ell=int(result.metrics["ell"]),
            kr_m_inv=float(result.metrics["kr_sample_m_inv"]),
            center_mode="centroid",
            reference_index=int(volume.get("peak_index", 0)),
        )
        radius_key = "core_radius_m" if int(result.metrics["ell"]) == 0 else "ring_radius_m"
        radius_um = np.asarray(per_z[radius_key], dtype=float) / BT_UM
        axes[1, col].plot(z, radius_um, color="white", lw=0.9, alpha=0.85)
        axes[1, col].plot(z, -radius_um, color="white", lw=0.9, alpha=0.85)
        axes[1, col].set_title(f"{label} XZ from surface")
        axes[1, col].set_xlabel("z [um from surface, in medium]")
        axes[1, col].set_ylabel("x [um, in-medium sample]")

    for result, label in zip(results, labels, strict=True):
        volume = result.volume_result
        z = np.asarray(volume["z"], dtype=float) / BT_UM
        peak = np.asarray(volume["peak"], dtype=float)
        peak_norm = peak / (float(np.nanmax(peak)) + BT_EPS)
        axes[0, 2].plot(z, peak_norm, label=f"{label} peak")
        if int(result.metrics["ell"]) == 0:
            trace = np.asarray(volume["onaxis"], dtype=float)
            trace_label = f"{label} on-axis"
        else:
            trace = peak
            trace_label = f"{label} ring peak"
        axes[0, 2].plot(z, trace / (float(np.nanmax(trace)) + BT_EPS), ls="--", label=trace_label)
        axes[0, 2].axvspan(
            float(result.metrics["zone_start_um"]),
            float(result.metrics["zone_end_um"]),
            alpha=0.08,
            lw=0,
        )
    axes[0, 2].set_title("Axial trace")
    axes[0, 2].set_xlabel("z [um from surface, in medium]")
    axes[0, 2].set_ylabel("normalised intensity")
    axes[0, 2].legend(loc="best")

    rows = [
        ("power rel. err", f"{corrected.energy_report['phase_only_power_relative_error']:.2e}"),
        ("spherical after", f"{corrected.correction_report['zernike_after']['spherical_waves']:.2e} waves"),
        ("corrected rel-L2", f"{corrected.correction_report['corrected_rel_l2_to_no_interface']:.2e}"),
        ("uncorrected rel-L2", f"{corrected.correction_report['uncorrected_rel_l2_to_no_interface']:.2e}"),
        ("zone corrected", f"{corrected.metrics['bessel_zone_um']:.2f} um"),
    ]
    axes[1, 2].axis("off")
    table = axes[1, 2].table(cellText=rows, colLabels=["check", "value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    axes[1, 2].set_title("Correction checks")

    if xy_artist is not None:
        cbar = fig.colorbar(xy_artist, ax=axes[0, :2], shrink=0.90)
        cbar.set_label("matched XY display intensity [a.u.]")
    if xz_artist is not None:
        cbar = fig.colorbar(xz_artist, ax=axes[1, :2], shrink=0.90)
        cbar.set_label("matched XZ display intensity [a.u.]")
    fig.suptitle(title or "Through-sample study: air SurfaceField into Cr:ZnSe")
    caption = (
        "Through-sample diagnostic. The input is the air-side SurfaceField; z is measured from "
        "the sample surface inside Cr:ZnSe. The correction branch is an ideal numerical phase-only "
        "interface cancellation, not an automatically lab-implemented correction."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"study_kind": "through_sample"})
    plt.close(fig)
    return out


__all__ = [
    "RefractedField",
    "SampleResult",
    "plot_sample_result_comparison",
    "refract_surface_field",
    "run_through_sample",
]
