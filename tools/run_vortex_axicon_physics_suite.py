from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import (
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.phase2e_spectral_propagation import (
    build_dense_spectral_propagation,
)
from vbb_study.digital_twin.vortex_axicon_oblique_wave import (
    build_carrier_tracked_oblique_axicon_route,
)
from vbb_study.digital_twin.vortex_axicon_tip_reference import tip_resolution
from vbb_study.digital_twin.vortex_following_propagation import (
    build_beam_following_propagation,
    transverse_morphology_axis,
)
from vbb_study.digital_twin.vortex_morphology_tracking import (
    LongitudinalAxisTrack,
    track_bessel_feature_axis,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_system_route,
)
from vbb_study.equations.fields import fft2c
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny


def _ell(case_id: str) -> int:
    try:
        return {"B0": 0, "V1": 1, "V3": 3}[case_id]
    except KeyError as exc:
        raise ValueError(f"unsupported case {case_id!r}") from exc


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _registry() -> dict[str, dict[str, Any]]:
    manifest = canonical_hardware_manifest()
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    lateral = (-500e-6, -250e-6, 0.0, 250e-6, 500e-6)
    tilt = tuple(math.radians(v) for v in (-10.0, -5.0, 0.0, 5.0, 10.0))
    tip_radius = (0.0, 100e-6, 200e-6, 400e-6, 800e-6)
    return {
        "axicon_lateral_decentre_x": {
            "values": lateral,
            "units": "m",
            "display": "um",
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(decentre_m=(float(v), 0.0))
            ),
            "axis": "x",
            "fidelity": "physical_axicon_coordinate_translation",
        },
        "axicon_lateral_decentre_y": {
            "values": lateral,
            "units": "m",
            "display": "um",
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(decentre_m=(0.0, float(v)))
            ),
            "axis": "y",
            "fidelity": "physical_axicon_coordinate_translation",
        },
        "axicon_rigid_tilt_x": {
            "values": tilt,
            "units": "rad",
            "display": "deg",
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(tilt_rad=(float(v), 0.0))
            ),
            "axis": "x",
            "fidelity": "REJECTED carrier-tracked scalar oblique thin axicon; use explicit refractive solver",
        },
        "axicon_rigid_tilt_y": {
            "values": tilt,
            "units": "rad",
            "display": "deg",
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(tilt_rad=(0.0, float(v)))
            ),
            "axis": "y",
            "fidelity": "REJECTED carrier-tracked scalar oblique thin axicon; use explicit refractive solver",
        },
        "axicon_round_tip_radius": {
            "values": tip_radius,
            "units": "m radial hyperbolic curvature scale",
            "display": "um",
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(
                    tip_model="sharp" if float(v) == 0.0 else "hyperboloidal_round",
                    # Production sag historically uses vertical parameter a.
                    # For f=v(sqrt(r^2+r_h^2)-r_h), a=v*r_h, so the external
                    # sweep is expressed in the physically legible radial scale.
                    rounding_parameter_m=float(v) * math.tan(gamma),
                )
            ),
            "axis": None,
            "tip_characteristic_radius": True,
            "fidelity": "hyperbolic_round_tip_radial_parameter; shallow-angle scalar diffraction",
        },
        "axicon_flat_tip_radius": {
            "values": tip_radius,
            "units": "m physical flat radius",
            "display": "um",
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(
                    tip_model="sharp" if float(v) == 0.0 else "flat_blunt",
                    flat_tip_radius_m=float(v),
                )
            ),
            "axis": None,
            "tip_characteristic_radius": True,
            "fidelity": "continuous_flat_centre_conical_outer_surface; shallow-angle scalar diffraction",
        },
        "axicon_base_angle_scale": {
            "values": (0.9, 0.95, 1.0, 1.05, 1.1),
            "units": "ratio",
            "display": "ratio",
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(base_angle_scale=float(v))
            ),
            "axis": None,
            "fidelity": "exact_normal_incidence_Snell_cone",
        },
        "axicon_index_scale": {
            "values": (0.99, 0.995, 1.0, 1.005, 1.01),
            "units": "ratio",
            "display": "ratio",
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(refractive_index_scale=float(v))
            ),
            "axis": None,
            "fidelity": "exact_normal_incidence_Snell_cone; dispersion calibration required",
        },
    }


def _format_value(value: float, display: str) -> str:
    if display == "um":
        return f"{float(value) * 1e6:+.0f} µm" if value != 0 else "0 µm"
    if display == "deg":
        return f"{math.degrees(float(value)):+.1f}°" if value != 0 else "0°"
    return f"{float(value):.4g}"


def _metrics(I: np.ndarray, grid: dict[str, Any]) -> dict[str, float]:
    arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    total = float(np.sum(arr))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    cx = float(np.sum(arr * X) / max(total, EPS))
    cy = float(np.sum(arr * Y) / max(total, EPS))
    radial_second_moment = float(
        np.sum(arr * ((X - cx) ** 2 + (Y - cy) ** 2)) / max(total, EPS)
    )
    return {
        "peak_au": float(np.max(arr)),
        "power_au": total * float(grid["dx"]) ** 2,
        "centroid_x_m": cx,
        "centroid_y_m": cy,
        "rms_radius_about_energy_centroid_m": float(
            np.sqrt(max(radial_second_moment, 0.0))
        ),
    }


def _line_metrics(I: np.ndarray, offsets: np.ndarray, z: np.ndarray) -> dict[str, float]:
    arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    p = np.sum(arr, axis=1)
    centre = np.sum(arr * offsets[None, :], axis=1) / np.maximum(p, EPS)
    var = np.sum(arr * (offsets[None, :] - centre[:, None]) ** 2, axis=1) / np.maximum(p, EPS)
    width = np.sqrt(np.maximum(var, 0.0))
    peak = np.max(arr, axis=1)
    active = peak >= 0.15 * max(float(np.max(peak)), EPS)
    return {
        "mean_width_m": float(np.mean(width[active])) if np.any(active) else float("nan"),
        "active_length_m": float(z[active][-1] - z[active][0]) if np.count_nonzero(active) >= 2 else 0.0,
        "centre_span_m": float(np.ptp(centre[active])) if np.any(active) else float("nan"),
    }


def _crop_about_axis(
    arr: np.ndarray,
    x: np.ndarray,
    *,
    axis_x_m: float,
    axis_y_m: float,
    halfwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.flatnonzero(np.abs(x - float(axis_x_m)) <= float(halfwidth))
    iy = np.flatnonzero(np.abs(x - float(axis_y_m)) <= float(halfwidth))
    if ix.size < 16 or iy.size < 16:
        raise ValueError("morphology-axis ROI has too few native samples")
    crop = np.asarray(arr)[np.ix_(iy, ix)]
    return (
        x[ix] - float(axis_x_m),
        x[iy] - float(axis_y_m),
        crop,
    )


def _spectral_crop(
    field: np.ndarray,
    grid: dict[str, Any],
    *,
    halfwidth_cpm: float = 35_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(fft2c(np.asarray(field, dtype=np.complex128))) ** 2
    axis = np.asarray(grid.get("fx", np.asarray(grid["FX"])[0]), dtype=float)
    mask = np.abs(axis) <= float(halfwidth_cpm)
    return axis[mask], spectrum[np.ix_(mask, mask)]


def _route(case_id: str, grid_n: int, config: SystemErrorConfig, family: str) -> dict[str, Any]:
    if family.startswith("axicon_rigid_tilt"):
        return build_carrier_tracked_oblique_axicon_route(
            case_id,
            grid_n=grid_n,
            config=config,
        )
    return build_system_route(case_id, grid_n=grid_n, config=config)


def _transverse_seed(family: str, value: float) -> tuple[float, float]:
    if family == "axicon_lateral_decentre_x":
        return float(value), 0.0
    if family == "axicon_lateral_decentre_y":
        return 0.0, float(value)
    return 0.0, 0.0


def _constant_track(value_m: float, count: int, method: str) -> LongitudinalAxisTrack:
    values = np.full(int(count), float(value_m), dtype=float)
    return LongitudinalAxisTrack(
        coordinate_m=values,
        detected_mask=np.ones(int(count), dtype=bool),
        method=str(method),
        seed_coordinate_m=float(value_m),
        detected_fraction=1.0,
        maximum_detected_step_m=0.0,
    )


def _tracked_paths(
    *,
    family: str,
    value: float,
    vortex_charge: int,
    fixed: Any,
    fixed_coordinate: np.ndarray,
    grid: dict[str, Any],
    wavelength_m: float,
    z: np.ndarray,
    scalar_field: np.ndarray,
) -> tuple[LongitudinalAxisTrack, LongitudinalAxisTrack]:
    """Find x(z), y(z) without allowing the orthogonal slice to miss the beam.

    For a lateral x translation, the primary x path is extracted from the wide
    laboratory x-z map.  A provisional direct spectral propagation then samples
    y through that x path, allowing the y core to be determined on the actual
    beam rather than on the empty lab x=0 plane.  The y-decentre case is the
    rotational analogue.  Symmetric tip/material families retain the exact
    symmetry-axis zero path.
    """

    seed_x, seed_y = _transverse_seed(family, value)
    if family == "axicon_lateral_decentre_x":
        x_track = track_bessel_feature_axis(
            fixed.xz_intensity,
            fixed_coordinate,
            vortex_charge=vortex_charge,
            seed_coordinate_m=seed_x,
            search_halfwidth_m=0.24e-3,
            maximum_step_m=55e-6,
        )
        provisional_offsets = np.linspace(-0.45e-3, 0.45e-3, 321)
        provisional = build_beam_following_propagation(
            grid=grid,
            wavelength_m=wavelength_m,
            z_values_m=z,
            transverse_offsets_m=provisional_offsets,
            scalar_field=scalar_field,
            x_axis_m=x_track.coordinate_m,
            y_axis_m=seed_y,
            source_label=f"orthogonal-y-track:{family}={value}",
        )
        y_relative = track_bessel_feature_axis(
            provisional.yz_intensity,
            provisional_offsets,
            vortex_charge=vortex_charge,
            seed_coordinate_m=0.0,
            search_halfwidth_m=0.20e-3,
            maximum_step_m=55e-6,
        )
        y_track = LongitudinalAxisTrack(
            coordinate_m=seed_y + y_relative.coordinate_m,
            detected_mask=y_relative.detected_mask,
            method="orthogonal_through_primary_x_core__" + y_relative.method,
            seed_coordinate_m=seed_y,
            detected_fraction=y_relative.detected_fraction,
            maximum_detected_step_m=y_relative.maximum_detected_step_m,
        )
        return x_track, y_track

    if family == "axicon_lateral_decentre_y":
        y_track = track_bessel_feature_axis(
            fixed.yz_intensity,
            fixed_coordinate,
            vortex_charge=vortex_charge,
            seed_coordinate_m=seed_y,
            search_halfwidth_m=0.24e-3,
            maximum_step_m=55e-6,
        )
        provisional_offsets = np.linspace(-0.45e-3, 0.45e-3, 321)
        provisional = build_beam_following_propagation(
            grid=grid,
            wavelength_m=wavelength_m,
            z_values_m=z,
            transverse_offsets_m=provisional_offsets,
            scalar_field=scalar_field,
            x_axis_m=seed_x,
            y_axis_m=y_track.coordinate_m,
            source_label=f"orthogonal-x-track:{family}={value}",
        )
        x_relative = track_bessel_feature_axis(
            provisional.xz_intensity,
            provisional_offsets,
            vortex_charge=vortex_charge,
            seed_coordinate_m=0.0,
            search_halfwidth_m=0.20e-3,
            maximum_step_m=55e-6,
        )
        x_track = LongitudinalAxisTrack(
            coordinate_m=seed_x + x_relative.coordinate_m,
            detected_mask=x_relative.detected_mask,
            method="orthogonal_through_primary_y_core__" + x_relative.method,
            seed_coordinate_m=seed_x,
            detected_fraction=x_relative.detected_fraction,
            maximum_detected_step_m=x_relative.maximum_detected_step_m,
        )
        return x_track, y_track

    # Tip, base-angle and index sweeps are axisymmetric by construction in this
    # branch.  Using exactly zero is preferable to letting numerical centroid or
    # lobe asymmetry invent a steering signal that the model does not contain.
    return (
        _constant_track(0.0, z.size, "declared_axisymmetry_x_zero"),
        _constant_track(0.0, z.size, "declared_axisymmetry_y_zero"),
    )


def run_family(
    family: str,
    *,
    case_id: str,
    grid_n: int,
    z_reference_m: float,
    output_root: Path,
    figure_root: Path,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    spec = _registry()[family]
    values = tuple(float(v) for v in spec["values"])
    nominal_value = (
        min(values, key=abs)
        if 0.0 in values
        else min(values, key=lambda v: abs(v - 1.0))
    )
    ell = _ell(case_id)
    z = np.arange(5e-3, 140e-3 + 2e-3, 2e-3)
    fixed_coordinate = np.linspace(-1.2e-3, 1.2e-3, 481)
    morphology_offset = np.linspace(-220e-6, 220e-6, 401)
    xy_halfwidth = 190e-6 if ell <= 1 else 250e-6

    records: list[dict[str, Any]] = []
    for value in values:
        config = spec["builder"](value)
        route = _route(case_id, grid_n, config, family)
        grid = dict(route["grid"])
        if spec.get("tip_characteristic_radius") and value != 0.0:
            resolution = tip_resolution(value, float(grid["dx"]), minimum_pixels=12.0)
            if not resolution.resolved:
                raise RuntimeError(
                    f"{family}={value:g} m is under-resolved: "
                    f"{resolution.radius_pixels:.2f} px < {resolution.minimum_pixels:.0f} px"
                )
        else:
            resolution = None

        wavelength = float(route["metadata"]["wavelength_m"])
        propagated = angular_spectrum_propagate_bl(
            route["post_axicon"],
            grid,
            wavelength,
            z_reference_m,
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        Ixy = np.abs(propagated) ** 2
        xy_metrics = _metrics(Ixy, grid)
        seed_x, seed_y = _transverse_seed(family, value)
        morphology_axis = transverse_morphology_axis(
            propagated,
            grid,
            vortex_charge=ell,
            seed_x_m=seed_x,
            seed_y_m=seed_y,
        )

        fixed = build_dense_spectral_propagation(
            grid=grid,
            wavelength_m=wavelength,
            z_values_m=z,
            transverse_coordinates_m=fixed_coordinate,
            scalar_field=route["post_axicon"],
            source_label=f"fixed:{case_id}:{family}={value}",
        )
        x_track, y_track = _tracked_paths(
            family=family,
            value=value,
            vortex_charge=ell,
            fixed=fixed,
            fixed_coordinate=fixed_coordinate,
            grid=grid,
            wavelength_m=wavelength,
            z=z,
            scalar_field=route["post_axicon"],
        )
        following = build_beam_following_propagation(
            grid=grid,
            wavelength_m=wavelength,
            z_values_m=z,
            transverse_offsets_m=morphology_offset,
            scalar_field=route["post_axicon"],
            x_axis_m=x_track.coordinate_m,
            y_axis_m=y_track.coordinate_m,
            source_label=f"following:{case_id}:{family}={value}",
        )
        fx, spectrum_crop = _spectral_crop(route["post_axicon"], grid)
        records.append(
            {
                "value": value,
                "route": route,
                "grid": grid,
                "Ixy": Ixy,
                "xy_metrics": xy_metrics,
                "morphology_axis": morphology_axis,
                "following": following,
                "x_track": x_track,
                "y_track": y_track,
                "fx": fx,
                "spectrum": spectrum_crop,
                "resolution": resolution,
            }
        )

    nominal = next(rec for rec in records if np.isclose(rec["value"], nominal_value))
    peak0 = float(nominal["xy_metrics"]["peak_au"])
    power0 = float(nominal["xy_metrics"]["power_au"])
    rows: list[dict[str, Any]] = []
    for rec in records:
        follow = rec["following"]
        mx = _line_metrics(follow.xz_intensity, morphology_offset, z)
        my = _line_metrics(follow.yz_intensity, morphology_offset, z)
        meta = rec["route"]["metadata"]
        to_meta = meta.get("lab_to_tilted", {})
        from_meta = meta.get("tilted_to_lab", {})
        ray = meta.get("independent_snell_ray_reference", {})
        resolution = rec["resolution"]
        morphology_axis = rec["morphology_axis"]
        x_track = rec["x_track"]
        y_track = rec["y_track"]
        rows.append(
            {
                "case_id": case_id,
                "vortex_charge": ell,
                "family": family,
                "value": rec["value"],
                "units": spec["units"],
                "fidelity": spec["fidelity"],
                "grid_n": int(grid_n),
                "dx_m": float(rec["grid"]["dx"]),
                **rec["xy_metrics"],
                "morphology_axis_x_m": float(morphology_axis.x_m),
                "morphology_axis_y_m": float(morphology_axis.y_m),
                "morphology_axis_method": morphology_axis.method,
                "detected_topological_charge": int(morphology_axis.detected_topological_charge),
                "selected_singularity_count": int(morphology_axis.selected_singularity_count),
                "morphology_axis_distance_from_seed_m": float(morphology_axis.distance_from_seed_m),
                "energy_centroid_minus_axis_x_m": float(rec["xy_metrics"]["centroid_x_m"] - morphology_axis.x_m),
                "energy_centroid_minus_axis_y_m": float(rec["xy_metrics"]["centroid_y_m"] - morphology_axis.y_m),
                "tracked_x_axis_mean_m": float(np.mean(x_track.coordinate_m)),
                "tracked_y_axis_mean_m": float(np.mean(y_track.coordinate_m)),
                "tracked_x_axis_span_m": float(np.ptp(x_track.coordinate_m)),
                "tracked_y_axis_span_m": float(np.ptp(y_track.coordinate_m)),
                "tracked_x_axis_method": x_track.method,
                "tracked_y_axis_method": y_track.method,
                "tracked_x_detected_fraction": float(x_track.detected_fraction),
                "tracked_y_detected_fraction": float(y_track.detected_fraction),
                "tracked_x_max_detected_step_m": float(x_track.maximum_detected_step_m),
                "tracked_y_max_detected_step_m": float(y_track.maximum_detected_step_m),
                "peak_ratio_to_nominal": float(rec["xy_metrics"]["peak_au"] / max(peak0, EPS)),
                "power_ratio_to_nominal": float(rec["xy_metrics"]["power_au"] / max(power0, EPS)),
                "xz_following_mean_width_m": mx["mean_width_m"],
                "yz_following_mean_width_m": my["mean_width_m"],
                "xz_following_active_length_m": mx["active_length_m"],
                "yz_following_active_length_m": my["active_length_m"],
                "wave_width_anisotropy_fraction": float(
                    abs(mx["mean_width_m"] - my["mean_width_m"])
                    / max(0.5 * (mx["mean_width_m"] + my["mean_width_m"]), EPS)
                ),
                "snell_ray_cone_anisotropy_fraction": float(ray.get("cone_radius_anisotropy_fraction", float("nan"))),
                "lab_to_tilted_spectral_power_ratio": float(to_meta.get("spectral_power_ratio", 1.0)),
                "tilted_to_lab_spectral_power_ratio": float(from_meta.get("spectral_power_ratio", 1.0)),
                "tip_radius_pixels": (
                    float(resolution.radius_pixels) if resolution is not None else float("nan")
                ),
            }
        )

    fig, axes = plt.subplots(
        4,
        len(records),
        figsize=(3.25 * len(records), 11.5),
        constrained_layout=True,
        squeeze=False,
    )
    x_native = np.asarray(records[0]["grid"]["x"], dtype=float)
    for col, rec in enumerate(records):
        value = rec["value"]
        label = _format_value(value, spec["display"])
        spec_img = rec["spectrum"]
        axes[0, col].imshow(
            spec_img / max(float(np.max(spec_img)), EPS),
            origin="lower",
            extent=[
                rec["fx"][0] / 1e3,
                rec["fx"][-1] / 1e3,
                rec["fx"][0] / 1e3,
                rec["fx"][-1] / 1e3,
            ],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("fx (10³ m⁻¹)")
        axes[0, col].set_ylabel("fy (10³ m⁻¹)")

        m = rec["xy_metrics"]
        morphology_axis = rec["morphology_axis"]
        xr, yr, crop = _crop_about_axis(
            rec["Ixy"],
            x_native,
            axis_x_m=morphology_axis.x_m,
            axis_y_m=morphology_axis.y_m,
            halfwidth=xy_halfwidth,
        )
        axes[1, col].imshow(
            crop / max(float(np.max(crop)), EPS),
            origin="lower",
            extent=[xr[0] * 1e6, xr[-1] * 1e6, yr[0] * 1e6, yr[-1] * 1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[1, col].axhline(0.0, linewidth=0.5, color="white", alpha=0.45)
        axes[1, col].axvline(0.0, linewidth=0.5, color="white", alpha=0.45)
        axes[1, col].set_xlabel("Δx from Bessel/vortex axis (µm)")
        axes[1, col].set_ylabel("Δy from Bessel/vortex axis (µm)")
        axes[1, col].text(
            0.02,
            0.98,
            (
                f"peak/0={m['peak_au']/max(peak0,EPS):.3f}\n"
                f"P/0={m['power_au']/max(power0,EPS):.3f}\n"
                f"axis=({morphology_axis.x_m*1e6:.0f},{morphology_axis.y_m*1e6:.0f}) µm\n"
                f"energy c=({m['centroid_x_m']*1e6:.0f},{m['centroid_y_m']*1e6:.0f}) µm"
            ),
            transform=axes[1, col].transAxes,
            ha="left",
            va="top",
            fontsize=6.7,
            color="white",
        )

        follow = rec["following"]
        for row, data, axis_name in (
            (2, follow.xz_intensity, "Δx"),
            (3, follow.yz_intensity, "Δy"),
        ):
            axes[row, col].imshow(
                (data / max(float(np.max(data)), EPS)).T,
                origin="lower",
                aspect="auto",
                extent=[
                    z[0] * 1e3,
                    z[-1] * 1e3,
                    morphology_offset[0] * 1e6,
                    morphology_offset[-1] * 1e6,
                ],
                vmin=0,
                vmax=1,
                cmap="inferno",
            )
            axes[row, col].axhline(0.0, linewidth=0.6, color="white", alpha=0.55)
            axes[row, col].set_xlabel("z from axicon (mm)")
            axes[row, col].set_ylabel(f"{axis_name} from tracked Bessel/vortex axis (µm)")

    axes[0, 0].set_ylabel("fy (10³ m⁻¹)\npost-axicon spectrum")
    fig.suptitle(
        (
            f"{case_id} — {family}\n{spec['fidelity']} | morphology-axis centred; "
            "panels individually normalised"
        ),
        fontsize=12,
    )
    path = figure_root / case_id / f"{family}_physics_diagnostic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    _write_csv(output_root / case_id / f"{family}_metrics.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resolution-aware axicon physics diagnostics."
    )
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/validation/axicon_physics_v2"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("outputs/figures/axicon_physics_v2"),
    )
    args = parser.parse_args()

    registry = _registry()
    families = list(registry) if args.families == ["all"] else list(args.families)
    unknown = [f for f in families if f not in registry]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}")

    all_rows: list[dict[str, Any]] = []
    for case_id in args.cases:
        for family in families:
            if family.startswith("axicon_rigid_tilt"):
                raise SystemExit(
                    "rigid-tilt figures are blocked in this runner: the carrier-tracked thin "
                    "surrogate is intentionally rejected; use the explicit refractive-axicon "
                    "reference branch once physical hardware geometry is supplied"
                )
            print(f"running {case_id} / {family}", flush=True)
            all_rows.extend(
                run_family(
                    family,
                    case_id=case_id,
                    grid_n=int(args.grid_n),
                    z_reference_m=float(args.z_mm) * 1e-3,
                    output_root=args.output_root,
                    figure_root=args.figure_root,
                )
            )

    manifest = {
        "outcome": "AXICON-PHYSICS-V2",
        "report_figures_authorised": False,
        "cases": list(args.cases),
        "families": families,
        "grid_n": int(args.grid_n),
        "diagnostic_contract": (
            "post-axicon angular spectrum + topological/central-peak centred xy + "
            "continuity-tracked Bessel/vortex xz/yz; orthogonal slice is sampled through "
            "the primary translated core rather than fixed lab zero; morphology panels "
            "individually normalised; energy centroid, absolute peak/power and steering "
            "retained separately in CSV"
        ),
        "tip_resolution_policy": "nonzero local tip radius >= 12 native 2-D pixels",
        "tilt_policy": (
            "thin rotated-phase rigid tilt is rejected and cannot be rendered here; "
            "explicit two-surface refractive/eikonal solver is the replacement reference"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "axicon_physics_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_root / "axicon_physics_metrics.csv", all_rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
