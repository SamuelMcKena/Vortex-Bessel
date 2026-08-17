from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_axicon_tip_reference import tip_resolution
from vbb_study.digital_twin.vortex_continuous_propagation import (
    adjacent_row_continuity_metrics,
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_following_propagation import transverse_morphology_axis
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig, build_system_route
from vbb_study.equations.fields import fft2c


EPS = np.finfo(float).tiny


def _ell(case_id: str) -> int:
    return {"B0": 0, "V1": 1, "V3": 3}[case_id]


def _registry() -> dict[str, dict[str, Any]]:
    manifest = canonical_hardware_manifest()
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    lateral = (-500e-6, -250e-6, 0.0, 250e-6, 500e-6)
    tip = (0.0, 100e-6, 200e-6, 400e-6, 800e-6)
    return {
        "axicon_lateral_decentre_x": {
            "values": lateral,
            "display": "um",
            "units": "m",
            "builder": lambda v: SystemErrorConfig(axicon=AxiconError(decentre_m=(float(v), 0.0))),
            "fidelity": "physical_axicon_coordinate_translation",
        },
        "axicon_lateral_decentre_y": {
            "values": lateral,
            "display": "um",
            "units": "m",
            "builder": lambda v: SystemErrorConfig(axicon=AxiconError(decentre_m=(0.0, float(v)))),
            "fidelity": "physical_axicon_coordinate_translation",
        },
        "axicon_round_tip_radius": {
            "values": tip,
            "display": "um",
            "units": "m radial hyperbolic curvature scale",
            "tip_radius": True,
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(
                    tip_model="sharp" if float(v) == 0.0 else "hyperboloidal_round",
                    rounding_parameter_m=float(v) * math.tan(gamma),
                )
            ),
            "fidelity": "hyperbolic_round_tip_radial_parameter; shallow-angle scalar diffraction",
        },
        "axicon_flat_tip_radius": {
            "values": tip,
            "display": "um",
            "units": "m physical flat radius",
            "tip_radius": True,
            "builder": lambda v: SystemErrorConfig(
                axicon=AxiconError(
                    tip_model="sharp" if float(v) == 0.0 else "flat_blunt",
                    flat_tip_radius_m=float(v),
                )
            ),
            "fidelity": "continuous_flat_centre_conical_outer_surface; shallow-angle scalar diffraction",
        },
        "axicon_base_angle_scale": {
            "values": (0.9, 0.95, 1.0, 1.05, 1.1),
            "display": "ratio",
            "units": "ratio",
            "builder": lambda v: SystemErrorConfig(axicon=AxiconError(base_angle_scale=float(v))),
            "fidelity": "exact_normal_incidence_Snell_cone",
        },
        "axicon_index_scale": {
            "values": (0.99, 0.995, 1.0, 1.005, 1.01),
            "display": "ratio",
            "units": "ratio",
            "builder": lambda v: SystemErrorConfig(axicon=AxiconError(refractive_index_scale=float(v))),
            "fidelity": "exact_normal_incidence_Snell_cone; dispersion calibration required",
        },
    }


def _label(value: float, display: str) -> str:
    if display == "um":
        return "0 µm" if value == 0.0 else f"{value * 1e6:+.0f} µm"
    return f"{value:.4g}"


def _seed(family: str, value: float) -> tuple[float, float]:
    if family == "axicon_lateral_decentre_x":
        return float(value), 0.0
    if family == "axicon_lateral_decentre_y":
        return 0.0, float(value)
    return 0.0, 0.0


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


def _spectral_crop(field: np.ndarray, grid: dict[str, Any], halfwidth_cpm: float = 40_000.0):
    spectrum = np.abs(fft2c(field)) ** 2
    axis = np.asarray(grid.get("fx", np.asarray(grid["FX"])[0]), dtype=float)
    mask = np.abs(axis) <= halfwidth_cpm
    return axis[mask], spectrum[np.ix_(mask, mask)]


def _crop(field_intensity: np.ndarray, x: np.ndarray, cx: float, cy: float, halfwidth: float):
    ix = np.flatnonzero(np.abs(x - cx) <= halfwidth)
    iy = np.flatnonzero(np.abs(x - cy) <= halfwidth)
    if ix.size < 16 or iy.size < 16:
        raise RuntimeError("transverse morphology ROI is under-sampled")
    return x[ix] - cx, x[iy] - cy, field_intensity[np.ix_(iy, ix)]


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
    ell = _ell(case_id)
    z = np.arange(5e-3, 140e-3 + 1e-12, 1e-3)
    offsets = np.linspace(-260e-6, 260e-6, 521)
    xy_halfwidth = 200e-6 if ell <= 1 else 280e-6
    records: list[dict[str, Any]] = []

    for value in values:
        route = build_system_route(case_id, grid_n=grid_n, config=spec["builder"](value))
        grid = dict(route["grid"])
        if spec.get("tip_radius") and value != 0.0:
            resolved = tip_resolution(value, float(grid["dx"]), minimum_pixels=12.0)
            if not resolved.resolved:
                raise RuntimeError(
                    f"{family}={value:g} m under-resolved: {resolved.radius_pixels:.2f} pixels"
                )
        else:
            resolved = None

        wavelength = float(route["metadata"]["wavelength_m"])
        propagator = build_fixed_support_spectrum(
            route["post_axicon"],
            grid,
            wavelength_m=wavelength,
            z_max_m=float(np.max(z)),
            minimum_retained_spectral_power=0.995,
        )
        uref = native_field_at_z(propagator, z_reference_m)
        iref = np.abs(uref) ** 2
        seed_x, seed_y = _seed(family, value)
        axis = transverse_morphology_axis(
            uref,
            grid,
            vortex_charge=ell,
            seed_x_m=seed_x,
            seed_y_m=seed_y,
        )

        # Both maps are genuine fixed physical planes crossing the detected
        # Bessel/vortex axis at z_reference_m.  Their coordinates never move with z.
        x_abs = float(axis.x_m) + offsets
        y_abs = float(axis.y_m) + offsets
        longitudinal = build_fixed_plane_longitudinal_map(
            propagator,
            z_values_m=z,
            x_coordinates_m=x_abs,
            y_coordinates_m=y_abs,
            fixed_x_m=float(axis.x_m),
            fixed_y_m=float(axis.y_m),
            source_label=f"{case_id}:{family}={value}",
        )
        ixref = int(np.argmin(np.abs(z - z_reference_m)))
        x_profile = longitudinal.xz_intensity[ixref].copy()
        y_profile = longitudinal.yz_intensity[ixref].copy()
        x_peak_z = np.max(longitudinal.xz_intensity, axis=1)
        y_peak_z = np.max(longitudinal.yz_intensity, axis=1)
        x_line_power_z = np.trapezoid(longitudinal.xz_intensity, x_abs, axis=1)
        y_line_power_z = np.trapezoid(longitudinal.yz_intensity, y_abs, axis=1)
        cx = adjacent_row_continuity_metrics(longitudinal.xz_intensity)
        cy = adjacent_row_continuity_metrics(longitudinal.yz_intensity)
        fx, spectrum = _spectral_crop(route["post_axicon"], grid)
        records.append(
            {
                "value": value,
                "label": _label(value, spec["display"]),
                "grid": grid,
                "axis": axis,
                "iref": iref,
                "x_abs": x_abs,
                "y_abs": y_abs,
                "longitudinal": longitudinal,
                "x_profile": x_profile,
                "y_profile": y_profile,
                "x_peak_z": x_peak_z,
                "y_peak_z": y_peak_z,
                "x_line_power_z": x_line_power_z,
                "y_line_power_z": y_line_power_z,
                "continuity_x": cx,
                "continuity_y": cy,
                "propagator": propagator,
                "fx": fx,
                "spectrum": spectrum,
                "tip_resolution": resolved,
            }
        )

    nominal_value = 0.0 if 0.0 in values else 1.0
    nominal = min(records, key=lambda r: abs(r["value"] - nominal_value))
    nominal_ref_peak = max(
        float(np.max(nominal["x_profile"])),
        float(np.max(nominal["y_profile"])),
        EPS,
    )
    nominal_axial_peak = max(
        float(np.max(nominal["x_peak_z"])),
        float(np.max(nominal["y_peak_z"])),
        EPS,
    )
    nominal_line_power = max(
        float(np.max(nominal["x_line_power_z"])),
        float(np.max(nominal["y_line_power_z"])),
        EPS,
    )

    rows: list[dict[str, Any]] = []
    for rec in records:
        axis = rec["axis"]
        prop = rec["propagator"]
        tip_res = rec["tip_resolution"]
        rows.append(
            {
                "case_id": case_id,
                "vortex_charge": ell,
                "family": family,
                "value": rec["value"],
                "units": spec["units"],
                "fidelity": spec["fidelity"],
                "grid_n": grid_n,
                "dx_m": float(rec["grid"]["dx"]),
                "z_reference_m": z_reference_m,
                "fixed_xz_plane_y_m": float(axis.y_m),
                "fixed_yz_plane_x_m": float(axis.x_m),
                "morphology_axis_x_m": float(axis.x_m),
                "morphology_axis_y_m": float(axis.y_m),
                "morphology_axis_method": axis.method,
                "detected_topological_charge": int(axis.detected_topological_charge),
                "selected_singularity_count": int(axis.selected_singularity_count),
                "fixed_support_retained_spectral_power_fraction": float(
                    prop.retained_spectral_power_fraction
                ),
                "xz_peak_at_reference_to_nominal": float(np.max(rec["x_profile"]) / nominal_ref_peak),
                "yz_peak_at_reference_to_nominal": float(np.max(rec["y_profile"]) / nominal_ref_peak),
                "xz_axial_peak_max_to_nominal": float(np.max(rec["x_peak_z"]) / nominal_axial_peak),
                "yz_axial_peak_max_to_nominal": float(np.max(rec["y_peak_z"]) / nominal_axial_peak),
                "xz_line_power_max_to_nominal": float(np.max(rec["x_line_power_z"]) / nominal_line_power),
                "yz_line_power_max_to_nominal": float(np.max(rec["y_line_power_z"]) / nominal_line_power),
                **{f"xz_{k}": v for k, v in rec["continuity_x"].items()},
                **{f"yz_{k}": v for k, v in rec["continuity_y"].items()},
                "tip_radius_pixels": (
                    float(tip_res.radius_pixels) if tip_res is not None else float("nan")
                ),
            }
        )

    # Main fixed-plane morphology figure.
    fig, axes = plt.subplots(4, len(records), figsize=(3.25 * len(records), 11.8), constrained_layout=True)
    native_x = np.asarray(records[0]["grid"]["x"], dtype=float)
    for col, rec in enumerate(records):
        fx = rec["fx"]
        spec_img = rec["spectrum"]
        axes[0, col].imshow(
            spec_img / max(float(np.max(spec_img)), EPS),
            origin="lower",
            extent=[fx[0] / 1e3, fx[-1] / 1e3, fx[0] / 1e3, fx[-1] / 1e3],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[0, col].set_title(rec["label"])
        axes[0, col].set_xlabel("fx (10³ m⁻¹)")
        axes[0, col].set_ylabel("fy (10³ m⁻¹)")

        axis = rec["axis"]
        xr, yr, crop = _crop(rec["iref"], native_x, float(axis.x_m), float(axis.y_m), xy_halfwidth)
        axes[1, col].imshow(
            crop / max(float(np.max(crop)), EPS),
            origin="lower",
            extent=[xr[0] * 1e6, xr[-1] * 1e6, yr[0] * 1e6, yr[-1] * 1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[1, col].axhline(0.0, linewidth=0.5, alpha=0.5)
        axes[1, col].axvline(0.0, linewidth=0.5, alpha=0.5)
        axes[1, col].set_xlabel("Δx from axis at zref (µm)")
        axes[1, col].set_ylabel("Δy from axis at zref (µm)")

        longi = rec["longitudinal"]
        axes[2, col].imshow(
            (longi.xz_intensity / max(float(np.max(longi.xz_intensity)), EPS)).T,
            origin="lower",
            aspect="auto",
            extent=[z[0] * 1e3, z[-1] * 1e3, offsets[0] * 1e6, offsets[-1] * 1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[3, col].imshow(
            (longi.yz_intensity / max(float(np.max(longi.yz_intensity)), EPS)).T,
            origin="lower",
            aspect="auto",
            extent=[z[0] * 1e3, z[-1] * 1e3, offsets[0] * 1e6, offsets[-1] * 1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        for row, name in ((2, "Δx"), (3, "Δy")):
            axes[row, col].axhline(0.0, linewidth=0.5, alpha=0.5)
            axes[row, col].axvline(z_reference_m * 1e3, linewidth=0.5, alpha=0.5)
            axes[row, col].set_xlabel("z from axicon (mm)")
            axes[row, col].set_ylabel(f"{name} from fixed zref axis (µm)")

    fig.suptitle(
        f"{case_id} — {family}\nfixed physical x-z/y-z planes; one spectral support for entire z sweep",
        fontsize=12,
    )
    figure_dir = figure_root / case_id
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{family}_continuous_fixed_plane.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Common-scale transverse line profiles at z_reference.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for rec in records:
        axes[0, 0].plot(rec["x_abs"] * 1e6, rec["x_profile"] / nominal_ref_peak, label=rec["label"])
        axes[0, 1].plot(rec["y_abs"] * 1e6, rec["y_profile"] / nominal_ref_peak, label=rec["label"])
        axes[1, 0].plot(offsets * 1e6, rec["x_profile"] / nominal_ref_peak, label=rec["label"])
        axes[1, 1].plot(offsets * 1e6, rec["y_profile"] / nominal_ref_peak, label=rec["label"])
    axes[0, 0].set_xlabel("laboratory x (µm)")
    axes[0, 1].set_xlabel("laboratory y (µm)")
    axes[1, 0].set_xlabel("Δx from detected axis (µm)")
    axes[1, 1].set_xlabel("Δy from detected axis (µm)")
    for ax in axes.ravel():
        ax.set_ylabel("I / nominal reference-plane peak")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{case_id} — {family}: transverse line profiles at z={z_reference_m*1e3:.0f} mm")
    fig.savefig(figure_dir / f"{family}_linear_profiles_zref.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Axial peak and line-integrated intensity on the same nominal scales.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), constrained_layout=True)
    for rec in records:
        axes[0].plot(z * 1e3, rec["x_peak_z"] / nominal_axial_peak, label=f"x: {rec['label']}")
        axes[0].plot(z * 1e3, rec["y_peak_z"] / nominal_axial_peak, linestyle="--", label=f"y: {rec['label']}")
        axes[1].plot(z * 1e3, rec["x_line_power_z"] / nominal_line_power, label=f"x: {rec['label']}")
        axes[1].plot(z * 1e3, rec["y_line_power_z"] / nominal_line_power, linestyle="--", label=f"y: {rec['label']}")
    axes[0].set_ylabel("slice peak I / nominal maximum")
    axes[1].set_ylabel("line-integrated I / nominal maximum")
    axes[1].set_xlabel("z from axicon (mm)")
    for ax in axes:
        ax.axvline(z_reference_m * 1e3, linewidth=0.7, alpha=0.5)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=7)
    fig.suptitle(f"{case_id} — {family}: absolute axial intensity comparison")
    fig.savefig(figure_dir / f"{family}_axial_intensity_profiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Save raw numerical arrays so visual artefacts can be audited directly.
    payload: dict[str, Any] = {"z_m": z, "offsets_m": offsets}
    for idx, rec in enumerate(records):
        payload[f"value_{idx}"] = np.asarray([rec["value"]], dtype=float)
        payload[f"xz_{idx}"] = rec["longitudinal"].xz_intensity
        payload[f"yz_{idx}"] = rec["longitudinal"].yz_intensity
        payload[f"x_profile_{idx}"] = rec["x_profile"]
        payload[f"y_profile_{idx}"] = rec["y_profile"]
        payload[f"x_peak_z_{idx}"] = rec["x_peak_z"]
        payload[f"y_peak_z_{idx}"] = rec["y_peak_z"]
    validation_dir = output_root / case_id
    validation_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(validation_dir / f"{family}_continuous_propagation_audit.npz", **payload)
    _write_csv(validation_dir / f"{family}_continuous_metrics.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/axicon_physics_v3"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/figures/axicon_physics_v3"))
    args = parser.parse_args()

    registry = _registry()
    families = list(registry) if args.families == ["all"] else list(args.families)
    unknown = [f for f in families if f not in registry]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}")

    rows: list[dict[str, Any]] = []
    for case_id in args.cases:
        for family in families:
            print(f"running continuous diagnostics {case_id}/{family}", flush=True)
            rows.extend(
                run_family(
                    family,
                    case_id=case_id,
                    grid_n=int(args.grid_n),
                    z_reference_m=float(args.z_mm) * 1e-3,
                    output_root=args.output_root,
                    figure_root=args.figure_root,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "axicon_physics_v3_metrics.csv", rows)
    manifest = {
        "outcome": "AXICON-PHYSICS-V3-CONTINUOUS-FIXED-PLANE",
        "report_figures_authorised": False,
        "grid_n": int(args.grid_n),
        "cases": list(args.cases),
        "families": families,
        "longitudinal_contract": "fixed physical planes; no per-z recentering or coordinate warp",
        "propagation_contract": "one max-z Matsushima support mask applied once, then continuous exp(i*kz*z)",
        "intensity_comparison_contract": "linear/axial curves share nominal-case intensity scales",
        "raw_array_audit": True,
    }
    (args.output_root / "axicon_physics_v3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
