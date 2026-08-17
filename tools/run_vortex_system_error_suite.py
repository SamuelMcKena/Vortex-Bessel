from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.vortex_continuous_propagation import (
    adjacent_row_continuity_metrics,
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_profile_evidence import (
    build_transverse_profile_evidence,
    profile_long_rows,
    profile_metrics,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import (
    blocked_or_data_driven_families,
    system_sweep_registry,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route


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


def _metrics(I: np.ndarray, grid: dict[str, Any]) -> dict[str, float]:
    arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    total = float(np.sum(arr))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    peak = float(np.max(arr))
    cx = float(np.sum(arr * X) / max(total, EPS))
    cy = float(np.sum(arr * Y) / max(total, EPS))
    radial_second_moment = float(
        np.sum(arr * ((X - cx) ** 2 + (Y - cy) ** 2)) / max(total, EPS)
    )
    return {
        "peak_au": peak,
        "power_au": total * float(grid["dx"]) ** 2,
        "centroid_x_m": cx,
        "centroid_y_m": cy,
        "rms_radius_m": float(np.sqrt(max(radial_second_moment, 0.0))),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel().copy()
    bb = np.asarray(b, dtype=float).ravel().copy()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= EPS:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.dot(aa, bb) / denom)


def _line_metrics(
    intensity_zx: np.ndarray,
    *,
    z_m: np.ndarray,
    coordinate_m: np.ndarray,
) -> dict[str, Any]:
    I = np.maximum(np.asarray(intensity_zx, dtype=float), 0.0)
    coord = np.asarray(coordinate_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    if I.shape != (z.size, coord.size):
        raise ValueError("line-map shape does not match z/transverse coordinates")

    line_power = np.trapezoid(I, coord, axis=1)
    discrete_weight = np.sum(I, axis=1)
    centre = np.sum(I * coord[None, :], axis=1) / np.maximum(discrete_weight, EPS)
    variance = (
        np.sum(I * (coord[None, :] - centre[:, None]) ** 2, axis=1)
        / np.maximum(discrete_weight, EPS)
    )
    width = np.sqrt(np.maximum(variance, 0.0))
    peak_trace = np.max(I, axis=1)

    active = peak_trace >= 0.15 * max(float(np.max(peak_trace)), EPS)
    if int(np.count_nonzero(active)) >= 2:
        slope, intercept = np.polyfit(z[active], centre[active], 1)
        active_length = float(z[active][-1] - z[active][0])
        mean_width = float(np.mean(width[active]))
        centre_span = float(np.max(centre[active]) - np.min(centre[active]))
    else:
        slope = float("nan")
        intercept = float("nan")
        active_length = 0.0
        mean_width = float("nan")
        centre_span = float("nan")

    return {
        "centre_m": centre,
        "width_m": width,
        "peak_trace": peak_trace,
        "line_integral_trace": line_power,
        "active_mask": active,
        "centre_slope_rad_approx": float(slope),
        "centre_intercept_m": float(intercept),
        "active_length_m": active_length,
        "mean_width_m": mean_width,
        "centre_span_m": centre_span,
    }


def _normalised_trace(trace: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(trace, dtype=float), 0.0)
    return arr / max(float(np.max(arr)), EPS)


def _crop_square(
    arr: np.ndarray,
    x: np.ndarray,
    *,
    centre_x_m: float,
    centre_y_m: float,
    halfwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.flatnonzero(np.abs(x - float(centre_x_m)) <= float(halfwidth))
    iy = np.flatnonzero(np.abs(x - float(centre_y_m)) <= float(halfwidth))
    if ix.size < 16 or iy.size < 16:
        raise RuntimeError("system-error morphology crop is under-sampled")
    return (
        x[ix] - float(centre_x_m),
        x[iy] - float(centre_y_m),
        np.asarray(arr)[np.ix_(iy, ix)],
    )


def _component_crop(
    arr: np.ndarray,
    x: np.ndarray,
    halfwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.abs(x) <= float(halfwidth)
    return x[mask], np.asarray(arr)[np.ix_(mask, mask)]


def _family_component_image(
    family: str,
    route: dict[str, Any],
    nominal: dict[str, Any],
) -> tuple[np.ndarray, str, str]:
    if family.startswith("beam_curvature"):
        phase = np.angle(
            np.asarray(route["input_beam"])
            * np.conj(np.asarray(nominal["input_beam"]))
        )
        return phase, "input beam phase difference vs nominal", "phase"
    if family.startswith("beam_"):
        return np.abs(route["input_beam"]) ** 2, "input beam intensity", "intensity"
    if family.startswith("slm"):
        return np.abs(route["fourier_plane_before_iris"]) ** 2, "4F Fourier plane before iris", "intensity"
    if family.startswith("fourf_iris"):
        after = np.asarray(route["fourier_plane_before_iris"]) * np.asarray(route["fourier_iris_mask"])
        return np.abs(after) ** 2, "4F Fourier plane after fixed iris", "intensity"
    if family.startswith("fourf_lens2"):
        return np.abs(route["post_4f_selected_order"]) ** 2, "4F output / selected-order plane before axicon", "intensity"
    if family.startswith("fourf"):
        return np.abs(route["fourier_plane_before_iris"]) ** 2, "4F Fourier plane before iris", "intensity"

    ratio = route["post_axicon_local"] / np.where(
        np.abs(route["field_on_axicon_plane"]) > 1e-9,
        route["field_on_axicon_plane"],
        1.0,
    )
    ratio0 = nominal["post_axicon_local"] / np.where(
        np.abs(nominal["field_on_axicon_plane"]) > 1e-9,
        nominal["field_on_axicon_plane"],
        1.0,
    )
    return np.angle(ratio * np.conj(ratio0)), "axicon phase difference vs nominal", "phase"


def _nominal_value(values: tuple[Any, ...]) -> Any:
    finite = [v for v in values if np.isfinite(float(v))]
    if any(np.isclose(float(v), 1.0) for v in finite):
        return min(finite, key=lambda v: abs(float(v) - 1.0))
    nonfinite = [v for v in values if not np.isfinite(float(v))]
    if nonfinite:
        return nonfinite[0]
    return min(values, key=lambda v: abs(float(v)))


def _same_value(a: Any, b: Any) -> bool:
    aa = float(a)
    bb = float(b)
    if not np.isfinite(aa) or not np.isfinite(bb):
        return (not np.isfinite(aa)) and (not np.isfinite(bb))
    return bool(np.isclose(aa, bb, rtol=0.0, atol=1e-15))


def _rotated_plane_diagnostics(route: dict[str, Any]) -> dict[str, float | str]:
    meta = route["metadata"]
    fourf = meta["fourf"]
    entries: dict[str, dict[str, Any]] = {
        "lens1_to": fourf["lens1"].get("lab_to_lens_plane", {}),
        "lens1_from": fourf["lens1"].get("lens_plane_to_lab", {}),
        "lens2_to": fourf["lens2"].get("lab_to_lens_plane", {}),
        "lens2_from": fourf["lens2"].get("lens_plane_to_lab", {}),
        "axicon_to": meta.get("lab_to_tilted", {}),
        "axicon_from": meta.get("tilted_to_lab", {}),
    }
    out: dict[str, float | str] = {}
    ratios: list[float] = []
    for name, entry in entries.items():
        ratio = float(entry.get("spectral_power_ratio", 1.0))
        out[f"{name}_spectral_power_ratio"] = ratio
        out[f"{name}_interpolation_model"] = str(entry.get("interpolation_model", "identity"))
        ratios.append(ratio)
    out["min_rotated_plane_spectral_power_ratio"] = float(min(ratios))
    return out


def _transverse_sampling(family: str) -> np.ndarray:
    wide_tokens = (
        "lateral_decentre",
        "lens1_decentre",
        "lens2_decentre",
        "lens1_tilt",
        "lens2_tilt",
    )
    if any(token in family for token in wide_tokens):
        return np.linspace(-0.65e-3, 0.65e-3, 401)
    return np.linspace(-0.22e-3, 0.22e-3, 321)


def _lab_profile_sampling(family: str) -> np.ndarray:
    if any(token in family for token in ("decentre", "tilt", "offset")):
        return np.linspace(-1.2e-3, 1.2e-3, 501)
    return np.linspace(-0.55e-3, 0.55e-3, 401)


def _plot_linear_profiles(
    *,
    records: list[dict[str, Any]],
    nominal_peak: float,
    case_id: str,
    family: str,
    units: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    for rec in records:
        profile = rec["profile"]
        label = f"{rec['value']:g} {units}" if units else f"{rec['value']:g}"
        axes[0, 0].plot(
            profile.lab_coordinate_m * 1e6,
            profile.lab_x_intensity / max(nominal_peak, EPS),
            label=label,
        )
        axes[0, 1].plot(
            profile.lab_coordinate_m * 1e6,
            profile.lab_y_intensity / max(nominal_peak, EPS),
            label=label,
        )
        axes[1, 0].plot(
            profile.relative_coordinate_m * 1e6,
            profile.axis_x_intensity / max(nominal_peak, EPS),
            label=label,
        )
        axes[1, 1].plot(
            profile.relative_coordinate_m * 1e6,
            profile.axis_y_intensity / max(nominal_peak, EPS),
            label=label,
        )
    axes[0, 0].set(title="Laboratory I(x), y=0", xlabel="x (µm)", ylabel="I / nominal 2-D peak")
    axes[0, 1].set(title="Laboratory I(y), x=0", xlabel="y (µm)", ylabel="I / nominal 2-D peak")
    axes[1, 0].set(title="Morphology-axis I(Δx)", xlabel="Δx from detected axis (µm)", ylabel="I / nominal 2-D peak")
    axes[1, 1].set(title="Morphology-axis I(Δy)", xlabel="Δy from detected axis (µm)", ylabel="I / nominal 2-D peak")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{case_id} — {family} | common physical intensity scale")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_shape_profiles(
    *,
    records: list[dict[str, Any]],
    case_id: str,
    family: str,
    units: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for rec in records:
        profile = rec["profile"]
        label = f"{rec['value']:g} {units}" if units else f"{rec['value']:g}"
        xshape = profile.axis_x_intensity / max(float(np.max(profile.axis_x_intensity)), EPS)
        yshape = profile.axis_y_intensity / max(float(np.max(profile.axis_y_intensity)), EPS)
        axes[0].plot(profile.relative_coordinate_m * 1e6, xshape, label=label)
        axes[1].plot(profile.relative_coordinate_m * 1e6, yshape, label=label)
    axes[0].set(title="I(Δx) shape only", xlabel="Δx (µm)", ylabel="own-peak normalised intensity")
    axes[1].set(title="I(Δy) shape only", xlabel="Δy (µm)", ylabel="own-peak normalised intensity")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{case_id} — {family} | shape-only supplementary view")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_axial_profiles(
    *,
    records: list[dict[str, Any]],
    nominal_record: dict[str, Any],
    z: np.ndarray,
    case_id: str,
    family: str,
    units: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    nominal_x = _line_metrics(nominal_record["xz"], z_m=z, coordinate_m=nominal_record["relative_coordinate"])
    nominal_y = _line_metrics(nominal_record["yz"], z_m=z, coordinate_m=nominal_record["relative_coordinate"])
    x_peak0 = max(float(np.max(nominal_x["peak_trace"])), EPS)
    y_peak0 = max(float(np.max(nominal_y["peak_trace"])), EPS)
    x_int0 = max(float(np.max(nominal_x["line_integral_trace"])), EPS)
    y_int0 = max(float(np.max(nominal_y["line_integral_trace"])), EPS)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    for rec in records:
        label = f"{rec['value']:g} {units}" if units else f"{rec['value']:g}"
        xm = _line_metrics(rec["xz"], z_m=z, coordinate_m=rec["relative_coordinate"])
        ym = _line_metrics(rec["yz"], z_m=z, coordinate_m=rec["relative_coordinate"])
        axes[0, 0].plot(z * 1e3, xm["peak_trace"] / x_peak0, label=label)
        axes[0, 1].plot(z * 1e3, ym["peak_trace"] / y_peak0, label=label)
        axes[1, 0].plot(z * 1e3, xm["line_integral_trace"] / x_int0, label=label)
        axes[1, 1].plot(z * 1e3, ym["line_integral_trace"] / y_int0, label=label)
    axes[0, 0].set(title="Fixed x–z plane peak", ylabel="peak / nominal axial max")
    axes[0, 1].set(title="Fixed y–z plane peak", ylabel="peak / nominal axial max")
    axes[1, 0].set(title="Fixed x–z line-integrated intensity", ylabel="line integral / nominal axial max")
    axes[1, 1].set(title="Fixed y–z line-integrated intensity", ylabel="line integral / nominal axial max")
    for ax in axes.ravel():
        ax.set_xlabel("z from axicon (mm)")
        ax.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{case_id} — {family} | fixed physical propagation planes")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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

    if family.startswith("axicon_"):
        raise RuntimeError(
            "axicon error families are superseded in this runner by the validated "
            "axicon-physics-v3 route; rigid tilt remains blocked pending real refractive geometry"
        )

    registry = system_sweep_registry()
    spec = registry[family]
    values = tuple(spec["values"])
    nominal_value = _nominal_value(values)
    nominal_config = spec["builder"](nominal_value)
    nominal_route = build_system_route(case_id, grid_n=grid_n, config=nominal_config)

    records: list[dict[str, Any]] = []
    z = np.arange(5e-3, 140e-3 + 2e-3, 2e-3)
    relative_coordinate = _transverse_sampling(family)
    lab_coordinate = _lab_profile_sampling(family)
    ell = _ell(case_id)

    for value in values:
        config = spec["builder"](value)
        route = build_system_route(case_id, grid_n=grid_n, config=config)
        grid = dict(route["grid"])
        wavelength = float(route["metadata"]["wavelength_m"])
        propagator = build_fixed_support_spectrum(
            route["post_axicon"],
            grid,
            wavelength_m=wavelength,
            z_max_m=float(np.max(np.abs(z))),
            n_medium=1.0,
            minimum_retained_spectral_power=0.995,
        )
        propagated = native_field_at_z(propagator, z_reference_m)
        Ixy = np.abs(propagated) ** 2
        profile = build_transverse_profile_evidence(
            propagated,
            grid,
            vortex_charge=ell,
            lab_coordinate_m=lab_coordinate,
            relative_coordinate_m=relative_coordinate,
            axis_search_radius_m=1.4e-3,
        )
        axis = profile.morphology_axis
        fixed_map = build_fixed_plane_longitudinal_map(
            propagator,
            z_values_m=z,
            x_coordinates_m=axis.x_m + relative_coordinate,
            y_coordinates_m=axis.y_m + relative_coordinate,
            fixed_x_m=axis.x_m,
            fixed_y_m=axis.y_m,
            source_label=f"{case_id}:{family}={value}",
        )
        component, component_label, component_kind = _family_component_image(family, route, nominal_route)
        records.append(
            {
                "value": value,
                "route": route,
                "Ixy": np.asarray(Ixy),
                "xz": np.asarray(fixed_map.xz_intensity),
                "yz": np.asarray(fixed_map.yz_intensity),
                "relative_coordinate": relative_coordinate,
                "profile": profile,
                "propagator": propagator,
                "map_metadata": fixed_map.metadata,
                "component": component,
                "component_label": component_label,
                "component_kind": component_kind,
            }
        )

    nominal_record = next(rec for rec in records if _same_value(rec["value"], nominal_value))
    nominal_xy = np.asarray(nominal_record["Ixy"])
    nominal_xz = np.asarray(nominal_record["xz"])
    nominal_yz = np.asarray(nominal_record["yz"])
    nominal_xy_metrics = _metrics(nominal_xy, dict(nominal_route["grid"]))
    nominal_peak = float(nominal_record["profile"].peak_2d_au)
    nominal_x_line = _line_metrics(nominal_xz, z_m=z, coordinate_m=relative_coordinate)

    rows: list[dict[str, Any]] = []
    long_profile_rows: list[dict[str, Any]] = []
    for rec in records:
        route = rec["route"]
        grid = dict(route["grid"])
        xy_metrics = _metrics(rec["Ixy"], grid)
        xline = _line_metrics(rec["xz"], z_m=z, coordinate_m=relative_coordinate)
        yline = _line_metrics(rec["yz"], z_m=z, coordinate_m=relative_coordinate)
        residual = _normalised_trace(xline["peak_trace"]) - _normalised_trace(nominal_x_line["peak_trace"])
        profile_row = profile_metrics(rec["profile"], nominal_peak_2d_au=nominal_peak)
        continuity_x = adjacent_row_continuity_metrics(rec["xz"])
        continuity_y = adjacent_row_continuity_metrics(rec["yz"])

        rows.append(
            {
                "case_id": case_id,
                "family": family,
                "value": rec["value"],
                "nominal_value": nominal_value,
                "units": spec["units"],
                "fidelity": spec["fidelity"],
                "grid_n": grid_n,
                "z_reference_m": z_reference_m,
                "longitudinal_plane_contract": "fixed physical planes through reference-plane morphology axis",
                "z_dependent_binary_mask": False,
                "per_z_recentering": False,
                "fixed_support_retained_spectral_power_fraction": float(rec["propagator"].retained_spectral_power_fraction),
                "fourf_iris_selected_fraction": route["metadata"]["fourf"]["iris_selected_power_fraction"],
                "axicon_tilt_status": route["metadata"]["axicon_tilt_status"],
                "exact_axicon_kr_m_inv": route["metadata"]["axicon"]["exact_kr_m_inv"],
                **xy_metrics,
                **profile_row,
                "output_power_ratio_to_nominal": float(xy_metrics["power_au"] / max(nominal_xy_metrics["power_au"], EPS)),
                "xy_corr_nominal": _correlation(rec["Ixy"], nominal_xy),
                "xz_corr_nominal": _correlation(rec["xz"], nominal_xz),
                "yz_corr_nominal": _correlation(rec["yz"], nominal_yz),
                "xz_center_slope_rad_approx": xline["centre_slope_rad_approx"],
                "yz_center_slope_rad_approx": yline["centre_slope_rad_approx"],
                "xz_center_span_m": xline["centre_span_m"],
                "yz_center_span_m": yline["centre_span_m"],
                "xz_mean_width_m": xline["mean_width_m"],
                "yz_mean_width_m": yline["mean_width_m"],
                "xz_active_length_m": xline["active_length_m"],
                "yz_active_length_m": yline["active_length_m"],
                "axial_peak_corr_nominal": _correlation(xline["peak_trace"], nominal_x_line["peak_trace"]),
                "axial_peak_residual_rms": float(np.sqrt(np.mean(residual * residual))),
                "xz_adjacent_row_change_max_over_median": continuity_x["adjacent_row_rms_change_max_over_median"],
                "yz_adjacent_row_change_max_over_median": continuity_y["adjacent_row_rms_change_max_over_median"],
                **_rotated_plane_diagnostics(route),
            }
        )
        long_profile_rows.extend(
            profile_long_rows(
                rec["profile"],
                case_id=case_id,
                family=family,
                sweep_value=float(rec["value"]),
                nominal_peak_2d_au=nominal_peak,
            )
        )

    family_output = output_root / case_id
    _write_csv(family_output / f"{family}_linear_profiles.csv", long_profile_rows)
    np.savez_compressed(
        family_output / f"{family}_raw_propagation_profiles.npz",
        sweep_values=np.asarray([float(rec["value"]) for rec in records], dtype=float),
        z_m=z,
        relative_coordinate_m=relative_coordinate,
        lab_coordinate_m=lab_coordinate,
        xz=np.asarray([rec["xz"] for rec in records], dtype=np.float32),
        yz=np.asarray([rec["yz"] for rec in records], dtype=np.float32),
        lab_x=np.asarray([rec["profile"].lab_x_intensity for rec in records], dtype=np.float32),
        lab_y=np.asarray([rec["profile"].lab_y_intensity for rec in records], dtype=np.float32),
        axis_x=np.asarray([rec["profile"].axis_x_intensity for rec in records], dtype=np.float32),
        axis_y=np.asarray([rec["profile"].axis_y_intensity for rec in records], dtype=np.float32),
        axis_xy_m=np.asarray([[rec["profile"].morphology_axis.x_m, rec["profile"].morphology_axis.y_m] for rec in records], dtype=float),
    )

    x = np.asarray(nominal_route["grid"]["x"], dtype=float)
    xy_common = max(float(np.max(rec["Ixy"])) for rec in records)
    xz_common = max(float(np.max(rec["xz"])) for rec in records)
    yz_common = max(float(np.max(rec["yz"])) for rec in records)
    comp_abs = max(float(np.max(np.abs(rec["component"]))) for rec in records)

    fig, axes = plt.subplots(4, len(records), figsize=(3.1 * len(records), 11.2), constrained_layout=True, squeeze=False)
    for col, rec in enumerate(records):
        value = rec["value"]
        comp = rec["component"]
        if rec["component_kind"] == "intensity":
            half = 3.0e-3 if (family.startswith("slm") or family.startswith("fourf")) else 2.5e-3
            xc, cc = _component_crop(comp, x, half)
            im0 = axes[0, col].imshow(
                cc / max(comp_abs, EPS),
                origin="lower",
                extent=[xc[0] * 1e3, xc[-1] * 1e3, xc[0] * 1e3, xc[-1] * 1e3],
                vmin=0,
                vmax=1,
                cmap="inferno",
            )
        else:
            xc, cc = _component_crop(comp, x, 2.5e-3)
            im0 = axes[0, col].imshow(
                cc,
                origin="lower",
                extent=[xc[0] * 1e3, xc[-1] * 1e3, xc[0] * 1e3, xc[-1] * 1e3],
                vmin=-np.pi,
                vmax=np.pi,
                cmap="twilight",
            )
        axes[0, col].set_xlabel("x (mm)")
        axes[0, col].set_ylabel("y (mm)")
        axes[0, col].set_title(str(value))

        axis = rec["profile"].morphology_axis
        half = float(np.max(np.abs(relative_coordinate)))
        xr, yr, Icrop = _crop_square(
            rec["Ixy"],
            x,
            centre_x_m=axis.x_m,
            centre_y_m=axis.y_m,
            halfwidth=half,
        )
        im1 = axes[1, col].imshow(
            Icrop / max(xy_common, EPS),
            origin="lower",
            extent=[xr[0] * 1e6, xr[-1] * 1e6, yr[0] * 1e6, yr[-1] * 1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[1, col].axhline(0.0, color="white", lw=0.5, alpha=0.45)
        axes[1, col].axvline(0.0, color="white", lw=0.5, alpha=0.45)
        axes[1, col].set_xlabel("Δx from detected axis (µm)")
        axes[1, col].set_ylabel("Δy from detected axis (µm)")

        im2 = axes[2, col].imshow(
            (rec["xz"] / max(xz_common, EPS)).T,
            origin="lower",
            aspect="auto",
            extent=[z[0] * 1e3, z[-1] * 1e3, relative_coordinate[0] * 1e6, relative_coordinate[-1] * 1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[2, col].set_xlabel("z from axicon (mm)")
        axes[2, col].set_ylabel("Δx in fixed axis-crossing plane (µm)")

        im3 = axes[3, col].imshow(
            (rec["yz"] / max(yz_common, EPS)).T,
            origin="lower",
            aspect="auto",
            extent=[z[0] * 1e3, z[-1] * 1e3, relative_coordinate[0] * 1e6, relative_coordinate[-1] * 1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[3, col].set_xlabel("z from axicon (mm)")
        axes[3, col].set_ylabel("Δy in fixed axis-crossing plane (µm)")

    fig.colorbar(im0, ax=axes[0, :].tolist(), label=records[0]["component_label"])
    fig.colorbar(im1, ax=axes[1, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im2, ax=axes[2, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im3, ax=axes[3, :].tolist(), label="I / sweep-global max")
    fig.suptitle(f"{case_id} — {family} | fixed-support, fixed physical propagation planes")
    figure_path = figure_root / case_id
    figure_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path / f"{family}_system_diagnostic_v4.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    units = str(spec.get("units", ""))
    _plot_linear_profiles(
        records=records,
        nominal_peak=nominal_peak,
        case_id=case_id,
        family=family,
        units=units,
        path=figure_path / f"{family}_linear_profiles.png",
    )
    _plot_shape_profiles(
        records=records,
        case_id=case_id,
        family=family,
        units=units,
        path=figure_path / f"{family}_shape_profiles.png",
    )
    _plot_axial_profiles(
        records=records,
        nominal_record=nominal_record,
        z=z,
        case_id=case_id,
        family=family,
        units=units,
        path=figure_path / f"{family}_axial_profiles.png",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run physically placed vortex/Bessel system-error sweeps.")
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/vortex_system_errors"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/figures/vortex_system_errors"))
    args = parser.parse_args()

    registry = system_sweep_registry()
    families = list(registry) if args.families == ["all"] else list(args.families)
    unknown = [name for name in families if name not in registry]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}")
    superseded_axicon = [name for name in families if name.startswith("axicon_")]
    if superseded_axicon:
        raise SystemExit(
            "axicon families are superseded by axicon-physics-v3 and are not authorised in this runner: "
            + ", ".join(superseded_axicon)
        )

    rows: list[dict[str, Any]] = []
    for case_id in args.cases:
        for family in families:
            print(f"running {case_id} / {family}", flush=True)
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

    _write_csv(args.output_root / "system_error_metrics.csv", rows)
    manifest = {
        "outcome": "VORTEX-SYSTEM-ERROR-PROPAGATION-AUDIT-V4",
        "report_figures_authorised": False,
        "cases": list(args.cases),
        "families": families,
        "grid_n": int(args.grid_n),
        "z_reference_mm": float(args.z_mm),
        "diagnostic_contract": (
            "component plane + morphology-centred xy + fixed physical xz/yz + common-scale laboratory "
            "and morphology-axis line profiles + shape-only supplementary profiles + axial line-peak/"
            "line-integral traces; one fixed Matsushima support is applied for the complete z sweep"
        ),
        "line_profile_contract": {
            "laboratory": "I(x,y=0), I(x=0,y) retains steering/decentre",
            "morphology_axis": "I(x,y=y_axis), I(x=x_axis,y) expressed as delta coordinate",
            "primary_normalisation": "all sweep curves / nominal case 2-D reference-plane peak",
            "shape_only_normalisation": "each morphology curve / its own peak; supplementary only",
            "raw_data": "long-form CSV plus compressed NPZ native/spectral line arrays",
        },
        "propagation_contract": {
            "spectral_support": "single Matsushima mask at maximum |z|, applied once",
            "z_dependent_binary_mask": False,
            "longitudinal_coordinates": "fixed physical planes through reference-plane morphology axis",
            "per_z_recentering": False,
            "minimum_retained_source_spectral_power": 0.995,
        },
        "executed_registry": {
            name: {k: v for k, v in registry[name].items() if k != "builder"}
            for name in families
        },
        "blocked_or_data_driven": blocked_or_data_driven_families(),
        "axicon_policy": "all axicon families use the separately validated axicon-physics-v3 route; old thin rigid-tilt surrogate remains rejected",
        "policy": (
            "Every executable family is a controlled relative-sensitivity study. Absolute bench-error "
            "magnitudes require calibration data and independent physical validation."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "system_error_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
