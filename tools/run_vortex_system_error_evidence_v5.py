from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.vortex_continuous_propagation import adjacent_row_continuity_metrics
from vbb_study.digital_twin.vortex_fixed_plane_bl_propagation import (
    build_bandlimited_fixed_plane_longitudinal_map,
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
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny


def _ell(case_id: str) -> int:
    return {"B0": 0, "V1": 1, "V3": 3}[case_id]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nominal_value(values: tuple[Any, ...]) -> Any:
    finite = [v for v in values if np.isfinite(float(v))]
    if any(np.isclose(float(v), 1.0) for v in finite):
        return min(finite, key=lambda value: abs(float(value) - 1.0))
    nonfinite = [v for v in values if not np.isfinite(float(v))]
    if nonfinite:
        return nonfinite[0]
    return min(values, key=lambda value: abs(float(value)))


def _same_value(a: Any, b: Any) -> bool:
    aa, bb = float(a), float(b)
    if not np.isfinite(aa) or not np.isfinite(bb):
        return (not np.isfinite(aa)) and (not np.isfinite(bb))
    return bool(np.isclose(aa, bb, rtol=0.0, atol=1e-15))


def _xy_metrics(intensity: np.ndarray, grid: dict[str, Any]) -> dict[str, float]:
    arr = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    total = float(np.sum(arr))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    cx = float(np.sum(arr * X) / max(total, EPS))
    cy = float(np.sum(arr * Y) / max(total, EPS))
    radius2 = float(np.sum(arr * ((X - cx) ** 2 + (Y - cy) ** 2)) / max(total, EPS))
    return {
        "peak_au": float(np.max(arr)),
        "power_au_m2": float(total * float(grid["dx"]) ** 2),
        "energy_centroid_x_m": cx,
        "energy_centroid_y_m": cy,
        "rms_radius_about_energy_centroid_m": float(np.sqrt(max(radius2, 0.0))),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel().copy()
    bb = np.asarray(b, dtype=float).ravel().copy()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def _transverse_coordinate(family: str) -> np.ndarray:
    if any(token in family for token in ("decentre", "tilt", "offset")):
        return np.linspace(-0.75e-3, 0.75e-3, 451)
    return np.linspace(-0.28e-3, 0.28e-3, 361)


def _lab_profile_coordinate(family: str) -> np.ndarray:
    if any(token in family for token in ("decentre", "tilt", "offset")):
        return np.linspace(-1.3e-3, 1.3e-3, 551)
    return np.linspace(-0.65e-3, 0.65e-3, 451)


def _crop_about_axis(
    intensity: np.ndarray,
    grid: dict[str, Any],
    *,
    axis_x_m: float,
    axis_y_m: float,
    halfwidth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(grid["x"], dtype=float)
    ix = np.flatnonzero(np.abs(x - float(axis_x_m)) <= float(halfwidth_m))
    iy = np.flatnonzero(np.abs(x - float(axis_y_m)) <= float(halfwidth_m))
    if ix.size < 16 or iy.size < 16:
        raise RuntimeError("transverse morphology ROI has too few native samples")
    return (
        x[ix] - float(axis_x_m),
        x[iy] - float(axis_y_m),
        np.asarray(intensity)[np.ix_(iy, ix)],
    )


def _line_metrics(intensity: np.ndarray, coordinate_m: np.ndarray) -> dict[str, np.ndarray]:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    coordinate = np.asarray(coordinate_m, dtype=float)
    return {
        "peak": np.max(values, axis=1),
        "integral": np.trapezoid(values, coordinate, axis=1),
    }


def _plot_xy_sweep(records: list[dict[str, Any]], *, case_id: str, family: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    common = max(float(np.max(rec["Ixy"])) for rec in records)
    fig, axes = plt.subplots(1, len(records), figsize=(3.15 * len(records), 3.25), constrained_layout=True)
    image = None
    for ax, rec in zip(np.atleast_1d(axes), records):
        axis = rec["profile"].morphology_axis
        half = 0.26e-3 if _ell(case_id) >= 3 else 0.20e-3
        xr, yr, crop = _crop_about_axis(
            rec["Ixy"],
            rec["grid"],
            axis_x_m=axis.x_m,
            axis_y_m=axis.y_m,
            halfwidth_m=half,
        )
        image = ax.imshow(
            crop / max(common, EPS),
            origin="lower",
            extent=[xr[0] * 1e6, xr[-1] * 1e6, yr[0] * 1e6, yr[-1] * 1e6],
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        ax.axhline(0.0, color="white", lw=0.45, alpha=0.45)
        ax.axvline(0.0, color="white", lw=0.45, alpha=0.45)
        ax.set_title(f"{float(rec['value']):g}")
        ax.set_xlabel("Δx (µm)")
        ax.set_ylabel("Δy (µm)")
    if image is not None:
        fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), label="I / sweep-global maximum")
    fig.suptitle(f"{case_id} — {family} | z = 60 mm, common physical intensity scale")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_longitudinal(records: list[dict[str, Any]], *, case_id: str, family: str, z: np.ndarray, coordinate: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    common = max(
        max(float(np.max(rec["xz"])), float(np.max(rec["yz"])))
        for rec in records
    )
    fig, axes = plt.subplots(2, len(records), figsize=(3.1 * len(records), 6.1), constrained_layout=True, squeeze=False)
    image = None
    for col, rec in enumerate(records):
        for row, data, label in (
            (0, rec["xz"], "Δx"),
            (1, rec["yz"], "Δy"),
        ):
            image = axes[row, col].imshow(
                (np.asarray(data) / max(common, EPS)).T,
                origin="lower",
                aspect="auto",
                extent=[z[0] * 1e3, z[-1] * 1e3, coordinate[0] * 1e6, coordinate[-1] * 1e6],
                cmap="inferno",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, col].axhline(0.0, color="white", lw=0.5, alpha=0.45)
            axes[row, col].set_xlabel("z from axicon (mm)")
            axes[row, col].set_ylabel(f"{label} in fixed plane (µm)")
        axes[0, col].set_title(f"{float(rec['value']):g}")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="I / family-global maximum")
    fig.suptitle(
        f"{case_id} — {family} | fixed physical planes, distance-aware BL-ASM"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_linear_profiles(records: list[dict[str, Any]], *, nominal_peak: float, case_id: str, family: str, units: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.1, 7.5), constrained_layout=True)
    for rec in records:
        p = rec["profile"]
        label = f"{float(rec['value']):g} {units}".strip()
        axes[0, 0].plot(p.lab_coordinate_m * 1e6, p.lab_x_intensity / max(nominal_peak, EPS), label=label)
        axes[0, 1].plot(p.lab_coordinate_m * 1e6, p.lab_y_intensity / max(nominal_peak, EPS), label=label)
        axes[1, 0].plot(p.relative_coordinate_m * 1e6, p.axis_x_intensity / max(nominal_peak, EPS), label=label)
        axes[1, 1].plot(p.relative_coordinate_m * 1e6, p.axis_y_intensity / max(nominal_peak, EPS), label=label)
    axes[0, 0].set(title="Laboratory I(x), y=0", xlabel="x (µm)", ylabel="I / nominal 2-D peak")
    axes[0, 1].set(title="Laboratory I(y), x=0", xlabel="y (µm)", ylabel="I / nominal 2-D peak")
    axes[1, 0].set(title="Morphology-axis I(Δx)", xlabel="Δx (µm)", ylabel="I / nominal 2-D peak")
    axes[1, 1].set(title="Morphology-axis I(Δy)", xlabel="Δy (µm)", ylabel="I / nominal 2-D peak")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{case_id} — {family} | common nominal intensity scale")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_axial(records: list[dict[str, Any]], *, nominal: dict[str, Any], case_id: str, family: str, units: str, z: np.ndarray, coordinate: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    nominal_x = _line_metrics(nominal["xz"], coordinate)
    nominal_y = _line_metrics(nominal["yz"], coordinate)
    scales = {
        "xp": max(float(np.max(nominal_x["peak"])), EPS),
        "yp": max(float(np.max(nominal_y["peak"])), EPS),
        "xi": max(float(np.max(nominal_x["integral"])), EPS),
        "yi": max(float(np.max(nominal_y["integral"])), EPS),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    for rec in records:
        xm = _line_metrics(rec["xz"], coordinate)
        ym = _line_metrics(rec["yz"], coordinate)
        label = f"{float(rec['value']):g} {units}".strip()
        axes[0, 0].plot(z * 1e3, xm["peak"] / scales["xp"], label=label)
        axes[0, 1].plot(z * 1e3, ym["peak"] / scales["yp"], label=label)
        axes[1, 0].plot(z * 1e3, xm["integral"] / scales["xi"], label=label)
        axes[1, 1].plot(z * 1e3, ym["integral"] / scales["yi"], label=label)
    axes[0, 0].set_title("x–z fixed-plane peak")
    axes[0, 1].set_title("y–z fixed-plane peak")
    axes[1, 0].set_title("x–z line-integrated intensity")
    axes[1, 1].set_title("y–z line-integrated intensity")
    for ax in axes.ravel():
        ax.set_xlabel("z from axicon (mm)")
        ax.set_ylabel("ratio to nominal axial maximum")
        ax.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{case_id} — {family} | quantitative axial comparison")
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
    if family.startswith("axicon_"):
        raise RuntimeError(
            "axicon families are governed by the separately validated axicon-physics-v3 route; "
            "the generic system-error evidence runner is not authorised for them"
        )
    registry = system_sweep_registry()
    spec = registry[family]
    values = tuple(spec["values"])
    nominal_value = _nominal_value(values)
    z = np.arange(5e-3, 140e-3 + 2e-3, 2e-3)
    relative = _transverse_coordinate(family)
    lab = _lab_profile_coordinate(family)
    records: list[dict[str, Any]] = []

    for value in values:
        route = build_system_route(case_id, grid_n=int(grid_n), config=spec["builder"](value))
        grid = dict(route["grid"])
        wavelength = float(route["metadata"]["wavelength_m"])
        field_ref = angular_spectrum_propagate_bl(
            route["post_axicon"],
            grid,
            wavelength,
            float(z_reference_m),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        Ixy = np.abs(field_ref) ** 2
        profile = build_transverse_profile_evidence(
            field_ref,
            grid,
            vortex_charge=_ell(case_id),
            lab_coordinate_m=lab,
            relative_coordinate_m=relative,
            axis_search_radius_m=1.5e-3,
        )
        axis = profile.morphology_axis
        longitudinal = build_bandlimited_fixed_plane_longitudinal_map(
            route["post_axicon"],
            grid,
            wavelength_m=wavelength,
            z_values_m=z,
            x_coordinates_m=axis.x_m + relative,
            y_coordinates_m=axis.y_m + relative,
            fixed_x_m=axis.x_m,
            fixed_y_m=axis.y_m,
            n_medium=1.0,
            minimum_retained_spectral_power=0.985,
            source_label=f"{case_id}:{family}={value}",
        )
        records.append(
            {
                "value": value,
                "route": route,
                "grid": grid,
                "Ixy": Ixy,
                "profile": profile,
                "xz": longitudinal.xz_intensity,
                "yz": longitudinal.yz_intensity,
                "support_retained": longitudinal.support_retained_spectral_power_fraction,
                "longitudinal_metadata": longitudinal.metadata,
            }
        )

    nominal = next(rec for rec in records if _same_value(rec["value"], nominal_value))
    nominal_peak = float(nominal["profile"].peak_2d_au)
    nominal_xy = np.asarray(nominal["Ixy"])
    nominal_power = float(nominal["profile"].total_2d_power_au_m2)
    nominal_xz = np.asarray(nominal["xz"])
    nominal_yz = np.asarray(nominal["yz"])
    rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for rec in records:
        profile_row = profile_metrics(rec["profile"], nominal_peak_2d_au=nominal_peak)
        continuity_x = adjacent_row_continuity_metrics(rec["xz"])
        continuity_y = adjacent_row_continuity_metrics(rec["yz"])
        metrics = _xy_metrics(rec["Ixy"], rec["grid"])
        rows.append(
            {
                "case_id": case_id,
                "family": family,
                "value": rec["value"],
                "nominal_value": nominal_value,
                "units": spec["units"],
                "fidelity": spec["fidelity"],
                "grid_n": int(grid_n),
                "z_reference_m": float(z_reference_m),
                "propagator": "distance-specific Matsushima band-limited angular spectrum",
                "longitudinal_coordinate_contract": "fixed physical x-z/y-z planes through reference-plane morphology axis",
                "per_z_recentering": False,
                "coordinate_warping": False,
                "minimum_support_retained_spectral_power_fraction": float(np.min(rec["support_retained"])),
                **metrics,
                **profile_row,
                "total_2d_power_ratio_to_nominal": float(rec["profile"].total_2d_power_au_m2 / max(nominal_power, EPS)),
                "xy_correlation_to_nominal": _correlation(rec["Ixy"], nominal_xy),
                "xz_correlation_to_nominal": _correlation(rec["xz"], nominal_xz),
                "yz_correlation_to_nominal": _correlation(rec["yz"], nominal_yz),
                "xz_adjacent_row_change_max_over_median": continuity_x["adjacent_row_rms_change_max_over_median"],
                "yz_adjacent_row_change_max_over_median": continuity_y["adjacent_row_rms_change_max_over_median"],
                "fourf_selected_fraction": rec["route"]["metadata"]["fourf"]["iris_selected_power_fraction"],
            }
        )
        long_rows.extend(
            profile_long_rows(
                rec["profile"],
                case_id=case_id,
                family=family,
                sweep_value=float(rec["value"]),
                nominal_peak_2d_au=nominal_peak,
            )
        )

    data_root = output_root / case_id
    fig_root = figure_root / case_id
    _write_csv(data_root / f"{family}_metrics.csv", rows)
    _write_csv(data_root / f"{family}_linear_profiles.csv", long_rows)
    np.savez_compressed(
        data_root / f"{family}_raw_evidence.npz",
        sweep_values=np.asarray([float(rec["value"]) for rec in records], dtype=float),
        z_m=z,
        relative_coordinate_m=relative,
        lab_coordinate_m=lab,
        xy=np.asarray([rec["Ixy"] for rec in records], dtype=np.float32),
        xz=np.asarray([rec["xz"] for rec in records], dtype=np.float32),
        yz=np.asarray([rec["yz"] for rec in records], dtype=np.float32),
        lab_x=np.asarray([rec["profile"].lab_x_intensity for rec in records], dtype=np.float32),
        lab_y=np.asarray([rec["profile"].lab_y_intensity for rec in records], dtype=np.float32),
        axis_x=np.asarray([rec["profile"].axis_x_intensity for rec in records], dtype=np.float32),
        axis_y=np.asarray([rec["profile"].axis_y_intensity for rec in records], dtype=np.float32),
        axis_xy_m=np.asarray([[rec["profile"].morphology_axis.x_m, rec["profile"].morphology_axis.y_m] for rec in records], dtype=float),
        support_retained=np.asarray([rec["support_retained"] for rec in records], dtype=float),
    )
    units = str(spec.get("units", ""))
    _plot_xy_sweep(records, case_id=case_id, family=family, path=fig_root / f"{family}_xy_sweep.png")
    _plot_longitudinal(records, case_id=case_id, family=family, z=z, coordinate=relative, path=fig_root / f"{family}_xz_yz_fixed_planes.png")
    _plot_linear_profiles(records, nominal_peak=nominal_peak, case_id=case_id, family=family, units=units, path=fig_root / f"{family}_linear_profiles.png")
    _plot_axial(records, nominal=nominal, case_id=case_id, family=family, units=units, z=z, coordinate=relative, path=fig_root / f"{family}_axial_profiles.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited B0/V1/V3 system-error evidence with 2-D and 1-D sweeps.")
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/system_error_evidence_v5"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/figures/system_error_evidence_v5"))
    args = parser.parse_args()

    registry = system_sweep_registry()
    families = list(registry) if args.families == ["all"] else list(args.families)
    unknown = [family for family in families if family not in registry]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}")
    axicon = [family for family in families if family.startswith("axicon_")]
    if axicon:
        raise SystemExit(
            "axicon families are intentionally excluded from v5 and must use axicon-physics-v3: "
            + ", ".join(axicon)
        )

    all_rows: list[dict[str, Any]] = []
    for case_id in args.cases:
        for family in families:
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

    _write_csv(args.output_root / "system_error_evidence_metrics.csv", all_rows)
    manifest = {
        "outcome": "SYSTEM-ERROR-EVIDENCE-V5",
        "report_figures_authorised": False,
        "cases": list(args.cases),
        "families": families,
        "grid_n": int(args.grid_n),
        "reference_plane_mm": float(args.z_mm),
        "transverse_evidence": "common-scale morphology-centred 2-D xy sweep plus lab/morphology complex-field line cuts",
        "longitudinal_evidence": "fixed physical x-z/y-z planes through the reference-plane morphology axis",
        "propagation": "distance-specific Matsushima BL-ASM; no z-dependent coordinate transform",
        "primary_line_normalisation": "nominal case 2-D reference-plane peak",
        "raw_evidence": "per-family CSV plus compressed NPZ arrays",
        "axicon_policy": "use axicon-physics-v3; rigid tilt remains blocked pending real refractive geometry",
        "blocked_or_data_driven": blocked_or_data_driven_families(),
        "claim_policy": "relative sensitivity until calibration; numerical propagation validation does not validate unknown hardware parameters",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
