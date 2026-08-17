from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_profile_evidence import (
    build_transverse_profile_evidence,
    profile_long_rows,
    profile_metrics,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.digital_twin.vortex_wavefront_errors import (
    ZERNIKE_NAMES,
    opd_to_phase_rad,
    zernike_opd_map_m,
)
from vbb_study.equations.fields import make_xy_grid


EPS = np.finfo(float).tiny


def _ell(case_id: str) -> int:
    return {"B0": 0, "V1": 1, "V3": 3}[case_id]


def _metrics(I: np.ndarray, grid: dict[str, Any]) -> dict[str, float]:
    arr = np.asarray(I, dtype=float)
    total = float(np.sum(arr))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return {
        "peak_au": float(np.max(arr)),
        "power_au": total * float(grid["dx"]) ** 2,
        "centroid_x_m": float(np.sum(arr * X) / max(total, EPS)),
        "centroid_y_m": float(np.sum(arr * Y) / max(total, EPS)),
    }


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
        raise RuntimeError("aberration morphology crop is under-sampled")
    return (
        x[ix] - float(axis_x_m),
        x[iy] - float(axis_y_m),
        np.asarray(intensity)[np.ix_(iy, ix)],
    )


def _plot_group(
    *,
    case_id: str,
    plane: str,
    mode: str,
    records: list[dict[str, Any]],
    nominal_peak: float,
    figure_root: Path,
) -> None:
    import matplotlib.pyplot as plt

    peak_common = max(float(np.max(rec["intensity"])) for rec in records)
    halfwidth = 250e-6 if _ell(case_id) >= 3 else 190e-6
    fig, axes = plt.subplots(1, len(records), figsize=(3.05 * len(records), 3.15), constrained_layout=True)
    for ax, rec in zip(np.atleast_1d(axes), records):
        profile = rec["profile"]
        xr, yr, crop = _crop_about_axis(
            rec["intensity"],
            rec["grid"],
            axis_x_m=profile.morphology_axis.x_m,
            axis_y_m=profile.morphology_axis.y_m,
            halfwidth_m=halfwidth,
        )
        image = ax.imshow(
            crop / max(peak_common, EPS),
            origin="lower",
            extent=[xr[0] * 1e6, xr[-1] * 1e6, yr[0] * 1e6, yr[-1] * 1e6],
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        ax.axhline(0.0, color="white", lw=0.45, alpha=0.4)
        ax.axvline(0.0, color="white", lw=0.45, alpha=0.4)
        ax.set_title(f"{rec['waves_rms']:+.2f} waves RMS")
        ax.set_xlabel("Δx (µm)")
        ax.set_ylabel("Δy (µm)")
    fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), label="I / sweep-global max")
    fig.suptitle(f"{case_id} — {plane} plane — {mode} | declared OPD sensitivity")
    target = figure_root / case_id / plane
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{mode}_xy_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    for rec in records:
        p = rec["profile"]
        label = f"{rec['waves_rms']:+.2f} waves"
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
    fig.suptitle(f"{case_id} — {plane} plane — {mode} | common-scale line profiles")
    fig.savefig(target / f"{mode}_linear_profiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run declared-plane Zernike OPD sensitivity studies.")
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--planes", nargs="+", choices=["input", "lens1", "lens2"], default=["input", "lens1", "lens2"])
    parser.add_argument("--modes", nargs="+", choices=list(ZERNIKE_NAMES), default=list(ZERNIKE_NAMES))
    parser.add_argument("--waves", nargs="+", type=float, default=[-0.5, -0.25, 0.0, 0.25, 0.5])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--pupil-radius-mm", type=float, default=2.0)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/vortex_declared_aberrations"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/figures/vortex_declared_aberrations"))
    args = parser.parse_args()

    N = int(args.grid_n)
    window = 10e-3
    grid_for_map = make_xy_grid(N, window / N)
    wavelength = 1029e-9
    pupil = float(args.pupil_radius_mm) * 1e-3
    z_ref = float(args.z_mm) * 1e-3
    lab_coordinate = np.linspace(-0.8e-3, 0.8e-3, 401)
    relative_coordinate = np.linspace(-0.30e-3, 0.30e-3, 321)
    rows: list[dict[str, Any]] = []

    for case_id in args.cases:
        for plane in args.planes:
            for mode in args.modes:
                group: list[dict[str, Any]] = []
                for waves in args.waves:
                    opd = zernike_opd_map_m(
                        mode,
                        grid_for_map,
                        wavelength_m=wavelength,
                        waves_rms=float(waves),
                        pupil_radius_m=pupil,
                    )
                    kwargs: dict[str, Any] = {}
                    if plane == "input":
                        kwargs["slm1_static_phase_map_rad"] = opd_to_phase_rad(opd, wavelength)
                    elif plane == "lens1":
                        kwargs["lens1_opd_map_m"] = opd
                    else:
                        kwargs["lens2_opd_map_m"] = opd

                    route = build_system_route(case_id, grid_n=N, **kwargs)
                    propagator = build_fixed_support_spectrum(
                        route["post_axicon"],
                        dict(route["grid"]),
                        wavelength_m=wavelength,
                        z_max_m=z_ref,
                        n_medium=1.0,
                        minimum_retained_spectral_power=0.995,
                    )
                    field = native_field_at_z(propagator, z_ref)
                    intensity = np.abs(field) ** 2
                    profile = build_transverse_profile_evidence(
                        field,
                        route["grid"],
                        vortex_charge=_ell(case_id),
                        lab_coordinate_m=lab_coordinate,
                        relative_coordinate_m=relative_coordinate,
                        axis_search_radius_m=1.2e-3,
                    )
                    group.append(
                        {
                            "waves_rms": float(waves),
                            "route": route,
                            "grid": dict(route["grid"]),
                            "field": field,
                            "intensity": intensity,
                            "profile": profile,
                            "retained_spectral_power_fraction": propagator.retained_spectral_power_fraction,
                        }
                    )

                nominal = min(group, key=lambda rec: abs(float(rec["waves_rms"])))
                nominal_peak = float(nominal["profile"].peak_2d_au)
                profile_rows: list[dict[str, Any]] = []
                for rec in group:
                    route = rec["route"]
                    row = {
                        "case_id": case_id,
                        "application_plane": plane,
                        "mode": mode,
                        "waves_rms": float(rec["waves_rms"]),
                        "pupil_radius_m": pupil,
                        "grid_n": N,
                        "z_reference_m": z_ref,
                        "propagation_support_model": "single Matsushima mask at z_reference applied once",
                        "fixed_support_retained_spectral_power_fraction": float(rec["retained_spectral_power_fraction"]),
                        "fourf_selected_fraction": route["metadata"]["fourf"]["iris_selected_power_fraction"],
                        **_metrics(rec["intensity"], route["grid"]),
                        **profile_metrics(rec["profile"], nominal_peak_2d_au=nominal_peak),
                    }
                    rows.append(row)
                    profile_rows.extend(
                        profile_long_rows(
                            rec["profile"],
                            case_id=case_id,
                            family=f"{plane}:{mode}",
                            sweep_value=float(rec["waves_rms"]),
                            nominal_peak_2d_au=nominal_peak,
                        )
                    )

                target = args.output_root / case_id / plane
                _write_csv(target / f"{mode}_linear_profiles.csv", profile_rows)
                np.savez_compressed(
                    target / f"{mode}_raw_profiles.npz",
                    waves_rms=np.asarray([rec["waves_rms"] for rec in group], dtype=float),
                    lab_coordinate_m=lab_coordinate,
                    relative_coordinate_m=relative_coordinate,
                    lab_x=np.asarray([rec["profile"].lab_x_intensity for rec in group], dtype=np.float32),
                    lab_y=np.asarray([rec["profile"].lab_y_intensity for rec in group], dtype=np.float32),
                    axis_x=np.asarray([rec["profile"].axis_x_intensity for rec in group], dtype=np.float32),
                    axis_y=np.asarray([rec["profile"].axis_y_intensity for rec in group], dtype=np.float32),
                    axis_xy_m=np.asarray([[rec["profile"].morphology_axis.x_m, rec["profile"].morphology_axis.y_m] for rec in group], dtype=float),
                )
                _plot_group(
                    case_id=case_id,
                    plane=plane,
                    mode=mode,
                    records=group,
                    nominal_peak=nominal_peak,
                    figure_root=args.figure_root,
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "declared_aberration_metrics.csv", rows)
    manifest = {
        "outcome": "VORTEX-DECLARED-ABERRATION-SENSITIVITY-WITH-PROFILES",
        "report_figures_authorised": False,
        "planes": list(args.planes),
        "modes": list(args.modes),
        "waves_rms": list(args.waves),
        "pupil_radius_mm": float(args.pupil_radius_mm),
        "line_profile_contract": (
            "laboratory and morphology-axis x/y complex-field line samples; all primary curves in a "
            "plane/mode sweep share the nominal 2-D peak normalisation"
        ),
        "policy": (
            "These are declared OPD sensitivities at named planes. They are not physical substitutes "
            "for lens/axicon/beam misalignment. Absolute values require measured wavefront or lens OPD data."
        ),
        "input_plane_note": (
            "The input phase screen is injected through the co-located SLM1 static-phase hook; phase factors commute at that plane."
        ),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
