"""Research-backed figures for physical vortex-Bessel error studies.

This module intentionally does not reuse the old generic sweep layout for input
beam angle.  Input pointing is an upstream error, so the figure exposes the
SLM/grating Fourier plane and fixed +1-order iris before showing the propagated
Bessel field.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.phase2e_spectral_propagation import build_dense_spectral_propagation
from vbb_study.digital_twin.vortex_error_reference_models import fourier_order_diagnostics
from vbb_study.digital_twin.vortex_physical_errors import (
    PhysicalPerturbation,
    build_physical_route_checkpoints,
)
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny
DEFAULT_ROOT = Path("outputs/figures/vortex_error_physics_rebuild")
DEFAULT_VALIDATION_ROOT = Path("outputs/validation/vortex_error_physics_rebuild")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _crop_xy(intensity: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float):
    x = np.asarray(grid["x"], dtype=float)
    keep = np.abs(x) <= float(halfwidth_m)
    return x[keep], np.asarray(intensity)[np.ix_(keep, keep)]


def _crop_spectrum(intensity: np.ndarray, grid: Mapping[str, Any], *, fx_center: float, fx_half: float, fy_half: float):
    fx_axis = np.asarray(grid["FX"], dtype=float)[0, :]
    fy_axis = np.asarray(grid["FY"], dtype=float)[:, 0]
    ix = np.abs(fx_axis - float(fx_center)) <= float(fx_half)
    iy = np.abs(fy_axis) <= float(fy_half)
    return fx_axis[ix], fy_axis[iy], np.asarray(intensity)[np.ix_(iy, ix)]


def _normalise(array: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(array, dtype=float) / max(float(scale), EPS)


def _save(fig, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [str(png), str(pdf)]


def render_input_beam_angle_study(
    case_id: str = "B0",
    *,
    grid_n: int = 1536,
    angles_rad: Sequence[float] = (-1e-3, -0.5e-3, 0.0, 0.5e-3, 1e-3),
    z_reference_m: float = 60e-3,
    z_values_m: Sequence[float] | None = None,
    xy_halfwidth_m: float = 0.16e-3,
    xz_halfwidth_m: float = 0.16e-3,
    figure_root: Path = DEFAULT_ROOT,
    validation_root: Path = DEFAULT_VALIDATION_ROOT,
) -> dict[str, Any]:
    """Render an input-pointing study with explicit SLM/4F diagnostics.

    Model scope:
    * incident pointing is a plane-wave direction cosine before SLM1;
    * the programmed carrier remains fixed on the SLM;
    * the +1-order iris remains fixed in the Fourier plane;
    * the propagated source uses the scalar thin-axicon route;
    * angle-dependent LCOS electro-optic phase response is not calibrated.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if z_values_m is None:
        z_values_m = np.arange(5e-3, 140e-3 + 1e-12, 2e-3)
    z = np.asarray(z_values_m, dtype=float)
    angles = tuple(float(v) for v in angles_rad)
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    iris_radius = float(hardware_value(manifest, "fourier_iris_radius_cpm"))
    focal_length = float(hardware_value(manifest, "fourf_focal_length_m"))

    transverse_coordinates = np.linspace(-xz_halfwidth_m, xz_halfwidth_m, 321)
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    for angle in angles:
        checkpoints = build_physical_route_checkpoints(
            case_id,
            grid_n=int(grid_n),
            perturbation=PhysicalPerturbation(input_beam_angle_rad=(angle, 0.0)),
        )
        grid = checkpoints["grid"]
        meta = dict(checkpoints["metadata"])
        diagnostic = fourier_order_diagnostics(
            checkpoints["post_slm"],
            grid,
            wavelength_m=wavelength,
            carrier_cpm=carrier,
            iris_radius_cpm=iris_radius,
            focal_length_m=focal_length,
            input_angle_x_rad=angle,
            input_angle_y_rad=0.0,
            order=1,
        )
        propagated = angular_spectrum_propagate_bl(
            checkpoints["post_axicon"],
            dict(grid),
            wavelength,
            float(z_reference_m),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        plane = np.asarray(np.abs(propagated) ** 2, dtype=np.float64)
        dense = build_dense_spectral_propagation(
            grid=grid,
            wavelength_m=wavelength,
            z_values_m=z,
            transverse_coordinates_m=transverse_coordinates,
            scalar_field=checkpoints["post_axicon"],
            source_label=f"{case_id} input pointing {angle:+.6e} rad research diagnostic",
        )
        records.append(
            {
                "angle": angle,
                "grid": grid,
                "meta": meta,
                "diagnostic": diagnostic,
                "plane": plane,
                "dense": dense,
            }
        )
        metrics.append(
            {
                "case_id": case_id,
                "grid_n": int(grid_n),
                "input_angle_x_rad": float(angle),
                "input_angle_x_mrad": float(angle * 1e3),
                "expected_plus1_fx_cpm": float(diagnostic["expected_fx_cpm"]),
                "expected_plus1_fourier_x_mm": float(diagnostic["expected_fourier_x_m"] * 1e3),
                "plus1_shift_from_nominal_mm": float(diagnostic["fourier_shift_x_m"] * 1e3),
                "order_offset_over_iris_radius": float(diagnostic["order_offset_over_iris_radius"]),
                "route_first_order_efficiency": float(meta["first_order_efficiency"]),
                "fixed_iris_selected_spectral_fraction": float(diagnostic["fixed_iris_selected_spectral_fraction"]),
                "expected_center_selected_spectral_fraction": float(diagnostic["expected_center_selected_spectral_fraction"]),
                "measured_local_centroid_fx_cpm": float(diagnostic["measured_local_centroid_fx_cpm"]),
                "centroid_error_cpm": float(diagnostic["centroid_error_cpm"]),
            }
        )

    xy_scale = max(float(np.max(r["plane"])) for r in records)
    xz_scale = max(float(np.max(r["dense"].xz_intensity)) for r in records)
    spectrum_scale = max(float(np.max(r["diagnostic"]["spectrum_intensity"])) for r in records)
    nominal_index = int(np.argmin(np.abs(np.asarray(angles))))
    nominal_plane = np.asarray(records[nominal_index]["plane"], dtype=float)
    difference_scale = EPS
    for record in records:
        _, current_crop = _crop_xy(record["plane"], record["grid"], xy_halfwidth_m)
        _, nominal_crop = _crop_xy(nominal_plane, record["grid"], xy_halfwidth_m)
        difference_scale = max(difference_scale, float(np.max(np.abs(current_crop - nominal_crop))))

    fig, axes = plt.subplots(
        4,
        len(records),
        figsize=(3.25 * len(records), 11.2),
        constrained_layout=True,
        squeeze=False,
    )

    for col, record in enumerate(records):
        angle = float(record["angle"])
        grid = record["grid"]
        diag = record["diagnostic"]
        title = f"{angle*1e3:+.2f} mrad"

        fx, fy, spectrum_crop = _crop_spectrum(
            diag["spectrum_intensity"],
            grid,
            fx_center=carrier,
            fx_half=4500.0,
            fy_half=3500.0,
        )
        shown_spec = np.log10(1.0 + 1e5 * _normalise(spectrum_crop, spectrum_scale))
        shown_spec /= max(float(np.max(shown_spec)), EPS)
        im0 = axes[0, col].imshow(
            shown_spec,
            origin="lower",
            aspect="equal",
            extent=[fx[0] / 1e3, fx[-1] / 1e3, fy[0] / 1e3, fy[-1] / 1e3],
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        axes[0, col].add_patch(
            Circle(
                (float(diag["fixed_iris_fx_cpm"]) / 1e3, 0.0),
                iris_radius / 1e3,
                fill=False,
                linestyle="--",
                linewidth=1.2,
            )
        )
        axes[0, col].plot(
            float(diag["expected_fx_cpm"]) / 1e3,
            float(diag["expected_fy_cpm"]) / 1e3,
            marker="x",
            markersize=7,
        )
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("$f_x$ (lp/mm)")
        axes[0, col].set_ylabel("$f_y$ (lp/mm)")
        axes[0, col].text(
            0.03,
            0.03,
            (
                f"shift={diag['fourier_shift_x_m']*1e3:+.3f} mm\n"
                f"offset={diag['order_offset_over_iris_radius']:.3f} iris R\n"
                f"fixed-iris η={record['meta']['first_order_efficiency']:.3f}"
            ),
            transform=axes[0, col].transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )

        x, xy_crop = _crop_xy(record["plane"], grid, xy_halfwidth_m)
        im1 = axes[1, col].imshow(
            _normalise(xy_crop, xy_scale),
            origin="lower",
            extent=[x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6],
            vmin=0.0,
            vmax=1.0,
            cmap="inferno",
        )
        axes[1, col].set_xlabel("x (µm)")
        axes[1, col].set_ylabel(f"y (µm)\nxy @ {z_reference_m*1e3:.0f} mm")

        im2 = axes[2, col].imshow(
            _normalise(record["dense"].xz_intensity, xz_scale).T,
            origin="lower",
            aspect="auto",
            extent=[z[0]*1e3, z[-1]*1e3, transverse_coordinates[0]*1e6, transverse_coordinates[-1]*1e6],
            vmin=0.0,
            vmax=1.0,
            cmap="inferno",
        )
        axes[2, col].set_xlabel("z from axicon (mm)")
        axes[2, col].set_ylabel("x (µm)")

        _, nominal_crop = _crop_xy(nominal_plane, grid, xy_halfwidth_m)
        diff = (xy_crop - nominal_crop) / difference_scale
        im3 = axes[3, col].imshow(
            diff,
            origin="lower",
            extent=[x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6],
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        axes[3, col].set_xlabel("x (µm)")
        axes[3, col].set_ylabel("y (µm)")

    fig.colorbar(im0, ax=axes[0, :].tolist(), label="log display of SLM Fourier intensity")
    fig.colorbar(im1, ax=axes[1, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im2, ax=axes[2, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im3, ax=axes[3, :].tolist(), label="(I - nominal) / max |difference|")
    fig.suptitle(
        (
            f"{case_id} — input pointing research diagnostic\n"
            "finite-angle input before SLM1 + fixed carrier/4F iris + scalar thin-axicon propagation; "
            "LCOS angle-dependent electro-optic response not calibrated"
        )
    )

    output_base = figure_root / f"{case_id.lower()}_input_beam_angle_research_diagnostic"
    files = _save(fig, output_base)
    plt.close(fig)

    metrics_path = validation_root / f"{case_id.lower()}_input_beam_angle_fourier_diagnostics.csv"
    _write_csv(metrics_path, metrics)
    outcome = {
        "outcome": "VORTEX-INPUT-ANGLE-REFERENCE-DIAGNOSTIC",
        "report_figures_authorised": False,
        "case_id": case_id,
        "grid_n": int(grid_n),
        "angles_mrad": [float(v * 1e3) for v in angles],
        "model_scope": (
            "unfolded scalar reflective-SLM geometry: finite-angle input before SLM1, "
            "fixed carrier and fixed Fourier iris, scalar thin-axicon diffraction"
        ),
        "required_before_report_authorisation": [
            "numerical +1-order centroid agrees with analytic grating-order translation",
            "fixed-iris efficiency response reviewed",
            "small-angle axicon result benchmarked to Zhao/Li thin-axicon theory",
            "moderate-angle standalone axicon benchmark reproduces known oblique astigmatism/caustic trend",
            "actual bench SLM incidence geometry and phase LUT bound",
        ],
        "files": files,
        "metrics_csv": str(metrics_path),
    }
    validation_root.mkdir(parents=True, exist_ok=True)
    (validation_root / f"{case_id.lower()}_input_beam_angle_outcome.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )
    return outcome
