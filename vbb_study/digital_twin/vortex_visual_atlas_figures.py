"""Figure generation for the physically grounded vortex-Bessel atlas.

Single transverse planes can hide the dominant effect of axicon/input errors.
Each sweep therefore shows both a tight transverse plane and an x-z propagation
map on common physical axes, plus a transverse difference map relative to the
nominal sweep member.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2e_spectral_propagation import build_dense_spectral_propagation
from vbb_study.digital_twin.vortex_visual_atlas import (
    DEFAULT_FIGURE_ROOT,
    DEFAULT_REPORT_N,
    ZERNIKE_REGISTRY,
    _source_kwargs,
    aberration_registry,
    alignment_registry,
    build_atlas_source,
    manufacturing_defect_registry,
    parameter_registry,
    propagate_selected_planes,
)


EPS = np.finfo(float).tiny


def _crop(intensity: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(grid["x"], dtype=float)
    mask = np.abs(x) <= float(halfwidth_m)
    return x[mask], np.asarray(intensity)[np.ix_(mask, mask)]


def _normalise(intensity: np.ndarray, reference: float) -> np.ndarray:
    return np.maximum(np.asarray(intensity, dtype=float), 0.0) / max(float(reference), EPS)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _sweep_values(family: str, parameter: str):
    if family == "parameter":
        return parameter_registry()[parameter]
    if family == "manufacturing":
        return manufacturing_defect_registry()[parameter]
    if family == "aberration":
        return aberration_registry()[parameter]
    if family == "alignment":
        return alignment_registry()[parameter]
    raise ValueError(family)


def _nominal_index(values: Sequence[Any]) -> int:
    numeric = np.asarray([float(v) for v in values], dtype=float)
    return int(np.argmin(np.abs(numeric)))


def _label_value(family: str, parameter: str, value: Any) -> str:
    v = float(value)
    if family == "aberration":
        return f"{v:+.2f} waves RMS"
    if "angle" in parameter and parameter.endswith("_rad"):
        return f"{v*1e3:+.2f} mrad"
    if "tilt" in parameter:
        return f"{np.rad2deg(v):+.2f}°"
    if "decentre" in parameter or "radius_m" in parameter or "rounding_parameter_m" in parameter:
        return f"{v*1e6:+.0f} µm"
    if "iris_offset_fraction" in parameter:
        return f"{v:+.2f} iris radii"
    return f"{value}"


def _family_subfolder(family: str) -> str:
    return {
        "parameter": "02_parameter_sweeps",
        "manufacturing": "03_manufacturing_defects",
        "aberration": "04_generic_wavefront_aberrations",
        "alignment": "05_alignment_errors",
    }[family]


def canonical_plane_grid(
    *,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    grid_n: int = DEFAULT_REPORT_N,
    cases: Sequence[str] = ("B0", "V1", "V3"),
    z_values_m: Sequence[float] = (20e-3, 40e-3, 60e-3, 80e-3, 100e-3),
    halfwidth_m: float = 0.16e-3,
) -> list[str]:
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    for case_id in cases:
        source, grid, meta = build_atlas_source(case_id, grid_n=grid_n)
        planes = propagate_selected_planes(source, grid, float(meta["wavelength_m"]), z_values_m)
        common_scale = max(float(np.max(p)) for p in planes)
        fig, axes = plt.subplots(1, len(planes), figsize=(3.0 * len(planes), 3.0), constrained_layout=True)
        for ax, z_m, plane in zip(np.atleast_1d(axes), z_values_m, planes):
            x, cropped = _crop(plane, grid, halfwidth_m)
            shown = _normalise(cropped, common_scale)
            im = ax.imshow(
                shown,
                origin="lower",
                extent=[x[0] * 1e6, x[-1] * 1e6, x[0] * 1e6, x[-1] * 1e6],
                vmin=0,
                vmax=1,
                cmap="inferno",
            )
            ax.set_title(f"z = {z_m*1e3:.0f} mm")
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
        fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), label="I / case-global max")
        fig.suptitle(f"{case_id} — nominal physical source-scale route")
        path = figure_root / "01_canonical_beams" / f"{case_id.lower()}_selected_xy_planes"
        _save(fig, path)
        plt.close(fig)
        outputs.extend([str(path.with_suffix(".png")), str(path.with_suffix(".pdf"))])
    return outputs


def sweep_transverse_and_propagation_grid(
    case_id: str,
    *,
    family: str,
    parameter: str,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    grid_n: int = DEFAULT_REPORT_N,
    z_reference_m: float = 60e-3,
    xy_halfwidth_m: float = 0.16e-3,
    xz_halfwidth_m: float = 0.16e-3,
    z_values_m: Sequence[float] | None = None,
) -> list[str]:
    """Render transverse, x-z and difference panels for one physical sweep."""

    import matplotlib.pyplot as plt

    values = tuple(_sweep_values(family, parameter))
    if z_values_m is None:
        z_values_m = np.arange(5e-3, 140e-3 + 1e-3, 2e-3)
    z = np.asarray(z_values_m, dtype=float)
    transverse_coordinates = np.linspace(-xz_halfwidth_m, xz_halfwidth_m, 321)

    records: list[dict[str, Any]] = []
    xy_scales: list[float] = []
    xz_scales: list[float] = []
    for value in values:
        source, grid, meta = build_atlas_source(
            case_id,
            grid_n=grid_n,
            **_source_kwargs(family, parameter, value),
        )
        plane = propagate_selected_planes(
            source, grid, float(meta["wavelength_m"]), (z_reference_m,)
        )[0]
        dense = build_dense_spectral_propagation(
            grid=grid,
            wavelength_m=float(meta["wavelength_m"]),
            z_values_m=z,
            transverse_coordinates_m=transverse_coordinates,
            scalar_field=source,
            source_label=(
                f"{case_id} {family}:{parameter}={value}; physical-error atlas source"
            ),
        )
        records.append({"value": value, "plane": plane, "grid": grid, "dense": dense, "meta": meta})
        xy_scales.append(float(np.max(plane)))
        xz_scales.append(float(np.max(dense.xz_intensity)))

    xy_common = max(xy_scales)
    xz_common = max(xz_scales)
    nominal_idx = _nominal_index(values)
    nominal_plane = np.asarray(records[nominal_idx]["plane"], dtype=float)
    difference_scale = 0.0
    for record in records:
        _, c = _crop(np.asarray(record["plane"]), record["grid"], xy_halfwidth_m)
        _, n = _crop(nominal_plane, record["grid"], xy_halfwidth_m)
        difference_scale = max(difference_scale, float(np.max(np.abs(c - n))))
    difference_scale = max(difference_scale, EPS)

    fig, axes = plt.subplots(
        3,
        len(values),
        figsize=(3.1 * len(values), 8.4),
        constrained_layout=True,
        squeeze=False,
    )
    for col, record in enumerate(records):
        value = record["value"]
        plane = np.asarray(record["plane"])
        grid = record["grid"]
        dense = record["dense"]

        x, cropped = _crop(plane, grid, xy_halfwidth_m)
        shown_xy = _normalise(cropped, xy_common)
        im_xy = axes[0, col].imshow(
            shown_xy,
            origin="lower",
            extent=[x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[0, col].set_title(_label_value(family, parameter, value))
        axes[0, col].set_xlabel("x (µm)")
        if col == 0:
            axes[0, col].set_ylabel(f"y (µm)\nxy @ {z_reference_m*1e3:.0f} mm")
        else:
            axes[0, col].set_ylabel("y (µm)")

        shown_xz = _normalise(dense.xz_intensity, xz_common)
        im_xz = axes[1, col].imshow(
            shown_xz.T,
            origin="lower",
            aspect="auto",
            extent=[z[0]*1e3, z[-1]*1e3, transverse_coordinates[0]*1e6, transverse_coordinates[-1]*1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[1, col].set_xlabel("z from axicon (mm)")
        axes[1, col].set_ylabel("x (µm)")

        _, nominal_crop = _crop(nominal_plane, grid, xy_halfwidth_m)
        diff = (cropped - nominal_crop) / difference_scale
        im_diff = axes[2, col].imshow(
            diff,
            origin="lower",
            extent=[x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6],
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        axes[2, col].set_xlabel("x (µm)")
        axes[2, col].set_ylabel("y (µm)")

    fig.colorbar(im_xy, ax=axes[0, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im_xz, ax=axes[1, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im_diff, ax=axes[2, :].tolist(), label="(I - nominal) / max |difference|")
    fig.suptitle(
        f"{case_id} — {parameter}: physical transverse + propagation response"
        if family != "aberration"
        else f"{case_id} — generic {parameter} wavefront error at axicon plane"
    )

    path = figure_root / _family_subfolder(family) / f"{case_id.lower()}_{parameter}_physical_sweep"
    _save(fig, path)
    plt.close(fig)
    return [str(path.with_suffix(".png")), str(path.with_suffix(".pdf"))]


def run_visual_atlas_figures(
    *,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    grid_n: int = DEFAULT_REPORT_N,
    cases: Sequence[str] = ("B0", "V1", "V3"),
    family_filter: str | None = None,
    parameter_filter: str | None = None,
) -> dict[str, Any]:
    outputs: list[str] = []
    outputs.extend(canonical_plane_grid(figure_root=figure_root, grid_n=grid_n, cases=cases))

    registries = (
        ("parameter", parameter_registry()),
        ("manufacturing", manufacturing_defect_registry()),
        ("aberration", aberration_registry()),
        ("alignment", alignment_registry()),
    )
    for case_id in cases:
        for family, registry in registries:
            if family_filter is not None and family != family_filter:
                continue
            for parameter in registry:
                if parameter_filter is not None and parameter != parameter_filter:
                    continue
                outputs.extend(
                    sweep_transverse_and_propagation_grid(
                        case_id,
                        family=family,
                        parameter=parameter,
                        figure_root=figure_root,
                        grid_n=grid_n,
                    )
                )

    manifest = {
        "outcome": "VORTEX-PHYSICAL-VISUAL-ATLAS-FIGURES",
        "grid_n": int(grid_n),
        "route": "physical-plane SLM/4F/axicon source-scale route",
        "cases": list(cases),
        "family_filter": family_filter,
        "parameter_filter": parameter_filter,
        "visual_contract": (
            "tight xy + xz propagation + nominal difference; common normalization within each sweep"
        ),
        "axicon_tilt_contract": (
            "small-angle rotated thin-element OPD; not full vector Snell/Fresnel"
        ),
        "figure_count": len(outputs),
        "files": outputs,
    }
    figure_root.mkdir(parents=True, exist_ok=True)
    (figure_root / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
