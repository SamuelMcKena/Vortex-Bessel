"""Report-grade figures for the Phase 2E visual bible."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2b_visual_cases import PHASE2B_HEX_REFERENCE_M
from vbb_study.digital_twin.phase2e_report_visualisation import (
    PHASE2E_3D_CASE_IDS,
    H1PolarisationCase,
    Phase2EData,
    ReportFigureStyle,
    ScalarVisualCase,
    SweepPlane,
)


EPS = np.finfo(float).eps
LINEAR_PROPAGATION_CONTRAST_CEILING = 0.01
CASE_TITLES = {
    "G0": "G0 Gaussian control",
    "B0": "B0 bright-core Bessel",
    "V1": "V1 charge-1 vortex Bessel",
    "V3": "V3 charge-3 vortex Bessel",
    "H1_CONTINUOUS": "H1 continuous orientation",
    "H1_AVERAGED": "H1 sector-averaged orientation",
}
ALIGNMENT_ERROR_SWEEPS = {
    "error_input_beam_decentre",
    "error_input_beam_tilt",
    "error_pupil_decentre",
    "error_axicon_decentre",
}


def _mpl(style: ReportFigureStyle) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    plt.rcParams.update({
        "font.family": style.font_family,
        "font.size": style.font_size,
        "axes.titlesize": style.title_size,
        "axes.labelsize": style.font_size,
        "xtick.labelsize": style.font_size - 1.0,
        "ytick.labelsize": style.font_size - 1.0,
        "legend.fontsize": style.font_size - 1.0,
        "lines.linewidth": style.line_width,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt, colors


def _save(fig: Any, stem: Path, style: ReportFigureStyle, *, dpi: int | None = None) -> tuple[Path, Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=int(dpi or style.raster_dpi))
    fig.savefig(pdf, dpi=int(dpi or style.raster_dpi))
    return png, pdf


def _record(
    figure_id: str,
    paths: tuple[Path, Path],
    *,
    family: str,
    role: str,
    case_ids: Sequence[str],
    source_artifacts: Sequence[str],
    data_basis: str,
    normalisation: str,
    linear_log_mode: str,
    x_unit: str,
    y_unit: str,
    z_unit: str = "not_applicable",
    x_limits: Sequence[float] | str = "panel_specific",
    y_limits: Sequence[float] | str = "panel_specific",
    z_limits: Sequence[float] | str = "not_applicable",
    comparison_group: str = "",
    matched_axes: bool = False,
    display_interpolation: str = "none",
    metric_bearing: bool = False,
    metrics_native: bool = True,
    roi_occupancy: Mapping[str, float] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "figure_family": family,
        "report_role": role,
        "png_path": paths[0].as_posix(),
        "pdf_path": paths[1].as_posix(),
        "case_ids": ";".join(case_ids),
        "source_artifacts": ";".join(source_artifacts),
        "data_basis": data_basis,
        "normalisation_policy": normalisation,
        "linear_log_mode": linear_log_mode,
        "x_unit": x_unit,
        "y_unit": y_unit,
        "z_unit": z_unit,
        "x_limits": list(x_limits) if not isinstance(x_limits, str) else x_limits,
        "y_limits": list(y_limits) if not isinstance(y_limits, str) else y_limits,
        "z_limits": list(z_limits) if not isinstance(z_limits, str) else z_limits,
        "comparison_group": comparison_group,
        "matched_axes": bool(matched_axes),
        "display_interpolation": display_interpolation,
        "metric_bearing": bool(metric_bearing),
        "metrics_computed_on_native_arrays": bool(metrics_native),
        "display_interpolation_used_for_metrics": False,
        "roi_occupancy": dict(roi_occupancy or {}),
        "notes": notes,
    }


def _normalise(array: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(array, dtype=float), 0.0)
    return values / max(float(np.max(values)), EPS)


def _radial_profile_plane(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    max_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    radius = np.hypot(X, Y)
    dr = min(abs(float(np.mean(np.diff(x)))), abs(float(np.mean(np.diff(y)))))
    bins = np.floor(radius.ravel() / max(dr, EPS)).astype(int)
    count = int(np.ceil(float(max_radius_m) / max(dr, EPS))) + 1
    totals = np.bincount(bins, weights=values.ravel(), minlength=count)[:count]
    counts = np.bincount(bins, minlength=count)[:count]
    profile = totals / np.maximum(counts, 1)
    radii = (np.arange(count, dtype=float) + 0.5) * dr
    return radii, profile


def _angular_profile_plane(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    radius_m: float,
    samples: int = 720,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.interpolate import RegularGridInterpolator

    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    theta = np.linspace(-np.pi, np.pi, int(samples), endpoint=False)
    points = np.column_stack((float(radius_m) * np.sin(theta), float(radius_m) * np.cos(theta)))
    interpolator = RegularGridInterpolator(
        (y, x), values, method="linear", bounds_error=False, fill_value=0.0
    )
    return theta, np.asarray(interpolator(points), dtype=float)


def _equal_power(array: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(array, dtype=float), 0.0)
    return values / max(float(np.sum(values)), EPS)


def _crop(
    array: np.ndarray,
    grid: Mapping[str, Any],
    halfwidth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    xi = np.flatnonzero(np.abs(x) <= float(halfwidth_m))
    yi = np.flatnonzero(np.abs(y) <= float(halfwidth_m))
    if xi.size < 8 or yi.size < 8:
        raise ValueError("Phase 2E crop contains fewer than eight samples")
    return (
        np.asarray(array)[int(yi[0]) : int(yi[-1]) + 1, int(xi[0]) : int(xi[-1]) + 1],
        x[xi],
        y[yi],
    )


def _crop_at(
    array: np.ndarray,
    grid: Mapping[str, Any],
    halfwidth_m: float,
    centre_x_m: float,
    centre_y_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a display ROI about a declared physical centre."""

    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    xi = np.flatnonzero(np.abs(x - float(centre_x_m)) <= float(halfwidth_m))
    yi = np.flatnonzero(np.abs(y - float(centre_y_m)) <= float(halfwidth_m))
    if xi.size < 8 or yi.size < 8:
        raise ValueError("Phase 2E centred crop contains fewer than eight samples")
    return (
        np.asarray(array)[int(yi[0]) : int(yi[-1]) + 1, int(xi[0]) : int(xi[-1]) + 1],
        x[xi],
        y[yi],
    )


def _intensity_centroid(array: np.ndarray, grid: Mapping[str, Any]) -> tuple[float, float]:
    values = np.maximum(np.asarray(array, dtype=float), 0.0)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    bright_support = values >= 0.35 * max(float(np.max(values)), EPS)
    weights = np.where(bright_support, values, 0.0)
    total = max(float(np.sum(weights)), EPS)
    return (
        float(np.sum(np.sum(weights, axis=0) * x) / total),
        float(np.sum(np.sum(weights, axis=1) * y) / total),
    )


def _extent_mm(x_m: np.ndarray, y_m: np.ndarray) -> list[float]:
    return [float(x_m[0] / 1e-3), float(x_m[-1] / 1e-3), float(y_m[0] / 1e-3), float(y_m[-1] / 1e-3)]


def _display_resample_real(
    values: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(factor) == 1:
        return np.asarray(values), np.asarray(x_m), np.asarray(y_m)
    from scipy.ndimage import zoom

    rendered = zoom(np.asarray(values, dtype=float), int(factor), order=3, prefilter=True)
    rendered = np.maximum(rendered, 0.0)
    return (
        rendered,
        np.linspace(float(x_m[0]), float(x_m[-1]), rendered.shape[1]),
        np.linspace(float(y_m[0]), float(y_m[-1]), rendered.shape[0]),
    )


def _display_resample_complex(
    values: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(factor) == 1:
        return np.asarray(values), np.asarray(x_m), np.asarray(y_m)
    from scipy.ndimage import zoom

    array = np.asarray(values, dtype=np.complex128)
    rendered = zoom(array.real, int(factor), order=3, prefilter=True) + 1j * zoom(
        array.imag, int(factor), order=3, prefilter=True
    )
    return (
        rendered,
        np.linspace(float(x_m[0]), float(x_m[-1]), rendered.shape[1]),
        np.linspace(float(y_m[0]), float(y_m[-1]), rendered.shape[0]),
    )


def _panel(ax: Any, index: int, style: ReportFigureStyle) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        0.015,
        0.985,
        f"({style.panel_labels[index]})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.panel_label_size,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=20,
    )


def _imshow(
    ax: Any,
    values: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "none",
) -> Any:
    return ax.imshow(
        values,
        origin="lower",
        extent=_extent_mm(x_m, y_m),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        aspect="equal",
    )


def _propagation_full_field_bundle(
    propagations: Sequence[Any],
) -> dict[str, Any]:
    """Return complete-window maps with one shared global linear normalisation."""

    if not propagations:
        raise ValueError("at least one propagation map is required")
    first = propagations[0]
    x_m = np.asarray(first.x_m, dtype=float)
    z_m = np.asarray(first.z_m, dtype=float)
    arrays: list[np.ndarray] = []
    for propagation in propagations:
        if not (
            np.allclose(x_m, propagation.x_m)
            and np.allclose(z_m, propagation.z_m)
        ):
            raise ValueError("matched propagation figures require common x/z coordinates")
        arrays.extend(
            (
                np.asarray(propagation.xz_intensity, dtype=float),
                np.asarray(propagation.yz_intensity, dtype=float),
            )
        )
    shared_peak = max(
        max(float(np.max(np.maximum(values, 0.0))), 0.0) for values in arrays
    )
    shared_peak = max(shared_peak, EPS)
    shared_z_peak = np.maximum.reduce(
        [np.max(np.maximum(values, 0.0), axis=1) for values in arrays]
    )
    displayed = [
        np.clip(np.maximum(values, 0.0) / shared_peak, 0.0, 1.0)
        for values in arrays
    ]
    return {
        "x_m": x_m,
        "z_m": z_m,
        "arrays": tuple(displayed),
        "shared_peak": shared_peak,
        "shared_z_peak": shared_z_peak,
        "shared_z_peak_normalised": shared_z_peak / shared_peak,
    }


def _add_bessel_zone_guides(ax: Any, propagation: Any) -> None:
    pupil_zone = propagation.metadata.get("geometric_pupil_bessel_zone_m")
    gaussian_zone = propagation.metadata.get("gaussian_radius_bessel_zone_m")
    if pupil_zone is not None:
        ax.axhline(
            float(pupil_zone) / 1e-3,
            color="#56B4E9",
            linestyle="--",
            linewidth=0.9,
            alpha=0.9,
            label="hard-pupil zone limit",
        )
    if gaussian_zone is not None:
        ax.axhline(
            float(gaussian_zone) / 1e-3,
            color="#009E73",
            linestyle="-.",
            linewidth=0.9,
            alpha=0.9,
            label="1/e beam-radius limit",
        )


def plot_scalar_core_case(
    case: ScalarVisualCase,
    reference: ScalarVisualCase,
    root: Path,
    style: ReportFigureStyle,
) -> dict[str, Any]:
    plt, _ = _mpl(style)
    fig, axes = plt.subplots(3, 3, figsize=(12.3, 11.0), constrained_layout=True)
    input_i, input_x, input_y = _crop(np.abs(case.input_field) ** 2, case.input_grid, 4.0e-3)
    input_p, _, _ = _crop(np.angle(case.input_field), case.input_grid, 4.0e-3)
    focus_field, focus_x, focus_y = _crop(case.focus_field, case.focus_grid, style.scalar_focus_halfwidth_m)
    focus_field, focus_x, focus_y = _display_resample_complex(
        focus_field, focus_x, focus_y, style.scalar_display_resample_factor
    )
    focus_i = np.abs(focus_field) ** 2
    focus_p = np.angle(focus_field)
    ref_field, ref_x, ref_y = _crop(reference.focus_field, reference.focus_grid, style.scalar_focus_halfwidth_m)
    ref_field, _, _ = _display_resample_complex(
        ref_field, ref_x, ref_y, style.scalar_display_resample_factor
    )
    ref_i = np.abs(ref_field) ** 2
    input_in = _normalise(input_i)
    focus_in = _normalise(focus_i)
    ref_in = _normalise(ref_i)
    phase_mask = np.ma.masked_where(input_in < 0.015, input_p)
    focus_phase = np.ma.masked_where(focus_in < 0.015, focus_p)
    _imshow(axes[0, 0], input_in, input_x, input_y, cmap=style.intensity_cmap, vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("axicon-exit intensity | per-panel normalised")
    p0 = _imshow(axes[0, 1], phase_mask, input_x, input_y, cmap=style.phase_cmap, vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("axicon-exit phase | wrapped radians")
    _imshow(axes[0, 2], focus_in, focus_x, focus_y, cmap=style.intensity_cmap, vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("z=60 mm SAS intensity | per-panel normalised")
    p1 = _imshow(axes[1, 0], focus_phase, focus_x, focus_y, cmap=style.phase_cmap, vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title("z=60 mm SAS phase | wrapped radians")
    propagation = case.propagation
    prop_view = _propagation_full_field_bundle((propagation,))
    prop_x = np.asarray(prop_view["x_m"], dtype=float)
    prop_z = np.asarray(prop_view["z_m"], dtype=float)
    extent_prop = [
        float(prop_x[0] / 1e-3),
        float(prop_x[-1] / 1e-3),
        float(prop_z[0] / 1e-3),
        float(prop_z[-1] / 1e-3),
    ]
    xz_display, yz_display = prop_view["arrays"]
    propagation_image = axes[1, 1].imshow(xz_display, origin="lower", extent=extent_prop, aspect="auto", cmap=style.intensity_cmap, vmin=0.0, vmax=LINEAR_PROPAGATION_CONTRAST_CEILING, interpolation="none")
    axes[1, 1].axhline(PHASE2B_HEX_REFERENCE_M / 1e-3, color="white", linestyle=":", linewidth=0.8, alpha=0.8)
    _add_bessel_zone_guides(axes[1, 1], propagation)
    axes[1, 1].set(title="dense x-z | global linear, saturates >0.01", xlabel="x (mm)", ylabel="z (mm)")
    axes[1, 2].imshow(yz_display, origin="lower", extent=extent_prop, aspect="auto", cmap=style.intensity_cmap, vmin=0.0, vmax=LINEAR_PROPAGATION_CONTRAST_CEILING, interpolation="none")
    axes[1, 2].axhline(PHASE2B_HEX_REFERENCE_M / 1e-3, color="white", linestyle=":", linewidth=0.8, alpha=0.8)
    _add_bessel_zone_guides(axes[1, 2], propagation)
    axes[1, 2].set(title="dense y-z | global linear, saturates >0.01", xlabel="y (mm)", ylabel="z (mm)")
    sas_native = np.abs(case.focus_field) ** 2
    radial_radius, radial = _radial_profile_plane(
        sas_native, case.focus_grid, style.scalar_focus_halfwidth_m
    )
    axes[2, 0].plot(radial_radius / 1e-3, radial / max(float(np.nanmax(radial)), EPS), color="#0072B2")
    axes[2, 0].set(title="physical SAS radial profile", xlabel="radius (mm)", ylabel="normalised intensity", xlim=(0, style.scalar_focus_halfwidth_m / 1e-3), ylim=(0, 1.05))
    ring_index = max(1, int(np.argmax(radial[1:]) + 1))
    angular_theta, angular = _angular_profile_plane(
        sas_native, case.focus_grid, float(radial_radius[ring_index])
    )
    if case.case_id != "G0" and np.any(np.isfinite(angular)):
        axes[2, 1].plot(np.degrees(angular_theta), angular / max(float(np.nanmax(angular)), EPS), color="#009E73")
        axes[2, 1].set(title="physical SAS angular profile on ring", xlabel="azimuth (deg)", ylabel="normalised intensity", xlim=(-180, 180), ylim=(0, 1.05))
    else:
        axes[2, 1].text(0.5, 0.5, "not applicable\n(no selected ring)", transform=axes[2, 1].transAxes, ha="center", va="center")
        axes[2, 1].set(title="native angular profile")
    signed = focus_in - ref_in
    limit = max(float(np.max(np.abs(signed))), EPS)
    _imshow(axes[2, 2], signed, focus_x, focus_y, cmap=style.difference_cmap, vmin=-limit, vmax=limit)
    axes[2, 2].set_title(f"signed difference vs {reference.case_id} | panel-normalised inputs")
    for index, ax in enumerate(axes.flat):
        _panel(ax, index, style)
        if index in {0, 1, 2, 3, 8}:
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(p0, ax=axes[0, 1], shrink=0.74, label="phase (rad)")
    fig.colorbar(p1, ax=axes[1, 0], shrink=0.74, label="phase (rad)")
    fig.colorbar(propagation_image, ax=axes[1, 1:3], shrink=0.74, label="I/global Imax | linear; saturated above 0.01")
    fig.suptitle(f"{CASE_TITLES[case.case_id]} | accepted finite-aperture route", fontsize=14.0)
    paths = _save(fig, root / "01_core_beams" / f"{case.case_id.lower()}_core_visual_bible", style)
    plt.close(fig)
    return _record(
        f"core_{case.case_id.lower()}",
        paths,
        family="core_beam_physics",
        role="main_text_candidate",
        case_ids=(case.case_id, reference.case_id),
        source_artifacts=("Phase 2A realistic_fixed_bench_route", "Phase 2E direct spectral propagation", "physical SAS focus field"),
        data_basis="accepted finite-aperture realistic fixed-bench field; fixed-coordinate BL-ASM spectral lines; physical SAS endpoint",
        normalisation="per-panel transverse; propagation global linear I/Imax with fixed 0.01 colour saturation",
        linear_log_mode="linear throughout",
        x_unit="mm",
        y_unit="mm",
        x_limits=(-style.scalar_focus_halfwidth_m / 1e-3, style.scalar_focus_halfwidth_m / 1e-3),
        y_limits=(-style.scalar_focus_halfwidth_m / 1e-3, style.scalar_focus_halfwidth_m / 1e-3),
        comparison_group="scalar_family_focus",
        matched_axes=True,
        display_interpolation=f"focus: complex cubic x{style.scalar_display_resample_factor}; propagation: none after {propagation.x_m.size}x{propagation.z_m.size} physical synthesis",
        metric_bearing=True,
        notes=(
            f"Propagation is a {propagation.x_m.size}x{propagation.z_m.size} physical spectral synthesis, not image interpolation. "
            "The longitudinal panels show the complete 10 mm source-grid window with one global linear normalisation; only x-y focus panels are cropped. "
            "Colour is explicitly saturated above 0.01 to expose low-intensity structure; no per-z renormalisation, logarithm, gamma or propagation interpolation is applied. "
            f"Native inverse-FFT parity error={propagation.metadata['native_line_max_abs_intensity_error']:.3e}; "
            "The accepted hard objective-pupil edge and fixed-bench hardware terms are retained. The boundary audit separates their physical diffraction from numerical band-limiting; phase masks are intensity-masked only for display."
        ),
    )


def plot_dense_propagation_atlas(
    data: Phase2EData,
    case_id: str,
    root: Path,
) -> dict[str, Any]:
    """Large x-z/y-z intensity views over the complete computational window."""

    style = data.config.style
    plt, _ = _mpl(style)
    if case_id in {"G0", "B0", "V1", "V3"}:
        propagation = data.scalar_cases[case_id].propagation
        source = "accepted finite-aperture realistic fixed-bench direct spectral propagation"
    else:
        propagation = data.h1_propagation[case_id]
        source = "projected-vector direct spectral propagation"
    view = _propagation_full_field_bundle((propagation,))
    x_m = np.asarray(view["x_m"], dtype=float)
    z_m = np.asarray(view["z_m"], dtype=float)
    extent = [
        float(x_m[0] / 1e-3),
        float(x_m[-1] / 1e-3),
        float(z_m[0] / 1e-3),
        float(z_m[-1] / 1e-3),
    ]
    fig = plt.figure(figsize=(15.2, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.72))
    full_axes = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
    contrast_axes = (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
    axial_ax = fig.add_subplot(gs[:, 2])
    full_image = None
    for index, (ax, values, transverse_axis) in enumerate(
        zip(full_axes, view["arrays"], ("x", "y"))
    ):
        full_image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=style.intensity_cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="none",
        )
        ax.axhline(
            PHASE2B_HEX_REFERENCE_M / 1e-3,
            color="white",
            linestyle=":",
            linewidth=0.9,
            alpha=0.85,
        )
        _add_bessel_zone_guides(ax, propagation)
        ax.set(
            title=f"{transverse_axis}-z | full field, linear 0--1",
            xlabel=f"{transverse_axis} (mm)",
            ylabel="z (mm)",
        )
        _panel(ax, index, style)
    contrast_image = None
    for index, (ax, values, transverse_axis) in enumerate(
        zip(contrast_axes, view["arrays"], ("x", "y")),
        start=3,
    ):
        contrast_image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=style.intensity_cmap,
            vmin=0.0,
            vmax=LINEAR_PROPAGATION_CONTRAST_CEILING,
            interpolation="none",
        )
        ax.axhline(
            PHASE2B_HEX_REFERENCE_M / 1e-3,
            color="white",
            linestyle=":",
            linewidth=0.9,
            alpha=0.85,
        )
        _add_bessel_zone_guides(ax, propagation)
        ax.set(
            title=(
                f"{transverse_axis}-z | linear low-intensity view, "
                f"saturated >{LINEAR_PROPAGATION_CONTRAST_CEILING:.2f}"
            ),
            xlabel=f"{transverse_axis} (mm)",
            ylabel="z (mm)",
        )
        _panel(ax, index, style)
    if (
        propagation.metadata.get("geometric_pupil_bessel_zone_m") is not None
        or propagation.metadata.get("gaussian_radius_bessel_zone_m") is not None
    ):
        full_axes[0].legend(loc="upper left", frameon=True, fontsize=7.4)
        contrast_axes[0].legend(loc="upper left", frameon=True, fontsize=7.4)
    axial = np.asarray(view["shared_z_peak_normalised"], dtype=float)
    axial_ax.plot(axial, z_m / 1e-3, color="#0072B2", linewidth=1.5)
    axial_ax.axhline(
        PHASE2B_HEX_REFERENCE_M / 1e-3,
        color="#555555",
        linestyle=":",
        linewidth=0.9,
    )
    _add_bessel_zone_guides(axial_ax, propagation)
    axial_ax.set(
        title="shared transverse peak | globally linear",
        xlabel="peak I(z) / global Imax",
        ylabel="z (mm)",
        xlim=(0.0, 1.05),
        ylim=(extent[2], extent[3]),
    )
    _panel(axial_ax, 2, style)
    assert full_image is not None
    fig.colorbar(
        full_image,
        ax=full_axes,
        shrink=0.86,
        label="I(x,z) / global Imax | linear 0--1",
    )
    assert contrast_image is not None
    fig.colorbar(
        contrast_image,
        ax=contrast_axes,
        shrink=0.86,
        label=(
            "I(x,z) / global Imax | colour saturated above "
            f"{LINEAR_PROPAGATION_CONTRAST_CEILING:.2f}"
        ),
    )
    fig.suptitle(
        f"{CASE_TITLES[case_id]} | full-field propagation intensity, direct BL-ASM",
        fontsize=14.0,
    )
    stem_id = case_id.lower()
    paths = _save(
        fig,
        root / "01b_propagation_maps" / f"{stem_id}_dense_xz_yz_global_linear_dual_range",
        style,
    )
    plt.close(fig)
    return _record(
        f"propagation_{stem_id}",
        paths,
        family="dense_propagation_full_field",
        role="main_text_candidate" if case_id != "G0" else "supplementary_candidate",
        case_ids=(case_id,),
        source_artifacts=("Phase 2E dense spectral propagation",),
        data_basis=source,
        normalisation="maps: paired uncapped and 0.01-saturated linear I/global Imax, shared across x-z/y-z; axial curve: global linear peak/Imax",
        linear_log_mode="linear throughout",
        x_unit="mm",
        y_unit="mm",
        x_limits=(extent[0], extent[1]),
        y_limits=(extent[2], extent[3]),
        comparison_group=f"{stem_id}_dense_propagation",
        matched_axes=True,
        display_interpolation=f"none after {propagation.x_m.size}x{propagation.z_m.size} physical spectral synthesis",
        metric_bearing=False,
        notes=(
            "Both longitudinal panels show the complete 10 mm source-grid window and the full z=0--200 mm interval. "
            "Top maps retain the full linear 0--1 range. Bottom maps show the identical ratios with colour saturation above 0.01 so low-intensity full-field morphology is visible; this is a fixed linear colour limit, not a data transform. "
            "The third panel preserves axial amplitude as the globally linear shared peak. No propagation ROI, per-z renormalisation, logarithm, gamma, percentile-based limit, spatial interpolation or display-derived metric is used. "
            "Scalar maps use the accepted finite-aperture realistic fixed-bench route; the separate boundary audit distinguishes hard-pupil diffraction from numerical band-limiting. Dashed guides mark applicable geometric Bessel-zone limits."
        ),
    )


def plot_b0_propagation_boundary_audit(
    data: Phase2EData,
    root: Path,
) -> dict[str, Any]:
    """Show which B0 longitudinal features come from the finite hard pupil."""

    style = data.config.style
    plt, _ = _mpl(style)
    audit = data.propagation_boundary_audit
    realistic = data.scalar_cases["B0"].propagation
    maps = (audit.ideal_untruncated, audit.hard_pupil, realistic)
    view = _propagation_full_field_bundle(maps)
    x_m = np.asarray(view["x_m"], dtype=float)
    z_m = np.asarray(view["z_m"], dtype=float)
    extent = [
        float(x_m[0] / 1e-3),
        float(x_m[-1] / 1e-3),
        float(z_m[0] / 1e-3),
        float(z_m[-1] / 1e-3),
    ]
    fig = plt.figure(figsize=(14.0, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.62))
    axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    image = None
    for index, (ax, values, title, propagation) in enumerate(
        zip(
            axes,
            (view["arrays"][0], view["arrays"][2], view["arrays"][4]),
            (
                "unclipped Gaussian-axicon control",
                "1.8 mm hard-pupil control",
                "accepted fixed-bench route",
            ),
            maps,
        )
    ):
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=style.intensity_cmap,
            vmin=0.0,
            vmax=LINEAR_PROPAGATION_CONTRAST_CEILING,
            interpolation="none",
        )
        _add_bessel_zone_guides(ax, propagation)
        ax.set_title(f"{title}\nglobal linear, saturates >0.01", fontsize=9.2)
        ax.set(xlabel="x (mm)", ylabel="z (mm)")
        _panel(ax, index, style)
    assert image is not None
    fig.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        label="I(x,z) / shared global Imax | linear; saturated above 0.01",
    )

    zero = int(np.argmin(np.abs(x_m)))
    curve_ax = fig.add_subplot(gs[1, 0:2])
    for propagation, label, color, linestyle in (
        (maps[0], "ideal untruncated", "#0072B2", "-"),
        (maps[1], "hard pupil", "#D55E00", "--"),
        (maps[2], "realistic route", "#009E73", ":"),
    ):
        axis_intensity = 0.5 * (
            propagation.xz_intensity[:, zero]
            + propagation.yz_intensity[:, zero]
        )
        curve_ax.plot(
            z_m / 1e-3,
            axis_intensity / max(float(np.max(axis_intensity)), EPS),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
        )
    pupil_zone = float(audit.metrics["geometric_pupil_bessel_zone_m"])
    gaussian_zone = float(audit.metrics["gaussian_radius_bessel_zone_m"])
    curve_ax.axvline(pupil_zone / 1e-3, color="#56B4E9", linestyle="--", linewidth=1.0)
    curve_ax.axvline(gaussian_zone / 1e-3, color="#009E73", linestyle="-.", linewidth=1.0)
    curve_ax.set(
        title="on-axis intensity | each curve peak-normalised",
        xlabel="z (mm)",
        ylabel="I / curve Imax",
        xlim=(extent[2], extent[3]),
        ylim=(0.0, 1.05),
    )
    curve_ax.legend(frameon=False, ncol=3)
    _panel(curve_ax, 3, style)

    metric_ax = fig.add_subplot(gs[1, 2])
    ripple_values = [
        float(audit.metrics[key]["ripple_rms_normalised"])
        for key in ("ideal_untruncated", "hard_pupil", "realistic_route")
    ]
    metric_ax.bar(
        ("ideal", "hard pupil", "realistic"),
        ripple_values,
        color=("#0072B2", "#D55E00", "#009E73"),
    )
    metric_ax.set(
        title="20--100 mm axial ripple",
        ylabel="RMS after 15 mm envelope removal",
    )
    metric_ax.tick_params(axis="x", rotation=18)
    metric_ax.text(
        0.02,
        0.96,
        f"BL/no-BL corr = {audit.metrics['bandlimited_to_unbandlimited_on_axis_correlation']:.8f}\n"
        f"hard-pupil power = {audit.metrics['hard_pupil_power_fraction']:.3f}",
        transform=metric_ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
    )
    _panel(metric_ax, 4, style)
    fig.suptitle(
        "B0 propagation-boundary audit | finite pupil versus numerical propagation",
        fontsize=14.0,
    )
    paths = _save(
        fig,
        root / "01b_propagation_maps" / "b0_pupil_boundary_truth_audit_linear",
        style,
    )
    plt.close(fig)
    return _record(
        "propagation_b0_boundary_audit",
        paths,
        family="propagation_boundary_truth_audit",
        role="main_text_candidate",
        case_ids=("B0",),
        source_artifacts=(
            "ideal Gaussian axicon control",
            "canonical 1.8 mm hard-pupil control",
            "Phase 2A realistic_fixed_bench_route",
        ),
        data_basis="matched full-window direct BL-ASM maps and native on-axis intensities",
        normalisation="maps use one shared global linear maximum with fixed 0.01 colour saturation; axial curves are individually peak-normalised and linear",
        linear_log_mode="linear throughout",
        x_unit="mm",
        y_unit="mm",
        x_limits=(extent[0], extent[1]),
        y_limits=(extent[2], extent[3]),
        comparison_group="b0_propagation_boundary",
        matched_axes=True,
        display_interpolation="none",
        metric_bearing=True,
        notes=(
            "The ideal control removes only hard-pupil clipping. The hard-pupil and realistic maps retain the "
            "canonical 1.8 mm stop. BL/no-BL agreement isolates the axial ripple and post-zone flare from the "
            "Matsushima mask; no logarithm or gamma is applied, and all reported metrics are computed on propagation arrays before rendering."
        ),
    )


def _surface_source(data: Phase2EData, case_id: str) -> tuple[np.ndarray, Mapping[str, Any], float, str]:
    style = data.config.style
    if case_id in {"B0", "V1", "V3"}:
        case = data.scalar_cases[case_id]
        return np.abs(case.focus_field) ** 2, case.focus_grid, style.scalar_focus_halfwidth_m, "physical SAS field"
    label = "continuous" if case_id == "H1_CONTINUOUS" else "sector_averaged"
    hero = data.hex_package.highn_hero
    return hero["sas_planes"][label], hero["sas_grids"][label], style.h1_focus_halfwidth_m, "accepted N=1536 H1 SAS endpoint"


def plot_intensity_surface(data: Phase2EData, case_id: str, root: Path) -> dict[str, Any]:
    style = data.config.style
    plt, _ = _mpl(style)
    raw, grid, halfwidth, source = _surface_source(data, case_id)
    crop, x, y = _crop(raw, grid, halfwidth)
    display_factor = (
        style.scalar_display_resample_factor
        if case_id in {"B0", "V1", "V3"}
        else style.h1_display_resample_factor
    )
    crop, x, y = _display_resample_real(crop, x, y, display_factor)
    intensity = _normalise(crop)
    stride = max(1, int(np.ceil(max(intensity.shape) / style.surface_render_max_n)))
    render = intensity[::stride, ::stride]
    xr = x[::stride] / 1e-3
    yr = y[::stride] / 1e-3
    X, Y = np.meshgrid(xr, yr, indexing="xy")
    fig = plt.figure(figsize=(13.8, 6.2), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    surface = ax3d.plot_surface(X, Y, render, cmap=style.intensity_cmap, vmin=0.0, vmax=1.0, linewidth=0, antialiased=True, shade=False)
    surface.set_rasterized(True)
    ax3d.view_init(elev=style.surface_elevation_deg, azim=style.surface_azimuth_deg)
    ax3d.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="I / Imax", zlim=style.surface_z_limits)
    ax3d.xaxis.labelpad = 9
    ax3d.yaxis.labelpad = 9
    ax3d.zaxis.labelpad = 5
    ax3d.set_xlim(-halfwidth / 1e-3, halfwidth / 1e-3)
    ax3d.set_ylim(-halfwidth / 1e-3, halfwidth / 1e-3)
    ax3d.set_box_aspect((1, 1, 0.55))
    ax3d.set_title("oblique intensity surface")
    im = _imshow(ax2d, intensity, x, y, cmap=style.intensity_cmap, vmin=0.0, vmax=1.0, interpolation="none")
    ax2d.set(title="top-down parity view | same array and colour limits", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im, ax=ax2d, shrink=0.82, pad=0.035, label="normalised intensity (linear)")
    _panel(ax3d, 0, style)
    _panel(ax2d, 1, style)
    fig.suptitle(f"{CASE_TITLES[case_id]} | pure transverse intensity at z=60 mm", fontsize=14.0)
    paths = _save(fig, root / "02_3d_surfaces" / f"{case_id.lower()}_pure_intensity_surface", style, dpi=400)
    plt.close(fig)
    limits = (-halfwidth / 1e-3, halfwidth / 1e-3)
    return _record(
        f"surface_{case_id.lower()}",
        paths,
        family="pure_3d_intensity",
        role="main_text_candidate" if case_id in PHASE2E_3D_CASE_IDS else "supplementary_candidate",
        case_ids=(case_id,),
        source_artifacts=(source,),
        data_basis="same beam-centred transverse intensity array in oblique and top-down panels",
        normalisation="per-panel normalised",
        linear_log_mode="linear",
        x_unit="mm",
        y_unit="mm",
        z_unit="normalised intensity",
        x_limits=limits,
        y_limits=limits,
        z_limits=style.surface_z_limits,
        comparison_group="scalar_3d" if case_id in {"B0", "V1", "V3"} else "h1_3d",
        matched_axes=True,
        display_interpolation=f"cubic x{display_factor} after physical SAS; surface plotting stride {stride}; display only",
        metric_bearing=False,
        notes="The top-down panel is the full display-resampled crop; no propagation axis is present and no metric uses this array.",
    )


def _h1_focus(package: Any, label: str) -> tuple[np.ndarray, Mapping[str, Any]]:
    hero = package.highn_hero
    return np.asarray(hero["sas_planes"][label]), hero["sas_grids"][label]


def plot_h1_matched_comparison(data: Phase2EData, root: Path) -> dict[str, Any]:
    style = data.config.style
    plt, _ = _mpl(style)
    package = data.hex_package
    continuous, grid = _h1_focus(package, "continuous")
    averaged, _ = _h1_focus(package, "sector_averaged")
    c_native, cx, cy = _crop(continuous, grid, style.h1_focus_halfwidth_m)
    a_native, a_x, a_y = _crop(averaged, grid, style.h1_focus_halfwidth_m)
    c, x, y = _display_resample_real(c_native, cx, cy, style.h1_display_resample_factor)
    a, _, _ = _display_resample_real(a_native, a_x, a_y, style.h1_display_resample_factor)
    cn = _normalise(c)
    an = _normalise(a)
    signed = _equal_power(c) - _equal_power(a)
    signed /= max(float(np.max(np.abs(signed))), EPS)
    fig = plt.figure(figsize=(14.0, 11.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    axes = [fig.add_subplot(gs[0, index]) for index in range(4)]
    _imshow(axes[0], cn, x, y, cmap=style.intensity_cmap, vmin=0, vmax=1)
    axes[0].set_title("continuous | per-panel normalised")
    _imshow(axes[1], an, x, y, cmap=style.intensity_cmap, vmin=0, vmax=1)
    axes[1].set_title("sector averaged | per-panel normalised")
    _imshow(axes[2], np.abs(cn - an), x, y, cmap="magma", vmin=0, vmax=max(float(np.max(np.abs(cn - an))), EPS))
    axes[2].set_title("absolute morphology difference")
    _imshow(axes[3], signed, x, y, cmap=style.difference_cmap, vmin=-1, vmax=1)
    axes[3].set_title("signed equal-power difference")
    result_c = package.continuous
    result_a = package.averaged
    prop_c = data.h1_propagation["H1_CONTINUOUS"]
    prop_a = data.h1_propagation["H1_AVERAGED"]
    prop_view = _propagation_full_field_bundle((prop_c, prop_a))
    prop_x = np.asarray(prop_view["x_m"], dtype=float)
    prop_z = np.asarray(prop_view["z_m"], dtype=float)
    extent = [
        float(prop_x[0] / 1e-3),
        float(prop_x[-1] / 1e-3),
        float(prop_z[0] / 1e-3),
        float(prop_z[-1] / 1e-3),
    ]
    prop_axes = [fig.add_subplot(gs[1, index]) for index in range(4)]
    propagation_image = None
    for ax, values, title, propagation in (
        (prop_axes[0], prop_view["arrays"][0], "continuous x-z", prop_c),
        (prop_axes[1], prop_view["arrays"][2], "averaged x-z", prop_a),
        (prop_axes[2], prop_view["arrays"][1], "continuous y-z", prop_c),
        (prop_axes[3], prop_view["arrays"][3], "averaged y-z", prop_a),
    ):
        propagation_image = ax.imshow(values, origin="lower", extent=extent, aspect="auto", cmap=style.intensity_cmap, vmin=0.0, vmax=LINEAR_PROPAGATION_CONTRAST_CEILING, interpolation="none")
        ax.axhline(PHASE2B_HEX_REFERENCE_M / 1e-3, color="white", linestyle=":", linewidth=0.8, alpha=0.8)
        _add_bessel_zone_guides(ax, propagation)
        ax.set(title=f"{title} | global linear, saturates >0.01", xlabel="transverse position (mm)", ylabel="z (mm)")
    profile_ax = fig.add_subplot(gs[2, 0:2])
    c_radius, c_radial = _radial_profile_plane(continuous, grid, style.h1_focus_halfwidth_m)
    a_radius, a_radial = _radial_profile_plane(averaged, grid, style.h1_focus_halfwidth_m)
    profile_ax.plot(c_radius / 1e-3, _normalise(c_radial), label="continuous", color="#0072B2")
    profile_ax.plot(a_radius / 1e-3, _normalise(a_radial), label="sector averaged", color="#D55E00", linestyle="--")
    profile_ax.set(title="high-N SAS radial ridge profile", xlabel="radius (mm)", ylabel="normalised intensity", xlim=(0, style.h1_focus_halfwidth_m / 1e-3), ylim=(0, 1.05))
    profile_ax.legend(frameon=False)
    cut_ax = fig.add_subplot(gs[2, 2:4])
    mid = c_native.shape[0] // 2
    cut_ax.plot(cx / 1e-3, _normalise(c_native[mid]), label="continuous horizontal", color="#0072B2")
    cut_ax.plot(a_x / 1e-3, _normalise(a_native[mid]), label="averaged horizontal", color="#D55E00", linestyle="--")
    cut_ax.plot(cy / 1e-3, _normalise(c_native[:, mid]), label="continuous vertical", color="#56B4E9", alpha=0.8)
    cut_ax.plot(a_y / 1e-3, _normalise(a_native[:, mid]), label="averaged vertical", color="#E69F00", linestyle=":")
    cut_ax.set(title="matched edge/ridge cuts", xlabel="position (mm)", ylabel="normalised intensity", xlim=(-style.h1_focus_halfwidth_m / 1e-3, style.h1_focus_halfwidth_m / 1e-3), ylim=(0, 1.05))
    cut_ax.legend(frameon=False, ncol=2)
    all_axes = axes + prop_axes + [profile_ax, cut_ax]
    for index, ax in enumerate(all_axes):
        _panel(ax, index, style)
        if index < 4:
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
    assert propagation_image is not None
    fig.colorbar(propagation_image, ax=prop_axes, shrink=0.75, label="I/global Imax | linear; saturated above 0.01")
    fig.suptitle("H1 continuous versus sector-averaged | matched physical scales and native metrics", fontsize=14.0)
    paths = _save(fig, root / "03_h1" / "h1_continuous_vs_averaged_matched", style)
    plt.close(fig)
    limits = (-style.h1_focus_halfwidth_m / 1e-3, style.h1_focus_halfwidth_m / 1e-3)
    return _record(
        "hero_h1_continuous_vs_averaged",
        paths,
        family="h1_continuous_averaged",
        role="hero_figure",
        case_ids=("H1_CONTINUOUS", "H1_AVERAGED"),
        source_artifacts=("Phase 2B accepted N=1536 H1 endpoints", "Phase 2E dense projected-vector spectral propagation", "MODE 2Y/2Z endpoint audit"),
        data_basis="high-N SAS endpoints and fixed-coordinate projected-vector BL-ASM spectral lines",
        normalisation="per-panel transverse; propagation shared-global linear I/Imax with fixed 0.01 colour saturation",
        linear_log_mode="linear throughout",
        x_unit="mm",
        y_unit="mm",
        x_limits=limits,
        y_limits=limits,
        comparison_group="h1_matched",
        matched_axes=True,
        display_interpolation=f"transverse: cubic x{style.h1_display_resample_factor}; propagation: none after {prop_c.x_m.size}x{prop_c.z_m.size} physical synthesis",
        metric_bearing=True,
        notes=(
            f"Propagation maps are {prop_c.x_m.size}x{prop_c.z_m.size} projected-vector spectral syntheses. "
            "Longitudinal panels show the complete 10 mm source-grid window and full z interval. "
            f"Native inverse-FFT parity errors are {prop_c.metadata['native_line_max_abs_intensity_error']:.3e} "
            f"and {prop_a.metadata['native_line_max_abs_intensity_error']:.3e}; edge/ridge profiles use native high-N SAS arrays."
        ),
    )


def _orientation_wrap(alpha: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(alpha, dtype=float) + 0.5 * np.pi, np.pi) - 0.5 * np.pi


def plot_h1_polarisation(data: Phase2EData, root: Path) -> dict[str, Any]:
    style = data.config.style
    plt, _ = _mpl(style)
    fig, axes = plt.subplots(2, 5, figsize=(16.0, 6.8), constrained_layout=True)
    for row, case_id in enumerate(("H1_CONTINUOUS", "H1_AVERAGED")):
        case: H1PolarisationCase = data.h1_polarisation[case_id]
        alpha, x, y = _crop(_orientation_wrap(case.input_orientation_rad), case.input_grid, style.h1_focus_halfwidth_m)
        input_i, _, _ = _crop(case.input_intensity, case.input_grid, style.h1_focus_halfwidth_m)
        alpha = np.ma.masked_where(_normalise(input_i) < 0.04, alpha)
        _imshow(axes[row, 0], alpha, x, y, cmap=style.phase_cmap, vmin=-0.5 * np.pi, vmax=0.5 * np.pi)
        axes[row, 0].set_title(f"{case_id.replace('H1_', '').lower()} input orientation")
        s0, sx, sy = _crop(case.stokes["S0"], case.input_grid, style.h1_focus_halfwidth_m)
        mask = s0 < 0.02 * max(float(np.max(s0)), EPS)
        focal_orientation, _, _ = _crop(_orientation_wrap(case.stokes["orientation_rad"]), case.input_grid, style.h1_focus_halfwidth_m)
        focal_orientation = np.ma.masked_where(mask, focal_orientation)
        _imshow(axes[row, 1], focal_orientation, sx, sy, cmap=style.phase_cmap, vmin=-0.5 * np.pi, vmax=0.5 * np.pi)
        axes[row, 1].set_title("focal transverse orientation")
        for column, stokes_id in enumerate(("S1", "S2", "S3"), start=2):
            component, _, _ = _crop(case.stokes[stokes_id], case.input_grid, style.h1_focus_halfwidth_m)
            ratio = np.ma.masked_where(mask, component / np.maximum(s0, EPS))
            _imshow(axes[row, column], ratio, sx, sy, cmap=style.signed_cmap, vmin=-1.0, vmax=1.0)
            axes[row, column].set_title(f"{stokes_id}/S0 at z=60 mm")
    for index, ax in enumerate(axes.flat):
        _panel(ax, index, style)
        ax.set(xlabel="x (mm)", ylabel="y (mm)")
    fig.suptitle("H1 local polarisation orientation and Stokes maps | intensity-masked display", fontsize=14.0)
    paths = _save(fig, root / "03_h1" / "h1_local_orientation_and_stokes", style)
    plt.close(fig)
    limits = (-style.h1_focus_halfwidth_m / 1e-3, style.h1_focus_halfwidth_m / 1e-3)
    return _record(
        "h1_local_orientation_and_stokes",
        paths,
        family="h1_polarisation",
        role="main_text_candidate",
        case_ids=("H1_CONTINUOUS", "H1_AVERAGED"),
        source_artifacts=("MODE 2Y continuous and sector-averaged local Jones fields",),
        data_basis="native vector components and Stokes definitions",
        normalisation="Stokes components divided by local S0",
        linear_log_mode="linear signed",
        x_unit="mm",
        y_unit="mm",
        x_limits=limits,
        y_limits=limits,
        comparison_group="h1_polarisation_matched",
        matched_axes=True,
        display_interpolation="none; low-S0 pixels masked for legibility",
        metric_bearing=False,
    )


def _sweep_x_values(planes: Sequence[SweepPlane]) -> tuple[np.ndarray, str]:
    sweep_id = planes[0].sweep_id
    values = np.asarray([plane.parameter_value for plane in planes], dtype=float)
    if sweep_id == "radial_wavevector":
        return values / 1e3, "k_r (krad/m)"
    if sweep_id in {"input_beam_radius", "propagation_distance", "aperture_radius"}:
        return values / 1e-3, f"{planes[0].parameter_name.replace('_', ' ')} (mm)"
    if sweep_id == "effective_objective_na":
        return values, "effective spectral NA"
    if sweep_id == "defocus_aberration":
        return values, "defocus coefficient (waves)"
    if sweep_id in {"error_input_beam_decentre", "error_axicon_decentre"}:
        return values / 1e-6, f"{planes[0].parameter_name.replace('_', ' ')} (um)"
    if sweep_id == "error_input_beam_tilt":
        return values / 1e-3, "input beam tilt (mrad)"
    if sweep_id == "error_slm_phase":
        return values, "SLM phase error coefficient (rad)"
    if sweep_id in {"error_fourier_iris_offset", "error_pupil_decentre"}:
        return values, planes[0].parameter_unit
    if sweep_id.startswith("error_zernike_"):
        return values, f"{sweep_id.removeprefix('error_zernike_')} coefficient (waves)"
    return values, "vortex charge ell"


def plot_sweep(planes: Sequence[SweepPlane], root: Path, style: ReportFigureStyle) -> dict[str, Any]:
    plt, _ = _mpl(style)
    sweep_id = planes[0].sweep_id
    physical_error = planes[0].provenance.get("sweep_kind") == "physical_error"
    halfwidth = (
        style.effective_na_sweep_halfwidth_m
        if sweep_id == "effective_objective_na"
        else style.realism_focus_halfwidth_m
        if sweep_id in ALIGNMENT_ERROR_SWEEPS
        else style.sweep_focus_halfwidth_m
    )
    fig = plt.figure(figsize=(15.0, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=(1.0, 0.72))
    for index, plane in enumerate(planes):
        ax = fig.add_subplot(gs[0, index])
        crop, x, y = _crop(plane.intensity, plane.grid, halfwidth)
        crop, x, y = _display_resample_real(
            crop, x, y, style.scalar_display_resample_factor
        )
        _imshow(ax, _normalise(crop), x, y, cmap=style.intensity_cmap, vmin=0, vmax=1)
        ax.set(title=plane.display_label, xlabel="x (mm)", ylabel="y (mm)")
        if sweep_id == "effective_objective_na" or sweep_id in ALIGNMENT_ERROR_SWEEPS:
            if sweep_id in ALIGNMENT_ERROR_SWEEPS:
                centre_x, centre_y = _intensity_centroid(plane.intensity, plane.grid)
                detail, detail_x, detail_y = _crop_at(
                    plane.intensity,
                    plane.grid,
                    style.sweep_focus_halfwidth_m,
                    centre_x,
                    centre_y,
                )
            else:
                detail, detail_x, detail_y = _crop(
                    plane.intensity, plane.grid, style.sweep_focus_halfwidth_m
                )
            detail, detail_x, detail_y = _display_resample_real(
                detail, detail_x, detail_y, style.scalar_display_resample_factor
            )
            inset = ax.inset_axes([0.57, 0.56, 0.39, 0.39])
            _imshow(
                inset,
                _normalise(detail),
                detail_x,
                detail_y,
                cmap=style.intensity_cmap,
                vmin=0,
                vmax=1,
            )
            inset.set_title(
                "beam ROI" if sweep_id in ALIGNMENT_ERROR_SWEEPS else "central ROI",
                fontsize=6.5,
                color="white",
                pad=1.0,
            )
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("white")
                spine.set_linewidth(0.8)
        _panel(ax, index, style)
    x_values, x_label = _sweep_x_values(planes)
    ax_ring = fig.add_subplot(gs[1, 0:3])
    ax_ratio = fig.add_subplot(gs[1, 3:5])
    if physical_error:
        centroid = np.asarray([plane.metrics["centroid_shift_m"] for plane in planes], dtype=float) / 1e-6
        deformation = np.asarray([plane.metrics["morphology_relative_l2_to_baseline"] for plane in planes], dtype=float)
        correlation = np.asarray([plane.metrics["morphology_correlation_to_baseline"] for plane in planes], dtype=float)
        peak = np.asarray([plane.metrics["peak_relative_to_baseline"] for plane in planes], dtype=float)
        power = np.asarray([plane.metrics["represented_power_fraction"] for plane in planes], dtype=float)
        ax_ring.plot(x_values, centroid, marker="o", color="#0072B2", label="centroid shift")
        ax_ring.set(title="native SAS displacement and deformation", xlabel=x_label, ylabel="centroid shift (um)")
        twin = ax_ring.twinx()
        twin.plot(x_values, deformation, marker="s", color="#D55E00", label="relative L2")
        twin.set_ylabel("morphology relative L2")
        handles = ax_ring.lines + twin.lines
        ax_ring.legend(handles, [line.get_label() for line in handles], frameon=False, loc="upper left")
        ax_ratio.plot(x_values, correlation, marker="o", label="correlation", color="#009E73")
        ax_ratio.plot(x_values, peak, marker="s", label="peak / baseline", color="#CC79A7")
        ax_ratio.plot(x_values, power, marker="^", label="represented power", color="#E69F00")
        ax_ratio.set(title="native morphology and power response", xlabel=x_label, ylabel="ratio")
    else:
        ring = np.asarray([plane.metrics["ring_radius_m"] for plane in planes], dtype=float) / 1e-3
        central = np.asarray([plane.metrics["central_intensity_ratio"] for plane in planes], dtype=float)
        sidelobe = np.asarray([plane.metrics["side_lobe_ratio"] for plane in planes], dtype=float)
        ax_ring.plot(x_values, ring, marker="o", color="#0072B2")
        ax_ring.set(title="native SAS ring/peak radius", xlabel=x_label, ylabel="radius (mm)")
        ax_ratio.plot(x_values, central, marker="o", label="central / peak", color="#D55E00")
        ax_ratio.plot(x_values, sidelobe, marker="s", label="outer max / radial peak", color="#009E73")
        ax_ratio.set(title="native contrast metrics", xlabel=x_label, ylabel="ratio", ylim=(0, max(1.05, 1.05 * float(np.max([central, sidelobe])))))
    ax_ratio.legend(frameon=False)
    _panel(ax_ring, 5, style)
    _panel(ax_ratio, 6, style)
    title_scope = "one-at-a-time physical error diagnostic" if physical_error else "not a fixed-bench claim"
    fig.suptitle(f"V1 diagnostic sweep: {sweep_id.replace('_', ' ')} | {title_scope}", fontsize=14.0)
    paths = _save(fig, root / "04_parameter_sweeps" / f"sweep_{sweep_id}", style)
    plt.close(fig)
    limits = (-halfwidth / 1e-3, halfwidth / 1e-3)
    return _record(
        f"sweep_{sweep_id}",
        paths,
        family="parameter_sweep",
        role="supplementary_candidate",
        case_ids=("V1",),
        source_artifacts=("Phase 2E analytic finite-energy vortex-Bessel screening sweep",),
        data_basis=(
            "canonical source-scale route with one physical pre-propagation perturbation"
            if physical_error
            else "new, isolated native SAS diagnostic arrays"
        ),
        normalisation="per-panel normalised for images; native metrics unnormalised before ratios",
        linear_log_mode="linear",
        x_unit="mm",
        y_unit="mm",
        x_limits=limits,
        y_limits=limits,
        comparison_group=f"sweep_{sweep_id}",
        matched_axes=True,
        display_interpolation=f"cubic x{style.scalar_display_resample_factor} after physical SAS; display only",
        metric_bearing=True,
        notes=(
            "Diagnostic screening only; every point passes explicit input-plane Nyquist and SAS validity checks. "
            + (
                f"Perturbation is applied at the {planes[0].provenance.get('physical_plane')} plane; baseline hardware quantisation, common 4F filtering, pupil and axicon remain active. "
                if physical_error
                else ""
            )
            + (
                f"Matched full-field panels include a common central +/-{style.sweep_focus_halfwidth_m / 1e-3:.2f} mm display-only ROI inset."
                if sweep_id == "effective_objective_na"
                else f"Matched wide panels include beam-centred +/-{style.sweep_focus_halfwidth_m / 1e-3:.2f} mm display-only ROI insets."
                if sweep_id in ALIGNMENT_ERROR_SWEEPS
                else "The common physical crop keeps the beam ROI report-legible."
            )
        ),
    )


def _centre_profile(array: np.ndarray) -> np.ndarray:
    values = _normalise(array)
    return values[values.shape[0] // 2]


def plot_vortex_realism(data: Phase2EData, root: Path) -> dict[str, Any]:
    style = data.config.style
    plt, _ = _mpl(style)
    variants = ("ideal_optical_route", "realistic_fixed_bench_route", "deliberately_degraded_route")
    titles = ("ideal optical", "realistic fixed bench", "deliberately degraded")
    fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    panel_index = 0
    for row, case_id in enumerate(("V1", "V3")):
        arrays: list[np.ndarray] = []
        grid = None
        for column, (variant, title) in enumerate(zip(variants, titles)):
            payload = data.realism_cases[f"{case_id}:{variant}"]
            grid = payload["grid"]
            crop, x, y = _crop(payload["reference_intensity"], grid, style.realism_focus_halfwidth_m)
            arrays.append(crop)
            ax = fig.add_subplot(gs[row, column])
            _imshow(ax, _normalise(crop), x, y, cmap=style.intensity_cmap, vmin=0, vmax=1, interpolation="bicubic")
            ax.set(title=f"{case_id} {title}", xlabel="x (mm)", ylabel="y (mm)")
            centre_x, centre_y = _intensity_centroid(payload["reference_intensity"], grid)
            detail, detail_x, detail_y = _crop_at(
                payload["reference_intensity"],
                grid,
                style.sweep_focus_halfwidth_m,
                centre_x,
                centre_y,
            )
            detail, detail_x, detail_y = _display_resample_real(
                detail, detail_x, detail_y, style.scalar_display_resample_factor
            )
            inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
            _imshow(
                inset,
                _normalise(detail),
                detail_x,
                detail_y,
                cmap=style.intensity_cmap,
                vmin=0,
                vmax=1,
            )
            inset.set_title("beam-centred ROI", fontsize=6.2, color="white", pad=1.0)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("white")
                spine.set_linewidth(0.8)
            _panel(ax, panel_index, style)
            panel_index += 1
        ax = fig.add_subplot(gs[row, 3])
        delta = np.abs(_normalise(arrays[2]) - _normalise(arrays[1]))
        _imshow(ax, delta, x, y, cmap="magma", vmin=0, vmax=max(float(np.max(delta)), EPS), interpolation="bicubic")
        ax.set(title=f"{case_id} |degraded-realistic|", xlabel="x (mm)", ylabel="y (mm)")
        _panel(ax, panel_index, style)
        panel_index += 1
    profile_ax1 = fig.add_subplot(gs[2, 0:2])
    profile_ax2 = fig.add_subplot(gs[2, 2:4])
    for ax, case_id in ((profile_ax1, "V1"), (profile_ax2, "V3")):
        for variant, title, colour, line in zip(variants, titles, ("#0072B2", "#009E73", "#D55E00"), ("-", "--", ":")):
            payload = data.realism_cases[f"{case_id}:{variant}"]
            crop, x, _ = _crop(payload["reference_intensity"], payload["grid"], style.realism_focus_halfwidth_m)
            ax.plot(x / 1e-3, _centre_profile(crop), label=title, color=colour, linestyle=line)
        ax.set(title=f"{case_id} matched horizontal profiles", xlabel="x (mm)", ylabel="normalised intensity", xlim=(-style.realism_focus_halfwidth_m / 1e-3, style.realism_focus_halfwidth_m / 1e-3), ylim=(0, 1.05))
        ax.legend(frameon=False)
        _panel(ax, panel_index, style)
        panel_index += 1
    fig.suptitle("Vortex ideal, realistic, and degraded routes | matched crops; display interpolation only", fontsize=14.0)
    paths = _save(fig, root / "05_realism" / "hero_vortex_ideal_realistic_degraded", style)
    plt.close(fig)
    limits = (-style.realism_focus_halfwidth_m / 1e-3, style.realism_focus_halfwidth_m / 1e-3)
    return _record(
        "hero_vortex_ideal_realistic_degraded",
        paths,
        family="realism_degradation",
        role="hero_figure",
        case_ids=("V1", "V3"),
        source_artifacts=("Phase 2A accepted route definitions and endpoint table",),
        data_basis="in-memory endpoint reproduction of accepted Phase 2A route variants",
        normalisation="per-panel normalised",
        linear_log_mode="linear",
        x_unit="mm",
        y_unit="mm",
        x_limits=limits,
        y_limits=limits,
        comparison_group="vortex_realism",
        matched_axes=True,
        display_interpolation=(
            f"bicubic wide view plus cubic x{style.scalar_display_resample_factor} beam-centred ROI insets; display only"
        ),
        metric_bearing=True,
        notes=(
            "Endpoint metrics are computed on native arrays and reproduce Phase 2A. Wide matched crops preserve "
            f"the imposed displacement; common +/-{style.sweep_focus_halfwidth_m / 1e-3:.2f} mm beam-centred insets make morphology legible."
        ),
    )


def plot_scalar_route_atlas(
    data: Phase2EData,
    case_id: str,
    root: Path,
) -> dict[str, Any]:
    """Detailed ideal/realistic/mild/severe route comparison for one beam."""

    style = data.config.style
    plt, _ = _mpl(style)
    variants = (
        "ideal_optical_route",
        "realistic_fixed_bench_route",
        "mild_error_route",
        "deliberately_degraded_route",
    )
    titles = ("ideal", "realistic fixed bench", "mild combined error", "severe combined error")
    fig = plt.figure(figsize=(14.0, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.72))
    arrays: list[np.ndarray] = []
    payloads: list[Mapping[str, Any]] = []
    panel_index = 0
    for column, (variant, title) in enumerate(zip(variants, titles)):
        payload = data.realism_cases[f"{case_id}:{variant}"]
        payloads.append(payload)
        crop, x, y = _crop(
            payload["reference_intensity"],
            payload["grid"],
            style.realism_focus_halfwidth_m,
        )
        normalised = _normalise(crop)
        arrays.append(normalised)
        ax = fig.add_subplot(gs[0, column])
        _imshow(
            ax,
            normalised,
            x,
            y,
            cmap=style.intensity_cmap,
            vmin=0,
            vmax=1,
            interpolation="bicubic",
        )
        centre_x, centre_y = _intensity_centroid(
            payload["reference_intensity"], payload["grid"]
        )
        detail, detail_x, detail_y = _crop_at(
            payload["reference_intensity"],
            payload["grid"],
            style.sweep_focus_halfwidth_m,
            centre_x,
            centre_y,
        )
        detail, detail_x, detail_y = _display_resample_real(
            detail, detail_x, detail_y, style.scalar_display_resample_factor
        )
        inset = ax.inset_axes([0.54, 0.54, 0.43, 0.43])
        _imshow(
            inset,
            _normalise(detail),
            detail_x,
            detail_y,
            cmap=style.intensity_cmap,
            vmin=0,
            vmax=1,
        )
        inset.set_title("beam ROI", fontsize=6.5, color="white", pad=1.0)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("white")
            spine.set_linewidth(0.8)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        _panel(ax, panel_index, style)
        panel_index += 1

    ideal = arrays[0]
    for column, (values, title) in enumerate(zip(arrays, titles)):
        ax = fig.add_subplot(gs[1, column])
        delta = np.abs(values - ideal)
        _imshow(
            ax,
            delta,
            x,
            y,
            cmap="magma",
            vmin=0,
            vmax=max(float(np.max(delta)), EPS),
            interpolation="bicubic",
        )
        ax.set(title=f"|{title} - ideal|", xlabel="x (mm)", ylabel="y (mm)")
        _panel(ax, panel_index, style)
        panel_index += 1

    profile_ax = fig.add_subplot(gs[2, 0:2])
    colours = ("#0072B2", "#009E73", "#E69F00", "#D55E00")
    lines = ("-", "--", "-.", ":")
    for values, title, colour, line in zip(arrays, titles, colours, lines):
        profile_ax.plot(
            x / 1e-3,
            _centre_profile(values),
            label=title,
            color=colour,
            linestyle=line,
        )
    profile_ax.set(
        title="matched horizontal profiles",
        xlabel="x (mm)",
        ylabel="normalised intensity",
        xlim=(-style.realism_focus_halfwidth_m / 1e-3, style.realism_focus_halfwidth_m / 1e-3),
        ylim=(0, 1.05),
    )
    profile_ax.legend(frameon=False, ncol=2)
    _panel(profile_ax, panel_index, style)
    panel_index += 1

    metric_ax = fig.add_subplot(gs[2, 2:4])
    ring = [float(payload["row"]["dominant_off_axis_ring_radius_m"]) / 1e-6 for payload in payloads]
    centre = [float(payload["row"]["central_intensity_ratio"]) for payload in payloads]
    positions = np.arange(len(variants), dtype=float)
    metric_ax.bar(positions - 0.18, centre, width=0.36, color="#009E73", label="central / peak")
    twin = metric_ax.twinx()
    twin.bar(positions + 0.18, ring, width=0.36, color="#0072B2", alpha=0.75, label="ring radius")
    metric_ax.set_xticks(positions, labels=("ideal", "realistic", "mild", "severe"))
    metric_ax.set(title="accepted endpoint metrics", ylabel="central / peak", ylim=(0, 1.05))
    twin.set_ylabel("ring radius (um)")
    handles = [metric_ax.containers[0], twin.containers[0]]
    metric_ax.legend(handles, ["central / peak", "ring radius"], frameon=False)
    _panel(metric_ax, panel_index, style)

    fig.suptitle(
        f"{CASE_TITLES[case_id]} | ideal, realistic, mild-error, and severe-error routes",
        fontsize=14.0,
    )
    paths = _save(
        fig,
        root / "05_realism" / f"{case_id.lower()}_ideal_realistic_error_atlas",
        style,
    )
    plt.close(fig)
    limits = (
        -style.realism_focus_halfwidth_m / 1e-3,
        style.realism_focus_halfwidth_m / 1e-3,
    )
    return _record(
        f"{case_id.lower()}_ideal_realistic_error_atlas",
        paths,
        family="realism_degradation",
        role="main_text_candidate" if case_id in {"V1", "V3"} else "supplementary_candidate",
        case_ids=(case_id,),
        source_artifacts=("Phase 2A accepted route definitions and endpoint table",),
        data_basis="native accepted endpoint reproductions for all four governed route variants",
        normalisation="per-panel normalised with matched wide field",
        linear_log_mode="linear",
        x_unit="mm",
        y_unit="mm",
        x_limits=limits,
        y_limits=limits,
        comparison_group=f"{case_id.lower()}_route_realism",
        matched_axes=True,
        display_interpolation=(
            f"bicubic wide view plus cubic x{style.scalar_display_resample_factor} beam-centred ROI; display only"
        ),
        metric_bearing=True,
        notes="All route metrics come from native arrays and reproduce the accepted Phase 2A table; insets are display-only.",
    )


def plot_h1_correction(data: Phase2EData, root: Path) -> dict[str, Any]:
    style = data.config.style
    plt, _ = _mpl(style)
    wanted = ("ideal_sequential", "realistic_sequential", "degraded_axicon_0p5mm", "corrected_axicon_0p5mm")
    by_id = {str(case["route_id"]): case for case in data.hex_package.cross_route_cases}
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), constrained_layout=True)
    realistic = by_id["realistic_sequential"]
    real_crop, rx, ry = _crop(realistic["sas_intensity"], realistic["sas_grid"], style.h1_focus_halfwidth_m)
    real_crop, x, y = _display_resample_real(
        real_crop, rx, ry, style.h1_display_resample_factor
    )
    real_norm = _normalise(real_crop)
    for column, route_id in enumerate(wanted):
        case = by_id[route_id]
        crop, cx, cy = _crop(case["sas_intensity"], case["sas_grid"], style.h1_focus_halfwidth_m)
        crop, _, _ = _display_resample_real(
            crop, cx, cy, style.h1_display_resample_factor
        )
        normalised = _normalise(crop)
        _imshow(axes[0, column], normalised, x, y, cmap=style.intensity_cmap, vmin=0, vmax=1)
        axes[0, column].set_title(str(case["label"]))
        delta = np.abs(normalised - real_norm)
        _imshow(axes[1, column], delta, x, y, cmap="magma", vmin=0, vmax=1)
        axes[1, column].set_title("absolute difference vs realistic")
    for index, ax in enumerate(axes.flat):
        _panel(ax, index, style)
        ax.set(xlabel="x (mm)", ylabel="y (mm)")
    fig.suptitle("H1 accepted realism, 0.5 mm axicon offset, and bounded digital recentering", fontsize=14.0)
    paths = _save(fig, root / "05_realism" / "h1_before_after_correction", style)
    plt.close(fig)
    limits = (-style.h1_focus_halfwidth_m / 1e-3, style.h1_focus_halfwidth_m / 1e-3)
    return _record(
        "h1_before_after_correction",
        paths,
        family="realism_degradation",
        role="supplementary_candidate",
        case_ids=("H1_CONTINUOUS",),
        source_artifacts=("Phase 2B accepted cross-route package",),
        data_basis="physical SAS display arrays from accepted ideal/realistic/degraded/corrected routes",
        normalisation="per-panel normalised",
        linear_log_mode="linear",
        x_unit="mm",
        y_unit="mm",
        x_limits=limits,
        y_limits=limits,
        comparison_group="h1_correction",
        matched_axes=True,
        display_interpolation=f"cubic x{style.h1_display_resample_factor} after physical SAS; display only",
        metric_bearing=True,
        notes="Correction is the accepted bounded digital recentring example; it is not promoted to strict eligibility.",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_energy_summary(root: Path, style: ReportFigureStyle) -> dict[str, Any]:
    plt, _ = _mpl(style)
    rows = _read_csv(Path("outputs/validation/phase2a/canonical_power_ledgers.csv"))
    realistic = [row for row in rows if row["route_variant"] == "realistic_fixed_bench_route"]
    cases = ("G0", "B0", "V1", "V3", "H1")
    by_case = {case: [row for row in realistic if row["case_id"] == case] for case in cases}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for case, colour in zip(cases, ("#7A7A7A", "#0072B2", "#009E73", "#D55E00", "#CC79A7")):
        values = by_case[case]
        axes[0, 0].plot([int(row["row_index"]) for row in values], [float(row["cumulative_efficiency"]) for row in values], marker="o", label=case, color=colour)
    axes[0, 0].set(title="cumulative realistic-route efficiency", xlabel="ledger row", ylabel="fraction of input", ylim=(0, 1.03))
    axes[0, 0].legend(frameon=False, ncol=3)
    final = [float(by_case[case][-1]["cumulative_efficiency"]) for case in cases]
    axes[0, 1].bar(cases, final, color=("#7A7A7A", "#0072B2", "#009E73", "#D55E00", "#CC79A7"))
    axes[0, 1].set(title="final model-plane efficiency", ylabel="fraction of input", ylim=(0, max(final) * 1.25))
    h1 = by_case["H1"]
    losses = [1.0 - float(row["stage_efficiency"]) for row in h1]
    labels = [str(row["factor_id"]).replace("simulated_", "sim. ").replace("configured_", "cfg. ") for row in h1]
    sources = [str(row["source_of_efficiency"]) for row in h1]
    colours = ["#0072B2" if source == "simulated" else "#E69F00" for source in sources]
    positions = np.arange(len(losses))
    axes[1, 0].barh(positions, losses, color=colours)
    max_loss = max(max(losses), 0.01)
    for position, (loss, source) in enumerate(zip(losses, sources)):
        if abs(float(loss)) <= 1.0e-12:
            axes[1, 0].scatter([0.0], [position], marker="|", s=120, color="#222222", zorder=4)
            axes[1, 0].text(0.015 * max_loss, position, f"unity {source}", va="center", fontsize=7.0)
        elif float(loss) < 1.0e-3:
            axes[1, 0].scatter([float(loss)], [position], marker="|", s=120, color="#222222", zorder=4)
            axes[1, 0].text(
                0.015 * max_loss,
                position,
                f"<0.001 {source}",
                va="center",
                fontsize=7.0,
            )
        else:
            axes[1, 0].text(float(loss) + 0.012 * max_loss, position, f"{loss:.3f}", va="center", fontsize=7.0)
    axes[1, 0].set_yticks(np.arange(len(losses)), labels=labels)
    axes[1, 0].set(title="H1 one-pass loss | zero and sub-0.001 rows labelled", xlabel="1 - stage efficiency", xlim=(0, 1.18 * max_loss))
    from matplotlib.patches import Patch
    axes[1, 0].legend(
        handles=(Patch(color="#0072B2", label="simulated"), Patch(color="#E69F00", label="assumed")),
        frameon=False,
        loc="lower right",
    )
    first_order = [
        next(float(row["stage_efficiency"]) for row in by_case[case] if row["factor_id"] == "simulated_selected_first_order")
        for case in cases
    ]
    axes[1, 1].bar(cases, first_order, color=("#7A7A7A", "#0072B2", "#009E73", "#D55E00", "#CC79A7"))
    axes[1, 1].set(title="simulated selected-first-order fraction", ylabel="fraction", ylim=(0, 1.03))
    for index, ax in enumerate(axes.flat):
        _panel(ax, index, style)
    fig.suptitle("Canonical energy, loss, and first-order selection ledger | relative model accounting", fontsize=14.0)
    paths = _save(fig, root / "06_energy" / "hero_energy_loss_efficiency", style)
    plt.close(fig)
    return _record(
        "hero_energy_loss_efficiency",
        paths,
        family="energy_and_efficiency",
        role="hero_figure",
        case_ids=cases,
        source_artifacts=("outputs/validation/phase2a/canonical_power_ledgers.csv",),
        data_basis="accepted ledger CSV",
        normalisation="fractions of model input energy",
        linear_log_mode="linear",
        x_unit="stage or case",
        y_unit="fraction",
        comparison_group="canonical_energy",
        matched_axes=False,
        metric_bearing=True,
        notes=(
            "Exactly one simulated_selected_first_order factor appears per filtered ledger. Zero-height loss rows are "
            "explicit unity assumptions/placeholders, including unmeasured axicon transmission; simulated losses below "
            "0.001 are also labelled rather than left visually blank. No damage or power-rating claim."
        ),
    )


def plot_pedagogy_fundamentals(root: Path, style: ReportFigureStyle) -> dict[str, Any]:
    plt, _ = _mpl(style)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
    x = np.linspace(-3.0, 3.0, 600)
    gaussian = np.exp(-x**2)
    axes[0, 0].plot(x, gaussian, color="#0072B2")
    axes[0, 0].fill_between(x, gaussian, color="#56B4E9", alpha=0.25)
    axes[0, 0].set(title="Gaussian amplitude envelope", xlabel="transverse position", ylabel="amplitude")
    axes[0, 1].plot(x, np.mod(5.0 * np.abs(x), 2 * np.pi), color="#D55E00")
    axes[0, 1].set(title="conical phase ramp", xlabel="radius", ylabel="wrapped phase")
    for start in np.linspace(-2.4, 2.4, 9):
        axes[0, 2].annotate("", xy=(0, 2.4), xytext=(start, 0), arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.2})
    axes[0, 2].set(title="axicon: conical wavevectors", xlim=(-3, 3), ylim=(-0.2, 2.8), xlabel="transverse plane", ylabel="propagation")
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    for angle in theta:
        axes[1, 0].annotate("", xy=(0, 0), xytext=(np.cos(angle), np.sin(angle)), arrowprops={"arrowstyle": "->", "color": "#009E73", "lw": 1.1})
    axes[1, 0].set(title="conical components interfere", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
    X, Y = np.meshgrid(np.linspace(-2, 2, 220), np.linspace(-2, 2, 220))
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    vortex = (R**2) * np.exp(-R**2)
    axes[1, 1].imshow(vortex, origin="lower", extent=(-2, 2, -2, 2), cmap=style.intensity_cmap)
    for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        axes[1, 1].plot([0, 1.6 * np.cos(angle)], [0, 1.6 * np.sin(angle)], color=plt.cm.twilight((angle % (2*np.pi))/(2*np.pi)), lw=1.1)
    axes[1, 1].set(title="exp(i ell phi): winding and dark core", xlabel="x", ylabel="y")
    z = np.linspace(0, 1, 500)
    envelope = np.clip(1.0 - np.abs(z - 0.5) / 0.44, 0, 1)
    axes[1, 2].plot(z, envelope, color="#CC79A7")
    axes[1, 2].fill_between(z, envelope, color="#CC79A7", alpha=0.25)
    axes[1, 2].annotate("finite aperture", (0.08, 0.2), xytext=(0.02, 0.72), arrowprops={"arrowstyle": "->"})
    axes[1, 2].set(title="finite-energy, finite Bessel region", xlabel="z / model span", ylabel="overlap envelope", ylim=(0, 1.05))
    for index, ax in enumerate(axes.flat):
        _panel(ax, index, style)
    fig.suptitle("Structured-beam fundamentals | schematic, not quantitative", fontsize=14.0)
    paths = _save(fig, root / "07_pedagogy" / "pedagogy_bessel_vortex_fundamentals", style)
    plt.close(fig)
    return _record(
        "pedagogy_bessel_vortex_fundamentals",
        paths,
        family="pedagogical_schematic",
        role="main_text_candidate",
        case_ids=("G0", "B0", "V1", "V3"),
        source_artifacts=("analytic schematic construction",),
        data_basis="schematic only",
        normalisation="schematic",
        linear_log_mode="linear",
        x_unit="conceptual",
        y_unit="conceptual",
        metric_bearing=False,
        metrics_native=False,
        notes="Covers Gaussian-to-conical phase, axicon wavevectors, interference, vortex dark-core creation, and finite truncation.",
    )


def plot_pedagogy_h1_orientation(root: Path, style: ReportFigureStyle) -> dict[str, Any]:
    plt, _ = _mpl(style)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True)
    gx = np.linspace(-1, 1, 17)
    X, Y = np.meshgrid(gx, gx)
    R = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    mask = (R > 0.18) & (R < 1.0)
    continuous = theta
    sector = np.floor(np.mod(theta, 2 * np.pi) / (np.pi / 3.0))
    centres = (sector + 0.5) * np.pi / 3.0
    for ax, alpha, title in (
        (axes[0], continuous, "continuous local radial orientation"),
        (axes[1], centres, "sector-averaged surrogate"),
    ):
        ax.quiver(X[mask], Y[mask], np.cos(alpha[mask]), np.sin(alpha[mask]), angles="xy", scale_units="xy", scale=9.0, pivot="middle", color="#243B64", width=0.006)
        circle = plt.Circle((0, 0), 1.0, fill=False, color="#D55E00", lw=1.4)
        ax.add_patch(circle)
        ax.set(title=title, xlabel="x", ylabel="y", xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal")
    _panel(axes[0], 0, style)
    _panel(axes[1], 1, style)
    fig.suptitle("H1 local orientation: continuous field versus six-sector surrogate", fontsize=14.0)
    paths = _save(fig, root / "07_pedagogy" / "pedagogy_h1_local_orientation", style)
    plt.close(fig)
    return _record(
        "pedagogy_h1_local_orientation",
        paths,
        family="pedagogical_schematic",
        role="main_text_candidate",
        case_ids=("H1_CONTINUOUS", "H1_AVERAGED"),
        source_artifacts=("MODE 2Y orientation convention",),
        data_basis="schematic headless line orientations",
        normalisation="not_applicable",
        linear_log_mode="not_applicable",
        x_unit="conceptual",
        y_unit="conceptual",
        metric_bearing=False,
        metrics_native=False,
    )


def _box(ax: Any, x: float, y: float, text: str, *, colour: str = "#E8EEF7", width: float = 0.115) -> None:
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch((x - width / 2, y - 0.08), width, 0.16, boxstyle="round,pad=0.01,rounding_size=0.01", facecolor=colour, edgecolor="#243B64", lw=1.1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.2)


def plot_pedagogy_bench(root: Path, style: ReportFigureStyle) -> dict[str, Any]:
    plt, _ = _mpl(style)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    ax = axes[0]
    labels = ("laser", "HWP", "SLM-H", "SLM-V", "4F", "QWP", "axicon", "camera")
    xs = np.linspace(0.07, 0.93, len(labels))
    for x, label in zip(xs, labels):
        _box(ax, float(x), 0.5, label, colour="#E8EEF7" if "SLM" not in label else "#FCE8D5", width=0.09)
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right - 0.05, 0.5), xytext=(left + 0.05, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#243B64"})
    ax.set(title="sequential dual-SLM + common 4F route", xlim=(0, 1), ylim=(0.2, 0.8))
    ax.axis("off")
    ax = axes[1]
    ax.axvline(0.0, color="#7A7A7A", lw=1.0)
    ax.scatter((0.0, 1.929), (0.0, 0.0), s=(120, 150), c=("#7A7A7A", "#D55E00"), zorder=3)
    iris = plt.Circle((1.929, 0.0), 0.772, fill=False, color="#0072B2", lw=2.0)
    ax.add_patch(iris)
    ax.annotate("zero order", (0, 0), xytext=(-0.5, 0.8), arrowprops={"arrowstyle": "->"})
    ax.annotate("selected +1", (1.929, 0), xytext=(2.2, 0.8), arrowprops={"arrowstyle": "->"})
    ax.set(title="Fourier plane iris selection", xlabel="Fourier-plane x (mm)", ylabel="y (mm)", xlim=(-1.2, 3.2), ylim=(-1.4, 1.4), aspect="equal")
    ax = axes[2]
    pupil = plt.Circle((0, 0), 1.0, fill=False, color="#0072B2", lw=2.0)
    ax.add_patch(pupil)
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ax.annotate("", xy=(0, 0), xytext=(np.cos(angle), np.sin(angle)), arrowprops={"arrowstyle": "->", "color": "#009E73", "lw": 1.0})
    ax.text(0, -1.28, "pupil spectrum -> objective focus", ha="center")
    ax.set(title="objective / focal mapping concept", xlim=(-1.4, 1.4), ylim=(-1.5, 1.25), aspect="equal")
    ax.axis("off")
    for index, ax in enumerate(axes):
        _panel(ax, index, style)
    fig.suptitle("Optical route and spatial filtering | schematic dimensions from canonical manifest", fontsize=14.0)
    paths = _save(fig, root / "07_pedagogy" / "pedagogy_sequential_route_fourier_objective", style)
    plt.close(fig)
    return _record(
        "pedagogy_sequential_route_fourier_objective",
        paths,
        family="pedagogical_schematic",
        role="main_text_candidate",
        case_ids=("G0", "B0", "V1", "V3", "H1_CONTINUOUS"),
        source_artifacts=("Phase 2A canonical hardware manifest", "docs/78 Nathan hardware closure"),
        data_basis="schematic with canonical 1029 nm/300 mm/6.25 lp-mm Fourier-plane dimensions",
        normalisation="not_applicable",
        linear_log_mode="not_applicable",
        x_unit="schematic; Fourier panel in mm",
        y_unit="schematic; Fourier panel in mm",
        metric_bearing=False,
        metrics_native=False,
    )


def plot_vortex_family_hero(data: Phase2EData, root: Path) -> dict[str, Any]:
    style = data.config.style
    plt, _ = _mpl(style)
    order = ("G0", "B0", "V1", "V3")
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.5), constrained_layout=True)
    for column, case_id in enumerate(order):
        case = data.scalar_cases[case_id]
        focus, x, y = _crop(case.focus_field, case.focus_grid, style.scalar_focus_halfwidth_m)
        focus, x, y = _display_resample_complex(
            focus, x, y, style.scalar_display_resample_factor
        )
        intensity = np.abs(focus) ** 2
        _imshow(axes[0, column], _normalise(intensity), x, y, cmap=style.intensity_cmap, vmin=0, vmax=1)
        axes[0, column].set(title=CASE_TITLES[case_id], xlabel="x (mm)", ylabel="y (mm)")
        phase = np.angle(focus)
        phase = np.ma.masked_where(_normalise(intensity) < 0.02, phase)
        _imshow(axes[1, column], phase, x, y, cmap=style.phase_cmap, vmin=-np.pi, vmax=np.pi)
        axes[1, column].set(title="focal phase", xlabel="x (mm)", ylabel="y (mm)")
    for index, ax in enumerate(axes.flat):
        _panel(ax, index, style)
    fig.suptitle("Canonical scalar beam family at z=60 mm | matched SAS crop and colour limits", fontsize=14.0)
    paths = _save(fig, root / "08_hero_figures" / "hero_vortex_beam_family", style)
    plt.close(fig)
    limits = (-style.scalar_focus_halfwidth_m / 1e-3, style.scalar_focus_halfwidth_m / 1e-3)
    return _record(
        "hero_vortex_beam_family",
        paths,
        family="report_hero",
        role="hero_figure",
        case_ids=order,
        source_artifacts=("Phase 2A/2B accepted scalar family",),
        data_basis="physical SAS complex focus fields reconstructed from accepted routes",
        normalisation="per-panel normalised",
        linear_log_mode="linear",
        x_unit="mm",
        y_unit="mm",
        x_limits=limits,
        y_limits=limits,
        comparison_group="scalar_family_focus",
        matched_axes=True,
        display_interpolation=f"complex cubic x{style.scalar_display_resample_factor} after physical SAS; display only",
        metric_bearing=False,
    )


def plot_parameter_hero(data: Phase2EData, root: Path) -> dict[str, Any]:
    style = data.config.style
    plt, _ = _mpl(style)
    chosen = ("vortex_charge", "radial_wavevector", "input_beam_radius", "propagation_distance")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)
    for index, (ax, sweep_id) in enumerate(zip(axes.flat, chosen)):
        planes = data.sweep_planes[sweep_id]
        x_values, x_label = _sweep_x_values(planes)
        ring = np.asarray([plane.metrics["ring_radius_m"] for plane in planes]) / 1e-3
        central = np.asarray([plane.metrics["central_intensity_ratio"] for plane in planes])
        ax.plot(x_values, ring, marker="o", color="#0072B2", label="ring radius (mm)")
        twin = ax.twinx()
        twin.plot(x_values, central, marker="s", color="#D55E00", label="central / peak")
        ax.set(title=sweep_id.replace("_", " "), xlabel=x_label, ylabel="radius (mm)")
        twin.set_ylabel("central / peak")
        _panel(ax, index, style)
    fig.suptitle("Vortex parameter dependence | Phase 2E diagnostic screening, not fixed-bench claims", fontsize=14.0)
    paths = _save(fig, root / "08_hero_figures" / "hero_vortex_parameter_dependence", style)
    plt.close(fig)
    return _record(
        "hero_vortex_parameter_dependence",
        paths,
        family="report_hero",
        role="hero_figure",
        case_ids=("V1",),
        source_artifacts=("Phase 2E diagnostic screening summary",),
        data_basis="native SAS diagnostic metrics",
        normalisation="native metric ratios",
        linear_log_mode="linear",
        x_unit="parameter-specific SI-derived display unit",
        y_unit="mm and ratio",
        metric_bearing=True,
        notes="Visual trend summary only; accepted fixed-bench and vector results retain authority.",
    )


def plot_scalar_vector_hero(root: Path, style: ReportFigureStyle) -> dict[str, Any]:
    plt, _ = _mpl(style)
    import matplotlib.image as mpimg

    sources = (
        Path("outputs/figures/phase2c/objective/v1_scalar_vs_debye.png"),
        Path("outputs/figures/phase2c/objective/v3_scalar_vs_debye.png"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(15.0, 9.0), constrained_layout=True)
    for index, (ax, source, title) in enumerate(zip(axes, sources, ("V1 scalar versus vector Debye", "V3 scalar versus vector Debye"))):
        ax.imshow(mpimg.imread(source), interpolation="none")
        ax.set_title(title, pad=2.0)
        ax.axis("off")
        _panel(ax, index, style)
    fig.suptitle("Objective benchmark plate | accepted Phase 2C figures reproduced without pixel-derived metrics", fontsize=14.0)
    paths = _save(fig, root / "08_hero_figures" / "hero_scalar_vector_objective_benchmark", style)
    plt.close(fig)
    return _record(
        "hero_scalar_vector_objective_benchmark",
        paths,
        family="report_hero",
        role="hero_figure",
        case_ids=("V1", "V3"),
        source_artifacts=tuple(path.as_posix() for path in sources),
        data_basis="accepted Phase 2C raster figures, presentation-only composite",
        normalisation="as declared by accepted Phase 2C source figures",
        linear_log_mode="linear",
        x_unit="um inside source panels",
        y_unit="um inside source panels",
        comparison_group="phase2c_objective",
        matched_axes=True,
        display_interpolation="none",
        metric_bearing=False,
        metrics_native=False,
        notes=(
            "No value is read from raster pixels; quantitative values remain in Phase 2C CSV files. "
            "The accepted wide source rasters are stacked vertically so each objective ROI remains report-legible."
        ),
    )


def generate_phase2e_figures(data: Phase2EData, root: Path) -> list[dict[str, Any]]:
    """Generate every required Phase 2E figure family."""

    style = data.config.style
    records: list[dict[str, Any]] = []
    for case_id in ("G0", "B0", "V1", "V3"):
        reference_id = "B0" if case_id != "B0" else "G0"
        records.append(
            plot_scalar_core_case(
                data.scalar_cases[case_id],
                data.scalar_cases[reference_id],
                root,
                style,
            )
        )
    for case_id in ("G0", "B0", "V1", "V3", "H1_CONTINUOUS", "H1_AVERAGED"):
        records.append(plot_dense_propagation_atlas(data, case_id, root))
    records.append(plot_b0_propagation_boundary_audit(data, root))
    for case_id in PHASE2E_3D_CASE_IDS:
        records.append(plot_intensity_surface(data, case_id, root))
    records.append(plot_h1_matched_comparison(data, root))
    records.append(plot_h1_polarisation(data, root))
    for sweep_id in (
        "vortex_charge",
        "radial_wavevector",
        "input_beam_radius",
        "propagation_distance",
        "aperture_radius",
        "effective_objective_na",
        "defocus_aberration",
        "error_input_beam_decentre",
        "error_input_beam_tilt",
        "error_slm_phase",
        "error_fourier_iris_offset",
        "error_pupil_decentre",
        "error_axicon_decentre",
        "error_zernike_defocus",
        "error_zernike_astigmatism",
        "error_zernike_coma",
        "error_zernike_spherical",
    ):
        records.append(plot_sweep(data.sweep_planes[sweep_id], root, style))
    records.append(plot_vortex_realism(data, root))
    for case_id in ("B0", "V1", "V3"):
        records.append(plot_scalar_route_atlas(data, case_id, root))
    records.append(plot_h1_correction(data, root))
    records.append(plot_energy_summary(root, style))
    records.append(plot_pedagogy_fundamentals(root, style))
    records.append(plot_pedagogy_h1_orientation(root, style))
    records.append(plot_pedagogy_bench(root, style))
    records.append(plot_vortex_family_hero(data, root))
    records.append(plot_parameter_hero(data, root))
    records.append(plot_scalar_vector_hero(root, style))
    return records


__all__ = ["generate_phase2e_figures"]
