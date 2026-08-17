"""Publication figures for the MODE 2Y propagation audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_local_vector_truth import (
    evaluate_local_vector_truth,
    line_orientation_error,
)
from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import (
    EPS,
    Mode2YStudyResult,
    PropagationRouteResult,
    _radial_profile_native,
    _selected_key,
)


def _display_ring_profile(
    image: np.ndarray,
    grid: Mapping[str, Any],
    ring_radius_m: float,
    *,
    angular_samples: int = 720,
    radial_samples: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a smooth display-only ring trace without changing native metrics."""

    from scipy.ndimage import gaussian_filter1d, map_coordinates

    arr = np.asarray(image, dtype=float)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    dx = float(grid["dx"])
    half_width = max(0.25 * float(ring_radius_m), dx)
    radii = np.linspace(
        max(0.0, float(ring_radius_m) - half_width),
        float(ring_radius_m) + half_width,
        int(radial_samples),
    )[:, None]
    theta = np.linspace(0.0, 2.0 * np.pi, int(angular_samples), endpoint=False)
    xs = radii * np.cos(theta)[None, :]
    ys = radii * np.sin(theta)[None, :]
    columns = (xs - float(x[0])) / dx
    rows = (ys - float(y[0])) / dx
    sampled = map_coordinates(arr, [rows, columns], order=1, mode="nearest")
    profile = np.mean(sampled, axis=0)
    profile = gaussian_filter1d(profile, sigma=2.0, mode="wrap")
    return theta, np.maximum(profile, 0.0)


def _mpl() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.rcParams.update({
        "font.size": 9.0,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
    return plt, LineCollection


def _save(fig: Any, stem: Path, *, dpi: int = 320) -> tuple[Path, Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    return png, pdf


def _extent_mm(grid: Mapping[str, Any], sy: slice | None = None, sx: slice | None = None) -> list[float]:
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    if sx is not None:
        x = x[sx]
    if sy is not None:
        y = y[sy]
    return [float(x[0] / 1e-3), float(x[-1] / 1e-3), float(y[0] / 1e-3), float(y[-1] / 1e-3)]


def _focus_slices(grid: Mapping[str, Any], ring_radius_m: float) -> tuple[slice, slice]:
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    half = max(3.6 * float(ring_radius_m), 0.65e-3)
    xi = np.where(np.abs(x) <= half)[0]
    yi = np.where(np.abs(y) <= half)[0]
    return slice(int(yi[0]), int(yi[-1]) + 1), slice(int(xi[0]), int(xi[-1]) + 1)


def _normalise_peak(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    return arr / max(float(np.max(arr)), EPS)


def _equal_power(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    return arr / max(float(np.sum(arr)), EPS)


def _line_segments(
    alpha: np.ndarray,
    extent: Sequence[float],
    *,
    samples: int = 23,
) -> np.ndarray:
    ny, nx = alpha.shape
    yi = np.linspace(0, ny - 1, samples, dtype=int)
    xi = np.linspace(0, nx - 1, samples, dtype=int)
    xx, yy = np.meshgrid(xi, yi)
    angles = np.asarray(alpha, dtype=float)[yy, xx]
    xcoord = np.interp(xx, [0, nx - 1], [extent[0], extent[1]])
    ycoord = np.interp(yy, [0, ny - 1], [extent[2], extent[3]])
    length = 0.070 * min(float(extent[1] - extent[0]), float(extent[3] - extent[2]))
    dx = 0.5 * length * np.cos(angles)
    dy = 0.5 * length * np.sin(angles)
    return np.stack(
        [np.stack([xcoord - dx, ycoord - dy], axis=-1), np.stack([xcoord + dx, ycoord + dy], axis=-1)],
        axis=-2,
    ).reshape(-1, 2, 2)


def _overlay_lines(ax: Any, alpha: np.ndarray, extent: Sequence[float], *, color: str = "white") -> None:
    _, LineCollection = _mpl()
    segments = _line_segments(alpha, extent)
    finite = np.all(np.isfinite(segments), axis=(1, 2))
    ax.add_collection(LineCollection(segments[finite], colors=color, linewidths=0.65, alpha=0.88))


def _input_truth_results(result: Mode2YStudyResult) -> tuple[Any, Any]:
    grid = result.data["grid"]
    cfg = result.data["config"]
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    continuous = evaluate_local_vector_truth(
        "continuous",
        result.inputs.continuous_ex,
        result.inputs.continuous_ey,
        x,
        y,
        result.inputs.continuous_alpha_rad,
        sector_rotation_rad=float(cfg.sector_rotation_rad),
        gate_class="ideal",
    )
    averaged = evaluate_local_vector_truth(
        "sector_averaged",
        result.inputs.averaged_ex,
        result.inputs.averaged_ey,
        x,
        y,
        result.inputs.continuous_alpha_rad,
        sector_rotation_rad=float(cfg.sector_rotation_rad),
        gate_class="ideal",
    )
    return continuous, averaged


def plot_pre_axicon_comparison(result: Mode2YStudyResult, root: Path) -> tuple[Path, Path]:
    plt, _ = _mpl()
    grid = result.data["grid"]
    continuous, averaged = _input_truth_results(result)
    sy, sx = _focus_slices(grid, 0.75e-3)
    extent = _extent_mm(grid, sy, sx)
    s0 = np.abs(result.inputs.continuous_ex) ** 2 + np.abs(result.inputs.continuous_ey) ** 2
    angle_delta = np.abs(line_orientation_error(
        result.inputs.averaged_alpha_rad,
        result.inputs.continuous_alpha_rad,
    ))
    continuous_purity = np.where(
        continuous.radial_sector_mask,
        continuous.basis_fields.radial_power_fraction,
        continuous.basis_fields.azimuthal_power_fraction,
    )
    averaged_purity = np.where(
        continuous.radial_sector_mask,
        averaged.basis_fields.radial_power_fraction,
        averaged.basis_fields.azimuthal_power_fraction,
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.5), constrained_layout=True)
    for ax, alpha, title in (
        (axes[0, 0], result.inputs.continuous_alpha_rad, "Continuous local field"),
        (axes[0, 1], result.inputs.averaged_alpha_rad, "One line per sector surrogate"),
    ):
        image = ax.imshow(s0[sy, sx], origin="lower", extent=extent, cmap="magma", interpolation="bicubic")
        _overlay_lines(ax, alpha[sy, sx], extent)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.72, label="S0")
    image = axes[0, 2].imshow(angle_delta[sy, sx], origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=np.pi / 6.0, interpolation="bicubic")
    axes[0, 2].set_title(r"Headless local-angle difference $|\Delta\alpha|$")
    fig.colorbar(image, ax=axes[0, 2], shrink=0.72, label="rad")
    for ax, array, title in (
        (axes[1, 0], continuous_purity, "Continuous desired-basis purity"),
        (axes[1, 1], averaged_purity, "Averaged desired-basis purity"),
        (axes[1, 2], continuous_purity - averaged_purity, "Purity difference (continuous - averaged)"),
    ):
        diverging = "coolwarm" if "difference" in title.lower() else "viridis"
        limits = {"vmin": -1.0, "vmax": 1.0} if diverging == "coolwarm" else {"vmin": 0.0, "vmax": 1.0}
        image = ax.imshow(array[sy, sx], origin="lower", extent=extent, cmap=diverging, interpolation="bicubic", **limits)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.72)
    for ax in axes.ravel():
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.suptitle("MODE 2Y input difference: continuous truth vs sector-average surrogate\nNative metrics; interpolation is display-only", fontsize=15)
    paths = _save(fig, root / "00_inputs/pre_axicon_continuous_vs_averaged")
    plt.close(fig)
    return paths


def _pair_routes(result: Mode2YStudyResult, optical_route: str) -> tuple[PropagationRouteResult, PropagationRouteResult]:
    if optical_route == "ideal":
        return result.routes["ideal_continuous"], result.routes["ideal_sector_averaged"]
    return result.routes["realistic_continuous_common_4f"], result.routes["realistic_sector_averaged_common_4f"]


def plot_xy_plane_grid(
    result: Mode2YStudyResult,
    root: Path,
    optical_route: str,
    *,
    focus: bool,
) -> tuple[Path, Path]:
    plt, _ = _mpl()
    continuous, averaged = _pair_routes(result, optical_route)
    grid = result.data["grid"]
    if focus:
        sy, sx = _focus_slices(grid, float(result.data["mode2y_ring_radius_m"]))
    else:
        sy, sx = slice(None), slice(None)
    extent = _extent_mm(grid, sy, sx)
    selected = result.config.selected_z_m
    fig, axes = plt.subplots(3, len(selected), figsize=(23.0, 9.0), constrained_layout=True)
    for column, z_m in enumerate(selected):
        key = _selected_key(float(z_m))
        c = np.asarray(continuous.selected_planes[key], dtype=float)[sy, sx]
        a = np.asarray(averaged.selected_planes[key], dtype=float)[sy, sx]
        c_eq = _equal_power(c)
        a_eq = _equal_power(a)
        panels = (_normalise_peak(c), _normalise_peak(a), np.abs(c_eq - a_eq) / max(float(np.max(np.abs(c_eq - a_eq))), EPS))
        for row, panel in enumerate(panels):
            image = axes[row, column].imshow(panel, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0, interpolation="bicubic")
            if row == 0:
                axes[row, column].set_title(f"z = {z_m / 1e-3:.0f} mm")
            if column == 0:
                axes[row, column].set_ylabel(("continuous", "sector averaged", "|equal-power difference|")[row] + "\ny (mm)")
            axes[row, column].set_xlabel("x (mm)")
        if column == len(selected) - 1:
            fig.colorbar(image, ax=axes[:, column], shrink=0.68, label="normalised display")
    scale = "focus crop" if focus else "wide field"
    fig.suptitle(f"{optical_route.capitalize()} route XY propagation: continuous vs averaged ({scale})", fontsize=15)
    paths = _save(fig, root / f"01_xy_planes/{optical_route}_xy_{'focus' if focus else 'wide'}")
    plt.close(fig)
    return paths


def plot_focused_hexagon(result: Mode2YStudyResult, root: Path, optical_route: str) -> tuple[Path, Path]:
    plt, _ = _mpl()
    continuous, averaged = _pair_routes(result, optical_route)
    grid = result.data["grid"]
    sy, sx = _focus_slices(grid, float(result.data["mode2y_ring_radius_m"]))
    extent = _extent_mm(grid, sy, sx)
    rows = (
        (_selected_key(60e-3), "z = 60 mm"),
        ("continuous_best_z", f"continuous best z = {continuous.best_z_m / 1e-3:.0f} mm"),
        ("averaged_best_z", f"averaged best z = {averaged.best_z_m / 1e-3:.0f} mm"),
    )
    fig, axes = plt.subplots(3, 6, figsize=(20.0, 10.8), constrained_layout=True)
    titles = ("continuous", "averaged", "|difference|", "continuous log", "averaged log", "|difference| log")
    for row, (key, label) in enumerate(rows):
        c = np.asarray(continuous.selected_planes[key], dtype=float)[sy, sx]
        a = np.asarray(averaged.selected_planes[key], dtype=float)[sy, sx]
        c = _normalise_peak(c)
        a = _normalise_peak(a)
        difference = np.abs(_equal_power(c) - _equal_power(a))
        difference = difference / max(float(np.max(difference)), EPS)
        panels = (
            c,
            a,
            difference,
            10.0 * np.log10(np.maximum(c, 1e-6)),
            10.0 * np.log10(np.maximum(a, 1e-6)),
            10.0 * np.log10(np.maximum(difference, 1e-6)),
        )
        for column, panel in enumerate(panels):
            is_log = column >= 3
            image = axes[row, column].imshow(
                panel,
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=-45.0 if is_log else 0.0,
                vmax=0.0 if is_log else 1.0,
                interpolation="bicubic",
            )
            if row == 0:
                axes[row, column].set_title(titles[column])
            if column == 0:
                axes[row, column].set_ylabel(label + "\ny (mm)")
            axes[row, column].set_xlabel("x (mm)")
        fig.colorbar(image, ax=axes[row, -1], shrink=0.62, label="dB")
    fig.suptitle(f"{optical_route.capitalize()} focused hexagon comparison", fontsize=15)
    paths = _save(fig, root / f"02_focus_crops/{optical_route}_focused_hexagon_comparison")
    plt.close(fig)
    return paths


def _normalise_rows(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    maxima = np.max(arr, axis=1, keepdims=True)
    return arr / np.maximum(maxima, EPS)


def plot_propagation_maps(result: Mode2YStudyResult, root: Path, optical_route: str) -> tuple[Path, Path]:
    plt, _ = _mpl()
    continuous, averaged = _pair_routes(result, optical_route)
    grid = result.data["grid"]
    x = np.asarray(grid["x"], dtype=float)
    half = max(4.5 * float(result.data["mode2y_ring_radius_m"]), 1.0e-3)
    indices = np.where(np.abs(x) <= half)[0]
    sx = slice(int(indices[0]), int(indices[-1]) + 1)
    extent = [float(x[indices[0]] / 1e-3), float(x[indices[-1]] / 1e-3), float(result.config.z_end_m / 1e-3), float(result.config.z_start_m / 1e-3)]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), constrained_layout=True)
    for row, (c_map, a_map, axis_name) in enumerate((
        (continuous.xz_map, averaged.xz_map, "x"),
        (continuous.yz_map, averaged.yz_map, "y"),
    )):
        c = _normalise_rows(np.asarray(c_map, dtype=float)[:, sx])
        a = _normalise_rows(np.asarray(a_map, dtype=float)[:, sx])
        difference = np.abs(c - a)
        for column, (panel, title) in enumerate(((c, "continuous"), (a, "averaged"), (difference, "|difference|"))):
            image = axes[row, column].imshow(panel, origin="upper", aspect="auto", extent=extent, cmap="magma", vmin=0.0, vmax=1.0, interpolation="bicubic")
            axes[row, column].set_title(f"{axis_name}-z {title}")
            axes[row, column].set_xlabel(f"{axis_name} (mm)")
            axes[row, column].set_ylabel("z (mm)")
        fig.colorbar(image, ax=axes[row, -1], shrink=0.72)
    fig.suptitle(f"{optical_route.capitalize()} route propagation persistence", fontsize=15)
    paths = _save(fig, root / f"03_propagation_maps/{optical_route}_xz_yz_comparison")
    plt.close(fig)
    return paths


def plot_profiles(result: Mode2YStudyResult, root: Path, optical_route: str) -> tuple[Path, Path]:
    plt, _ = _mpl()
    continuous, averaged = _pair_routes(result, optical_route)
    grid = result.data["grid"]
    ring_radius = float(result.data["mode2y_ring_radius_m"])
    x_mm = np.asarray(grid["x"], dtype=float) / 1e-3
    z_mm = np.asarray(continuous.z_values_m, dtype=float) / 1e-3
    c60 = np.asarray(continuous.selected_planes[_selected_key(60e-3)], dtype=float)
    a60 = np.asarray(averaged.selected_planes[_selected_key(60e-3)], dtype=float)
    mid = c60.shape[0] // 2
    radial_r, radial_c = _radial_profile_native(c60, grid)
    _, radial_a = _radial_profile_native(a60, grid)
    theta, angular_c = _display_ring_profile(c60, grid, ring_radius)
    _, angular_a = _display_ring_profile(a60, grid, ring_radius)
    focus_half_mm = max(3.6 * ring_radius / 1e-3, 0.65)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), constrained_layout=True)
    axes[0, 0].plot(x_mm, c60[mid] / max(float(np.max(c60[mid])), EPS), label="continuous x", color="#0072B2")
    axes[0, 0].plot(x_mm, a60[mid] / max(float(np.max(a60[mid])), EPS), label="averaged x", color="#D55E00")
    axes[0, 0].plot(x_mm, c60[:, mid] / max(float(np.max(c60[:, mid])), EPS), "--", label="continuous y", color="#0072B2")
    axes[0, 0].plot(x_mm, a60[:, mid] / max(float(np.max(a60[:, mid])), EPS), "--", label="averaged y", color="#D55E00")
    axes[0, 0].set(xlabel="position (mm)", ylabel="normalised intensity", title="Central x/y profiles at z=60 mm")
    axes[0, 0].set_xlim(-focus_half_mm, focus_half_mm)
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[0, 1].plot(radial_r / 1e-3, radial_c / max(float(np.max(radial_c)), EPS), label="continuous", color="#0072B2")
    axes[0, 1].plot(radial_r / 1e-3, radial_a / max(float(np.max(radial_a)), EPS), label="averaged", color="#D55E00")
    axes[0, 1].set(xlabel="radius (mm)", ylabel="normalised intensity", title="Native radial profile at z=60 mm")
    axes[0, 1].set_xlim(0.0, focus_half_mm)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(np.rad2deg(theta), angular_c / max(float(np.max(angular_c)), EPS), label="continuous", color="#0072B2")
    axes[1, 0].plot(np.rad2deg(theta), angular_a / max(float(np.max(angular_a)), EPS), label="averaged", color="#D55E00")
    axes[1, 0].set(
        xlabel="azimuth (deg)",
        ylabel="normalised ring intensity",
        title="Bilinear display profile on V0 ring at z=60 mm",
    )
    axes[1, 0].legend(frameon=False)
    c_score = [metric.sharpness_composite for metric in continuous.z_metrics]
    a_score = [metric.sharpness_composite for metric in averaged.z_metrics]
    axes[1, 1].plot(z_mm, c_score, label="continuous", color="#0072B2")
    axes[1, 1].plot(z_mm, a_score, label="averaged", color="#D55E00")
    axes[1, 1].axvline(60.0, color="0.35", linestyle="--", linewidth=0.9)
    axes[1, 1].set(xlabel="z (mm)", ylabel="sharpness composite", title="Native sharpness metric versus z")
    axes[1, 1].legend(frameon=False)
    for ax in axes.ravel():
        ax.grid(alpha=0.22)
    fig.suptitle(f"{optical_route.capitalize()} route profiles and sharpness", fontsize=15)
    paths = _save(fig, root / f"04_profiles/{optical_route}_profile_comparison")
    plt.close(fig)
    return paths


def plot_metric_dashboard(result: Mode2YStudyResult, root: Path) -> tuple[Path, Path]:
    plt, _ = _mpl()
    rows = list(result.summary_rows)
    labels = ["ideal\ncontinuous", "ideal\naveraged", "realistic\ncontinuous", "realistic\naveraged"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), constrained_layout=True)
    panels = (
        ("z60_correlation_to_v0", "z60 correlation to V0", None),
        ("edge_gradient_sharpness_mm_inv", "Edge-gradient sharpness (mm$^{-1}$)", None),
        ("threshold_transition_width_mm", "80%-20% transition width (mm)", None),
        ("corner_concentration", "Corner concentration", None),
        ("bright_ridge_fwhm_mm", "Bright-ridge FWHM (mm)", None),
        ("dark_core_ratio", "Dark-core ratio", "log"),
    )
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for ax, (key, title, scale) in zip(axes.ravel(), panels):
        values = [float(row[key]) for row in rows]
        ax.bar(x, values, color=colours)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        if scale:
            ax.set_yscale(scale)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(f"MODE 2Y metric dashboard - outcome {result.outcome}\n{result.outcome_reason}", fontsize=15)
    paths = _save(fig, root / "05_metrics/continuous_vs_averaged_metric_dashboard")
    plt.close(fig)
    return paths


def write_mode2y_figures(result: Mode2YStudyResult, output_root: str | Path) -> dict[str, Path]:
    """Write every required MODE 2Y PNG/PDF figure pair."""

    root = Path(output_root)
    paths: dict[str, Path] = {}
    for name, generated in (("pre_axicon", plot_pre_axicon_comparison(result, root)),):
        paths[f"{name}_png"], paths[f"{name}_pdf"] = generated
    for route in ("ideal", "realistic"):
        for focus in (False, True):
            generated = plot_xy_plane_grid(result, root, route, focus=focus)
            key = f"{route}_xy_{'focus' if focus else 'wide'}"
            paths[f"{key}_png"], paths[f"{key}_pdf"] = generated
        generated = plot_focused_hexagon(result, root, route)
        paths[f"{route}_focused_png"], paths[f"{route}_focused_pdf"] = generated
        generated = plot_propagation_maps(result, root, route)
        paths[f"{route}_propagation_png"], paths[f"{route}_propagation_pdf"] = generated
        generated = plot_profiles(result, root, route)
        paths[f"{route}_profiles_png"], paths[f"{route}_profiles_pdf"] = generated
    generated = plot_metric_dashboard(result, root)
    paths["dashboard_png"], paths["dashboard_pdf"] = generated
    return paths


__all__ = [
    "plot_focused_hexagon",
    "plot_metric_dashboard",
    "plot_pre_axicon_comparison",
    "plot_profiles",
    "plot_propagation_maps",
    "plot_xy_plane_grid",
    "write_mode2y_figures",
]
