"""Publication figures for the MODE 2Z orientation-fidelity sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_local_vector_truth import line_orientation_error
from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import EPS, _selected_key
from vbb_study.digital_twin.nathan_mode2z_orientation_interpolation import (
    Mode2ZSweepResult,
    mode2z_route_id,
)


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


def _line_segments(alpha: np.ndarray, extent: Sequence[float], *, samples: int = 23) -> np.ndarray:
    ny, nx = alpha.shape
    yi = np.linspace(0, ny - 1, samples, dtype=int)
    xi = np.linspace(0, nx - 1, samples, dtype=int)
    xx, yy = np.meshgrid(xi, yi)
    angles = np.asarray(alpha, dtype=float)[yy, xx]
    xcoord = np.interp(xx, [0, nx - 1], [extent[0], extent[1]])
    ycoord = np.interp(yy, [0, ny - 1], [extent[2], extent[3]])
    length = 0.065 * min(float(extent[1] - extent[0]), float(extent[3] - extent[2]))
    dx = 0.5 * length * np.cos(angles)
    dy = 0.5 * length * np.sin(angles)
    return np.stack([
        np.stack([xcoord - dx, ycoord - dy], axis=-1),
        np.stack([xcoord + dx, ycoord + dy], axis=-1),
    ], axis=-2).reshape(-1, 2, 2)


def _overlay_lines(ax: Any, alpha: np.ndarray, extent: Sequence[float]) -> None:
    _, LineCollection = _mpl()
    segments = _line_segments(alpha, extent)
    ax.add_collection(LineCollection(segments, colors="white", linewidths=0.62, alpha=0.9))


def _nearest_eta_values(result: Mode2ZSweepResult, requested: Sequence[float]) -> list[float]:
    available = np.asarray(result.config.eta_values, dtype=float)
    return [float(available[int(np.argmin(np.abs(available - value)))]) for value in requested]


def plot_input_interpolation(result: Mode2ZSweepResult, root: Path) -> tuple[Path, Path]:
    plt, _ = _mpl()
    eta_values = _nearest_eta_values(result, (0.0, 0.25, 0.5, 0.75, 1.0))
    grid = result.data["grid"]
    extent = _extent_mm(grid)
    amplitude = _normalise_peak(np.asarray(result.data["A"], dtype=float) ** 2)
    target = np.asarray(result.mode2y_inputs.continuous_alpha_rad, dtype=float)
    fig, axes = plt.subplots(2, len(eta_values), figsize=(16.5, 6.8), constrained_layout=True)
    error_image = None
    for column, eta in enumerate(eta_values):
        alpha = np.asarray(result.alpha_by_eta[eta], dtype=float)
        axes[0, column].imshow(amplitude, origin="lower", extent=extent, cmap="gray", interpolation="bilinear")
        _overlay_lines(axes[0, column], alpha, extent)
        axes[0, column].set_title(f"eta = {eta:.1f}")
        error = np.rad2deg(np.abs(line_orientation_error(alpha, target)))
        error_image = axes[1, column].imshow(
            error,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=30.0,
            interpolation="bilinear",
        )
        axes[1, column].set_title("local line error")
        for row in range(2):
            axes[row, column].set_xlabel("x (mm)")
            if column == 0:
                axes[row, column].set_ylabel("y (mm)")
    if error_image is not None:
        fig.colorbar(error_image, ax=axes[1, :], label="degrees", shrink=0.78)
    fig.suptitle("MODE 2Z input interpolation: sector-centre lines to true local field", fontsize=15)
    paths = _save(fig, root / "00_inputs/orientation_interpolation_inputs")
    plt.close(fig)
    return paths


def plot_xy_sweep(result: Mode2ZSweepResult, root: Path, optical_route: str) -> tuple[Path, Path]:
    plt, _ = _mpl()
    grid = result.data["grid"]
    sy, sx = _focus_slices(grid, float(result.data["mode2z_ring_radius_m"]))
    extent = _extent_mm(grid, sy, sx)
    row_by_eta = {
        float(row["eta"]): row for row in result.summary_rows if row["optical_route"] == optical_route
    }
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 10.0), constrained_layout=True)
    image = None
    for ax, eta in zip(axes.ravel(), result.config.eta_values):
        route = result.routes[mode2z_route_id(optical_route, float(eta))]
        plane = np.asarray(route.selected_planes[_selected_key(60e-3)], dtype=float)[sy, sx]
        image = ax.imshow(
            _normalise_peak(plane),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="bicubic",
        )
        strict = "PASS" if row_by_eta[float(eta)]["strict_hexagon_pass"] else "fail"
        ax.set_title(f"eta={float(eta):.1f} | {strict}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    for ax in axes.ravel()[len(result.config.eta_values):]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes[:, -1], label="normalised intensity", shrink=0.75)
    fig.suptitle(f"{optical_route.capitalize()} route at z=60 mm", fontsize=15)
    paths = _save(fig, root / f"01_xy_sweep/{optical_route}_eta_xy_focus")
    plt.close(fig)
    return paths


def plot_metric_trends(result: Mode2ZSweepResult, root: Path) -> tuple[Path, Path]:
    plt, _ = _mpl()
    panels = (
        ("z60_correlation_to_v0", "V0 correlation", None),
        ("edge_gradient_sharpness_mm_inv", "Edge gradient (mm$^{-1}$)", None),
        ("threshold_transition_width_mm", "80-20 width (mm)", None),
        ("bright_ridge_fwhm_mm", "Ridge FWHM (mm)", None),
        ("peak_intensity", "Peak intensity", None),
        ("useful_region_power", "Useful-region energy", None),
        ("dark_core_ratio", "Dark-core ratio", "log"),
        ("morphology_quality_index", "Morphology quality index", None),
    )
    styles = {"ideal": ("#0072B2", "-", "o"), "realistic": ("#D55E00", "--", "s")}
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.0), constrained_layout=True)
    for ax, (metric, title, scale) in zip(axes.ravel(), panels):
        for route in ("ideal", "realistic"):
            rows = sorted(
                (row for row in result.summary_rows if row["optical_route"] == route),
                key=lambda row: float(row["eta"]),
            )
            colour, linestyle, marker = styles[route]
            ax.plot(
                [row["eta"] for row in rows],
                [row[metric] for row in rows],
                label=route,
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markersize=3.8,
            )
        ax.set_title(title)
        ax.set_xlabel("orientation fidelity eta")
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("MODE 2Z morphology and energy trends", fontsize=15)
    paths = _save(fig, root / "02_metric_trends/orientation_fidelity_metric_trends")
    plt.close(fig)
    return paths


def plot_axial_trends(result: Mode2ZSweepResult, root: Path) -> tuple[Path, Path]:
    plt, _ = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for route, colour, marker in (("ideal", "#0072B2", "o"), ("realistic", "#D55E00", "s")):
        rows = sorted(
            (row for row in result.summary_rows if row["optical_route"] == route),
            key=lambda row: float(row["eta"]),
        )
        eta = [row["eta"] for row in rows]
        axes[0].plot(eta, [row["best_z_mm"] for row in rows], marker=marker, label=route, color=colour)
        axes[1].plot(
            eta,
            [row["propagation_persistence_fraction"] for row in rows],
            marker=marker,
            label=route,
            color=colour,
        )
    axes[0].set(xlabel="orientation fidelity eta", ylabel="best z (mm)", title="Best sharpness plane")
    axes[1].set(
        xlabel="orientation fidelity eta",
        ylabel="persistence fraction",
        title="Fraction within 80% of route maximum",
    )
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Axial evolution versus orientation fidelity", fontsize=14)
    paths = _save(fig, root / "02_metric_trends/orientation_fidelity_axial_trends")
    plt.close(fig)
    return paths


def _normalise_rows(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    return arr / np.maximum(np.max(arr, axis=1, keepdims=True), EPS)


def plot_propagation_maps(result: Mode2ZSweepResult, root: Path) -> tuple[Path, Path]:
    plt, _ = _mpl()
    eta_values = _nearest_eta_values(result, (0.0, 0.5, 1.0))
    grid = result.data["grid"]
    x = np.asarray(grid["x"], dtype=float)
    indices = np.where(np.abs(x) <= 0.9e-3)[0]
    sx = slice(int(indices[0]), int(indices[-1]) + 1)
    extent = [
        float(x[indices[0]] / 1e-3),
        float(x[indices[-1]] / 1e-3),
        float(result.config.z_end_m / 1e-3),
        float(result.config.z_start_m / 1e-3),
    ]
    fig, axes = plt.subplots(2, len(eta_values), figsize=(14.0, 8.5), constrained_layout=True)
    image = None
    for column, eta in enumerate(eta_values):
        route = result.routes[mode2z_route_id("realistic", eta)]
        for row, (array, axis_name) in enumerate(((route.xz_map, "x-z"), (route.yz_map, "y-z"))):
            image = axes[row, column].imshow(
                _normalise_rows(np.asarray(array, dtype=float)[:, sx]),
                origin="upper",
                aspect="auto",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                interpolation="bicubic",
            )
            axes[row, column].set_title(f"{axis_name} | eta={eta:.1f}")
            axes[row, column].set_xlabel(f"{axis_name[0]} (mm)")
            axes[row, column].set_ylabel("z (mm)")
    if image is not None:
        fig.colorbar(image, ax=axes[:, -1], label="row-normalised intensity", shrink=0.76)
    fig.suptitle("Realistic-route propagation as local orientation becomes continuous", fontsize=15)
    paths = _save(fig, root / "03_propagation/realistic_eta_xz_yz")
    plt.close(fig)
    return paths


def plot_tradeoff(result: Mode2ZSweepResult, root: Path) -> tuple[Path, Path]:
    plt, _ = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, route in zip(axes, ("ideal", "realistic")):
        rows = sorted(
            (row for row in result.summary_rows if row["optical_route"] == route),
            key=lambda row: float(row["eta"]),
        )
        base_energy = max(float(rows[0]["useful_region_power"]), EPS)
        colours = ["#009E73" if row["morphology_energy_pareto"] else "0.65" for row in rows]
        x = [float(row["useful_region_power"]) / base_energy for row in rows]
        y = [row["morphology_quality_index"] for row in rows]
        ax.plot(x, y, color="0.75", linewidth=0.9)
        ax.scatter(x, y, c=colours, s=42, zorder=3)
        for xx, yy, row in zip(x, y, rows):
            ax.annotate(f"{float(row['eta']):.1f}", (xx, yy), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set(
            xlabel="useful-region energy / eta=0",
            ylabel="morphology quality index",
            title=f"{route.capitalize()} route",
        )
        ax.grid(alpha=0.22)
    fig.suptitle("Morphology-energy trade-off; green points are Pareto non-dominated", fontsize=14)
    paths = _save(fig, root / "04_tradeoff/orientation_fidelity_pareto")
    plt.close(fig)
    return paths


def plot_gate_dashboard(result: Mode2ZSweepResult, root: Path) -> tuple[Path, Path]:
    plt, _ = _mpl()
    input_rows = sorted(result.input_rows, key=lambda row: float(row["eta"]))
    eta_input = [row["eta"] for row in input_rows]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
    axes[0, 0].plot(eta_input, np.rad2deg([row["local_angle_rms_rad"] for row in input_rows]), marker="o")
    axes[0, 0].set(title="Local line RMS error", ylabel="degrees")
    axes[0, 1].plot(eta_input, [row["radial_purity"] for row in input_rows], marker="o", label="radial")
    axes[0, 1].plot(eta_input, [row["azimuthal_purity"] for row in input_rows], marker="s", label="azimuthal")
    axes[0, 1].set(title="Local cylindrical purity", ylabel="power fraction")
    axes[0, 1].legend(frameon=False)
    for route, colour in (("ideal", "#0072B2"), ("realistic", "#D55E00")):
        rows = sorted(
            (row for row in result.summary_rows if row["optical_route"] == route),
            key=lambda row: float(row["eta"]),
        )
        eta = [row["eta"] for row in rows]
        axes[0, 2].step(eta, [int(row["strict_hexagon_pass"]) for row in rows], where="mid", label=route, color=colour)
        axes[1, 0].plot(eta, [row["h3_over_h6"] for row in rows], marker="o", label=route, color=colour)
        axes[1, 1].plot(eta, [row["delta_c120_minus_c60"] for row in rows], marker="o", label=route, color=colour)
        axes[1, 2].plot(eta, [row["corner_concentration"] for row in rows], marker="o", label=route, color=colour)
    axes[0, 2].set(title="Repaired strict gate", yticks=(0, 1), yticklabels=("fail", "pass"))
    axes[1, 0].set(title="Triangular leakage", ylabel="H3 / H6")
    axes[1, 1].set(title="C6 discriminator", ylabel="C120 - C60")
    axes[1, 2].set(title="Corner concentration", ylabel="corner / side")
    for ax in axes.ravel():
        ax.set_xlabel("orientation fidelity eta")
        ax.grid(alpha=0.22)
    axes[0, 2].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    fig.suptitle(f"MODE 2Z local truth and morphology gates: {result.outcome}", fontsize=15)
    paths = _save(fig, root / "05_gates/orientation_fidelity_gate_dashboard")
    plt.close(fig)
    return paths


def write_mode2z_figures(result: Mode2ZSweepResult, root: Path) -> dict[str, Path]:
    """Write every MODE 2Z PNG/PDF figure pair."""

    paths: dict[str, Path] = {}
    generated = plot_input_interpolation(result, root)
    paths["inputs_png"], paths["inputs_pdf"] = generated
    for route in ("ideal", "realistic"):
        generated = plot_xy_sweep(result, root, route)
        paths[f"{route}_xy_png"], paths[f"{route}_xy_pdf"] = generated
    generated = plot_metric_trends(result, root)
    paths["metrics_png"], paths["metrics_pdf"] = generated
    generated = plot_axial_trends(result, root)
    paths["axial_png"], paths["axial_pdf"] = generated
    generated = plot_propagation_maps(result, root)
    paths["propagation_png"], paths["propagation_pdf"] = generated
    generated = plot_tradeoff(result, root)
    paths["tradeoff_png"], paths["tradeoff_pdf"] = generated
    generated = plot_gate_dashboard(result, root)
    paths["gates_png"], paths["gates_pdf"] = generated
    return paths


__all__ = [
    "plot_axial_trends",
    "plot_gate_dashboard",
    "plot_input_interpolation",
    "plot_metric_trends",
    "plot_propagation_maps",
    "plot_tradeoff",
    "plot_xy_sweep",
    "write_mode2z_figures",
]
