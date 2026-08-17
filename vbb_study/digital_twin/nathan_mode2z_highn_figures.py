"""Publication figures for the targeted MODE 2Z high-N confirmation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_mode2z_highn_confirmation import Mode2ZHighNResult
from vbb_study.digital_twin.nathan_vector_hexagon import EPS


def _mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    return plt


def _save(fig: Any, stem: Path, *, dpi: int = 320) -> tuple[Path, Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    return png, pdf


def _focus_slices(grid: Mapping[str, Any], ring_radius_m: float) -> tuple[slice, slice]:
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    half_width_m = max(3.6 * float(ring_radius_m), 0.65e-3)
    xi = np.where(np.abs(x) <= half_width_m)[0]
    yi = np.where(np.abs(y) <= half_width_m)[0]
    return slice(int(yi[0]), int(yi[-1]) + 1), slice(int(xi[0]), int(xi[-1]) + 1)


def _extent_mm(grid: Mapping[str, Any], sy: slice, sx: slice) -> list[float]:
    x = np.asarray(grid["x"], dtype=float)[sx]
    y = np.asarray(grid.get("y", grid["x"]), dtype=float)[sy]
    return [float(x[0] / 1e-3), float(x[-1] / 1e-3), float(y[0] / 1e-3), float(y[-1] / 1e-3)]


def _normalise_peak(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    return values / max(float(np.max(values)), EPS)


def _row_map(rows: Sequence[Mapping[str, Any]]) -> dict[float, Mapping[str, Any]]:
    return {float(row["eta"]): row for row in rows}


def plot_highn_focus_grid(result: Mode2ZHighNResult, root: Path) -> tuple[Path, Path]:
    """Show every selected eta on its native N=1536 focus plane."""

    plt = _mpl()
    grid = result.data["grid"]
    sy, sx = _focus_slices(grid, float(result.data["mode2z_hn_ring_radius_m"]))
    extent = _extent_mm(grid, sy, sx)
    rows = _row_map(result.summary_rows)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.2), constrained_layout=True)
    image = None
    for ax, eta in zip(axes.ravel(), result.config.eta_values):
        plane = _normalise_peak(np.asarray(result.planes_by_eta[float(eta)], dtype=float)[sy, sx])
        image = ax.imshow(
            plane,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="bicubic",
        )
        gate = "PASS" if bool(rows[float(eta)]["strict_hexagon_pass"]) else "fail"
        ax.set_title(f"eta={float(eta):.1f} | {gate}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
    if image is not None:
        fig.colorbar(image, ax=axes[:, -1], label="normalised intensity", shrink=0.78)
    fig.suptitle(
        f"Realistic sequential route at z=60 mm, native N={result.config.grid_n}",
        fontsize=14.5,
    )
    paths = _save(fig, root / "01_focus_planes/highn_selected_eta_focus")
    plt.close(fig)
    return paths


def plot_metric_convergence(result: Mode2ZHighNResult, root: Path) -> tuple[Path, Path]:
    """Compare selected-eta native metrics at N=1024 and high N."""

    plt = _mpl()
    panels = (
        ("z60_correlation_to_v0", "V0 correlation", False),
        ("edge_gradient_sharpness_mm_inv", "Edge gradient / eta=0", True),
        ("threshold_transition_width_mm", "80-20 width (mm)", False),
        ("bright_ridge_fwhm_mm", "Ridge FWHM (mm)", False),
        ("peak_intensity", "Peak intensity / eta=0", True),
        ("useful_region_power", "Useful-region energy / eta=0", True),
    )
    high = sorted(result.summary_rows, key=lambda row: float(row["eta"]))
    low = sorted(result.baseline_rows, key=lambda row: float(row["eta"]))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.89, wspace=0.13, hspace=0.25)
    for ax, (metric, title, normalise) in zip(axes.ravel(), panels):
        if low:
            low_values = np.asarray([float(row[metric]) for row in low], dtype=float)
            if normalise:
                low_values = low_values / max(abs(float(low_values[0])), EPS)
            ax.plot(
                [row["eta"] for row in low],
                low_values,
                color="#0072B2",
                marker="o",
                linewidth=1.4,
                label=f"N={int(low[0]['grid_n'])}",
            )
        high_values = np.asarray([float(row[metric]) for row in high], dtype=float)
        if normalise:
            high_values = high_values / max(abs(float(high_values[0])), EPS)
        ax.plot(
            [row["eta"] for row in high],
            high_values,
            color="#D55E00",
            marker="s",
            linewidth=1.4,
            label=f"N={result.config.grid_n}",
        )
        ax.axvline(0.7, color="0.35", linestyle=":", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("orientation fidelity eta")
        ax.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("MODE 2Z selected-eta native-grid convergence", fontsize=14.5, y=0.965)
    paths = _save(fig, root / "02_convergence/n1024_vs_highn_metrics")
    plt.close(fig)
    return paths


def _threshold_eta_values(result: Mode2ZHighNResult) -> list[float]:
    available = np.asarray(result.config.eta_values, dtype=float)
    return [float(available[int(np.argmin(np.abs(available - value)))]) for value in (0.6, 0.7, 0.8)]


def plot_threshold_detail(result: Mode2ZHighNResult, root: Path) -> tuple[Path, Path]:
    """Resolve the selected strict-gate neighbourhood in linear and log display."""

    plt = _mpl()
    grid = result.data["grid"]
    sy, sx = _focus_slices(grid, float(result.data["mode2z_hn_ring_radius_m"]))
    extent = _extent_mm(grid, sy, sx)
    rows = _row_map(result.summary_rows)
    eta_values = _threshold_eta_values(result)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.2), constrained_layout=True)
    linear_image = None
    log_image = None
    for column, eta in enumerate(eta_values):
        plane = _normalise_peak(np.asarray(result.planes_by_eta[eta], dtype=float)[sy, sx])
        gate = "PASS" if bool(rows[eta]["strict_hexagon_pass"]) else "fail"
        linear_image = axes[0, column].imshow(
            plane,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="bicubic",
        )
        log_image = axes[1, column].imshow(
            10.0 * np.log10(np.maximum(plane, 1e-5)),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=-50.0,
            vmax=0.0,
            interpolation="bicubic",
        )
        axes[0, column].set_title(f"eta={eta:.1f} | {gate}")
        axes[1, column].set_title(f"eta={eta:.1f} | log view")
        for row_index in range(2):
            axes[row_index, column].set_xlabel("x (mm)")
            axes[row_index, column].set_ylabel("y (mm)")
            axes[row_index, column].set_aspect("equal")
    if linear_image is not None:
        fig.colorbar(linear_image, ax=axes[0, :], label="normalised intensity", shrink=0.78)
    if log_image is not None:
        fig.colorbar(log_image, ax=axes[1, :], label="relative intensity (dB)", shrink=0.78)
    fig.suptitle("High-N detail around the selected-grid strict onset", fontsize=14.5)
    paths = _save(fig, root / "03_threshold/highn_eta_060_070_080_detail")
    plt.close(fig)
    return paths


def plot_width_resolution(result: Mode2ZHighNResult, root: Path) -> tuple[Path, Path]:
    """Expose width quantisation and any surviving high-N plateaus."""

    plt = _mpl()
    panels = (
        ("threshold_transition_width_mm", "80-20 transition width"),
        ("bright_ridge_fwhm_mm", "Bright-ridge FWHM"),
    )
    low = sorted(result.baseline_rows, key=lambda row: float(row["eta"]))
    high = sorted(result.summary_rows, key=lambda row: float(row["eta"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    for ax, (metric, title) in zip(axes, panels):
        if low:
            ax.step(
                [row["eta"] for row in low],
                [row[metric] for row in low],
                where="mid",
                marker="o",
                color="#0072B2",
                label=f"N={int(low[0]['grid_n'])}",
            )
        ax.step(
            [row["eta"] for row in high],
            [row[metric] for row in high],
            where="mid",
            marker="s",
            color="#D55E00",
            label=f"N={result.config.grid_n}",
        )
        ax.set_title(title)
        ax.set_xlabel("orientation fidelity eta")
        ax.set_ylabel("mm")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Native-grid morphology-width resolution; no metric interpolation", fontsize=14.0)
    paths = _save(fig, root / "02_convergence/width_fwhm_resolution")
    plt.close(fig)
    return paths


def write_mode2z_highn_figures(result: Mode2ZHighNResult, root: Path) -> dict[str, Path]:
    """Write all targeted MODE 2Z high-N PNG/PDF figure pairs."""

    paths: dict[str, Path] = {}
    generated = plot_highn_focus_grid(result, root)
    paths["focus_png"], paths["focus_pdf"] = generated
    generated = plot_metric_convergence(result, root)
    paths["convergence_png"], paths["convergence_pdf"] = generated
    generated = plot_threshold_detail(result, root)
    paths["threshold_png"], paths["threshold_pdf"] = generated
    generated = plot_width_resolution(result, root)
    paths["width_png"], paths["width_pdf"] = generated
    return paths


__all__ = [
    "plot_highn_focus_grid",
    "plot_metric_convergence",
    "plot_threshold_detail",
    "plot_width_resolution",
    "write_mode2z_highn_figures",
]
