"""Reusable quick-look and diagnostic visualisations.

The helpers in this module are deliberately thin wrappers around Matplotlib:
they do not compute beam metrics or improve numerical sampling.  Their job is
to keep axes, units, colourbars, display transforms, and visual-only
interpolation labels consistent across notebooks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from vbb_study import vbb_style

UM = 1e-6
MM = 1e-3
DISPLAY_INTERPOLATION_LABEL = "display interpolation only; metrics use the raw grid"
UNDERSAMPLING_WARNING = "coarse display grid; rerun balanced/publication mode for decisions"

_UNIT_SCALE = {
    "m": 1.0,
    "mm": MM,
    "um": UM,
    "pixel": 1.0,
    "lp/mm": 1.0,
}


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _finite_max(values: np.ndarray) -> float:
    arr = _as_array(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmax(finite))


def _positive_max(values: np.ndarray) -> float:
    return max(_finite_max(np.maximum(_as_array(values), 0.0)), 0.0)


def _axis_from_grid(grid: Mapping[str, Any] | None, n: int, unit: str = "um") -> np.ndarray:
    scale = _UNIT_SCALE.get(unit, UM)
    if grid is not None and "x" in grid:
        return np.asarray(grid["x"], dtype=float) / scale
    dx = 1.0
    if grid is not None and "dx" in grid:
        dx = float(grid["dx"]) / scale
    idx = np.arange(int(n), dtype=float) - 0.5 * (int(n) - 1)
    return idx * dx


def _extent_from_grid(grid: Mapping[str, Any] | None, shape: tuple[int, int], unit: str = "um") -> tuple[float, float, float, float]:
    y = _axis_from_grid(grid, shape[0], unit)
    x = _axis_from_grid(grid, shape[1], unit)
    return float(x[0]), float(x[-1]), float(y[0]), float(y[-1])


def _axis_from_values(values: Sequence[float] | np.ndarray | None, n: int, unit: str = "um") -> np.ndarray:
    scale = _UNIT_SCALE.get(unit, UM)
    if values is None:
        return np.arange(int(n), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size != int(n):
        raise ValueError(f"axis length {arr.size} does not match data dimension {n}")
    if np.nanmax(np.abs(arr)) < 1e-2 and unit in {"um", "mm"}:
        return arr / scale
    return arr


def _display_intensity(
    values: Any,
    *,
    normalise: bool = True,
    reference_max: float | None = None,
    log_scale: bool = False,
    gamma: float | None = None,
) -> tuple[np.ndarray, str]:
    raw = np.maximum(_as_array(values), 0.0)
    raw[~np.isfinite(raw)] = 0.0
    ref = float(reference_max) if reference_max is not None else _positive_max(raw)
    if normalise:
        arr = raw / max(ref, np.finfo(float).eps)
        base = "normalised intensity"
    else:
        arr = raw
        base = "intensity"
    if log_scale:
        return np.log10(arr + np.finfo(float).eps), f"log10 {base} [a.u.]"
    if gamma is not None:
        arr = np.maximum(arr, 0.0) ** float(gamma)
        return arr, f"display {base}^{{gamma={float(gamma):.2g}}} [a.u.]"
    return arr, f"{base} [a.u.]"


def _resolve_interpolation(display_interpolation: bool, interpolation: str | None) -> tuple[str, str]:
    method = str(interpolation or ("bilinear" if display_interpolation else "nearest")).strip()
    if display_interpolation and method == "nearest":
        method = "bilinear"
    label = DISPLAY_INTERPOLATION_LABEL if method != "nearest" else "not_applicable"
    return method, label


def _add_display_labels(
    ax: plt.Axes,
    *,
    interpolation_label: str,
    warning: str,
    qa_label: str | None,
) -> None:
    y = 0.015
    if interpolation_label != "not_applicable":
        ax.text(
            0.015,
            y,
            interpolation_label,
            transform=ax.transAxes,
            fontsize=7,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "alpha": 0.55, "linewidth": 0.0},
        )
        y += 0.075
    if warning:
        ax.text(
            0.015,
            y,
            warning,
            transform=ax.transAxes,
            fontsize=7,
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#fff4b8", "alpha": 0.95, "linewidth": 0.0},
        )
    if qa_label:
        ax.text(
            0.985,
            0.985,
            str(qa_label),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "alpha": 0.55, "linewidth": 0.0},
        )


def _warning_for_array(values: np.ndarray, min_pixels: int = 30) -> str:
    arr = np.asarray(values)
    if arr.ndim >= 2 and min(arr.shape[-2:]) < int(min_pixels):
        return UNDERSAMPLING_WARNING
    return ""


def _metadata(
    *,
    interpolation: str,
    interpolation_label: str,
    warning: str,
    colorbar_label: str,
    units_present: str = "yes",
    qa_label: str | None = None,
) -> dict[str, Any]:
    return {
        "interpolation_used": "yes" if interpolation != "nearest" else "no",
        "interpolation_method": interpolation,
        "interpolation_label": interpolation_label,
        "interpolation_label_present": "yes" if interpolation_label != "not_applicable" else "not_applicable",
        "undersampling_warning": warning,
        "undersampling_warning_visible": bool(warning),
        "units_present": units_present,
        "colourbar_label": colorbar_label,
        "colourbar_label_present": "yes" if colorbar_label else "no",
        "qa_label_visible": "yes" if qa_label else "not_applicable",
    }


def plot_xy_intensity(
    intensity: Any,
    grid: Mapping[str, Any] | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    unit: str = "um",
    normalise: bool = True,
    reference_max: float | None = None,
    log_scale: bool = False,
    gamma: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = vbb_style.INTENSITY_CMAP,
    display_interpolation: bool = False,
    interpolation: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    ring_radius_um: float | None = None,
    core_radius_um: float | None = None,
    qa_label: str | None = None,
    caveat_text: str | None = None,
) -> dict[str, Any]:
    """Plot an XY intensity-like plane with explicit display metadata."""

    vbb_style.apply_style()
    arr, default_label = _display_intensity(
        intensity,
        normalise=normalise,
        reference_max=reference_max,
        log_scale=log_scale,
        gamma=gamma,
    )
    label = str(colorbar_label or default_label)
    interpolation_method, interpolation_label = _resolve_interpolation(display_interpolation, interpolation)
    warning = _warning_for_array(arr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(
        arr,
        origin="lower",
        extent=_extent_from_grid(grid, arr.shape, unit),
        cmap=cmap,
        interpolation=interpolation_method,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    if title:
        ax.set_title(title)
    if ring_radius_um is not None and np.isfinite(float(ring_radius_um)):
        ax.add_patch(patches.Circle((0.0, 0.0), float(ring_radius_um), fill=False, edgecolor="#56B4E9", linewidth=1.1))
    if core_radius_um is not None and np.isfinite(float(core_radius_um)):
        ax.add_patch(patches.Circle((0.0, 0.0), float(core_radius_um), fill=False, edgecolor="#009E73", linewidth=1.1))
    if colorbar:
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
    else:
        cbar = None
    _add_display_labels(ax, interpolation_label=interpolation_label, warning=warning, qa_label=qa_label)
    if caveat_text:
        ax.text(0.5, -0.16, caveat_text, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    return {
        "fig": fig,
        "ax": ax,
        "image": image,
        "colorbar": cbar,
        "metadata": _metadata(
            interpolation=interpolation_method,
            interpolation_label=interpolation_label,
            warning=warning,
            colorbar_label=label,
            qa_label=qa_label,
        ),
    }


def plot_xz_intensity(
    xz: Any,
    z_um: Sequence[float] | np.ndarray | None = None,
    transverse_um: Sequence[float] | np.ndarray | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    normalise: bool = True,
    reference_max: float | None = None,
    log_scale: bool = False,
    gamma: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = vbb_style.INTENSITY_CMAP,
    display_interpolation: bool = False,
    interpolation: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    canonical_zone_um: tuple[float, float] | float | None = None,
    strict_region_um: tuple[float, float] | float | None = None,
    interface_z_um: float | None = None,
    qa_label: str | None = None,
    caveat_text: str | None = None,
) -> dict[str, Any]:
    """Plot an XZ intensity trace with z and transverse axes in micrometres."""

    vbb_style.apply_style()
    arr, default_label = _display_intensity(
        xz,
        normalise=normalise,
        reference_max=reference_max,
        log_scale=log_scale,
        gamma=gamma,
    )
    label = str(colorbar_label or default_label)
    x_axis = _axis_from_values(transverse_um if transverse_um is not None else (None if grid is None else grid.get("x")), arr.shape[0], "um")
    z_axis = _axis_from_values(z_um, arr.shape[1], "um")
    interpolation_method, interpolation_label = _resolve_interpolation(display_interpolation, interpolation)
    warning = _warning_for_array(arr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(
        arr,
        origin="lower",
        extent=(float(z_axis[0]), float(z_axis[-1]), float(x_axis[0]), float(x_axis[-1])),
        cmap=cmap,
        interpolation=interpolation_method,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xlabel("z [um]")
    ax.set_ylabel("x [um]")
    if title:
        ax.set_title(title)
    _shade_zone(ax, canonical_zone_um, "#0072B2", "canonical zone")
    _shade_zone(ax, strict_region_um, "#D55E00", "strict region")
    if interface_z_um is not None and np.isfinite(float(interface_z_um)):
        ax.axvline(float(interface_z_um), color="white", linewidth=1.0, linestyle="--")
        ax.text(float(interface_z_um), 0.98, "interface", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=8, color="white")
    if colorbar:
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
    else:
        cbar = None
    _add_display_labels(ax, interpolation_label=interpolation_label, warning=warning, qa_label=qa_label)
    if caveat_text:
        ax.text(0.5, -0.18, caveat_text, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    return {
        "fig": fig,
        "ax": ax,
        "image": image,
        "colorbar": cbar,
        "metadata": _metadata(
            interpolation=interpolation_method,
            interpolation_label=interpolation_label,
            warning=warning,
            colorbar_label=label,
            qa_label=qa_label,
        ),
    }


def _shade_zone(ax: plt.Axes, zone: tuple[float, float] | float | None, colour: str, label: str) -> None:
    if zone is None:
        return
    if isinstance(zone, tuple):
        start, end = zone
    else:
        start, end = 0.0, float(zone)
    if not (np.isfinite(start) and np.isfinite(end)):
        return
    lo, hi = sorted((float(start), float(end)))
    ax.axvspan(lo, hi, color=colour, alpha=0.13, linewidth=0.0, label=label)
    ax.text(0.5 * (lo + hi), 0.02, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=colour)


def plot_xy_xz_pair(
    xy: Any,
    xz: Any,
    *,
    grid: Mapping[str, Any] | None = None,
    z_um: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    shared_color_scale: bool = True,
    display_interpolation: bool = False,
    interpolation: str | None = None,
    log_scale: bool = False,
    qa_label: str | None = None,
    caveat_text: str | None = None,
) -> dict[str, Any]:
    """Plot a coupled XY/XZ view and an optional metric summary box."""

    vbb_style.apply_style()
    cols = 3 if metrics else 2
    fig, axes = plt.subplots(1, cols, figsize=(5.2 * cols, 4.2), constrained_layout=True)
    if cols == 2:
        ax_xy, ax_xz = axes
        ax_box = None
    else:
        ax_xy, ax_xz, ax_box = axes
    ref = max(_positive_max(_as_array(xy)), _positive_max(_as_array(xz))) if shared_color_scale else None
    xy_result = plot_xy_intensity(
        xy,
        grid,
        ax=ax_xy,
        title="XY peak plane" if title is None else f"{title}: XY",
        reference_max=ref,
        display_interpolation=display_interpolation,
        interpolation=interpolation,
        log_scale=log_scale,
        qa_label=qa_label,
        caveat_text=caveat_text,
    )
    xz_result = plot_xz_intensity(
        xz,
        z_um,
        None,
        grid=grid,
        ax=ax_xz,
        title="XZ axial trace" if title is None else f"{title}: XZ",
        reference_max=ref,
        display_interpolation=display_interpolation,
        interpolation=interpolation,
        log_scale=log_scale,
        qa_label=qa_label,
        caveat_text=caveat_text,
    )
    summary = None
    if ax_box is not None:
        summary = plot_metric_summary_box(metrics or {}, ax=ax_box, title="Key metrics")
    return {
        "fig": fig,
        "axes": axes,
        "xy": xy_result,
        "xz": xz_result,
        "summary": summary,
        "metadata": {
            "shared_color_scale": bool(shared_color_scale),
            "xy": xy_result["metadata"],
            "xz": xz_result["metadata"],
        },
    }


def plot_slm_phase_mask(
    phase: Any,
    grid: Mapping[str, Any] | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    unit: str = "mm",
    display_interpolation: bool = False,
    interpolation: str | None = None,
    phase_quantisation_bits: int | None = None,
    labels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plot a wrapped SLM phase mask in radians."""

    vbb_style.apply_style()
    wrapped = (np.asarray(phase, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    interpolation_method, interpolation_label = _resolve_interpolation(display_interpolation, interpolation)
    warning = _warning_for_array(wrapped)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(
        wrapped,
        origin="lower",
        extent=_extent_from_grid(grid, wrapped.shape, unit),
        cmap=vbb_style.PHASE_CMAP,
        interpolation=interpolation_method,
        vmin=-np.pi,
        vmax=np.pi,
        aspect="equal",
    )
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.set_title(title or "SLM wrapped phase mask")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("wrapped phase [rad]")
    note = []
    if phase_quantisation_bits is not None:
        note.append(f"{int(phase_quantisation_bits)} bit phase")
    for key, value in dict(labels or {}).items():
        note.append(f"{key}: {value}")
    if note:
        ax.text(
            0.015,
            0.985,
            "\n".join(note[:6]),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.55, "linewidth": 0.0},
        )
    _add_display_labels(ax, interpolation_label=interpolation_label, warning=warning, qa_label=None)
    return {
        "fig": fig,
        "ax": ax,
        "image": image,
        "colorbar": cbar,
        "metadata": _metadata(
            interpolation=interpolation_method,
            interpolation_label=interpolation_label,
            warning=warning,
            colorbar_label="wrapped phase [rad]",
        ),
    }


def plot_fourier_filter_plane(
    spectrum: Any,
    grid: Mapping[str, Any] | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    fx_lpmm: Sequence[float] | np.ndarray | None = None,
    fy_lpmm: Sequence[float] | np.ndarray | None = None,
    carrier_lpmm: float | None = None,
    filter_radius_lpmm: float | None = None,
    selected_fraction: float | None = None,
    display_interpolation: bool = False,
    interpolation: str | None = None,
) -> dict[str, Any]:
    """Plot a Fourier-plane power map with optional first-order stop overlay."""

    vbb_style.apply_style()
    raw = np.asarray(spectrum)
    power = np.abs(raw) ** 2 if np.iscomplexobj(raw) else np.maximum(np.asarray(raw, dtype=float), 0.0)
    arr, label = _display_intensity(power, normalise=True, log_scale=True)
    if grid is not None and "FX" in grid:
        fx_axis = np.asarray(grid["FX"][0], dtype=float) / 1e3
        fy_axis = np.asarray(grid["FY"][:, 0], dtype=float) / 1e3
    else:
        fx_axis = _axis_from_values(fx_lpmm, arr.shape[1], "lp/mm")
        fy_axis = _axis_from_values(fy_lpmm, arr.shape[0], "lp/mm")
    interpolation_method, interpolation_label = _resolve_interpolation(display_interpolation, interpolation)
    warning = _warning_for_array(arr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(
        arr,
        origin="lower",
        extent=(float(fx_axis[0]), float(fx_axis[-1]), float(fy_axis[0]), float(fy_axis[-1])),
        cmap=vbb_style.INTENSITY_CMAP,
        interpolation=interpolation_method,
        aspect="equal",
    )
    ax.set_xlabel("fx [lp/mm]")
    ax.set_ylabel("fy [lp/mm]")
    ax.set_title(title or "Fourier filter plane")
    ax.plot([0.0], [0.0], marker="x", color="white", markersize=7, label="zero order")
    if carrier_lpmm is not None and filter_radius_lpmm is not None:
        ax.add_patch(
            patches.Circle(
                (float(carrier_lpmm), 0.0),
                float(filter_radius_lpmm),
                fill=False,
                edgecolor="#56B4E9",
                linewidth=1.2,
            )
        )
        ax.plot([float(carrier_lpmm)], [0.0], marker="+", color="#56B4E9", markersize=8, label="+1 order")
        ax.legend(loc="lower right", frameon=True, fontsize=7)
    if selected_fraction is not None and np.isfinite(float(selected_fraction)):
        rejected = max(0.0, 1.0 - float(selected_fraction))
        ax.text(
            0.985,
            0.985,
            f"selected: {float(selected_fraction):.3g}\nrejected: {rejected:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "alpha": 0.55, "linewidth": 0.0},
        )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    _add_display_labels(ax, interpolation_label=interpolation_label, warning=warning, qa_label=None)
    return {
        "fig": fig,
        "ax": ax,
        "image": image,
        "colorbar": cbar,
        "metadata": _metadata(
            interpolation=interpolation_method,
            interpolation_label=interpolation_label,
            warning=warning,
            colorbar_label=label,
        )
        | {
            "zero_order_marker_visible": "yes",
            "plus_one_order_marker_visible": "yes" if carrier_lpmm is not None else "no",
            "selected_fraction": float(selected_fraction) if selected_fraction is not None and np.isfinite(float(selected_fraction)) else np.nan,
            "rejected_fraction": max(0.0, 1.0 - float(selected_fraction)) if selected_fraction is not None and np.isfinite(float(selected_fraction)) else np.nan,
        },
    }


def plot_pupil_plane(
    pupil: Any,
    grid: Mapping[str, Any] | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    unit: str = "mm",
    pupil_radius: float | None = None,
    display_interpolation: bool = False,
    interpolation: str | None = None,
) -> dict[str, Any]:
    """Plot a pupil amplitude or mask plane."""

    vbb_style.apply_style()
    arr = _as_array(pupil)
    interpolation_method, interpolation_label = _resolve_interpolation(display_interpolation, interpolation)
    warning = _warning_for_array(arr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(
        arr,
        origin="lower",
        extent=_extent_from_grid(grid, arr.shape, unit),
        cmap="viridis",
        interpolation=interpolation_method,
        aspect="equal",
    )
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.set_title(title or "Pupil plane")
    if pupil_radius is not None and np.isfinite(float(pupil_radius)):
        ax.add_patch(patches.Circle((0.0, 0.0), float(pupil_radius), fill=False, edgecolor="white", linewidth=1.1))
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("pupil amplitude or mask [a.u.]")
    _add_display_labels(ax, interpolation_label=interpolation_label, warning=warning, qa_label=None)
    return {
        "fig": fig,
        "ax": ax,
        "image": image,
        "colorbar": cbar,
        "metadata": _metadata(
            interpolation=interpolation_method,
            interpolation_label=interpolation_label,
            warning=warning,
            colorbar_label="pupil amplitude or mask [a.u.]",
        ),
    }


def plot_metric_summary_box(
    metrics: Mapping[str, Any],
    *,
    ax: plt.Axes | None = None,
    title: str = "Metric summary",
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Render key scalar metrics as a compact, caveat-preserving text box."""

    vbb_style.apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.8), constrained_layout=True)
    else:
        fig = ax.figure
    ax.axis("off")
    ax.set_title(title)
    preferred = list(
        keys
        or [
            "case_id",
            "path",
            "ell",
            "canonical_zone_um",
            "strict_bessel_region_um",
            "ring_radius_um",
            "core_radius_um",
            "core_or_ring_peak_fluence_J_cm2",
            "incubated_threshold_J_cm2",
            "side_to_core_peak_ratio",
            "first_order_selected_fraction",
            "propagation_power_label",
            "validity_label",
        ]
    )
    rows = [(key, metrics.get(key)) for key in preferred if key in metrics]
    if not rows:
        rows = list(metrics.items())[:10]
    text = "\n".join(f"{_human_label(key)}: {_format_value(value)}" for key, value in rows)
    ax.text(
        0.02,
        0.96,
        text or "No metrics supplied",
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f4f4", "edgecolor": "#bdbdbd", "linewidth": 0.7},
    )
    return {"fig": fig, "ax": ax, "metadata": {"units_present": "yes", "qa_label_visible": "yes"}}


def _human_label(key: str) -> str:
    return str(key).replace("_", " ")


def _format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def plot_case_comparison_grid(
    cases: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    shared_color_scale: bool = True,
    display_interpolation: bool = False,
    interpolation: str | None = None,
    include_phase: bool = True,
    include_metrics: bool = True,
    log_scale: bool = False,
) -> dict[str, Any]:
    """Plot phase, XY, XZ, and metrics for multiple labelled cases."""

    vbb_style.apply_style()
    if not cases:
        raise ValueError("cases must contain at least one case mapping")
    cols: list[str] = []
    if include_phase and any("phase" in case for case in cases):
        cols.append("phase")
    cols.extend(["xy", "xz"])
    if include_metrics:
        cols.append("metrics")
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(4.7 * len(cols), 3.7 * len(cases)), squeeze=False, constrained_layout=True)
    if title:
        fig.suptitle(title)
    ref = None
    if shared_color_scale:
        vals = []
        for case in cases:
            if "xy" in case:
                vals.append(_positive_max(_as_array(case["xy"])))
            if "xz" in case:
                vals.append(_positive_max(_as_array(case["xz"])))
        ref = max(vals) if vals else None
    cell_meta: list[dict[str, Any]] = []
    for row, case in enumerate(cases):
        for col, kind in enumerate(cols):
            ax = axes[row, col]
            label = str(case.get("label", case.get("case_id", f"case {row + 1}")))
            if kind == "phase":
                if "phase" in case:
                    result = plot_slm_phase_mask(
                        case["phase"],
                        case.get("slm_grid") or case.get("grid"),
                        ax=ax,
                        title=f"{label}: phase",
                        display_interpolation=display_interpolation,
                        interpolation=interpolation,
                        phase_quantisation_bits=case.get("phase_bits"),
                        labels=case.get("phase_labels"),
                    )
                    cell_meta.append(result["metadata"])
                else:
                    ax.axis("off")
            elif kind == "xy":
                result = plot_xy_intensity(
                    case["xy"],
                    case.get("grid"),
                    ax=ax,
                    title=f"{label}: XY",
                    reference_max=ref,
                    display_interpolation=display_interpolation,
                    interpolation=interpolation,
                    log_scale=log_scale,
                    qa_label=case.get("qa_label"),
                )
                cell_meta.append(result["metadata"])
            elif kind == "xz":
                result = plot_xz_intensity(
                    case["xz"],
                    case.get("z_um"),
                    None,
                    grid=case.get("grid"),
                    ax=ax,
                    title=f"{label}: XZ",
                    reference_max=ref,
                    display_interpolation=display_interpolation,
                    interpolation=interpolation,
                    log_scale=log_scale,
                    qa_label=case.get("qa_label"),
                )
                cell_meta.append(result["metadata"])
            elif kind == "metrics":
                result = plot_metric_summary_box(case.get("metrics", {}), ax=ax, title=f"{label}: metrics")
                cell_meta.append(result["metadata"])
    return {
        "fig": fig,
        "axes": axes,
        "metadata": {
            "shared_color_scale": bool(shared_color_scale),
            "case_count": len(cases),
            "cell_metadata": cell_meta,
        },
    }


def plot_parameter_delta_comparison(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    title: str = "Parameter delta comparison",
    shared_color_scale: bool = True,
    display_interpolation: bool = False,
    interpolation: str | None = None,
) -> dict[str, Any]:
    """Plot before/after beam panels and report metric deltas."""

    cases = [dict(before), dict(after)]
    cases[0].setdefault("label", "before")
    cases[1].setdefault("label", "after")
    result = plot_case_comparison_grid(
        cases,
        title=title,
        shared_color_scale=shared_color_scale,
        display_interpolation=display_interpolation,
        interpolation=interpolation,
        include_phase=any("phase" in case for case in cases),
        include_metrics=True,
    )
    deltas: dict[str, float] = {}
    before_metrics = before.get("metrics", {})
    after_metrics = after.get("metrics", {})
    for key, value in after_metrics.items():
        if key in before_metrics:
            try:
                b = float(before_metrics[key])
                a = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(a) and np.isfinite(b):
                deltas[f"delta_{key}"] = a - b
    result["metadata"]["metric_deltas"] = deltas
    return result


__all__ = [
    "DISPLAY_INTERPOLATION_LABEL",
    "UNDERSAMPLING_WARNING",
    "plot_case_comparison_grid",
    "plot_fourier_filter_plane",
    "plot_metric_summary_box",
    "plot_parameter_delta_comparison",
    "plot_pupil_plane",
    "plot_slm_phase_mask",
    "plot_xy_intensity",
    "plot_xy_xz_pair",
    "plot_xz_intensity",
]
