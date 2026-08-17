"""
Stage 8C diagnostic figure builder for energy-scaled optical fields.

Produces a single diagnostic preview figure from a canonical optical field
(plane or stack) and its energy-scaled fluence result.  The figure is
DIAGNOSTIC ONLY and is stamped with:

    stage               = stage8c_surfacefield_energy_scaled_cockpit
    figure_status       = diagnostic_allowed
    model_status        = fluence_prediction
    final_export_allowed= False

It deliberately draws NO material-threshold contours, NO modification regions,
and NO microscopy proxies.  It only visualises the optical field and its
energy-scaled fluence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.digital_twin.field_coupling import (
    OpticalFieldPlane,
    OpticalFieldStack,
)
from vbb_study.digital_twin.field_fluence import (
    FluencePlaneResult,
    FluenceStackResult,
    peak_intensity_from_fluence_result,
)

STAGE = "stage8c_surfacefield_energy_scaled_cockpit"
MODEL_STATUS = "fluence_prediction"
FIGURE_STATUS = "diagnostic_allowed"
FINAL_EXPORT_ALLOWED = False

FIGURE_METADATA_BASE = {
    "stage": STAGE,
    "figure_status": FIGURE_STATUS,
    "model_status": MODEL_STATUS,
    "final_export_allowed": "False",
}


class CaveatsRequiredError(RuntimeError):
    """Raised when a Stage 8C figure is saved with caveats disabled."""


def _extent_um(n: int, d_um: float) -> tuple[float, float, float, float]:
    half = 0.5 * n * d_um
    return (-half, half, -half, half)


def _extent_from_coords(x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float, float, float]:
    """Return imshow extent from pixel-centre coordinate arrays."""
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    dx = float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    dy = float(np.mean(np.abs(np.diff(y)))) if y.size > 1 else 1.0
    return (
        float(np.min(x) - 0.5 * dx),
        float(np.max(x) + 0.5 * dx),
        float(np.min(y) - 0.5 * dy),
        float(np.max(y) + 0.5 * dy),
    )


def plot_stage8c_field_fluence_preview(
    stack_or_plane: OpticalFieldStack | OpticalFieldPlane,
    fluence_result: FluenceStackResult | FluencePlaneResult,
    *,
    energy_ledger: Any | None = None,
    pulse_duration_fs: float | None = None,
    output_path: str | Path | None = None,
    title: str = "Stage 8C energy-scaled optical field preview",
    show_caveats: bool = True,
    dpi: int = 180,
    metadata: Mapping[str, Any] | None = None,
) -> "matplotlib.figure.Figure":
    """Build (and optionally save) the Stage 8C diagnostic preview figure.

    Saving while ``show_caveats=False`` raises :class:`CaveatsRequiredError`.
    The figure is stamped diagnostic-only and ``final_export_allowed=False``.
    """
    is_stack = isinstance(stack_or_plane, OpticalFieldStack)
    is_plane = isinstance(stack_or_plane, OpticalFieldPlane)
    if not (is_stack or is_plane):
        raise TypeError(
            "stack_or_plane must be OpticalFieldStack or OpticalFieldPlane; "
            f"got {type(stack_or_plane).__name__}."
        )

    fig_meta = dict(FIGURE_METADATA_BASE)
    fig_meta["source_status"] = stack_or_plane.source_status
    if metadata:
        fig_meta.update({str(k): str(v) for k, v in metadata.items()})

    peak_I = None
    if pulse_duration_fs is not None:
        peak_I = peak_intensity_from_fluence_result(fluence_result, pulse_duration_fs)

    energy_at_sample = None
    if energy_ledger is not None:
        energy_at_sample = float(getattr(energy_ledger, "energy_at_sample_uJ", float("nan")))

    if is_stack:
        fig = _plot_stack(
            stack_or_plane, fluence_result, peak_I, energy_at_sample, pulse_duration_fs, title
        )
    else:
        fig = _plot_plane(
            stack_or_plane, fluence_result, peak_I, energy_at_sample, pulse_duration_fs, title
        )

    # Attach metadata to the figure object for programmatic inspection.
    fig.stage8c_metadata = fig_meta  # type: ignore[attr-defined]

    if output_path is not None:
        if not show_caveats:
            plt.close(fig)
            raise CaveatsRequiredError(
                "Refusing to save a Stage 8C figure with show_caveats=False. "
                "Diagnostic outputs must carry their caveats."
            )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
            # PNG tEXt chunks (read back by governance tests).
            "Title": title,
            "Description": (
                "Stage 8C energy-scaled optical fluence diagnostic. "
                "Optical fluence prediction only; not absorbed energy / dose / "
                "material modification / damage."
            ),
            "stage": STAGE,
            "figure_status": FIGURE_STATUS,
            "model_status": MODEL_STATUS,
            "final_export_allowed": "False",
            "source_status": str(stack_or_plane.source_status),
        })

    return fig


def _caveat_text(model_status: str, caveat: str) -> str:
    return (
        f"model_status = {model_status}\n"
        f"final_export_allowed = False\n"
        f"figure_status = diagnostic_allowed\n\n"
        f"{caveat}"
    )


def _plot_plane(
    plane: OpticalFieldPlane,
    res: FluencePlaneResult,
    peak_I: float | None,
    energy_at_sample: float | None,
    pulse_duration_fs: float | None,
    title: str,
) -> "matplotlib.figure.Figure":
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle(title, fontsize=12)

    ny, nx = plane.intensity.shape
    extent = _extent_from_coords(plane.x_um, plane.y_um)

    im0 = axes[0].imshow(plane.intensity, origin="lower", extent=extent, cmap="inferno", aspect="equal")
    axes[0].set_title("XY intensity (a.u.)")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(res.fluence_j_cm2, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    axes[1].set_title("XY fluence (J/cm²)")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("y (µm)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].axis("off")
    lines = [
        f"field: {plane.field_label}",
        f"source_status: {plane.source_status}",
        f"pulse energy (scaled): {res.pulse_energy_uJ:.4g} µJ",
    ]
    if energy_at_sample is not None and np.isfinite(energy_at_sample):
        lines.append(f"ledger energy@sample: {energy_at_sample:.4g} µJ")
    lines += [
        f"integrated energy: {res.integrated_energy_uJ:.4g} µJ",
        f"peak fluence: {res.peak_fluence_j_cm2:.4g} J/cm²",
        f"mean fluence: {res.mean_fluence_j_cm2:.4g} J/cm²",
    ]
    if peak_I is not None:
        lines.append(f"peak intensity (approx): {peak_I:.3e} W/cm²")
    if pulse_duration_fs is not None:
        lines.append(f"pulse duration: {pulse_duration_fs:.0f} fs")
    lines.append(f"dx, dy: {plane.dx_um:.4g}, {plane.dy_um:.4g} µm")
    if plane.z_um is not None:
        lines.append(f"z: {plane.z_um:.4g} µm")
    axes[2].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    axes[2].text(
        0.0, 0.30, _caveat_text(res.model_status, res.caveat),
        va="top", ha="left", fontsize=8, color="#7a0000", wrap=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _plot_stack(
    stack: OpticalFieldStack,
    res: FluenceStackResult,
    peak_I: float | None,
    energy_at_sample: float | None,
    pulse_duration_fs: float | None,
    title: str,
) -> "matplotlib.figure.Figure":
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=12)

    nz, ny, nx = stack.intensity_zyx.shape
    peak_idx = int(np.argmax(res.peak_fluence_by_z_j_cm2))
    extent_xy = _extent_from_coords(stack.x_um, stack.y_um)

    # (0,0) XY intensity at peak plane
    im = axes[0, 0].imshow(
        stack.intensity_zyx[peak_idx], origin="lower", extent=extent_xy, cmap="inferno", aspect="equal"
    )
    axes[0, 0].set_title(f"XY intensity @ peak z={res.peak_z_um:.2g} µm")
    axes[0, 0].set_xlabel("x (µm)")
    axes[0, 0].set_ylabel("y (µm)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (0,1) XY fluence at peak plane
    im = axes[0, 1].imshow(
        res.fluence_zyx_j_cm2[peak_idx], origin="lower", extent=extent_xy, cmap="viridis", aspect="equal"
    )
    axes[0, 1].set_title("XY fluence @ peak z (J/cm²)")
    axes[0, 1].set_xlabel("x (µm)")
    axes[0, 1].set_ylabel("y (µm)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (0,2) XZ fluence (slice through y centre)
    yc = ny // 2
    xz = res.fluence_zyx_j_cm2[:, yc, :].T  # shape [x, z]
    z = np.asarray(stack.z_um, dtype=float)
    extent_xz = (
        float(np.min(z)),
        float(np.max(z)),
        float(np.min(stack.x_um) - 0.5 * stack.dx_um),
        float(np.max(stack.x_um) + 0.5 * stack.dx_um),
    )
    im = axes[0, 2].imshow(xz, origin="lower", extent=extent_xz, cmap="viridis", aspect="auto")
    axes[0, 2].set_title("XZ fluence (y=0 slice)")
    axes[0, 2].set_xlabel("z (µm)")
    axes[0, 2].set_ylabel("x (µm)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (1,0) peak fluence vs z
    axes[1, 0].plot(z, res.peak_fluence_by_z_j_cm2, color="#1f77b4")
    axes[1, 0].axvline(res.peak_z_um, color="#d62728", ls="--", lw=1, label=f"peak z={res.peak_z_um:.2g} µm")
    axes[1, 0].set_title("Peak fluence vs z")
    axes[1, 0].set_xlabel("z (µm)")
    axes[1, 0].set_ylabel("peak fluence (J/cm²)")
    axes[1, 0].legend(fontsize=8)

    # (1,1) raw captured transverse power drift vs z
    axes[1, 1].plot(z, res.raw_captured_power_fraction_by_z, color="#2ca02c")
    axes[1, 1].axhline(1.0, color="#888", ls=":", lw=1, label="max raw integral")
    axes[1, 1].set_title("Raw captured power fraction vs z")
    axes[1, 1].set_xlabel("z (µm)")
    axes[1, 1].set_ylabel("fraction of max raw integral")
    axes[1, 1].legend(fontsize=8)

    # (1,2) text panel
    axes[1, 2].axis("off")
    lines = [
        f"field: {stack.field_label}",
        f"source_status: {stack.source_status}",
        f"planes: {nz}",
        f"pulse energy (scaled): {res.pulse_energy_uJ:.4g} µJ",
    ]
    if energy_at_sample is not None and np.isfinite(energy_at_sample):
        lines.append(f"ledger energy@sample: {energy_at_sample:.4g} µJ")
    lines += [
        f"peak fluence: {float(np.max(res.peak_fluence_by_z_j_cm2)):.4g} J/cm²",
        f"peak z: {res.peak_z_um:.4g} µm",
        f"prop. energy drift: {res.propagation_energy_drift_fraction:.3%}",
    ]
    if peak_I is not None:
        lines.append(f"peak intensity (approx): {peak_I:.3e} W/cm²")
    if pulse_duration_fs is not None:
        lines.append(f"pulse duration: {pulse_duration_fs:.0f} fs")
    lines.append(f"dx, dy: {stack.dx_um:.4g}, {stack.dy_um:.4g} µm")
    axes[1, 2].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    axes[1, 2].text(
        0.0, 0.34, _caveat_text(res.model_status, res.caveat),
        va="top", ha="left", fontsize=8, color="#7a0000", wrap=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
