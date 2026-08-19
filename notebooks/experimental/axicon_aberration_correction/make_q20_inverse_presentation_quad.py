"""Presentation-ready one-plane q=20 inverse-correction figure.

This renderer deliberately reuses the validated q20 modal analysis rather than
re-labelling a saved bitmap.  The four panels are therefore generated from the
same arrays used for the quantitative comparison:

    measured -> best-fit model -> predicted after correction -> ideal target

The corrected panel is a *per-plane model prediction*, not a post-SLM camera
measurement.  The metric banner reports ROI-normalized RMSE and Pearson
correlation to the ideal target before and after the model correction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modal_vortex_bessel import load_first_scan, estimate_global_kr, fit_plane
from q20_modal_analysis import _cartesian_stack

EPS = 1e-12


def _similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    av = np.asarray(a, float)[mask]
    bv = np.asarray(b, float)[mask]
    corr = float(np.corrcoef(av, bv)[0, 1])
    rmse = float(np.sqrt(np.mean((av - bv) ** 2)))
    return corr, rmse


def make_q20_inverse_presentation_quad(
    data_dir: Path,
    output_path: Path,
    *,
    z_positions_mm: np.ndarray | None = None,
    z_target_mm: float = -10.0,
    pixel_pitch_m: float = 5.5e-6,
    q: int = 20,
    m_max: int = 8,
    limit_um: float = 180.0,
    size: int = 241,
) -> dict:
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    images = load_first_scan(data_dir)
    if not images:
        raise FileNotFoundError(f"No z*_*.bmg files found in {data_dir.resolve()}")

    if z_positions_mm is None:
        # Current realigned q20 acquisition convention: -17 ... 0 mm.
        z_positions_mm = np.arange(-17, -17 + len(images), dtype=float)
    z_positions_mm = np.asarray(z_positions_mm, float)
    if len(z_positions_mm) != len(images):
        raise ValueError("z_positions_mm must contain one physical z value per BMG plane")

    kr, _ = estimate_global_kr(images, pixel_pitch_m, q, .55)
    outputs = [
        fit_plane(im, i, pixel_pitch_m, q, kr, m_max=m_max,
                  rmax_um=220, n_r=44, n_theta=96)
        for i, im in enumerate(images)
    ]
    fits = [item[0] for item in outputs]
    aux = [item[1] for item in outputs]

    axis, measured, model, corrected, ideal = _cartesian_stack(
        images, fits, aux, pixel_pitch_m, q, kr,
        limit_um=limit_um, size=size)

    iz = int(np.argmin(np.abs(z_positions_mm - float(z_target_mm))))
    selected_z = float(z_positions_mm[iz])

    XX, YY = np.meshgrid(axis, axis)
    roi = np.hypot(XX, YY) <= 160.0
    before_corr, before_rmse = _similarity(measured[iz], ideal, roi)
    fit_corr, fit_rmse = _similarity(model[iz], measured[iz], roi)
    after_corr, after_rmse = _similarity(corrected[iz], ideal, roi)

    arrays = (measured[iz], model[iz], corrected[iz], ideal)
    titles = (
        "measured",
        "best-fit model",
        "predicted after correction",
        "ideal target",
    )
    extent = [axis[0], axis[-1], axis[0], axis[-1]]

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.35), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, image, title in zip(axes, arrays, titles):
        shown = ax.imshow(
            image, origin="lower", extent=extent, cmap="inferno",
            vmin=0, vmax=1, interpolation="nearest"
        )
        ax.axhline(0, color="#00a6a6", lw=.45, alpha=.55)
        ax.axvline(0, color="#00a6a6", lw=.45, alpha=.55)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("signed x (µm)")
        ax.set_ylabel("signed y (µm)")
        ax.set_aspect("equal")

    cbar = fig.colorbar(shown, ax=axes, label="plane-normalized intensity", shrink=.82, pad=.015)
    cbar.ax.tick_params(labelsize=8)

    # Keep technical detail subordinate to the visual comparison.
    fig.suptitle(
        f"Representative inverse-correction plane  |  q={q}, z={selected_z:g} mm",
        fontsize=13, y=1.02,
    )

    # Primary quantitative takeaway: same ROI and normalization as the figure.
    metric_text = (
        f"RMSE to ideal:  {before_rmse:.3f}  →  {after_rmse:.3f}"
        f"     |     correlation to ideal:  {before_corr:.3f}  →  {after_corr:.3f}"
        f"     |     best-fit vs measured: r = {fit_corr:.3f}"
    )
    fig.text(
        0.5, -0.015, metric_text,
        ha="center", va="top", fontsize=10.5, weight="bold", color="#202428",
    )
    fig.text(
        0.5, -0.075,
        "Predicted after correction is a per-plane model result, not a post-SLM camera measurement.",
        ha="center", va="top", fontsize=8.5, color="#59636b",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    result = {
        "q": int(q),
        "z_mm": selected_z,
        "roi_radius_um": 160.0,
        "measured_vs_ideal_corr_before": before_corr,
        "measured_vs_ideal_rmse_before": before_rmse,
        "best_fit_vs_measured_corr": fit_corr,
        "best_fit_vs_measured_rmse": fit_rmse,
        "corrected_model_vs_ideal_corr_after": after_corr,
        "corrected_model_vs_ideal_rmse_after": after_rmse,
        "scope": "predicted after correction is a per-plane model prediction, not a measured post-SLM field",
    }
    output_path.with_suffix(".json").write_text(
        __import__("json").dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=here / "z-scan 2 1010")
    p.add_argument(
        "--output",
        type=Path,
        default=here / "outputs" / "slm_closed_loop_alignment" / "modal_q20" /
                "q20_inverse_correction_presentation_zminus10.png",
    )
    p.add_argument("--z-mm", type=float, default=-10.0)
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    result = make_q20_inverse_presentation_quad(a.data_dir, a.output, z_target_mm=a.z_mm)
    print(__import__("json").dumps(result, indent=2))
